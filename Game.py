# Game.py - 基于Numpy向量的暗棋环境 (已加入最终奖惩)
import random
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ==============================================================================
# --- 类型定义与常量 ---
# ==============================================================================

class PieceType(Enum):
    """定义棋子类型及其对应索引值"""
    SOLDIER = 0
    CANNON = 1
    HORSE = 2
    CHARIOT = 3
    ELEPHANT = 4
    ADVISOR = 5
    GENERAL = 6

class Piece:
    """棋子对象，存储棋子本身的属性（类型，玩家，是否翻开）。"""
    def __init__(self, piece_type, player):
        self.piece_type, self.player, self.revealed = piece_type, player, False
    def __repr__(self):
        return f"{'R' if self.revealed else 'H'}_{'R'if self.player == 1 else 'B'}{self.piece_type.name}"

# --- 游戏核心常量 ---
BOARD_ROWS, BOARD_COLS = 4, 4
NUM_PIECE_TYPES = 7
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS

# --- 动作空间定义 (统一查找表驱动的稠密空间) ---
REVEAL_ACTIONS_COUNT = 16
REGULAR_MOVE_ACTIONS_COUNT = 48  # (4*2 for corners) + (8*3 for edges) + (4*4 for center) = 48
CANNON_ATTACK_ACTIONS_COUNT = 48 # 4x4棋盘上所有几何可能的炮击路径
ACTION_SPACE_SIZE = REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT # 16 + 48 + 48 = 112

MAX_CONSECUTIVE_MOVES = 40
WINNING_SCORE = 60

# --- 棋子属性 ---
PIECE_VALUES = {pt: val for pt, val in zip(PieceType, [4, 10, 10, 10, 10, 20, 30])}
PIECE_MAX_COUNTS = {pt: val for pt, val in zip(PieceType, [2, 1, 1, 1, 1, 1, 1])}

# --- 位置转换工具 ---
POS_TO_SQ = {(r, c): r * BOARD_COLS + c for r in range(BOARD_ROWS) for c in range(BOARD_COLS)}
SQ_TO_POS = {sq: (sq // BOARD_COLS, sq % BOARD_COLS) for sq in range(TOTAL_POSITIONS)}


class GameEnvironment(gym.Env):
    """
    基于Numpy布尔向量的暗棋Gym环境。
    - 使用Numpy向量代替Bitboard进行状态表示。
    - 采用与Cython版一致的112个动作的稠密动作空间。
    - 纯Python实现，无需Cython编译。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- 状态和动作空间定义 ---
        my_pieces_plane_size = NUM_PIECE_TYPES * TOTAL_POSITIONS
        opponent_pieces_plane_size = NUM_PIECE_TYPES * TOTAL_POSITIONS
        hidden_pieces_plane_size = TOTAL_POSITIONS
        empty_plane_size = TOTAL_POSITIONS
        scalar_features_size = 3
        state_size = (my_pieces_plane_size + opponent_pieces_plane_size +
                      hidden_pieces_plane_size + empty_plane_size + scalar_features_size)

        # 定义状态向量中各个部分的起始索引
        self._my_pieces_plane_start_idx = 0
        self._opponent_pieces_plane_start_idx = my_pieces_plane_size
        self._hidden_pieces_plane_start_idx = self._opponent_pieces_plane_start_idx + opponent_pieces_plane_size
        self._empty_plane_start_idx = self._hidden_pieces_plane_start_idx + hidden_pieces_plane_size
        self._scalar_features_start_idx = self._empty_plane_start_idx + empty_plane_size
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # --- 核心数据结构 ---
        # 棋盘，存储Piece对象，主要用于渲染和逻辑判断
        self.board = np.empty(TOTAL_POSITIONS, dtype=object)

        # 使用Numpy布尔向量代替Bitboard
        self.piece_vectors = {p: [np.zeros(TOTAL_POSITIONS, dtype=bool) for _ in range(NUM_PIECE_TYPES)] for p in [1, -1]}
        self.revealed_vectors = {p: np.zeros(TOTAL_POSITIONS, dtype=bool) for p in [1, -1]}
        self.hidden_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.empty_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)

        # 游戏状态变量
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

        # --- 统一查找表 ---
        self.attack_tables = {}
        self.action_to_coords = {}  # action_index -> coords
        self.coords_to_action = {}  # coords -> action_index
        self._initialize_lookup_tables()

    def _initialize_lookup_tables(self):
        """预计算所有需要的查找表，构建统一动作空间。"""
        # 1. 炮的射线表 (用于action_masks)
        ray_attacks = np.zeros((4, TOTAL_POSITIONS, TOTAL_POSITIONS), dtype=bool)
        for sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[sq]
            # 0: North, 1: South, 2: West, 3: East
            for i in range(r - 1, -1, -1): ray_attacks[0, sq, POS_TO_SQ[(i, c)]] = True
            for i in range(r + 1, 4):      ray_attacks[1, sq, POS_TO_SQ[(i, c)]] = True
            for i in range(c - 1, -1, -1): ray_attacks[2, sq, POS_TO_SQ[(r, i)]] = True
            for i in range(c + 1, 4):      ray_attacks[3, sq, POS_TO_SQ[(r, i)]] = True
        self.attack_tables['rays'] = ray_attacks
        
        # --- 构建统一动作查找表 ---
        action_idx = 0
        
        # 2. 翻棋动作 (索引 0-15)
        for sq in range(TOTAL_POSITIONS):
            pos = tuple(SQ_TO_POS[sq]) # 确保是元组
            self.action_to_coords[action_idx] = pos
            self.coords_to_action[pos] = action_idx
            action_idx += 1
            
        # 3. 普通移动动作 (索引 16-63)
        for from_sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[from_sq]
            from_pos = tuple((r, c)) # 确保是元组
            # 上
            if r > 0:
                to_sq = from_sq - 4
                to_pos = tuple(SQ_TO_POS[to_sq])
                self.action_to_coords[action_idx] = (from_pos, to_pos)
                self.coords_to_action[(from_pos, to_pos)] = action_idx
                action_idx += 1
            # 下
            if r < 3:
                to_sq = from_sq + 4
                to_pos = tuple(SQ_TO_POS[to_sq])
                self.action_to_coords[action_idx] = (from_pos, to_pos)
                self.coords_to_action[(from_pos, to_pos)] = action_idx
                action_idx += 1
            # 左
            if c > 0:
                to_sq = from_sq - 1
                to_pos = tuple(SQ_TO_POS[to_sq])
                self.action_to_coords[action_idx] = (from_pos, to_pos)
                self.coords_to_action[(from_pos, to_pos)] = action_idx
                action_idx += 1
            # 右
            if c < 3:
                to_sq = from_sq + 1
                to_pos = tuple(SQ_TO_POS[to_sq])
                self.action_to_coords[action_idx] = (from_pos, to_pos)
                self.coords_to_action[(from_pos, to_pos)] = action_idx
                action_idx += 1

        # 4. 炮的攻击动作 (索引 64-111)
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                from_pos = (r1, c1)
                # 水平攻击
                for c2 in range(BOARD_COLS):
                    if abs(c1 - c2) > 1:
                        to_pos = (r1, c2)
                        key = (from_pos, to_pos)
                        if key not in self.coords_to_action:
                            self.action_to_coords[action_idx] = key
                            self.coords_to_action[key] = action_idx
                            action_idx += 1
                # 垂直攻击
                for r2 in range(BOARD_ROWS):
                     if abs(r1 - r2) > 1:
                        to_pos = (r2, c1)
                        key = (from_pos, to_pos)
                        if key not in self.coords_to_action:
                            self.action_to_coords[action_idx] = key
                            self.coords_to_action[key] = action_idx
                            action_idx += 1
                            
    def _initialize_board(self):
        """初始化棋盘和所有状态变量。"""
        pieces = [Piece(pt, p) for pt, count in PIECE_MAX_COUNTS.items() for p in [1, -1] for _ in range(count)]
        if hasattr(self, 'np_random') and self.np_random is not None:
            self.np_random.shuffle(pieces)
        else:
            random.shuffle(pieces)
        
        for sq in range(TOTAL_POSITIONS):
            self.board[sq] = pieces[sq]

        # 重置所有状态向量
        for player in [1, -1]:
            for pt_idx in range(NUM_PIECE_TYPES):
                self.piece_vectors[player][pt_idx].fill(False)
            self.revealed_vectors[player].fill(False)
        
        self.hidden_vector.fill(True)
        self.empty_vector.fill(False)
        
        # 重置游戏状态变量
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_board()
        return self.get_state(), {'action_mask': self.action_masks()}
    
    def get_state(self):
        """根据当前的状态向量动态生成供模型观察的状态。"""
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        my_player = self.current_player
        opponent_player = -self.current_player

        # 填充棋子位置平面
        for pt_val in range(NUM_PIECE_TYPES):
            # 我方棋子
            start_idx = self._my_pieces_plane_start_idx + pt_val * TOTAL_POSITIONS
            state[start_idx : start_idx + TOTAL_POSITIONS] = self.piece_vectors[my_player][pt_val].astype(np.float32)
            # 对方棋子
            start_idx = self._opponent_pieces_plane_start_idx + pt_val * TOTAL_POSITIONS
            state[start_idx : start_idx + TOTAL_POSITIONS] = self.piece_vectors[opponent_player][pt_val].astype(np.float32)
        
        # 填充暗棋和空格平面
        state[self._hidden_pieces_plane_start_idx : self._hidden_pieces_plane_start_idx + TOTAL_POSITIONS] = self.hidden_vector.astype(np.float32)
        state[self._empty_plane_start_idx : self._empty_plane_start_idx + TOTAL_POSITIONS] = self.empty_vector.astype(np.float32)
        
        # 填充标量特征
        score_norm = WINNING_SCORE if WINNING_SCORE > 0 else 1.0
        move_norm = MAX_CONSECUTIVE_MOVES if MAX_CONSECUTIVE_MOVES > 0 else 1.0
        scalar_idx = self._scalar_features_start_idx
        
        state[scalar_idx] = self.scores[my_player] / score_norm
        state[scalar_idx + 1] = self.scores[opponent_player] / score_norm
        state[scalar_idx + 2] = self.move_counter / move_norm
        
        return state

    # === 修改: step函数被重构以加入最终奖惩 ===
    def step(self, action_index):
        # 记录下做出动作的玩家，以便在最后正确地分配奖惩
        acting_player = self.current_player

        reward = -0.0005  # 时间惩罚
        coords = self.action_to_coords.get(action_index)
        if coords is None:
            raise ValueError(f"无效的动作索引: {action_index}")

        # 应用动作并计算即时奖励
        if action_index < REVEAL_ACTIONS_COUNT:
            from_sq = POS_TO_SQ[coords]
            self._apply_reveal_update(from_sq)
            self.move_counter = 0
            reward += 0.0005 # 翻棋抵消时间惩罚
        else:
            from_sq = POS_TO_SQ[coords[0]]
            to_sq = POS_TO_SQ[coords[1]]
            raw_reward = self._apply_move_action(from_sq, to_sq)
            # 将吃子得分归一化，避免其与最终胜负奖励差距过大
            reward += raw_reward / WINNING_SCORE if WINNING_SCORE > 0 else raw_reward
        
        # --- 检查游戏结束条件 ---
        terminated, truncated, winner = False, False, None
        
        # 1. 检查得分胜利
        if self.scores[1] >= WINNING_SCORE:
            winner, terminated = 1, True
        elif self.scores[-1] >= WINNING_SCORE:
            winner, terminated = -1, True
        
        # 切换玩家，为下一步做准备
        self.current_player = -self.current_player
        
        # 2. 检查新玩家是否无棋可走 (在切换玩家后检查)
        action_mask = self.action_masks()
        if not terminated and not truncated and np.sum(action_mask) == 0:
            winner = acting_player # 做出最后一步的玩家(acting_player)获胜
            terminated = True

        # 3. 检查是否平局 (在所有胜利条件之后检查)
        if not terminated and not truncated and self.move_counter >= MAX_CONSECUTIVE_MOVES:
            winner, truncated = 0, True

        # --- 新增: 基于最终结果，修正 acting_player 的奖励 ---
        if terminated:
            if winner == acting_player:
                reward += 1.0  # 胜利，给予巨大正奖励
            elif winner == -acting_player:
                reward -= 1.0  # 失败，给予巨大负奖励
        elif truncated: # 平局
            reward -= 0.5 # 平局给予一个负奖励，鼓励分出胜负

        # 准备返回信息
        info = {'winner': winner, 'action_mask': action_mask}
        if (terminated or truncated) and self.render_mode == "human":
            self.render()

        return self.get_state(), np.float32(reward), terminated, truncated, info

    def _apply_reveal_update(self, from_sq):
        """应用翻棋动作并更新状态向量。"""
        piece = self.board[from_sq]
        piece.revealed = True
        
        # 更新状态向量
        self.hidden_vector[from_sq] = False
        self.revealed_vectors[piece.player][from_sq] = True
        self.piece_vectors[piece.player][piece.piece_type.value][from_sq] = True

    def _apply_move_action(self, from_sq, to_sq):
        """应用移动或攻击动作并更新状态向量。"""
        attacker = self.board[from_sq]
        defender = self.board[to_sq]
        reward = 0.0

        # 情况一: 移动到空位
        if defender is None:
            self.board[to_sq], self.board[from_sq] = attacker, None
            # 更新攻击方位置
            self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
            self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
            self.revealed_vectors[attacker.player][from_sq] = False
            self.revealed_vectors[attacker.player][to_sq] = True
            # 更新空格状态
            self.empty_vector[from_sq] = True
            self.empty_vector[to_sq] = False
            self.move_counter += 1
            return 0.0

        # 情况二: 攻击
        points = PIECE_VALUES[defender.piece_type]
        is_cannon_attack = attacker.piece_type == PieceType.CANNON
        
        # 计算得分和奖励
        if is_cannon_attack and attacker.player == defender.player:
            self.scores[-attacker.player] += points
            reward = -float(points)
        else:
            self.scores[attacker.player] += points
            reward = float(points)
        
        # 更新攻击方位置
        self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
        self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
        self.revealed_vectors[attacker.player][from_sq] = False
        self.revealed_vectors[attacker.player][to_sq] = True
        
        # 移除被吃方
        if defender.revealed:
            self.piece_vectors[defender.player][defender.piece_type.value][to_sq] = False
            self.revealed_vectors[defender.player][to_sq] = False
        else:
            self.hidden_vector[to_sq] = False
        
        # 更新棋盘和空格状态
        self.empty_vector[from_sq] = True
        self.dead_pieces[defender.player].append(defender)
        self.board[to_sq], self.board[from_sq] = attacker, None
        self.move_counter = 0
        
        return reward

    def action_masks(self):
        """生成当前玩家所有合法动作的掩码。"""
        action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        my_player = self.current_player
        opponent_player = -self.current_player
        
        # 1. 翻棋动作
        hidden_squares = np.where(self.hidden_vector)[0]
        for sq in hidden_squares:
            action_mask[self.coords_to_action[SQ_TO_POS[sq]]] = 1
            
        # 2. 普通棋子移动/攻击
        # 预计算所有有效目标 (空格 + 对方棋子)
        target_vectors = {}
        cumulative_targets = self.empty_vector.copy()
        for pt in PieceType:
            cumulative_targets |= self.piece_vectors[opponent_player][pt.value]
            target_vectors[pt.value] = cumulative_targets.copy()
        # 特殊规则: 兵吃将, 将不吃兵
        target_vectors[PieceType.SOLDIER.value] |= self.piece_vectors[opponent_player][PieceType.GENERAL.value]
        target_vectors[PieceType.GENERAL.value] &= ~self.piece_vectors[opponent_player][PieceType.SOLDIER.value]

        # 遍历所有普通棋子
        for pt_val in range(NUM_PIECE_TYPES):
            if pt_val == PieceType.CANNON.value: continue
            
            my_pieces_sqs = np.where(self.piece_vectors[my_player][pt_val])[0]
            if my_pieces_sqs.size == 0: continue

            valid_targets_for_pt = target_vectors[pt_val]

            for from_sq in my_pieces_sqs:
                r, c = SQ_TO_POS[from_sq]
                # 上下左右检查
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    to_r, to_c = r + dr, c + dc
                    if 0 <= to_r < BOARD_ROWS and 0 <= to_c < BOARD_COLS:
                        to_sq = POS_TO_SQ[(to_r, to_c)]
                        if valid_targets_for_pt[to_sq]:
                            action_index = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[to_sq]))
                            if action_index is not None:
                                action_mask[action_index] = 1

        # 3. 炮的攻击
        my_cannons_sqs = np.where(self.piece_vectors[my_player][PieceType.CANNON.value])[0]
        if my_cannons_sqs.size > 0:
            all_pieces_vector = ~self.empty_vector
            valid_cannon_targets = ~self.revealed_vectors[my_player]
            
            for from_sq in my_cannons_sqs:
                for direction_idx in range(4): # 4个方向
                    ray_vec = self.attack_tables['rays'][direction_idx, from_sq]
                    blockers_on_ray_indices = np.where(ray_vec & all_pieces_vector)[0]
                    
                    if blockers_on_ray_indices.size < 2: continue # 必须至少有炮架和目标

                    # 根据方向确定炮架和目标
                    if direction_idx == 0 or direction_idx == 2: # North, West (远端)
                        screen_sq = np.max(blockers_on_ray_indices)
                        target_candidates = blockers_on_ray_indices[blockers_on_ray_indices < screen_sq]
                        if not target_candidates.size: continue
                        target_sq = np.max(target_candidates)
                    else: # South, East (近端)
                        screen_sq = np.min(blockers_on_ray_indices)
                        target_candidates = blockers_on_ray_indices[blockers_on_ray_indices > screen_sq]
                        if not target_candidates.size: continue
                        target_sq = np.min(target_candidates)

                    if valid_cannon_targets[target_sq]:
                        from_pos, to_pos = SQ_TO_POS[from_sq], SQ_TO_POS[target_sq]
                        action_index = self.coords_to_action.get((from_pos, to_pos))
                        if action_index is not None:
                            action_mask[action_index] = 1
            
        return action_mask

    def render(self):
        """以人类可读的方式在终端打印当前棋盘状态。"""
        if self.render_mode != 'human': return

        red_map = {
            PieceType.GENERAL: "帥", PieceType.ADVISOR: "仕", PieceType.ELEPHANT: "相",
            PieceType.CHARIOT: "俥", PieceType.HORSE: "傌", PieceType.CANNON: "炮", PieceType.SOLDIER: "兵"
        }
        black_map = {
            PieceType.GENERAL: "將", PieceType.ADVISOR: "士", PieceType.ELEPHANT: "象",
            PieceType.CHARIOT: "車", PieceType.HORSE: "馬", PieceType.CANNON: "炮", PieceType.SOLDIER: "卒"
        }
        
        print("  " + "-" * 21)
        for r in range(BOARD_ROWS):
            print(f"{r} |", end="")
            for c in range(BOARD_COLS):
                sq = POS_TO_SQ[(r, c)]
                if self.empty_vector[sq]:
                    print("    |", end="")
                else:
                    piece = self.board[sq]
                    if not piece.revealed:
                        print(f" \033[90m暗\033[0m  |", end="")
                    elif piece.player == 1:
                        print(f" \033[91m{red_map[piece.piece_type]}\033[0m  |", end="")
                    else:
                        print(f" \033[94m{black_map[piece.piece_type]}\033[0m  |", end="")
            print()
            print("  " + "-" * 21)
        print("    " + "   ".join(str(c) for c in range(BOARD_COLS)))

        player_str = "\033[91m红方\033[0m" if self.current_player == 1 else "\033[94m黑方\033[0m"
        print(f"\n当前玩家: {player_str}, 得分: (红) {self.scores[1]} - {self.scores[-1]} (黑), "
              f"连续未吃/翻子: {self.move_counter}/{MAX_CONSECUTIVE_MOVES}\n")

    def close(self):
        """清理环境资源，符合Gym接口。"""
        pass