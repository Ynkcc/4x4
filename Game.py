# Game.py - 基于Numpy向量的暗棋环境 (已集成课程学习)
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
        state = 'R' if self.revealed else 'H'
        player_char = 'R' if self.player == 1 else 'B'
        return f"{state}_{player_char}{self.piece_type.name}"

# --- 游戏核心常量 ---
BOARD_ROWS, BOARD_COLS = 4, 4
NUM_PIECE_TYPES = 7
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS

# --- 动作空间定义 ---
REVEAL_ACTIONS_COUNT = 16
REGULAR_MOVE_ACTIONS_COUNT = 48
CANNON_ATTACK_ACTIONS_COUNT = 48
ACTION_SPACE_SIZE = REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT # 112

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
    基于Numpy布尔向量的暗棋Gym环境 (支持课程学习)。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    # --- 修改: __init__ 方法，增加 curriculum_stage 参数 ---
    def __init__(self, render_mode=None, curriculum_stage=4):
        super().__init__()
        self.render_mode = render_mode
        # 新增: 保存课程学习阶段
        self.curriculum_stage = curriculum_stage

        # --- 状态和动作空间定义 (无变化) ---
        my_pieces_plane_size = NUM_PIECE_TYPES * TOTAL_POSITIONS
        opponent_pieces_plane_size = NUM_PIECE_TYPES * TOTAL_POSITIONS
        hidden_pieces_plane_size = TOTAL_POSITIONS
        empty_plane_size = TOTAL_POSITIONS
        scalar_features_size = 3
        state_size = (my_pieces_plane_size + opponent_pieces_plane_size +
                      hidden_pieces_plane_size + empty_plane_size + scalar_features_size)

        self._my_pieces_plane_start_idx = 0
        self._opponent_pieces_plane_start_idx = my_pieces_plane_size
        self._hidden_pieces_plane_start_idx = self._opponent_pieces_plane_start_idx + opponent_pieces_plane_size
        self._empty_plane_start_idx = self._hidden_pieces_plane_start_idx + hidden_pieces_plane_size
        self._scalar_features_start_idx = self._empty_plane_start_idx + empty_plane_size
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # --- 核心数据结构 (无变化) ---
        self.board = np.empty(TOTAL_POSITIONS, dtype=object)
        self.piece_vectors = {p: [np.zeros(TOTAL_POSITIONS, dtype=bool) for _ in range(NUM_PIECE_TYPES)] for p in [1, -1]}
        self.revealed_vectors = {p: np.zeros(TOTAL_POSITIONS, dtype=bool) for p in [1, -1]}
        self.hidden_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.empty_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)

        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

        self.attack_tables = {}
        self.action_to_coords = {}
        self.coords_to_action = {}
        self._initialize_lookup_tables()

    def _initialize_lookup_tables(self):
        """预计算所有需要的查找表 (无变化)。"""
        ray_attacks = np.zeros((4, TOTAL_POSITIONS, TOTAL_POSITIONS), dtype=bool)
        for sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[sq]
            for i in range(r - 1, -1, -1): ray_attacks[0, sq, POS_TO_SQ[(i, c)]] = True
            for i in range(r + 1, 4):      ray_attacks[1, sq, POS_TO_SQ[(i, c)]] = True
            for i in range(c - 1, -1, -1): ray_attacks[2, sq, POS_TO_SQ[(r, i)]] = True
            for i in range(c + 1, 4):      ray_attacks[3, sq, POS_TO_SQ[(r, i)]] = True
        self.attack_tables['rays'] = ray_attacks
        
        action_idx = 0
        for sq in range(TOTAL_POSITIONS):
            self.action_to_coords[action_idx] = tuple(SQ_TO_POS[sq])
            self.coords_to_action[tuple(SQ_TO_POS[sq])] = action_idx
            action_idx += 1
            
        for from_sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[from_sq]
            from_pos = tuple((r, c))
            for dr, dc in [( -1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= r + dr < BOARD_ROWS and 0 <= c + dc < BOARD_COLS:
                    to_sq = POS_TO_SQ[(r + dr, c + dc)]
                    to_pos = tuple(SQ_TO_POS[to_sq])
                    self.action_to_coords[action_idx] = (from_pos, to_pos)
                    self.coords_to_action[(from_pos, to_pos)] = action_idx
                    action_idx += 1

        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                from_pos = (r1, c1)
                for c2 in range(BOARD_COLS):
                    if abs(c1 - c2) > 1:
                        to_pos = (r1, c2)
                        if (from_pos, to_pos) not in self.coords_to_action:
                            self.action_to_coords[action_idx] = (from_pos, to_pos)
                            self.coords_to_action[(from_pos, to_pos)] = action_idx
                            action_idx += 1
                for r2 in range(BOARD_ROWS):
                     if abs(r1 - r2) > 1:
                        to_pos = (r2, c1)
                        if (from_pos, to_pos) not in self.coords_to_action:
                            self.action_to_coords[action_idx] = (from_pos, to_pos)
                            self.coords_to_action[(from_pos, to_pos)] = action_idx
                            action_idx += 1

    def _reset_all_vectors_and_state(self):
        """辅助函数：重置所有状态向量和游戏变量。"""
        for player in [1, -1]:
            for pt_idx in range(NUM_PIECE_TYPES):
                self.piece_vectors[player][pt_idx].fill(False)
            self.revealed_vectors[player].fill(False)
        
        self.hidden_vector.fill(False)
        self.empty_vector.fill(True) # 棋盘默认是空的
        self.board.fill(None)
        
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

    def _update_vectors_from_board(self):
        """辅助函数：根据 self.board 的内容，更新所有Numpy状态向量。"""
        for sq, piece in enumerate(self.board):
            if piece:
                self.empty_vector[sq] = False
                if piece.revealed:
                    self.revealed_vectors[piece.player][sq] = True
                    self.piece_vectors[piece.player][piece.piece_type.value][sq] = True
                else:
                    self.hidden_vector[sq] = True

    # --- 修改: _initialize_board 函数，根据课程阶段设置棋盘 ---
    def _initialize_board(self):
        """根据课程学习阶段初始化棋盘和所有状态变量。"""
        self._reset_all_vectors_and_state()

        # 阶段 1: 吃子入门
        if self.curriculum_stage == 1:
            # 红“車”在(1,1)，黑“卒”在(1,2)
            red_chariot = Piece(PieceType.CHARIOT, 1)
            red_chariot.revealed = True
            black_soldier = Piece(PieceType.SOLDIER, -1)
            black_soldier.revealed = True
            
            self.board[POS_TO_SQ[(1, 1)]] = red_chariot
            self.board[POS_TO_SQ[(1, 2)]] = black_soldier
            self.current_player = 1 # 确保红方先手

        # 阶段 2: 简单战斗 (炮吃子)
        elif self.curriculum_stage == 2:
            # 红“炮”在(0,0), 炮架在(0,1), 目标黑“卒”在(0,2)
            red_cannon = Piece(PieceType.CANNON, 1)
            red_cannon.revealed = True
            red_horse_mount = Piece(PieceType.HORSE, 1) # 炮架
            red_horse_mount.revealed = True
            black_soldier_target = Piece(PieceType.SOLDIER, -1)
            black_soldier_target.revealed = True

            self.board[POS_TO_SQ[(0, 0)]] = red_cannon
            self.board[POS_TO_SQ[(0, 1)]] = red_horse_mount
            self.board[POS_TO_SQ[(0, 2)]] = black_soldier_target
            self.current_player = 1

        # 阶段 3: 探索与决策
        elif self.curriculum_stage == 3:
            # 左半边明棋，右半边暗棋
            pieces = [
                Piece(PieceType.CHARIOT, 1), Piece(PieceType.HORSE, 1),
                Piece(PieceType.CANNON, -1), Piece(PieceType.SOLDIER, -1),
                Piece(PieceType.ELEPHANT, 1), Piece(PieceType.ADVISOR, -1)
            ]
            # 明棋
            for i, piece in enumerate(pieces[:3]):
                piece.revealed = True
                self.board[POS_TO_SQ[(i, 0)]] = piece
            # 暗棋
            for i, piece in enumerate(pieces[3:]):
                piece.revealed = False
                self.board[POS_TO_SQ[(i, 3)]] = piece
            self.current_player = 1

        # 阶段 4: 完整对局 (原始逻辑)
        else:
            pieces = [Piece(pt, p) for pt, count in PIECE_MAX_COUNTS.items() for p in [1, -1] for _ in range(count)]
            if hasattr(self, 'np_random') and self.np_random is not None:
                self.np_random.shuffle(pieces)
            else:
                random.shuffle(pieces)
            
            for sq in range(TOTAL_POSITIONS):
                self.board[sq] = pieces[sq]
            
            # 完整对局中，所有棋子开局都是暗的
            self.hidden_vector.fill(True)
            self.empty_vector.fill(False)
            return # 直接返回，因为向量已经设置好

        # 对于课程1,2,3，需要根据board内容更新向量
        self._update_vectors_from_board()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_board()
        return self.get_state(), {'action_mask': self.action_masks()}
    
    def get_state(self):
        """根据当前的状态向量动态生成供模型观察的状态 (无变化)。"""
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        my_player = self.current_player
        opponent_player = -self.current_player

        for pt_val in range(NUM_PIECE_TYPES):
            start_idx = self._my_pieces_plane_start_idx + pt_val * TOTAL_POSITIONS
            state[start_idx : start_idx + TOTAL_POSITIONS] = self.piece_vectors[my_player][pt_val].astype(np.float32)
            start_idx = self._opponent_pieces_plane_start_idx + pt_val * TOTAL_POSITIONS
            state[start_idx : start_idx + TOTAL_POSITIONS] = self.piece_vectors[opponent_player][pt_val].astype(np.float32)
        
        state[self._hidden_pieces_plane_start_idx : self._hidden_pieces_plane_start_idx + TOTAL_POSITIONS] = self.hidden_vector.astype(np.float32)
        state[self._empty_plane_start_idx : self._empty_plane_start_idx + TOTAL_POSITIONS] = self.empty_vector.astype(np.float32)
        
        score_norm = WINNING_SCORE if WINNING_SCORE > 0 else 1.0
        move_norm = MAX_CONSECUTIVE_MOVES if MAX_CONSECUTIVE_MOVES > 0 else 1.0
        scalar_idx = self._scalar_features_start_idx
        
        state[scalar_idx] = self.scores[my_player] / score_norm
        state[scalar_idx + 1] = self.scores[opponent_player] / score_norm
        state[scalar_idx + 2] = self.move_counter / move_norm
        
        return state

    # --- 修改: step函数，根据课程阶段调整奖励和结束条件 ---
    def step(self, action_index):
        acting_player = self.current_player
        coords = self.action_to_coords.get(action_index)
        if coords is None:
            raise ValueError(f"无效的动作索引: {action_index}")

        # --- 课程学习的特殊逻辑 ---
        # 阶段 1 & 2: 目标驱动的短期对局
        if self.curriculum_stage in [1, 2]:
            reward = -0.1 # 每一步都有一个小惩罚，鼓励快速完成目标
            terminated = False
            truncated = False
            
            # 只有移动/攻击动作
            if action_index >= REVEAL_ACTIONS_COUNT:
                from_sq = POS_TO_SQ[coords[0]]
                to_sq = POS_TO_SQ[coords[1]]
                
                # 检查是否是吃子
                if self.board[to_sq] is not None and self.board[to_sq].player == -acting_player:
                    reward = 1.0  # 成功吃子，巨大奖励
                    terminated = True # 游戏结束
                    self._apply_move_action(from_sq, to_sq) # 执行动作
                else:
                    self._apply_move_action(from_sq, to_sq) # 只是移动
            
            # 如果走了5步还没吃子，就结束
            if not terminated and self.move_counter >= 5:
                truncated = True
                reward = -1.0 # 失败惩罚

            self.current_player = -self.current_player
            info = {'winner': acting_player if terminated else None, 'action_mask': self.action_masks()}
            return self.get_state(), np.float32(reward), terminated, truncated, info

        # --- 阶段 3 & 4: 完整游戏逻辑 ---
        reward = -0.0005  # 时间惩罚
        
        if action_index < REVEAL_ACTIONS_COUNT:
            from_sq = POS_TO_SQ[coords]
            self._apply_reveal_update(from_sq)
            self.move_counter = 0
            # 阶段3: 鼓励翻棋
            if self.curriculum_stage == 3:
                reward += 0.001 
        else:
            from_sq = POS_TO_SQ[coords[0]]
            to_sq = POS_TO_SQ[coords[1]]
            raw_reward = self._apply_move_action(from_sq, to_sq)
            reward += raw_reward / WINNING_SCORE if WINNING_SCORE > 0 else raw_reward
        
        terminated, truncated, winner = False, False, None
        
        if self.scores[1] >= WINNING_SCORE:
            winner, terminated = 1, True
        elif self.scores[-1] >= WINNING_SCORE:
            winner, terminated = -1, True
        
        self.current_player = -self.current_player
        
        action_mask = self.action_masks()
        if not terminated and not truncated and np.sum(action_mask) == 0:
            winner = acting_player
            terminated = True

        if not terminated and not truncated and self.move_counter >= MAX_CONSECUTIVE_MOVES:
            winner, truncated = 0, True

        if terminated:
            if winner == acting_player:
                reward += 1.0
            elif winner == -acting_player:
                reward -= 1.0
        elif truncated:
            reward -= 0.5

        info = {'winner': winner, 'action_mask': action_mask}
        if (terminated or truncated) and self.render_mode == "human":
            self.render()

        return self.get_state(), np.float32(reward), terminated, truncated, info

    def _apply_reveal_update(self, from_sq):
        """应用翻棋动作并更新状态向量。"""
        piece = self.board[from_sq]
        piece.revealed = True
        self.hidden_vector[from_sq] = False
        self.revealed_vectors[piece.player][from_sq] = True
        self.piece_vectors[piece.player][piece.piece_type.value][from_sq] = True

    def _apply_move_action(self, from_sq, to_sq):
        """应用移动或攻击动作并更新状态向量。"""
        attacker = self.board[from_sq]
        defender = self.board[to_sq]
        reward = 0.0

        if defender is None:
            self.board[to_sq], self.board[from_sq] = attacker, None
            self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
            self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
            self.revealed_vectors[attacker.player][from_sq] = False
            self.revealed_vectors[attacker.player][to_sq] = True
            self.empty_vector[from_sq] = True
            self.empty_vector[to_sq] = False
            self.move_counter += 1
            return 0.0

        points = PIECE_VALUES[defender.piece_type]
        is_cannon_attack = attacker.piece_type == PieceType.CANNON
        
        if is_cannon_attack and attacker.player == defender.player:
            self.scores[-attacker.player] += points
            reward = -float(points)
        else:
            self.scores[attacker.player] += points
            reward = float(points)
        
        self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
        self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
        self.revealed_vectors[attacker.player][from_sq] = False
        self.revealed_vectors[attacker.player][to_sq] = True
        
        if defender.revealed:
            self.piece_vectors[defender.player][defender.piece_type.value][to_sq] = False
            self.revealed_vectors[defender.player][to_sq] = False
        else:
            self.hidden_vector[to_sq] = False
        
        self.empty_vector[from_sq] = True
        self.dead_pieces[defender.player].append(defender)
        self.board[to_sq], self.board[from_sq] = attacker, None
        self.move_counter = 0
        
        return reward

    def action_masks(self):
        """生成当前玩家所有合法动作的掩码 (无变化)。"""
        action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        my_player = self.current_player
        
        hidden_squares = np.where(self.hidden_vector)[0]
        for sq in hidden_squares:
            action_mask[self.coords_to_action[SQ_TO_POS[sq]]] = 1
            
        target_vectors = {}
        cumulative_targets = self.empty_vector.copy()
        for pt in PieceType:
            cumulative_targets |= self.piece_vectors[-my_player][pt.value]
            target_vectors[pt.value] = cumulative_targets.copy()
        target_vectors[PieceType.SOLDIER.value] |= self.piece_vectors[-my_player][PieceType.GENERAL.value]
        target_vectors[PieceType.GENERAL.value] &= ~self.piece_vectors[-my_player][PieceType.SOLDIER.value]

        for pt_val in range(NUM_PIECE_TYPES):
            if pt_val == PieceType.CANNON.value: continue
            
            my_pieces_sqs = np.where(self.piece_vectors[my_player][pt_val])[0]
            if my_pieces_sqs.size == 0: continue

            valid_targets_for_pt = target_vectors[pt_val]

            for from_sq in my_pieces_sqs:
                r, c = SQ_TO_POS[from_sq]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    to_r, to_c = r + dr, c + dc
                    if 0 <= to_r < BOARD_ROWS and 0 <= to_c < BOARD_COLS:
                        to_sq = POS_TO_SQ[(to_r, to_c)]
                        if valid_targets_for_pt[to_sq]:
                            action_index = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[to_sq]))
                            if action_index is not None:
                                action_mask[action_index] = 1

        my_cannons_sqs = np.where(self.piece_vectors[my_player][PieceType.CANNON.value])[0]
        if my_cannons_sqs.size > 0:
            all_pieces_vector = ~self.empty_vector
            valid_cannon_targets = ~self.revealed_vectors[my_player]
            
            for from_sq in my_cannons_sqs:
                for direction_idx in range(4):
                    ray_vec = self.attack_tables['rays'][direction_idx, from_sq]
                    blockers_on_ray_indices = np.where(ray_vec & all_pieces_vector)[0]
                    
                    if blockers_on_ray_indices.size < 2: continue

                    if direction_idx == 0 or direction_idx == 2:
                        screen_sq = np.max(blockers_on_ray_indices)
                        target_candidates = blockers_on_ray_indices[blockers_on_ray_indices < screen_sq]
                        if not target_candidates.size: continue
                        target_sq = np.max(target_candidates)
                    else:
                        screen_sq = np.min(blockers_on_ray_indices)
                        target_candidates = blockers_on_ray_indices[blockers_on_ray_indices > screen_sq]
                        if not target_candidates.size: continue
                        target_sq = np.min(target_candidates)

                    if valid_cannon_targets[target_sq]:
                        action_index = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[target_sq]))
                        if action_index is not None:
                            action_mask[action_index] = 1
            
        return action_mask

    def render(self):
        """以人类可读的方式在终端打印当前棋盘状态 (无变化)。"""
        if self.render_mode != 'human': return

        red_map = {p: c for p, c in zip(PieceType, "兵炮马俥相仕帥")}
        black_map = {p: c for p, c in zip(PieceType, "卒炮馬車象士將")}
        
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
        pass
