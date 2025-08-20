# src_code/environment.py
import random
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

# 导入统一配置
from config import (
    WINNING_SCORE, MAX_CONSECUTIVE_MOVES_FOR_DRAW, MAX_STEPS_PER_EPISODE,
    ACTION_SPACE_SIZE, HISTORY_WINDOW_SIZE, REVEAL_ACTIONS_COUNT,
    REGULAR_MOVE_ACTIONS_COUNT, CANNON_ATTACK_ACTIONS_COUNT
)

# ==============================================================================
class PieceType(Enum):
    SOLDIER = 0
    CANNON = 1
    HORSE = 2
    CHARIOT = 3
    ELEPHANT = 4
    ADVISOR = 5
    GENERAL = 6


class Piece:
    """棋子对象，存储其类型、所属玩家和是否翻开"""

    def __init__(self, piece_type, player):
        self.piece_type, self.player, self.revealed = piece_type, player, False

    def __repr__(self):
        state = 'R' if self.revealed else 'H'
        player_char = 'R' if self.player == 1 else 'B'
        return f"{state}_{player_char}{self.piece_type.name}"


BOARD_ROWS, BOARD_COLS = 4, 4
NUM_PIECE_TYPES = 7
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS

PIECE_VALUES = {pt: val for pt, val in zip(PieceType, [4, 10, 10, 10, 10, 20, 30])}
PIECE_MAX_COUNTS = {pt: val for pt, val in zip(PieceType, [2, 1, 1, 1, 1, 1, 1])}

POS_TO_SQ = {(r, c): r * BOARD_COLS + c for r in range(BOARD_ROWS) for c in range(BOARD_COLS)}
SQ_TO_POS = {sq: (sq // BOARD_COLS, sq % BOARD_COLS) for sq in range(TOTAL_POSITIONS)}


def _get_piece_survival_vector(dead_pieces: list) -> np.ndarray:
    """辅助函数，计算存活棋子的向量"""
    vec = np.ones(8, dtype=np.float32)
    s_count = 0
    for p in dead_pieces:
        if p.piece_type == PieceType.SOLDIER:
            if s_count < 2:
                vec[s_count] = 0.0
            s_count += 1
        else:
            vec[p.piece_type.value + 1] = 0.0
    return vec


class GameEnvironment(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self,
                 render_mode=None):
        super().__init__()
        self.render_mode = render_mode


        # --- Gym 环境空间定义 ---
        # 棋盘状态的通道数
        num_channels = NUM_PIECE_TYPES * 2 + 2
        board_shape = (num_channels, BOARD_ROWS, BOARD_COLS)
        # 标量状态的维度
        # 原始标量 (3) + 我方存活 (8) + 敌方存活 (8) + 历史动作 (HISTORY_WINDOW_SIZE + 1) * ACTION_SPACE_SIZE
        scalar_shape = (3 + 8 + 8 + (HISTORY_WINDOW_SIZE + 1) * ACTION_SPACE_SIZE, )
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0.0, high=1.0, shape=board_shape, dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=scalar_shape, dtype=np.float32)
        })
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # --- 游戏内部状态 ---
        self.board = np.empty(TOTAL_POSITIONS, dtype=object)
        self.piece_vectors = {p: [np.zeros(TOTAL_POSITIONS, dtype=bool) for _ in range(NUM_PIECE_TYPES)] for p in [1, -1]}
        self.revealed_vectors = {p: np.zeros(TOTAL_POSITIONS, dtype=bool) for p in [1, -1]}
        self.hidden_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.empty_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.total_step_counter = 0
        self.scores = {-1: 0, 1: 0}
        self.action_history = [] # 新增：历史动作序列

        # --- 预计算表 ---
        self.attack_tables = {}
        self.action_to_coords = {}
        self.coords_to_action = {}
        self._initialize_lookup_tables()

    def reset(self, seed=None, options=None):
        """重置游戏到初始状态"""
        super().reset(seed=seed)
        self._initialize_board()
        self.action_history = [] # 重置历史动作
        return self.get_state(), {'action_mask': self.action_masks(), 'winner': None}

    def step(self, action_index):
        """执行一步游戏逻辑"""
        if not self.action_masks()[action_index]:
            raise ValueError(f"错误：试图执行无效动作! 索引: {action_index}")

        self.total_step_counter += 1

        # 记录动作
        self.action_history.append(action_index)

        # 执行动作
        if action_index < REVEAL_ACTIONS_COUNT:
            # 翻棋动作
            self._apply_reveal_update(POS_TO_SQ[self.action_to_coords[action_index]])
            self.move_counter = 0
        else:
            # 移动或吃子动作
            from_sq, to_sq = POS_TO_SQ[self.action_to_coords[action_index][0]], POS_TO_SQ[
                self.action_to_coords[action_index][1]]
            self._apply_move_action(from_sq, to_sq)

        # --- 检查游戏结束条件 ---
        winner, terminated, truncated = None, False, False
        if self.scores[1] >= WINNING_SCORE:
            winner, terminated = 1, True
        elif self.scores[-1] >= WINNING_SCORE:
            winner, terminated = -1, True

        # 检查对手是否无棋可走
        if not terminated and not truncated and not np.any(self.action_masks(-self.current_player)):
            winner, terminated = self.current_player, True

        # 和棋检测
        if not terminated and not truncated and self.move_counter >= MAX_CONSECUTIVE_MOVES_FOR_DRAW:
            winner, terminated = 0, True
        if not terminated and not truncated and self.total_step_counter >= MAX_STEPS_PER_EPISODE:
            winner, truncated = 0, True

        # 计算当前玩家的奖励
        if winner is None:
            # 游戏还未结束
            reward = 0.0
        elif winner == self.current_player:
            # 当前玩家胜利
            reward = 1.0
        elif winner == -self.current_player:
            # 当前玩家失败
            reward = -1.0
        else:
            # 平局 (winner == 0)
            reward = 0.0

        # 切换玩家
        self.current_player *= -1

        return self.get_state(), np.float32(reward), terminated, truncated, {
            'winner': winner,
            'action_mask': self.action_masks()
        }

    def _apply_move_action(self, from_sq, to_sq):
        """处理移动和吃子动作"""
        attacker = self.board[from_sq]
        defender = self.board[to_sq]

        if defender is None:  # 移动
            self.board[to_sq], self.board[from_sq] = attacker, None
            # 更新 bit vectors
            self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
            self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
            self.revealed_vectors[attacker.player][from_sq] = False
            self.revealed_vectors[attacker.player][to_sq] = True
            self.empty_vector[from_sq], self.empty_vector[to_sq] = True, False
            self.move_counter += 1
        else:  # 吃子
            points = PIECE_VALUES[defender.piece_type]
            if defender.player == attacker.player:
                opponent = -attacker.player
                self.scores[opponent] += points  # 吃自己棋子，对方得分
            else:
                self.scores[attacker.player] += points
            self.board[to_sq], self.board[from_sq] = attacker, None
            # 更新被吃棋子的 bit vectors
            if defender.revealed:
                self.piece_vectors[defender.player][defender.piece_type.value][to_sq] = False
                self.revealed_vectors[defender.player][to_sq] = False
            else:
                self.hidden_vector[to_sq] = False

            # 更新攻击方棋子的 bit vectors
            self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
            self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
            self.revealed_vectors[attacker.player][from_sq] = False
            self.revealed_vectors[attacker.player][to_sq] = True
            self.empty_vector[from_sq] = True

            self.dead_pieces[defender.player].append(defender)
            self.move_counter = 0

    def close(self):
        """清理资源"""
        pass

    def get_state(self):
        """获取当前玩家视角的观察状态"""
        my_player, opponent_player = self.current_player, -self.current_player

        # --- 棋盘状态 (16, 4, 4) ---
        board_state_list = []
        # 我方棋子 (7个通道)
        board_state_list.extend(
            [self.piece_vectors[my_player][pt.value].reshape(BOARD_ROWS, BOARD_COLS) for pt in PieceType])
        # 敌方棋子 (7个通道)
        board_state_list.extend(
            [self.piece_vectors[opponent_player][pt.value].reshape(BOARD_ROWS, BOARD_COLS) for pt in PieceType])
        # 暗棋和空位 (2个通道)
        board_state_list.append(self.hidden_vector.reshape(BOARD_ROWS, BOARD_COLS))
        board_state_list.append(self.empty_vector.reshape(BOARD_ROWS, BOARD_COLS))

        # --- 标量状态 ---
        scalar_state = np.concatenate([
            np.array([
                self.scores[my_player] / WINNING_SCORE,
                self.scores[opponent_player] / WINNING_SCORE,
                self.move_counter / MAX_CONSECUTIVE_MOVES_FOR_DRAW
            ],
                     dtype=np.float32),
            _get_piece_survival_vector(self.dead_pieces[my_player]),  # 我方存活棋子 (8,)
            _get_piece_survival_vector(self.dead_pieces[opponent_player]),  # 敌方存活棋子 (8,)
            self._get_action_history_vector() # 新增：历史动作向量
        ])

        return {"board": np.array(board_state_list, dtype=np.float32), "scalars": scalar_state}

    def _get_action_history_vector(self):
        """
        将历史动作序列编码为固定长度的向量，并添加一个额外的空槽位用于动作填充。
        """
        # 截取最近的 HISTORY_WINDOW_SIZE 个历史动作
        history = self.action_history[-HISTORY_WINDOW_SIZE:]
        
        # 历史动作向量的大小为 (HISTORY_WINDOW_SIZE + 1) * ACTION_SPACE_SIZE
        # 额外的一个 ACTION_SPACE_SIZE 空间用于在训练时拼接动作
        history_vector = np.zeros((HISTORY_WINDOW_SIZE + 1) * ACTION_SPACE_SIZE, dtype=np.float32)

        # 编码历史动作
        for i, action in enumerate(history):
            start_index = i * ACTION_SPACE_SIZE
            history_vector[start_index + action] = 1.0

        return history_vector
    
    def action_masks(self, player_id: int = None):
        """为当前玩家计算所有合法动作的掩码"""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        my_player = player_id if player_id is not None else self.current_player

        # 1. 翻棋动作
        for sq in np.where(self.hidden_vector)[0]:
            mask[self.coords_to_action[SQ_TO_POS[sq]]] = 1

        # --- 移动和吃子动作 ---
        # 【重要修复】采用 temp_game.py 的逻辑，正确实现棋子等级制度
        # 3. 计算不同棋子的目标位置 (可移动/可吃)，遵守等级规则
        target_vectors = {}
        # 从空位开始，逐级加入更小或同级的敌方棋子作为可攻击目标
        cumulative_targets = self.empty_vector.copy()
        for pt in PieceType: # PieceType的枚举顺序决定了棋子等级
            cumulative_targets |= self.piece_vectors[-my_player][pt.value]
            target_vectors[pt.value] = cumulative_targets.copy()
        
        # 应用特殊规则: 兵可以吃帅
        target_vectors[PieceType.SOLDIER.value] |= self.piece_vectors[-my_player][PieceType.GENERAL.value]
        # 应用特殊规则: 帅不能吃兵
        target_vectors[PieceType.GENERAL.value] &= ~self.piece_vectors[-my_player][PieceType.SOLDIER.value]

        # 4. 普通棋子 (非炮) 的移动
        for pt_val in range(NUM_PIECE_TYPES):
            if pt_val == PieceType.CANNON.value:
                continue
            
            # 获取该等级棋子的所有可攻击/移动目标
            valid_targets_for_pt = target_vectors[pt_val]

            for from_sq in np.where(self.piece_vectors[my_player][pt_val])[0]:
                r, c = SQ_TO_POS[from_sq]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= r + dr < BOARD_ROWS and 0 <= c + dc < BOARD_COLS:
                        to_sq = POS_TO_SQ[(r + dr, c + dc)]
                        # 如果目标位置在合法目标集合中
                        if valid_targets_for_pt[to_sq]:
                            idx = self.coords_to_action.get((SQ_TO_POS[from_sq], (r + dr, c + dc)))
                            if idx is not None:
                                mask[idx] = 1
        
        # 5. 炮的移动和攻击 (此部分逻辑保持不变)
        all_pieces_vector = ~self.empty_vector
        # 炮不能吃自己的明棋, 但可以吃自己的暗棋、所有敌方棋子和空位(尽管攻击逻辑需要目标不能是空位)
        valid_cannon_targets = ~self.revealed_vectors[my_player] 
        for from_sq in np.where(self.piece_vectors[my_player][PieceType.CANNON.value])[0]:
            for d_idx in range(4): # 4个方向
                ray = self.attack_tables['rays'][d_idx, from_sq]
                blockers = np.where(ray & all_pieces_vector)[0]
                if len(blockers) >= 2: # 必须至少有2个棋子在射线上
                    # 找到炮架
                    screen = np.max(blockers) if d_idx in [0, 2] else np.min(blockers)
                    # 找到炮架后的目标
                    targets_on_ray = blockers[blockers < screen] if d_idx in [0, 2] else blockers[blockers > screen]
                    if targets_on_ray.size > 0:
                        target_sq = np.max(targets_on_ray) if d_idx in [0, 2] else np.min(targets_on_ray)
                        if valid_cannon_targets[target_sq]:
                            idx = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[target_sq]))
                            if idx is not None:
                                mask[idx] = 1
        return mask
    
    def copy(self, shuffle_hidden: bool = False):
        """
        返回当前环境的一个深拷贝。
        如果 shuffle_hidden 为 True，将打乱所有未翻开的棋子，模拟信息不确定性。
        """
        new_env = GameEnvironment()

        # 复制基本状态
        new_env.board = np.copy(self.board)
        new_env.piece_vectors = {p: [vec.copy() for vec in vecs] for p, vecs in self.piece_vectors.items()}
        new_env.revealed_vectors = {p: vec.copy() for p, vec in self.revealed_vectors.items()}
        new_env.hidden_vector = self.hidden_vector.copy()
        new_env.empty_vector = self.empty_vector.copy()
        new_env.dead_pieces = {p: list(pieces) for p, pieces in self.dead_pieces.items()}
        new_env.current_player = self.current_player
        new_env.move_counter = self.move_counter
        new_env.total_step_counter = self.total_step_counter
        new_env.scores = self.scores.copy()
        new_env.attack_tables = self.attack_tables
        new_env.action_to_coords = self.action_to_coords
        new_env.coords_to_action = self.coords_to_action
        new_env.action_history = self.action_history.copy() # 复制历史动作

        
        # 复制棋子对象本身，因为它们是可变对象
        for sq in range(TOTAL_POSITIONS):
            if self.board[sq] is not None:
                piece = self.board[sq]
                new_piece = Piece(piece.piece_type, piece.player)
                new_piece.revealed = piece.revealed
                new_env.board[sq] = new_piece

        if shuffle_hidden:
            hidden_pieces_and_positions = []
            for sq in np.where(self.hidden_vector)[0]:
                piece = new_env.board[sq]
                # 确保是未翻开的棋子
                if piece and not piece.revealed:
                    hidden_pieces_and_positions.append((piece, sq))

            if hidden_pieces_and_positions:
                # 提取棋子对象，打乱它们
                pieces_to_shuffle = [p for p, _ in hidden_pieces_and_positions]
                random.shuffle(pieces_to_shuffle)
                
                # 将打乱后的棋子放回原来的位置
                for i, (_, original_sq) in enumerate(hidden_pieces_and_positions):
                    new_env.board[original_sq] = pieces_to_shuffle[i]

        return new_env

    # ==============================================================================
    # --- 内部初始化辅助函数 ---
    # ==============================================================================

    def _initialize_lookup_tables(self):
        """预计算攻击射线和动作映射表"""
        # 射线攻击表 (用于炮)
        ray_attacks = np.zeros((4, TOTAL_POSITIONS, TOTAL_POSITIONS), dtype=bool)
        for sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[sq]
            for i in range(r - 1, -1, -1): ray_attacks[0, sq, POS_TO_SQ[(i, c)]] = True  # 上
            for i in range(r + 1, 4):      ray_attacks[1, sq, POS_TO_SQ[(i, c)]] = True  # 下
            for i in range(c - 1, -1, -1): ray_attacks[2, sq, POS_TO_SQ[(r, i)]] = True  # 左
            for i in range(c + 1, 4):      ray_attacks[3, sq, POS_TO_SQ[(r, i)]] = True  # 右
        self.attack_tables['rays'] = ray_attacks
        
        # 动作索引与坐标的映射
        action_idx = 0
        # 翻棋
        for sq in range(TOTAL_POSITIONS):
            self.action_to_coords[action_idx] = tuple(SQ_TO_POS[sq])
            self.coords_to_action[tuple(SQ_TO_POS[sq])] = action_idx
            action_idx += 1
        # 普通移动
        for from_sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[from_sq]
            from_pos = (r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= r + dr < BOARD_ROWS and 0 <= c + dc < BOARD_COLS:
                    to_sq = POS_TO_SQ[(r + dr, c + dc)]
                    self.action_to_coords[action_idx] = (from_pos, tuple(SQ_TO_POS[to_sq]))
                    self.coords_to_action[(from_pos, tuple(SQ_TO_POS[to_sq]))] = action_idx
                    action_idx += 1
        # 炮的攻击
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                from_pos = (r1, c1)
                for c2 in range(BOARD_COLS):
                    if abs(c1 - c2) > 1 and (from_pos, (r1, c2)) not in self.coords_to_action:
                        self.action_to_coords[action_idx] = (from_pos, (r1, c2))
                        self.coords_to_action[(from_pos, (r1, c2))] = action_idx
                        action_idx += 1
                for r2 in range(BOARD_ROWS):
                    if abs(r1 - r2) > 1 and (from_pos, (r2, c1)) not in self.coords_to_action:
                        self.action_to_coords[action_idx] = (from_pos, (r2, c1))
                        self.coords_to_action[(from_pos, (r2, c1))] = action_idx
                        action_idx += 1

    def _reset_all_vectors_and_state(self):
        """重置所有状态变量和bit vectors"""
        self.board.fill(None)
        for p in [1, -1]:
            for i in range(NUM_PIECE_TYPES):
                self.piece_vectors[p][i].fill(False)
            self.revealed_vectors[p].fill(False)
        self.hidden_vector.fill(False)
        self.empty_vector.fill(True)
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.total_step_counter = 0
        self.scores = {-1: 0, 1: 0}

    def _initialize_board(self):
        """初始化棋盘，随机放置棋子"""
        self._reset_all_vectors_and_state()
        pieces = [Piece(pt, p) for pt, count in PIECE_MAX_COUNTS.items() for p in [1, -1] for _ in range(count)]
        if hasattr(self, 'np_random') and self.np_random:
            self.np_random.shuffle(pieces)
        else:
            random.shuffle(pieces)
        for sq, piece in enumerate(pieces):
            self.board[sq] = piece
            self.empty_vector[sq] = False
            self.hidden_vector[sq] = True

    def _apply_reveal_update(self, from_sq):
        """处理翻棋动作并更新 bit vectors"""
        piece = self.board[from_sq]
        piece.revealed = True
        self.hidden_vector[from_sq] = False
        self.revealed_vectors[piece.player][from_sq] = True
        self.piece_vectors[piece.player][piece.piece_type.value][from_sq] = True