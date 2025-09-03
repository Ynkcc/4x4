# rl_code/rllib_version/core/environment.py

import os
import random
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any, List, Dict, Tuple
import math
from collections import deque

# --- 导入调试和状态堆叠常量 ---
from utils.constants import USE_FIXED_SEED_FOR_TRAINING, FIXED_SEED_VALUE, STATE_STACK_SIZE

# ==============================================================================
# --- 游戏常量与类型定义 ---
# ==============================================================================

WINNING_SCORE = 60
MAX_CONSECUTIVE_MOVES_FOR_DRAW = 24
MAX_STEPS_PER_EPISODE = 100
BOARD_ROWS, BOARD_COLS = 4, 4
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS
NUM_PIECE_TYPES = 5  # 棋子类型从7种减少到5种
INITIAL_REVEALED_PIECES = 16  # 游戏开局时随机翻开的棋子数量

class PieceType(Enum):
    # 移除了 HORSE 和 CHARIOT，并重新编号
    SOLDIER = 0
    CANNON = 1
    ELEPHANT = 2
    ADVISOR = 3
    GENERAL = 4

# 移除了 HORSE 和 CHARIOT
PIECE_VALUES = {
    PieceType.SOLDIER: 4,
    PieceType.CANNON: 10,
    PieceType.ELEPHANT: 10,
    PieceType.ADVISOR: 10,
    PieceType.GENERAL: 30
}

# 更新了棋子数量
PIECE_MAX_COUNTS = {
    PieceType.SOLDIER: 3,
    PieceType.CANNON: 1,
    PieceType.ELEPHANT: 1,
    PieceType.ADVISOR: 2,
    PieceType.GENERAL: 1
}

# 根据新的棋子数量和类型重新定义生存向量
PIECE_SURVIVAL_VEC_INFO = {
    PieceType.SOLDIER:  {'start_idx': 0, 'count': 3}, # 3个兵
    PieceType.CANNON:   {'start_idx': 3, 'count': 1}, # 1个炮
    PieceType.ELEPHANT: {'start_idx': 4, 'count': 1}, # 1个象
    PieceType.ADVISOR:  {'start_idx': 5, 'count': 2}, # 2个士
    PieceType.GENERAL:  {'start_idx': 7, 'count': 1}, # 1个将
}

REVEAL_ACTIONS_COUNT = 16
REGULAR_MOVE_ACTIONS_COUNT = 48
CANNON_ATTACK_ACTIONS_COUNT = 48
ACTION_SPACE_SIZE = REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT

POS_TO_SQ = {(r, c): r * BOARD_COLS + c for r in range(BOARD_ROWS) for c in range(BOARD_COLS)}
SQ_TO_POS = {sq: (sq // BOARD_COLS, sq % BOARD_COLS) for sq in range(TOTAL_POSITIONS)}


# ==============================================================================
# --- Piece 类 ---
# ==============================================================================

class Piece:
    """棋子对象，存储其类型、所属玩家和是否翻开"""
    def __init__(self, piece_type: PieceType, player: int):
        self.piece_type = piece_type
        self.player = player
        self.revealed = False

    def __repr__(self) -> str:
        state = 'R' if self.revealed else 'H'
        player_char = 'R' if self.player == 1 else 'B'
        return f"{state}_{player_char}{self.piece_type.name}"


# ==============================================================================
# --- GameEnvironment 类 (RLlib 兼容版) ---
# ==============================================================================

class DarkChessEnv(gym.Env):
    """
    暗棋游戏核心逻辑环境，兼容Gymnasium接口。
    此版本被设计为由 RLlib 的 MultiAgentEnv 封装，因此它只处理单步逻辑。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        
        # --- 状态堆叠核心 ---
        self.stack_size = STATE_STACK_SIZE
        self.board_history = deque([], maxlen=self.stack_size)
        self.scalar_history = deque([], maxlen=self.stack_size)
        
        self._initialize_spaces()
        
        # --- 游戏内部状态 ---
        self.board: List[Optional[Piece]] = [None] * TOTAL_POSITIONS
        self.piece_vectors: Dict[int, List[np.ndarray]] = {}
        self.revealed_vectors: Dict[int, np.ndarray] = {}
        self.hidden_vector: np.ndarray = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.empty_vector: np.ndarray = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.dead_pieces: Dict[int, List[Piece]] = {}
        self.current_player: int = 1
        self.move_counter: int = 0
        self.total_step_counter: int = 0
        self.scores: Dict[int, int] = {}
        self.survival_vectors: Dict[int, np.ndarray] = {}
        self.last_action: int = -1 
        
        # --- 预计算查找表 ---
        self.attack_tables: Dict[str, np.ndarray] = {}
        self.action_to_coords: Dict[int, Any] = {}
        self.coords_to_action: Dict[Any, int] = {}
        
        self._initialize_lookup_tables()

    def _initialize_spaces(self):
        """根据堆叠大小定义观察空间和动作空间"""
        num_channels_single = NUM_PIECE_TYPES * 2 + 2
        scalar_shape_single = (3 + 8 + 8 + 1 + ACTION_SPACE_SIZE,)
        
        board_shape_stacked = (num_channels_single * self.stack_size, BOARD_ROWS, BOARD_COLS)
        scalar_shape_stacked = (scalar_shape_single[0] * self.stack_size,)
        
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0.0, high=1.0, shape=board_shape_stacked, dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=scalar_shape_stacked, dtype=np.float32)
        })
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    def _initialize_lookup_tables(self):
        """预计算所有查找表以加速游戏。"""
        self._precompute_ray_attacks()
        self._precompute_action_mappings()

    def _reset_internal_state(self):
        """重置内部状态，并用零状态填充历史队列"""
        self.board = [None] * TOTAL_POSITIONS
        self.piece_vectors = {p: [np.zeros(TOTAL_POSITIONS, dtype=bool) for _ in range(NUM_PIECE_TYPES)] for p in [1, -1]}
        self.revealed_vectors = {p: np.zeros(TOTAL_POSITIONS, dtype=bool) for p in [1, -1]}
        self.hidden_vector.fill(False)
        self.empty_vector.fill(True)
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.total_step_counter = 0
        self.scores = {-1: 0, 1: 0}
        self.survival_vectors = {p: np.ones(8, dtype=np.float32) for p in [1, -1]}
        self.last_action = -1

        # 填充历史状态
        initial_board_state = self._get_board_state_tensor()
        initial_scalar_state = self._get_scalar_state_vector()
        
        self.board_history.clear()
        self.scalar_history.clear()
        
        for _ in range(self.stack_size):
            self.board_history.append(initial_board_state.copy())
            self.scalar_history.append(initial_scalar_state.copy())

    def _internal_reset(self, seed: Optional[int] = None):
        """私有重置方法，由RLlib封装器调用"""
        if USE_FIXED_SEED_FOR_TRAINING:
            seed = FIXED_SEED_VALUE
        super().reset(seed=seed)
        self._reset_internal_state() 
        self._initialize_board()
        
        # 初始化后，用真实的初始状态更新历史记录
        self._update_history()
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        """重置游戏，返回初始观察和信息"""
        self._internal_reset(seed=seed)
        info = {'action_mask': self.action_masks(), 'winner': None}
        return self.get_state(), info

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """
        执行一步动作。此方法只处理当前玩家的动作，不处理对手逻辑。
        """
        _, terminated, truncated, winner = self._internal_apply_action(action)
        reward = 0.0
        info = {'winner': winner, 'action_mask': self.action_masks()}
        return self.get_state(), reward, terminated, truncated, info

    def _internal_apply_action(self, action_index: int) -> Tuple[float, bool, bool, Optional[int]]:
        """应用一个动作并更新游戏状态，返回结果。"""
        # 暂停有效动作检查
        #if not self.action_masks()[action_index]: raise ValueError(f"错误：试图执行无效动作! 索引: {action_index}")

        self.last_action = action_index
        self.total_step_counter += 1
        
        shaping_reward = 0.0
        if action_index < REVEAL_ACTIONS_COUNT:
            sq = POS_TO_SQ[self.action_to_coords[action_index]]
            piece = self.board[sq]
            assert piece is not None, f"试图翻开空位置 {sq}"
            piece.revealed = True
            self.hidden_vector[sq] = False
            self.revealed_vectors[piece.player][sq] = True
            self.piece_vectors[piece.player][piece.piece_type.value][sq] = True
            self.move_counter = 0
        else:
            from_sq, to_sq = POS_TO_SQ[self.action_to_coords[action_index][0]], POS_TO_SQ[self.action_to_coords[action_index][1]]
            shaping_reward = self._apply_move_action(from_sq, to_sq)
        
        self._update_history()
        terminated, truncated, winner = self._check_game_over_conditions()
        return shaping_reward, terminated, truncated, winner
    
    def _update_history(self):
        """获取当前单帧状态并将其添加到历史队列中"""
        current_board = self._get_board_state_tensor()
        current_scalars = self._get_scalar_state_vector()
        self.board_history.append(current_board)
        self.scalar_history.append(current_scalars)

    def _apply_move_action(self, from_sq: int, to_sq: int) -> float:
        """应用移动或吃子动作，返回塑形奖励"""
        attacker = self.board[from_sq]
        defender = self.board[to_sq]
        self.board[to_sq], self.board[from_sq] = attacker, None
        
        assert attacker is not None, f"试图移动空位置 {from_sq} 的棋子"
        self._update_vectors_for_move(attacker.player, attacker.piece_type, from_sq, to_sq)
        if defender is None:
            self.move_counter += 1
            return 0.0
        else:
            self._update_vectors_for_capture(defender, to_sq)
            self.dead_pieces[defender.player].append(defender)
            self._update_survival_vector_on_capture(defender)
            self.move_counter = 0
            points = PIECE_VALUES[defender.piece_type]
            if defender.player == attacker.player:
                self.scores[-attacker.player] += points
                return -points / WINNING_SCORE
            else:
                self.scores[attacker.player] += points
                return points / WINNING_SCORE

    def _check_game_over_conditions(self) -> Tuple[bool, bool, Optional[int]]:
        """检查游戏是否结束。"""
        if self.scores[1] >= WINNING_SCORE: return True, False, 1
        if self.scores[-1] >= WINNING_SCORE: return True, False, -1
        if not np.any(self.action_masks(-self.current_player)): return True, False, self.current_player
        if self.move_counter >= MAX_CONSECUTIVE_MOVES_FOR_DRAW: return True, False, 0
        if self.total_step_counter >= MAX_STEPS_PER_EPISODE: return False, True, 0
        return False, False, None

    def get_state(self) -> Dict[str, np.ndarray]:
        """从历史队列拼接并返回最终的堆叠状态"""
        stacked_board = np.concatenate(list(self.board_history), axis=0).astype(np.float32)
        stacked_scalars = np.concatenate(list(self.scalar_history), axis=0).astype(np.float32)
        return {"board": stacked_board, "scalars": stacked_scalars}

    def _get_board_state_tensor(self) -> np.ndarray:
        """获取【单帧】棋盘状态张量"""
        my_player, opponent_player = self.current_player, -self.current_player
        tensors = [vec.reshape(BOARD_ROWS, BOARD_COLS) for vec in self.piece_vectors[my_player]]
        tensors.extend([vec.reshape(BOARD_ROWS, BOARD_COLS) for vec in self.piece_vectors[opponent_player]])
        tensors.append(self.hidden_vector.reshape(BOARD_ROWS, BOARD_COLS))
        tensors.append(self.empty_vector.reshape(BOARD_ROWS, BOARD_COLS))
        return np.array(tensors, dtype=np.float32)

    def _get_scalar_state_vector(self) -> np.ndarray:
        """获取【单帧】标量状态向量"""
        my_player, opponent_player = self.current_player, -self.current_player
        base_scalars = np.array([
            self.scores[my_player] / WINNING_SCORE, 
            self.scores[opponent_player] / WINNING_SCORE,
            self.move_counter / MAX_CONSECUTIVE_MOVES_FOR_DRAW
        ], dtype=np.float32)
        total_steps_scalar = np.array([self.total_step_counter / MAX_STEPS_PER_EPISODE], dtype=np.float32)
        last_action_one_hot = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if self.last_action != -1:
            last_action_one_hot[self.last_action] = 1.0
        return np.concatenate([
            base_scalars,
            self.survival_vectors[my_player], 
            self.survival_vectors[opponent_player],
            total_steps_scalar,
            last_action_one_hot
        ])

    # ==============================================================================
    # --- 【已还原】有效动作计算 (原始逻辑) ---
    # ==============================================================================
    
    def action_masks(self, player_id: Optional[int] = None) -> np.ndarray:
        player = player_id if player_id is not None else self.current_player
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        
        # 翻棋动作
        reveal_actions = np.where(self.hidden_vector)[0]
        for sq in reveal_actions:
            action_idx = self.coords_to_action.get(SQ_TO_POS[sq])
            if action_idx is not None:
                mask[action_idx] = 1
        
        # 移动和吃子
        self._add_regular_move_masks(mask, player)
        self._add_cannon_attack_masks(mask, player)
        return mask

    def _get_valid_target_vectors(self, player: int) -> Dict[PieceType, np.ndarray]:
        opponent = -player
        target_vectors, cumulative_targets = {}, self.empty_vector.copy()
        
        # 按照棋子等级从低到高累积可以被吃掉的敌方棋子
        for pt in PieceType:
            cumulative_targets |= self.piece_vectors[opponent][pt.value]
            target_vectors[pt] = cumulative_targets.copy()

        # 应用特殊规则
        # 兵可以吃将
        target_vectors[PieceType.SOLDIER] |= self.piece_vectors[opponent][PieceType.GENERAL.value]
        # 将不能吃兵
        target_vectors[PieceType.GENERAL] &= ~self.piece_vectors[opponent][PieceType.SOLDIER.value]
        
        return target_vectors

    def _add_regular_move_masks(self, mask: np.ndarray, player: int):
        target_vectors = self._get_valid_target_vectors(player)
        for pt in PieceType:
            if pt == PieceType.CANNON: continue
            
            valid_targets_for_pt = target_vectors[pt]
            piece_locations = np.where(self.piece_vectors[player][pt.value])[0]

            for from_sq in piece_locations:
                r, c = SQ_TO_POS[from_sq]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= r + dr < BOARD_ROWS and 0 <= c + dc < BOARD_COLS:
                        to_sq = POS_TO_SQ[(r + dr, c + dc)]
                        if valid_targets_for_pt[to_sq]:
                            idx = self.coords_to_action.get((SQ_TO_POS[from_sq], (r + dr, c + dc)))
                            if idx is not None: mask[idx] = 1

    def _add_cannon_attack_masks(self, mask: np.ndarray, player: int):
        all_pieces_vector = ~self.empty_vector
        # 炮不能攻击已翻开的己方棋子
        valid_cannon_targets = ~self.revealed_vectors[player]
        cannon_locations = np.where(self.piece_vectors[player][PieceType.CANNON.value])[0]

        for from_sq in cannon_locations:
            for d_idx in range(4): # 4个方向
                ray = self.attack_tables['rays'][d_idx, from_sq]
                blockers = np.where(ray & all_pieces_vector)[0]
                
                if len(blockers) >= 2:
                    is_pos_dir = d_idx in [1, 3] # DOWN, RIGHT
                    
                    # 找到第一个和第二个阻挡物
                    screen = np.min(blockers) if is_pos_dir else np.max(blockers)
                    targets_after_screen = blockers[blockers > screen] if is_pos_dir else blockers[blockers < screen]
                    
                    if targets_after_screen.size > 0:
                        target_sq = np.min(targets_after_screen) if is_pos_dir else np.max(targets_after_screen)
                        
                        if valid_cannon_targets[target_sq]:
                            idx = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[int(target_sq)]))
                            if idx is not None: mask[idx] = 1
    # ==============================================================================

    def _update_vectors_for_move(self, player: int, pt: PieceType, from_sq: int, to_sq: int):
        """更新移动后的向量"""
        pt_val = pt.value
        self.piece_vectors[player][pt_val][from_sq] = False
        self.piece_vectors[player][pt_val][to_sq] = True
        self.revealed_vectors[player][from_sq] = False
        self.revealed_vectors[player][to_sq] = True
        self.empty_vector[from_sq], self.empty_vector[to_sq] = True, False

    def _update_vectors_for_capture(self, defender: Piece, at_sq: int):
        """更新吃子后的向量"""
        if defender.revealed:
            self.piece_vectors[defender.player][defender.piece_type.value][at_sq] = False
            self.revealed_vectors[defender.player][at_sq] = False
        else:
            self.hidden_vector[at_sq] = False

    def _update_survival_vector_on_capture(self, captured_piece: Piece):
        """更新存活向量"""
        player, pt = captured_piece.player, captured_piece.piece_type
        info = PIECE_SURVIVAL_VEC_INFO[pt]
        start_idx = info['start_idx']
        
        if info['count'] > 1:
            for i in range(info['count']):
                if self.survival_vectors[player][start_idx + i] == 1.0:
                    self.survival_vectors[player][start_idx + i] = 0.0
                    break
        else: 
            self.survival_vectors[player][start_idx] = 0.0

    def _precompute_ray_attacks(self):
        """预计算炮的攻击射线"""
        rays = np.zeros((4, TOTAL_POSITIONS, TOTAL_POSITIONS), dtype=bool)
        for sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[sq]
            # UP
            for i in range(r - 1, -1, -1): rays[0, sq, POS_TO_SQ[(i, c)]] = True
            # DOWN
            for i in range(r + 1, BOARD_ROWS): rays[1, sq, POS_TO_SQ[(i, c)]] = True
            # LEFT
            for i in range(c - 1, -1, -1): rays[2, sq, POS_TO_SQ[(r, i)]] = True
            # RIGHT
            for i in range(c + 1, BOARD_COLS): rays[3, sq, POS_TO_SQ[(r, i)]] = True
        self.attack_tables['rays'] = rays

    def _precompute_action_mappings(self):
        """预计算动作索引与坐标的映射"""
        idx = 0
        # 翻棋动作
        for sq in range(TOTAL_POSITIONS):
            pos = tuple(SQ_TO_POS[sq])
            self.action_to_coords[idx], self.coords_to_action[pos] = pos, idx
            idx += 1
        
        # 常规移动
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                f_pos = (r1, c1)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= r1 + dr < BOARD_ROWS and 0 <= c1 + dc < BOARD_COLS:
                        t_pos = (r1 + dr, c1 + dc)
                        self.action_to_coords[idx], self.coords_to_action[(f_pos, t_pos)] = (f_pos, t_pos), idx
                        idx += 1
        
        # 炮的攻击
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                f_pos = (r1, c1)
                # 水平攻击
                for c2 in range(BOARD_COLS):
                    if abs(c1 - c2) > 1:
                        t_pos = (r1, c2)
                        if (f_pos, t_pos) not in self.coords_to_action:
                            self.action_to_coords[idx], self.coords_to_action[(f_pos, t_pos)] = (f_pos, t_pos), idx
                            idx += 1
                # 垂直攻击
                for r2 in range(BOARD_ROWS):
                    if abs(r1 - r2) > 1:
                        t_pos = (r2, c1)
                        if (f_pos, t_pos) not in self.coords_to_action:
                            self.action_to_coords[idx], self.coords_to_action[(f_pos, t_pos)] = (f_pos, t_pos), idx
                            idx += 1

    def _initialize_board(self):
        """初始化棋盘布局"""
        pieces = [Piece(pt, p) for pt, count in PIECE_MAX_COUNTS.items() for p in [1, -1] for _ in range(count)]
        
        if self.np_random is None:
             random.shuffle(pieces)
        else:
             self.np_random.shuffle(pieces)

        for sq, piece in enumerate(pieces):
            self.board[sq] = piece
            self.empty_vector[sq] = False
            self.hidden_vector[sq] = True
        
        if INITIAL_REVEALED_PIECES > 0:
            reveal_count = min(INITIAL_REVEALED_PIECES, TOTAL_POSITIONS)
            positions_to_reveal = self.np_random.choice(TOTAL_POSITIONS, size=reveal_count, replace=False)
            for sq in positions_to_reveal:
                piece = self.board[sq]
                if piece:
                    piece.revealed = True
                    self.hidden_vector[sq] = False
                    self.revealed_vectors[piece.player][sq] = True
                    self.piece_vectors[piece.player][piece.piece_type.value][sq] = True

    def apply_single_action(self, action_index):
        return self._internal_apply_action(action_index)