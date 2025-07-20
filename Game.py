# Game.py (Bitboard Version - Final Architecture)

import random
from enum import Enum
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces

# ==============================================================================
# --- Bitboard 核心工具 ---
# ==============================================================================

POS_TO_SQ = np.array([[(r * 4 + c) for c in range(4)] for r in range(4)], dtype=np.int32)
SQ_TO_POS = {sq: (sq // 4, sq % 4) for sq in range(16)}

def ULL(x):
    """一个帮助函数，用于创建一个Bitboard（无符号长整型），仅将第x位置为1。"""
    return 1 << x

# 边栏常量以处理换行问题
FILE_A = sum(ULL(i) for i in [0, 4, 8, 12]); FILE_H = sum(ULL(i) for i in [3, 7, 11, 15])
NOT_FILE_A = ~FILE_A; NOT_FILE_H = ~FILE_H

# 动作类型枚举
ACTION_TYPE_REVEAL = 0
ACTION_TYPE_MOVE = 1
ACTION_TYPE_CANNON_ATTACK = 2
Move = collections.namedtuple('Move', ['from_sq', 'to_sq', 'action_type'])

# --- 枚举和Piece类定义 ---

class PieceType(Enum):
    """定义棋子类型及其大小等级。值越大，等级越高。"""
    A = 0; B = 1; C = 2; D = 3; E = 4; F = 5; G = 6 # 兵/卒, 炮, 马, 车, 象, 士, 将

class Piece:
    """棋子对象，存储棋子本身的属性（类型，玩家，是否翻开）。"""
    def __init__(self, piece_type, player):
        self.piece_type, self.player, self.revealed = piece_type, player, False
    def __repr__(self):
        return f"{'R' if self.revealed else 'H'}_{'R' if self.player == 1 else 'B'}{self.piece_type.name}"


class GameEnvironment(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    # --- 游戏核心常量 ---
    BOARD_ROWS, BOARD_COLS, NUM_PIECE_TYPES = 4, 4, 7
    TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS
    
    # --- 动作空间定义 (按方向组织的稠密空间) ---
    REVEAL_ACTIONS_COUNT = 16
    UP_ACTIONS_COUNT = 12
    DOWN_ACTIONS_COUNT = 12
    LEFT_ACTIONS_COUNT = 12
    RIGHT_ACTIONS_COUNT = 12
    CANNON_ATTACK_ACTIONS_COUNT = 64
    ACTION_SPACE_SIZE = (REVEAL_ACTIONS_COUNT + UP_ACTIONS_COUNT + DOWN_ACTIONS_COUNT + 
                         LEFT_ACTIONS_COUNT + RIGHT_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT) # 16+12+12+12+12+64 = 128

    MAX_CONSECUTIVE_MOVES = 40; WINNING_SCORE = 60
    PIECE_VALUES = {pt: val for pt, val in zip(PieceType, [4, 10, 10, 10, 10, 20, 30])}
    PIECE_MAX_COUNTS = {pt: val for pt, val in zip(PieceType, [2, 1, 1, 1, 1, 1, 1])}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        my_pieces_plane_size = self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS
        opponent_pieces_plane_size = self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS
        hidden_pieces_plane_size = self.TOTAL_POSITIONS; empty_plane_size = self.TOTAL_POSITIONS
        scalar_features_size = 3
        self.state_size = ( my_pieces_plane_size + opponent_pieces_plane_size + hidden_pieces_plane_size + empty_plane_size + scalar_features_size )

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)

        # 核心状态数据结构
        self.board = np.empty(self.TOTAL_POSITIONS, dtype=object)
        self.piece_bitboards = {p: [0] * self.NUM_PIECE_TYPES for p in [1, -1]}
        self.revealed_bitboards = {p: 0 for p in [1, -1]}
        self.hidden_bitboard = 0; self.empty_bitboard = 0
        self.dead_pieces = {-1: [], 1: []}; self.current_player = 1
        self.move_counter = 0; self.scores = {-1: 0, 1: 0}

        # 核心查找表
        self.attack_tables = {}
        self.action_to_move_map = {} # 反向查找表: action_index -> Move
        self.move_to_action_map = {} # 正向查找表: (from, to) -> action_index
        self.cannon_attack_to_action_map = {} # 炮正向查找表: (from, dir) -> action_index
        self._initialize_lookup_tables()
    
    def _initialize_lookup_tables(self):
        """在游戏开始前，一次性预计算所有需要的查找表，实现最高效率。"""
        # 1. 炮的射线表
        ray_attacks = [[0] * self.TOTAL_POSITIONS for _ in range(4)]
        for sq in range(self.TOTAL_POSITIONS):
            r, c = SQ_TO_POS[sq]
            for i in range(r - 1, -1, -1): ray_attacks[0][sq] |= ULL(POS_TO_SQ[i, c]) # N
            for i in range(r + 1, 4):      ray_attacks[1][sq] |= ULL(POS_TO_SQ[i, c]) # S
            for i in range(c - 1, -1, -1): ray_attacks[2][sq] |= ULL(POS_TO_SQ[r, i]) # W
            for i in range(c + 1, 4):      ray_attacks[3][sq] |= ULL(POS_TO_SQ[r, i]) # E
        self.attack_tables['rays'] = ray_attacks
        
        # 2. 构建稠密、按方向组织的普通移动查找表
        # action_index -> Move and (from,to) -> action_index
        action_idx = self.REVEAL_ACTIONS_COUNT # 从16开始
        
        # 方向顺序: UP, DOWN, LEFT, RIGHT
        # UP moves
        for from_sq in range(4, 16): self.action_to_move_map[action_idx] = Move(from_sq, from_sq - 4, ACTION_TYPE_MOVE); self.move_to_action_map[(from_sq, from_sq - 4)] = action_idx; action_idx += 1
        # DOWN moves
        for from_sq in range(12): self.action_to_move_map[action_idx] = Move(from_sq, from_sq + 4, ACTION_TYPE_MOVE); self.move_to_action_map[(from_sq, from_sq + 4)] = action_idx; action_idx += 1
        # LEFT moves
        for r in range(4):
            for c in range(1, 4): from_sq = r * 4 + c; self.action_to_move_map[action_idx] = Move(from_sq, from_sq - 1, ACTION_TYPE_MOVE); self.move_to_action_map[(from_sq, from_sq - 1)] = action_idx; action_idx += 1
        # RIGHT moves
        for r in range(4):
            for c in range(3): from_sq = r * 4 + c; self.action_to_move_map[action_idx] = Move(from_sq, from_sq + 1, ACTION_TYPE_MOVE); self.move_to_action_map[(from_sq, from_sq + 1)] = action_idx; action_idx += 1

        # 3. 构建稠密的炮攻击查找表
        # (from_sq, direction_idx) -> action_index
        base_idx = self.REVEAL_ACTIONS_COUNT + self.UP_ACTIONS_COUNT + self.DOWN_ACTIONS_COUNT + self.LEFT_ACTIONS_COUNT + self.RIGHT_ACTIONS_COUNT
        for from_sq in range(self.TOTAL_POSITIONS):
            for direction_idx in range(4):
                self.cannon_attack_to_action_map[(from_sq, direction_idx)] = base_idx + from_sq * 4 + direction_idx

    def _initialize_board(self):
        """初始化棋盘和所有状态变量。"""
        pieces = [Piece(pt, p) for pt, count in self.PIECE_MAX_COUNTS.items() for p in [1, -1] for _ in range(count)]
        if hasattr(self, 'np_random') and self.np_random is not None: self.np_random.shuffle(pieces)
        else: random.shuffle(pieces)
        for sq in range(self.TOTAL_POSITIONS): self.board[sq] = pieces[sq]

        self.piece_bitboards = {p: [0] * self.NUM_PIECE_TYPES for p in [1, -1]}
        self.revealed_bitboards = {p: 0 for p in [1, -1]}
        self.hidden_bitboard = (1 << self.TOTAL_POSITIONS) - 1
        self.empty_bitboard = 0; self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1; self.move_counter = 0; self.scores = {-1: 0, 1: 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._initialize_board()
        return self.get_state(), {'action_mask': self.action_masks()}
    
    def get_state(self):
        state = np.zeros(self.state_size, dtype=np.float32)
        my_player, opponent_player = self.current_player, -self.current_player
        for pt_val in range(self.NUM_PIECE_TYPES):
            start_idx = self._my_pieces_plane_start_idx + pt_val * self.TOTAL_POSITIONS
            state[start_idx : start_idx + self.TOTAL_POSITIONS] = self._bitboard_to_plane(self.piece_bitboards[my_player][pt_val])
            start_idx = self._opponent_pieces_plane_start_idx + pt_val * self.TOTAL_POSITIONS
            state[start_idx : start_idx + self.TOTAL_POSITIONS] = self._bitboard_to_plane(self.piece_bitboards[opponent_player][pt_val])
        state[self._hidden_pieces_plane_start_idx : self._hidden_pieces_plane_start_idx + self.TOTAL_POSITIONS] = self._bitboard_to_plane(self.hidden_bitboard)
        state[self._empty_plane_start_idx : self._empty_plane_start_idx + self.TOTAL_POSITIONS] = self._bitboard_to_plane(self.empty_bitboard)
        score_norm = self.WINNING_SCORE or 1.0; move_norm = self.MAX_CONSECUTIVE_MOVES or 1.0
        scalar_idx = self._scalar_features_start_idx
        state[scalar_idx] = self.scores[my_player] / score_norm; state[scalar_idx + 1] = self.scores[opponent_player] / score_norm
        state[scalar_idx + 2] = self.move_counter / move_norm
        return state

    def _bitboard_to_plane(self, bb): return (bb >> np.arange(self.TOTAL_POSITIONS)) & 1

    def step(self, action_index):
        raw_reward = 0
        # 翻棋动作直接用action_index作为sq
        if 0 <= action_index < self.REVEAL_ACTIONS_COUNT:
            move = Move(action_index, action_index, ACTION_TYPE_REVEAL)
        else: # 其他移动/攻击动作，通过反向查找表获取move对象
            move = self.action_to_move_map.get(action_index)
        
        if move is None: raise ValueError(f"Invalid action_index: {action_index}")

        if move.action_type == ACTION_TYPE_REVEAL: self._apply_reveal_update(move); self.move_counter = 0
        elif move.action_type == ACTION_TYPE_MOVE: raw_reward = self._apply_move_action(move)
        elif move.action_type == ACTION_TYPE_CANNON_ATTACK: raw_reward = self._handle_cannon_attack(move)
        
        self.current_player = -self.current_player
        reward = raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward
        return self.get_state(), reward, False, False, {'action_mask': self.action_masks()}

    def _apply_reveal_update(self, move: Move):
        piece = self.board[move.from_sq]; piece.revealed = True
        self.hidden_bitboard ^= ULL(move.from_sq); self.revealed_bitboards[piece.player] |= ULL(move.from_sq)
        self.piece_bitboards[piece.player][piece.piece_type.value] |= ULL(move.from_sq)

    def _apply_move_action(self, move: Move):
        attacker = self.board[move.from_sq]
        if self.board[move.to_sq] is None:
            move_mask = ULL(move.from_sq) | ULL(move.to_sq)
            self.piece_bitboards[attacker.player][attacker.piece_type.value] ^= move_mask
            self.revealed_bitboards[attacker.player] ^= move_mask; self.empty_bitboard ^= move_mask
            self.board[move.to_sq], self.board[move.from_sq] = attacker, None
            self.move_counter += 1; return 0
        else:
            defender = self.board[move.to_sq]
            attacker_move_mask = ULL(move.from_sq) | ULL(move.to_sq)
            self.piece_bitboards[attacker.player][attacker.piece_type.value] ^= attacker_move_mask
            self.revealed_bitboards[attacker.player] ^= attacker_move_mask
            defender_remove_mask = ULL(move.to_sq)
            self.piece_bitboards[defender.player][defender.piece_type.value] ^= defender_remove_mask
            self.revealed_bitboards[defender.player] ^= defender_remove_mask
            self.empty_bitboard |= ULL(move.from_sq); self.dead_pieces[defender.player].append(defender)
            self.board[move.to_sq], self.board[move.from_sq] = attacker, None
            self.move_counter = 0; points = self.PIECE_VALUES[defender.piece_type]
            self.scores[attacker.player] += points; return float(points)

    def _handle_cannon_attack(self, move: Move):
        attacker, defender = self.board[move.from_sq], self.board[move.to_sq]
        points = self.PIECE_VALUES[defender.piece_type]
        if defender.player == attacker.player: self.scores[-attacker.player] += points
        else: self.scores[attacker.player] += points
        attacker_move_mask = ULL(move.from_sq) | ULL(move.to_sq)
        self.piece_bitboards[attacker.player][attacker.piece_type.value] ^= attacker_move_mask
        self.revealed_bitboards[attacker.player] ^= attacker_move_mask
        defender_remove_mask = ULL(move.to_sq)
        if defender.revealed: self.piece_bitboards[defender.player][defender.piece_type.value] ^= defender_remove_mask
        else: self.hidden_bitboard ^= defender_remove_mask
        self.empty_bitboard |= ULL(move.from_sq); self.dead_pieces[defender.player].append(defender)
        self.board[move.to_sq], self.board[move.from_sq] = attacker, None
        self.move_counter = 0
        return float(points) if defender.player != attacker.player else -float(points)

    def action_masks(self):
        action_mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=int)
        my_player, opponent_player = self.current_player, -self.current_player
        
        # 1. 翻棋动作 (0-15)
        action_mask[0:16] = self._bitboard_to_plane(self.hidden_bitboard)
            
        # 2. 普通棋子移动/攻击 (16-63)
        my_revealed_bb = self.revealed_bitboards[my_player]; target_bbs = {}
        cumulative_targets = self.empty_bitboard
        for pt_val in range(self.NUM_PIECE_TYPES):
            cumulative_targets |= self.piece_bitboards[opponent_player][pt_val]; target_bbs[pt_val] = cumulative_targets
        target_bbs[PieceType.A.value] |= self.piece_bitboards[opponent_player][PieceType.G.value]
        target_bbs[PieceType.G.value] &= ~self.piece_bitboards[opponent_player][PieceType.A.value]

        # -- 按方向，并行计算所有普通棋子的走法 --
        up_moves_bb, down_moves_bb, left_moves_bb, right_moves_bb = 0, 0, 0, 0
        for pt_val in range(self.NUM_PIECE_TYPES):
            if pt_val == PieceType.B.value: continue 
            my_pieces_bb = self.piece_bitboards[my_player][pt_val]; valid_targets = target_bbs[pt_val]
            up_moves_bb |= ((my_pieces_bb) >> 4) & valid_targets
            down_moves_bb |= ((my_pieces_bb) << 4) & valid_targets
            left_moves_bb |= ((my_pieces_bb & NOT_FILE_A) >> 1) & valid_targets
            right_moves_bb |= ((my_pieces_bb & NOT_FILE_H) << 1) & valid_targets
        
        # -- 将bitboard映射到稠密的action_index --
        for to_sq_bb, shift in [(up_moves_bb, -4), (down_moves_bb, 4), (left_moves_bb, -1), (right_moves_bb, 1)]:
            temp_to_bb = to_sq_bb
            while temp_to_bb > 0:
                to_sq = temp_to_bb.bit_length() - 1; from_sq = to_sq - shift
                action_index = self.move_to_action_map.get((from_sq, to_sq))
                if action_index is not None: action_mask[action_index] = 1
                temp_to_bb ^= ULL(to_sq)

        # 3. 炮的攻击 (64-127)
        my_cannons_bb = self.piece_bitboards[my_player][PieceType.B.value]; all_pieces_bb = ~self.empty_bitboard
        valid_cannon_targets = ~self.revealed_bitboards[my_player]
        temp_cannons_bb = my_cannons_bb
        while temp_cannons_bb > 0:
            from_sq = temp_cannons_bb.bit_length() - 1
            for direction_idx in range(4):
                ray_bb = self.attack_tables['rays'][direction_idx][from_sq]; blockers = ray_bb & all_pieces_bb
                if blockers == 0: continue
                screen_sq = blockers.bit_length() - 1 if direction_idx in [0, 2] else (blockers & -blockers).bit_length() - 1
                after_screen_ray = self.attack_tables['rays'][direction_idx][screen_sq]; targets = after_screen_ray & all_pieces_bb
                if targets == 0: continue
                target_sq = targets.bit_length() - 1 if direction_idx in [0, 2] else (targets & -targets).bit_length() - 1
                if ULL(target_sq) & valid_cannon_targets:
                    action_index = self.cannon_attack_to_action_map[(from_sq, direction_idx)]
                    # 炮的攻击需要保存(from,to)到反向查找表，以便step查询
                    self.action_to_move_map[action_index] = Move(from_sq, target_sq, ACTION_TYPE_CANNON_ATTACK)
                    action_mask[action_index] = 1
            temp_cannons_bb ^= ULL(from_sq)
            
        return action_mask

    def render(self): pass
    def close(self): pass