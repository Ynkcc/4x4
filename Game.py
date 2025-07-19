# Game.py (Bitboard Version - Refactoring Part 3 Complete)

import random
from enum import Enum
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces

# ==============================================================================
# --- Bitboard 核心工具 ---
# ==============================================================================

# 棋盘位置索引 (0-15) 与 (行, 列) 的转换字典
POS_TO_SQ = np.array([[(r * 4 + c) for c in range(4)] for r in range(4)], dtype=np.int32)
SQ_TO_POS = {sq: (sq // 4, sq % 4) for sq in range(16)}

def ULL(x):
    """一个帮助函数，用于创建一个Bitboard（无符号长整型），仅将第x位置为1。"""
    return 1 << x

# 定义走法类型的枚举
ACTION_TYPE_MOVE = 0
ACTION_TYPE_REVEAL = 1

# 使用命名元组来表示一个“走法”
Move = collections.namedtuple('Move', ['from_sq', 'to_sq', 'action_type'])

# --- 枚举和Piece类定义 ---

class PieceType(Enum):
    """定义棋子类型及其大小等级。值越大，等级越高。"""
    A = 0; B = 1; C = 2; D = 3; E = 4; F = 5; G = 6 # 兵/卒, 炮, 马, 车, 象, 士, 将

class Piece:
    """棋子对象，仅存储棋子本身的属性（类型，玩家，是否翻开）。"""
    def __init__(self, piece_type, player):
        self.piece_type, self.player, self.revealed = piece_type, player, False
    def __repr__(self):
        return f"{'R' if self.revealed else 'H'}_{'R' if self.player == 1 else 'B'}{self.piece_type.name}"


class GameEnvironment(gym.Env):
    """
    基于Bitboard的暗棋Gym环境。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    # --- 游戏核心常量 ---
    BOARD_ROWS, BOARD_COLS, NUM_PIECE_TYPES = 4, 4, 7
    TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS
    # 动作空间: 16个翻棋动作 + (16个位置 * 4个方向的移动) = 16 + 64 = 80
    REVEAL_ACTIONS_COUNT = TOTAL_POSITIONS
    MOVE_ACTIONS_COUNT = TOTAL_POSITIONS * 4
    ACTION_SPACE_SIZE = REVEAL_ACTIONS_COUNT + MOVE_ACTIONS_COUNT

    MAX_CONSECUTIVE_MOVES = 40
    WINNING_SCORE = 60
    PIECE_VALUES = {pt: val for pt, val in zip(PieceType, [4, 10, 10, 10, 10, 20, 30])}
    PIECE_MAX_COUNTS = {pt: val for pt, val in zip(PieceType, [2, 1, 1, 1, 1, 1, 1])}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # --- Gym环境所需的状态和动作空间定义 ---
        my_pieces_plane_size = self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS
        opponent_pieces_plane_size = self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS
        hidden_pieces_plane_size = self.TOTAL_POSITIONS
        empty_plane_size = self.TOTAL_POSITIONS
        scalar_features_size = 3
        
        self.state_size = (
            my_pieces_plane_size + 
            opponent_pieces_plane_size + 
            hidden_pieces_plane_size + 
            empty_plane_size + 
            scalar_features_size
        )

        self._my_pieces_plane_start_idx = 0
        self._opponent_pieces_plane_start_idx = my_pieces_plane_size
        self._hidden_pieces_plane_start_idx = self._opponent_pieces_plane_start_idx + opponent_pieces_plane_size
        self._empty_plane_start_idx = self._hidden_pieces_plane_start_idx + hidden_pieces_plane_size
        self._scalar_features_start_idx = self._empty_plane_start_idx + empty_plane_size

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)

        self.board = np.empty(self.TOTAL_POSITIONS, dtype=object)
        self.piece_bitboards = {p: [0] * self.NUM_PIECE_TYPES for p in [1, -1]}
        self.revealed_bitboards = {p: 0 for p in [1, -1]}
        self.hidden_bitboard = 0
        self.empty_bitboard = 0
        
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}
        
    def _initialize_board(self):
        """初始化棋盘，包括随机放置棋子对象，并根据其建立所有Bitboards。"""
        pieces = []
        for piece_type, count in self.PIECE_MAX_COUNTS.items():
            for _ in range(count):
                pieces.extend([Piece(piece_type, -1), Piece(piece_type, 1)])
        
        if hasattr(self, 'np_random') and self.np_random is not None:
             self.np_random.shuffle(pieces)
        else:
            random.shuffle(pieces)

        for sq in range(self.TOTAL_POSITIONS):
            self.board[sq] = pieces[sq]

        self.piece_bitboards = {p: [0] * self.NUM_PIECE_TYPES for p in [1, -1]}
        self.revealed_bitboards = {p: 0 for p in [1, -1]}
        self.hidden_bitboard = ULL(self.TOTAL_POSITIONS) - 1
        self.empty_bitboard = 0
        
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

    def reset(self, seed=None, options=None):
        """重置游戏环境到初始状态，符合Gym接口。"""
        super().reset(seed=seed)
        self._initialize_board()
        observation = self.get_state()
        info = {'action_mask': self.action_masks()}
        return observation, info
    
    def _bitboard_to_plane(self, bb):
        """将一个bitboard整数转换为一个16位的numpy数组平面"""
        return (bb >> np.arange(self.TOTAL_POSITIONS)) & 1

    def get_state(self):
        """获取当前玩家视角的状态向量。采用临时拼接方式构建。"""
        state = np.zeros(self.state_size, dtype=np.float32)
        my_player, opponent_player = self.current_player, -self.current_player

        start_idx = self._my_pieces_plane_start_idx
        for pt_val in range(self.NUM_PIECE_TYPES):
            bb = self.piece_bitboards[my_player][pt_val]
            state[start_idx : start_idx + self.TOTAL_POSITIONS] = self._bitboard_to_plane(bb)
            start_idx += self.TOTAL_POSITIONS
            
        start_idx = self._opponent_pieces_plane_start_idx
        for pt_val in range(self.NUM_PIECE_TYPES):
            bb = self.piece_bitboards[opponent_player][pt_val]
            state[start_idx : start_idx + self.TOTAL_POSITIONS] = self._bitboard_to_plane(bb)
            start_idx += self.TOTAL_POSITIONS

        state[self._hidden_pieces_plane_start_idx : self._hidden_pieces_plane_start_idx + self.TOTAL_POSITIONS] = self._bitboard_to_plane(self.hidden_bitboard)
        state[self._empty_plane_start_idx : self._empty_plane_start_idx + self.TOTAL_POSITIONS] = self._bitboard_to_plane(self.empty_bitboard)

        score_norm = self.WINNING_SCORE if self.WINNING_SCORE > 0 else 1.0
        move_norm = self.MAX_CONSECUTIVE_MOVES if self.MAX_CONSECUTIVE_MOVES > 0 else 1.0
        scalar_idx = self._scalar_features_start_idx
        state[scalar_idx] = self.scores[my_player] / score_norm
        state[scalar_idx + 1] = self.scores[opponent_player] / score_norm
        state[scalar_idx + 2] = self.move_counter / move_norm

        return state

    def step(self, action_index):
        """执行一个动作，更新游戏状态。"""
        # --- 1. 解码并执行动作 ---
        # 假定输入的action_index始终有效，移除合法性检查以优化性能
        if 0 <= action_index < self.REVEAL_ACTIONS_COUNT:
            # 这是一个翻棋动作
            from_sq = action_index
            move = Move(from_sq=from_sq, to_sq=from_sq, action_type=ACTION_TYPE_REVEAL)
            self._apply_reveal_update(move)
            self.move_counter = 0 # 翻棋重置计数器
        
        elif self.REVEAL_ACTIONS_COUNT <= action_index < self.ACTION_SPACE_SIZE:
            # 这是移动/攻击动作 (占位符)
            pass
        
        # --- 2. 切换玩家 ---
        self.current_player = -self.current_player
        
        # --- 3. 准备返回值 (奖励和结束判断为占位符) ---
        observation = self.get_state()
        reward = 0.0
        terminated = False
        truncated = False
        info = {'action_mask': self.action_masks()}
        return observation, reward, terminated, truncated, info

    def _apply_reveal_update(self, move: Move):
        """增量更新：处理翻棋动作。"""
        sq = move.from_sq
        piece = self.board[sq]
        piece.revealed = True
        
        # 更新Bitboards: 使用异或(XOR)操作来翻转比特位
        # 1. 从 hidden_bitboard 中移除该位置
        self.hidden_bitboard ^= ULL(sq)
        # 2. 在对应玩家的 revealed_bitboards 中添加该位置
        self.revealed_bitboards[piece.player] |= ULL(sq)
        # 3. 在对应玩家的特定棋子类型 piece_bitboards 中添加该位置
        self.piece_bitboards[piece.player][piece.piece_type.value] |= ULL(sq)

    def _apply_move_update(self, move: Move):
        """增量更新：处理移动到空位的动作。(占位符)"""
        pass

    def _apply_attack_update(self, move: Move):
        """增量更新：处理攻击动作。(占位符)"""
        pass

    def action_masks(self):
        """生成当前玩家所有合法动作的掩码。"""
        actions = np.zeros(self.ACTION_SPACE_SIZE, dtype=int)
        
        # 1. 生成翻棋动作 (Reveal)
        # 任何未翻开的棋子都可以被当前玩家翻开
        hidden_bb = self.hidden_bitboard
        temp_bb = hidden_bb
        while temp_bb > 0:
            # `bit_length() - 1` 是一个快速找到最高位(Most Significant Bit, MSB)索引的方法
            sq = temp_bb.bit_length() - 1
            actions[sq] = 1
            # 使用XOR将该位清零，以便处理下一个未翻开的棋子
            temp_bb ^= ULL(sq)

        # 2. 生成移动和攻击动作 (Move & Attack) (占位符)
        # 备注: 这部分逻辑将在后续重构中实现
        # actions[self.REVEAL_ACTIONS_COUNT:] = ...
        
        return actions

    def render(self):
        """(占位符)"""
        pass

    def close(self):
        """清理环境资源，符合Gym接口。"""
        pass