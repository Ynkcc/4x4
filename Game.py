# Game.py (Bitboard Version - Refactoring Part 5 Complete)

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

# 定义棋盘边界的Bitboard掩码，用于在生成走法时防止棋子“穿越”棋盘边界。
FILE_A = sum(ULL(i) for i in [0, 4, 8, 12])  # 第1列 (A列)
FILE_D = sum(ULL(i) for i in [3, 7, 11, 15]) # 第4列 (D列)
NOT_FILE_A = ~FILE_A # 按位取反，得到所有不是第1列的格子
NOT_FILE_D = ~FILE_D # 按位取反，得到所有不是第4列的格子

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
    """棋子对象，仅用于初始化、翻棋和被炮攻击时的类型查询。"""
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

    def _find_piece_type_at(self, player, sq):
        """通过遍历bitboards找到指定位置的棋子类型。"""
        sq_bb = ULL(sq)
        for pt_val in range(self.NUM_PIECE_TYPES):
            if self.piece_bitboards[player][pt_val] & sq_bb:
                return PieceType(pt_val)
        return None

    def step(self, action_index):
        """执行一个动作，更新游戏状态。"""
        raw_reward = 0
        if 0 <= action_index < self.REVEAL_ACTIONS_COUNT:
            from_sq = action_index
            move = Move(from_sq=from_sq, to_sq=from_sq, action_type=ACTION_TYPE_REVEAL)
            self._apply_reveal_update(move)
            self.move_counter = 0
        
        elif self.REVEAL_ACTIONS_COUNT <= action_index < self.ACTION_SPACE_SIZE:
            move_action_idx = action_index - self.REVEAL_ACTIONS_COUNT
            from_sq = move_action_idx // 4
            direction_idx = move_action_idx % 4
            to_sq = from_sq + [-4, 4, -1, 1][direction_idx]
            
            move = Move(from_sq=from_sq, to_sq=to_sq, action_type=ACTION_TYPE_MOVE)
            
            # 先判断是否是炮的移动
            attacker_type = self._find_piece_type_at(self.current_player, from_sq)
            if attacker_type == PieceType.B:
                raw_reward = self._handle_cannon_move(move) # 占位符
            else:
                raw_reward = self._apply_move_action(move, attacker_type)

        self.current_player = -self.current_player
        
        observation = self.get_state()
        # 备注: 此处的最终reward计算可以根据需要变得更复杂
        reward = raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward
        terminated = False
        truncated = False
        info = {'action_mask': self.action_masks()}
        return observation, reward, terminated, truncated, info

    def _apply_reveal_update(self, move: Move):
        """增量更新：处理翻棋动作。"""
        sq = move.from_sq
        piece = self.board[sq]
        piece.revealed = True
        
        self.hidden_bitboard ^= ULL(sq)
        self.revealed_bitboards[piece.player] |= ULL(sq)
        self.piece_bitboards[piece.player][piece.piece_type.value] |= ULL(sq)

    def _apply_move_action(self, move: Move, attacker_type: PieceType):
        """
        处理所有非炮棋子的移动和攻击，并返回原始奖励。
        此函数不依赖self.board。
        """
        to_sq_bb = ULL(move.to_sq)
        
        # 优先检查是否移动到空位
        if self.empty_bitboard & to_sq_bb:
            # --- 移动到空位 ---
            move_mask = ULL(move.from_sq) | to_sq_bb
            self.piece_bitboards[self.current_player][attacker_type.value] ^= move_mask
            self.revealed_bitboards[self.current_player] ^= move_mask
            self.empty_bitboard ^= move_mask
            
            # 更新棋盘数组以保持同步（仅用于翻棋和炮的逻辑）
            self.board[move.to_sq], self.board[move.from_sq] = self.board[move.from_sq], None
            
            self.move_counter += 1
            return 0
        else:
            # --- 吃子 ---
            opponent_player = -self.current_player
            defender_type = self._find_piece_type_at(opponent_player, move.to_sq)

            # 更新攻击方
            attacker_move_mask = ULL(move.from_sq) | to_sq_bb
            self.piece_bitboards[self.current_player][attacker_type.value] ^= attacker_move_mask
            self.revealed_bitboards[self.current_player] ^= attacker_move_mask
            
            # 移除被吃方
            self.piece_bitboards[opponent_player][defender_type.value] ^= to_sq_bb
            self.revealed_bitboards[opponent_player] ^= to_sq_bb
            
            # 更新空位
            self.empty_bitboard |= ULL(move.from_sq)
            
            # 更新棋盘数组
            defender_obj = self.board[move.to_sq]
            self.dead_pieces[opponent_player].append(defender_obj)
            self.board[move.to_sq], self.board[move.from_sq] = self.board[move.from_sq], None

            self.move_counter = 0
            
            # 计算并返回与棋子价值挂钩的原始奖励
            points = self.PIECE_VALUES[defender_type]
            self.scores[self.current_player] += points
            return float(points)

    def _handle_cannon_move(self, move: Move):
        """处理炮的移动/攻击。(占位符)"""
        # 备注: 炮的逻辑将在后续实现
        return 0

    def _generate_cannon_action_masks(self, actions):
        """生成炮的所有合法动作。(占位符)"""
        # 备注: 炮的逻辑将在后续实现
        return actions

    def action_masks(self):
        """生成当前玩家所有合法动作的掩码。"""
        actions = np.zeros(self.ACTION_SPACE_SIZE, dtype=int)
        my_player = self.current_player
        opponent_player = -my_player
        
        hidden_bb = self.hidden_bitboard
        temp_bb = hidden_bb
        while temp_bb > 0:
            sq = temp_bb.bit_length() - 1
            actions[sq] = 1
            temp_bb ^= ULL(sq)

        my_revealed_bb = self.revealed_bitboards[my_player]
        
        target_bbs = {}
        cumulative_targets = self.empty_bitboard
        for pt_val in range(self.NUM_PIECE_TYPES):
            cumulative_targets |= self.piece_bitboards[opponent_player][pt_val]
            target_bbs[pt_val] = cumulative_targets
        
        target_bbs[PieceType.A.value] |= self.piece_bitboards[opponent_player][PieceType.G.value] # 兵吃将
        target_bbs[PieceType.G.value] &= ~self.piece_bitboards[opponent_player][PieceType.A.value] # 将不吃兵

        for pt_val in range(self.NUM_PIECE_TYPES):
            if pt_val == PieceType.B.value: continue 

            my_pieces_bb = self.piece_bitboards[my_player][pt_val]
            valid_targets = target_bbs[pt_val]

            # 向上
            potential_to_sq = (my_pieces_bb >> 4) & valid_targets
            from_sq_bb = (potential_to_sq << 4) & my_revealed_bb
            temp_from_bb = from_sq_bb
            while temp_from_bb > 0:
                sq = temp_from_bb.bit_length() - 1
                actions[self.REVEAL_ACTIONS_COUNT + sq * 4 + 0] = 1
                temp_from_bb ^= ULL(sq)
            # 向下
            potential_to_sq = (my_pieces_bb << 4) & valid_targets
            from_sq_bb = (potential_to_sq >> 4) & my_revealed_bb
            temp_from_bb = from_sq_bb
            while temp_from_bb > 0:
                sq = temp_from_bb.bit_length() - 1
                actions[self.REVEAL_ACTIONS_COUNT + sq * 4 + 1] = 1
                temp_from_bb ^= ULL(sq)
            # 向左
            potential_to_sq = ((my_pieces_bb & NOT_FILE_A) >> 1) & valid_targets
            from_sq_bb = (potential_to_sq << 1) & my_revealed_bb
            temp_from_bb = from_sq_bb
            while temp_from_bb > 0:
                sq = temp_from_bb.bit_length() - 1
                actions[self.REVEAL_ACTIONS_COUNT + sq * 4 + 2] = 1
                temp_from_bb ^= ULL(sq)
            # 向右
            potential_to_sq = ((my_pieces_bb & NOT_FILE_D) << 1) & valid_targets
            from_sq_bb = (potential_to_sq >> 1) & my_revealed_bb
            temp_from_bb = from_sq_bb
            while temp_from_bb > 0:
                sq = temp_from_bb.bit_length() - 1
                actions[self.REVEAL_ACTIONS_COUNT + sq * 4 + 3] = 1
                temp_from_bb ^= ULL(sq)

        actions = self._generate_cannon_action_masks(actions)
        
        return actions

    def render(self):
        """(占位符)"""
        pass

    def close(self):
        """清理环境资源，符合Gym接口。"""
        pass