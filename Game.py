# Game.py (Bitboard Version - Final with Render)

import random
from enum import Enum
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces

# ==============================================================================
# --- Bitboard 核心工具 ---
# ==============================================================================


POS_TO_SQ = {(r, c): r * 4 + c for r in range(4) for c in range(4)}
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
    SOLDIER = 0   # 兵/卒
    CANNON = 1    # 炮
    HORSE = 2     # 马
    CHARIOT = 3   # 车
    ELEPHANT = 4  # 象
    ADVISOR = 5   # 士
    GENERAL = 6   # 将

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
    
    # --- 动作空间定义 (统一查找表驱动的稠密空间) ---
    REVEAL_ACTIONS_COUNT = 16
    REGULAR_MOVE_ACTIONS_COUNT = 48 # (4*2 for corners) + (8*3 for edges) + (4*4 for center)
    CANNON_ATTACK_ACTIONS_COUNT = 48 # 4x4棋盘上所有几何可能的炮击路径
    ACTION_SPACE_SIZE = (REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + 
                         CANNON_ATTACK_ACTIONS_COUNT) # 16 + 48 + 48 = 112

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

        # 定义状态向量中各个部分的起始索引
        self._my_pieces_plane_start_idx = 0
        self._opponent_pieces_plane_start_idx = my_pieces_plane_size
        self._hidden_pieces_plane_start_idx = my_pieces_plane_size + opponent_pieces_plane_size
        self._empty_plane_start_idx = my_pieces_plane_size + opponent_pieces_plane_size + hidden_pieces_plane_size
        self._scalar_features_start_idx = my_pieces_plane_size + opponent_pieces_plane_size + hidden_pieces_plane_size + empty_plane_size

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)

        # 核心状态数据结构
        self.board = np.empty(self.TOTAL_POSITIONS, dtype=object)
        self.piece_bitboards = {p: [0] * self.NUM_PIECE_TYPES for p in [1, -1]}
        self.revealed_bitboards = {p: 0 for p in [1, -1]}
        self.hidden_bitboard = 0; self.empty_bitboard = 0
        self.dead_pieces = {-1: [], 1: []}; self.current_player = 1
        self.move_counter = 0; self.scores = {-1: 0, 1: 0}

        # 统一查找表
        self.attack_tables = {}
        self.action_to_coords = {} # 反向查找表: action_index -> coords
        self.coords_to_action = {} # 正向查找表: coords -> action_index
        self._initialize_lookup_tables()
    
    def _initialize_lookup_tables(self):
        """在游戏开始前，一次性预计算所有需要的查找表，构建统一动作空间。"""
        # 1. 炮的射线表 (用于action_masks)
        ray_attacks = [[0] * self.TOTAL_POSITIONS for _ in range(4)]
        for sq in range(self.TOTAL_POSITIONS):
            r, c = SQ_TO_POS[sq]
            for i in range(r-1,-1,-1): ray_attacks[0][sq] |= ULL(POS_TO_SQ[i,c]) # N
            for i in range(r+1,4):     ray_attacks[1][sq] |= ULL(POS_TO_SQ[i,c]) # S
            for i in range(c-1,-1,-1): ray_attacks[2][sq] |= ULL(POS_TO_SQ[r,i]) # W
            for i in range(c+1,4):     ray_attacks[3][sq] |= ULL(POS_TO_SQ[r,i]) # E
        self.attack_tables['rays'] = ray_attacks
        
        # --- 构建统一查找表 ---
        action_idx = 0
        
        # 2. 翻棋动作 (索引 0-15)
        for sq in range(self.TOTAL_POSITIONS):
            pos = SQ_TO_POS[sq]
            self.action_to_coords[action_idx] = pos
            self.coords_to_action[pos] = action_idx
            action_idx += 1
            
        # 3. 普通移动动作 (索引 16-63)
        for from_sq in range(self.TOTAL_POSITIONS):
            r, c = SQ_TO_POS[from_sq]
            from_pos = (r, c)
            if r > 0: to_sq = from_sq - 4; self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq]); self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx; action_idx += 1
            if r < 3: to_sq = from_sq + 4; self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq]); self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx; action_idx += 1
            if c > 0: to_sq = from_sq - 1; self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq]); self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx; action_idx += 1
            if c < 3: to_sq = from_sq + 1; self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq]); self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx; action_idx += 1

        # 4. 炮的攻击动作 (索引 64-111)
        for r1 in range(self.BOARD_ROWS):
            for c1 in range(self.BOARD_COLS):
                from_pos = (r1, c1)
                for c2 in range(c1 + 2, self.BOARD_COLS): self.action_to_coords[action_idx] = (from_pos, (r1, c2)); self.coords_to_action[(from_pos, (r1, c2))] = action_idx; action_idx += 1
                for c2 in range(c1 - 2, -1, -1): self.action_to_coords[action_idx] = (from_pos, (r1, c2)); self.coords_to_action[(from_pos, (r1, c2))] = action_idx; action_idx += 1
                for r2 in range(r1 + 2, self.BOARD_ROWS): self.action_to_coords[action_idx] = (from_pos, (r2, c1)); self.coords_to_action[(from_pos, (r2, c1))] = action_idx; action_idx += 1
                for r2 in range(r1 - 2, -1, -1): self.action_to_coords[action_idx] = (from_pos, (r2, c1)); self.coords_to_action[(from_pos, (r2, c1))] = action_idx; action_idx += 1

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
        for pt in PieceType:
            pt_val = pt.value
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
        coords = self.action_to_coords.get(action_index)
        if coords is None: raise ValueError(f"Invalid action_index: {action_index}")

        if isinstance(coords[0], int):
            from_sq = POS_TO_SQ[coords]
            move = Move(from_sq, from_sq, ACTION_TYPE_REVEAL)
            self._apply_reveal_update(move)
            self.move_counter = 0
        else:
            from_sq, to_sq = POS_TO_SQ[coords[0]], POS_TO_SQ[coords[1]]
            attacker = self.board[from_sq]
            if attacker.piece_type == PieceType.CANNON:
                move = Move(from_sq, to_sq, ACTION_TYPE_CANNON_ATTACK)
            else:
                move = Move(from_sq, to_sq, ACTION_TYPE_MOVE)
            raw_reward = self._apply_move_action(move)
        
        self.current_player = -self.current_player
        reward = raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward
        return self.get_state(), reward, False, False, {'action_mask': self.action_masks()}

    def _apply_reveal_update(self, move: Move):
        piece = self.board[move.from_sq]; piece.revealed = True
        self.hidden_bitboard ^= ULL(move.from_sq); self.revealed_bitboards[piece.player] |= ULL(move.from_sq)
        self.piece_bitboards[piece.player][piece.piece_type.value] |= ULL(move.from_sq)

    def _apply_move_action(self, move: Move):
        attacker = self.board[move.from_sq]
        
        # 处理移动到空位的情况（只有普通移动会遇到）
        if self.board[move.to_sq] is None:
            move_mask = ULL(move.from_sq) | ULL(move.to_sq)
            self.piece_bitboards[attacker.player][attacker.piece_type.value] ^= move_mask
            self.revealed_bitboards[attacker.player] ^= move_mask
            self.empty_bitboard ^= move_mask
            self.board[move.to_sq], self.board[move.from_sq] = attacker, None
            self.move_counter += 1
            return 0
        
        # 处理攻击/吃子的情况
        defender = self.board[move.to_sq]
        points = self.PIECE_VALUES[defender.piece_type]
        
        # 计算得分（炮攻击有特殊的友军伤害逻辑）
        if move.action_type == ACTION_TYPE_CANNON_ATTACK:
            if defender.player == attacker.player:
                self.scores[-attacker.player] += points
                reward = -float(points)
            else:
                self.scores[attacker.player] += points
                reward = float(points)
        else:
            # 普通攻击只能攻击敌方棋子
            self.scores[attacker.player] += points
            reward = float(points)
        
        # 更新攻击方的bitboard
        attacker_move_mask = ULL(move.from_sq) | ULL(move.to_sq)
        self.piece_bitboards[attacker.player][attacker.piece_type.value] ^= attacker_move_mask
        self.revealed_bitboards[attacker.player] ^= attacker_move_mask
        
        # 更新被攻击方的bitboard
        defender_remove_mask = ULL(move.to_sq)
        if defender.revealed:
            self.piece_bitboards[defender.player][defender.piece_type.value] ^= defender_remove_mask
            self.revealed_bitboards[defender.player] ^= defender_remove_mask
        else:
            # 只有炮攻击才能攻击隐藏的棋子
            self.hidden_bitboard ^= defender_remove_mask
        
        # 更新棋盘状态
        self.empty_bitboard |= ULL(move.from_sq)
        self.dead_pieces[defender.player].append(defender)
        self.board[move.to_sq], self.board[move.from_sq] = attacker, None
        self.move_counter = 0
        
        return reward

    def action_masks(self):
        action_mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=int)
        my_player, opponent_player = self.current_player, -self.current_player
        
        # 1. 翻棋动作
        temp_bb = self.hidden_bitboard
        while temp_bb > 0:
            sq = temp_bb.bit_length() - 1
            action_index = self.coords_to_action[SQ_TO_POS[sq]]
            action_mask[action_index] = 1
            temp_bb ^= ULL(sq)
            
        # 2. 普通棋子移动/攻击
        my_revealed_bb = self.revealed_bitboards[my_player]; target_bbs = {}
        cumulative_targets = self.empty_bitboard
        for pt in PieceType:
            cumulative_targets |= self.piece_bitboards[opponent_player][pt.value]
            target_bbs[pt] = cumulative_targets
        target_bbs[PieceType.SOLDIER] |= self.piece_bitboards[opponent_player][PieceType.GENERAL.value]
        target_bbs[PieceType.GENERAL] &= ~self.piece_bitboards[opponent_player][PieceType.SOLDIER.value]

        for pt in PieceType:
            if pt == PieceType.CANNON: continue 
            my_pieces_bb = self.piece_bitboards[my_player][pt.value]; valid_targets = target_bbs[pt]
            for shift, wrap_check in [(-4, 0), (4, 0), (-1, NOT_FILE_A), (1, NOT_FILE_H)]:
                from_sq_bb = my_pieces_bb & wrap_check if wrap_check else my_pieces_bb
                potential_to_sq_bb = (from_sq_bb << shift) if shift > 0 else (from_sq_bb >> -shift)
                actual_to_sq_bb = potential_to_sq_bb & valid_targets
                temp_to_bb = actual_to_sq_bb
                while temp_to_bb > 0:
                    to_sq = temp_to_bb.bit_length() - 1; from_sq = to_sq - shift
                    action_index = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[to_sq]))
                    if action_index is not None: action_mask[action_index] = 1
                    temp_to_bb ^= ULL(to_sq)

        # 3. 炮的攻击
        my_cannons_bb = self.piece_bitboards[my_player][PieceType.CANNON.value]; all_pieces_bb = ~self.empty_bitboard
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
                    from_pos, to_pos = SQ_TO_POS[from_sq], SQ_TO_POS[target_sq]
                    action_index = self.coords_to_action.get((from_pos, to_pos))
                    if action_index is not None: action_mask[action_index] = 1
            temp_cannons_bb ^= ULL(from_sq)
            
        return action_mask

    def render(self):
        """以人类可读的方式在终端打印当前棋盘状态。"""
        if self.render_mode != 'human': return

        # 定义棋子在终端的显示字符
        red_map = {
            PieceType.GENERAL: "帥", PieceType.ADVISOR: "仕", PieceType.ELEPHANT: "相",
            PieceType.CHARIOT: "俥", PieceType.HORSE: "傌", PieceType.CANNON: "炮", PieceType.SOLDIER: "兵"
        }
        black_map = {
            PieceType.GENERAL: "將", PieceType.ADVISOR: "士", PieceType.ELEPHANT: "象",
            PieceType.CHARIOT: "車", PieceType.HORSE: "馬", PieceType.CANNON: "炮", PieceType.SOLDIER: "卒"
        }
        
        # 打印棋盘
        print("  " + "-" * 21)
        for r in range(self.BOARD_ROWS):
            print(f"{r} |", end="")
            for c in range(self.BOARD_COLS):
                sq = POS_TO_SQ[r, c]
                # 优先使用bitboard判断状态，效率更高
                if self.empty_bitboard & ULL(sq):
                    print("    |", end="") # 空位
                else:
                    piece = self.board[sq]
                    if not piece.revealed:
                        print(f" \033[90m暗\033[0m  |", end="") # 未翻开 (灰色)
                    elif piece.player == 1:
                        print(f" \033[91m{red_map[piece.piece_type]}\033[0m  |", end="") # 红方棋子 (红色)
                    else: # player == -1
                        print(f" \033[94m{black_map[piece.piece_type]}\033[0m  |", end="") # 黑方棋子 (蓝色)
            print()
            print("  " + "-" * 21)
        print("    " + "   ".join(str(c) for c in range(self.BOARD_COLS)))

        # 打印游戏状态信息
        player_str = "\033[91m红方\033[0m" if self.current_player == 1 else "\033[94m黑方\033[0m"
        print(f"\n当前玩家: {player_str}, 得分: (红) {self.scores[1]} - {self.scores[-1]} (黑), "
              f"连续未吃/翻子: {self.move_counter}/{self.MAX_CONSECUTIVE_MOVES}\n")

    def close(self):
        """清理环境资源，符合Gym接口。"""
        pass