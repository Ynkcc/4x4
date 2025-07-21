# Game_cython_simple.pyx - 简化的 Cython 优化版本

import random
from enum import Enum
import numpy as np
cimport numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces
import cython

# 基本常量
DEF BOARD_ROWS = 4
DEF BOARD_COLS = 4
DEF NUM_PIECE_TYPES = 7
DEF TOTAL_POSITIONS = 16
DEF REVEAL_ACTIONS_COUNT = 16
DEF REGULAR_MOVE_ACTIONS_COUNT = 48
DEF CANNON_ATTACK_ACTIONS_COUNT = 48
DEF ACTION_SPACE_SIZE = 112
DEF MAX_CONSECUTIVE_MOVES = 40
DEF WINNING_SCORE = 60

# 使用 ctypedef 定义类型
ctypedef unsigned long long bitboard

# 枚举定义
cdef enum PieceTypeEnum:
    SOLDIER = 0
    CANNON = 1  
    HORSE = 2
    CHARIOT = 3
    ELEPHANT = 4
    ADVISOR = 5
    GENERAL = 6

# 常量数组 (使用全局数组)
cdef int[7] PIECE_VALUES = [4, 10, 10, 10, 10, 20, 30]  # SOLDIER to GENERAL
cdef int[7] PIECE_MAX_COUNTS = [2, 1, 1, 1, 1, 1, 1]   # SOLDIER to GENERAL

# 辅助函数
@cython.cfunc
@cython.inline
cdef bitboard ULL(int x):
    return 1ULL << x

@cython.cfunc
@cython.inline  
cdef int trailing_zeros(bitboard bb):
    if bb == 0:
        return 64
    cdef int count = 0
    while (bb & 1) == 0:
        bb >>= 1
        count += 1
    return count

# 位置转换
POS_TO_SQ = {(r, c): r * 4 + c for r in range(4) for c in range(4)}
SQ_TO_POS = {sq: (sq // 4, sq % 4) for sq in range(16)}

# 边界常量
cdef bitboard FILE_A = ULL(0) | ULL(4) | ULL(8) | ULL(12)
cdef bitboard FILE_H = ULL(3) | ULL(7) | ULL(11) | ULL(15)
cdef bitboard NOT_FILE_A = ~FILE_A
cdef bitboard NOT_FILE_H = ~FILE_H

# Piece 类
class Piece:
    def __init__(self, piece_type, player):
        self.piece_type = piece_type
        self.player = player
        self.revealed = False
    
    def __repr__(self):
        return f"{'R' if self.revealed else 'H'}_{'R' if self.player == 1 else 'B'}{self.piece_type.name}"

class PieceType(Enum):
    SOLDIER = 0; CANNON = 1; HORSE = 2; CHARIOT = 3
    ELEPHANT = 4; ADVISOR = 5; GENERAL = 6

# 主类 - 使用常规Python类但内部使用Cython优化函数
class GameEnvironment(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 状态大小计算
        my_pieces_plane_size = NUM_PIECE_TYPES * TOTAL_POSITIONS
        opponent_pieces_plane_size = NUM_PIECE_TYPES * TOTAL_POSITIONS
        hidden_pieces_plane_size = TOTAL_POSITIONS
        empty_plane_size = TOTAL_POSITIONS
        scalar_features_size = 3
        self.state_size = (my_pieces_plane_size + opponent_pieces_plane_size + 
                          hidden_pieces_plane_size + empty_plane_size + scalar_features_size)

        self._my_pieces_plane_start_idx = 0
        self._opponent_pieces_plane_start_idx = my_pieces_plane_size
        self._hidden_pieces_plane_start_idx = my_pieces_plane_size + opponent_pieces_plane_size
        self._empty_plane_start_idx = self._hidden_pieces_plane_start_idx + hidden_pieces_plane_size
        self._scalar_features_start_idx = self._empty_plane_start_idx + empty_plane_size

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # 状态数据结构
        self.board = np.empty(TOTAL_POSITIONS, dtype=object)
        self.dead_pieces = {-1: [], 1: []}
        self.scores = {-1: 0, 1: 0}

        # Bitboards - 使用普通Python变量，在C函数中作为参数传递
        self.piece_bitboards = np.zeros((2, NUM_PIECE_TYPES), dtype=np.uint64)
        self.revealed_bitboards = np.zeros(2, dtype=np.uint64)
        self.hidden_bitboard = (1 << TOTAL_POSITIONS) - 1
        self.empty_bitboard = 0
        self.current_player = 1
        self.move_counter = 0

        # 查找表
        self.attack_tables = {}
        self.action_to_coords = {}
        self.coords_to_action = {}
        self._initialize_lookup_tables()

    def _initialize_lookup_tables(self):
        # 炮的射线表
        ray_attacks = [[0] * TOTAL_POSITIONS for _ in range(4)]
        for sq in range(TOTAL_POSITIONS):
            r = sq // 4
            c = sq % 4
            for i in range(r - 1, -1, -1): ray_attacks[0][sq] |= ULL(i * 4 + c)  # N
            for i in range(r + 1, 4):      ray_attacks[1][sq] |= ULL(i * 4 + c)  # S
            for i in range(c - 1, -1, -1): ray_attacks[2][sq] |= ULL(r * 4 + i)  # W
            for i in range(c + 1, 4):      ray_attacks[3][sq] |= ULL(r * 4 + i)  # E
        self.attack_tables['rays'] = ray_attacks
        
        # 动作映射
        action_idx = 0
        
        # 翻棋动作
        for sq in range(TOTAL_POSITIONS):
            self.action_to_coords[action_idx] = SQ_TO_POS[sq]
            self.coords_to_action[SQ_TO_POS[sq]] = action_idx
            action_idx += 1
            
        # 普通移动动作
        for from_sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[from_sq]
            from_pos = (r, c)
            if r > 0: 
                to_sq = from_sq - 4
                self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq])
                self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx
                action_idx += 1
            if r < 3: 
                to_sq = from_sq + 4
                self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq])
                self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx
                action_idx += 1
            if c > 0: 
                to_sq = from_sq - 1
                self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq])
                self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx
                action_idx += 1
            if c < 3: 
                to_sq = from_sq + 1
                self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq])
                self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx
                action_idx += 1

        # 炮的攻击动作
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                from_pos = (r1, c1)
                for c2 in range(c1 + 2, BOARD_COLS): 
                    self.action_to_coords[action_idx] = (from_pos, (r1, c2))
                    self.coords_to_action[(from_pos, (r1, c2))] = action_idx
                    action_idx += 1
                for c2 in range(c1 - 2, -1, -1): 
                    self.action_to_coords[action_idx] = (from_pos, (r1, c2))
                    self.coords_to_action[(from_pos, (r1, c2))] = action_idx
                    action_idx += 1
                for r2 in range(r1 + 2, BOARD_ROWS): 
                    self.action_to_coords[action_idx] = (from_pos, (r2, c1))
                    self.coords_to_action[(from_pos, (r2, c1))] = action_idx
                    action_idx += 1
                for r2 in range(r1 - 2, -1, -1): 
                    self.action_to_coords[action_idx] = (from_pos, (r2, c1))
                    self.coords_to_action[(from_pos, (r2, c1))] = action_idx
                    action_idx += 1

    def _initialize_board(self):
        # 创建棋子列表
        pieces = []
        for pt_val in range(NUM_PIECE_TYPES):
            count = PIECE_MAX_COUNTS[pt_val]
            for p in [1, -1]:
                for _ in range(count):
                    pieces.append(Piece(PieceType(pt_val), p))

        if hasattr(self, 'np_random') and self.np_random is not None:
            self.np_random.shuffle(pieces)
        else:
            random.shuffle(pieces)
        
        for sq in range(TOTAL_POSITIONS):
            self.board[sq] = pieces[sq]

        # 重置bitboards
        self.piece_bitboards.fill(0)
        self.revealed_bitboards.fill(0)
        self.hidden_bitboard = (1 << TOTAL_POSITIONS) - 1
        self.empty_bitboard = 0
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_board()
        return self.get_state(), {'action_mask': self.action_masks()}

    @cython.boundscheck(False)
    @cython.wraparound(False)  
    def get_state(self):
        cdef np.ndarray[np.float32_t, ndim=1] state = np.zeros(self.state_size, dtype=np.float32)
        cdef int my_player_idx = 1 if self.current_player == 1 else 0
        cdef int opponent_player_idx = 1 - my_player_idx
        
        # 填充棋子平面
        for pt_val in range(NUM_PIECE_TYPES):
            # My pieces
            bb = self.piece_bitboards[my_player_idx, pt_val]
            start_idx = self._my_pieces_plane_start_idx + pt_val * TOTAL_POSITIONS
            _fill_bitboard_plane(bb, state, start_idx)
            
            # Opponent pieces  
            bb = self.piece_bitboards[opponent_player_idx, pt_val]
            start_idx = self._opponent_pieces_plane_start_idx + pt_val * TOTAL_POSITIONS
            _fill_bitboard_plane(bb, state, start_idx)

        # Hidden and Empty planes
        _fill_bitboard_plane(self.hidden_bitboard, state, self._hidden_pieces_plane_start_idx)
        _fill_bitboard_plane(self.empty_bitboard, state, self._empty_plane_start_idx)
        
        # Scalar features
        cdef double score_norm = WINNING_SCORE if WINNING_SCORE > 0 else 1.0
        cdef double move_norm = MAX_CONSECUTIVE_MOVES if MAX_CONSECUTIVE_MOVES > 0 else 1.0
        cdef int scalar_idx = self._scalar_features_start_idx
        
        state[scalar_idx] = self.scores[self.current_player] / score_norm
        state[scalar_idx + 1] = self.scores[-self.current_player] / score_norm
        state[scalar_idx + 2] = self.move_counter / move_norm
        
        return state

    def step(self, action_index):
        cdef double reward = -0.0005
        coords = self.action_to_coords.get(action_index)
        if coords is None:
            raise ValueError(f"Invalid action_index: {action_index}")

        # 检查是否是翻棋动作
        if action_index < REVEAL_ACTIONS_COUNT:
            from_sq = POS_TO_SQ[coords]
            self._apply_reveal_update(from_sq)
            self.move_counter = 0
            reward += 0.0005
        else:
            from_sq = POS_TO_SQ[coords[0]]
            to_sq = POS_TO_SQ[coords[1]]
            attacker = self.board[from_sq]
            
            raw_reward = self._apply_move_action(from_sq, to_sq, attacker.piece_type.value)
            reward += raw_reward / WINNING_SCORE if WINNING_SCORE > 0 else raw_reward
        
        # 检查游戏结束
        terminated = False
        truncated = False
        winner = None

        if self.scores[1] >= WINNING_SCORE:
            winner = 1
            terminated = True
        elif self.scores[-1] >= WINNING_SCORE:
            winner = -1
            terminated = True
        elif self.move_counter >= MAX_CONSECUTIVE_MOVES:
            winner = 0
            truncated = True

        self.current_player = -self.current_player
        
        if not terminated and not truncated and np.sum(self.action_masks()) == 0:
            winner = -self.current_player
            terminated = True
            
        info = {'winner': winner, 'action_mask': self.action_masks()}
        
        if (terminated or truncated) and self.render_mode == "human":
            self.render()

        return self.get_state(), reward, terminated, truncated, info

    def _apply_reveal_update(self, from_sq):
        piece = self.board[from_sq]
        piece.revealed = True
        mask = np.uint64(1 << from_sq)
        self.hidden_bitboard ^= int(mask)
        
        player_idx = 1 if piece.player == 1 else 0
        self.revealed_bitboards[player_idx] = np.uint64(self.revealed_bitboards[player_idx] | mask)
        self.piece_bitboards[player_idx, piece.piece_type.value] = np.uint64(self.piece_bitboards[player_idx, piece.piece_type.value] | mask)

    def _apply_move_action(self, from_sq, to_sq, attacker_type_val):
        attacker = self.board[from_sq]
        defender = self.board[to_sq]
        attacker_mask = np.uint64(1 << from_sq)
        defender_mask = np.uint64(1 << to_sq)
        attacker_player_idx = 1 if attacker.player == 1 else 0

        # 移动到空格
        if defender is None:
            self.board[to_sq], self.board[from_sq] = attacker, None
            move_mask = attacker_mask | defender_mask
            self.piece_bitboards[attacker_player_idx, attacker_type_val] = np.uint64(self.piece_bitboards[attacker_player_idx, attacker_type_val] ^ move_mask)
            self.revealed_bitboards[attacker_player_idx] = np.uint64(self.revealed_bitboards[attacker_player_idx] ^ move_mask)
            self.empty_bitboard ^= int(move_mask)
            self.move_counter += 1
            return 0.0

        # 攻击/吃子
        defender_player_idx = 1 if defender.player == 1 else 0
        points = PIECE_VALUES[defender.piece_type.value]
        
        # 计算得分
        is_cannon_attack = attacker_type_val == CANNON
        if is_cannon_attack and attacker.player == defender.player:
            self.scores[-attacker.player] += points
            reward = -float(points)
        else:
            self.scores[attacker.player] += points
            reward = float(points)
        
        # 更新Bitboards
        combined_mask = attacker_mask | defender_mask
        self.piece_bitboards[attacker_player_idx, attacker_type_val] = np.uint64(self.piece_bitboards[attacker_player_idx, attacker_type_val] ^ combined_mask)
        self.revealed_bitboards[attacker_player_idx] = np.uint64(self.revealed_bitboards[attacker_player_idx] ^ combined_mask)
        
        if defender.revealed:
            self.piece_bitboards[defender_player_idx, defender.piece_type.value] = np.uint64(self.piece_bitboards[defender_player_idx, defender.piece_type.value] ^ defender_mask)
            self.revealed_bitboards[defender_player_idx] = np.uint64(self.revealed_bitboards[defender_player_idx] ^ defender_mask)
        else:
            self.hidden_bitboard ^= int(defender_mask)
        
        self.empty_bitboard |= int(attacker_mask)
        
        # 更新棋盘和死亡列表
        self.dead_pieces[defender.player].append(defender)
        self.board[to_sq], self.board[from_sq] = attacker, None
        self.move_counter = 0
        
        return reward

    def action_masks(self):
        return _compute_action_masks(
            self.current_player,
            self.hidden_bitboard,
            self.piece_bitboards,
            self.revealed_bitboards, 
            self.empty_bitboard,
            self.attack_tables,
            self.coords_to_action
        )

    def render(self):
        if self.render_mode != 'human': 
            return
            
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
                sq = POS_TO_SQ[r, c]
                if self.empty_bitboard & (1 << sq):
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

# Cython 优化的辅助函数
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_bitboard_plane(bitboard bb, np.ndarray[np.float32_t, ndim=1] plane, int start_idx):
    cdef int i
    for i in range(TOTAL_POSITIONS):
        if (bb >> i) & 1:
            plane[start_idx + i] = 1.0
        else:
            plane[start_idx + i] = 0.0

@cython.boundscheck(False)
@cython.wraparound(False)            
def _compute_action_masks(int current_player, bitboard hidden_bitboard,
                         np.ndarray[np.uint64_t, ndim=2] piece_bitboards,
                         np.ndarray[np.uint64_t, ndim=1] revealed_bitboards,
                         bitboard empty_bitboard, dict attack_tables, dict coords_to_action):
    
    cdef np.ndarray[np.int32_t, ndim=1] action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
    cdef int my_player_idx = 1 if current_player == 1 else 0
    cdef int opponent_player_idx = 1 - my_player_idx
    cdef int sq, from_sq, to_sq, action_index
    cdef bitboard temp_bb, target_bbs[7]
    cdef bitboard cumulative_targets = empty_bitboard
    
    # 1. 翻棋动作
    temp_bb = hidden_bitboard
    while temp_bb > 0:
        sq = trailing_zeros(temp_bb)
        action_mask[coords_to_action[SQ_TO_POS[sq]]] = 1
        temp_bb &= temp_bb - 1
        
    # 2. 普通移动/攻击 (简化版本)
    for pt_val in range(NUM_PIECE_TYPES-1, -1, -1):
        cumulative_targets |= piece_bitboards[opponent_player_idx, pt_val]
        target_bbs[pt_val] = cumulative_targets
    target_bbs[0] |= piece_bitboards[opponent_player_idx, 6]  # SOLDIER gets GENERAL
    target_bbs[6] &= ~piece_bitboards[opponent_player_idx, 0]  # GENERAL excludes SOLDIER

    # 简化的移动检查 (为了编译成功)
    for pt_val in range(NUM_PIECE_TYPES):
        if pt_val == 1:  # CANNON = 1
            continue
        from_sq_bb = piece_bitboards[my_player_idx, pt_val]
        if from_sq_bb == 0: 
            continue
            
        # 简单的邻接移动检查
        temp_bb = from_sq_bb
        while temp_bb > 0:
            from_sq = trailing_zeros(temp_bb)
            r, c = from_sq // 4, from_sq % 4
            
            # 检查四个方向
            if r > 0:  # 上
                to_sq = from_sq - 4
                if (target_bbs[pt_val] >> to_sq) & 1:
                    action_index = coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[to_sq]))
                    if action_index is not None:
                        action_mask[action_index] = 1
            if r < 3:  # 下
                to_sq = from_sq + 4
                if (target_bbs[pt_val] >> to_sq) & 1:
                    action_index = coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[to_sq]))
                    if action_index is not None:
                        action_mask[action_index] = 1
            if c > 0:  # 左
                to_sq = from_sq - 1
                if (target_bbs[pt_val] >> to_sq) & 1:
                    action_index = coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[to_sq]))
                    if action_index is not None:
                        action_mask[action_index] = 1
            if c < 3:  # 右
                to_sq = from_sq + 1
                if (target_bbs[pt_val] >> to_sq) & 1:
                    action_index = coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[to_sq]))
                    if action_index is not None:
                        action_mask[action_index] = 1
                        
            temp_bb &= temp_bb - 1

    # 3. 炮的攻击 (简化版本)
    my_cannons_bb = piece_bitboards[my_player_idx, 1]  # CANNON = 1
    if my_cannons_bb > 0:
        all_pieces_bb = ~empty_bitboard
        valid_cannon_targets = ~revealed_bitboards[my_player_idx]
        
        temp_bb = my_cannons_bb
        while temp_bb > 0:
            from_sq = trailing_zeros(temp_bb)
            
            for direction_idx in range(4):
                ray_bb = attack_tables['rays'][direction_idx][from_sq]
                blockers = ray_bb & all_pieces_bb
                if blockers == 0:
                    continue
                
                # 简化的炮攻击逻辑
                screen_sq = trailing_zeros(blockers) if direction_idx < 2 else trailing_zeros(blockers & -blockers)
                after_screen_ray = attack_tables['rays'][direction_idx][screen_sq]
                targets = after_screen_ray & all_pieces_bb
                if targets == 0:
                    continue

                target_sq = trailing_zeros(targets) if direction_idx < 2 else trailing_zeros(targets & -targets)
                    
                if (valid_cannon_targets >> target_sq) & 1:
                    from_pos = SQ_TO_POS[from_sq]
                    to_pos = SQ_TO_POS[target_sq]
                    action_index = coords_to_action.get((from_pos, to_pos))
                    if action_index is not None:
                        action_mask[action_index] = 1
            
            temp_bb &= temp_bb - 1
            
    return action_mask
