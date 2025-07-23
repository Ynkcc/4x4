# Game_cython.pyx - 整合优化版本
# distutils: language=c++

import random
from enum import Enum
import numpy as np
cimport numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces, Wrapper

# 导入cython的装饰器，用于微调性能
import cython
from libc.stdlib cimport rand, srand

np.import_array()
# ==============================================================================
# --- 类型定义 ---
# ==============================================================================
# 使用 cdef 定义Bitboard类型，long long 保证64位
ctypedef unsigned long long bitboard

# ==============================================================================
# --- Cython 优化: 位操作C函数 ---
# ==============================================================================

@cython.cfunc
@cython.inline
cdef bitboard ULL(int x):
    """一个帮助函数，用于创建一个Bitboard（无符号长整型），仅将第x位置为1。"""
    return 1ULL << x

@cython.cfunc
@cython.inline
cdef int trailing_zeros(bitboard bb):
    """计算末尾零的数量，等效于找到最低位1的位置 (LSB)"""
    if bb == 0:
        return 64
    cdef int count = 0
    while (bb & 1) == 0:
        bb >>= 1
        count += 1
    return count

@cython.cfunc
@cython.inline
cdef int msb_pos(bitboard n):
    """计算最高有效位的位置 (MSB), 类似于 Python 的 int.bit_length() - 1"""
    if n == 0: return -1
    cdef int pos = 0
    if n & 0xffffffff00000000ULL: pos += 32; n >>= 32
    if n & 0x00000000ffff0000ULL: pos += 16; n >>= 16
    if n & 0x000000000000ff00ULL: pos += 8;  n >>= 8
    if n & 0x00000000000000f0ULL: pos += 4;  n >>= 4
    if n & 0x000000000000000cULL: pos += 2;  n >>= 2
    if n & 0x0000000000000002ULL: pos += 1
    return pos

@cython.cfunc
@cython.inline
cdef int pop_lsb(bitboard* bb):
    """弹出并返回最低位的位置，同时更新bitboard"""
    if bb[0] == 0: return -1
    cdef int pos = trailing_zeros(bb[0])
    bb[0] &= bb[0] - 1
    return pos

# ==============================================================================
# --- Cython 优化: 类型定义和常量 ---
# ==============================================================================

cdef enum PieceTypeEnum:
    SOLDIER, CANNON, HORSE, CHARIOT, ELEPHANT, ADVISOR, GENERAL

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

cdef int[7] PIECE_VALUES = [4, 10, 10, 10, 10, 20, 30]
cdef int[7] PIECE_MAX_COUNTS = [2, 1, 1, 1, 1, 1, 1]

POS_TO_SQ = {(r, c): r * 4 + c for r in range(4) for c in range(4)}
SQ_TO_POS = {sq: (sq // 4, sq % 4) for sq in range(16)}

cdef bitboard FILE_A = ULL(0) | ULL(4) | ULL(8) | ULL(12)
cdef bitboard FILE_H = ULL(3) | ULL(7) | ULL(11) | ULL(15)
cdef bitboard NOT_FILE_A = ~FILE_A
cdef bitboard NOT_FILE_H = ~FILE_H

class Piece:
    """棋子对象，存储棋子本身的属性（类型，玩家，是否翻开）。"""
    def __init__(self, piece_type, player):
        self.piece_type, self.player, self.revealed = piece_type, player, False
    def __repr__(self):
        return f"{'R' if self.revealed else 'H'}_{'R' if self.player == 1 else 'B'}{self.piece_type.name}"

class PieceType(Enum):
    SOLDIER = 0; CANNON = 1; HORSE = 2; CHARIOT = 3
    ELEPHANT = 4; ADVISOR = 5; GENERAL = 6

cdef class GameEnvironment:
    cdef bitboard piece_bitboards[2][7]
    cdef bitboard revealed_bitboards[2]
    cdef bitboard hidden_bitboard, empty_bitboard
    cdef public int current_player
    cdef public int move_counter
    cdef public object scores, dead_pieces, board
    cdef public object action_to_coords, coords_to_action, attack_tables
    cdef public object np_random, observation_space, action_space, render_mode
    cdef public int ACTION_SPACE_SIZE, REVEAL_ACTIONS_COUNT, REGULAR_MOVE_ACTIONS_COUNT, MAX_CONSECUTIVE_MOVES
    cdef public int _my_pieces_plane_start_idx
    cdef public int _opponent_pieces_plane_start_idx
    cdef public int _hidden_pieces_plane_start_idx
    cdef public int _empty_plane_start_idx
    cdef public int _scalar_features_start_idx

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.ACTION_SPACE_SIZE = ACTION_SPACE_SIZE
        self.REVEAL_ACTIONS_COUNT = REVEAL_ACTIONS_COUNT
        self.REGULAR_MOVE_ACTIONS_COUNT = REGULAR_MOVE_ACTIONS_COUNT
        self.MAX_CONSECUTIVE_MOVES = MAX_CONSECUTIVE_MOVES
        
        my_pieces_plane_size = NUM_PIECE_TYPES * TOTAL_POSITIONS
        opponent_pieces_plane_size = NUM_PIECE_TYPES * TOTAL_POSITIONS
        hidden_pieces_plane_size = TOTAL_POSITIONS
        empty_plane_size = TOTAL_POSITIONS
        scalar_features_size = 3
        state_size = ( my_pieces_plane_size + opponent_pieces_plane_size + 
                            hidden_pieces_plane_size + empty_plane_size + scalar_features_size )

        self._my_pieces_plane_start_idx = 0
        self._opponent_pieces_plane_start_idx = my_pieces_plane_size
        self._hidden_pieces_plane_start_idx = my_pieces_plane_size + opponent_pieces_plane_size
        self._empty_plane_start_idx = self._hidden_pieces_plane_start_idx + hidden_pieces_plane_size
        self._scalar_features_start_idx = self._empty_plane_start_idx + empty_plane_size

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self.board = np.empty(TOTAL_POSITIONS, dtype=object)
        self.dead_pieces = {-1: [], 1: []}
        self.scores = {-1: 0, 1: 0}

        self.attack_tables = {}
        self.action_to_coords = {}
        self.coords_to_action = {}
        self._initialize_lookup_tables()
    
    cpdef _initialize_lookup_tables(self):
        ray_attacks = [[0] * TOTAL_POSITIONS for _ in range(4)]
        cdef int sq, r, c, i
        for sq in range(TOTAL_POSITIONS):
            r, c = sq // 4, sq % 4
            for i in range(r - 1, -1, -1): ray_attacks[0][sq] |= ULL(i * 4 + c)
            for i in range(r + 1, 4):      ray_attacks[1][sq] |= ULL(i * 4 + c)
            for i in range(c - 1, -1, -1): ray_attacks[2][sq] |= ULL(r * 4 + i)
            for i in range(c + 1, 4):      ray_attacks[3][sq] |= ULL(r * 4 + i)
        self.attack_tables['rays'] = ray_attacks
        
        cdef int action_idx = 0
        for sq in range(TOTAL_POSITIONS):
            self.action_to_coords[action_idx] = SQ_TO_POS[sq]
            self.coords_to_action[SQ_TO_POS[sq]] = action_idx
            action_idx += 1
            
        cdef int from_sq, to_sq
        for from_sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[from_sq]
            from_pos = (r, c)
            if r > 0: to_sq = from_sq - 4; self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq]); self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx; action_idx += 1
            if r < 3: to_sq = from_sq + 4; self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq]); self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx; action_idx += 1
            if c > 0: to_sq = from_sq - 1; self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq]); self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx; action_idx += 1
            if c < 3: to_sq = from_sq + 1; self.action_to_coords[action_idx] = (from_pos, SQ_TO_POS[to_sq]); self.coords_to_action[(from_pos, SQ_TO_POS[to_sq])] = action_idx; action_idx += 1

        cdef int r1, c1, r2, c2
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                from_pos = (r1, c1)
                for c2 in range(c1 + 2, BOARD_COLS): self.action_to_coords[action_idx] = (from_pos, (r1, c2)); self.coords_to_action[(from_pos, (r1, c2))] = action_idx; action_idx += 1
                for c2 in range(c1 - 2, -1, -1): self.action_to_coords[action_idx] = (from_pos, (r1, c2)); self.coords_to_action[(from_pos, (r1, c2))] = action_idx; action_idx += 1
                for r2 in range(r1 + 2, BOARD_ROWS): self.action_to_coords[action_idx] = (from_pos, (r2, c1)); self.coords_to_action[(from_pos, (r2, c1))] = action_idx; action_idx += 1
                for r2 in range(r1 - 2, -1, -1): self.action_to_coords[action_idx] = (from_pos, (r2, c1)); self.coords_to_action[(from_pos, (r2, c1))] = action_idx; action_idx += 1

    @cython.cfunc
    cdef void _initialize_board(self):
        pieces = []
        for pt in PieceType:
            count = PIECE_MAX_COUNTS[pt.value]
            for p in [1, -1]:
                 for _ in range(count):
                    pieces.append(Piece(pt, p))

        if self.np_random is not None:
            self.np_random.shuffle(pieces)
        else:
            random.shuffle(pieces)
        
        cdef int sq
        for sq in range(TOTAL_POSITIONS):
            self.board[sq] = pieces[sq]

        cdef int player_idx, pt_idx
        for player_idx in range(2):
            self.revealed_bitboards[player_idx] = 0
            for pt_idx in range(NUM_PIECE_TYPES):
                self.piece_bitboards[player_idx][pt_idx] = 0
        
        self.hidden_bitboard = (1ULL << TOTAL_POSITIONS) - 1
        self.empty_bitboard = 0
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

    @property
    def unwrapped(self):
        """返回环境本身，用于 Gymnasium 兼容性"""
        return self

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._initialize_board()
        return self.get_state(), {'action_mask': self.action_masks()}
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_state(self):
        cdef np.ndarray[np.float32_t, ndim=1] state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        cdef int my_player = self.current_player
        cdef int opponent_player = -self.current_player
        cdef int my_player_idx = 1 if my_player == 1 else 0
        cdef int opponent_player_idx = 1 if opponent_player == 1 else 0
        cdef int pt_val, start_idx
        cdef bitboard bb

        for pt_val in range(NUM_PIECE_TYPES):
            bb = self.piece_bitboards[my_player_idx][pt_val]
            start_idx = self._my_pieces_plane_start_idx + pt_val * TOTAL_POSITIONS
            self._bitboard_to_plane(bb, state, start_idx)
            
            bb = self.piece_bitboards[opponent_player_idx][pt_val]
            start_idx = self._opponent_pieces_plane_start_idx + pt_val * TOTAL_POSITIONS
            self._bitboard_to_plane(bb, state, start_idx)

        self._bitboard_to_plane(self.hidden_bitboard, state, self._hidden_pieces_plane_start_idx)
        self._bitboard_to_plane(self.empty_bitboard, state, self._empty_plane_start_idx)
        
        cdef double score_norm = WINNING_SCORE if WINNING_SCORE > 0 else 1.0
        cdef double move_norm = MAX_CONSECUTIVE_MOVES if MAX_CONSECUTIVE_MOVES > 0 else 1.0
        cdef int scalar_idx = self._scalar_features_start_idx
        
        state[scalar_idx] = self.scores[my_player] / score_norm
        state[scalar_idx + 1] = self.scores[opponent_player] / score_norm
        state[scalar_idx + 2] = self.move_counter / move_norm
        
        return state

    @cython.cfunc
    @cython.inline
    cdef void _bitboard_to_plane(self, bitboard bb, np.ndarray[np.float32_t, ndim=1] plane, int start_idx):
        cdef int i
        for i in range(TOTAL_POSITIONS):
            if (bb >> i) & 1:
                plane[start_idx + i] = 1.0
            else:
                plane[start_idx + i] = 0.0

    def step(self, int action_index):
        cdef double reward = -0.0005
        cdef object coords = self.action_to_coords.get(action_index)
        if coords is None:
             raise ValueError(f"Invalid action_index: {action_index}")

        cdef int from_sq, to_sq
        cdef object attacker
        cdef double raw_reward

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
        
        cdef bint terminated = False
        cdef bint truncated = False
        cdef object winner = None

        if self.scores[1] >= WINNING_SCORE:
            winner = 1; terminated = True
        elif self.scores[-1] >= WINNING_SCORE:
            winner = -1; terminated = True
        elif self.move_counter >= MAX_CONSECUTIVE_MOVES:
            winner = 0; truncated = True

        self.current_player = -self.current_player
        
        if not terminated and not truncated and np.sum(self.action_masks()) == 0:
            winner = -self.current_player; terminated = True
            
        info = {'winner': winner, 'action_mask': self.action_masks()}
        
        if (terminated or truncated) and self.render_mode == "human":
            self.render()

        return self.get_state(), np.float32(reward), terminated, truncated, info

    @cython.cfunc
    cdef void _apply_reveal_update(self, int from_sq):
        cdef object piece = self.board[from_sq]
        piece.revealed = True
        cdef bitboard mask = ULL(from_sq)
        self.hidden_bitboard ^= mask
        
        cdef int player_idx = 1 if piece.player == 1 else 0
        self.revealed_bitboards[player_idx] |= mask
        self.piece_bitboards[player_idx][piece.piece_type.value] |= mask

    @cython.cfunc
    cdef double _apply_move_action(self, int from_sq, int to_sq, int attacker_type_val):
        cdef object attacker = self.board[from_sq]
        cdef object defender = self.board[to_sq]
        cdef bitboard attacker_mask = ULL(from_sq)
        cdef bitboard defender_mask = ULL(to_sq)
        cdef int attacker_player_idx = 1 if attacker.player == 1 else 0
        cdef bitboard move_mask
        cdef int defender_player_idx
        cdef int points
        cdef double reward = 0.0

        if defender is None:
            self.board[to_sq], self.board[from_sq] = attacker, None
            move_mask = attacker_mask | defender_mask
            self.piece_bitboards[attacker_player_idx][attacker_type_val] ^= move_mask
            self.revealed_bitboards[attacker_player_idx] ^= move_mask
            self.empty_bitboard ^= move_mask
            self.move_counter += 1
            return 0.0

        defender_player_idx = 1 if defender.player == 1 else 0
        points = PIECE_VALUES[defender.piece_type.value]

        is_cannon_attack = attacker_type_val == 1
        if is_cannon_attack and attacker.player == defender.player:
            self.scores[-attacker.player] += points
            reward = -float(points)
        else:
            self.scores[attacker.player] += points
            reward = float(points)
        
        self.piece_bitboards[attacker_player_idx][attacker_type_val] ^= (attacker_mask | defender_mask)
        self.revealed_bitboards[attacker_player_idx] ^= (attacker_mask | defender_mask)
        
        if defender.revealed:
            self.piece_bitboards[defender_player_idx][defender.piece_type.value] ^= defender_mask
            self.revealed_bitboards[defender_player_idx] ^= defender_mask
        else:
            self.hidden_bitboard ^= defender_mask
        
        self.empty_bitboard |= attacker_mask
        
        self.dead_pieces[defender.player].append(defender)
        self.board[to_sq], self.board[from_sq] = attacker, None
        self.move_counter = 0
        
        return reward

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def action_masks(self):
        cdef np.ndarray[int, ndim=1] action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        cdef int my_player_idx = 1 if self.current_player == 1 else 0
        cdef int opponent_player_idx = 1 - my_player_idx
        cdef int sq, action_index, from_sq, to_sq
        cdef object from_pos, to_pos
        cdef bitboard temp_bb = self.hidden_bitboard
        cdef bitboard target_bbs[7]
        cdef bitboard cumulative_targets = self.empty_bitboard
        cdef int shift, pt_val, i
        cdef bitboard wrap_check, from_sq_bb, temp_from_bb, potential_to_sq_bb, actual_to_sq_bb
        cdef bitboard my_cannons_bb
        cdef bitboard all_pieces_bb, valid_cannon_targets
        cdef bitboard temp_cannons_bb
        cdef int direction_idx
        cdef bitboard ray_bb, blockers, after_screen_ray, targets
        cdef int screen_sq, target_sq
        cdef int shifts[4]
        cdef bitboard wrap_checks[4]

        while temp_bb > 0:
            sq = pop_lsb(&temp_bb)
            if sq != -1:
                action_mask[self.coords_to_action[SQ_TO_POS[sq]]] = 1
            
        for pt_val in range(NUM_PIECE_TYPES):
            cumulative_targets |= self.piece_bitboards[opponent_player_idx][pt_val]
            target_bbs[pt_val] = cumulative_targets
        target_bbs[0] |= self.piece_bitboards[opponent_player_idx][6]
        target_bbs[6] &= ~self.piece_bitboards[opponent_player_idx][0]

        shifts[0] = -4; shifts[1] = 4; shifts[2] = -1; shifts[3] = 1
        wrap_checks[0] = 0; wrap_checks[1] = 0; wrap_checks[2] = NOT_FILE_A; wrap_checks[3] = NOT_FILE_H

        for pt_val in range(NUM_PIECE_TYPES):
            if pt_val == 1: continue
            from_sq_bb = self.piece_bitboards[my_player_idx][pt_val]
            if from_sq_bb == 0: continue
            
            for i in range(4):
                shift = shifts[i]
                wrap_check = wrap_checks[i]
                
                temp_from_bb = from_sq_bb & wrap_check if wrap_check != 0 else from_sq_bb
                if temp_from_bb == 0: continue

                if shift > 0:
                    potential_to_sq_bb = temp_from_bb << shift
                else:
                    potential_to_sq_bb = temp_from_bb >> -shift
                
                actual_to_sq_bb = potential_to_sq_bb & target_bbs[pt_val]
                
                temp_bb = actual_to_sq_bb
                while temp_bb > 0:
                    to_sq = pop_lsb(&temp_bb)
                    if to_sq != -1:
                        from_sq = to_sq - shift
                        action_index = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[to_sq]))
                        if action_index is not None: action_mask[action_index] = 1

        my_cannons_bb = self.piece_bitboards[my_player_idx][1]
        if my_cannons_bb > 0:
            all_pieces_bb = ~self.empty_bitboard
            valid_cannon_targets = ~self.revealed_bitboards[my_player_idx]
            
            temp_cannons_bb = my_cannons_bb
            while temp_cannons_bb > 0:
                from_sq = pop_lsb(&temp_cannons_bb)
                if from_sq == -1: continue
                
                for direction_idx in range(4):
                    ray_bb = self.attack_tables['rays'][direction_idx][from_sq]
                    blockers = ray_bb & all_pieces_bb
                    if blockers == 0: continue
                    
                    if direction_idx == 0 or direction_idx == 2:
                        screen_sq = msb_pos(blockers)
                    else:
                        screen_sq = trailing_zeros(blockers)
                    
                    if screen_sq == -1: continue
                    after_screen_ray = self.attack_tables['rays'][direction_idx][screen_sq]
                    targets = after_screen_ray & all_pieces_bb
                    if targets == 0: continue

                    if direction_idx == 0 or direction_idx == 2:
                        target_sq = msb_pos(targets)
                    else:
                        target_sq = trailing_zeros(targets)
                        
                    if target_sq != -1 and (ULL(target_sq) & valid_cannon_targets):
                        from_pos = SQ_TO_POS[from_sq]
                        to_pos = SQ_TO_POS[target_sq]
                        action_index = self.coords_to_action.get((from_pos, to_pos))
                        if action_index is not None: action_mask[action_index] = 1
            
        return action_mask

    def render(self):
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
                sq = POS_TO_SQ[r, c]
                if self.empty_bitboard & ULL(sq):
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

    def get_hidden_bitboard(self):
        return int(self.hidden_bitboard)
    
    def get_empty_bitboard(self):
        return int(self.empty_bitboard)
    
    def get_piece_bitboard(self, player, piece_type):
        cdef int player_idx = 1 if player == 1 else 0
        return int(self.piece_bitboards[player_idx][piece_type])
    
    def get_revealed_bitboard(self, player):
        cdef int player_idx = 1 if player == 1 else 0
        return int(self.revealed_bitboards[player_idx])

# ==============================================================================
# --- Gymnasium 兼容性包装器 ---
# ==============================================================================

class BanqiEnvironment(Wrapper):
    """
    Gymnasium兼容性包装器，现在继承自 gym.Wrapper 以便与 SubprocVecEnv 完美配合。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(self, render_mode=None):
        game_env = GameEnvironment(render_mode=render_mode)
        super().__init__(game_env)
    
    def action_masks(self):
        return self.env.action_masks()
    
    def get_state(self):
        return self.env.get_state()
        
    def get_wrapper_attr(self, name: str):
        """
        手动实现 get_wrapper_attr 以停止在非Gym环境的基类上进行递归。
        这可以防止在 stable-baselines3 的 SubprocVecEnv 中出现AttributeError。
        """
        if hasattr(self, name):
            return getattr(self, name)
        elif hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' and its base environment '{type(self.env).__name__}' do not have an attribute called '{name}'")


# 为了向后兼容，保持原有的类名
GameEnvironment_Wrapper = BanqiEnvironment