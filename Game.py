# Game.py 

import random
from enum import Enum
import numpy as np
import copy
import gymnasium as gym
from gymnasium import spaces

# 定义棋子类型的枚举
class PieceType(Enum):
    A = 0  # 兵/卒
    B = 1  # 炮/炮
    C = 2  # 马/傌
    D = 3  # 车/俥
    E = 4  # 象/相
    F = 5  # 士/仕
    G = 6  # 将/帅

class Piece:
    def __init__(self, piece_type, player):
        self.piece_type = piece_type
        self.player = player # -1 表示黑方, 1 表示红方
        self.revealed = False

    def __repr__(self):
        """提供棋子对象的字符串表示，方便调试。"""
        state = "R" if self.revealed else "H" # Revealed or Hidden
        player_char = "B" if self.player == -1 else "R" # Black or Red
        return f"{state}_{player_char}{self.piece_type.name}"

class GameEnvironment(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    BOARD_ROWS, BOARD_COLS = 4, 4
    NUM_PIECE_TYPES = 7
    TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS
    ACTION_SPACE_SIZE = TOTAL_POSITIONS * 5

    MAX_CONSECUTIVE_MOVES = 40
    WINNING_SCORE = 60

    PIECE_VALUES = {
        PieceType.A: 4, PieceType.B: 10, PieceType.C: 10,
        PieceType.D: 10, PieceType.E: 10, PieceType.F: 20, PieceType.G: 30
    }
    PIECE_MAX_COUNTS = {
        PieceType.A: 2, PieceType.B: 1, PieceType.C: 1,
        PieceType.D: 1, PieceType.E: 1, PieceType.F: 1, PieceType.G: 1
    }

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        dead_piece_counts_size_per_player = sum(self.PIECE_MAX_COUNTS.values())
        self.state_size = (
            (self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS * 2) + self.TOTAL_POSITIONS * 2 +
            (dead_piece_counts_size_per_player * 2) + 2 + 1 + self.ACTION_SPACE_SIZE * 2
        )
                          
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)

        self.board = None
        self.dead_pieces = {-1: [], 1: []} 
        self.current_player = 1 
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}
        self.unrevealed_pieces_pos = set() 
        self.player_pieces_pos = {-1: set(), 1: set()}

        self._state_vector_p1 = np.zeros(self.state_size, dtype=np.float32)
        self._state_vector_p_neg1 = np.zeros(self.state_size, dtype=np.float32)

        self._my_pieces_plane_start_idx = 0
        self._opponent_pieces_plane_start_idx = self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS
        self._hidden_pieces_plane_start_idx = self._opponent_pieces_plane_start_idx + self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS
        self._threat_plane_start_idx = self._hidden_pieces_plane_start_idx + self.TOTAL_POSITIONS
        self._my_dead_count_start_idx = self._threat_plane_start_idx + self.TOTAL_POSITIONS
        self._opponent_dead_count_start_idx = self._my_dead_count_start_idx + dead_piece_counts_size_per_player
        self._scores_start_idx = self._opponent_dead_count_start_idx + dead_piece_counts_size_per_player
        self._move_counter_idx = self._scores_start_idx + 2
        self._opportunity_vector_start_idx = self._move_counter_idx + 1
        self._threat_vector_start_idx = self._opportunity_vector_start_idx + self.ACTION_SPACE_SIZE

    def _initialize_board(self):
        """初始化棋盘基础状态。"""
        self.board = np.empty((self.BOARD_ROWS, self.BOARD_COLS), dtype=object)
        pieces = []
        for piece_type, count in self.PIECE_MAX_COUNTS.items():
            for _ in range(count):
                pieces.extend([Piece(piece_type, -1), Piece(piece_type, 1)])
        self.np_random.shuffle(pieces)
        
        self.unrevealed_pieces_pos = set(divmod(i, self.BOARD_COLS) for i in range(self.TOTAL_POSITIONS))
        self.player_pieces_pos = {-1: set(), 1: set()}
        
        for i, pos in enumerate(self.unrevealed_pieces_pos):
            self.board[pos] = pieces[i]
        
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1 
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

    def reset(self, seed=None, options=None):
        """重置游戏，并从零开始构建初始状态。"""
        super().reset(seed=seed)
        self._initialize_board()
        action_mask = self._build_state_from_scratch()
        observation = self.get_state()
        info = {'action_mask': action_mask}
        return observation, info
    
    def get_state(self):
        """直接返回当前玩家对应的预计算好的状态向量。"""
        return self._state_vector_p1.copy() if self.current_player == 1 else self._state_vector_p_neg1.copy()

    def step(self, action_index):
        """执行动作，并调用增量更新函数来改变状态。"""
        # 1. 执行动作并完成状态的增量更新
        raw_reward = self._execute_action_and_update_state(action_index)
        
        # 2. 计算归一化奖励
        reward = -0.0005 # 时间惩罚
        if raw_reward != 0:
            reward += raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward

        # 3. 检查游戏结束条件
        terminated, truncated, winner = False, False, None
        if self.scores[1] >= self.WINNING_SCORE: winner, terminated = 1, True
        elif self.scores[-1] >= self.WINNING_SCORE: winner, terminated = -1, True
        elif self.move_counter >= self.MAX_CONSECUTIVE_MOVES: winner, truncated = 0, True

        # 4. 切换玩家并获取下一回合的动作掩码
        self.current_player = -self.current_player
        action_mask = self._update_complex_state_vectors() # 只更新需重算的复杂部分
        
        # 5. 检查新玩家是否无棋可走
        if not terminated and not truncated and np.sum(action_mask) == 0:
            winner, terminated = -self.current_player, True
            
        observation = self.get_state()
        info = {'winner': winner, 'action_mask': action_mask}
        
        if (terminated or truncated) and self.render_mode == "human": self.render()
        return observation, reward, terminated, truncated, info

    def _execute_action_and_update_state(self, action_index):
        """
        核心执行函数：执行动作，并对简单状态进行直接、增量的修改。
        返回原始奖励值。
        """
        pos_idx, action_sub_idx = divmod(action_index, 5)
        from_pos = divmod(pos_idx, self.BOARD_COLS)
        raw_reward = 0

        # --- 分类执行动作并更新棋盘基础状态 ---
        if action_sub_idx == 4: # 翻棋
            piece = self.board[from_pos]
            self._apply_reveal_update(from_pos, piece)
            self.move_counter = 0
        else: # 移动或攻击
            attacker = self.board[from_pos]
            dr, dc = [(-1,0), (1,0), (0,-1), (0,1)][action_sub_idx]
            
            if attacker.piece_type == PieceType.B: # 炮
                to_pos = self._get_cannon_target(from_pos, (dr, dc))
                defender = self.board[to_pos]
                raw_reward = self._apply_attack_update(from_pos, to_pos, attacker, defender)
            else: # 普通棋子
                to_pos = (from_pos[0] + dr, from_pos[1] + dc)
                defender = self.board[to_pos]
                if defender is None: # 移动
                    self._apply_move_update(from_pos, to_pos, attacker)
                    self.move_counter += 1
                else: # 攻击
                    raw_reward = self._apply_attack_update(from_pos, to_pos, attacker, defender)
                    self.move_counter = 0
        
        # --- 更新计数器和分数状态 ---
        score_norm = self.WINNING_SCORE if self.WINNING_SCORE > 0 else 1.0
        move_norm = self.MAX_CONSECUTIVE_MOVES if self.MAX_CONSECUTIVE_MOVES > 0 else 1.0
        for vec in [self._state_vector_p1, self._state_vector_p_neg1]:
            p = 1 if np.array_equal(vec, self._state_vector_p1) else -1
            vec[self._scores_start_idx] = self.scores[p] / score_norm
            vec[self._scores_start_idx + 1] = self.scores[-p] / score_norm
            vec[self._move_counter_idx] = self.move_counter / move_norm

        return raw_reward

    def _apply_reveal_update(self, pos, piece):
        """增量更新：翻棋"""
        piece.revealed = True
        self.unrevealed_pieces_pos.remove(pos)
        self.player_pieces_pos[piece.player].add(pos)
        
        pos_flat = pos[0] * self.BOARD_COLS + pos[1]
        
        # 对于双方视角，都将该位置从“暗棋”层移到对应的“明棋”层
        for vec in [self._state_vector_p1, self._state_vector_p_neg1]:
            p = 1 if np.array_equal(vec, self._state_vector_p1) else -1
            vec[self._hidden_pieces_plane_start_idx + pos_flat] = 0
            if piece.player == p: # 己方棋子
                vec[self._my_pieces_plane_start_idx + piece.piece_type.value * self.TOTAL_POSITIONS + pos_flat] = 1
            else: # 对方棋子
                vec[self._opponent_pieces_plane_start_idx + piece.piece_type.value * self.TOTAL_POSITIONS + pos_flat] = 1

    def _apply_move_update(self, from_pos, to_pos, piece):
        """增量更新：移动"""
        self.board[to_pos] = piece
        self.board[from_pos] = None
        self.player_pieces_pos[piece.player].remove(from_pos)
        self.player_pieces_pos[piece.player].add(to_pos)

        from_flat = from_pos[0] * self.BOARD_COLS + from_pos[1]
        to_flat = to_pos[0] * self.BOARD_COLS + to_pos[1]

        # 对于双方视角，都只更新棋子位置层的两个bit
        for vec in [self._state_vector_p1, self._state_vector_p_neg1]:
            p = 1 if np.array_equal(vec, self._state_vector_p1) else -1
            plane_start_idx = self._my_pieces_plane_start_idx if piece.player == p else self._opponent_pieces_plane_start_idx
            offset = piece.piece_type.value * self.TOTAL_POSITIONS
            vec[plane_start_idx + offset + from_flat] = 0
            vec[plane_start_idx + offset + to_flat] = 1

    def _apply_attack_update(self, from_pos, to_pos, attacker, defender):
        """增量更新：攻击"""
        # 1. 更新基础棋盘状态
        original_dead_count = len(self.dead_pieces[defender.player])
        self.dead_pieces[defender.player].append(defender)
        
        points = self.PIECE_VALUES[defender.piece_type]
        raw_reward = 0
        if attacker.player != defender.player:
            self.scores[attacker.player] += points
            raw_reward = float(points) if defender.revealed else 0
        else: # 炮误伤
            self.scores[-attacker.player] += points
            raw_reward = -float(points)

        self.board[to_pos] = attacker
        self.board[from_pos] = None
        self.player_pieces_pos[attacker.player].remove(from_pos)
        self.player_pieces_pos[attacker.player].add(to_pos)

        if defender.revealed: self.player_pieces_pos[defender.player].remove(to_pos)
        else: self.unrevealed_pieces_pos.remove(to_pos)
        
        # 2. 增量更新状态向量
        from_flat = from_pos[0] * self.BOARD_COLS + from_pos[1]
        to_flat = to_pos[0] * self.BOARD_COLS + to_pos[1]
        dead_piece_offset = sum(self.PIECE_MAX_COUNTS[pt] for pt in PieceType if pt.value < defender.piece_type.value)

        for vec in [self._state_vector_p1, self._state_vector_p_neg1]:
            p = 1 if np.array_equal(vec, self._state_vector_p1) else -1
            # 2a. 更新攻击方位置
            att_plane_start = self._my_pieces_plane_start_idx if attacker.player == p else self._opponent_pieces_plane_start_idx
            att_offset = attacker.piece_type.value * self.TOTAL_POSITIONS
            vec[att_plane_start + att_offset + from_flat] = 0
            vec[att_plane_start + att_offset + to_flat] = 1
            
            # 2b. 更新被吃方位置（从明棋或暗棋层移除）
            if defender.revealed:
                def_plane_start = self._my_pieces_plane_start_idx if defender.player == p else self._opponent_pieces_plane_start_idx
                def_offset = defender.piece_type.value * self.TOTAL_POSITIONS
                vec[def_plane_start + def_offset + to_flat] = 0
            else:
                vec[self._hidden_pieces_plane_start_idx + to_flat] = 0
            
            # 2c. 更新死亡棋子计数
            dead_plane_start = self._my_dead_count_start_idx if defender.player == p else self._opponent_dead_count_start_idx
            vec[dead_plane_start + dead_piece_offset + original_dead_count] = 1

        return raw_reward

    def _update_complex_state_vectors(self):
        """
        只重新计算并更新那些具有非局部效应的复杂状态。
        返回当前玩家的动作掩码。
        """
        final_action_mask = None
        for player_perspective in [1, -1]:
            state_vector = self._state_vector_p1 if player_perspective == 1 else self._state_vector_p_neg1
            
            # 更新威胁图层
            threat_plane = self._get_threat_plane(-player_perspective)
            state_vector[self._threat_plane_start_idx : self._my_dead_count_start_idx] = threat_plane
            
            # 更新行动向量
            opportunity_vector, threat_vector, action_mask = self._get_action_vectors(player_perspective)
            state_vector[self._opportunity_vector_start_idx : self._threat_vector_start_idx] = opportunity_vector
            state_vector[self._threat_vector_start_idx:] = threat_vector
            
            if player_perspective == self.current_player:
                final_action_mask = action_mask
        return final_action_mask

    def _build_state_from_scratch(self):
        """在reset时，从零开始完整地构建双方的状态向量。"""
        for p in [1, -1]:
            state_vector = self._state_vector_p1 if p == 1 else self._state_vector_p_neg1
            # 初始化棋子位置
            for r, c in self.unrevealed_pieces_pos:
                pos_flat = r * self.BOARD_COLS + c
                state_vector[self._hidden_pieces_plane_start_idx + pos_flat] = 1.0
        # 其他所有状态在游戏开始时都为0，由zeros()初始化完成
        # 初始化后，计算所有复杂状态
        return self._update_complex_state_vectors()

    # --- 以下为各类辅助函数，与之前版本基本一致 ---
    
    def _get_threat_plane(self, player):
        threat_plane = np.zeros(self.TOTAL_POSITIONS, dtype=np.float32)
        for r_from, c_from in self.player_pieces_pos[player]:
            piece = self.board[r_from, c_from]
            if not piece.revealed: continue
            if piece.piece_type == PieceType.B:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    target_pos = self._get_cannon_target((r_from, c_from), (dr, dc))
                    if target_pos: threat_plane[target_pos[0] * self.BOARD_COLS + target_pos[1]] = 1.0
            else:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r_to, c_to = r_from + dr, c_from + dc
                    if (0 <= r_to < self.BOARD_ROWS and 0 <= c_to < self.BOARD_COLS):
                        threat_plane[r_to * self.BOARD_COLS + c_to] = 1.0
        return threat_plane

    def _is_position_threatened_after_move(self, piece_to_move, to_pos, from_pos, by_player):
        for r_opp, c_opp in self.player_pieces_pos[by_player]:
            opp_piece = self.board[r_opp, c_opp]
            if not opp_piece.revealed: continue
            
            if opp_piece.piece_type != PieceType.B:
                if abs(to_pos[0] - r_opp) + abs(to_pos[1] - c_opp) == 1 and self.can_attack(opp_piece, piece_to_move):
                    return True
            else:
                if not (r_opp == to_pos[0] or c_opp == to_pos[1]): continue
                mount_count = 0
                if r_opp == to_pos[0]:
                    start_col, end_col = min(c_opp, to_pos[1]), max(c_opp, to_pos[1])
                    for c in range(start_col + 1, end_col):
                        if (r_opp, c) == from_pos: continue
                        if self.board[r_opp, c] is not None: mount_count += 1
                else:
                    start_row, end_row = min(r_opp, to_pos[0]), max(r_opp, to_pos[0])
                    for r in range(start_row + 1, end_row):
                        if (r, c_opp) == from_pos: continue
                        if self.board[r, c_opp] is not None: mount_count += 1
                if mount_count == 1: return True
        return False

    def _get_action_vectors(self, for_player):
        action_mask = self.action_masks(for_player)
        opportunity_vector = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
        threat_vector = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
        
        valid_action_indices = np.where(action_mask == 1)[0]
        for action_index in valid_action_indices:
            pos_idx, action_sub_idx = divmod(action_index, 5)
            r_from, c_from = divmod(pos_idx, self.BOARD_COLS)
            from_pos = (r_from, c_from)
            if action_sub_idx == 4: continue

            moving_piece = self.board[r_from, c_from]
            is_capture = False
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action_sub_idx]

            if moving_piece.piece_type == PieceType.B:
                to_pos = self._get_cannon_target(from_pos, (dr, dc))
                if to_pos: is_capture = True
            else:
                to_pos = (r_from + dr, c_from + dc)
                if self.board[to_pos] is not None: is_capture = True
            
            if is_capture: opportunity_vector[action_index] = 1.0
            if self._is_position_threatened_after_move(moving_piece, to_pos, from_pos, -for_player):
                threat_vector[action_index] = 1.0
                
        return opportunity_vector, threat_vector, action_mask

    def action_masks(self, for_player=None):
        player = for_player if for_player is not None else self.current_player
        valid_actions_arr = np.zeros(self.ACTION_SPACE_SIZE, dtype=int)

        if player == self.current_player:
            for r_from, c_from in self.unrevealed_pieces_pos:
                pos_idx = r_from * self.BOARD_COLS + c_from
                valid_actions_arr[pos_idx * 5 + 4] = 1

        for r_from, c_from in self.player_pieces_pos[player]:
            piece = self.board[r_from, c_from]
            pos_idx = r_from * self.BOARD_COLS + c_from
            if piece.piece_type == PieceType.B:
                for move_sub_idx, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    target_pos = self._get_cannon_target((r_from, c_from), (dr, dc))
                    if target_pos and (not self.board[target_pos].revealed or self.board[target_pos].player != player):
                         valid_actions_arr[pos_idx * 5 + move_sub_idx] = 1
            else:
                for move_sub_idx, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    r_to, c_to = r_from + dr, c_from + dc
                    if not (0 <= r_to < self.BOARD_ROWS and 0 <= c_to < self.BOARD_COLS): continue
                    
                    target_piece = self.board[r_to, c_to]
                    action_idx = pos_idx * 5 + move_sub_idx
                    if target_piece is None: valid_actions_arr[action_idx] = 1
                    elif target_piece.player != player and target_piece.revealed and self.can_attack(piece, target_piece):
                        valid_actions_arr[action_idx] = 1
        return valid_actions_arr

    def can_attack(self, attacker, defender):
        if attacker.piece_type == PieceType.G and defender.piece_type == PieceType.A: return False
        if attacker.piece_type == PieceType.A and defender.piece_type == PieceType.G: return True
        return attacker.piece_type.value >= defender.piece_type.value

    def render(self):
        if self.render_mode is None: return
        red_map = {PieceType.G: "帥", PieceType.F: "仕", PieceType.E: "相", PieceType.D: "俥", PieceType.C: "傌", PieceType.B: "炮", PieceType.A: "兵"}
        black_map = {PieceType.G: "將", PieceType.F: "士", PieceType.E: "象", PieceType.D: "車", PieceType.C: "馬", PieceType.B: "炮", PieceType.A: "卒"}
        print("-" * (self.BOARD_COLS * 5 + 1))
        for r in range(self.BOARD_ROWS):
            print("|", end="")
            for c in range(self.BOARD_COLS):
                p = self.board[r,c]
                if p is None: print("    |", end="")
                elif not p.revealed: print(f" \033[90m暗\033[0m  |", end="")
                elif p.player == 1: print(f" \033[91m{red_map[p.piece_type]}\033[0m  |", end="")
                else: print(f" \033[94m{black_map[p.piece_type]}\033[0m  |", end="")
            print()
        print("-" * (self.BOARD_COLS * 5 + 1))
        p_str = "\033[91m红方\033[0m" if self.current_player == 1 else "\033[94m黑方\033[0m"
        print(f"Player: {p_str}, Scores: R={self.scores[1]} B={self.scores[-1]}, Moves: {self.move_counter}/{self.MAX_CONSECUTIVE_MOVES}\n")

    def close(self): pass