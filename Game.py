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
    # --- Gymnasium 环境元数据 ---
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    # --- 类常量定义 ---
    BOARD_ROWS = 4
    BOARD_COLS = 4
    NUM_PIECE_TYPES = 7  # A 到 G
    TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS
    ACTION_SPACE_SIZE = TOTAL_POSITIONS * 5 # 每个位置: 上,下,左,右,翻

    MAX_CONSECUTIVE_MOVES = 40 # 最大连续未吃子/翻子移动步数，超过则平局
    WINNING_SCORE = 60         # 胜利所需分数

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

        # --- 状态向量大小计算 ---
        dead_piece_counts_size_per_player = sum(self.PIECE_MAX_COUNTS.values())
        self.dead_piece_counts_total_size = dead_piece_counts_size_per_player * 2

        self.state_size = (
            (self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS * 2) +
            self.TOTAL_POSITIONS +
            self.TOTAL_POSITIONS +
            self.dead_piece_counts_total_size +
            2 + 1 +
            self.ACTION_SPACE_SIZE * 2
        )
                          
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)

        # --- 游戏内部状态 ---
        self.board = None
        self.dead_pieces = {-1: [], 1: []} 
        self.current_player = 1 
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}
        self.reveal_counter = 0 
        self.unrevealed_pieces_pos = set() 
        self.player_pieces_pos = {-1: set(), 1: set()}

        # --- 修改: 持久化存储双方视角的状态向量 ---
        self._state_vector_p1 = np.zeros(self.state_size, dtype=np.float32)
        self._state_vector_p_neg1 = np.zeros(self.state_size, dtype=np.float32)

        # --- 状态向量索引计算 ---
        dead_piece_counts_size_per_player = sum(self.PIECE_MAX_COUNTS.values())
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
        """初始化棋盘，随机放置双方的棋子，并填充位置追踪集合。"""
        self.board = np.empty((self.BOARD_ROWS, self.BOARD_COLS), dtype=object)
        
        piece_counts = {k: v for k, v in self.PIECE_MAX_COUNTS.items()}
        pieces = []
        for piece_type, count in piece_counts.items():
            for _ in range(count):
                pieces.append(Piece(piece_type, -1))
                pieces.append(Piece(piece_type, 1))

        self.np_random.shuffle(pieces)
        
        self.unrevealed_pieces_pos.clear()
        self.player_pieces_pos[-1].clear()
        self.player_pieces_pos[1].clear()

        idx = 0
        for row in range(self.BOARD_ROWS):
            for col in range(self.BOARD_COLS):
                self.board[row, col] = pieces[idx]
                self.unrevealed_pieces_pos.add((row, col))
                idx += 1
        
        # # --- 暂时改为完美信息 ---
        # # 强制所有棋子变为明棋
        # for r in range(self.BOARD_ROWS):
        #     for c in range(self.BOARD_COLS):
        #         piece = self.board[r, c]
        #         if piece:
        #             piece.revealed = True # 强制设置为已翻开

        # # 相应地更新位置追踪集合
        # self.player_pieces_pos[-1].clear()
        # self.player_pieces_pos[1].clear()
        # for r in range(self.BOARD_ROWS):
        #     for c in range(self.BOARD_COLS):
        #         piece = self.board[r, c]
        #         if piece:
        #             self.player_pieces_pos[piece.player].add((r, c))
        # self.unrevealed_pieces_pos.clear() # 已经没有未翻开的棋子了

        # #------------------------ 

        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1 
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}
        self.reveal_counter = 0

    def reset(self, seed=None, options=None):
        """重置游戏环境，并计算初始状态。"""
        super().reset(seed=seed)
        self._initialize_board()
        
        # 核心修改: 在reset时计算一次完整的状态
        action_mask = self._update_all_state_vectors()
        
        observation = self.get_state()
        info = {'action_mask': action_mask}
        
        return observation, info
    
    def get_state(self):
        """
        简化后的方法：直接返回当前玩家对应的预计算好的状态向量的拷贝。
        """
        if self.current_player == 1:
            return self._state_vector_p1.copy()
        else:
            return self._state_vector_p_neg1.copy()

    def _update_all_state_vectors(self):
        """
        核心计算函数：根据当前局面，更新双方视角的状态向量。
        此函数取代了旧的 get_state() 的计算功能。
        """
        # 为两个玩家（1 和 -1）分别生成状态
        for player_perspective in [1, -1]:
            state_vector = self._state_vector_p1 if player_perspective == 1 else self._state_vector_p_neg1
            state_vector.fill(0.0)

            # 定义当前视角下的“我”和“对手”
            current_p = player_perspective
            opponent_p = -player_perspective

            # 填充棋子位置图层 (我方, 对手, 暗棋)
            for r, c in self.unrevealed_pieces_pos:
                pos_flat_idx = r * self.BOARD_COLS + c
                state_vector[self._hidden_pieces_plane_start_idx + pos_flat_idx] = 1.0
            
            for r, c in self.player_pieces_pos[current_p]:
                piece = self.board[r, c]
                piece_type_idx = piece.piece_type.value
                pos_flat_idx = r * self.BOARD_COLS + c
                state_vector[self._my_pieces_plane_start_idx + piece_type_idx * self.TOTAL_POSITIONS + pos_flat_idx] = 1.0

            for r, c in self.player_pieces_pos[opponent_p]:
                piece = self.board[r, c]
                piece_type_idx = piece.piece_type.value
                pos_flat_idx = r * self.BOARD_COLS + c
                state_vector[self._opponent_pieces_plane_start_idx + piece_type_idx * self.TOTAL_POSITIONS + pos_flat_idx] = 1.0

            # 填充对手威胁图层
            threat_plane = self._get_threat_plane(opponent_p)
            state_vector[self._threat_plane_start_idx : self._my_dead_count_start_idx] = threat_plane
                
            # 填充死亡棋子二元计数
            my_dead_counts = {pt: 0 for pt in PieceType}
            for p in self.dead_pieces[current_p]: my_dead_counts[p.piece_type] += 1
            
            current_idx = self._my_dead_count_start_idx
            for piece_type in sorted(PieceType, key=lambda pt: pt.value):
                max_count = self.PIECE_MAX_COUNTS[piece_type]
                dead_count = my_dead_counts.get(piece_type, 0)
                for i in range(max_count):
                    state_vector[current_idx + i] = 1.0 if i < dead_count else 0.0
                current_idx += max_count

            opp_dead_counts = {pt: 0 for pt in PieceType}
            for p in self.dead_pieces[opponent_p]: opp_dead_counts[p.piece_type] += 1

            current_idx = self._opponent_dead_count_start_idx
            for piece_type in sorted(PieceType, key=lambda pt: pt.value):
                max_count = self.PIECE_MAX_COUNTS[piece_type]
                dead_count = opp_dead_counts.get(piece_type, 0)
                for i in range(max_count):
                    state_vector[current_idx + i] = 1.0 if i < dead_count else 0.0
                current_idx += max_count

            # 填充分数和步数计数器
            score_norm = self.WINNING_SCORE if self.WINNING_SCORE > 0 else 1.0
            state_vector[self._scores_start_idx] = self.scores[current_p] / score_norm
            state_vector[self._scores_start_idx + 1] = self.scores[opponent_p] / score_norm

            move_norm = self.MAX_CONSECUTIVE_MOVES if self.MAX_CONSECUTIVE_MOVES > 0 else 1.0
            state_vector[self._move_counter_idx] = self.move_counter / move_norm

            # 填充行动机会和威胁向量
            opportunity_vector, threat_vector, _ = self._get_action_vectors(player_perspective)
            state_vector[self._opportunity_vector_start_idx : self._threat_vector_start_idx] = opportunity_vector
            state_vector[self._threat_vector_start_idx:] = threat_vector
        
        # 返回当前玩家的动作掩码，供外部使用
        return self.action_masks()


    def step(self, action_index):
        """执行动作，并在动作后更新所有状态向量。"""
        reward = -0.0005
        pos_idx = action_index // 5
        action_sub_idx = action_index % 5
        row, col = pos_idx // self.BOARD_COLS, pos_idx % self.BOARD_COLS
        from_pos = (row, col)

        # 执行动作并更新棋盘基础状态
        if action_sub_idx == 4:
            self.reveal(from_pos)
            self.move_counter = 0
            reward += 0.0005
        else:
            d_row, d_col = [(-1,0), (1,0), (0,-1), (0,1)][action_sub_idx]
            attacker = self.board[row, col]
            
            if attacker.piece_type == PieceType.B:
                to_pos = self._get_cannon_target(from_pos, (d_row, d_col))
                if to_pos is None: raise ValueError(f"炮从 {from_pos} 的移动/攻击路径无效")
                raw_reward = self.attack(from_pos, to_pos)
                reward += raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward
                self.move_counter = 0
            else:
                to_pos = (row + d_row, col + d_col)
                target = self.board[to_pos[0], to_pos[1]]
                if target is None:
                    self.move(from_pos, to_pos)
                    self.move_counter += 1
                else:
                    raw_reward = self.attack(from_pos, to_pos)
                    reward += raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward
                    self.move_counter = 0
        
        # 检查游戏结束条件
        terminated, truncated, winner = False, False, None
        if self.scores[1] >= self.WINNING_SCORE:
            winner, terminated = 1, True
        elif self.scores[-1] >= self.WINNING_SCORE:
            winner, terminated = -1, True
        elif self.move_counter >= self.MAX_CONSECUTIVE_MOVES:
            winner, truncated = 0, True

        # 切换玩家
        self.current_player = -self.current_player
        
        # 核心修改: 动作执行完毕，更新双方的完整状态向量
        action_mask = self._update_all_state_vectors()
        
        # 基于新的状态，检查新玩家是否有棋可走
        if not terminated and not truncated and np.sum(action_mask) == 0:
            winner = -self.current_player
            terminated = True
            
        # 准备返回信息
        observation = self.get_state()
        info = {'winner': winner, 'action_mask': action_mask}
        
        if (terminated or truncated) and self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """在控制台渲染当前棋盘状态。"""
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        red_map = {PieceType.G: "帥", PieceType.F: "仕", PieceType.E: "相", PieceType.D: "俥", PieceType.C: "傌", PieceType.B: "炮", PieceType.A: "兵"}
        black_map = {PieceType.G: "將", PieceType.F: "士", PieceType.E: "象", PieceType.D: "車", PieceType.C: "馬", PieceType.B: "炮", PieceType.A: "卒"}
        
        print("-" * (self.BOARD_COLS * 5 + 1))
        for r in range(self.BOARD_ROWS):
            print("|", end="")
            for c in range(self.BOARD_COLS):
                piece = self.board[r, c]
                if piece is None: print("    |", end="")
                elif not piece.revealed: print(f" \033[90m暗\033[0m  |", end="")
                elif piece.player == 1: print(f" \033[91m{red_map[piece.piece_type]}\033[0m  |", end="")
                else: print(f" \033[94m{black_map[piece.piece_type]}\033[0m  |", end="")
            print(f"  Row {r}")
        print("-" * (self.BOARD_COLS * 5 + 1))
        
        current_player_str = "\033[91m红方\033[0m" if self.current_player == 1 else "\033[94m黑方\033[0m"
        print(f"当前玩家: {current_player_str}")
        print(f"分数: \033[91m红方 {self.scores[1]}\033[0m vs \033[94m黑方 {self.scores[-1]}\033[0m")
        print(f"连续未吃子/翻子步数: {self.move_counter}/{self.MAX_CONSECUTIVE_MOVES}")
        print("\n")


    def reveal(self, position):
        """翻开指定位置的棋子。"""
        piece_to_reveal = self.board[position]
        if piece_to_reveal is None: raise ValueError(f"尝试翻开空位置 {position}。")
        piece_to_reveal.revealed = True
        self.reveal_counter += 1
        self.unrevealed_pieces_pos.remove(position)
        self.player_pieces_pos[piece_to_reveal.player].add(position)

    def move(self, from_pos, to_pos):
        """移动棋子。"""
        moving_piece = self.board[from_pos]
        self.board[to_pos] = moving_piece
        self.board[from_pos] = None
        self.player_pieces_pos[moving_piece.player].remove(from_pos)
        self.player_pieces_pos[moving_piece.player].add(to_pos)

    def attack(self, from_pos, to_pos):
        """攻击并返回原始奖励值。"""
        attacker = self.board[from_pos]
        defender = self.board[to_pos]
        
        self.dead_pieces[defender.player].append(defender)
        points = self.PIECE_VALUES[defender.piece_type]

        if attacker.player != defender.player:
            self.scores[attacker.player] += points
            reward = float(points)
        else:
            self.scores[-attacker.player] += points
            reward = -float(points)

        if defender.revealed:
            self.player_pieces_pos[defender.player].remove(to_pos)
        else:
            self.unrevealed_pieces_pos.remove(to_pos)
            reward = 0

        self.board[to_pos] = attacker
        self.board[from_pos] = None
        self.player_pieces_pos[attacker.player].remove(from_pos)
        self.player_pieces_pos[attacker.player].add(to_pos)
        return reward

    def can_attack(self, attacker, defender):
        """判断攻击方是否能吃掉防守方。"""
        if attacker.piece_type == PieceType.G and defender.piece_type == PieceType.A: return False
        if attacker.piece_type == PieceType.A and defender.piece_type == PieceType.G: return True
        if attacker.piece_type.value < defender.piece_type.value: return False
        return True
    
    def _get_cannon_target(self, from_pos, direction):
        """计算炮在指定方向上的攻击目标位置。"""
        dr, dc = direction
        r_check, c_check = from_pos
        mount_found = False
        while True:
            r_check, c_check = r_check + dr, c_check + dc
            if not (0 <= r_check < self.BOARD_ROWS and 0 <= c_check < self.BOARD_COLS): return None
            if self.board[r_check, c_check]:
                if not mount_found:
                    mount_found = True
                    continue
                else: return (r_check, c_check)
        return None

    def _get_threat_plane(self, player):
        """计算指定玩家的威胁图层。"""
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
        """检查在一次移动后，棋子在目标位置是否会受到威胁。"""
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
        """计算指定玩家视角的行动机会和威胁向量。"""
        action_mask = self.action_masks(for_player)
        opportunity_vector = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
        threat_vector = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
        
        opponent_p = -for_player
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
            if self._is_position_threatened_after_move(moving_piece, to_pos, from_pos, opponent_p):
                threat_vector[action_index] = 1.0
                
        return opportunity_vector, threat_vector, action_mask

    def action_masks(self, for_player=None):
        """获取指定玩家所有合法的动作。如果未指定，则为当前玩家。"""
        player = for_player if for_player is not None else self.current_player
        valid_actions_arr = np.zeros(self.ACTION_SPACE_SIZE, dtype=int)

        if player == self.current_player: # 只有当前玩家可以翻棋
            for r_from, c_from in self.unrevealed_pieces_pos:
                pos_idx = r_from * self.BOARD_COLS + c_from
                valid_actions_arr[pos_idx * 5 + 4] = 1

        for r_from, c_from in self.player_pieces_pos[player]:
            piece = self.board[r_from, c_from]
            pos_idx = r_from * self.BOARD_COLS + c_from
            
            if piece.piece_type == PieceType.B:
                for move_sub_idx, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    target_pos = self._get_cannon_target((r_from, c_from), (dr, dc))
                    if target_pos:
                        target_piece = self.board[target_pos]
                        if not target_piece.revealed or target_piece.player != player:
                             valid_actions_arr[pos_idx * 5 + move_sub_idx] = 1
            else:
                for move_sub_idx, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    r_to, c_to = r_from + dr, c_from + dc
                    if not (0 <= r_to < self.BOARD_ROWS and 0 <= c_to < self.BOARD_COLS): continue
                    
                    target_piece = self.board[r_to, c_to]
                    action_idx = pos_idx * 5 + move_sub_idx
                    if target_piece is None:
                        valid_actions_arr[action_idx] = 1
                    elif target_piece.player != player and target_piece.revealed and self.can_attack(piece, target_piece):
                        valid_actions_arr[action_idx] = 1
        return valid_actions_arr

    def close(self):
        """清理环境资源。"""
        pass