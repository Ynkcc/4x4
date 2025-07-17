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
    # 每种棋子的总数，用于新的二元死亡棋子表示
    PIECE_MAX_COUNTS = {
        PieceType.A: 2, PieceType.B: 1, PieceType.C: 1,
        PieceType.D: 1, PieceType.E: 1, PieceType.F: 1, PieceType.G: 1
    }

    def __init__(self, render_mode=None):
        super().__init__()
        
        # --- Gymnasium所需的环境定义 ---
        self.render_mode = render_mode

        # --- 状态向量大小计算 (已更新) ---
        dead_piece_counts_size_per_player = sum(self.PIECE_MAX_COUNTS.values())
        self.dead_piece_counts_total_size = dead_piece_counts_size_per_player * 2

        self.state_size = (
            (self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS * 2) +  # 我方和对手的棋子位置图层
            self.TOTAL_POSITIONS +                              # 暗棋位置图层
            self.TOTAL_POSITIONS +                              # 对手威胁图层 (新)
            self.dead_piece_counts_total_size +                 # 死亡棋子计数 (新表示)
            2 +                                                 # 双方分数
            1 +                                                 # 连续未吃子步数
            self.ACTION_SPACE_SIZE +                            # 行动机会向量 (新)
            self.ACTION_SPACE_SIZE                              # 行动威胁向量 (新)
        )
                          
        # 观测空间 (状态向量)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32
        )
        
        # 动作空间 (离散动作)
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)

        # --- 游戏内部状态 ---
        self.board = None # 棋盘将在 reset 时初始化
        self.dead_pieces = {-1: [], 1: []} 
        self.current_player = 1 
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}
        self.reveal_counter = 0 
        
        # --- 用于优化 action_masks 的数据结构 ---
        self.unrevealed_pieces_pos = set() 
        self.player_pieces_pos = {-1: set(), 1: set()}

        # --- 状态向量索引计算 (已更新) ---
        self._state_vector = np.zeros(self.state_size, dtype=np.float32)

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
        """
        初始化棋盘，随机放置双方的棋子，并填充位置追踪集合。
        """
        self.board = np.empty((self.BOARD_ROWS, self.BOARD_COLS), dtype=object)
        
        piece_counts = {
            PieceType.A: 2, PieceType.B: 1, PieceType.C: 1,
            PieceType.D: 1, PieceType.E: 1, PieceType.F: 1, PieceType.G: 1
        }
        pieces = []
        for piece_type, count in piece_counts.items():
            for _ in range(count):
                pieces.append(Piece(piece_type, -1)) # 黑方棋子
                pieces.append(Piece(piece_type, 1))  # 红方棋子

        # 使用 Gymnasium 提供的随机数生成器以保证可复现性
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
        """
        重置游戏环境到初始状态，符合 Gymnasium API。
        """
        # 设置随机种子
        super().reset(seed=seed)
        
        self._initialize_board()
        
        observation = self.get_state()
        info = {'action_mask': self.action_masks()} # 为初始状态提供动作掩码
        
        return observation, info

    def get_state(self):
        """
        获取当前游戏状态，返回一个更新后的、包含新图层和向量的特征向量。
        """
        current_p = self.current_player
        opponent_p = -current_p

        self._state_vector.fill(0.0)

        # 填充棋子位置图层 (我方, 对手, 暗棋)
        for r, c in self.unrevealed_pieces_pos:
            pos_flat_idx = r * self.BOARD_COLS + c
            idx = self._hidden_pieces_plane_start_idx + pos_flat_idx
            self._state_vector[idx] = 1.0
        
        for r, c in self.player_pieces_pos[current_p]:
            piece = self.board[r, c]
            piece_type_idx = piece.piece_type.value
            pos_flat_idx = r * self.BOARD_COLS + c
            idx = self._my_pieces_plane_start_idx + piece_type_idx * self.TOTAL_POSITIONS + pos_flat_idx
            self._state_vector[idx] = 1.0

        for r, c in self.player_pieces_pos[opponent_p]:
            piece = self.board[r, c]
            piece_type_idx = piece.piece_type.value
            pos_flat_idx = r * self.BOARD_COLS + c
            idx = self._opponent_pieces_plane_start_idx + piece_type_idx * self.TOTAL_POSITIONS + pos_flat_idx
            self._state_vector[idx] = 1.0

        # --- 新增: 填充对手威胁图层 ---
        threat_plane = self._get_threat_plane(opponent_p)
        self._state_vector[self._threat_plane_start_idx : self._my_dead_count_start_idx] = threat_plane
            
        # --- 修改: 填充死亡棋子二元计数 ---
        my_dead_counts = {pt: 0 for pt in PieceType}
        for p in self.dead_pieces[current_p]:
            my_dead_counts[p.piece_type] += 1
        
        current_idx = self._my_dead_count_start_idx
        for piece_type in sorted(PieceType, key=lambda pt: pt.value): # 保证顺序
            max_count = self.PIECE_MAX_COUNTS[piece_type]
            dead_count = my_dead_counts[piece_type]
            for i in range(max_count):
                self._state_vector[current_idx + i] = 1.0 if i < dead_count else 0.0
            current_idx += max_count

        opp_dead_counts = {pt: 0 for pt in PieceType}
        for p in self.dead_pieces[opponent_p]:
            opp_dead_counts[p.piece_type] += 1

        current_idx = self._opponent_dead_count_start_idx
        for piece_type in sorted(PieceType, key=lambda pt: pt.value): # 保证顺序
            max_count = self.PIECE_MAX_COUNTS[piece_type]
            dead_count = opp_dead_counts[piece_type]
            for i in range(max_count):
                self._state_vector[current_idx + i] = 1.0 if i < dead_count else 0.0
            current_idx += max_count

        # 填充分数和步数计数器
        score_norm = self.WINNING_SCORE if self.WINNING_SCORE > 0 else 1.0
        self._state_vector[self._scores_start_idx] = self.scores[current_p] / score_norm
        self._state_vector[self._scores_start_idx + 1] = self.scores[opponent_p] / score_norm

        move_norm = self.MAX_CONSECUTIVE_MOVES if self.MAX_CONSECUTIVE_MOVES > 0 else 1.0
        self._state_vector[self._move_counter_idx] = self.move_counter / move_norm

        # --- 新增: 填充行动机会和威胁向量 ---
        opportunity_vector, threat_vector = self._get_action_vectors()
        self._state_vector[self._opportunity_vector_start_idx : self._threat_vector_start_idx] = opportunity_vector
        self._state_vector[self._threat_vector_start_idx:] = threat_vector
        
        return self._state_vector.copy() #这里需要一个深拷贝，不要删除此条注释


    def step(self, action_index):
        """
        执行一个动作，更新游戏状态，符合 Gymnasium API。
        返回: (observation, reward, terminated, truncated, info)
        """
        reward = -0.0005

        pos_idx = action_index // 5
        action_sub_idx = action_index % 5
        
        row = pos_idx // self.BOARD_COLS
        col = pos_idx % self.BOARD_COLS
        from_pos = (row, col)

        info = {'winner': None}

        # 翻棋动作
        if action_sub_idx == 4:
            self.reveal(from_pos)
            self.move_counter = 0
            reward += 0.0005
        else:
            d_row, d_col = 0, 0
            if action_sub_idx == 0: d_row = -1
            elif action_sub_idx == 1: d_row = 1
            elif action_sub_idx == 2: d_col = -1
            elif action_sub_idx == 3: d_col = 1
            
            attacker = self.board[row, col]
            
            # 炮的攻击逻辑
            if attacker.piece_type == PieceType.B:
                to_pos = self._get_cannon_target(from_pos, (d_row, d_col))
                if to_pos is None:
                    raise ValueError(f"炮从 {from_pos} 的移动/攻击路径无效")
                
                raw_reward = self.attack(from_pos, to_pos)
                reward += raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward
                self.move_counter = 0
            # 其他棋子的移动/攻击逻辑
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
        
        terminated = False
        truncated = False
        winner = None

        if self.scores[1] >= self.WINNING_SCORE:
            winner = 1
            terminated = True
        elif self.scores[-1] >= self.WINNING_SCORE:
            winner = -1
            terminated = True
        elif self.move_counter >= self.MAX_CONSECUTIVE_MOVES:
            winner = 0 
            truncated = True

        self.current_player = -self.current_player
        
        if not terminated and not truncated and np.sum(self.action_masks()) == 0:
            winner = -self.current_player
            terminated = True
            
        info['winner'] = winner
        info['action_mask'] = self.action_masks()
        
        if terminated or truncated:
            if self.render_mode == "human":
                self.render()

        return self.get_state(), reward, terminated, truncated, info

    def render(self):
        """
        在控制台渲染当前棋盘状态。
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                "e.g. gym.make(env_id, render_mode='human')"
            )
            return

        red_map = {PieceType.G: "帥", PieceType.F: "仕", PieceType.E: "相", PieceType.D: "俥", PieceType.C: "傌", PieceType.B: "炮", PieceType.A: "兵"}
        black_map = {PieceType.G: "將", PieceType.F: "士", PieceType.E: "象", PieceType.D: "車", PieceType.C: "馬", PieceType.B: "炮", PieceType.A: "卒"}
        
        print("-" * (self.BOARD_COLS * 5 + 1))
        for r in range(self.BOARD_ROWS):
            print("|", end="")
            for c in range(self.BOARD_COLS):
                piece = self.board[r, c]
                if piece is None:
                    print("    |", end="")
                else:
                    if not piece.revealed:
                        print(f" \033[90m暗\033[0m  |", end="")
                    elif piece.player == 1:
                        print(f" \033[91m{red_map[piece.piece_type]}\033[0m  |", end="")
                    else:
                        print(f" \033[94m{black_map[piece.piece_type]}\033[0m  |", end="")
            print(f"  Row {r}")
        print("-" * (self.BOARD_COLS * 5 + 1))
        
        current_player_str = "\033[91m红方\033[0m" if self.current_player == 1 else "\033[94m黑方\033[0m"
        print(f"当前玩家: {current_player_str}")
        print(f"分数: \033[91m红方 {self.scores[1]}\033[0m vs \033[94m黑方 {self.scores[-1]}\033[0m")
        print(f"连续未吃子/翻子步数: {self.move_counter}/{self.MAX_CONSECUTIVE_MOVES}")
        print("\n")


    def reveal(self, position):
        """翻开指定位置的棋子。"""
        row, col = position
        piece_to_reveal = self.board[row, col]
        if piece_to_reveal is None:
            raise ValueError(f"尝试翻开空位置 ({row},{col})。")
        piece_to_reveal.revealed = True
        
        self.reveal_counter += 1
        
        self.unrevealed_pieces_pos.remove(position)
        self.player_pieces_pos[piece_to_reveal.player].add(position)


    def move(self, from_pos, to_pos):
        """移动棋子。"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        moving_piece = self.board[from_row, from_col]
        self.board[to_row, to_col] = moving_piece
        self.board[from_row, from_col] = None

        self.player_pieces_pos[moving_piece.player].remove(from_pos)
        self.player_pieces_pos[moving_piece.player].add(to_pos)


    def attack(self, from_pos, to_pos):
        """攻击并返回原始奖励值。"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        attacker = self.board[from_row, from_col]
        defender = self.board[to_row, to_col]
        
        self.dead_pieces[defender.player].append(defender)
        points = self.PIECE_VALUES[defender.piece_type]

        if attacker.player != defender.player:
            self.scores[attacker.player] += points
            reward = float(points)
        else: # 炮误伤己方暗棋
            self.scores[-attacker.player] += points
            reward = -float(points)

        if defender.revealed:
            self.player_pieces_pos[defender.player].remove(to_pos)
        else:
            self.unrevealed_pieces_pos.remove(to_pos)
            reward = 0 #暗子死亡不显式计分，以降低模型学习炮的策略的难度。不要移除此条注释

        self.board[to_row, to_col] = attacker
        self.board[from_row, from_col] = None

        self.player_pieces_pos[attacker.player].remove(from_pos)
        self.player_pieces_pos[attacker.player].add(to_pos)
        
        return reward

    def can_attack(self, attacker, defender):
        """判断攻击方是否能吃掉防守方。"""
        if attacker.piece_type == PieceType.G and defender.piece_type == PieceType.A:
            return False
        if attacker.piece_type == PieceType.A and defender.piece_type == PieceType.G:
            return True
        if attacker.piece_type.value < defender.piece_type.value:
            return False
        return True
    
    def _get_cannon_target(self, from_pos, direction):
        """计算炮在指定方向上的攻击目标位置。"""
        dr, dc = direction
        r_check, c_check = from_pos
        mount_found = False
        while True:
            r_check += dr
            c_check += dc
            if not (0 <= r_check < self.BOARD_ROWS and 0 <= c_check < self.BOARD_COLS):
                return None
            
            target_on_path = self.board[r_check, c_check]
            if target_on_path:
                if not mount_found:
                    mount_found = True
                    continue
                else:
                    return (r_check, c_check)
        return None

    # --- 新增和修改的辅助函数 ---

    def _get_threat_plane(self, player):
        """计算指定玩家的威胁图层。"""
        threat_plane = np.zeros(self.TOTAL_POSITIONS, dtype=np.float32)
        for r_from, c_from in self.player_pieces_pos[player]:
            piece = self.board[r_from, c_from]
            if not piece.revealed: continue

            if piece.piece_type == PieceType.B:  # 炮
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    target_pos = self._get_cannon_target((r_from, c_from), (dr, dc))
                    if target_pos:
                        pos_flat_idx = target_pos[0] * self.BOARD_COLS + target_pos[1]
                        threat_plane[pos_flat_idx] = 1.0
            else:  # 其他棋子
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r_to, c_to = r_from + dr, c_from + dc
                    if (0 <= r_to < self.BOARD_ROWS and 0 <= c_to < self.BOARD_COLS):
                        pos_flat_idx = r_to * self.BOARD_COLS + c_to
                        threat_plane[pos_flat_idx] = 1.0
        return threat_plane

    def _is_position_threatened_after_move(self, piece_to_move, to_pos, from_pos, by_player):
        """检查在一次移动后，棋子在目标位置是否会受到威胁。"""
        for r_opp, c_opp in self.player_pieces_pos[by_player]:
            opp_piece = self.board[r_opp, c_opp]
            if not opp_piece.revealed:
                continue
            
            # 检查普通棋子攻击
            if opp_piece.piece_type != PieceType.B:
                if abs(to_pos[0] - r_opp) + abs(to_pos[1] - c_opp) == 1:
                    if self.can_attack(opp_piece, piece_to_move):
                        return True
            # 检查炮攻击
            else:
                if not (r_opp == to_pos[0] or c_opp == to_pos[1]):
                    continue

                mount_count = 0
                if r_opp == to_pos[0]:  # 同一行
                    start_col, end_col = min(c_opp, to_pos[1]), max(c_opp, to_pos[1])
                    for c in range(start_col + 1, end_col):
                        if (r_opp, c) == from_pos: continue
                        if self.board[r_opp, c] is not None:
                            mount_count += 1
                else:  # 同一列
                    start_row, end_row = min(r_opp, to_pos[0]), max(r_opp, to_pos[0])
                    for r in range(start_row + 1, end_row):
                        if (r, c_opp) == from_pos: continue
                        if self.board[r, c_opp] is not None:
                            mount_count += 1
                
                if mount_count == 1:
                    return True
        return False

    def _get_action_vectors(self):
        """计算行动机会和行动威胁向量。"""
        action_mask = self.action_masks()
        opportunity_vector = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
        threat_vector = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
        
        current_p = self.current_player
        opponent_p = -current_p

        valid_action_indices = np.where(action_mask == 1)[0]

        for action_index in valid_action_indices:
            pos_idx = action_index // 5
            action_sub_idx = action_index % 5
            r_from = pos_idx // self.BOARD_COLS
            c_from = pos_idx % self.BOARD_COLS
            from_pos = (r_from, c_from)

            if action_sub_idx == 4:  # 翻棋
                continue

            moving_piece = self.board[r_from, c_from]
            
            to_pos = None
            is_capture = False
            
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action_sub_idx]

            if moving_piece.piece_type == PieceType.B:  # 炮
                to_pos = self._get_cannon_target(from_pos, (dr, dc))
                if to_pos:
                    is_capture = True
            else:  # 其他棋子
                to_pos = (r_from + dr, c_from + dc)
                target_piece = self.board[to_pos[0], to_pos[1]]
                if target_piece is not None:
                    is_capture = True
            
            if is_capture:
                opportunity_vector[action_index] = 1.0

            if self._is_position_threatened_after_move(moving_piece, to_pos, from_pos, opponent_p):
                threat_vector[action_index] = 1.0
                
        return opportunity_vector, threat_vector


    def action_masks(self):
        """获取当前玩家所有合法的动作。"""
        valid_actions_arr = np.zeros(self.ACTION_SPACE_SIZE, dtype=int)
        current_p = self.current_player

        for r_from, c_from in self.unrevealed_pieces_pos:
            pos_idx = r_from * self.BOARD_COLS + c_from
            action_idx = pos_idx * 5 + 4
            valid_actions_arr[action_idx] = 1

        for r_from, c_from in self.player_pieces_pos[current_p]:
            piece = self.board[r_from, c_from]
            pos_idx = r_from * self.BOARD_COLS + c_from
            
            if piece.piece_type == PieceType.B:
                for move_sub_idx, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    target_pos = self._get_cannon_target((r_from, c_from), (dr, dc))
                    if target_pos:
                        target_piece = self.board[target_pos[0], target_pos[1]]
                        if not target_piece.revealed or target_piece.player != current_p:
                             action_idx = pos_idx * 5 + move_sub_idx
                             valid_actions_arr[action_idx] = 1
            else:
                for move_sub_idx, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    r_to, c_to = r_from + dr, c_from + dc
                    if not (0 <= r_to < self.BOARD_ROWS and 0 <= c_to < self.BOARD_COLS):
                        continue

                    action_idx = pos_idx * 5 + move_sub_idx
                    target_piece = self.board[r_to, c_to]

                    if target_piece is None:
                        valid_actions_arr[action_idx] = 1
                    elif target_piece.player != current_p and target_piece.revealed and self.can_attack(piece, target_piece):
                        valid_actions_arr[action_idx] = 1
        return valid_actions_arr

    def close(self):
        """
        清理环境资源（如果需要）。
        """
        pass