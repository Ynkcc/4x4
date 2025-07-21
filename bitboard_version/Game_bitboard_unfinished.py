# Game_bitboard_unfinished

import random
from enum import Enum
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces

# ==============================================================================
# --- Bitboard 核心工具 ---
# Bitboard是一种用单个整数的比特位来表示棋盘状态的高性能技术。
# 每一个比特位对应棋盘上的一个格子。例如，一个64位的整数可以表示一个8x8棋盘。
# 在我们的4x4棋盘中，一个16位或更高位的整数就足够了。
# 我们将所有相关的常量、转换函数和预计算表的生成逻辑放在这里。
# ==============================================================================

# 棋盘位置索引 (0-15) 与 (行, 列) 的转换字典
# 棋盘布局 (从0到15的索引):
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
POS_TO_SQ = np.array([[(r * 4 + c) for c in range(4)] for r in range(4)], dtype=np.int32)
SQ_TO_POS = {sq: (sq // 4, sq % 4) for sq in range(16)}

def ULL(x):
    """一个帮助函数，用于创建一个Bitboard（无符号长整型），仅将第x位置为1。"""
    # 例如 ULL(3) -> ...00001000 (二进制)，表示第3个格子上有棋子。
    return 1 << x

# 定义棋盘边界的Bitboard掩码（Mask），用于在生成走法时防止棋子“穿越”棋盘边界。
# 掩码是一个特殊的Bitboard，其中特定位置为1，用于通过位运算快速筛选或修改棋盘状态。
FILE_A = sum(ULL(i) for i in [0, 4, 8, 12])  # 第1列 (A列)
FILE_D = sum(ULL(i) for i in [3, 7, 11, 15]) # 第4列 (D列)
NOT_FILE_A = ~FILE_A # 按位取反，得到所有不是第1列的格子
NOT_FILE_D = ~FILE_D # 按位取反，得到所有不是第4列的格子

# 定义走法类型的枚举，比使用魔法数字（如0、1）更清晰、更易于维护。
ACTION_TYPE_MOVE = 0
ACTION_TYPE_REVEAL = 1

# 使用命名元组来表示一个“走法”，使代码更具可读性。
# move = Move(from_sq=1, to_sq=5, action_type=ACTION_TYPE_MOVE)
Move = collections.namedtuple('Move', ['from_sq', 'to_sq', 'action_type'])

# --- 枚举和Piece类定义 ---

class PieceType(Enum):
    """定义棋子类型及其大小等级。值越大，等级越高。"""
    A = 0; B = 1; C = 2; D = 3; E = 4; F = 5; G = 6 # 兵/卒, 炮, 马, 车, 象, 士, 将

class Piece:
    """棋子对象，仅存储棋子本身的属性（类型，玩家，是否翻开）。
    这个对象主要用于方便渲染和处理复杂的吃子逻辑，而棋盘位置状态由Bitboard管理。
    """
    def __init__(self, piece_type, player):
        self.piece_type, self.player, self.revealed = piece_type, player, False
    def __repr__(self):
        # 这个方法用于在调试时打印对象信息，使其更具可读性。
        return f"{'R' if self.revealed else 'H'}_{'R' if self.player == 1 else 'B'}{self.piece_type.name}"


class GameEnvironment(gym.Env):
    """
    基于Bitboard的暗棋Gym环境。
    实现了OpenAI Gym (现为Gymnasium) 的标准接口 (reset, step, render, close)。
    使用Bitboard进行状态表示和走法生成，以获得高性能。
    使用增量更新来修改状态，避免在每一步都重新计算整个状态，从而提高效率。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    # --- 游戏核心常量 ---
    BOARD_ROWS, BOARD_COLS, NUM_PIECE_TYPES = 4, 4, 7
    TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS  # 16
    # 动作空间大小: 每个格子(16)可以移动到4个方向(上、下、左、右)或翻开(1)，总共 16 * 5 = 80 个可能的动作。
    ACTION_SPACE_SIZE = TOTAL_POSITIONS * 5

    MAX_CONSECUTIVE_MOVES = 40  # 连续未吃子或未翻棋达到此步数，判为和棋。
    WINNING_SCORE = 60 # 任何一方达到或超过此分数即获胜。

    # 定义每种棋子的价值，用于计算得分。
    PIECE_VALUES = {pt: val for pt, val in zip(PieceType, [4, 10, 10, 10, 10, 20, 30])}
    # 定义每种棋子的最大数量（用于状态向量中的死亡棋子计数）。
    PIECE_MAX_COUNTS = {pt: val for pt, val in zip(PieceType, [2, 1, 1, 1, 1, 1, 1])}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # --- Gym环境所需的状态和动作空间定义 ---
        # 计算死亡棋子计数部分在状态向量中的大小
        dead_piece_counts_size_per_player = sum(self.PIECE_MAX_COUNTS.values())
        
        # 定义状态向量的总大小。这是一个扁平化的（一维）向量，包含了AI决策所需的所有信息。
        self.state_size = (
            # 1. 我方棋子位置平面 (7种棋子 * 16个位置)
            (self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS) +
            # 2. 对方棋子位置平面 (7种棋子 * 16个位置)
            (self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS) +
            # 3. 未翻开棋子位置平面 (1 * 16个位置)
            self.TOTAL_POSITIONS +
            # 4. 威胁平面 (我方棋子可以攻击的位置)
            self.TOTAL_POSITIONS +
            # 5. 我方死亡棋子计数 (每种棋子死亡了多少个)
            dead_piece_counts_size_per_player +
            # 6. 对方死亡棋子计数
            dead_piece_counts_size_per_player +
            # 7. 分数 (我方分数, 对方分数)
            2 +
            # 8. 连续未吃子/翻棋的步数计数器
            1 +
            # 9. 机会向量 (我方每个合法动作能带来的潜在收益)
            self.ACTION_SPACE_SIZE +
            # 10. 威胁向量 (我方每个合法动作是否会使执行该动作的棋子被吃)
            self.ACTION_SPACE_SIZE
        )
        # 观测空间：一个连续的一维向量，值被归一化到 [0.0, 1.0] 之间。
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32)
        # 动作空间：一个离散的整数，范围从 0 到 ACTION_SPACE_SIZE - 1。
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)

        # --- 核心数据结构: Bitboards ---
        # self.board 是一个Numpy数组，存储实际的Piece对象，主要用于渲染和复杂的吃子逻辑判断。
        self.board = np.empty(self.TOTAL_POSITIONS, dtype=object)
        # piece_bitboards: 按玩家和棋子类型存储位置信息。例如 self.piece_bitboards[1][PieceType.A.value] 是一个整数，
        # 其比特位表示所有红方兵的位置。
        self.piece_bitboards = {p: [0] * self.NUM_PIECE_TYPES for p in [1, -1]}
        # occupied_bitboards: 按玩家存储所有已翻开棋子的位置。
        # occupied_bitboards[0] 存储所有棋子（无论翻开与否）的位置。
        self.occupied_bitboards = {1: 0, -1: 0, 0: 0}
        # hidden_pieces_bitboard: 存储所有未翻开棋子的位置。
        self.hidden_pieces_bitboard = 0

        # --- 游戏状态变量 ---
        self.dead_pieces = {-1: [], 1: []} # 存储被吃掉的Piece对象
        self.current_player = 1 # 1: 红方, -1: 黑方
        self.move_counter = 0 # 连续未吃子/翻棋的步数
        self.scores = {-1: 0, 1: 0} # 双方得分

        # --- 持久化状态向量 ---
        # 为了效率，我们为红方和黑方各维护一个状态向量，并在每一步进行增量更新。
        # _state_vector_p1 是以红方(player=1)视角的状态。
        # _state_vector_p_neg1 是以黑方(player=-1)视角的状态。
        self._state_vector_p1 = np.zeros(self.state_size, dtype=np.float32)
        self._state_vector_p_neg1 = np.zeros(self.state_size, dtype=np.float32)
        
        # --- 状态向量索引定义 ---
        # 为了方便地更新状态向量的特定部分，我们预先定义好每个部分的起始索引。
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
        
        # --- 预计算攻击表 ---
        # 预先计算和存储所有可能的走法、攻击模式，以在游戏过程中通过查表快速获取信息，避免重复计算。
        self.attack_tables = {}
        self._initialize_bitboard_tables()


    def _initialize_bitboard_tables(self):
        """一次性预计算所有与位置相关的查找表，典型的“空间换时间”策略。"""
        # 1. 普通棋子（可以看作国际象棋中的“王”）的攻击范围
        # 对于每个格子，计算其上下左右相邻的格子，并存为一个Bitboard。
        king_attacks = [0] * self.TOTAL_POSITIONS
        for sq in range(self.TOTAL_POSITIONS):
            target = ULL(sq)
            attacks = 0
            # East (右移一位)
            if (target & NOT_FILE_D) > 0: attacks |= (target << 1)
            # West (左移一位)
            if (target & NOT_FILE_A) > 0: attacks |= (target >> 1)
            # North (上移四位)
            if sq > 3: attacks |= (target >> 4)
            # South (下移四位)
            if sq < 12: attacks |= (target << 4)
            king_attacks[sq] = attacks
        self.attack_tables['king'] = king_attacks

        # 2. 炮的射线范围 (不考虑棋子阻挡)
        # 对于每个格子，计算其四个方向（上、下、左、右）上的所有格子，分别存为一个Bitboard。
        ray_attacks = [[0] * self.TOTAL_POSITIONS for _ in range(4)] # 0:N, 1:S, 2:W, 3:E
        for sq in range(self.TOTAL_POSITIONS):
            r, c = SQ_TO_POS[sq]
            for i in range(r - 1, -1, -1): ray_attacks[0][sq] |= ULL(POS_TO_SQ[i, c]) # North
            for i in range(r + 1, 4):      ray_attacks[1][sq] |= ULL(POS_TO_SQ[i, c]) # South
            for i in range(c - 1, -1, -1): ray_attacks[2][sq] |= ULL(POS_TO_SQ[r, i]) # West
            for i in range(c + 1, 4):      ray_attacks[3][sq] |= ULL(POS_TO_SQ[r, i]) # East
        self.attack_tables['rays'] = ray_attacks
        
        # 3. 动作索引转换表
        # 将 (from_sq, to_sq) 的移动映射到一个唯一的动作索引 action_index。
        # action_index = from_sq * 5 + sub_idx (0-3 for move, 4 for reveal)
        move_to_action = {}
        for from_sq in range(self.TOTAL_POSITIONS):
            move_to_action[from_sq] = {}
            r, c = SQ_TO_POS[from_sq]
            # 上(North)移
            if r > 0: move_to_action[from_sq][from_sq - 4] = from_sq * 5 + 0
            # 下(South)移
            if r < 3: move_to_action[from_sq][from_sq + 4] = from_sq * 5 + 1
            # 左(West)移
            if c > 0: move_to_action[from_sq][from_sq - 1] = from_sq * 5 + 2
            # 右(East)移
            if c < 3: move_to_action[from_sq][from_sq + 1] = from_sq * 5 + 3
        self.attack_tables['move_to_action'] = move_to_action


    def _initialize_board(self):
        """初始化棋盘，包括随机放置棋子对象，并根据其建立所有Bitboards。"""
        # 1. 创建一个包含所有棋子对象的列表，并随机打乱顺序。
        pieces = []
        for piece_type, count in self.PIECE_MAX_COUNTS.items():
            for _ in range(count):
                # 每种棋子都创建红黑双方各一个。
                pieces.extend([Piece(piece_type, -1), Piece(piece_type, 1)])
        self.np_random.shuffle(pieces)
        for sq in range(self.TOTAL_POSITIONS):
            self.board[sq] = pieces[sq]

        # 2. 根据 `self.board` 数组，从零开始建立所有Bitboards。
        # 重置所有 bitboards
        for p in [1, -1]:
            self.piece_bitboards[p] = [0] * self.NUM_PIECE_TYPES
            self.occupied_bitboards[p] = 0
        
        # 初始时，所有棋子都是未翻开的。
        self.hidden_pieces_bitboard = ULL(16) - 1 # ULL(16)-1 会生成一个所有16个低位都为1的整数
        # occupied_bitboards[0] 表示棋盘上所有被占据的位置（无论棋子是否翻开）。
        self.occupied_bitboards[0] = self.hidden_pieces_bitboard
        
        # 3. 重置游戏状态变量。
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

    def reset(self, seed=None, options=None):
        """重置游戏环境到初始状态，符合Gym接口。"""
        super().reset(seed=seed)
        self._initialize_board()
        # 从头构建状态向量和动作掩码
        action_mask = self._build_state_from_scratch()
        # 返回初始观测和信息字典
        return self.get_state(), {'action_mask': action_mask}

    def get_state(self):
        """获取当前玩家视角的状态向量。"""
        # 直接返回预先计算好的、对应当前玩家的状态向量的拷贝。
        # 返回拷贝是为了防止外部代码意外修改内部状态。
        return self._state_vector_p1.copy() if self.current_player == 1 else self._state_vector_p_neg1.copy()

    def step(self, action_index):
        """
        执行一个动作，更新游戏状态，并返回(observation, reward, terminated, truncated, info)。
        这是环境的核心交互函数。
        """
        # 1. 执行动作，更新Bitboards、棋盘数组和简单状态变量（如得分、步数），并获取原始奖励。
        raw_reward, move = self._execute_action_and_update_state(action_index)
        
        # 2. 计算最终奖励。这里引入了时间惩罚，鼓励AI尽快获胜。
        reward = -0.0005 # 每走一步都有一个微小的负奖励（时间成本）
        if move.action_type == ACTION_TYPE_MOVE:
             # 如果是移动/吃子，奖励是吃子得分（归一化后）减去时间成本
             reward += raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward
        else: # 翻棋动作的奖励刚好抵消时间惩罚，鼓励探索。
             reward += 0.0005

        # 3. 检查游戏是否结束 (terminated) 或截断 (truncated)。
        terminated, truncated, winner = False, False, None
        if self.scores[1] >= self.WINNING_SCORE: winner, terminated = 1, True       # 红方胜利
        elif self.scores[-1] >= self.WINNING_SCORE: winner, terminated = -1, True  # 黑方胜利
        elif self.move_counter >= self.MAX_CONSECUTIVE_MOVES: winner, truncated = 0, True # 和棋

        # 4. 切换玩家，并更新状态向量中的复杂部分（如威胁、机会等）。
        self.current_player = -self.current_player
        action_mask = self._update_complex_state_vectors()
        
        # 5. 检查切换后的玩家是否有棋可走。如果无棋可走，则判负。
        if not terminated and not truncated and np.sum(action_mask) == 0:
            winner = -self.current_player # 对手获胜
            terminated = True
            
        # 准备返回值
        observation = self.get_state()
        info = {'winner': winner, 'action_mask': action_mask}
        
        # 如果游戏结束并且处于人类渲染模式，则渲染最终棋盘状态。
        if (terminated or truncated) and self.render_mode == "human": self.render()
        
        return observation, reward, terminated, truncated, info

    def _execute_action_and_update_state(self, action_index):
        """
        核心执行函数：解码动作，调用对应的增量更新函数，修改Bitboards和状态向量的简单部分。
        """
        # --- 1. 解码 Action ---
        # 从一维的action_index解码出棋子的起始位置(from_sq)和具体操作(sub_idx)。
        from_sq, sub_idx = divmod(action_index, 5)
        
        # sub_idx == 4 表示这是一个翻棋动作。
        if sub_idx == 4:
            move = Move(from_sq=from_sq, to_sq=from_sq, action_type=ACTION_TYPE_REVEAL)
            raw_reward = self._apply_reveal_update(move)
            self.move_counter = 0 # 翻棋会重置连续步数计数器
        else:
            # sub_idx 0-3 表示移动。从预计算表中查找目标位置(to_sq)。
            to_sq = self.attack_tables['move_to_action'][from_sq].get(from_sq + [-4, 4, -1, 1][sub_idx])
            
            # 这是一个健壮性检查，理论上如果action_mask正确，to_sq不会是None。
            if to_sq is None or to_sq < 0 or to_sq >= 16:
                # 如果是无效动作（例如，试图移出棋盘），则不执行任何操作。
                # 返回一个虚拟的翻棋Move对象，因为它不改变棋盘状态。
                return 0, Move(from_sq=from_sq, to_sq=from_sq, action_type=ACTION_TYPE_REVEAL)
                
            move = Move(from_sq=from_sq, to_sq=to_sq, action_type=ACTION_TYPE_MOVE)
            
            # 判断目标位置是空位还是有棋子
            if self.board[to_sq] is None:
                # 移动到空位
                raw_reward = self._apply_move_update(move)
                self.move_counter += 1
            else:
                # 攻击目标位置的棋子
                raw_reward = self._apply_attack_update(move)
                self.move_counter = 0 # 吃子会重置连续步数计数器

        # --- 2. 更新状态向量中的共享部分 (分数和计数器) ---
        # 对分数和步数进行归一化，使其范围在[0, 1]之间，便于神经网络处理。
        score_norm = self.WINNING_SCORE if self.WINNING_SCORE > 0 else 1.0
        move_norm = self.MAX_CONSECUTIVE_MOVES if self.MAX_CONSECUTIVE_MOVES > 0 else 1.0
        # 同时更新双方视角的状态向量
        for vec, p in [(self._state_vector_p1, 1), (self._state_vector_p_neg1, -1)]:
            vec[self._scores_start_idx] = self.scores[p] / score_norm
            vec[self._scores_start_idx + 1] = self.scores[-p] / score_norm
            vec[self._move_counter_idx] = self.move_counter / move_norm
        
        return raw_reward, move

    def _apply_reveal_update(self, move: Move):
        """增量更新：处理翻棋动作。只修改Bitboards和状态向量的相关位。"""
        piece = self.board[move.from_sq]
        piece.revealed = True
        
        # --- Bitboard更新 ---
        # 使用XOR(^)操作来翻转比特位。一个数与另一个数异或两次，会变回原来的数。
        # 从“未翻开”集合中移除该位置
        self.hidden_pieces_bitboard ^= ULL(move.from_sq)
        # 在对应玩家的对应棋子类型Bitboard中添加该位置
        self.piece_bitboards[piece.player][piece.piece_type.value] |= ULL(move.from_sq)
        # 在对应玩家的“已翻开”集合中添加该位置
        self.occupied_bitboards[piece.player] |= ULL(move.from_sq)
        
        # --- 状态向量更新 ---
        # 同时更新双方视角的状态向量
        for vec, p in [(self._state_vector_p1, 1), (self._state_vector_p_neg1, -1)]:
            # 在“未翻开平面”中将该位置设为0
            vec[self._hidden_pieces_plane_start_idx + move.from_sq] = 0
            # 判断翻开的棋子是“我方”还是“对方”，并更新相应的平面
            plane_start = self._my_pieces_plane_start_idx if piece.player == p else self._opponent_pieces_plane_start_idx
            # 在对应棋子类型的平面中将该位置设为1
            vec[plane_start + piece.piece_type.value * self.TOTAL_POSITIONS + move.from_sq] = 1
        return 0 # 翻棋本身没有原始奖励

    def _apply_move_update(self, move: Move):
        """增量更新：处理移动到空位的动作。"""
        attacker = self.board[move.from_sq]
        # 创建一个掩码，其中from_sq和to_sq对应的位为1，用于一次性更新两个位置。
        move_mask = ULL(move.from_sq) | ULL(move.to_sq)

        # --- Bitboard更新 ---
        # 使用XOR(^)可以同时清除from_sq的位并设置to_sq的位。
        self.piece_bitboards[attacker.player][attacker.piece_type.value] ^= move_mask
        self.occupied_bitboards[attacker.player] ^= move_mask
        self.occupied_bitboards[0] ^= move_mask # 更新总棋盘状态
        
        # 更新棋盘对象数组
        self.board[move.to_sq], self.board[move.from_sq] = attacker, None
        
        # --- 状态向量更新 ---
        for vec, p in [(self._state_vector_p1, 1), (self._state_vector_p_neg1, -1)]:
            plane_start = self._my_pieces_plane_start_idx if attacker.player == p else self._opponent_pieces_plane_start_idx
            offset = attacker.piece_type.value * self.TOTAL_POSITIONS
            # 将旧位置清零，新位置设为1
            vec[plane_start + offset + move.from_sq] = 0
            vec[plane_start + offset + move.to_sq] = 1
        return 0 # 移动到空位没有原始奖励

    def _apply_attack_update(self, move: Move):
        """增量更新：处理攻击动作。"""
        attacker, defender = self.board[move.from_sq], self.board[move.to_sq]
        # 记录被吃棋子的原始死亡计数，用于更新状态向量
        original_dead_count = len(self.dead_pieces[defender.player])
        self.dead_pieces[defender.player].append(defender)
        
        # --- 计算奖励 ---
        points = self.PIECE_VALUES[defender.piece_type]
        raw_reward = 0
        if attacker.player != defender.player: # 攻击对方棋子
            self.scores[attacker.player] += points
            # 只有攻击已翻开的棋子才算作有效得分（用于模型训练的奖励信号）
            raw_reward = float(points) if defender.revealed else 0
        else: # 炮误伤己方棋子
            self.scores[-attacker.player] += points # 对手得分
            raw_reward = -float(points)

        # --- Bitboard更新 ---
        # 更新攻击方位置 (from -> to)
        att_move_mask = ULL(move.from_sq) | ULL(move.to_sq)
        self.piece_bitboards[attacker.player][attacker.piece_type.value] ^= att_move_mask
        self.occupied_bitboards[attacker.player] ^= att_move_mask
        
        # 移除被吃方棋子
        def_remove_mask = ULL(move.to_sq)
        if defender.revealed:
            # 从对应类型的Bitboard中移除
            self.piece_bitboards[defender.player][defender.piece_type.value] ^= def_remove_mask
        else:
            # 从“未翻开”Bitboard中移除
            self.hidden_pieces_bitboard ^= def_remove_mask
        # 更新总棋盘Bitboard: 攻击方从from_sq移走，被吃方在to_sq被移除，攻击方占据to_sq。
        # 效果等同于只从总棋盘中移除from_sq的位置，因为to_sq的位置仍然被占据。
        self.occupied_bitboards[0] ^= ULL(move.from_sq)
        
        # 更新棋盘对象数组
        self.board[move.to_sq], self.board[move.from_sq] = attacker, None

        # --- 状态向量更新 ---
        # 计算死亡棋子在状态向量中的偏移量
        dead_piece_offset = sum(self.PIECE_MAX_COUNTS[pt] for pt in PieceType if pt.value < defender.piece_type.value)
        for vec, p in [(self._state_vector_p1, 1), (self._state_vector_p_neg1, -1)]:
            # 1. 更新攻击方位置
            att_plane_start = self._my_pieces_plane_start_idx if attacker.player == p else self._opponent_pieces_plane_start_idx
            vec[att_plane_start + attacker.piece_type.value * self.TOTAL_POSITIONS + move.from_sq] = 0
            vec[att_plane_start + attacker.piece_type.value * self.TOTAL_POSITIONS + move.to_sq] = 1
            
            # 2. 移除被吃方位置
            if defender.revealed:
                def_plane_start = self._my_pieces_plane_start_idx if defender.player == p else self._opponent_pieces_plane_start_idx
                vec[def_plane_start + defender.piece_type.value * self.TOTAL_POSITIONS + move.to_sq] = 0
            else: # 如果被吃的是未翻开的棋子
                vec[self._hidden_pieces_plane_start_idx + move.to_sq] = 0
                
            # 3. 更新死亡计数
            dead_plane_start = self._my_dead_count_start_idx if defender.player == p else self._opponent_dead_count_start_idx
            vec[dead_plane_start + dead_piece_offset + original_dead_count] = 1
        
        return raw_reward

    def _update_complex_state_vectors(self):
        """
        在玩家切换后，重新计算并更新状态向量中具有非局部效应的复杂部分，
        例如威胁平面、机会向量等。这些值的计算依赖于整个棋盘的局势。
        (此部分为简洁起见暂时省略了具体实现，但在完整的AI训练中至关重要)
        """
        # ... [此处应包含计算威胁、机会等逻辑]
        # ... [例如，调用 _get_threat_plane, _get_opportunity_vector 等函数来填充状态向量]
        return self.action_masks() # 临时返回当前玩家的合法动作掩码

    def _build_state_from_scratch(self):
        """
        在 `reset` 时，从零开始完整地构建双方的状态向量。
        与增量更新不同，此函数会遍历整个棋盘来初始化所有状态信息。
        (此部分为简洁起见暂时省略了具体实现)
        """
        # ... [此处应包含遍历self.board并填充self._state_vector_p1和self._state_vector_p_neg1的逻辑]
        # 初始化完成后，调用一次复杂状态更新
        return self._update_complex_state_vectors()

    def action_masks(self):
        """
        生成当前玩家所有合法动作的掩码。这是走法生成的核心。
        使用Bitboard和预计算表，性能极高。
        返回一个大小为 ACTION_SPACE_SIZE 的数组，合法动作为1，非法为0。
        """
        actions = np.zeros(self.ACTION_SPACE_SIZE, dtype=int)
        player = self.current_player
        
        # 1. 生成翻棋动作 (Reveal)
        # 遍历所有未翻开的棋子，为它们生成翻棋动作。
        hidden_bb = self.hidden_pieces_bitboard
        while hidden_bb > 0:
            # `bit_length() - 1` 是一个快速找到最高位(Most Significant Bit, MSB)索引的方法。
            sq = int(hidden_bb).bit_length() - 1
            # 翻棋动作的 sub_idx 是 4。
            actions[sq * 5 + 4] = 1
            # 使用XOR将该位清零，以便处理下一个未翻开的棋子。
            hidden_bb ^= ULL(sq)

        # 2. 生成移动和攻击动作 (Move & Attack)
        my_pieces_bb = self.occupied_bitboards[player] # 我方所有已翻开的棋子
        all_pieces_bb = self.occupied_bitboards[0]    # 棋盘上所有的棋子

        # 遍历每一种棋子类型
        for pt_val in range(self.NUM_PIECE_TYPES):
            pt = PieceType(pt_val)
            # 获取当前玩家、当前类型的所有棋子的Bitboard
            pieces_bb = self.piece_bitboards[player][pt_val]
            
            # 遍历该类型下的每一个棋子
            while pieces_bb > 0:
                from_sq = int(pieces_bb).bit_length() - 1
                
                if pt == PieceType.B: # 特殊逻辑：炮 (Cannon)
                    # 遍历四个射线方向
                    for i in range(4):
                        # 从预计算表中获取该方向的射线Bitboard
                        ray_bb = self.attack_tables['rays'][i][from_sq]
                        # `&` 运算找到射线上所有的棋子（阻挡物）
                        blockers = ray_bb & all_pieces_bb
                        
                        first_blocker_sq = -1
                        # LSB/MSB: `& -blockers` 快速找到最低位(LSB), `bit_length() - 1` 找到最高位(MSB)
                        # 方向决定了我们是找最近的还是最远的第一个阻挡物
                        if i in [0, 2]: # North, West -> 远离0的方向，找MSB
                            if blockers > 0: first_blocker_sq = int(blockers).bit_length() - 1
                        else: # South, East -> 靠近0的方向，找LSB
                            if blockers > 0: first_blocker_sq = int(blockers & -blockers).bit_length() - 1
                        
                        # 如果在第一个“炮架”后面还能找到棋子，才可能构成攻击
                        if first_blocker_sq != -1:
                            # 获取越过第一个炮架后的射线
                            remaining_ray = self.attack_tables['rays'][i][first_blocker_sq]
                            # 找到这条新射线上的所有棋子
                            blockers_after = remaining_ray & all_pieces_bb
                            target_sq = -1
                            # 再次使用 LSB/MSB 找到第二个棋子（真正的攻击目标）
                            if i in [0, 2]: # North, West
                                if blockers_after > 0: target_sq = int(blockers_after).bit_length() - 1
                            else: # South, East
                                if blockers_after > 0: target_sq = int(blockers_after & -blockers_after).bit_length() - 1
                            
                            # 如果找到了攻击目标
                            if target_sq != -1:
                                # 检查是否满足炮的攻击规则 (不能吃自己人等)
                                if self.board[target_sq] is not None and self.can_attack(self.board[from_sq], self.board[target_sq]):
                                    # 从 (from, to) 反查 action_index 并设为1
                                    # 这里存在一个逻辑上的可优化点：炮的移动方向和普通棋子不同，
                                    # 但为了共用move_to_action表，这里通过遍历查找。
                                    for k, v in self.attack_tables['move_to_action'][from_sq].items():
                                        if k == target_sq:
                                            actions[v] = 1
                                            break
                else: # 普通棋子 (King-like moves)
                    # 从预计算表中获取其一步可达的位置
                    attacks_bb = self.attack_tables['king'][from_sq]
                    # 目标位置不能有我方棋子 (`& ~my_pieces_bb`)
                    valid_moves_bb = attacks_bb & ~my_pieces_bb
                    
                    # 遍历所有有效的目标位置
                    while valid_moves_bb > 0:
                        to_sq = int(valid_moves_bb).bit_length() - 1
                        # 检查目标位置是空地还是可以攻击的敌方棋子
                        if self.board[to_sq] is None or self.can_attack(self.board[from_sq], self.board[to_sq]):
                           # 从预计算表中直接查到 action_index 并设为1
                           actions[self.attack_tables['move_to_action'][from_sq][to_sq]] = 1
                        # 将处理过的位置清零，继续循环
                        valid_moves_bb ^= ULL(to_sq)
                
                # 将处理过的棋子清零，继续处理该类型的下一个棋子
                pieces_bb ^= ULL(from_sq)
        return actions

    def can_attack(self, attacker, defender):
        """根据游戏规则判断攻击是否合法。"""
        # 攻击方必须是已翻开的棋子
        if not attacker.revealed:
            return False
        
        # --- 炮的特殊规则 ---
        if attacker.piece_type == PieceType.B:
            # 炮不能攻击己方已翻开的棋子（但可以隔着己方棋子打对方或未翻开的棋子）
            if defender.revealed and attacker.player == defender.player:
                return False
            return True # 可以攻击对方棋子或未翻开的棋子
        
        # --- 非炮棋子的规则 ---
        # 非炮棋子不能攻击未翻开的棋子
        if not defender.revealed:
            return False
            
        # 不能攻击己方棋子
        if attacker.player == defender.player:
            return False
            
        # 特殊规则：兵(A)能吃将(G)
        if attacker.piece_type == PieceType.A and defender.piece_type == PieceType.G: 
            return True
        
        # 特殊规则：将(G)不能吃兵(A)
        if attacker.piece_type == PieceType.G and defender.piece_type == PieceType.A:
            return False
            
        # 通用规则：大子吃小子
        # Enum的值在这里被用作棋力等级 (G=6 > ... > A=0)
        # 等级相同或更高则可以攻击
        return attacker.piece_type.value >= defender.piece_type.value
        
    def render(self):
        """以人类可读的方式在终端打印当前棋盘状态。"""
        if self.render_mode != 'human': return
        # 定义棋子在终端的显示字符
        red_map = {PieceType.G: "帥", PieceType.F: "仕", PieceType.E: "相", PieceType.D: "俥", PieceType.C: "傌", PieceType.B: "炮", PieceType.A: "兵"}
        black_map = {PieceType.G: "將", PieceType.F: "士", PieceType.E: "象", PieceType.D: "車", PieceType.C: "馬", PieceType.B: "炮", PieceType.A: "卒"}
        print("-" * 21)
        for r in range(4):
            print("|", end="")
            for c in range(4):
                sq = POS_TO_SQ[r, c]
                p = self.board[sq]
                # 使用 `occupied_bitboards[0]` 来判断格子上是否有棋子，这比检查 p is None 更高效。
                if not (self.occupied_bitboards[0] & ULL(sq)):
                    print("    |", end="") # 空位
                elif not p.revealed:
                    print(f" \033[90m暗\033[0m  |", end="") # 未翻开
                elif p.player == 1:
                    print(f" \033[91m{red_map[p.piece_type]}\033[0m  |", end="") # 红方棋子
                else:
                    print(f" \033[94m{black_map[p.piece_type]}\033[0m  |", end="") # 黑方棋子
            print()
        print("-" * 21)
        p_str = "\033[91m红方\033[0m" if self.current_player == 1 else "\033[94m黑方\033[0m"
        print(f"Player: {p_str}, Scores: R={self.scores[1]} B={self.scores[-1]}, Moves: {self.move_counter}/{self.MAX_CONSECUTIVE_MOVES}\n")

    def close(self):
        """清理环境资源，符合Gym接口。"""
        pass