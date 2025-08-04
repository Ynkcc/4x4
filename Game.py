# Game.py - 基于Numpy向量的暗棋环境 (已修改以支持CNN)
import os
import random
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ==============================================================================
# --- 类型定义与常量 ---
# ==============================================================================

class PieceType(Enum):
    """定义棋子类型及其对应索引值"""
    SOLDIER = 0
    CANNON = 1
    HORSE = 2
    CHARIOT = 3
    ELEPHANT = 4
    ADVISOR = 5
    GENERAL = 6

class Piece:
    """棋子对象，存储棋子本身的属性（类型，玩家，是否翻开）。"""
    def __init__(self, piece_type, player):
        self.piece_type, self.player, self.revealed = piece_type, player, False
    def __repr__(self):
        state = 'R' if self.revealed else 'H'
        player_char = 'R' if self.player == 1 else 'B'
        return f"{state}_{player_char}{self.piece_type.name}"

# --- 游戏核心常量 ---
BOARD_ROWS, BOARD_COLS = 4, 4
NUM_PIECE_TYPES = 7
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS

# --- 动作空间定义 ---
REVEAL_ACTIONS_COUNT = 16
REGULAR_MOVE_ACTIONS_COUNT = 48
CANNON_ATTACK_ACTIONS_COUNT = 48
ACTION_SPACE_SIZE = REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT # 112

MAX_CONSECUTIVE_MOVES = 40
WINNING_SCORE = 60

# --- 棋子属性 ---
PIECE_VALUES = {pt: val for pt, val in zip(PieceType, [4, 10, 10, 10, 10, 20, 30])}
# 每个玩家的棋子数量: 2兵, 1炮, 1马, 1车, 1相, 1仕, 1帅
PIECE_MAX_COUNTS = {pt: val for pt, val in zip(PieceType, [2, 1, 1, 1, 1, 1, 1])}

# --- 位置转换工具 ---
POS_TO_SQ = {(r, c): r * BOARD_COLS + c for r in range(BOARD_ROWS) for c in range(BOARD_COLS)}
SQ_TO_POS = {sq: (sq // BOARD_COLS, sq % BOARD_COLS) for sq in range(TOTAL_POSITIONS)}


class GameEnvironment(gym.Env):
    """
    基于Numpy布尔向量的暗棋Gym环境 (支持课程学习)。
    【重要修改】: 状态空间已修改为支持CNN的字典格式。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode=None, curriculum_stage=4, opponent_policy=None):
        super().__init__()
        self.render_mode = render_mode
        self.curriculum_stage = curriculum_stage
        
        # 自我对弈相关属性
        self.learning_player_id = 1  # 学习者始终是玩家1（红方）
        self.opponent_model = None
        
        # 如果提供了对手策略路径，则加载对手模型
        if opponent_policy and os.path.exists(opponent_policy):
            print(f"环境加载对手策略: {opponent_policy}")
            try:
                from sb3_contrib import MaskablePPO
                self.opponent_model = MaskablePPO.load(opponent_policy)
            except Exception as e:
                print(f"警告：无法加载对手模型 {opponent_policy}: {e}")
                self.opponent_model = None

        # --- 【重要修改】状态空间定义 ---
        # 状态被分为两部分：棋盘的“图像”表示和全局的“标量”特征
        # 1. 棋盘 "图像" 部分: (通道数, 高, 宽)
        # 通道包括: 我方7种棋子 + 敌方7种棋子 + 暗棋 + 空位 = 16个通道
        num_channels = NUM_PIECE_TYPES * 2 + 2
        board_shape = (num_channels, BOARD_ROWS, BOARD_COLS)
        
        # 2. 标量特征部分: [我方得分, 敌方得分, 连续未吃子步数, 我方存活棋子(8), 敌方存活棋子(8)]
        # 存活棋子向量的顺序为: [兵, 兵, 炮, 马, 车, 相, 仕, 帅]
        scalar_shape = (3 + 8 + 8,)

        # 使用Dict空间来组合这两部分
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0.0, high=1.0, shape=board_shape, dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=scalar_shape, dtype=np.float32)
        })

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # --- 核心数据结构 (无变化) ---
        self.board = np.empty(TOTAL_POSITIONS, dtype=object)
        self.piece_vectors = {p: [np.zeros(TOTAL_POSITIONS, dtype=bool) for _ in range(NUM_PIECE_TYPES)] for p in [1, -1]}
        self.revealed_vectors = {p: np.zeros(TOTAL_POSITIONS, dtype=bool) for p in [1, -1]}
        self.hidden_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.empty_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)

        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

        self.attack_tables = {}
        self.action_to_coords = {}
        self.coords_to_action = {}
        self._initialize_lookup_tables()

    def _initialize_lookup_tables(self):
        """预计算所有需要的查找表 (无变化)。"""
        ray_attacks = np.zeros((4, TOTAL_POSITIONS, TOTAL_POSITIONS), dtype=bool)
        for sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[sq]
            for i in range(r - 1, -1, -1): ray_attacks[0, sq, POS_TO_SQ[(i, c)]] = True
            for i in range(r + 1, 4):      ray_attacks[1, sq, POS_TO_SQ[(i, c)]] = True
            for i in range(c - 1, -1, -1): ray_attacks[2, sq, POS_TO_SQ[(r, i)]] = True
            for i in range(c + 1, 4):      ray_attacks[3, sq, POS_TO_SQ[(r, i)]] = True
        self.attack_tables['rays'] = ray_attacks
        
        action_idx = 0
        for sq in range(TOTAL_POSITIONS):
            self.action_to_coords[action_idx] = tuple(SQ_TO_POS[sq])
            self.coords_to_action[tuple(SQ_TO_POS[sq])] = action_idx
            action_idx += 1
            
        for from_sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[from_sq]
            from_pos = tuple((r, c))
            for dr, dc in [( -1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= r + dr < BOARD_ROWS and 0 <= c + dc < BOARD_COLS:
                    to_sq = POS_TO_SQ[(r + dr, c + dc)]
                    to_pos = tuple(SQ_TO_POS[to_sq])
                    self.action_to_coords[action_idx] = (from_pos, to_pos)
                    self.coords_to_action[(from_pos, to_pos)] = action_idx
                    action_idx += 1

        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                from_pos = (r1, c1)
                for c2 in range(BOARD_COLS):
                    if abs(c1 - c2) > 1:
                        to_pos = (r1, c2)
                        if (from_pos, to_pos) not in self.coords_to_action:
                            self.action_to_coords[action_idx] = (from_pos, to_pos)
                            self.coords_to_action[(from_pos, to_pos)] = action_idx
                            action_idx += 1
                for r2 in range(BOARD_ROWS):
                     if abs(r1 - r2) > 1:
                        to_pos = (r2, c1)
                        if (from_pos, to_pos) not in self.coords_to_action:
                            self.action_to_coords[action_idx] = (from_pos, to_pos)
                            self.coords_to_action[(from_pos, to_pos)] = action_idx
                            action_idx += 1

    def _reset_all_vectors_and_state(self):
        """辅助函数：重置所有状态向量和游戏变量。"""
        for player in [1, -1]:
            for pt_idx in range(NUM_PIECE_TYPES):
                self.piece_vectors[player][pt_idx].fill(False)
            self.revealed_vectors[player].fill(False)
        
        self.hidden_vector.fill(False)
        self.empty_vector.fill(True)
        self.board.fill(None)
        
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}

    def _update_vectors_from_board(self):
        """辅助函数：根据 self.board 的内容，更新所有Numpy状态向量。"""
        for sq, piece in enumerate(self.board):
            if piece:
                self.empty_vector[sq] = False
                if piece.revealed:
                    self.revealed_vectors[piece.player][sq] = True
                    self.piece_vectors[piece.player][piece.piece_type.value][sq] = True
                else:
                    self.hidden_vector[sq] = True

    def _initialize_board(self):
        """根据课程学习阶段初始化棋盘和所有状态变量。"""
        self._reset_all_vectors_and_state()

        # 阶段 1: 吃子入门
        if self.curriculum_stage == 1:
            red_chariot = Piece(PieceType.CHARIOT, 1)
            red_chariot.revealed = True
            black_soldier = Piece(PieceType.SOLDIER, -1)
            black_soldier.revealed = True
            self.board[POS_TO_SQ[(1, 1)]] = red_chariot
            self.board[POS_TO_SQ[(1, 2)]] = black_soldier
            self.current_player = 1

        # 阶段 2: 简单战斗 (炮吃子)
        elif self.curriculum_stage == 2:
            red_cannon = Piece(PieceType.CANNON, 1)
            red_cannon.revealed = True
            red_horse_mount = Piece(PieceType.HORSE, 1)
            red_horse_mount.revealed = True
            black_soldier_target = Piece(PieceType.SOLDIER, -1)
            black_soldier_target.revealed = True
            self.board[POS_TO_SQ[(0, 0)]] = red_cannon
            self.board[POS_TO_SQ[(0, 1)]] = red_horse_mount
            self.board[POS_TO_SQ[(0, 2)]] = black_soldier_target
            self.current_player = 1

        # 【重要修改】移除阶段3，所有其他情况都视为完整对局
        # 阶段 4: 完整对局 (原始逻辑)
        else:
            pieces = [Piece(pt, p) for pt, count in PIECE_MAX_COUNTS.items() for p in [1, -1] for _ in range(count)]
            if hasattr(self, 'np_random') and self.np_random is not None:
                self.np_random.shuffle(pieces)
            else:
                random.shuffle(pieces)
            
            for sq in range(TOTAL_POSITIONS):
                self.board[sq] = pieces[sq]
            
            self.hidden_vector.fill(True)
            self.empty_vector.fill(False)
            return

        self._update_vectors_from_board()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_board()
        return self.get_state(), {'action_mask': self.action_masks()}
    
    def get_state(self):
        """
        【重要修改】根据当前的状态向量动态生成供CNN模型观察的字典格式状态。
        新增了双方棋子存活状态的向量。
        """
        my_player = self.current_player
        opponent_player = -self.current_player

        # 1. 构建棋盘 "图像" 部分 (16, 4, 4)
        board_state_list = []
        # 我方棋子 (7个通道)
        for pt_val in range(NUM_PIECE_TYPES):
            board_state_list.append(self.piece_vectors[my_player][pt_val].reshape(BOARD_ROWS, BOARD_COLS))
        # 敌方棋子 (7个通道)
        for pt_val in range(NUM_PIECE_TYPES):
            board_state_list.append(self.piece_vectors[opponent_player][pt_val].reshape(BOARD_ROWS, BOARD_COLS))
        # 暗棋 (1个通道)
        board_state_list.append(self.hidden_vector.reshape(BOARD_ROWS, BOARD_COLS))
        # 空位 (1个通道)
        board_state_list.append(self.empty_vector.reshape(BOARD_ROWS, BOARD_COLS))
        
        board_state = np.array(board_state_list, dtype=np.float32)

        # 2. 构建标量特征部分
        # 2.1 基础标量 (得分, 步数)
        score_norm = WINNING_SCORE if WINNING_SCORE > 0 else 1.0
        move_norm = MAX_CONSECUTIVE_MOVES if MAX_CONSECUTIVE_MOVES > 0 else 1.0
        
        base_scalars = np.array([
            self.scores[my_player] / score_norm,
            self.scores[opponent_player] / score_norm,
            self.move_counter / move_norm
        ], dtype=np.float32)

        # 2.2 棋子存活状态向量
        def _get_piece_survival_vector(player):
            """
            生成指定玩家的棋子存活向量 (8个元素)。
            顺序: 兵, 兵, 炮, 马, 车, 相, 仕, 帅
            存活为1, 死亡为0.
            """
            # 向量顺序: [SOLDIER, SOLDIER, CANNON, HORSE, CHARIOT, ELEPHANT, ADVISOR, GENERAL]
            survival_vector = np.ones(8, dtype=np.float32)
            dead_soldier_count = 0
            
            for dead_piece in self.dead_pieces[player]:
                pt = dead_piece.piece_type
                if pt == PieceType.SOLDIER:
                    if dead_soldier_count < 2:
                        survival_vector[dead_soldier_count] = 0.0
                    dead_soldier_count += 1
                else:
                    # pt.value: CANNON=1, HORSE=2, ... GENERAL=6
                    # index: CANNON=2, HORSE=3, ... GENERAL=7
                    # index = pt.value + 1
                    idx = pt.value + 1
                    survival_vector[idx] = 0.0
            return survival_vector

        my_survival_vector = _get_piece_survival_vector(my_player)
        opponent_survival_vector = _get_piece_survival_vector(opponent_player)
        
        # 2.3 组合所有标量
        scalar_state = np.concatenate([base_scalars, my_survival_vector, opponent_survival_vector])
        
        # 3. 组合成字典返回
        return {"board": board_state, "scalars": scalar_state}

    def _internal_apply_action(self, action_index):
        """
        内部方法：应用动作并返回奖励、terminated、truncated状态
        这个方法被 step 和自我对弈逻辑共同使用
        """
        coords = self.action_to_coords.get(action_index)
        if coords is None:
            raise ValueError(f"无效的动作索引: {action_index}")

        # 阶段 1 & 2: 目标驱动的短期对局
        if self.curriculum_stage in [1, 2]:
            reward = -0.1
            terminated = False
            truncated = False
            
            if action_index >= REVEAL_ACTIONS_COUNT:
                from_sq = POS_TO_SQ[coords[0]]
                to_sq = POS_TO_SQ[coords[1]]
                
                if self.board[to_sq] is not None and self.board[to_sq].player == -self.current_player:
                    reward = 1.0
                    terminated = True
                    self._apply_move_action(from_sq, to_sq)
                else:
                    self._apply_move_action(from_sq, to_sq)
            
            if not terminated and self.move_counter >= 5:
                truncated = True
                reward = -1.0

            return reward, terminated, truncated

        # 完整游戏逻辑
        reward = -0.0005
        
        if action_index < REVEAL_ACTIONS_COUNT:
            from_sq = POS_TO_SQ[coords]
            self._apply_reveal_update(from_sq)
            self.move_counter = 0
        else:
            from_sq = POS_TO_SQ[coords[0]]
            to_sq = POS_TO_SQ[coords[1]]
            raw_reward = self._apply_move_action(from_sq, to_sq)
            reward += raw_reward / WINNING_SCORE if WINNING_SCORE > 0 else raw_reward
        
        terminated, truncated, winner = False, False, None
        
        if self.scores[1] >= WINNING_SCORE:
            winner, terminated = 1, True
        elif self.scores[-1] >= WINNING_SCORE:
            winner, terminated = -1, True
        
        # 检查是否有合法动作
        temp_current_player = self.current_player
        self.current_player = -self.current_player  # 临时切换以检查对手的动作
        action_mask = self.action_masks()
        self.current_player = temp_current_player  # 切换回来
        
        if not terminated and not truncated and np.sum(action_mask) == 0:
            winner = self.current_player
            terminated = True

        if not terminated and not truncated and self.move_counter >= MAX_CONSECUTIVE_MOVES:
            winner, truncated = 0, True

        if terminated:
            if winner == self.current_player:
                reward += 1.0
            elif winner == -self.current_player:
                reward -= 1.0
        elif truncated:
            reward -= 0.5

        return reward, terminated, truncated

    def step(self, action_index):
        acting_player = self.current_player
    def step(self, action_index):
        acting_player = self.current_player
        
        # 应用学习者的动作
        reward, terminated, truncated = self._internal_apply_action(action_index)
        
        # 阶段 1 & 2: 目标驱动的短期对局 (保持原有逻辑)
        if self.curriculum_stage in [1, 2]:
            self.current_player = -self.current_player
            info = {'winner': acting_player if terminated else None, 'action_mask': self.action_masks()}
            return self.get_state(), np.float32(reward), terminated, truncated, info

        # 切换玩家
        self.current_player = -self.current_player

        # 自我对弈逻辑：如果是对手的回合并且游戏未结束
        while (self.current_player != self.learning_player_id and 
               self.opponent_model is not None and not terminated and not truncated):

            # 1. 对手观察当前状态
            opponent_obs = self.get_state()
            opponent_mask = self.action_masks()

            # 2. 对手决策
            try:
                opponent_action, _ = self.opponent_model.predict(
                    opponent_obs, action_masks=opponent_mask, deterministic=True
                )
                opponent_action = int(opponent_action)
            except Exception as e:
                print(f"警告：对手模型预测失败: {e}")
                # 如果预测失败，随机选择一个合法动作
                valid_actions = np.where(opponent_mask)[0]
                if len(valid_actions) > 0:
                    opponent_action = np.random.choice(valid_actions)
                else:
                    break  # 没有合法动作，跳出循环

            # 3. 对手执行动作
            try:
                opponent_reward, term, trunc = self._internal_apply_action(opponent_action)
                
                # 4. 累加奖励并更新结束标志
                reward += opponent_reward
                terminated = terminated or term
                truncated = truncated or trunc

                # 5. 如果游戏未结束，切换回学习者
                if not terminated and not truncated:
                    self.current_player = -self.current_player
                else:
                    break  # 游戏已结束，跳出循环
            except Exception as e:
                print(f"警告：对手动作执行失败: {e}")
                break

        # 生成最终的动作掩码
        action_mask = self.action_masks()
        
        # 如果游戏未结束但没有合法动作，设置游戏结束
        if not terminated and not truncated and np.sum(action_mask) == 0:
            winner = acting_player
            terminated = True
            if winner == acting_player:
                reward += 1.0
            elif winner == -acting_player:
                reward -= 1.0

        info = {'winner': acting_player if terminated else None, 'action_mask': action_mask}
        if (terminated or truncated) and self.render_mode == "human":
            self.render()

        return self.get_state(), np.float32(reward), terminated, truncated, info

    def _apply_reveal_update(self, from_sq):
        """应用翻棋动作并更新状态向量。"""
        piece = self.board[from_sq]
        piece.revealed = True
        self.hidden_vector[from_sq] = False
        self.revealed_vectors[piece.player][from_sq] = True
        self.piece_vectors[piece.player][piece.piece_type.value][from_sq] = True

    def _apply_move_action(self, from_sq, to_sq):
        """应用移动或攻击动作并更新状态向量。"""
        attacker = self.board[from_sq]
        defender = self.board[to_sq]
        reward = 0.0

        if defender is None:
            self.board[to_sq], self.board[from_sq] = attacker, None
            self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
            self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
            self.revealed_vectors[attacker.player][from_sq] = False
            self.revealed_vectors[attacker.player][to_sq] = True
            self.empty_vector[from_sq] = True
            self.empty_vector[to_sq] = False
            self.move_counter += 1
            return 0.0

        points = PIECE_VALUES[defender.piece_type]
        is_cannon_attack = attacker.piece_type == PieceType.CANNON
        
        if is_cannon_attack and attacker.player == defender.player:
            self.scores[-attacker.player] += points
            reward = -float(points)
        else:
            self.scores[attacker.player] += points
            reward = float(points)
        
        self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
        self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
        self.revealed_vectors[attacker.player][from_sq] = False
        self.revealed_vectors[attacker.player][to_sq] = True
        
        if defender.revealed:
            self.piece_vectors[defender.player][defender.piece_type.value][to_sq] = False
            self.revealed_vectors[defender.player][to_sq] = False
        else:
            self.hidden_vector[to_sq] = False
        
        self.empty_vector[from_sq] = True
        self.dead_pieces[defender.player].append(defender)
        self.board[to_sq], self.board[from_sq] = attacker, None
        self.move_counter = 0
        
        return reward

    def action_masks(self):
        """生成当前玩家所有合法动作的掩码 (无变化)。"""
        action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        my_player = self.current_player
        
        hidden_squares = np.where(self.hidden_vector)[0]
        for sq in hidden_squares:
            action_mask[self.coords_to_action[SQ_TO_POS[sq]]] = 1
            
        target_vectors = {}
        cumulative_targets = self.empty_vector.copy()
        for pt in PieceType:
            cumulative_targets |= self.piece_vectors[-my_player][pt.value]
            target_vectors[pt.value] = cumulative_targets.copy()
        target_vectors[PieceType.SOLDIER.value] |= self.piece_vectors[-my_player][PieceType.GENERAL.value]
        target_vectors[PieceType.GENERAL.value] &= ~self.piece_vectors[-my_player][PieceType.SOLDIER.value]

        for pt_val in range(NUM_PIECE_TYPES):
            if pt_val == PieceType.CANNON.value: continue
            
            my_pieces_sqs = np.where(self.piece_vectors[my_player][pt_val])[0]
            if my_pieces_sqs.size == 0: continue

            valid_targets_for_pt = target_vectors[pt_val]

            for from_sq in my_pieces_sqs:
                r, c = SQ_TO_POS[from_sq]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    to_r, to_c = r + dr, c + dc
                    if 0 <= to_r < BOARD_ROWS and 0 <= to_c < BOARD_COLS:
                        to_sq = POS_TO_SQ[(to_r, to_c)]
                        if valid_targets_for_pt[to_sq]:
                            action_index = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[to_sq]))
                            if action_index is not None:
                                action_mask[action_index] = 1

        my_cannons_sqs = np.where(self.piece_vectors[my_player][PieceType.CANNON.value])[0]
        if my_cannons_sqs.size > 0:
            all_pieces_vector = ~self.empty_vector
            valid_cannon_targets = ~self.revealed_vectors[my_player]
            
            for from_sq in my_cannons_sqs:
                for direction_idx in range(4):
                    ray_vec = self.attack_tables['rays'][direction_idx, from_sq]
                    blockers_on_ray_indices = np.where(ray_vec & all_pieces_vector)[0]
                    
                    if blockers_on_ray_indices.size < 2: continue

                    if direction_idx == 0 or direction_idx == 2:
                        screen_sq = np.max(blockers_on_ray_indices)
                        target_candidates = blockers_on_ray_indices[blockers_on_ray_indices < screen_sq]
                        if not target_candidates.size: continue
                        target_sq = np.max(target_candidates)
                    else:
                        screen_sq = np.min(blockers_on_ray_indices)
                        target_candidates = blockers_on_ray_indices[blockers_on_ray_indices > screen_sq]
                        if not target_candidates.size: continue
                        target_sq = np.min(target_candidates)

                    if valid_cannon_targets[target_sq]:
                        action_index = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[target_sq]))
                        if action_index is not None:
                            action_mask[action_index] = 1
            
        return action_mask

    def render(self):
        """以人类可读的方式在终端打印当前棋盘状态 (无变化)。"""
        if self.render_mode != 'human': return

        red_map = {p: c for p, c in zip(PieceType, "兵炮马俥相仕帥")}
        black_map = {p: c for p, c in zip(PieceType, "卒炮馬車象士將")}
        
        print("  " + "-" * 21)
        for r in range(BOARD_ROWS):
            print(f"{r} |", end="")
            for c in range(BOARD_COLS):
                sq = POS_TO_SQ[(r, c)]
                if self.empty_vector[sq]:
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