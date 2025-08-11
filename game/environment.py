# game/environment.py
# Game.py - 基于Numpy向量的暗棋环境 (已修改以支持CNN和健壮的对手模型更新)
# 【V5 优化版】: 实现了对手模型的“即时加载”(Just-in-Time Loading)与缓存机制，解决了多环境训练时的内存冗余问题。
import os
import random
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any, List, Dict

# 【新增】导入模型类
try:
    from sb3_contrib import MaskablePPO
except ImportError:
    MaskablePPO = None # 允许在没有安装sb3的情况下至少能导入环境

# ==============================================================================
# --- 类型定义与常量 ---
# ==============================================================================

# --- 游戏核心规则 ---
WINNING_SCORE = 60  # 达到该分数获胜
MAX_CONSECUTIVE_MOVES_FOR_DRAW = 50 # 连续这么多步没有翻棋或吃子，则判为平局 (已采纳建议修改)
MAX_STEPS_PER_EPISODE = 100 # 每局游戏的最大步数，超出则强制截断为平局

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

# --- 棋子属性 ---
PIECE_VALUES = {pt: val for pt, val in zip(PieceType, [4, 10, 10, 10, 10, 20, 30])}
PIECE_MAX_COUNTS = {pt: val for pt, val in zip(PieceType, [2, 1, 1, 1, 1, 1, 1])}

# --- 位置转换工具 ---
POS_TO_SQ = {(r, c): r * BOARD_COLS + c for r in range(BOARD_ROWS) for c in range(BOARD_COLS)}
SQ_TO_POS = {sq: (sq // BOARD_COLS, sq % BOARD_COLS) for sq in range(TOTAL_POSITIONS)}


class GameEnvironment(gym.Env):
    """
    基于Numpy布尔向量的暗棋Gym环境。
    【V5 优化版】: 实现对手模型的即时加载与缓存。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode=None, opponent_agent: Optional[Any] = None, 
                 opponent_pool: Optional[List[str]] = None, opponent_weights: Optional[List[float]] = None):
        super().__init__()
        self.render_mode = render_mode
        
        self.learning_player_id = 1  # 学习者始终是玩家1（红方）
        
        # --- 对手管理核心变更 ---
        self.opponent_agent_for_eval = opponent_agent # 用于评估模式的固定对手
        self.opponent_pool = opponent_pool if opponent_pool else []
        self.opponent_weights = opponent_weights if opponent_weights else []
        
        # 【优化】loaded_opponents 现在作为一个缓存存在，而不是预加载所有模型
        self.loaded_opponents: Dict[str, Any] = {} 
        self.active_opponent: Optional[Any] = None

        # 如果是评估模式，则直接加载并设置固定对手
        if self.opponent_agent_for_eval:
            self.active_opponent = self.opponent_agent_for_eval
        
        # 【移除】不再在初始化时调用 _load_opponent_pool()

        # --- 状态空间和动作空间定义 (不变) ---
        num_channels = NUM_PIECE_TYPES * 2 + 2
        board_shape = (num_channels, BOARD_ROWS, BOARD_COLS)
        scalar_shape = (3 + 8 + 8,)

        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0.0, high=1.0, shape=board_shape, dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=scalar_shape, dtype=np.float32)
        })

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # --- 游戏状态变量 (不变) ---
        self.board = np.empty(TOTAL_POSITIONS, dtype=object)
        self.piece_vectors = {p: [np.zeros(TOTAL_POSITIONS, dtype=bool) for _ in range(NUM_PIECE_TYPES)] for p in [1, -1]}
        self.revealed_vectors = {p: np.zeros(TOTAL_POSITIONS, dtype=bool) for p in [1, -1]}
        self.hidden_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.empty_vector = np.zeros(TOTAL_POSITIONS, dtype=bool)

        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.total_step_counter = 0
        self.scores = {-1: 0, 1: 0}

        self.attack_tables = {}
        self.action_to_coords = {}
        self.coords_to_action = {}
        self._initialize_lookup_tables()

    # 【移除】_load_opponent_pool 方法被移除，其功能被合并到 reset 方法中

    def reload_opponent_pool(self, new_pool: List[str], new_weights: List[float]):
        """
        【优化】公开的方法，用于更新对手池路径和权重。
        此方法现在只更新列表，不执行任何模型加载，因此非常迅速。
        同时会清空模型缓存，以确保能加载到最新的模型文件。
        """
        print(f"进程 {os.getpid()}: 收到指令，正在更新对手池路径...")
        self.opponent_pool = new_pool
        self.opponent_weights = new_weights
        
        # 清空缓存，以便在下次需要时能加载更新后的模型文件（例如main_opponent.zip被覆盖后）
        self.loaded_opponents.clear() 
        print(f"进程 {os.getpid()}: 对手池路径已更新，模型缓存已清空。")
        return True

    def reset(self, seed=None, options=None):
        """
        【核心优化】在重置游戏时，按需选择并加载单个对手模型。
        """
        super().reset(seed=seed)

        # 仅在训练模式下（即非评估模式）且对手池非空时，才选择和加载对手
        if not self.opponent_agent_for_eval and self.opponent_pool:
            if MaskablePPO is None:
                raise ImportError("需要安装 `sb3-contrib` 库来加载对手模型。")

            # 1. 根据权重选择一个对手模型的 *路径*
            normalized_weights = np.array(self.opponent_weights) / np.sum(self.opponent_weights)
            chosen_path = self.np_random.choice(self.opponent_pool, p=normalized_weights)
            
            # 2. 检查缓存中是否已有该模型
            if chosen_path in self.loaded_opponents:
                self.active_opponent = self.loaded_opponents[chosen_path]
            else:
                # 3. 如果不在缓存中，则从磁盘加载并存入缓存
                try:
                    if not os.path.exists(chosen_path):
                         raise FileNotFoundError(f"找不到对手模型文件: {chosen_path}")
                    # print(f"进程 {os.getpid()}: 正在加载对手 {os.path.basename(chosen_path)}...") # 可选的调试信息
                    model = MaskablePPO.load(chosen_path, device='cpu')
                    self.loaded_opponents[chosen_path] = model
                    self.active_opponent = model
                except Exception as e:
                    print(f"环境内错误：加载对手模型 {chosen_path} 失败: {e}。本局游戏将没有对手。")
                    self.active_opponent = None
        
        elif self.opponent_agent_for_eval:
             # 评估模式下，使用固定的对手
             self.active_opponent = self.opponent_agent_for_eval
        else:
             # 如果没有对手池也没有评估对手，则对手为None
             self.active_opponent = None

        self._initialize_board()
        info = {'action_mask': self.action_masks(self.current_player), 'winner': None}
        return self.get_state(), info

    def step(self, action_index):
        # 1. 应用学习者的动作
        reward, terminated, truncated, winner = self._internal_apply_action(action_index)

        # 2. 如果学习者的动作直接结束游戏，准备信息并返回
        if terminated or truncated:
            info = {'winner': winner, 'action_mask': self.action_masks(self.current_player)}
            # 在返回前，增加对中间奖励的应用
            final_reward = reward
            if terminated:
                if winner == self.learning_player_id: final_reward = 1.0
                elif winner == -self.learning_player_id: final_reward = -1.0
                else: final_reward = 0.0
            elif truncated:
                final_reward = -0.5
            return self.get_state(), np.float32(final_reward), terminated, truncated, info

        # 3. 进入对手回合
        self.current_player = -self.current_player
        opponent_final_reward = 0.0

        if self.active_opponent is not None:
            opponent_obs = self.get_state()
            opponent_mask = self.action_masks(self.current_player)

            if np.sum(opponent_mask) == 0:
                winner = self.learning_player_id
                terminated = True
            else:
                try:
                    opponent_action, _ = self.active_opponent.predict(
                        opponent_obs, action_masks=opponent_mask, deterministic=True
                    )
                    opponent_action = int(opponent_action)
                except Exception as e:
                    print(f"警告：对手模型预测失败: {e}。将随机选择一个有效动作。")
                    valid_actions = np.where(opponent_mask)[0]
                    opponent_action = np.random.choice(valid_actions)
                
                opponent_reward, terminated, truncated, winner = self._internal_apply_action(opponent_action)
                
                # 计算对手的最终回合奖励
                if terminated:
                    if winner == self.current_player: opponent_final_reward = 1.0
                    elif winner == -self.current_player: opponent_final_reward = -1.0
                    else: opponent_final_reward = 0.0
                elif truncated:
                    opponent_final_reward = -0.5
                else:
                    opponent_final_reward = opponent_reward
        
        # 4. 学习者的总奖励 = 自己的中间奖励 - 对手的最终回合奖励
        final_reward = reward - opponent_final_reward
        
        # 5. 准备返回
        self.current_player = self.learning_player_id
        
        final_obs = self.get_state()
        final_mask = self.action_masks(self.current_player)
        info = {'winner': winner, 'action_mask': final_mask}

        if (terminated or truncated) and self.render_mode == "human":
            self.render()

        return final_obs, np.float32(final_reward), terminated, truncated, info

    def _internal_apply_action(self, action_index):
        # ... (此函数前半部分不变) ...
        coords = self.action_to_coords.get(action_index)
        if coords is None:
            raise ValueError(f"错误：无效的动作索引: {action_index}")
        
        action_mask = self.action_masks(self.current_player)
        if not action_mask[action_index]:
            debug_info = self.get_debug_state_string()
            error_msg = (f"\n{'='*80}\n错误：试图执行无效动作!\n"
                         f"  - 动作索引: {action_index}\n  - 动作坐标: {coords}\n"
                         f"  - 当前玩家: {self.current_player}\n{'='*80}\n"
                         f"环境当前状态:\n{debug_info}\n"
                         f"  - 当前合法的动作掩码中，值为1的数量: {np.sum(action_mask)}\n{'='*80}")
            raise ValueError(error_msg)

        self.total_step_counter += 1
        winner = None
        intermediate_reward = 0.0 # 【修改】使用中间奖励
        
        if action_index < REVEAL_ACTIONS_COUNT:
            from_sq = POS_TO_SQ[coords]
            self._apply_reveal_update(from_sq)
            self.move_counter = 0
        else:
            from_sq = POS_TO_SQ[coords[0]]
            to_sq = POS_TO_SQ[coords[1]]
            intermediate_reward = self._apply_move_action(from_sq, to_sq) # 【修改】接收吃子奖励
        
        terminated, truncated = False, False
        
        if self.scores[1] >= WINNING_SCORE:
            winner, terminated = 1, True
        elif self.scores[-1] >= WINNING_SCORE:
            winner, terminated = -1, True
        
        opponent_player_id = -self.current_player
        opponent_action_mask = self.action_masks(opponent_player_id)
        if not terminated and not truncated and np.sum(opponent_action_mask) == 0:
            winner = self.current_player
            terminated = True

        if not terminated and not truncated and self.move_counter >= MAX_CONSECUTIVE_MOVES_FOR_DRAW:
            winner, terminated = 0, True

        if not terminated and not truncated and self.total_step_counter >= MAX_STEPS_PER_EPISODE:
            winner, truncated = 0, True

        # 【修改】此函数现在只返回中间奖励。最终奖励在 step() 中计算。
        return intermediate_reward, terminated, truncated, winner


    def _apply_move_action(self, from_sq, to_sq):
        attacker = self.board[from_sq]
        defender = self.board[to_sq]
        if attacker is None or not attacker.revealed:
            raise ValueError(f"错误：试图从 {from_sq} 移动空或未翻开的棋子")
        
        if defender is None:
            # 移动到空格
            self.board[to_sq], self.board[from_sq] = attacker, None
            self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
            self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
            self.revealed_vectors[attacker.player][from_sq] = False
            self.revealed_vectors[attacker.player][to_sq] = True
            self.empty_vector[from_sq] = True
            self.empty_vector[to_sq] = False
            self.move_counter += 1
            return 0.0 # 【修改】移动不产生奖励
        
        # --- 吃子逻辑 ---
        points = PIECE_VALUES[defender.piece_type]
        is_cannon_attack = attacker.piece_type == PieceType.CANNON
        
        if is_cannon_attack and attacker.player == defender.player:
            self.scores[-attacker.player] += points
        else:
            self.scores[attacker.player] += points
            
        # 更新棋盘向量
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

        # 【修改】返回归一化的吃子奖励
        return (points / WINNING_SCORE) * 0.1

    def close(self):
        """关闭环境时，清空缓存，释放模型占用的内存。"""
        self.loaded_opponents.clear()
        self.active_opponent = None
        # print(f"进程 {os.getpid()}: 环境关闭，模型缓存已清空。") # 可选的调试信息
        pass
    
    # ==========================================================================
    # --- 以下是无需修改的辅助函数 ---
    # _initialize_lookup_tables, _reset_all_vectors_and_state, 
    # _update_vectors_from_board, _initialize_board, get_state,
    # _apply_reveal_update, action_masks, render, get_debug_state_string
    # ==========================================================================
    def _initialize_lookup_tables(self):
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
        self.total_step_counter = 0
        self.scores = {-1: 0, 1: 0}

    def _update_vectors_from_board(self):
        for sq, piece in enumerate(self.board):
            if piece:
                self.empty_vector[sq] = False
                if piece.revealed:
                    self.revealed_vectors[piece.player][sq] = True
                    self.piece_vectors[piece.player][piece.piece_type.value][sq] = True
                else:
                    self.hidden_vector[sq] = True

    def _initialize_board(self):
        self._reset_all_vectors_and_state()
        pieces = [Piece(pt, p) for pt, count in PIECE_MAX_COUNTS.items() for p in [1, -1] for _ in range(count)]
        if hasattr(self, 'np_random') and self.np_random is not None:
            self.np_random.shuffle(pieces)
        else:
            random.shuffle(pieces)
        
        for sq in range(TOTAL_POSITIONS):
            self.board[sq] = pieces[sq]
        
        self.hidden_vector.fill(True)
        self.empty_vector.fill(False)
        self._update_vectors_from_board()
    
    def get_state(self):
        my_player = self.current_player
        opponent_player = -self.current_player

        board_state_list = []
        for pt_val in range(NUM_PIECE_TYPES):
            board_state_list.append(self.piece_vectors[my_player][pt_val].reshape(BOARD_ROWS, BOARD_COLS))
        for pt_val in range(NUM_PIECE_TYPES):
            board_state_list.append(self.piece_vectors[opponent_player][pt_val].reshape(BOARD_ROWS, BOARD_COLS))
        board_state_list.append(self.hidden_vector.reshape(BOARD_ROWS, BOARD_COLS))
        board_state_list.append(self.empty_vector.reshape(BOARD_ROWS, BOARD_COLS))
        board_state = np.array(board_state_list, dtype=np.float32)

        score_norm = WINNING_SCORE if WINNING_SCORE > 0 else 1.0
        move_norm = MAX_CONSECUTIVE_MOVES_FOR_DRAW if MAX_CONSECUTIVE_MOVES_FOR_DRAW > 0 else 1.0
        
        base_scalars = np.array([
            self.scores[my_player] / score_norm,
            self.scores[opponent_player] / score_norm,
            self.move_counter / move_norm
        ], dtype=np.float32)

        def _get_piece_survival_vector(player):
            survival_vector = np.ones(8, dtype=np.float32)
            dead_soldier_count = 0
            for dead_piece in self.dead_pieces[player]:
                pt = dead_piece.piece_type
                if pt == PieceType.SOLDIER:
                    if dead_soldier_count < 2:
                        survival_vector[dead_soldier_count] = 0.0
                    dead_soldier_count += 1
                else:
                    idx = pt.value + 1
                    survival_vector[idx] = 0.0
            return survival_vector

        my_survival_vector = _get_piece_survival_vector(my_player)
        opponent_survival_vector = _get_piece_survival_vector(opponent_player)
        scalar_state = np.concatenate([base_scalars, my_survival_vector, opponent_survival_vector])
        return {"board": board_state, "scalars": scalar_state}

    def _apply_reveal_update(self, from_sq):
        piece = self.board[from_sq]
        if piece is None or piece.revealed:
            raise ValueError(f"错误：试图翻开空或已翻开的位置 {from_sq}")
        piece.revealed = True
        self.hidden_vector[from_sq] = False
        self.revealed_vectors[piece.player][from_sq] = True
        self.piece_vectors[piece.player][piece.piece_type.value][from_sq] = True

    def action_masks(self, player_id: int = None):
        action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        my_player = player_id if player_id is not None else self.current_player
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
              f"连续未吃/翻子: {self.move_counter}/{MAX_CONSECUTIVE_MOVES_FOR_DRAW}\n")

    def get_debug_state_string(self) -> str:
        red_map = {p: c for p, c in zip(PieceType, "兵炮马俥相仕帥")}
        black_map = {p: c for p, c in zip(PieceType, "卒炮馬車象士將")}
        board_str = "  " + "-" * 21 + "\n"
        for r in range(4):
            board_str += f"{r} |"
            for c in range(4):
                sq = POS_TO_SQ[(r, c)]
                piece = self.board[sq]
                if piece is None:
                    board_str += "    |"
                elif not piece.revealed:
                    board_str += " 暗  |"
                elif piece.player == 1:
                    board_str += f" {red_map[piece.piece_type]}  |"
                else:
                    board_str += f" {black_map[piece.piece_type]}  |"
            board_str += "\n  " + "-" * 21 + "\n"
        board_str += "    " + "   ".join(str(c) for c in range(4))
        state_str = (
            f"棋盘布局:\n{board_str}\n\n"
            f"核心状态:\n"
            f"  - 当前玩家: {'红方(1)' if self.current_player == 1 else '黑方(-1)'}\n"
            f"  - 连续未吃子步数: {self.move_counter}\n"
            f"  - 总步数: {self.total_step_counter}\n"
            f"  - 得分 (红/黑): {self.scores[1]} / {self.scores[-1]}\n"
        )
        return state_str