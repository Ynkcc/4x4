# src_code/game/environment.py (已集成威胁与收益塑形)
import os
import random
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any, List, Dict

try:
    from sb3_contrib import MaskablePPO
except ImportError:
    MaskablePPO = None

# ==============================================================================
# --- 游戏常量定义 ---
# ==============================================================================
WINNING_SCORE = 60
MAX_CONSECUTIVE_MOVES_FOR_DRAW = 24
MAX_STEPS_PER_EPISODE = 100

class PieceType(Enum):
    SOLDIER = 0
    CANNON = 1
    HORSE = 2
    CHARIOT = 3
    ELEPHANT = 4
    ADVISOR = 5
    GENERAL = 6

class Piece:
    """棋子对象，存储其类型、所属玩家和是否翻开"""

    def __init__(self, piece_type, player):
        self.piece_type, self.player, self.revealed = piece_type, player, False

    def __repr__(self):
        state = 'R' if self.revealed else 'H'
        player_char = 'R' if self.player == 1 else 'B'
        return f"{state}_{player_char}{self.piece_type.name}"

BOARD_ROWS, BOARD_COLS = 4, 4
NUM_PIECE_TYPES = 7
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS

REVEAL_ACTIONS_COUNT = 16
REGULAR_MOVE_ACTIONS_COUNT = 48
CANNON_ATTACK_ACTIONS_COUNT = 48
ACTION_SPACE_SIZE = REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT

PIECE_VALUES = {pt: val for pt, val in zip(PieceType, [4, 10, 10, 10, 10, 20, 30])}
PIECE_MAX_COUNTS = {pt: val for pt, val in zip(PieceType, [2, 1, 1, 1, 1, 1, 1])}

POS_TO_SQ = {(r, c): r * BOARD_COLS + c for r in range(BOARD_ROWS) for c in range(BOARD_COLS)}
SQ_TO_POS = {sq: (sq // BOARD_COLS, sq % BOARD_COLS) for sq in range(TOTAL_POSITIONS)}


def _get_piece_survival_vector(dead_pieces: list) -> np.ndarray:
    """辅助函数，计算存活棋子的向量"""
    vec = np.ones(8, dtype=np.float32)
    s_count = 0
    for p in dead_pieces:
        if p.piece_type == PieceType.SOLDIER:
            if s_count < 2:
                vec[s_count] = 0.0
            s_count += 1
        else:
            vec[p.piece_type.value + 1] = 0.0
    return vec


class GameEnvironment(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self,
                 render_mode=None,
                 opponent_agent: Optional[Any] = None,
                 opponent_data: Optional[List[Dict[str, Any]]] = None, # 【修改】新的合并参数
                 shaping_coef: float = 0.1):
        super().__init__()
        self.render_mode = render_mode
        self.learning_player_id = 1
        self.shaping_coef = shaping_coef

        # 对手配置
        self.opponent_agent_for_eval = opponent_agent
        # 【修改】现在只使用一个数据结构
        self.opponent_data = opponent_data if opponent_data else []
        self.active_opponent: Optional[Any] = None
        if self.opponent_agent_for_eval:
            self.active_opponent = self.opponent_agent_for_eval

        # --- Gym 环境空间定义 ---
        num_channels = NUM_PIECE_TYPES * 2 + 2
        board_shape = (num_channels, BOARD_ROWS, BOARD_COLS)
        scalar_shape = (3 + 8 + 8, )
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0.0, high=1.0, shape=board_shape, dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=scalar_shape, dtype=np.float32)
        })
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # --- 游戏内部状态 ---
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

        # --- 预计算表 ---
        self.attack_tables = {}
        self.action_to_coords = {}
        self.coords_to_action = {}
        self._initialize_lookup_tables()

    # 【修改】新版本的 reload_opponent_pool 方法，只接收一个参数
    def reload_opponent_pool(self, new_opponent_data: List[Dict[str, Any]]):
        """在训练期间从外部动态更新对手池数据。"""
        self.opponent_data = new_opponent_data
        return True

    def reset(self, seed=None, options=None):
        """重置游戏到初始状态"""
        super().reset(seed=seed)

        if not self.opponent_agent_for_eval and self.opponent_data:
            # 【修改】使用新的数据结构进行采样
            paths = [item['path'] for item in self.opponent_data]
            weights = [item['weight'] for item in self.opponent_data]
            
            chosen_path = self.np_random.choice(paths, p=weights)
            
            # 【修改】直接从数据中获取预加载的模型实例
            chosen_model_data = next((item for item in self.opponent_data if item['path'] == chosen_path), None)
            
            if chosen_model_data is None:
                raise ValueError(f"环境内错误：找不到预加载的对手模型 {chosen_path}。")
            self.active_opponent = chosen_model_data['model']
        elif self.opponent_agent_for_eval:
            self.active_opponent = self.opponent_agent_for_eval
        else:
            self.active_opponent = None

        self._initialize_board()
        return self.get_state(), {'action_mask': self.action_masks(), 'winner': None}

    def step(self, action_index):
        """执行一步完整的游戏逻辑，并应用奖励塑形"""
        reward = 0.0
        
        prev_threat_potential = 0.0
        if self.shaping_coef > 0.0:
            prev_threat_potential = self._get_max_capture_value_for(-self.learning_player_id)

        move_reward, terminated, truncated, winner = self._internal_apply_action(action_index)
        reward += self.shaping_coef * move_reward
        
        if terminated or truncated:
            final_reward = reward
            if terminated:
                if winner == self.learning_player_id: final_reward += 1.0
                elif winner == -self.learning_player_id: final_reward += -0.8
                else: final_reward += -0.1 # 平局略微惩罚
            elif truncated:
                final_reward += -0.5
            return self.get_state(), np.float32(final_reward), terminated, truncated, {
                'winner': winner,
                'action_mask': self.action_masks()
            }

        if self.shaping_coef > 0.0:
            new_threat_potential = self._get_max_capture_value_for(-self.learning_player_id)
            new_benefit_potential = self._get_max_capture_value_for(self.learning_player_id)

            denom = WINNING_SCORE if WINNING_SCORE > 0 else 1.0

            threat_reward = (prev_threat_potential - new_threat_potential) / denom
            benefit_reward = new_benefit_potential / denom
            
            reward += self.shaping_coef * (threat_reward + benefit_reward)

        self.current_player *= -1
        if self.active_opponent:
            obs, mask = self.get_state(), self.action_masks()
            if not np.any(mask):
                terminated, winner, reward = True, self.learning_player_id, reward + 1.0
            else:
                try:
                    opp_action, _ = self.active_opponent.predict(obs, action_masks=mask, deterministic=True)
                    opp_reward, terminated, truncated, winner = self._internal_apply_action(int(opp_action))
                    reward -= self.shaping_coef * opp_reward
                    
                    if terminated:
                        if winner == self.learning_player_id: reward += 1.0
                        elif winner == -self.learning_player_id: reward += -0.8
                        else: reward += -0.1 # 平局略微惩罚
                    elif truncated:
                        reward += -0.5
                except Exception as e:
                    valid_actions = np.where(mask)[0]
                    opp_action = np.random.choice(valid_actions)
                    opp_reward, terminated, truncated, winner = self._internal_apply_action(opp_action)
                    reward -= self.shaping_coef * opp_reward
                    print(f"警告: 对手预测失败 ({e}), 已随机选择动作。")

        self.current_player *= -1
        return self.get_state(), np.float32(reward), terminated, truncated, {
            'winner': winner,
            'action_mask': self.action_masks()
        }

    def _internal_apply_action(self, action_index):
        """仅应用一个动作并更新内部状态，返回即时奖励和游戏是否结束"""
        if not self.action_masks()[action_index]:
            raise ValueError(f"错误：试图执行无效动作! 索引: {action_index}")

        self.total_step_counter += 1
        reward = 0.0

        if action_index < REVEAL_ACTIONS_COUNT:
            self._apply_reveal_update(POS_TO_SQ[self.action_to_coords[action_index]])
            self.move_counter = 0
            reward = 0
        else:
            from_sq, to_sq = POS_TO_SQ[self.action_to_coords[action_index][0]], POS_TO_SQ[
                self.action_to_coords[action_index][1]]
            reward = self._apply_move_action(from_sq, to_sq)

        winner, terminated, truncated = None, False, False
        if self.scores[1] >= WINNING_SCORE:
            winner, terminated = 1, True
        elif self.scores[-1] >= WINNING_SCORE:
            winner, terminated = -1, True

        if not terminated and not truncated and not np.any(self.action_masks(-self.current_player)):
            winner, terminated = self.current_player, True

        if not terminated and not truncated and self.move_counter >= MAX_CONSECUTIVE_MOVES_FOR_DRAW:
            winner, terminated = 0, True
        if not terminated and not truncated and self.total_step_counter >= MAX_STEPS_PER_EPISODE:
            winner, truncated = 0, True

        return reward, terminated, truncated, winner

    def _apply_move_action(self, from_sq, to_sq):
        """处理移动和吃子动作，返回奖励"""
        attacker = self.board[from_sq]
        defender = self.board[to_sq]

        if defender is None:
            self.board[to_sq], self.board[from_sq] = attacker, None
            self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
            self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
            self.revealed_vectors[attacker.player][from_sq] = False
            self.revealed_vectors[attacker.player][to_sq] = True
            self.empty_vector[from_sq], self.empty_vector[to_sq] = True, False
            self.move_counter += 1
            return 0.0
        else:
            points = PIECE_VALUES[defender.piece_type]
            if defender.player == attacker.player:
                opponent = -attacker.player
                self.scores[opponent] += points
                points = - points
            else:
                self.scores[attacker.player] += points
            self.board[to_sq], self.board[from_sq] = attacker, None

            if defender.revealed:
                self.piece_vectors[defender.player][defender.piece_type.value][to_sq] = False
                self.revealed_vectors[defender.player][to_sq] = False
            else:
                self.hidden_vector[to_sq] = False

            self.piece_vectors[attacker.player][attacker.piece_type.value][from_sq] = False
            self.piece_vectors[attacker.player][attacker.piece_type.value][to_sq] = True
            self.revealed_vectors[attacker.player][from_sq] = False
            self.revealed_vectors[attacker.player][to_sq] = True
            self.empty_vector[from_sq] = True

            self.dead_pieces[defender.player].append(defender)
            self.move_counter = 0
            return points / WINNING_SCORE

    def _get_max_capture_value_for(self, player_id: int) -> float:
        """【新增】计算指定玩家一步内能吃掉的对方棋子的最大价值"""
        max_value = 0.0
        opponent_id = -player_id
        action_mask = self.action_masks(player_id)
        valid_actions = np.where(action_mask)[0]

        for action in valid_actions:
            if action >= REVEAL_ACTIONS_COUNT:
                coords = self.action_to_coords.get(action)
                to_sq = POS_TO_SQ[coords[1]]
                defender = self.board[to_sq]

                if defender is not None and defender.player == opponent_id:
                    value = PIECE_VALUES[defender.piece_type]
                    if value > max_value:
                        max_value = value
        return max_value

    def close(self):
        """清理资源"""
        self.active_opponent = None

    def get_state(self):
        """获取当前玩家视角的观察状态"""
        my_player, opponent_player = self.current_player, -self.current_player

        board_state_list = []
        board_state_list.extend(
            [self.piece_vectors[my_player][pt.value].reshape(BOARD_ROWS, BOARD_COLS) for pt in PieceType])
        board_state_list.extend(
            [self.piece_vectors[opponent_player][pt.value].reshape(BOARD_ROWS, BOARD_COLS) for pt in PieceType])
        board_state_list.append(self.hidden_vector.reshape(BOARD_ROWS, BOARD_COLS))
        board_state_list.append(self.empty_vector.reshape(BOARD_ROWS, BOARD_COLS))

        scalar_state = np.concatenate([
            np.array([
                self.scores[my_player] / WINNING_SCORE,
                self.scores[opponent_player] / WINNING_SCORE,
                self.move_counter / MAX_CONSECUTIVE_MOVES_FOR_DRAW
            ],
                     dtype=np.float32),
            _get_piece_survival_vector(self.dead_pieces[my_player]),
            _get_piece_survival_vector(self.dead_pieces[opponent_player])
        ])

        return {"board": np.array(board_state_list, dtype=np.float32), "scalars": scalar_state}

    def action_masks(self, player_id: int = None):
        """为当前玩家计算所有合法动作的掩码"""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        my_player = player_id if player_id is not None else self.current_player

        for sq in np.where(self.hidden_vector)[0]:
            mask[self.coords_to_action[SQ_TO_POS[sq]]] = 1

        target_vectors = {}
        cumulative_targets = self.empty_vector.copy()
        for pt in PieceType:
            cumulative_targets |= self.piece_vectors[-my_player][pt.value]
            target_vectors[pt.value] = cumulative_targets.copy()
        
        target_vectors[PieceType.SOLDIER.value] |= self.piece_vectors[-my_player][PieceType.GENERAL.value]
        target_vectors[PieceType.GENERAL.value] &= ~self.piece_vectors[-my_player][PieceType.SOLDIER.value]

        for pt_val in range(NUM_PIECE_TYPES):
            if pt_val == PieceType.CANNON.value:
                continue
            
            valid_targets_for_pt = target_vectors[pt_val]

            for from_sq in np.where(self.piece_vectors[my_player][pt_val])[0]:
                r, c = SQ_TO_POS[from_sq]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= r + dr < BOARD_ROWS and 0 <= c + dc < BOARD_COLS:
                        to_sq = POS_TO_SQ[(r + dr, c + dc)]
                        if valid_targets_for_pt[to_sq]:
                            idx = self.coords_to_action.get((SQ_TO_POS[from_sq], (r + dr, c + dc)))
                            if idx is not None:
                                mask[idx] = 1
        
        all_pieces_vector = ~self.empty_vector
        valid_cannon_targets = ~self.revealed_vectors[my_player] 
        for from_sq in np.where(self.piece_vectors[my_player][PieceType.CANNON.value])[0]:
            for d_idx in range(4):
                ray = self.attack_tables['rays'][d_idx, from_sq]
                blockers = np.where(ray & all_pieces_vector)[0]
                if len(blockers) >= 2:
                    screen = np.max(blockers) if d_idx in [0, 2] else np.min(blockers)
                    targets_on_ray = blockers[blockers < screen] if d_idx in [0, 2] else blockers[blockers > screen]
                    if targets_on_ray.size > 0:
                        target_sq = np.max(targets_on_ray) if d_idx in [0, 2] else np.min(targets_on_ray)
                        if valid_cannon_targets[target_sq]:
                            idx = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[target_sq]))
                            if idx is not None:
                                mask[idx] = 1
        return mask

    def _initialize_lookup_tables(self):
        """预计算攻击射线和动作映射表"""
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
            from_pos = (r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= r + dr < BOARD_ROWS and 0 <= c + dc < BOARD_COLS:
                    to_sq = POS_TO_SQ[(r + dr, c + dc)]
                    self.action_to_coords[action_idx] = (from_pos, tuple(SQ_TO_POS[to_sq]))
                    self.coords_to_action[(from_pos, tuple(SQ_TO_POS[to_sq]))] = action_idx
                    action_idx += 1
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                from_pos = (r1, c1)
                for c2 in range(BOARD_COLS):
                    if abs(c1 - c2) > 1 and (from_pos, (r1, c2)) not in self.coords_to_action:
                        self.action_to_coords[action_idx] = (from_pos, (r1, c2))
                        self.coords_to_action[(from_pos, (r1, c2))] = action_idx
                        action_idx += 1
                for r2 in range(BOARD_ROWS):
                    if abs(r1 - r2) > 1 and (from_pos, (r2, c1)) not in self.coords_to_action:
                        self.action_to_coords[action_idx] = (from_pos, (r2, c1))
                        self.coords_to_action[(from_pos, (r2, c1))] = action_idx
                        action_idx += 1

    def _reset_all_vectors_and_state(self):
        """重置所有状态变量和bit vectors"""
        self.board.fill(None)
        for p in [1, -1]:
            for i in range(NUM_PIECE_TYPES):
                self.piece_vectors[p][i].fill(False)
            self.revealed_vectors[p].fill(False)
        self.hidden_vector.fill(False)
        self.empty_vector.fill(True)
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.total_step_counter = 0
        self.scores = {-1: 0, 1: 0}

    def _initialize_board(self):
        """初始化棋盘，随机放置棋子"""
        self._reset_all_vectors_and_state()
        pieces = [Piece(pt, p) for pt, count in PIECE_MAX_COUNTS.items() for p in [1, -1] for _ in range(count)]
        if hasattr(self, 'np_random') and self.np_random:
            self.np_random.shuffle(pieces)
        else:
            random.shuffle(pieces)
        for sq, piece in enumerate(pieces):
            self.board[sq] = piece
            self.empty_vector[sq] = False
            self.hidden_vector[sq] = True

    def _apply_reveal_update(self, from_sq):
        """处理翻棋动作并更新 bit vectors"""
        piece = self.board[from_sq]
        piece.revealed = True
        self.hidden_vector[from_sq] = False
        self.revealed_vectors[piece.player][from_sq] = True
        self.piece_vectors[piece.player][piece.piece_type.value][from_sq] = True

    def apply_single_action(self, action_index):
        """
        为GUI提供的公共方法：只执行单个动作，不自动切换玩家或处理对手回合。
        返回: (reward, terminated, truncated, winner)
        """
        return self._internal_apply_action(action_index)