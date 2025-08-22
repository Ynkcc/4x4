# src_code/game/environment.py (最终重构版)
import os
import random
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any, List, Dict, Tuple

# ==============================================================================
# --- 游戏常量与类型定义 ---
# ==============================================================================

WINNING_SCORE = 60
MAX_CONSECUTIVE_MOVES_FOR_DRAW = 24
MAX_STEPS_PER_EPISODE = 100
BOARD_ROWS, BOARD_COLS = 4, 4
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS
NUM_PIECE_TYPES = 7

class PieceType(Enum):
    SOLDIER = 0
    CANNON = 1
    HORSE = 2
    CHARIOT = 3
    ELEPHANT = 4
    ADVISOR = 5
    GENERAL = 6

PIECE_VALUES = {pt: val for pt, val in zip(PieceType, [4, 10, 10, 10, 10, 20, 30])}
PIECE_MAX_COUNTS = {pt: val for pt, val in zip(PieceType, [2, 1, 1, 1, 1, 1, 1])}

PIECE_SURVIVAL_VEC_INFO = {
    PieceType.SOLDIER:  {'start_idx': 0, 'count': 2},
    PieceType.CANNON:   {'start_idx': 2, 'count': 1},
    PieceType.HORSE:    {'start_idx': 3, 'count': 1},
    PieceType.CHARIOT:  {'start_idx': 4, 'count': 1},
    PieceType.ELEPHANT: {'start_idx': 5, 'count': 1},
    PieceType.ADVISOR:  {'start_idx': 6, 'count': 1},
    PieceType.GENERAL:  {'start_idx': 7, 'count': 1},
}

REVEAL_ACTIONS_COUNT = 16
REGULAR_MOVE_ACTIONS_COUNT = 48
CANNON_ATTACK_ACTIONS_COUNT = 48
ACTION_SPACE_SIZE = REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT

POS_TO_SQ = {(r, c): r * BOARD_COLS + c for r in range(BOARD_ROWS) for c in range(BOARD_COLS)}
SQ_TO_POS = {sq: (sq // BOARD_COLS, sq % BOARD_COLS) for sq in range(TOTAL_POSITIONS)}


# ==============================================================================
# --- Piece 类 ---
# ==============================================================================

class Piece:
    """棋子对象，存储其类型、所属玩家和是否翻开"""
    def __init__(self, piece_type: PieceType, player: int):
        self.piece_type = piece_type
        self.player = player
        self.revealed = False

    def __repr__(self) -> str:
        state = 'R' if self.revealed else 'H'
        player_char = 'R' if self.player == 1 else 'B'
        return f"{state}_{player_char}{self.piece_type.name}"


# ==============================================================================
# --- GameEnvironment 类 ---
# ==============================================================================

class GameEnvironment(gym.Env):
    """
    暗棋游戏环境，兼容Gymnasium接口，并为自我对弈训练进行了优化。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 opponent_agent: Optional[Any] = None,
                 opponent_data: Optional[List[Dict[str, Any]]] = None,
                 shaping_coef: float = 0.1):
        super().__init__()
        self.render_mode = render_mode
        self.shaping_coef = shaping_coef
        self.learning_player_id = 1
        self.opponent_data = opponent_data if opponent_data else []
        self.active_opponent = opponent_agent
        self._initialize_spaces()
        self.board: np.ndarray = np.empty(TOTAL_POSITIONS, dtype=object)
        self.piece_vectors: Dict[int, List[np.ndarray]] = {}
        self.revealed_vectors: Dict[int, np.ndarray] = {}
        self.hidden_vector: np.ndarray = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.empty_vector: np.ndarray = np.zeros(TOTAL_POSITIONS, dtype=bool)
        self.dead_pieces: Dict[int, List[Piece]] = {}
        self.current_player: int = 1
        self.move_counter: int = 0
        self.total_step_counter: int = 0
        self.scores: Dict[int, int] = {}
        self.survival_vectors: Dict[int, np.ndarray] = {}
        self.attack_tables: Dict[str, np.ndarray] = {}
        self.action_to_coords: Dict[int, Any] = {}
        self.coords_to_action: Dict[Any, int] = {}
        self._initialize_lookup_tables()

    def _initialize_spaces(self):
        num_channels = NUM_PIECE_TYPES * 2 + 2
        board_shape = (num_channels, BOARD_ROWS, BOARD_COLS)
        scalar_shape = (3 + 8 + 8,)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0.0, high=1.0, shape=board_shape, dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=scalar_shape, dtype=np.float32)
        })
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    def _initialize_lookup_tables(self):
        """预计算所有查找表以加速游戏。"""
        self._precompute_ray_attacks()
        self._precompute_action_mappings()

    def _reset_internal_state(self):
        self.board.fill(None)
        self.piece_vectors = {p: [np.zeros(TOTAL_POSITIONS, dtype=bool) for _ in range(NUM_PIECE_TYPES)] for p in [1, -1]}
        self.revealed_vectors = {p: np.zeros(TOTAL_POSITIONS, dtype=bool) for p in [1, -1]}
        self.hidden_vector.fill(False)
        self.empty_vector.fill(True)
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.total_step_counter = 0
        self.scores = {-1: 0, 1: 0}
        self.survival_vectors = {p: np.ones(8, dtype=np.float32) for p in [1, -1]}

    def reload_opponent_pool(self, new_opponent_data: List[Dict[str, Any]]):
        self.opponent_data = new_opponent_data
        return True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        super().reset(seed=seed)
        self._select_active_opponent()
        self._initialize_board()
        info = {'action_mask': self.action_masks(), 'winner': None}
        return self.get_state(), info

    def _select_active_opponent(self):
        if self.active_opponent and not self.opponent_data: return
        if not self.opponent_data:
            self.active_opponent = None
            return
        paths = [item['path'] for item in self.opponent_data]
        weights = [item['weight'] for item in self.opponent_data]
        chosen_path = self.np_random.choice(paths, p=weights)
        chosen_model_data = next((item for item in self.opponent_data if item['path'] == chosen_path), None)
        if chosen_model_data is None: raise ValueError(f"环境内错误：找不到预加载的对手模型 {chosen_path}。")
        self.active_opponent = chosen_model_data['model']

    def step(self, action_index: int) -> Tuple[dict, float, bool, bool, dict]:
        prev_threat = self._calculate_threat_potential()
        move_reward, terminated, truncated, winner = self._internal_apply_action(action_index)
        reward = self.shaping_coef * move_reward
        if terminated or truncated:
            final_reward = self._calculate_final_reward(reward, winner, terminated)
            return self.get_state(), np.float32(final_reward), terminated, truncated, {'winner': winner, 'action_mask': self.action_masks()}
        reward += self._calculate_shaping_reward(prev_threat)
        self.current_player *= -1
        if self.active_opponent:
            opp_reward, terminated, truncated, winner = self._execute_opponent_move()
            reward -= self.shaping_coef * opp_reward
            if terminated or truncated:
                final_reward = self._calculate_final_reward(reward, winner, terminated)
                return self.get_state(), np.float32(final_reward), terminated, truncated, {'winner': winner, 'action_mask': self.action_masks()}
        self.current_player *= -1
        return self.get_state(), np.float32(reward), terminated, truncated, {'winner': winner, 'action_mask': self.action_masks()}

    def _internal_apply_action(self, action_index: int) -> Tuple[float, bool, bool, Optional[int]]:
        if not self.action_masks()[action_index]: raise ValueError(f"错误：试图执行无效动作! 索引: {action_index}")
        self.total_step_counter += 1
        immediate_reward = 0.0
        if action_index < REVEAL_ACTIONS_COUNT:
            sq = POS_TO_SQ[self.action_to_coords[action_index]]
            piece = self.board[sq]
            piece.revealed = True
            self.hidden_vector[sq] = False
            self.revealed_vectors[piece.player][sq] = True
            self.piece_vectors[piece.player][piece.piece_type.value][sq] = True
            self.move_counter = 0
        else:
            from_sq, to_sq = POS_TO_SQ[self.action_to_coords[action_index][0]], POS_TO_SQ[self.action_to_coords[action_index][1]]
            immediate_reward = self._apply_move_action(from_sq, to_sq)
        terminated, truncated, winner = self._check_game_over_conditions()
        return immediate_reward, terminated, truncated, winner

    def _apply_move_action(self, from_sq: int, to_sq: int) -> float:
        attacker = self.board[from_sq]
        defender = self.board[to_sq]
        self.board[to_sq], self.board[from_sq] = attacker, None
        self._update_vectors_for_move(attacker.player, attacker.piece_type, from_sq, to_sq)
        if defender is None:
            self.move_counter += 1
            return 0.0
        else:
            self._update_vectors_for_capture(defender, to_sq)
            self.dead_pieces[defender.player].append(defender)
            self._update_survival_vector_on_capture(defender)
            self.move_counter = 0
            points = PIECE_VALUES[defender.piece_type]
            if defender.player == attacker.player:
                self.scores[-attacker.player] += points
                return -points / WINNING_SCORE
            else:
                self.scores[attacker.player] += points
                return points / WINNING_SCORE

    def _check_game_over_conditions(self) -> Tuple[bool, bool, Optional[int]]:
        if self.scores[1] >= WINNING_SCORE: return True, False, 1
        if self.scores[-1] >= WINNING_SCORE: return True, False, -1
        if not np.any(self.action_masks(-self.current_player)): return True, False, self.current_player
        if self.move_counter >= MAX_CONSECUTIVE_MOVES_FOR_DRAW: return True, False, 0
        if self.total_step_counter >= MAX_STEPS_PER_EPISODE: return False, True, 0
        return False, False, None

    def get_state(self) -> Dict[str, np.ndarray]:
        return {"board": self._get_board_state_tensor(), "scalars": self._get_scalar_state_vector()}

    def _get_board_state_tensor(self) -> np.ndarray:
        my_player, opponent_player = self.current_player, -self.current_player
        tensors = [vec.reshape(BOARD_ROWS, BOARD_COLS) for vec in self.piece_vectors[my_player]]
        tensors.extend([vec.reshape(BOARD_ROWS, BOARD_COLS) for vec in self.piece_vectors[opponent_player]])
        tensors.append(self.hidden_vector.reshape(BOARD_ROWS, BOARD_COLS))
        tensors.append(self.empty_vector.reshape(BOARD_ROWS, BOARD_COLS))
        return np.array(tensors, dtype=np.float32)

    def _get_scalar_state_vector(self) -> np.ndarray:
        my_player, opponent_player = self.current_player, -self.current_player
        return np.concatenate([
            np.array([self.scores[my_player] / WINNING_SCORE, self.scores[opponent_player] / WINNING_SCORE,
                      self.move_counter / MAX_CONSECUTIVE_MOVES_FOR_DRAW], dtype=np.float32),
            self.survival_vectors[my_player], self.survival_vectors[opponent_player]
        ])

    def action_masks(self, player_id: int = None) -> np.ndarray:
        player = player_id if player_id is not None else self.current_player
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        reveal_actions = np.where(self.hidden_vector)[0]
        for sq in reveal_actions: mask[self.coords_to_action[SQ_TO_POS[sq]]] = 1
        self._add_regular_move_masks(mask, player)
        self._add_cannon_attack_masks(mask, player)
        return mask

    def _get_valid_target_vectors(self, player: int) -> Dict[PieceType, np.ndarray]:
        opponent = -player
        target_vectors, cumulative_targets = {}, self.empty_vector.copy()
        for pt in PieceType:
            cumulative_targets |= self.piece_vectors[opponent][pt.value]
            target_vectors[pt] = cumulative_targets.copy()
        target_vectors[PieceType.SOLDIER] |= self.piece_vectors[opponent][PieceType.GENERAL.value]
        target_vectors[PieceType.GENERAL] &= ~self.piece_vectors[opponent][PieceType.SOLDIER.value]
        return target_vectors

    def _add_regular_move_masks(self, mask: np.ndarray, player: int):
        target_vectors = self._get_valid_target_vectors(player)
        for pt in PieceType:
            if pt == PieceType.CANNON: continue
            valid_targets_for_pt = target_vectors[pt]
            piece_locations = np.where(self.piece_vectors[player][pt.value])[0]
            for from_sq in piece_locations:
                r, c = SQ_TO_POS[from_sq]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= r + dr < BOARD_ROWS and 0 <= c + dc < BOARD_COLS:
                        to_sq = POS_TO_SQ[(r + dr, c + dc)]
                        if valid_targets_for_pt[to_sq]:
                            idx = self.coords_to_action.get((SQ_TO_POS[from_sq], (r + dr, c + dc)))
                            if idx is not None: mask[idx] = 1

    def _add_cannon_attack_masks(self, mask: np.ndarray, player: int):
        all_pieces_vector, valid_cannon_targets = ~self.empty_vector, ~self.revealed_vectors[player]
        cannon_locations = np.where(self.piece_vectors[player][PieceType.CANNON.value])[0]
        for from_sq in cannon_locations:
            for d_idx in range(4):
                ray = self.attack_tables['rays'][d_idx, from_sq]
                blockers = np.where(ray & all_pieces_vector)[0]
                if len(blockers) >= 2:
                    is_pos_dir = d_idx in [1, 3]
                    screen = np.min(blockers) if is_pos_dir else np.max(blockers)
                    targets = blockers[blockers > screen] if is_pos_dir else blockers[blockers < screen]
                    if targets.size > 0:
                        target_sq = np.min(targets) if is_pos_dir else np.max(targets)
                        if valid_cannon_targets[target_sq]:
                            idx = self.coords_to_action.get((SQ_TO_POS[from_sq], SQ_TO_POS[target_sq]))
                            if idx is not None: mask[idx] = 1

    def _execute_opponent_move(self) -> Tuple[float, bool, bool, Optional[int]]:
        obs, mask = self.get_state(), self.action_masks()
        if not np.any(mask): return 0.0, True, False, -self.current_player
        try:
            opp_action, _ = self.active_opponent.predict(obs, action_masks=mask, deterministic=True)
            return self._internal_apply_action(int(opp_action))
        except Exception as e:
            print(f"警告: 对手预测失败 ({e}), 已随机选择动作。")
            valid_actions = np.where(mask)[0]
            return self._internal_apply_action(np.random.choice(valid_actions))

    def _calculate_final_reward(self, reward: float, winner: int, term: bool) -> float:
        if term:
            if winner == self.learning_player_id: return reward + 1.0
            if winner == -self.learning_player_id: return reward - 0.8
            return reward - 0.1
        return reward - 0.5

    def _calculate_shaping_reward(self, prev_threat: float) -> float:
        if self.shaping_coef <= 0.0: return 0.0
        new_threat = self._get_max_capture_value_for(-self.learning_player_id)
        benefit = self._get_max_capture_value_for(self.learning_player_id)
        denom = max(WINNING_SCORE, 1.0)
        return self.shaping_coef * ((prev_threat - new_threat) / denom + benefit / denom)

    def _calculate_threat_potential(self) -> float:
        return self._get_max_capture_value_for(-self.learning_player_id) if self.shaping_coef > 0.0 else 0.0

    def _update_vectors_for_move(self, player: int, pt: PieceType, from_sq: int, to_sq: int):
        pt_val = pt.value
        self.piece_vectors[player][pt_val][from_sq] = False
        self.piece_vectors[player][pt_val][to_sq] = True
        self.revealed_vectors[player][from_sq] = False
        self.revealed_vectors[player][to_sq] = True
        self.empty_vector[from_sq], self.empty_vector[to_sq] = True, False

    def _update_vectors_for_capture(self, defender: Piece, at_sq: int):
        if defender.revealed:
            self.piece_vectors[defender.player][defender.piece_type.value][at_sq] = False
            self.revealed_vectors[defender.player][at_sq] = False
        else:
            self.hidden_vector[at_sq] = False

    def _update_survival_vector_on_capture(self, captured_piece: Piece):
        player, pt = captured_piece.player, captured_piece.piece_type
        info, start_idx = PIECE_SURVIVAL_VEC_INFO[pt], PIECE_SURVIVAL_VEC_INFO[pt]['start_idx']
        if info['count'] > 1:
            for i in range(info['count']):
                if self.survival_vectors[player][start_idx + i] == 1.0:
                    self.survival_vectors[player][start_idx + i] = 0.0
                    break
        else: self.survival_vectors[player][start_idx] = 0.0

    def _get_max_capture_value_for(self, player_id: int) -> float:
        max_val, opp_id = 0.0, -player_id
        valid_actions = np.where(self.action_masks(player_id))[0]
        for action in valid_actions:
            if action >= REVEAL_ACTIONS_COUNT:
                _, to_sq = self.action_to_coords[action]
                defender = self.board[POS_TO_SQ[to_sq]]
                if defender is not None and defender.player == opp_id:
                    val = PIECE_VALUES[defender.piece_type]
                    if val > max_val: max_val = val
        return max_val

    def _precompute_ray_attacks(self):
        rays = np.zeros((4, TOTAL_POSITIONS, TOTAL_POSITIONS), dtype=bool)
        for sq in range(TOTAL_POSITIONS):
            r, c = SQ_TO_POS[sq]
            for i in range(r - 1, -1, -1): rays[0, sq, POS_TO_SQ[(i, c)]] = True
            for i in range(r + 1, 4):      rays[1, sq, POS_TO_SQ[(i, c)]] = True
            for i in range(c - 1, -1, -1): rays[2, sq, POS_TO_SQ[(r, i)]] = True
            for i in range(c + 1, 4):      rays[3, sq, POS_TO_SQ[(r, i)]] = True
        self.attack_tables['rays'] = rays

    def _precompute_action_mappings(self):
        idx = 0
        for sq in range(TOTAL_POSITIONS):
            pos = tuple(SQ_TO_POS[sq])
            self.action_to_coords[idx], self.coords_to_action[pos] = pos, idx
            idx += 1
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                f_pos = (r1, c1)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= r1 + dr < BOARD_ROWS and 0 <= c1 + dc < BOARD_COLS:
                        t_pos = (r1 + dr, c1 + dc)
                        self.action_to_coords[idx], self.coords_to_action[(f_pos, t_pos)] = (f_pos, t_pos), idx
                        idx += 1
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                f_pos = (r1, c1)
                for c2 in range(BOARD_COLS):
                    if abs(c1 - c2) > 1:
                        t_pos = (r1, c2)
                        if (f_pos, t_pos) not in self.coords_to_action:
                            self.action_to_coords[idx], self.coords_to_action[(f_pos, t_pos)] = (f_pos, t_pos), idx
                            idx += 1
                for r2 in range(BOARD_ROWS):
                    if abs(r1 - r2) > 1:
                        t_pos = (r2, c1)
                        if (f_pos, t_pos) not in self.coords_to_action:
                            self.action_to_coords[idx], self.coords_to_action[(f_pos, t_pos)] = (f_pos, t_pos), idx
                            idx += 1

    def _initialize_board(self):
        self._reset_internal_state()
        pieces = [Piece(pt, p) for pt, count in PIECE_MAX_COUNTS.items() for p in [1, -1] for _ in range(count)]
        rng = self.np_random if hasattr(self, 'np_random') and self.np_random else random
        rng.shuffle(pieces)
        for sq, piece in enumerate(pieces):
            self.board[sq] = piece
            self.empty_vector[sq] = False
            self.hidden_vector[sq] = True

    def apply_single_action(self, action_index):
        return self._internal_apply_action(action_index)