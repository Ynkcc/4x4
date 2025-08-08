# training/simple_agent.py
import numpy as np
from typing import Dict, Tuple, Any

class SimpleAgent:
    """
    一个执行更智能策略的Agent，用于作为基准对手。
    策略：
    1. 如果存在吃子动作（不包括炮的隔山打牛），优先执行能获得最高分值的吃子动作。
    2. 如果没有吃子动作，则从所有合法的动作中随机选择一个。
    3. 如果没有可用动作，则抛出异常。
    """
    def __init__(self, action_to_coords: Dict, piece_values: Dict, piece_types: Any, reveal_actions_count: int, regular_move_actions_count: int):
        """
        初始化SimpleAgent。

        :param action_to_coords: 将动作索引映射到坐标的字典。
        :param piece_values: 包含每个棋子类型分值的字典。
        :param piece_types: PieceType枚举类，用于访问棋子类型。
        :param reveal_actions_count: 翻棋动作的数量。
        :param regular_move_actions_count: 普通移动动作的数量。
        """
        self.action_to_coords = action_to_coords
        self.piece_values = piece_values
        self.PieceType = piece_types
        self.REVEAL_ACTIONS_COUNT = reveal_actions_count
        # 普通移动和吃子动作的范围
        self.MOVE_ACTION_END = reveal_actions_count + regular_move_actions_count

    def predict(self, observation: Dict, action_masks: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        根据观察和动作掩码，选择一个最佳动作。

        :param observation: 当前环境的观察值字典，包含 'board' 和 'scalars'。
        :param action_masks: 一个布尔或整数数组，标记了所有合法动作。
        :param deterministic: 是否使用确定性策略 (此agent不使用)。
        :return: 一个元组，包含选择的动作索引和None。
        """
        valid_actions = np.where(action_masks)[0]

        if len(valid_actions) == 0:
            raise ValueError("SimpleAgent: 接收到没有可用动作的状态，训练应停止。")
            
        capture_moves = {}
        board_state = observation['board']  # shape: (16, 4, 4)

        # 棋盘状态通道定义: 0-6是我方棋子, 7-13是敌方棋子
        opponent_piece_channels = range(7, 14) 
        # 将通道索引映射回棋子类型
        channel_to_piece_type = {
            7: self.PieceType.SOLDIER,
            8: self.PieceType.CANNON,
            9: self.PieceType.HORSE,
            10: self.PieceType.CHARIOT,
            11: self.PieceType.ELEPHANT,
            12: self.PieceType.ADVISOR,
            13: self.PieceType.GENERAL,
        }

        for action in valid_actions:
            # 只考虑在常规移动/吃子范围内的动作，排除翻棋和炮的攻击
            if self.REVEAL_ACTIONS_COUNT <= action < self.MOVE_ACTION_END:
                coords = self.action_to_coords.get(action)
                if coords and isinstance(coords, tuple) and len(coords) == 2:
                    # coords 是 ((from_row, from_col), (to_row, to_col))
                    from_pos, to_pos = coords
                    
                    # 检查目标位置是否有敌方棋子
                    for channel_idx in opponent_piece_channels:
                        if board_state[channel_idx, to_pos[0], to_pos[1]] == 1.0:
                            # 这是一个吃子动作
                            piece_type = channel_to_piece_type[channel_idx]
                            score = self.piece_values[piece_type]
                            capture_moves[action] = score
                            break  # 找到棋子后无需再检查该位置的其他通道

        # 决策逻辑
        if capture_moves:
            # 如果有吃子动作，选择得分最高的那个
            best_action = max(capture_moves, key=capture_moves.get)
            return int(best_action), None
        else:
            # 如果没有吃子动作，从所有合法动作中随机选择一个
            action = np.random.choice(valid_actions)
            return int(action), None