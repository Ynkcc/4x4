# training/simple_agent.py
import numpy as np
from typing import Dict, Tuple

class SimpleAgent:
    """
    一个执行简单策略的Agent，用于作为基准对手。
    策略：在所有合法的动作中随机选择一个。
    """
    def predict(self, observation: Dict, action_masks: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        根据动作掩码，随机选择一个有效的动作。

        :param observation: 当前环境的观察值 (此agent不使用)。
        :param action_masks: 一个布尔或整数数组，标记了所有合法动作。
        :param deterministic: 是否使用确定性策略 (此agent不使用)。
        :return: 一个元组，包含选择的动作索引和None（表示无附加状态）。
        """
        # 找到所有合法动作的索引
        valid_actions = np.where(action_masks)[0]
        
        # 如果不存在合法动作，这是一个异常情况，但为了健壮性返回一个默认值
        if len(valid_actions) == 0:
            # 在实际游戏中，这种情况通常意味着游戏已经结束，或者逻辑有误
            # 但作为独立的预测函数，我们返回一个不可能的动作索引-1
            return -1, None
            
        # 从合法动作中随机选择一个
        action = np.random.choice(valid_actions)
        
        return int(action), None