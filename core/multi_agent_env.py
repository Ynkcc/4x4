import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple

from core.environment import DarkChessEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv, AgentID

class RLLibMultiAgentEnv(MultiAgentEnv):
    """
    一个适配 Ray RLlib MultiAgentEnv 接口的暗棋环境封装器。
    """
    def __init__(self, env_config: Dict = None):
        super().__init__()
        self.env = DarkChessEnv()
        self.agent_ids = {"player1", "player2"}

        # 核心改动：为每个智能体定义独立的观察和动作空间
        # RLlib 需要一个从 AgentID 到其对应空间的字典
        self._agent_ids = set(self.agent_ids)
        self.observation_space = spaces.Dict(
            {
                agent_id: self.env.observation_space
                for agent_id in self.agent_ids
            }
        )
        self.action_space = spaces.Dict(
            {
                agent_id: self.env.action_space
                for agent_id in self.agent_ids
            }
        )

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, Dict]]:
        """
        重置环境并返回初始观察值。
        """
        obs = self.env.reset()
        # RLlib 需要一个从 AgentID 到观察值的字典
        agent_obs = {"player1": obs}
        return agent_obs, {}

    def step(self, action_dict: Dict[AgentID, int]) -> Tuple[
        Dict[AgentID, np.ndarray],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict]
    ]:
        """
        执行一个时间步。
        """
        # 从字典中提取当前玩家的动作
        player_id = f"player{self.env.current_player}"
        action = action_dict[player_id]

        obs, reward, terminated, truncated, info = self.env.step(action)

        # 确定下一个玩家
        next_player_id = f"player{self.env.current_player}"

        # RLlib 需要返回每个智能体对应的值的字典
        obs_dict = {next_player_id: obs}
        # 奖励只给做出动作的玩家
        rew_dict = {player_id: reward}
        # __all__ 键用于表示整个 episode 是否结束
        terminated_dict = {"__all__": terminated}
        truncated_dict = {"__all__": truncated}

        return obs_dict, rew_dict, terminated_dict, truncated_dict, {}

    def render(self, mode="human"):
        """
        渲染环境状态（可选）。
        """
        return self.env.render(mode)