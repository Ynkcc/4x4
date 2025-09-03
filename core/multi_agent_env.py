import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Set

from core.environment import DarkChessEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv, AgentID

class RLLibMultiAgentEnv(MultiAgentEnv):
    """
    一个适配 Ray RLlib MultiAgentEnv 接口的暗棋环境封装器。
    """
    def __init__(self, env_config: Dict = None):
        super().__init__()
        self.env = DarkChessEnv()
        
        # RLlib 需要一个 agent ID 的集合
        self._agent_ids: Set[AgentID] = {"player1", "player2"}

        # 为 RLlib 定义观察和动作空间（所有智能体共享相同的空间）
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # 内部用于映射 agent_id 到游戏内的玩家编号 (1 或 -1)
        self._player_map = {
            "player1": 1,
            "player2": -1,
        }
        self._agent_to_player_map = {}


    def reset(self, *, seed=None, options=None) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, Dict]]:
        """
        重置环境并返回初始观察值。
        """
        obs, info = self.env.reset(seed=seed, options=options)
        
        # 随机决定哪个 agent 对应哪个 player
        agents = list(self._agent_ids)
        np.random.shuffle(agents)
        self._agent_to_player_map = {agents[0]: 1, agents[1]: -1}

        # 初始观察值只给当前行动的玩家
        current_player_id = self._get_agent_id(self.env.current_player)
        agent_obs = {current_player_id: obs}
        agent_info = {current_player_id: info}

        return agent_obs, agent_info

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
        # action_dict 在这种回合制游戏中应该只有一个 key
        player_id = list(action_dict.keys())[0]
        action = action_dict[player_id]

        # 执行环境的 step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 获取行动前后的玩家ID
        prev_player_id = player_id
        next_player_id = self._get_agent_id(self.env.current_player)

        # RLlib 需要返回每个智能体对应的值的字典
        obs_dict = {next_player_id: obs}
        rew_dict = {prev_player_id: reward, self._get_opponent_agent_id(prev_player_id): 0.0}

        # __all__ 键用于表示整个 episode 是否结束
        terminated_dict = {"__all__": terminated}
        truncated_dict = {"__all__": truncated}

        # 为两个玩家都提供 info
        info_dict = {
            prev_player_id: info,
            self._get_opponent_agent_id(prev_player_id): info
        }
        
        # 如果游戏结束，确保两个玩家都收到终止信号
        if terminated or truncated:
            terminated_dict["player1"] = True
            terminated_dict["player2"] = True
            truncated_dict["player1"] = True
            truncated_dict["player2"] = True


        return obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict

    def _get_agent_id(self, player: int) -> AgentID:
        """根据玩家编号 (1 或 -1) 获取 agent_id"""
        for agent_id, p_num in self._agent_to_player_map.items():
            if p_num == player:
                return agent_id
        raise ValueError(f"找不到玩家编号 {player} 对应的智能体。")

    def _get_opponent_agent_id(self, agent_id: AgentID) -> AgentID:
        """获取对手的 agent_id"""
        return "player2" if agent_id == "player1" else "player1"

    def render(self, mode="human"):
        """
        渲染环境状态（可选）。
        """
        return self.env.render(mode)