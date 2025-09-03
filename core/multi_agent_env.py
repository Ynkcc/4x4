# rllib_version_complete/core/multi_agent_env.py

import numpy as np
from typing import Dict, Any, Optional, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from core.environment import GameEnvironment

class RLLibMultiAgentEnv(MultiAgentEnv):
    """
    将单智能体 GameEnvironment 包装成与 RLlib 兼容的 MultiAgentEnv。
    这个环境模拟了两个玩家轮流下棋的场景。
    """
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        self._env = GameEnvironment()
        self.agent_ids = {"player1", "player2"}
        self._agent_to_player_map: Dict[str, int] = {}
        
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
        self._skip_env_checking = True

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境，随机决定哪个智能体先手，并返回第一个智能体的观察。"""
        self._env._internal_reset(seed=seed)
        
        if np.random.rand() < 0.5:
            self._agent_to_player_map = {"player1": 1, "player2": -1}
            first_agent = "player1"
        else:
            self._agent_to_player_map = {"player1": -1, "player2": 1}
            first_agent = "player2"
            
        self._env.current_player = self._agent_to_player_map[first_agent]
        
        obs = self._get_obs_for_current_player()
        return {first_agent: obs}, {}

    def step(self, action_dict: Dict[str, int]):
        """
        为一个智能体执行动作，然后轮到下一个智能体。
        返回下一个智能体的观察、奖励等信息 (重构版)。
        """
        agent_id = list(action_dict.keys())[0]
        action = action_dict[agent_id]
        
        if self._env.current_player != self._agent_to_player_map[agent_id]:
             raise ValueError(f"错误的智能体 {agent_id} 试图行动，但现在是玩家 {self._env.current_player} 的回合。")

        _, terminated, truncated, winner = self._env._internal_apply_action(action)
        
        # 初始化返回字典
        rewards: Dict[str, float] = {}
        terminateds: Dict[str, bool] = {"__all__": False}
        truncateds: Dict[str, bool] = {"__all__": False}
        infos: Dict[str, dict] = {}
        obs: Dict[str, Any] = {}

        if terminated or truncated:
            # 标记 episode 结束
            if terminated:
                terminateds["__all__"] = True
            if truncated:
                truncateds["__all__"] = True

            # 分配奖励
            if winner == 0: # 平局
                rewards = {"player1": 0.0, "player2": 0.0}
            else:
                for agent, player in self._agent_to_player_map.items():
                    rewards[agent] = 1.0 if player == winner else -1.0
            
            # 为双方都提供包含胜者信息的 info 字典
            infos["player1"] = {"winner": winner}
            infos["player2"] = {"winner": winner}
        else:
            # 游戏未结束，切换玩家
            self._env.current_player *= -1
            next_agent = self._get_agent_id_for_player(self._env.current_player)
            obs = {next_agent: self._get_obs_for_current_player()}

        # RLlib v2.x 期望的返回格式
        return obs, rewards, terminateds, truncateds, infos

    def _get_obs_for_current_player(self):
        """获取当前玩家的观察，并附加上合法的动作掩码。"""
        state = self._env.get_state()
        state["action_mask"] = self._env.action_masks()
        return state

    def _get_agent_id_for_player(self, player_id: int) -> str:
        """根据玩家ID（1或-1）找到对应的智能体ID（'player1'或'player2'）。"""
        for agent, player in self._agent_to_player_map.items():
            if player == player_id:
                return agent
        raise ValueError(f"找不到玩家ID {player_id} 对应的智能体。")