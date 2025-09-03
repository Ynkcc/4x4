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
        
        # 直接从内部环境中获取观察和动作空间
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
        # RLlib需要这个属性来跳过一些默认检查，因为我们的reset/step返回的是
        # 单个智能体的信息，而不是所有智能体的。
        self._skip_env_checking = True

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境，随机决定哪个智能体先手，并返回第一个智能体的观察。"""
        self._env._internal_reset(seed=seed)
        
        # 随机分配 agent 到 player (1 或 -1)
        if np.random.rand() < 0.5:
            self._agent_to_player_map = {"player1": 1, "player2": -1}
            first_agent = "player1"
        else:
            self._agent_to_player_map = {"player1": -1, "player2": 1}
            first_agent = "player2"
            
        self._env.current_player = self._agent_to_player_map[first_agent]
        
        obs = self._get_obs_for_current_player()
        # RLlib期望的返回格式：{agent_id: observation}, {agent_id: info}
        return {first_agent: obs}, {}

    def step(self, action_dict: Dict[str, int]):
        """
        为一个智能体执行动作，然后轮到下一个智能体。
        返回下一个智能体的观察、奖励等信息。
        """
        # action_dict 在我们的例子中总是只包含一个键值对
        agent_id = list(action_dict.keys())[0]
        action = action_dict[agent_id]
        
        # 安全检查：确保是正确的智能体在行动
        if self._env.current_player != self._agent_to_player_map[agent_id]:
             raise ValueError(f"错误的智能体 {agent_id} 试图行动，但现在是玩家 {self._env.current_player} 的回合。")

        # 调用内部环境的逻辑来应用动作
        _, terminated, truncated, winner = self._env._internal_apply_action(action)
        
        done = terminated or truncated
        
        # 初始化返回字典
        rewards: Dict[str, float] = {}
        # RLlib 使用 "__all__" 键来表示整个 episode 是否结束
        dones: Dict[str, bool] = {"__all__": False}
        infos: Dict[str, dict] = {}
        obs: Dict[str, Any] = {}

        if done:
            dones["__all__"] = True
            if winner == 0: # 平局
                rewards = {"player1": 0.0, "player2": 0.0}
            else:
                # 根据胜负结果分配奖励
                for agent, player in self._agent_to_player_map.items():
                    rewards[agent] = 1.0 if player == winner else -1.0
            
            # 在游戏结束时，为双方都提供包含胜者信息的info字典
            # 这对于回调函数中的数据分析很重要
            infos["player1"] = {"winner": winner}
            infos["player2"] = {"winner": winner}
            
            # 游戏结束时，观察字典为空
        else:
            # 游戏未结束，切换玩家
            self._env.current_player *= -1
            next_agent = self._get_agent_id_for_player(self._env.current_player)
            # 仅返回下一个行动的智能体的观察
            obs = {next_agent: self._get_obs_for_current_player()}

        # RLlib 期望的返回格式
        return obs, rewards, dones, dones, infos

    def _get_obs_for_current_player(self):
        """获取当前玩家的观察，并附加上合法的动作掩码。"""
        state = self._env.get_state()
        # RLlib的自定义模型会自动在 "obs" 键下查找观察
        # 我们需要在这里加入 "action_mask"
        state["action_mask"] = self._env.action_masks()
        return state

    def _get_agent_id_for_player(self, player_id: int) -> str:
        """根据玩家ID（1或-1）找到对应的智能体ID（'player1'或'player2'）。"""
        for agent, player in self._agent_to_player_map.items():
            if player == player_id:
                return agent
        raise ValueError(f"找不到玩家ID {player_id} 对应的智能体。")
