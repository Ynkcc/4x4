# src_code/core/multi_agent_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Set, Any

# 修复点：导入正确的底层环境
from core.environment import DarkChessEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv, AgentID

class RLLibMultiAgentEnv(MultiAgentEnv):
    """
    一个适配 Ray RLlib MultiAgentEnv 接口的暗棋环境封装器。
    (已为新版 RLModule API 修复)
    """
    def __init__(self, env_config: Dict = None):
        """
        重构的构造函数，以正确定义包含 action_mask 的观察空间。
        """
        super().__init__()
        self.env = DarkChessEnv()
        self._agent_ids: Set[AgentID] = {"player_0", "player_1"}
        self._agent_to_player_map: Dict[AgentID, int] = {}

        # 核心修复点 1: 创建一个包含 'action_mask' 的新的、正确的单一智能体观察空间
        # 这是为了适配 RLModule API，它会自动从观察中寻找 'action_mask'
        single_agent_obs_space_with_mask = spaces.Dict({
            # 保留原始环境的观察空间结构
            **self.env.observation_space.spaces,
            # 添加动作掩码空间
            "action_mask": spaces.Box(0, 1, shape=(self.env.action_space.n,), dtype=np.float32)
        })

        # 核心修复点 2: 将 observation_space 和 action_space 定义为 gym.spaces.Dict
        # 这是 RLlib 多智能体环境的强制要求。
        self.observation_space = spaces.Dict(
            {agent_id: single_agent_obs_space_with_mask for agent_id in self._agent_ids}
        )
        self.action_space = spaces.Dict(
            {agent_id: self.env.action_space for agent_id in self._agent_ids}
        )

    def reset(self, *, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[AgentID, Any], Dict[AgentID, Any]]:
        """
        重置游戏环境，并返回符合新观察空间格式的初始状态。
        """
        obs_dict, info = self.env.reset(seed=seed, options=options)
        
        # 随机分配 agent_id 到玩家编号 (1 和 -1)
        agents = list(self._agent_ids)
        np.random.shuffle(agents)
        self._agent_to_player_map = {agents[0]: 1, agents[1]: -1}

        current_agent_id = self._get_agent_id(self.env.current_player)
        
        # 核心修复点 3: 将 action_mask 添加到观察字典中
        obs_with_mask = self._add_action_mask_to_obs(obs_dict, info['action_mask'])
        
        # RLlib 期望 reset 返回当前需要行动的智能体的观察
        agent_obs = {current_agent_id: obs_with_mask}
        agent_info = {current_agent_id: info}

        return agent_obs, agent_info

    def step(self, action_dict: Dict[AgentID, int]) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Any]
    ]:
        """
        根据传入的动作执行一步游戏，并返回符合新观察空间格式的状态。
        """
        # 在回合制游戏中，action_dict 通常只包含一个当前行动的智能体
        acting_agent_id = list(action_dict.keys())[0]
        action = action_dict[acting_agent_id]

        # 底层环境返回不带 mask 的 observation
        obs_dict, reward_shaping, terminated, truncated, info = self.env.step(action)

        next_agent_id = self._get_agent_id(self.env.current_player)
        
        # 核心修复点 4: 再次将 action_mask 添加到观察字典中
        obs_with_mask = self._add_action_mask_to_obs(obs_dict, info['action_mask'])

        # 新版RLlib要求：当游戏结束时，必须为所有参与的agent提供最后的观察值
        if terminated or truncated:
            # 游戏结束时，为两个智能体都提供最后的观察状态
            obs_return = {
                acting_agent_id: obs_with_mask,
                self._get_opponent_agent_id(acting_agent_id): obs_with_mask
            }
        else:
            # 游戏继续时，只为下一个行动的智能体提供观察
            obs_return = {next_agent_id: obs_with_mask}

        # 奖励只给当前行动的智能体
        rew_dict = {
            acting_agent_id: reward_shaping,
            self._get_opponent_agent_id(acting_agent_id): 0.0
        }

        terminated_dict = {"__all__": terminated}
        truncated_dict = {"__all__": truncated}
        
        # 为两个智能体都提供info
        info_dict = {
            acting_agent_id: info.copy(),
            self._get_opponent_agent_id(acting_agent_id): info.copy()
        }

        return obs_return, rew_dict, terminated_dict, truncated_dict, info_dict

    def _add_action_mask_to_obs(self, obs: dict, mask: np.ndarray) -> dict:
        """
        辅助函数，将 action_mask 合并到 observation 字典中。
        """
        obs_with_mask = obs.copy()
        obs_with_mask['action_mask'] = mask.astype(np.float32)
        return obs_with_mask

    def _get_agent_id(self, player_num: int) -> AgentID:
        for agent_id, p_num in self._agent_to_player_map.items():
            if p_num == player_num:
                return agent_id
        # 如果找不到，返回一个默认值以避免崩溃
        return "player_0"

    def _get_opponent_agent_id(self, agent_id: AgentID) -> AgentID:
        return "player_1" if agent_id == "player_0" else "player_0"