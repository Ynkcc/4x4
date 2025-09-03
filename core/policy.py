# src_code/core/policy.py
import gymnasium as gym
import torch as th
import torch.nn as nn
from typing import Dict, Any, cast, Callable, List

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from utils.constants import (
    NETWORK_NUM_RES_BLOCKS, 
    NETWORK_NUM_HIDDEN_CHANNELS,
    SCALAR_ENCODER_OUTPUT_DIM
)

# --- 激活函数选择 ---
ACTIVATION_FN = nn.SiLU

# --- 核心网络块 (RLlib版) ---

class ResidualBlock(nn.Module):
    """
    一个标准的残差块 (Pre-activation 结构)。
    结构: BN -> SiLU -> Conv -> BN -> SiLU -> Conv -> +
    """
    def __init__(self, num_channels: int):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.activation = ACTIVATION_FN()

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = x
        out = self.activation(self.bn1(x))
        out = self.conv1(out)
        out = self.activation(self.bn2(out))
        out = self.conv2(out)
        out += residual
        return out

# --- RLlib 版本的自定义网络 ---

class RLLibCustomNetwork(TorchModelV2, nn.Module):
    """
    RLlib 版本的自定义网络 (重构版)。
    - forward 方法只返回 logits。
    - value_function 方法独立计算价值。
    - 统一处理展平和非展平的观察输入。
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 内部变量，用于缓存 forward pass 计算出的特征
        self._features = None
        
        # 检查观察空间是否是原始的 Dict 空间
        if isinstance(obs_space, gym.spaces.Dict) and "board" in obs_space.spaces:
            board_shape = obs_space["board"].shape
            scalar_shape = obs_space["scalars"].shape
            self._is_obs_flattened = False
        else: # 否则，是展平的 Box 空间
            # 根据已知维度反推
            # (NUM_PIECE_TYPES * 2 + 2) * stack_size
            # (5 * 2 + 2) * 4 = 12 * 4 = 48
            board_channels = 48 
            board_size = 4 * 4
            self._board_flat_size = board_channels * board_size
            scalar_size = obs_space.shape[0] - self._board_flat_size
            
            board_shape = (board_channels, 4, 4)
            scalar_shape = (scalar_size,)
            self._is_obs_flattened = True

        # --- 棋盘处理分支 ---
        self.board_conv = nn.Sequential(
            nn.Conv2d(board_shape[0], NETWORK_NUM_HIDDEN_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(NETWORK_NUM_HIDDEN_CHANNELS),
            ACTIVATION_FN(),
            *[ResidualBlock(NETWORK_NUM_HIDDEN_CHANNELS) for _ in range(NETWORK_NUM_RES_BLOCKS)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # --- 标量处理分支 ---
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_shape[0], SCALAR_ENCODER_OUTPUT_DIM),
            ACTIVATION_FN(),
            nn.Linear(SCALAR_ENCODER_OUTPUT_DIM, SCALAR_ENCODER_OUTPUT_DIM),
            ACTIVATION_FN(),
        )

        # --- 计算总特征维度 ---
        board_out_dim = NETWORK_NUM_HIDDEN_CHANNELS
        scalar_out_dim = SCALAR_ENCODER_OUTPUT_DIM
        features_dim = board_out_dim + scalar_out_dim

        # --- 策略头 (Actor) ---
        self.policy_head = SlimFC(features_dim, num_outputs)

        # --- 价值头 (Critic) ---
        self.value_head = SlimFC(features_dim, 1)

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        计算动作 logits。
        同时，将计算出的共享特征缓存到 self._features 中，供 value_function 调用。
        """
        obs = input_dict["obs"]
        
        # 根据观察空间是展平还是字典来解析输入
        if self._is_obs_flattened:
            board_obs = obs[:, :self._board_flat_size].reshape(-1, 48, 4, 4)
            scalar_obs = obs[:, self._board_flat_size:]
        else:
            # RLlib 在送入模型前会将 Dict 观察值放入 "obs" 键下
            board_obs = obs["board"]
            scalar_obs = obs["scalars"]

        # --- 特征提取 ---
        board_features = self.board_conv(board_obs)
        scalar_features = self.scalar_encoder(scalar_obs)

        # 拼接特征并缓存，以供 value_function 使用
        self._features = th.cat([board_features, scalar_features], dim=1)

        # --- 策略输出 ---
        logits = self.policy_head(self._features)

        # 只返回 logits 和 state
        return logits, state

    def value_function(self) -> TensorType:
        """
        【新增】独立的价值函数计算方法。
        RLlib 会在需要计算价值时自动调用此方法。
        """
        assert self._features is not None, "must call forward() first"
        value = self.value_head(self._features).squeeze(-1)
        return value