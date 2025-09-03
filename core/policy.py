# src_code/game/policy.py
import gymnasium as gym
import torch as th
import torch.nn as nn
from typing import Dict, Any, cast, Callable

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from utils.constants import (
    NETWORK_NUM_RES_BLOCKS, 
    NETWORK_NUM_HIDDEN_CHANNELS,
    SCALAR_ENCODER_OUTPUT_DIM
)

# --- 激活函数选择 ---
ACTIVATION_FN = nn.SiLU

# --- 【V8 修改】核心网络块 ---

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

# --- 【V8 修改】主网络架构 ---

class CustomNetwork(BaseFeaturesExtractor):
    """
    【V8 版】自定义特征提取器。
    - CNN处理棋盘，MLP处理标量。
    - 两者的输出被拼接（Concatenate）后，送入Actor/Critic头。
    - 【V9 兼容】: 无需修改即可自动适应状态堆叠后的输入维度。
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int):
        super(CustomNetwork, self).__init__(observation_space, features_dim)

        # --- 棋盘处理分支 ---
        board_channels = observation_space["board"].shape[0]
        self.board_conv = nn.Sequential(
            nn.Conv2d(board_channels, NETWORK_NUM_HIDDEN_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(NETWORK_NUM_HIDDEN_CHANNELS),
            ACTIVATION_FN(),
            *[ResidualBlock(NETWORK_NUM_HIDDEN_CHANNELS) for _ in range(NETWORK_NUM_RES_BLOCKS)],
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
        )

        # --- 标量处理分支 ---
        scalar_dim = observation_space["scalars"].shape[0]
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, SCALAR_ENCODER_OUTPUT_DIM),
            ACTIVATION_FN(),
            nn.Linear(SCALAR_ENCODER_OUTPUT_DIM, SCALAR_ENCODER_OUTPUT_DIM),
            ACTIVATION_FN(),
        )

        # --- 计算总特征维度 ---
        board_out_dim = NETWORK_NUM_HIDDEN_CHANNELS
        scalar_out_dim = SCALAR_ENCODER_OUTPUT_DIM
        total_features = board_out_dim + scalar_out_dim

        # --- 最终特征融合 ---
        self.final_encoder = nn.Sequential(
            nn.Linear(total_features, features_dim),
            ACTIVATION_FN(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        board_obs = observations["board"]
        scalar_obs = observations["scalars"]

        # 处理棋盘
        board_features = self.board_conv(board_obs)

        # 处理标量
        scalar_features = self.scalar_encoder(scalar_obs)

        # 拼接特征
        combined = th.cat([board_features, scalar_features], dim=1)

        # 最终编码
        return self.final_encoder(combined)

# --- 【V8 修改】自定义策略 ---

class CustomMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """
    【V8 版】自定义Maskable策略。
    - 使用自定义特征提取器。
    - 支持动作掩码。
    """
    def __init__(self, *args, **kwargs):
        super(CustomMaskableActorCriticPolicy, self).__init__(
            *args,
            features_extractor_class=CustomNetwork,
            **kwargs
        )

# --- RLlib 版本的自定义网络 ---

import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.misc import SlimFC
from typing import List

class RLLibCustomNetwork(TorchModelV2, nn.Module):
    """
    RLlib 版本的自定义网络。
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 假设观察空间是 Dict 空间
        board_shape = obs_space["board"].shape
        scalar_shape = obs_space["scalars"].shape

        # 棋盘处理分支
        self.board_conv = nn.Sequential(
            nn.Conv2d(board_shape[0], NETWORK_NUM_HIDDEN_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(NETWORK_NUM_HIDDEN_CHANNELS),
            ACTIVATION_FN(),
            *[ResidualBlock(NETWORK_NUM_HIDDEN_CHANNELS) for _ in range(NETWORK_NUM_RES_BLOCKS)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # 标量处理分支
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_shape[0], SCALAR_ENCODER_OUTPUT_DIM),
            ACTIVATION_FN(),
            nn.Linear(SCALAR_ENCODER_OUTPUT_DIM, SCALAR_ENCODER_OUTPUT_DIM),
            ACTIVATION_FN(),
        )

        # 计算特征维度
        board_out_dim = NETWORK_NUM_HIDDEN_CHANNELS
        scalar_out_dim = SCALAR_ENCODER_OUTPUT_DIM
        features_dim = board_out_dim + scalar_out_dim

        # 策略头
        self.policy_head = SlimFC(features_dim, num_outputs)

        # 价值头
        self.value_head = SlimFC(features_dim, 1)

    def forward(self, input_dict, state, seq_lens):
        board_obs = input_dict["obs"]["board"]
        scalar_obs = input_dict["obs"]["scalars"]

        # 处理棋盘
        board_features = self.board_conv(board_obs)

        # 处理标量
        scalar_features = self.scalar_encoder(scalar_obs)

        # 拼接
        features = th.cat([board_features, scalar_features], dim=1)

        # 策略输出
        logits = self.policy_head(features)

        # 价值输出
        value = self.value_head(features).squeeze(-1)

        return logits, state

    def value_function(self):
        return self.value_head_output

    def custom_loss(self, policy_loss, vf_loss):
        return policy_loss + vf_loss
