# src_code/game/policy.py
import gymnasium as gym
import torch as th
import torch.nn as nn
from typing import Dict, Any

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
    def __init__(self, 
                 observation_space: gym.spaces.Dict, 
                 num_res_blocks: int = NETWORK_NUM_RES_BLOCKS, 
                 num_hidden_channels: int = NETWORK_NUM_HIDDEN_CHANNELS,
                 scalar_encoder_output_dim: int = SCALAR_ENCODER_OUTPUT_DIM):
        
        # 计算最终拼接后特征向量的总维度
        features_dim = num_hidden_channels + scalar_encoder_output_dim
        super(CustomNetwork, self).__init__(observation_space, features_dim=features_dim)
        
        board_space = observation_space['board']
        scalars_space = observation_space['scalars']
        
        # --- 1. CNN 分支 (处理棋盘) ---
        # 这行代码会自动获取堆叠后的通道数
        in_channels = board_space.shape[0]
        self.cnn_head = nn.Conv2d(in_channels, num_hidden_channels, kernel_size=3, padding=1, bias=False)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_hidden_channels) for _ in range(num_res_blocks)])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # --- 2. MLP 分支 (处理标量) ---
        # 这行代码会自动获取堆叠后的标量维度
        scalar_input_dim = scalars_space.shape[0]
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_input_dim, 256),
            nn.LayerNorm(256),
            ACTIVATION_FN(),
            nn.Linear(256, scalar_encoder_output_dim),
            ACTIVATION_FN()
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        board_obs = observations['board']
        scalars_obs = observations['scalars']

        # 1. 通过CNN处理棋盘
        cnn_features = self.cnn_head(board_obs)
        for block in self.res_blocks:
            cnn_features = block(cnn_features)
        
        # 2. 全局池化并展平
        pooled_features = self.global_pool(cnn_features)
        cnn_flat = th.flatten(pooled_features, 1)

        # 3. 通过MLP处理标量
        scalar_features = self.scalar_encoder(scalars_obs)
        
        # 4. 拼接两部分特征
        combined_features = th.cat([cnn_flat, scalar_features], dim=1)
        
        return combined_features

# ==============================================================================
# --- V6 结构保持不变 - 分离的Actor和Critic ---
# ==============================================================================
class CustomActorCriticPolicy(MaskableActorCriticPolicy):
    """
    自定义的Actor-Critic策略，为策略(Actor)和价值(Critic)网络使用独立的特征提取器。
    此实现利用了MaskableActorCriticPolicy基类的内置逻辑，只需在初始化时指定
    自定义的特征提取器并强制不共享即可。
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        *args: Any,
        **kwargs: Any,
    ):
        # 关键: 强制使用独立的特征提取器，实现Actor和Critic解耦
        kwargs["share_features_extractor"] = False
        
        # 调用父类构造函数，并传入我们自定义的特征提取器类。
        # 父类将为actor和critic网络分别创建独立的CustomNetwork实例。
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomNetwork,
            features_extractor_kwargs=dict(
                # 这里可以传递 CustomNetwork 初始化所需的参数
                num_res_blocks=NETWORK_NUM_RES_BLOCKS,
                num_hidden_channels=NETWORK_NUM_HIDDEN_CHANNELS,
                scalar_encoder_output_dim=SCALAR_ENCODER_OUTPUT_DIM
            ),
            *args,
            **kwargs,
        )