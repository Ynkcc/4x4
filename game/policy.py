# custom_policy.py - V6: 解耦 Actor-Critic 特征提取器
import gymnasium as gym
import torch as th
import torch.nn as nn
from typing import Dict, Tuple, List, Type, Any

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from utils.constants import NETWORK_FEATURES_DIM, NETWORK_NUM_RES_BLOCKS, NETWORK_NUM_HIDDEN_CHANNELS

# --- 激活函数选择 ---
ACTIVATION_FN = nn.SiLU

# --- 注意力与调制模块 ---

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) 模块，实现通道注意力。
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = in_channels // reduction_ratio
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            ACTIVATION_FN(),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) 层。
    """
    def __init__(self):
        super(FiLMLayer, self).__init__()

    def forward(self, cnn_features: th.Tensor, film_params: th.Tensor) -> th.Tensor:
        gamma, beta = th.chunk(film_params, 2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3).expand_as(cnn_features)
        beta = beta.unsqueeze(2).unsqueeze(3).expand_as(cnn_features)
        return gamma * cnn_features + beta

# --- 核心网络块 ---

class ResidualAttentionFiLMBlock(nn.Module):
    """
    采用 "Pre-activation" 结构，并在内部集成了FiLM调制的注意力残差块。
    结构: BN -> SiLU -> Conv -> FiLM -> BN -> SiLU -> Conv -> Attention -> +
    """
    def __init__(self, num_channels: int):
        super(ResidualAttentionFiLMBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.attention = SqueezeExcitation(num_channels)
        self.activation = ACTIVATION_FN()
        self.film_layer = FiLMLayer()

    def forward(self, x: th.Tensor, film_params: th.Tensor) -> th.Tensor:
        residual = x
        out = self.activation(self.bn1(x))
        out = self.conv1(out)
        out = self.film_layer(out, film_params)
        out = self.activation(self.bn2(out))
        out = self.conv2(out)
        out = self.attention(out)
        out += residual
        return out

# --- 主网络架构 ---

class CustomNetwork(BaseFeaturesExtractor):
    """
    自定义的神经网络，作为特征提取器。
    <--- V6 优化 --->
    - 采用Pervasive FiLM深度调制和GAP结构。
    - 【移除】了最终的 fusion_head，让策略/价值头直接从池化后的特征学习，增加灵活性。
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = NETWORK_FEATURES_DIM, num_res_blocks: int = NETWORK_NUM_RES_BLOCKS, num_hidden_channels: int = NETWORK_NUM_HIDDEN_CHANNELS):
        # MlpExtractor 会在此基础上构建策略和价值头
        super(CustomNetwork, self).__init__(observation_space, features_dim=num_hidden_channels)
        
        board_space = observation_space['board']
        scalars_space = observation_space['scalars']
        
        # --- 1. CNN 分支定义 (用于处理棋盘) ---
        in_channels = board_space.shape[0]
        self.cnn_head = nn.Conv2d(in_channels, num_hidden_channels, kernel_size=3, padding=1, bias=False)
        self.res_blocks = nn.ModuleList([ResidualAttentionFiLMBlock(num_hidden_channels) for _ in range(num_res_blocks)])
        
        # --- 2. 增强的FC分支 (用于处理标量并生成FiLM参数) ---
        scalar_input_dim = scalars_space.shape[0]
        film_params_dim_per_block = num_hidden_channels * 2
        total_film_params_dim = film_params_dim_per_block * num_res_blocks
        
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_input_dim, 256),
            nn.LayerNorm(256),
            ACTIVATION_FN(),
            nn.Linear(256, total_film_params_dim),
        )

        # --- 3. 全局平均池化 (GAP) ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        board_obs = observations['board']
        scalars_obs = observations['scalars']

        # 1. 从标量输入生成所有FiLM参数
        all_film_params = self.scalar_encoder(scalars_obs)
        film_params_per_block = th.chunk(all_film_params, len(self.res_blocks), dim=1)
        
        # 2. 通过CNN头部和带有FiLM调制的残差块处理棋盘观测
        cnn_features = self.cnn_head(board_obs)
        for i, block in enumerate(self.res_blocks):
            cnn_features = block(cnn_features, film_params_per_block[i])
        
        # 3. 全局池化并展平
        pooled_features = self.global_pool(cnn_features)
        features_flat = th.flatten(pooled_features, 1)
        
        # 4. 直接返回CNN部分的原始特征
        return features_flat

# ==============================================================================
# --- 【核心优化】V6 - 分离Actor和Critic的特征提取器 ---
# ==============================================================================
class CustomActorCriticPolicy(MaskableActorCriticPolicy):
    """
    【V6 核心修改】
    自定义的Actor-Critic策略，为策略(Actor)和价值(Critic)网络使用独立的特征提取器，
    以解决梯度冲突问题，专门用于提升价值函数的学习效果。
    
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
                features_dim=NETWORK_FEATURES_DIM, # 此参数为兼容性保留
                num_res_blocks=NETWORK_NUM_RES_BLOCKS,
                num_hidden_channels=NETWORK_NUM_HIDDEN_CHANNELS,
            ),
            *args,
            **kwargs,
        )