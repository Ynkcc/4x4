# custom_policy.py - V5 终极版: 采用 Pervasive FiLM 深度调制, GAP 和 Pre-activation 结构
import gymnasium as gym
import torch as th
import torch.nn as nn
from typing import Dict, Tuple, List

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from utils.constants import NETWORK_FEATURES_DIM, NETWORK_NUM_RES_BLOCKS, NETWORK_NUM_HIDDEN_CHANNELS

# --- 激活函数选择 ---
# 使用 SiLU (Swish) 替代 ReLU，因为它在许多现代架构中表现更好
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
        # film_params 的维度是 (batch_size, 2 * num_cnn_features)
        # 将其分割为 gamma 和 beta
        gamma, beta = th.chunk(film_params, 2, dim=1)
        
        # 调整 gamma 和 beta 的形状以匹配 cnn_features
        # cnn_features: (batch, channels, height, width)
        # gamma/beta: (batch, channels) -> (batch, channels, 1, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3).expand_as(cnn_features)
        beta = beta.unsqueeze(2).unsqueeze(3).expand_as(cnn_features)
        
        # 应用FiLM: modulated_features = gamma * cnn_features + beta
        return gamma * cnn_features + beta

# --- 核心网络块 ---

class ResidualAttentionFiLMBlock(nn.Module):
    """
    【V5 改进】采用 "Pre-activation" 结构，并在内部集成了FiLM调制的注意力残差块。
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
        self.film_layer = FiLMLayer() # 每个残差块都有自己的FiLM层

    def forward(self, x: th.Tensor, film_params: th.Tensor) -> th.Tensor:
        residual = x
        # Pre-activation
        out = self.activation(self.bn1(x))
        out = self.conv1(out)
        
        # 在残差块内部应用FiLM调制
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
    <--- V5 终极版 --->
    采用Pervasive FiLM深度调制，在每个残差块中都进行特征融合。
    使用全局平均池化(GAP)替代Flatten，增强模型鲁棒性。
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = NETWORK_FEATURES_DIM, num_res_blocks: int = NETWORK_NUM_RES_BLOCKS, num_hidden_channels: int = NETWORK_NUM_HIDDEN_CHANNELS):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        
        board_space = observation_space['board']
        scalars_space = observation_space['scalars']
        
        # --- 1. CNN 分支定义 (用于处理棋盘) ---
        in_channels = board_space.shape[0]
        # 初始卷积层，将输入通道数转换为隐藏通道数
        self.cnn_head = nn.Conv2d(in_channels, num_hidden_channels, kernel_size=3, padding=1, bias=False)
        
        # V5 改进: 改为 ModuleList 以便迭代并传入FiLM参数
        self.res_blocks = nn.ModuleList([ResidualAttentionFiLMBlock(num_hidden_channels) for _ in range(num_res_blocks)])
        
        # --- 2. 增强的FC分支 (用于处理标量并生成FiLM参数) ---
        scalar_input_dim = scalars_space.shape[0]
        # V5 改进: 为每一个残差块都生成一组独立的FiLM参数 (gamma + beta)
        film_params_dim_per_block = num_hidden_channels * 2
        total_film_params_dim = film_params_dim_per_block * num_res_blocks
        
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_input_dim, 256),
            nn.LayerNorm(256),
            ACTIVATION_FN(),
            nn.Linear(256, total_film_params_dim),
        )

        # --- 3. 最终的 Fusion Head ---
        # V5 改进: 使用全局平均池化层替代Flatten
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # V5 改进: fusion_head的输入维度现在只与隐藏通道数相关，不再与棋盘尺寸相关
        self.fusion_head = nn.Sequential(
            nn.Linear(num_hidden_channels, 512),
            nn.LayerNorm(512),
            ACTIVATION_FN(),
            nn.Linear(512, features_dim) # 最终输出，不加激活函数
        )
        
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        board_obs = observations['board']
        scalars_obs = observations['scalars']

        # 1. 标量路径: 一次性生成所有残差块所需的FiLM参数
        all_film_params = self.scalar_encoder(scalars_obs)
        # 将参数分割成每个块对应的部分
        film_params_per_block = th.chunk(all_film_params, len(self.res_blocks), dim=1)
        
        # 2. CNN路径:
        # a. 初始卷积
        cnn_features = self.cnn_head(board_obs)
        # b. V5 改进: 依次通过每个残差块，并传入对应的FiLM参数进行深度调制
        for i, block in enumerate(self.res_blocks):
            cnn_features = block(cnn_features, film_params_per_block[i])
        
        # 3. V5 改进: 使用全局平均池化并展平
        pooled_features = self.global_pool(cnn_features)
        features_flat = th.flatten(pooled_features, 1)
        
        # 4. 将最终处理过的特征送入Fusion Head
        final_features = self.fusion_head(features_flat)
        
        return final_features

class CustomActorCriticPolicy(MaskableActorCriticPolicy):
    """
    自定义的Actor-Critic策略。
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        kwargs["features_extractor_class"] = CustomNetwork
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )