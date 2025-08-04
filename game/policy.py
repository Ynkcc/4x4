# custom_policy.py - 为stable-baselines3定义一个带有残差块和双输入的自定义策略网络
import gymnasium as gym
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class ResidualBlock(nn.Module):
    """
    ResNet 的基本残差块。
    通过"跳跃连接"允许梯度更有效地传播，使得训练更深的网络成为可能。
    """
    def __init__(self, num_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 核心的残差连接
        out = F.relu(out)
        return out

class CustomNetwork(BaseFeaturesExtractor):
    """
    自定义的神经网络，作为特征提取器。
    它包含两个分支：
    1. CNN分支：处理棋盘的"图像"表示 (16, 4, 4)，使用残差块提取空间特征。
       - 16个通道：我方7种棋子 + 敌方7种棋子 + 暗棋 + 空位
    2. FC分支：处理得分等全局"标量"信息 (19,)。
       - 19个标量：我方得分 + 敌方得分 + 连续未吃子步数 + 我方存活棋子(8) + 敌方存活棋子(8)
    最后将两个分支的特征融合在一起。
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, num_res_blocks: int = 4, num_hidden_channels: int = 64):
        # features_dim 是融合后特征的总维度，这里我们动态计算它，并将其传递给策略和价值头
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        
        board_space = observation_space['board']
        scalars_space = observation_space['scalars']
        
        # --- CNN 分支定义 ---
        in_channels = board_space.shape[0]
        # 初始卷积层
        cnn_head = [
            nn.Conv2d(in_channels, num_hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden_channels),
            nn.ReLU()
        ]
        # 中间的残差块
        res_blocks = [ResidualBlock(num_hidden_channels) for _ in range(num_res_blocks)]
        self.cnn = nn.Sequential(*cnn_head, *res_blocks)
        
        # 计算CNN输出展平后的大小
        # 我们需要一个虚拟输入来动态计算这个大小
        with th.no_grad():
            dummy_input = th.zeros(1, *board_space.shape)
            cnn_output = self.cnn(dummy_input)
            # 排除批次维度，只计算特征维度
            self.cnn_flat_size = cnn_output.shape[1:].numel()

        # --- 全连接 (FC) 分支定义 ---
        self.fc = nn.Sequential(
            nn.Linear(scalars_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        fc_output_size = 64

        # --- 计算融合后的总特征维度 ---
        self._features_dim = self.cnn_flat_size + fc_output_size

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # 从字典中分离出两部分输入
        board_obs = observations['board']
        scalars_obs = observations['scalars']

        # 1. 通过CNN分支处理棋盘图像
        cnn_features = self.cnn(board_obs)
        # 展平CNN的输出，以便与FC分支拼接
        cnn_features_flat = cnn_features.reshape(cnn_features.size(0), -1)
        
        # 2. 通过FC分支处理标量信息
        fc_features = self.fc(scalars_obs)
        
        # 3. 融合两个分支的特征
        combined_features = th.cat([cnn_features_flat, fc_features], dim=1)
        
        return combined_features

class CustomActorCriticPolicy(MaskableActorCriticPolicy):
    """
    自定义的Actor-Critic策略。
    这个类告诉stable-baselines3使用我们上面定义的 `CustomNetwork` 作为特征提取器，
    而不是默认的CNN或MLP网络。
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        # 显式地将我们的自定义网络设置为特征提取器
        kwargs["features_extractor_class"] = CustomNetwork
        # 可以通过 policy_kwargs 传递参数给 CustomNetwork, 例如 {'num_res_blocks': 5}
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
