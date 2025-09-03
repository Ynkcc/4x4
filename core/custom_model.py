# src_code/core/custom_model.py

import torch
from torch import nn
import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import normc_initializer
from gymnasium.spaces import Box, Dict, Discrete

class CustomDarkChessModel(TorchModelV2, nn.Module):
    """
    自定义模型，用于处理由 "board" (图像) 和 "scalars" (向量) 组成的字典观察空间。
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 确保观察空间是 Dict 类型
        assert isinstance(obs_space, Dict)
        
        # --- 定义处理棋盘 (board) 的 CNN ---
        board_space = obs_space["board"]
        board_shape = board_space.shape
        self.board_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=board_shape[0], # (channels, height, width)
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算 CNN 输出的大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, *board_shape)
            cnn_output_size = self.board_cnn(dummy_input).shape[1]

        # --- 定义处理标量 (scalars) 的 MLP ---
        scalars_space = obs_space["scalars"]
        scalars_input_size = scalars_space.shape[0]
        self.scalars_mlp = nn.Sequential(
            nn.Linear(scalars_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # --- 融合后的主干网络 ---
        combined_input_size = cnn_output_size + 64
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_input_size, 256),
            nn.ReLU(),
        )

        # --- 策略头和价值头 ---
        self.action_branch = nn.Linear(256, num_outputs)
        self.value_branch = nn.Linear(256, 1)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


    def forward(self, input_dict, state, seq_lens):
        """
        前向传播函数。
        """
        obs = input_dict["obs"]
        
        # 分别处理 board 和 scalars
        board_features = self.board_cnn(obs["board"])
        scalar_features = self.scalars_mlp(obs["scalars"])
        
        # 拼接特征
        combined_features = torch.cat([board_features, scalar_features], dim=1)
        
        # 通过主干网络
        model_out = self.combined_mlp(combined_features)
        
        # 计算动作 logits 和价值
        self._last_features = model_out
        logits = self.action_branch(model_out)
        
        return logits, state

    def value_function(self):
        """
        返回当前观察的价值估计。
        """
        assert self._last_features is not None, "must call forward() first"
        return torch.squeeze(self.value_branch(self._last_features), -1)