# src_code/core/custom_model.py

import torch
from torch import nn
from typing import Dict, Any, Optional

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.columns import Columns
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Space
# 修复点: 移除了不存在的 RLModuleConfig 导入

class CustomDarkChessRLModule(TorchRLModule, ValueFunctionAPI):
    """
    一个为暗棋设计的自定义 RLModule，适配 RLlib 的新版 RLModule API。
    它处理由 "board" (图像) 和 "scalars" (向量) 组成的字典观察空间。
    """
    # 修复点: 更新 __init__ 签名并直接调用 super()
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        *,
        model_config: Dict,
        **kwargs,
    ):
        # 直接将关键字参数传递给父类构造函数
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            **kwargs
        )
        
        assert isinstance(observation_space, DictSpace)
        
        # --- 定义处理棋盘 (board) 的 CNN ---
        board_space = observation_space["board"]
        board_shape = board_space.shape
        self.board_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=board_shape[0],
                out_channels=32,
                kernel_size=3, stride=1, padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3, stride=1, padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *board_shape)
            cnn_output_size = self.board_cnn(dummy_input).shape[1]

        # --- 定义处理标量 (scalars) 的 MLP ---
        scalars_space = observation_space["scalars"]
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
        self.action_branch = nn.Linear(256, action_space.n)
        self.value_branch = nn.Linear(256, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def _get_shared_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        obs = batch[Columns.OBS]
        board_features = self.board_cnn(obs["board"])
        scalar_features = self.scalars_mlp(obs["scalars"])
        combined_features = torch.cat([board_features, scalar_features], dim=1)
        return self.combined_mlp(combined_features)

    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        with torch.no_grad():
            shared_features = self._get_shared_features(batch)
            action_logits = self.action_branch(shared_features)
            return {Columns.ACTION_DIST_INPUTS: action_logits}

    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        shared_features = self._get_shared_features(batch)
        action_logits = self.action_branch(shared_features)
        return {Columns.ACTION_DIST_INPUTS: action_logits}

    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        shared_features = self._get_shared_features(batch)
        action_logits = self.action_branch(shared_features)
        value_preds = torch.squeeze(self.value_branch(shared_features), -1)
        
        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: value_preds
        }
    
    def compute_values(self, batch: Dict[str, Any]) -> torch.Tensor:
        features = self._get_shared_features(batch)
        return torch.squeeze(self.value_branch(features), -1)

# --- 保留旧版 ModelV2 API 以供参考 ---
class CustomDarkChessModel(TorchModelV2, nn.Module):
    # ... (旧代码保持不变) ...
    pass