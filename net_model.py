import torch as th
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple

# 导入自定义常量
from .constants import (NETWORK_NUM_HIDDEN_CHANNELS, NETWORK_NUM_RES_BLOCKS, 
                        LSTM_HIDDEN_SIZE, ACTION_SPACE_SIZE, HISTORY_WINDOW_SIZE)

class ResidualBlock(nn.Module):
    """
    ResNet 的基本残差块。
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
        out += residual
        out = F.relu(out)
        return out

class CustomNetwork(nn.Module):
    """
    自定义的神经网络，作为价值预测的特征提取器。
    它包含三个分支：CNN(棋盘), MLP(标量), LSTM(历史动作)。
    """
    def __init__(self, board_shape, scalars_shape):
        super(CustomNetwork, self).__init__()
        
        # --- CNN 分支定义 (处理棋盘状态) ---
        in_channels = board_shape[0]
        cnn_head = [
            nn.Conv2d(in_channels, NETWORK_NUM_HIDDEN_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(NETWORK_NUM_HIDDEN_CHANNELS),
            nn.ReLU()
        ]
        res_blocks = [ResidualBlock(NETWORK_NUM_HIDDEN_CHANNELS) for _ in range(NETWORK_NUM_RES_BLOCKS)]
        self.cnn = nn.Sequential(*cnn_head, *res_blocks)
        
        # --- MLP 分支定义 (处理标量信息) ---
        # 减去历史动作向量的维度
        mlp_input_dim = scalars_shape[0] - (HISTORY_WINDOW_SIZE * ACTION_SPACE_SIZE)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # --- LSTM 分支定义 (处理历史动作) ---
        self.lstm = nn.LSTM(ACTION_SPACE_SIZE, LSTM_HIDDEN_SIZE, batch_first=True)
        
        # 计算CNN输出展平后的大小
        with th.no_grad():
            dummy_input = th.zeros(1, *board_shape)
            cnn_output = self.cnn(dummy_input)
            self.cnn_flat_size = cnn_output.shape[1:].numel()
        
        # --- 最终价值头 ---
        # 融合后的特征维度: CNN_flat + MLP_output + LSTM_output
        combined_features_dim = self.cnn_flat_size + 64 + LSTM_HIDDEN_SIZE
        self.value_head = nn.Sequential(
            nn.Linear(combined_features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # 分离输入
        board_obs = observations['board']
        scalars_obs = observations['scalars']
        
        # 提取历史动作向量和剩余标量
        history_vector = scalars_obs[:, -HISTORY_WINDOW_SIZE * ACTION_SPACE_SIZE:].view(-1, HISTORY_WINDOW_SIZE, ACTION_SPACE_SIZE)
        other_scalars = scalars_obs[:, :-HISTORY_WINDOW_SIZE * ACTION_SPACE_SIZE]

        # 1. CNN分支处理棋盘图像
        cnn_features = self.cnn(board_obs)
        cnn_features_flat = cnn_features.view(cnn_features.size(0), -1)
        
        # 2. MLP分支处理标量信息
        mlp_features = self.mlp(other_scalars)
        
        # 3. LSTM分支处理历史动作
        lstm_output, _ = self.lstm(history_vector)
        lstm_features = lstm_output[:, -1, :] # 取最后一个时间步的输出

        # 4. 融合所有特征
        combined_features = th.cat([cnn_features_flat, mlp_features, lstm_features], dim=1)
        
        # 5. 价值头预测
        value = self.value_head(combined_features)
        
        return value

class Model:
    """
    用于封装神经网络和提供预测接口的类。
    """
    def __init__(self, observation_space, device):
        self.device = device
        self.network = CustomNetwork(observation_space['board'].shape, observation_space['scalars'].shape).to(device)
        self.network.eval()
    
    def to(self, device):
        self.device = device
        self.network.to(device)

    def predict_values(self, obs: Dict[str, np.ndarray], legal_actions: np.ndarray) -> np.ndarray:
        """
        为所有合法动作预测价值。
        """
        with th.no_grad():
            board_obs = th.from_numpy(obs['board']).float().unsqueeze(0).to(self.device)
            scalars_obs = th.from_numpy(obs['scalars']).float().unsqueeze(0).to(self.device)
            
            # 为每个合法动作创建一个观察批次
            num_legal_actions = len(legal_actions)
            board_batch = board_obs.repeat(num_legal_actions, 1, 1, 1)
            scalars_batch = scalars_obs.repeat(num_legal_actions, 1)
            
            # 临时修改历史动作向量以模拟每个合法动作后的状态
            # 这是一种简化的处理方式，更严谨的做法是需要环境拷贝和rollout
            history_vector_batch = scalars_batch[:, -HISTORY_WINDOW_SIZE * ACTION_SPACE_SIZE:]
            for i, action in enumerate(legal_actions):
                # 将最后一个动作替换为当前合法动作
                history_vector_batch[i, -(ACTION_SPACE_SIZE - action)] = 1.0
                history_vector_batch[i, -ACTION_SPACE_SIZE:] = 0.0
                history_vector_batch[i, -ACTION_SPACE_SIZE + action] = 1.0
            
            obs_batch = {
                'board': board_batch,
                'scalars': th.cat([scalars_batch[:, :-HISTORY_WINDOW_SIZE * ACTION_SPACE_SIZE], history_vector_batch], dim=1)
            }
            
            values = self.network(obs_batch)
            return values.cpu().numpy().flatten()
    
    def predict(self, obs, legal_actions) -> int:
        """
        根据价值预测选择最佳动作。
        """
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # 预测所有合法动作的价值
        all_action_values = self.predict_values(obs, legal_actions)
        
        # 选择价值最高的动作
        best_action_index = np.argmax(all_action_values)
        return legal_actions[best_action_index]
    
    def share_memory(self):
        self.network.share_memory()