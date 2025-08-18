# src_code/learner.py
import os
import time
import numpy as np

import torch
from torch import nn
from constants import *
from environment import GameEnvironment
from net_model import Model

def compute_loss(values, targets):
    """计算均方误差损失"""
    return ((values.squeeze(-1) - targets)**2).mean()

def check_batch_consistency(batch):
    """
    检查批次数据的形状和类型，确保输入正确。
    """
    if not isinstance(batch, dict) or 'board' not in batch or 'scalars' not in batch or 'target' not in batch:
        raise ValueError("批次数据格式不正确，应为包含 'board', 'scalars', 'target' 的字典。")
    if not all(isinstance(v, torch.Tensor) for v in batch.values()):
        raise ValueError("批次数据中的值必须是torch.Tensor。")

def learn(learner_model, optimizer, batch):
    """执行一步学习（优化）"""
    device = learner_model.device
    
    check_batch_consistency(batch)
    
    board_obs = batch['board'].to(device)
    scalars_obs = batch['scalars'].to(device)
    target = batch['target'].to(device)
    
    learner_model.network.train()
    
    # 网络的输入是状态特征和动作特征的组合，在这里表现为board和包含动作编码的scalars
    values = learner_model.network({'board': board_obs, 'scalars': scalars_obs}).squeeze(-1)
    loss = compute_loss(values, target)
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(learner_model.network.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    learner_model.network.eval()
    
    return loss.item()

def train():
    """主训练函数"""
    # 检查硬件配置是否合理
    if not ACTOR_DEVICE_CPU and TRAINING_DEVICE == 'cpu':
        raise ValueError("当训练设备为CPU时，Actor设备也必须为CPU。请检查配置。")
        
    if not ACTOR_DEVICE_CPU and not torch.cuda.is_available():
        raise ValueError("CUDA不可用。请使用 `ACTOR_DEVICE_CPU=True` 和 `TRAINING_DEVICE='cpu'` 参数进行CPU训练。")

    print("启动训练...")
    
    # 准备模型和优化器
    device = torch.device(f'cuda:{TRAINING_DEVICE}' if TRAINING_DEVICE != 'cpu' else 'cpu')
    env = GameEnvironment()

    # 模型加载或创建逻辑
    model_path = os.path.join(SAVEDIR, 'model.pt')
    if os.path.exists(model_path):
        print(f"检测到已存在模型，从 {model_path} 加载。")
        try:
            # 检查模型文件中保存的是完整模型还是state_dict
            loaded_data = torch.load(model_path, map_location=device, weights_only=False)
            
            if isinstance(loaded_data, dict):
                # 如果是state_dict，正常加载
                learner_model = Model(env.observation_space, device)
                learner_model.network.load_state_dict(loaded_data)
            else:
                # 如果保存的是完整模型，直接使用加载的网络
                learner_model = Model(env.observation_space, device)
                learner_model.network = loaded_data.to(device)
        except Exception as e:
            raise ValueError(f"加载模型失败: {e}")
    else:
        print("未找到模型，创建新模型。")
        os.makedirs(SAVEDIR, exist_ok=True)
        learner_model = Model(env.observation_space, device)
    
    optimizer = torch.optim.RMSprop(
        learner_model.network.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        eps=EPSILON,
        alpha=ALPHA
    )
    
    # 警告: 由于移除了 Redis 相关代码，此训练循环无法获取实际数据。
    # 此处仅为演示训练流程，需自行实现数据加载逻辑。
    print("警告: 训练数据获取逻辑已移除。当前无法进行完整训练。")
    print("训练流程将在无法获取数据时停止。")

    total_frames = 0
    last_log_time = time.time()
    
    # 模拟训练主循环
    try:
        while total_frames < TOTAL_FRAMES:
            # 假设这里有一个 `get_batch()` 函数能够从缓冲区中获取数据
            # 该批次数据应包含 board、带有动作编码的 scalars 和蒙特卡洛目标值
            batch = None # 模拟无法获取数据
            if batch is None:
                time.sleep(1)
                continue
            
            # 执行一步学习
            loss = learn(learner_model, optimizer, batch)
            
            total_frames += BATCH_SIZE * UNROLL_LENGTH
            
            # 简化版日志输出
            if time.time() - last_log_time > LOG_INTERVAL_SEC:
                print(f"Frames: {total_frames}, Loss: {loss:.4f}")
                last_log_time = time.time()
        
    except KeyboardInterrupt:
        print("训练中断")
    finally:
        # 训练结束后保存模型
        print(f"训练结束，保存模型到 {model_path}")
        torch.save(learner_model.network.state_dict(), model_path)
        
if __name__ == '__main__':
    train()