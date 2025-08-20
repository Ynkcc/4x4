# src_code/train.py
import os
import time
import numpy as np
import pickle
import random
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F

# 导入统一配置
from config import (
    WINNING_SCORE, MAX_CONSECUTIVE_MOVES_FOR_DRAW, MAX_STEPS_PER_EPISODE,
    ACTION_SPACE_SIZE, HISTORY_WINDOW_SIZE, NETWORK_NUM_HIDDEN_CHANNELS,
    NETWORK_NUM_RES_BLOCKS, LSTM_HIDDEN_SIZE, EXP_EPSILON,
    get_device, TRAINING_CONFIG
)

from environment import GameEnvironment
from net_model import Model

# DMC 模式的配置
CONFIG = {
    'batch_size': TRAINING_CONFIG.BATCH_SIZE,         # 训练的batch大小
    'epochs': TRAINING_CONFIG.EPOCHS,               # 每次更新的train_step数量 (DMC通常为1)
    'buffer_size': TRAINING_CONFIG.BUFFER_SIZE,      # 经验池大小
    'game_batch_num': TRAINING_CONFIG.GAME_BATCH_NUM,    # 训练更新的总次数
    'train_update_interval': TRAINING_CONFIG.TRAIN_UPDATE_INTERVAL # 每次更新的间隔时间(秒)
}

class TrainPipeline:
    def __init__(self, init_model_path=None):
        # 训练参数
        self.batch_size = CONFIG['batch_size']
        self.epochs = CONFIG['epochs']
        self.game_batch_num = CONFIG['game_batch_num']
        
        # 数据缓冲区
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)

        # 环境与模型
        self.device = get_device()
        print(f"训练设备: {self.device}")
        self.env = GameEnvironment()
        self.learner_model = Model(self.env.observation_space, self.device)
        
        # 加载已有模型
        if init_model_path and os.path.exists(init_model_path):
            print(f"检测到已存在模型，从 {init_model_path} 加载。")
            try:
                self.learner_model.network.load_state_dict(torch.load(init_model_path, map_location=self.device))
            except Exception as e:
                print(f"加载模型失败: {e}。将使用新模型。")
        else:
            print("未找到模型，创建新模型。")
            os.makedirs(TRAINING_CONFIG.SAVEDIR, exist_ok=True)

        # 使用与原 learner.py 相同的 RMSprop 优化器
        self.optimizer = torch.optim.RMSprop(
            self.learner_model.network.parameters(),
            lr=TRAINING_CONFIG.LEARNING_RATE,
            momentum=TRAINING_CONFIG.MOMENTUM,
            eps=TRAINING_CONFIG.EPSILON,
            alpha=TRAINING_CONFIG.ALPHA
        )

    def update_network(self):
        """核心训练步骤：使用DMC数据更新价值网络"""
        if len(self.data_buffer) < self.batch_size:
            print(f"数据不足 ({len(self.data_buffer)}/{self.batch_size})，跳过本轮训练。")
            return

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        
        # 解包 DMC 数据: (board, scalars_with_action, target_value)
        # 此时 board 和 scalars 还是 numpy 数组
        board_batch_np = np.array([data[0] for data in mini_batch])
        scalars_batch_np = np.array([data[1] for data in mini_batch])
        target_batch_np = np.array([data[2] for data in mini_batch])

        # 转换为 torch 张量并移动到指定设备
        board_batch = torch.from_numpy(board_batch_np).float().to(self.device)
        scalars_batch = torch.from_numpy(scalars_batch_np).float().to(self.device)
        target_batch = torch.from_numpy(target_batch_np).float().to(self.device)
        
        # 开始训练
        for i in range(self.epochs):
            self.learner_model.network.train()
            
            # --- 【核心修复】 ---
            # 将 board 和 scalars 数据打包成一个字典，以匹配模型的 forward 方法签名
            observations_batch = {
                'board': board_batch,
                'scalars': scalars_batch
            }
            
            # 模型的输入是包含状态特征和动作特征的观测字典
            values = self.learner_model.network(observations_batch)
            values = values.squeeze(-1) # 从 [B, 1] 压缩到 [B]

            # --- 计算损失函数 (仅价值损失 MSE) ---
            loss = F.mse_loss(values, target_batch)
            
            # --- 反向传播和优化 ---
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(self.learner_model.network.parameters(), TRAINING_CONFIG.MAX_GRAD_NORM)
            self.optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")

    def run(self):
        """启动主训练循环"""
        model_path = os.path.join(TRAINING_CONFIG.SAVEDIR, 'model.pt')
        buffer_path = TRAINING_CONFIG.TRAIN_DATA_BUFFER_PATH  # 使用配置中的路径

        try:
            for i in range(self.game_batch_num):
                print(f"--- 批次 {i+1}/{self.game_batch_num} ---")
                
                # --- 从文件加载数据 ---
                if os.path.exists(buffer_path):
                    try:
                        with open(buffer_path, 'rb') as f:
                            # 添加文件锁或重试机制可以提高文件读取的稳定性，但这里保持简单
                            data_file = pickle.load(f)
                            self.data_buffer = data_file['data_buffer']
                        print(f"成功载入 {len(self.data_buffer)} 条数据。")
                    except Exception as e:
                        print(f"载入数据失败: {e}, 等待下一轮...")
                        time.sleep(CONFIG['train_update_interval'])
                        continue
                else:
                    print("未找到数据文件，等待Actor生成数据...")
                    time.sleep(CONFIG['train_update_interval'])
                    continue

                # --- 执行训练 ---
                self.update_network()
                
                # 保存最新模型
                try:
                    torch.save(self.learner_model.network.state_dict(), model_path)
                    print(f"模型已保存至: {model_path}")
                except Exception as e:
                    print(f"模型保存失败: {e}")

                # 等待下一次更新
                print(f"等待 {CONFIG['train_update_interval']} 秒...")
                time.sleep(CONFIG['train_update_interval'])

        except KeyboardInterrupt:
            print("\n\r训练被手动中断。")
        finally:
            print(f"训练结束，保存最终模型到 {model_path}")
            torch.save(self.learner_model.network.state_dict(), model_path)

if __name__ == '__main__':
    # 确保SAVEDIR存在
    os.makedirs(TRAINING_CONFIG.SAVEDIR, exist_ok=True)
    
    model_file_path = os.path.join(TRAINING_CONFIG.SAVEDIR, 'model.pt')
    training_pipeline = TrainPipeline(init_model_path=model_file_path)
    training_pipeline.run()