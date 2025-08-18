# src_code/learner.py
import os
import time
import numpy as np
import pickle
import random
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F

# 游戏相关常量
WINNING_SCORE = 60
MAX_CONSECUTIVE_MOVES_FOR_DRAW = 12
MAX_STEPS_PER_EPISODE = 100
ACTION_SPACE_SIZE = 112
HISTORY_WINDOW_SIZE = 15

# 网络相关常量
NETWORK_NUM_HIDDEN_CHANNELS = 64
NETWORK_NUM_RES_BLOCKS = 5
LSTM_HIDDEN_SIZE = 128
EXP_EPSILON = 0.01

# 训练相关常量
TRAINING_DEVICE = 'cpu'
SAVEDIR = 'saved_models'
LEARNING_RATE = 0.0001
MOMENTUM = 0
EPSILON = 1e-5
ALPHA = 0.99
MAX_GRAD_NORM = 40.0
from environment import GameEnvironment
from net_model import Model

# DMC 模式的配置
CONFIG = {
    'batch_size': 512,         # 训练的batch大小
    'epochs': 1,               # 每次更新的train_step数量 (DMC通常为1)
    'buffer_size': 10000,      # 经验池大小
    'game_batch_num': 1500,    # 训练更新的总次数
    'train_update_interval': 10 # 每次更新的间隔时间(秒)
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
        self.device = torch.device(f'cuda:{TRAINING_DEVICE}' if TRAINING_DEVICE != 'cpu' else 'cpu')
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
            os.makedirs(SAVEDIR, exist_ok=True)

        # 使用与原 learner.py 相同的 RMSprop 优化器
        self.optimizer = torch.optim.RMSprop(
            self.learner_model.network.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            eps=EPSILON,
            alpha=ALPHA
        )

    def update_network(self):
        """核心训练步骤：使用DMC数据更新价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        
        # 解包 DMC 数据: (board, scalars_with_action, target_value)
        board_batch = torch.FloatTensor(np.array([data[0] for data in mini_batch])).to(self.device)
        scalars_batch = torch.FloatTensor(np.array([data[1] for data in mini_batch])).to(self.device)
        target_batch = torch.FloatTensor(np.array([data[2] for data in mini_batch])).to(self.device)
        
        # 开始训练
        for i in range(self.epochs):
            self.learner_model.network.train()
            
            # 网络的输入是状态特征和one-hot编码的动作特征
            # 假设网络输出的是单一的价值预测值
            values = self.learner_model.network(board_batch, scalars_batch)
            values = values.squeeze(-1)

            # --- 计算损失函数 (仅价值损失 MSE) ---
            loss = F.mse_loss(values, target_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.learner_model.network.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")

    def run(self):
        """启动主训练循环"""
        model_path = os.path.join(SAVEDIR, 'model.pt')
        buffer_path = os.path.join(SAVEDIR, 'train_data_buffer.pkl')

        try:
            for i in range(self.game_batch_num):
                # --- 从文件加载数据 ---
                if os.path.exists(buffer_path):
                    try:
                        with open(buffer_path, 'rb') as f:
                            data_file = pickle.load(f)
                            self.data_buffer = data_file['data_buffer']
                        print(f"批次 {i+1}/{self.game_batch_num}: 成功载入 {len(self.data_buffer)} 条数据。")
                    except Exception as e:
                        print(f"载入数据失败: {e}, 等待下一轮...")
                        time.sleep(CONFIG['train_update_interval'])
                        continue
                else:
                    print("未找到数据文件，等待Actor生成数据...")
                    time.sleep(CONFIG['train_update_interval'])
                    continue

                # --- 执行训练 ---
                if len(self.data_buffer) > self.batch_size:
                    self.update_network()
                    # 保存最新模型
                    torch.save(self.learner_model.network.state_dict(), model_path)
                else:
                    print(f"数据不足 ({len(self.data_buffer)}/{self.batch_size})，跳过本轮训练。")

                time.sleep(CONFIG['train_update_interval'])

        except KeyboardInterrupt:
            print("\n\r训练中断。")
        finally:
            print(f"训练结束，保存最终模型到 {model_path}")
            torch.save(self.learner_model.network.state_dict(), model_path)

if __name__ == '__main__':
    model_file_path = os.path.join(SAVEDIR, 'model.pt')
    training_pipeline = TrainPipeline(init_model_path=model_file_path)
    training_pipeline.run()