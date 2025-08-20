# src_code/train.py
import os
import time
import numpy as np
import pickle
import random
from collections import deque
from queue import Empty

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
    'epochs': TRAINING_CONFIG.EPOCHS,               # 每次更新的train_step数量
    'buffer_size': TRAINING_CONFIG.BUFFER_SIZE,      # 经验池大小
    'min_buffer_size': TRAINING_CONFIG.MIN_BUFFER_SIZE,  # 开始训练的最小经验池大小
    'training_frequency': TRAINING_CONFIG.TRAINING_FREQUENCY,  # 训练频率
    'samples_per_update': TRAINING_CONFIG.SAMPLES_PER_UPDATE,  # 每次训练的采样批次数
    'save_interval': 300 # 每隔多少秒保存一次模型
}

class TrainPipeline:
    def __init__(self, shared_model, data_queue, stop_event):
        # 训练参数
        self.batch_size = CONFIG['batch_size']
        self.epochs = CONFIG['epochs']
        self.min_buffer_size = CONFIG['min_buffer_size']
        self.training_frequency = CONFIG['training_frequency']
        self.samples_per_update = CONFIG['samples_per_update']
        
        # 数据缓冲区
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        
        # 训练计数器
        self.experiences_collected = 0
        self.last_training_count = 0
        self.training_count = 0
        self.total_training_time = 0.0
        
        # 共享资源
        self.data_queue = data_queue
        self.stop_event = stop_event

        # 环境与设备
        self.device = get_device()
        print(f"训练设备: {self.device}")
        
        # 共享模型引用（在CPU上）
        self.shared_model = shared_model
        
        # 创建训练器的本地模型副本（可以在GPU上训练）
        env = GameEnvironment()
        self.learner_model = Model(env.observation_space, self.device)
        
        # 从共享模型同步初始权重
        self.sync_from_shared_model()
        
        # 确保模型在正确的设备上
        self.learner_model.to(self.device)

        # 使用与原 learner.py 相同的 RMSprop 优化器
        self.optimizer = torch.optim.RMSprop(
            self.learner_model.network.parameters(),
            lr=TRAINING_CONFIG.LEARNING_RATE,
            momentum=TRAINING_CONFIG.MOMENTUM,
            eps=TRAINING_CONFIG.EPSILON,
            alpha=TRAINING_CONFIG.ALPHA
        )
        
        # 确保目录存在
        os.makedirs(TRAINING_CONFIG.SAVEDIR, exist_ok=True)
        print("训练器初始化完成。")

    def sync_from_shared_model(self):
        """从共享模型同步权重到本地训练模型"""
        try:
            self.learner_model.network.load_state_dict(self.shared_model.network.state_dict())
        except Exception as e:
            print(f"从共享模型同步权重失败: {e}")
    
    def sync_to_shared_model(self):
        """将训练后的权重同步到共享模型"""
        try:
            # 将本地模型权重复制到共享模型（需要在CPU上）
            cpu_state_dict = {}
            for key, value in self.learner_model.network.state_dict().items():
                cpu_state_dict[key] = value.cpu()
            self.shared_model.network.load_state_dict(cpu_state_dict)
        except Exception as e:
            print(f"同步权重到共享模型失败: {e}")

    def update_network(self):
        """核心训练步骤：使用经验池数据更新价值网络"""
        if len(self.data_buffer) < self.min_buffer_size:
            # 数据不足，不进行训练
            return False

        training_start_time = time.time()
        total_loss = 0.0
        
        # 进行多次采样和训练以提高GPU利用率
        for sample_idx in range(self.samples_per_update):
            # 从经验池中随机采样一个batch
            mini_batch = random.sample(self.data_buffer, min(self.batch_size, len(self.data_buffer)))
            
            # 解包数据: (board, scalars, target_value)
            board_batch_np = np.array([data[0] for data in mini_batch])
            scalars_batch_np = np.array([data[1] for data in mini_batch])
            target_batch_np = np.array([data[2] for data in mini_batch])

            # 转换为 torch 张量并移动到指定设备
            board_batch = torch.from_numpy(board_batch_np).float().to(self.device)
            scalars_batch = torch.from_numpy(scalars_batch_np).float().to(self.device)
            target_batch = torch.from_numpy(target_batch_np).float().to(self.device)
            
            # 开始训练
            for epoch in range(self.epochs):
                self.learner_model.network.train() # 切换到训练模式
                
                # 将 board 和 scalars 数据打包成一个字典
                observations_batch = {
                    'board': board_batch,
                    'scalars': scalars_batch
                }
                
                # 模型的输入是包含状态特征和动作特征的观测字典
                values = self.learner_model.network(observations_batch)
                values = values.squeeze(-1) # 从 [B, 1] 压缩到 [B]

                # 计算损失函数 (MSE)
                loss = F.mse_loss(values, target_batch)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                nn.utils.clip_grad_norm_(self.learner_model.network.parameters(), TRAINING_CONFIG.MAX_GRAD_NORM)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # 训练完成后，将更新后的权重同步到共享模型
        self.sync_to_shared_model()
        
        training_time = time.time() - training_start_time
        self.total_training_time += training_time
        self.training_count += 1
        
        avg_loss = total_loss / (self.samples_per_update * self.epochs)
        print(f"批量训练完成 - 平均Loss: {avg_loss:.4f}, 经验池大小: {len(self.data_buffer)}, "
              f"采样批次: {self.samples_per_update}, 训练轮次: {self.epochs}, "
              f"训练耗时: {training_time:.2f}s, 累计训练: {self.training_count}次")
        
        return True

    def run(self):
        """启动主训练循环"""
        model_path = TRAINING_CONFIG.MODEL_PATH
        last_save_time = time.time()

        try:
            while not self.stop_event.is_set():
                # --- 从队列加载数据 ---
                data_collected_this_round = 0
                try:
                    # 尝试从队列中获取一批数据，设置超时以避免永久阻塞
                    batch_data = self.data_queue.get(timeout=1.0)
                    data_collected_this_round = len(batch_data)
                    self.data_buffer.extend(batch_data)
                    self.experiences_collected += data_collected_this_round
                    
                except Empty:
                    # 队列为空是正常现象，继续循环
                    time.sleep(0.1) # 短暂休眠，避免CPU空转
                    continue
                except (IOError, EOFError):
                    # 队列可能已损坏或关闭
                    print("数据队列出现问题，训练终止。")
                    break

                # --- 基于收集的经验数量决定是否训练 ---
                experiences_since_last_training = self.experiences_collected - self.last_training_count
                
                if (len(self.data_buffer) >= self.min_buffer_size and 
                    experiences_since_last_training >= self.training_frequency):
                    
                    print(f"触发训练 - 新经验: {experiences_since_last_training}, 总经验池: {len(self.data_buffer)}")
                    
                    if self.update_network():
                        self.last_training_count = self.experiences_collected
                
                # --- 定时保存模型 ---
                current_time = time.time()
                if current_time - last_save_time > CONFIG['save_interval']:
                    try:
                        torch.save(self.learner_model.network.state_dict(), model_path)
                        print(f"模型已保存至: {model_path}")
                        last_save_time = current_time
                    except Exception as e:
                        print(f"模型保存失败: {e}")

        except KeyboardInterrupt:
            print("\n\r训练器被手动中断。")
        except Exception as e:
            print(f"训练器发生未知错误: {e}")
        finally:
            print(f"训练结束，保存最终模型到 {model_path}")
            torch.save(self.learner_model.network.state_dict(), model_path)