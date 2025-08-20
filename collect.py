# src_code/collect.py
import os
import time
import numpy as np
import torch
import pickle
import random
from collections import deque
import multiprocessing as mp
from queue import Empty
import traceback

# 导入统一配置
from config import (
    ACTION_SPACE_SIZE,
    get_device,
    COLLECT_CONFIG
)

from environment import GameEnvironment
from net_model import Model

# 工作进程 (生产者)
class Worker(mp.Process):
    def __init__(self, worker_id, shared_model, data_queue, stop_event):
        super().__init__()
        self.worker_id = worker_id
        self.data_queue = data_queue
        self.stop_event = stop_event
        
        # 每个工作进程拥有自己的环境和模型实例
        # 模型实例的权重会指向共享内存
        self.env = GameEnvironment()
        self.device = get_device()
        self.policy_value_net = shared_model
        
        print(f"工作进程 #{self.worker_id} 已启动，使用设备: {self.device}")

    def predict_action(self, obs, legal_actions):
        """使用 Epsilon-Greedy 策略预测动作"""
        # 确保模型在评估模式
        self.policy_value_net.network.eval()
        return self.policy_value_net.predict(obs, legal_actions)

    def run(self):
        """工作进程的主循环：持续生产数据"""
        try:
            # 每个进程维护一个小的数据缓冲区
            local_buffer = []
            
            while not self.stop_event.is_set():
                # --- 1. 收集一局游戏的数据 ---
                game_steps = []
                obs, info = self.env.reset()
                terminated, truncated = False, False
                
                while not terminated and not truncated:
                    current_player = self.env.current_player
                    legal_actions = np.where(info.get('action_mask'))[0]

                    if len(legal_actions) == 0:
                        terminated = True
                        info['winner'] = -current_player
                        break
                    
                    chosen_action = self.predict_action(obs, legal_actions)

                    # 构造包含动作信息的输入向量
                    obs_with_action = obs['scalars'].copy()
                    action_slot_start = obs_with_action.shape[0] - ACTION_SPACE_SIZE
                    obs_with_action[action_slot_start:] = 0.0
                    obs_with_action[action_slot_start + chosen_action] = 1.0

                    # 存储(棋盘, 标量, 当前玩家)
                    game_steps.append({
                        'board': obs['board'],
                        'scalars': obs_with_action,
                        'player': current_player
                    })
                    
                    obs, _, terminated, truncated, info = self.env.step(chosen_action)
                
                # --- 2. 整理数据并添加到本地缓冲区 ---
                winner = info.get('winner', 0)
                # 根据游戏结果计算每个状态的目标价值
                for step in game_steps:
                    player_id = step['player']
                    target_value = 1.0 if player_id == winner else -1.0 if winner != 0 else 0.0
                    local_buffer.append((step['board'], step['scalars'], target_value))
                
                # --- 3. 当本地缓冲区达到一定大小时，发送到主队列 ---
                # 这个大小可以根据需要调整，以平衡通信开销和数据新鲜度
                if len(local_buffer) >= COLLECT_CONFIG.MAIN_PROCESS_BATCH_SIZE / COLLECT_CONFIG.NUM_THREADS:
                    try:
                        self.data_queue.put(local_buffer)
                        local_buffer = [] # 清空本地缓冲区
                    except Exception as e:
                        print(f"进程 #{self.worker_id} 发送数据失败: {e}")

        except KeyboardInterrupt:
            # 当接收到Ctrl+C时，子进程安静地退出
            pass
        except Exception as e:
            if not self.stop_event.is_set():
                print(f"进程 #{self.worker_id} 发生错误: {e}")
                traceback.print_exc()