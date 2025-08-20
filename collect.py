# actor.py
import os
import time
import numpy as np
import torch
import pickle
import random
from collections import deque

# 导入统一配置
from config import (
    WINNING_SCORE, MAX_CONSECUTIVE_MOVES_FOR_DRAW, MAX_STEPS_PER_EPISODE,
    ACTION_SPACE_SIZE, HISTORY_WINDOW_SIZE, NETWORK_NUM_HIDDEN_CHANNELS,
    NETWORK_NUM_RES_BLOCKS, LSTM_HIDDEN_SIZE, EXP_EPSILON,
    get_device, COLLECT_CONFIG
)

# 假设这些模块存在于您的项目中
from environment import GameEnvironment
from net_model import Model

# 配置字典，移除了MCTS相关参数，增加了epsilon
CONFIG = {
    'epsilon': COLLECT_CONFIG.EPSILON,  # Epsilon for epsilon-greedy exploration
    'buffer_size': COLLECT_CONFIG.BUFFER_SIZE, # 经验池大小
    'pytorch_model_path': COLLECT_CONFIG.PYTORCH_MODEL_PATH, # 模型路径
    'train_data_buffer_path': COLLECT_CONFIG.TRAIN_DATA_BUFFER_PATH # 训练数据缓存路径
}


class CollectPipeline:
    """
    数据收集流程主类，使用 Deep Monte Carlo (DMC) 方法。
    """

    def __init__(self, init_model_path=None):
        self.env = GameEnvironment()
        self.epsilon = CONFIG['epsilon']
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.init_model_path = init_model_path

        # 自动选择设备
        self.device = get_device()
        print(f"Using device: {self.device}")

    def load_model(self):
        """从主体加载最新模型"""
        model_path = CONFIG['pytorch_model_path']
        self.policy_value_net = Model(self.env.observation_space, self.device)

        if os.path.exists(model_path):
            try:
                self.policy_value_net.network.load_state_dict(torch.load(model_path, map_location=self.device))
                print('已加载最新模型')
            except Exception as e:
                print(f"加载模型权重失败: {e}, 将使用初始模型。")
        else:
            print('未找到模型文件，使用初始模型')
        
        # 将模型设置为评估模式
        self.policy_value_net.network.eval()

    def predict_action(self, obs, legal_actions):
        """
        使用 Epsilon-Greedy 策略预测动作。
        """
        # 使用Model类的predict方法，它已经实现了epsilon-greedy策略
        return self.policy_value_net.predict(obs, legal_actions)


    def collect_selfplay_data(self, n_games=1):
        """
        收集自我对弈的数据。
        此函数遵循 DMC 方法，记录完整的游戏轨迹，并使用最终结果作为目标值。
        """
        for i in range(n_games):
            self.load_model()  # 每局开始前加载最新模型

            # 为每个玩家存储一个单独的轨迹
            trajectories = {1: {'board': [], 'scalars': []}, -1: {'board': [], 'scalars': []}}
            
            # 启动新游戏
            obs, info = self.env.reset()
            terminated, truncated = False, False
            
            while not terminated and not truncated:
                current_player = self.env.current_player
                legal_actions = np.where(info.get('action_mask'))[0]

                if len(legal_actions) == 0:
                    terminated = True
                    info['winner'] = -current_player # 没有合法动作，对手获胜
                    break
                
                # 使用DMC的epsilon-greedy策略选择动作
                chosen_action = self.predict_action(obs, legal_actions)

                # --- 存储当前状态和动作（One-Hot编码） ---
                obs_with_action = obs['scalars'].copy()
                # 假设动作空间在 'scalars' 特征的末尾
                action_slot_start = obs_with_action.shape[0] - ACTION_SPACE_SIZE
                
                # 将动作部分清零，然后设置one-hot
                obs_with_action[action_slot_start:] = 0.0
                obs_with_action[action_slot_start + chosen_action] = 1.0

                trajectories[current_player]['board'].append(obs['board'])
                trajectories[current_player]['scalars'].append(obs_with_action)
                
                # 执行动作
                obs, _, terminated, truncated, info = self.env.step(chosen_action)
            
            # --- 游戏结束，整理训练数据 ---
            winner = info.get('winner', 0)
            play_data = []

            for player_id, trajectory in trajectories.items():
                if not trajectory['board']: # 如果该玩家没有走过棋，则跳过
                    continue

                if player_id == winner:
                    target_value = 1.0
                elif winner == 0:
                    target_value = 0.0
                else:
                    target_value = -1.0

                # 为该玩家轨迹中的每一步都赋予相同的目标值
                for board, scalars in zip(trajectory['board'], trajectory['scalars']):
                    play_data.append((board, scalars, target_value))

            self.episode_len = len(play_data)
            print(f"游戏结束。胜利者: {winner}, 收集到 {self.episode_len} 条训练数据。")

            # --- 数据存储逻辑 (与之前相同) ---
            if not play_data:
                print("本局未产生有效数据，跳过存储。")
                continue
            
            data_path = CONFIG['train_data_buffer_path']
            
            if os.path.exists(data_path):
                while True:
                    try:
                        with open(data_path, 'rb') as f:
                            saved_data = pickle.load(f)
                            self.data_buffer = saved_data['data_buffer']
                            self.iters = saved_data.get('iters', 0)
                        print('成功载入历史数据')
                        break
                    except Exception as e:
                        print(f"载入数据失败: {e}, 30秒后重试...")
                        time.sleep(30)
            
            self.data_buffer.extend(play_data)
            self.iters += 1
            
            try:
                with open(data_path, 'wb') as f:
                    pickle.dump({'data_buffer': self.data_buffer, 'iters': self.iters}, f)
                print("数据存储完成")
            except Exception as e:
                print(f"数据存储失败: {e}")

        return self.iters

    def run(self):
        """启动数据收集的无限循环"""
        try:
            while True:
                iters = self.collect_selfplay_data()
                print('批次 i: {}, 本局长度: {}'.format(
                    self.iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\r已退出')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Actor进程发生严重错误: {e}")


if __name__ == '__main__':
    pipeline = CollectPipeline(init_model_path=CONFIG['pytorch_model_path'])
    pipeline.run()