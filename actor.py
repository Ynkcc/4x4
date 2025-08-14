# src_code/actor.py
import os
import time
import numpy as np
import torch
import random 

from constants import *
from environment import GameEnvironment
from net_model import Model

def act():
    """
    Actor进程，负责生成训练数据。
    此函数已修改为遵循 DouZero 的 Deep Monte Carlo (DMC) 方法。
    它将执行完整的游戏局，并使用游戏的最终结果作为整个轨迹中所有状态-动作对的目标值。
    """
    try:
        T = UNROLL_LENGTH
        env = GameEnvironment()
        
        # 自动选择设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(env.observation_space, device)
        model.network.eval() # 确保actor模型处于评估模式
        
        while True:
            # 为每个玩家存储一个单独的轨迹
            trajectories = {1: {'board': [], 'scalars': []}, -1: {'board': [], 'scalars': []}}
            
            # 启动新游戏
            obs, info = env.reset()
            episode_steps = 0
            
            terminated, truncated = False, False
            while not terminated and not truncated:
                current_player = env.current_player
                legal_actions = np.where(info.get('action_mask'))[0]

                if len(legal_actions) == 0:
                    # 无棋可走，提前结束
                    terminated = True
                    info['winner'] = -current_player
                    break
                else:
                    # 使用新的predict方法，其中包含了epsilon-贪心策略
                    chosen_action = model.predict(obs, legal_actions)
                    
                # 将当前状态和选择的动作存储到当前玩家的轨迹中
                obs_with_action = obs['scalars'].copy()
                
                action_slot_start = obs_with_action.shape[0] - ACTION_SPACE_SIZE
                
                obs_with_action[action_slot_start:] = 0.0
                obs_with_action[action_slot_start + chosen_action] = 1.0

                trajectories[current_player]['board'].append(obs['board'])
                trajectories[current_player]['scalars'].append(obs_with_action)
                
                # 执行动作
                obs, _, terminated, truncated, info = env.step(chosen_action)
                episode_steps += 1
            
            # --- 游戏结束，应用蒙特卡洛目标值 ---
            winner = info.get('winner', 0)
            
            # 根据胜负结果为每个玩家分配奖励
            for player_id in trajectories.keys():
                trajectory = trajectories[player_id]
                
                if player_id == winner:
                    target_value = 1.0
                elif winner == 0:
                    target_value = 0.0
                else:
                    target_value = -1.0

                trajectory['target'] = [target_value] * len(trajectory['board'])
                
                # 在此可以实现将数据发送到队列的逻辑，例如：
                # data_queue.put(trajectory)
                if len(trajectory['board']) > 0:
                    print(f"Actor 生成了玩家 {player_id} 的完整游戏轨迹，长度为 {len(trajectory['board'])}。")
                    # data_queue.put(trajectory)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Actor进程发生错误: {str(e)}")

if __name__ == '__main__':
    act()