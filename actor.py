# src_code/actor.py
import os
import time
import numpy as np
import torch
import random # 导入random库以使用epsilon-greedy

from constants import *
from environment import GameEnvironment
from net_model import Model

def run_rollout(env, model, player_to_move):
    """
    执行一个蒙特卡洛Rollout到游戏结束，并返回最终奖励。
    在不确定性环境下，此函数在每次Rollout时都会随机化隐藏棋子。
    已添加 epsilon-贪心探索策略。
    """
    rollout_env = env.copy(shuffle_hidden=True)
    
    # 确保游戏可以开始
    action_mask = rollout_env.action_masks()
    if np.any(action_mask):
        first_action = np.random.choice(np.where(action_mask)[0])
        rollout_obs, _, terminated, truncated, info = rollout_env.step(first_action)
    else:
        return 0.0 # 游戏无法开始，返回0奖励

    while not terminated and not truncated:
        legal_actions = np.where(rollout_env.action_masks())[0]
        if len(legal_actions) == 0:
            # 无棋可走，提前结束
            terminated = True
            info['winner'] = -rollout_env.current_player
            break
        
        rollout_obs = rollout_env.get_state()
        
        # --- Epsilon-Greedy 探索策略 ---
        if random.random() < EXP_EPSILON:
            best_action = random.choice(legal_actions)
        else:
            action_values = model.predict_values(rollout_obs, legal_actions)
            best_action = legal_actions[np.argmax(action_values)]
        
        rollout_obs, _, terminated, truncated, info = rollout_env.step(best_action)
        
    final_reward = info.get('winner', 0)
    if final_reward == player_to_move:
        return 1.0
    elif final_reward == -player_to_move:
        return -1.0
    return 0.0

def act():
    """
    Actor进程，负责生成训练数据。
    警告：此函数已移除 Redis 依赖，不再能将数据推送到队列。
    """
    try:
        T = UNROLL_LENGTH
        env = GameEnvironment()
        
        # 自动选择设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(env.observation_space, device)
        obs, info = env.reset()
        episode_steps = 0
        
        while True:
            # 警告: 已移除模型更新逻辑，模型版本将不会更新。
            
            batch_data = {'board': [], 'scalars': [], 'target': [], 'episode_length': []}
            
            while len(batch_data['board']) < T:
                legal_actions = np.where(env.action_masks())[0]
                if len(legal_actions) == 0:
                    break
                
                action_rewards = []
                for action in legal_actions:
                    rewards = []
                    for _ in range(NUM_IMPERFECT_INFO_ROLLOUTS):
                        rewards.append(run_rollout(env, model, env.current_player))

                    action_rewards.append(np.mean(rewards))

                best_action = legal_actions[np.argmax(action_rewards)]
                target_value = np.max(action_rewards)
                
                batch_data['board'].append(obs['board'])
                batch_data['scalars'].append(obs['scalars'])
                batch_data['target'].append(target_value)
                
                obs, _, terminated, truncated, info = env.step(best_action)
                episode_steps += 1

                if terminated or truncated:
                    batch_data['episode_length'].append(episode_steps)
                    break
            
            if len(batch_data['board']) > 0:
                # 警告: 已移除将数据推送到队列的逻辑。数据将不会被保存或传输。
                print(f"Actor 生成了 {len(batch_data['board'])} 个数据点，但未将其保存。")
            
            if terminated or truncated:
                obs, info = env.reset()
                episode_steps = 0

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Actor进程发生错误: {str(e)}")

if __name__ == '__main__':
    act()