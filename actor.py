# src_code/actor.py
import os
import time
import numpy as np
import torch
from torch import multiprocessing as mp

from constants import *
from environment import GameEnvironment
from net_model import Model

def run_rollout(env, model, player_to_move):
    """
    执行一个蒙特卡洛Rollout到游戏结束，并返回最终奖励。
    在不确定性环境下，此函数在每次Rollout时都会随机化隐藏棋子。
    """
    rollout_env = env.copy(shuffle_hidden=True)
    rollout_obs, _, terminated, truncated, info = rollout_env.step(np.random.choice(np.where(rollout_env.action_masks())[0]))

    while not terminated and not truncated:
        legal_actions = np.where(rollout_env.action_masks())[0]
        if len(legal_actions) == 0:
            # 无棋可走，提前结束
            terminated = True
            info['winner'] = -rollout_env.current_player
            break
        
        rollout_obs = rollout_env.get_state()
        action_values = model.predict_values(rollout_obs, legal_actions)
        best_action = legal_actions[np.argmax(action_values)]
        
        rollout_obs, _, terminated, truncated, info = rollout_env.step(best_action)
        
    final_reward = info.get('winner', 0)
    if final_reward == player_to_move:
        return 1.0
    elif final_reward == -player_to_move:
        return -1.0
    return 0.0

def act(actor_id, device, free_queue, full_queue, model, buffers):
    """
    Actor进程，负责生成训练数据。
    """
    try:
        T = UNROLL_LENGTH
        env = GameEnvironment()
        
        obs, info = env.reset()
        episode_steps = 0
        
        while True:
            # 收集一个回合的数据
            batch_data = {'board': [], 'scalars': [], 'target': [], 'episode_length': []}
            
            while len(batch_data['board']) < T:
                legal_actions = np.where(env.action_masks())[0]
                if len(legal_actions) == 0:
                    # 无合法动作，游戏结束
                    break
                
                # 为每个合法动作执行蒙特卡洛Rollout
                action_rewards = []
                for action in legal_actions:
                    rewards = []
                    # 每次Rollout都重新随机化隐藏棋子
                    for _ in range(NUM_IMPERFECT_INFO_ROLLOUTS):
                        rewards.append(run_rollout(env, model, env.current_player))

                    action_rewards.append(np.mean(rewards))

                # 根据Rollout结果选择价值最高的动作
                best_action = legal_actions[np.argmax(action_rewards)]
                target_value = np.max(action_rewards)
                
                # 存储数据
                batch_data['board'].append(obs['board'])
                batch_data['scalars'].append(obs['scalars'])
                batch_data['target'].append(target_value)
                
                # 执行选定的动作
                obs, _, terminated, truncated, info = env.step(best_action)
                episode_steps += 1

                if terminated or truncated:
                    batch_data['episode_length'].append(episode_steps)
                    break
            
            # 将收集到的数据放入缓冲区
            if len(batch_data['board']) > 0:
                index = free_queue.get()
                if index is None:
                    continue
                
                # 将数据转换为张量并放入缓冲区
                buffers['board'][index][:len(batch_data['board'])] = torch.from_numpy(np.stack(batch_data['board'])).to(device)
                buffers['scalars'][index][:len(batch_data['scalars'])] = torch.from_numpy(np.stack(batch_data['scalars'])).to(device)
                buffers['target'][index][:len(batch_data['target'])] = torch.from_numpy(np.stack(batch_data['target'])).to(device)
                
                # 存储回合长度。由于每个批次可能包含多个回合，我们只记录结束的回合长度，其余填充0
                episode_lengths_tensor = torch.zeros(UNROLL_LENGTH, dtype=torch.int32)
                end_index = len(batch_data['episode_length'])
                episode_lengths_tensor[:end_index] = torch.from_numpy(np.array(batch_data['episode_length'], dtype=np.int32))
                buffers['episode_length'][index] = episode_lengths_tensor.to(device)
                
                full_queue.put(index)
            
            # 游戏结束，重置环境
            if terminated or truncated:
                obs, info = env.reset()
                episode_steps = 0

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Actor进程 {actor_id} 发生错误: {str(e)}")