# src_code/actor.py
import os
import time
import numpy as np
import torch
import argparse

from constants import *
from environment import GameEnvironment
from net_model import Model
from redis_utils import (get_redis_connection, set_buffer_data, push_to_full_queue, 
                         pop_from_free_queue)
from model_utils import get_latest_model_from_redis

def run_rollout(env, model, player_to_move):
    """
    执行一个蒙特卡洛Rollout到游戏结束，并返回最终奖励。
    在不确定性环境下，此函数在每次Rollout时都会随机化隐藏棋子。
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
        action_values = model.predict_values(rollout_obs, legal_actions)
        best_action = legal_actions[np.argmax(action_values)]
        
        rollout_obs, _, terminated, truncated, info = rollout_env.step(best_action)
        
    final_reward = info.get('winner', 0)
    if final_reward == player_to_move:
        return 1.0
    elif final_reward == -player_to_move:
        return -1.0
    return 0.0

def act(actor_id, device):
    """
    Actor进程，负责生成训练数据。
    """
    try:
        T = UNROLL_LENGTH
        env = GameEnvironment()
        
        model = Model(env.observation_space, device)
        obs, info = env.reset()
        episode_steps = 0
        
        redis_conn = get_redis_connection()
        current_model_version = -1
        last_model_check_time = time.time()

        while True:
            # 定期检查模型更新，每隔10秒检查一次
            if time.time() - last_model_check_time > 10:
                latest_version_str = redis_conn.get(REDIS_MODEL_VERSION_KEY)
                if latest_version_str:
                    latest_version = int(latest_version_str.decode())
                    if latest_version > current_model_version:
                        print(f"Actor {actor_id} 检测到新模型版本 {latest_version}，正在下载...")
                        new_version = get_latest_model_from_redis(model)
                        if new_version:
                            current_model_version = new_version
                last_model_check_time = time.time()
            
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
                index = pop_from_free_queue(redis_conn, timeout=1)
                if index is None:
                    continue
                
                set_buffer_data(redis_conn, index, batch_data)
                push_to_full_queue(redis_conn, index)
            
            if terminated or truncated:
                obs, info = env.reset()
                episode_steps = 0

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Actor进程 {actor_id} 发生错误: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Actor Process for RL training")
    parser.add_argument("--actor_id", type=int, default=0, help="Unique ID for the actor")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., 'cpu', 'cuda:0')")
    args = parser.parse_args()
    
    act(args.actor_id, args.device)