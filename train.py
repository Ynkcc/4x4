import os
import time
import threading
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from .constants import *
from .environment import GameEnvironment
from .net_model import Model

def get_batch(free_queue, full_queue, buffers, lock):
    """从缓冲区获取一个训练批次"""
    with lock:
        indices = [full_queue.get() for _ in range(BATCH_SIZE)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def compute_loss(values, targets):
    """计算均方误差损失"""
    return ((values.squeeze(-1) - targets)**2).mean()

def learn(model, learner_model, optimizer, batch, lock):
    """执行一步学习（优化）"""
    device = torch.device('cuda:'+str(TRAINING_DEVICE) if TRAINING_DEVICE != 'cpu' else 'cpu')
    
    board_obs = torch.flatten(batch['board'].to(device), 0, 1)
    scalars_obs = torch.flatten(batch['scalars'].to(device), 0, 1)
    target = torch.flatten(batch['target'].to(device), 0, 1)

    with lock:
        values = learner_model.network({'board': board_obs, 'scalars': scalars_obs}).squeeze(-1)
        loss = compute_loss(values, target)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(learner_model.network.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        # 将更新后的模型参数同步给所有Actor模型
        model.network.load_state_dict(learner_model.network.state_dict())
        
        return {'loss': loss.item()}

def run_rollout(env, model, player_to_move, rollout_depth=5):
    """
    执行一个蒙特卡洛Rollout到游戏结束，并返回最终奖励。
    在不确定性环境下，此函数在每次Rollout时都会随机化隐藏棋子。
    """
    rollout_env = env.copy(shuffle_hidden=True)
    rollout_obs, _, terminated, truncated, info = rollout_env.step(np.random.choice(rollout_env.action_space.n, p=np.array(rollout_env.action_masks())/np.sum(rollout_env.action_masks())))

    while not terminated and not truncated:
        legal_actions = np.where(rollout_env.action_masks())[0]
        if len(legal_actions) == 0:
            # 无棋可走，提前结束
            terminated = True
            info['winner'] = -rollout_env.current_player
            break
        
        action_values = model.predict_values(rollout_obs, legal_actions)
        best_action = legal_actions[np.argmax(action_values)]
        
        rollout_obs, _, terminated, truncated, info = rollout_env.step(best_action)
        
    final_reward = info['winner']
    if final_reward == player_to_move:
        return 1.0
    elif final_reward == -player_to_move:
        return -1.0
    return 0.0


def act(i, device, free_queue, full_queue, model, buffers):
    """
    Actor进程，负责生成训练数据。
    """
    try:
        T = UNROLL_LENGTH
        env = GameEnvironment()
        
        obs, info = env.reset()
        
        while True:
            # 收集一个回合的数据
            batch_data = {'board': [], 'scalars': [], 'target': []}
            
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
                        rollout_env = env.copy(shuffle_hidden=True)
                        _, _, terminated, truncated, info_rollout = rollout_env.step(action)
                        
                        while not terminated and not truncated:
                            rollout_legal_actions = np.where(rollout_env.action_masks())[0]
                            if len(rollout_legal_actions) == 0:
                                terminated = True
                                info_rollout['winner'] = -rollout_env.current_player
                                break
                                
                            rollout_obs = rollout_env.get_state()
                            rollout_action_values = model.predict_values(rollout_obs, rollout_legal_actions)
                            rollout_best_action = rollout_legal_actions[np.argmax(rollout_action_values)]
                            
                            _, _, terminated, truncated, info_rollout = rollout_env.step(rollout_best_action)

                        if info_rollout['winner'] == env.current_player:
                            rewards.append(1.0)
                        elif info_rollout['winner'] == -env.current_player:
                            rewards.append(-1.0)
                        else:
                            rewards.append(0.0)

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

                if terminated or truncated:
                    break
            
            # 将收集到的数据放入缓冲区
            if len(batch_data['board']) > 0:
                index = free_queue.get()
                if index is None:
                    continue
                
                for key in batch_data:
                    buffers[key][index][:len(batch_data[key])] = torch.from_numpy(np.stack(batch_data[key])).to(device)
                
                full_queue.put(index)
            
            # 游戏结束，重置环境
            if terminated or truncated:
                obs, info = env.reset()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Actor进程 {i} 发生错误: {str(e)}")

def train():
    """主训练函数"""
    if ACTOR_DEVICE_CPU or TRAINING_DEVICE == 'cpu':
        if not torch.cuda.is_available():
            raise ValueError("CUDA不可用。请使用 `--actor_device_cpu` 和 `--training_device cpu` 参数进行CPU训练。")

    print("启动训练...")
    
    # 初始化模型
    device = torch.device('cuda:' + TRAINING_DEVICE if TRAINING_DEVICE != 'cpu' else 'cpu')
    env = GameEnvironment()
    learner_model = Model(env.observation_space, device)
    
    # 创建优化器
    optimizer = torch.optim.RMSprop(
        learner_model.network.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        eps=EPSILON,
        alpha=ALPHA
    )
    
    # 初始化Actor模型
    actor_device_ids = [int(i) for i in GPU_DEVICES.split(',')] if not ACTOR_DEVICE_CPU else ['cpu']
    actor_models = {}
    for dev_id in actor_device_ids:
        actor_models[dev_id] = Model(env.observation_space, dev_id)
        actor_models[dev_id].share_memory()
        
    # 创建共享缓冲区
    buffers = {
        'board': [torch.empty((UNROLL_LENGTH, *env.observation_space['board'].shape), dtype=torch.float32).share_memory_() for _ in range(NUM_BUFFERS)],
        'scalars': [torch.empty((UNROLL_LENGTH, *env.observation_space['scalars'].shape), dtype=torch.float32).share_memory_() for _ in range(NUM_BUFFERS)],
        'target': [torch.empty((UNROLL_LENGTH,), dtype=torch.float32).share_memory_() for _ in range(NUM_BUFFERS)],
    }
    
    # 创建队列
    ctx = mp.get_context('spawn')
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    # 启动Actor进程
    actor_processes = []
    for dev_id in actor_device_ids:
        for i in range(NUM_ACTORS):
            actor = ctx.Process(
                target=act,
                args=(i, dev_id, free_queue, full_queue, actor_models[dev_id], buffers)
            )
            actor.start()
            actor_processes.append(actor)

    # 初始化缓冲区队列
    for i in range(NUM_BUFFERS):
        free_queue.put(i)

    # 启动Learner线程
    threads = []
    lock = threading.Lock()
    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=lambda: learn(actor_models[0], learner_model, optimizer, get_batch(free_queue, full_queue, buffers, lock), lock),
            name=f'learner-thread-{i}'
        )
        thread.start()
        threads.append(thread)
    
    # 主循环和监控
    try:
        last_save_time = time.time()
        while True:
            time.sleep(10)
            if time.time() - last_save_time > SAVE_INTERVAL_MIN * 60:
                print("保存模型...")
                torch.save(learner_model.network.state_dict(), f'{SAVEDIR}/model.pt')
                last_save_time = time.time()
    except KeyboardInterrupt:
        print("训练中断")
    finally:
        for p in actor_processes:
            p.join()
        for t in threads:
            t.join()
        print("训练结束")

if __name__ == '__main__':
    train()