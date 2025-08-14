# src_code/learner.py
import os
import time
import threading
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from constants import *
from environment import GameEnvironment
from net_model import Model
from actor import act # 从新文件导入 act 函数

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

def learn(actor_model, learner_model, optimizer, batch, lock):
    """执行一步学习（优化）"""
    device = torch.device('cpu' if TRAINING_DEVICE == 'cpu' else f'cuda:{TRAINING_DEVICE}')
    
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

        # 将更新后的模型参数同步给Actor模型
        actor_model.network.load_state_dict(learner_model.network.state_dict())
        
        return {'loss': loss.item()}

def train():
    """主训练函数"""
    if not ACTOR_DEVICE_CPU and TRAINING_DEVICE == 'cpu':
        raise ValueError("当训练设备为CPU时，Actor设备也必须为CPU。请检查配置。")
        
    if not ACTOR_DEVICE_CPU and not torch.cuda.is_available():
        raise ValueError("CUDA不可用。请使用 `ACTOR_DEVICE_CPU=True` 和 `TRAINING_DEVICE='cpu'` 参数进行CPU训练。")

    print("启动训练...")
    
    # 初始化模型
    device = torch.device('cpu' if TRAINING_DEVICE == 'cpu' else f'cuda:{TRAINING_DEVICE}')
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
    actor_device_ids = ['cpu'] if ACTOR_DEVICE_CPU else [int(i) for i in GPU_DEVICES.split(',')]
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
    
    def learner_thread():
        while True:
            try:
                batch = get_batch(free_queue, full_queue, buffers, lock)
                if batch is None:
                    continue
                result = learn(actor_models['cpu'], learner_model, optimizer, batch, lock)
                if result:
                    print(f"Loss: {result['loss']:.6f}")
            except Exception as e:
                print(f"Learner thread error: {e}")
                break
    
    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=learner_thread,
            name=f'learner-thread-{i}'
        )
        thread.start()
        threads.append(thread)
    
    # 主循环和监控
    try:
        os.makedirs(SAVEDIR, exist_ok=True)
        last_save_time = time.time()
        while True:
            time.sleep(10)
            if time.time() - last_save_time > SAVE_INTERVAL_MIN * 60:
                print("保存模型...")
                torch.save(learner_model.network.state_dict(), os.path.join(SAVEDIR, 'model.pt'))
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