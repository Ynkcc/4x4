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
from torch.utils.tensorboard import SummaryWriter

from constants import *
from environment import GameEnvironment
from net_model import Model
from actor import act

# file_writer.py
# 一个简化的文件写入类，用于将日志写入CSV文件
class FileWriter:
    def __init__(self, log_dir, xpid):
        self.log_dir = os.path.join(log_dir, xpid)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, 'logs.csv')
        self.file = open(self.log_path, 'w')
        self.fieldnames = None
        
    def log(self, to_log):
        if self.fieldnames is None:
            self.fieldnames = sorted(to_log.keys())
            self.file.write(','.join(self.fieldnames) + '\n')
        
        row = [str(to_log.get(k, '')) for k in self.fieldnames]
        self.file.write(','.join(row) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()

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

def learn(actor_model, learner_model, optimizer, batch, lock, writer, file_writer, total_frames):
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

        # 将更新后的模型参数同步给Actor模型
        actor_model.network.load_state_dict(learner_model.network.state_dict())
        
        # 记录日志
        current_frames = total_frames.value
        writer.add_scalar('train/loss', loss.item(), current_frames)
        writer.add_scalar('train/mean_target_value', target.mean().item(), current_frames)
        
        stats = {
            'frames': current_frames,
            'loss': loss.item(),
            'mean_target_value': target.mean().item(),
        }
        file_writer.log(stats)
        
        total_frames.value += BATCH_SIZE * UNROLL_LENGTH
        
        return stats

def train():
    """主训练函数"""
    if not ACTOR_DEVICE_CPU and TRAINING_DEVICE == 'cpu':
        raise ValueError("当训练设备为CPU时，Actor设备也必须为CPU。请检查配置。")
        
    if not ACTOR_DEVICE_CPU and not torch.cuda.is_available():
        raise ValueError("CUDA不可用。请使用 `ACTOR_DEVICE_CPU=True` 和 `TRAINING_DEVICE='cpu'` 参数进行CPU训练。")

    print("启动训练...")
    
    # 初始化模型
    device = torch.device('cuda:' + str(TRAINING_DEVICE) if TRAINING_DEVICE != 'cpu' else 'cpu')
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
    total_frames = ctx.Value('i', 0)

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
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    file_writer = FileWriter(os.path.join(SAVEDIR, 'logs'), XPID)

    for i in range(NUM_THREADS):
        thread = threading.Thread(
            target=lambda: learn(actor_models[0], learner_model, optimizer, get_batch(free_queue, full_queue, buffers, lock), lock, writer, file_writer, total_frames),
            name=f'learner-thread-{i}'
        )
        thread.start()
        threads.append(thread)
    
    # 主循环和监控
    try:
        os.makedirs(SAVEDIR, exist_ok=True)
        last_log_time = time.time()
        while total_frames.value < TOTAL_FRAMES:
            time.sleep(LOG_INTERVAL_SEC)
            current_frames = total_frames.value
            if time.time() - last_log_time > LOG_INTERVAL_SEC:
                print(f"训练步数: {current_frames}, 损失: {learner_model.network.loss}")
                last_log_time = time.time()

    except KeyboardInterrupt:
        print("训练中断")
    finally:
        writer.close()
        file_writer.close()
        for p in actor_processes:
            p.join()
        for t in threads:
            t.join()
        print("训练结束")

if __name__ == '__main__':
    train()