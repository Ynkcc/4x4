# src_code/learner.py
import os
import time
import numpy as np
from tqdm import tqdm
import argparse

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from constants import *
from environment import GameEnvironment
from net_model import Model
from redis_utils import (get_redis_connection, get_buffer_data, push_to_free_queue, 
                         pop_from_full_queue, clear_redis_queues_and_buffers)
from model_utils import save_model_to_redis, get_latest_model_from_redis

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

def get_batch(redis_conn):
    """从 Redis 缓冲区获取一个训练批次"""
    indices = []
    # 尝试非阻塞地从队列中获取BATCH_SIZE个索引
    for _ in range(BATCH_SIZE):
        item = redis_conn.blpop(REDIS_FULL_QUEUE_KEY, timeout=1)
        if item:
            indices.append(int(item[1]))
        else:
            break
            
    if not indices:
        return None

    batch_list = [get_buffer_data(redis_conn, m) for m in indices]
    
    board_list = [torch.from_numpy(np.stack(data['board'])).float() for data in batch_list]
    scalars_list = [torch.from_numpy(np.stack(data['scalars'])).float() for data in batch_list]
    target_list = [torch.from_numpy(np.stack(data['target'])).float() for data in batch_list]
    episode_lengths_list = [l for data in batch_list for l in data['episode_length']]
    
    batch = {
        'board': torch.cat(board_list, dim=0),
        'scalars': torch.cat(scalars_list, dim=0),
        'target': torch.cat(target_list, dim=0),
        'episode_length': torch.from_numpy(np.array(episode_lengths_list, dtype=np.int32))
    }

    for m in indices:
        push_to_free_queue(redis_conn, m)
        
    return batch


def compute_loss(values, targets):
    """计算均方误差损失"""
    return ((values.squeeze(-1) - targets)**2).mean()

def learn(learner_model, optimizer, batch, writer, file_writer, total_frames):
    """执行一步学习（优化）"""
    device = learner_model.device
    
    board_obs = batch['board'].to(device)
    scalars_obs = batch['scalars'].to(device)
    target = batch['target'].to(device)
    
    episode_lengths = batch['episode_length'][batch['episode_length'] > 0]
    
    learner_model.network.train()
    
    values = learner_model.network({'board': board_obs, 'scalars': scalars_obs}).squeeze(-1)
    loss = compute_loss(values, target)
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(learner_model.network.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    learner_model.network.eval()
    
    current_frames = total_frames
    writer.add_scalar('train/loss', loss.item(), current_frames)
    writer.add_scalar('train/mean_target_value', target.mean().item(), current_frames)
    
    stats = {
        'frames': current_frames,
        'loss': loss.item(),
        'mean_target_value': target.mean().item(),
    }
    if episode_lengths.numel() > 0:
        mean_ep_len = episode_lengths.float().mean().item()
        writer.add_scalar('rollout/ep_len_mean', mean_ep_len, current_frames)
        stats['ep_len_mean'] = mean_ep_len
         
    file_writer.log(stats)
    
    return stats


def train():
    """主训练函数"""
    if not ACTOR_DEVICE_CPU and TRAINING_DEVICE == 'cpu':
        raise ValueError("当训练设备为CPU时，Actor设备也必须为CPU。请检查配置。")
        
    if not ACTOR_DEVICE_CPU and not torch.cuda.is_available():
        raise ValueError("CUDA不可用。请使用 `ACTOR_DEVICE_CPU=True` 和 `TRAINING_DEVICE='cpu'` 参数进行CPU训练。")

    print("启动训练...")
    
    redis_conn = get_redis_connection()
    clear_redis_queues_and_buffers(redis_conn)
    
    device = torch.device(f'cuda:{TRAINING_DEVICE}' if TRAINING_DEVICE != 'cpu' else 'cpu')
    env = GameEnvironment()
    learner_model = Model(env.observation_space, device)
    
    current_model_version = 0
    save_model_to_redis(learner_model, current_model_version)
    
    optimizer = torch.optim.RMSprop(
        learner_model.network.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        eps=EPSILON,
        alpha=ALPHA
    )
    
    for i in range(NUM_BUFFERS):
        push_to_free_queue(redis_conn, i)

    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    file_writer = FileWriter(os.path.join(SAVEDIR, 'logs'), XPID)

    total_frames = 0
    last_save_time = time.time()
    
    try:
        os.makedirs(SAVEDIR, exist_ok=True)
        with tqdm(total=TOTAL_FRAMES, desc="训练进度") as pbar:
            last_frames = 0
            while total_frames < TOTAL_FRAMES:
                batch = get_batch(redis_conn)
                if batch is None:
                    time.sleep(1)
                    continue
                
                learn(learner_model, optimizer, batch, writer, file_writer, total_frames)
                
                total_frames += BATCH_SIZE * UNROLL_LENGTH
                
                if time.time() - last_save_time > SAVE_INTERVAL_MIN * 60:
                    current_model_version += 1
                    save_model_to_redis(learner_model, current_model_version)
                    last_save_time = time.time()

                if total_frames - last_frames >= LOG_INTERVAL_FRAMES:
                    pbar.update(total_frames - last_frames)
                    last_frames = total_frames

            pbar.update(TOTAL_FRAMES - last_frames)

    except KeyboardInterrupt:
        print("训练中断")
    finally:
        writer.close()
        file_writer.close()
        print("训练结束")

if __name__ == '__main__':
    train()