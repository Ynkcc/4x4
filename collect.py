# actor.py
import os
import time
import numpy as np
import torch
import pickle
import random
from collections import deque
import multiprocessing as mp
from queue import Empty # 导入队列的Empty异常
import traceback

# 导入统一配置
from config import (
    ACTION_SPACE_SIZE,
    get_device,
    COLLECT_CONFIG # 导入整个配置类
)

# 假设这些模块存在于您的项目中
from environment import GameEnvironment
from net_model import Model

# 工作进程 (生产者)
class Worker(mp.Process):
    def __init__(self, worker_id, shared_model, data_queue, stop_event):
        super().__init__()
        self.worker_id = worker_id
        self.policy_value_net = shared_model
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.env = GameEnvironment()
        self.device = get_device()
        print(f"工作进程 #{self.worker_id} 已启动，使用设备: {self.device}")

    def predict_action(self, obs, legal_actions):
        """使用 Epsilon-Greedy 策略预测动作"""
        return self.policy_value_net.predict(obs, legal_actions)

    def run(self):
        """工作进程的主循环：持续生产数据"""
        try:
            while not self.stop_event.is_set():
                try:
                    # --- 1. 收集一局游戏的数据 ---
                    game_steps = []
                    obs, info = self.env.reset()
                    terminated, truncated = False, False
                    
                    while not terminated and not truncated:
                        current_player = self.env.current_player
                        legal_actions = np.where(info.get('action_mask'))[0]

                        # 如果没有合法动作，判定当前玩家输棋
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
                    
                    # --- 2. 整理数据并发送到主进程 ---
                    winner = info.get('winner', 0)
                    play_data = []
                    # 根据游戏结果计算每个状态的目标价值
                    for step in game_steps:
                        player_id = step['player']
                        target_value = 1.0 if player_id == winner else -1.0 if winner != 0 else 0.0
                        play_data.append((step['board'], step['scalars'], target_value))
                    
                    if play_data:
                        self.data_queue.put(play_data)

                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"进程 #{self.worker_id} 发生错误: {e}")
                        traceback.print_exc()
                        time.sleep(5)
        except KeyboardInterrupt:
            # 当接收到Ctrl+C时，子进程安静地退出
            pass

# 主控流程 (消费者)
class CollectPipeline:
    def __init__(self, init_model_path=None):
        self.init_model_path = init_model_path
        self.device = get_device()
        print(f"主进程使用设备: {self.device}")

        self.env = GameEnvironment()
        self.policy_value_net = Model(self.env.observation_space, self.device)

        # 加载初始模型
        if self.init_model_path and os.path.exists(self.init_model_path):
            try:
                self.policy_value_net.network.load_state_dict(torch.load(self.init_model_path, map_location=self.device))
                print(f'已从 {self.init_model_path} 加载初始模型')
            except Exception as e:
                print(f"加载初始模型权重失败: {e}, 将使用随机初始化的模型。")
        else:
            print('未找到初始模型文件，使用随机初始化的模型')
        
        self.policy_value_net.network.eval()
        self.total_samples_collected = 0

    def _save_batch_to_file(self, batch_data):
        """将收集到的批次数据保存为独立的pkl文件"""
        try:
            data_dir = COLLECT_CONFIG.TRAIN_DATA_DIR
            timestamp = int(time.time() * 1000)
            filename = f"batch_{timestamp}_{random.randint(100, 999)}.pkl"
            filepath = os.path.join(data_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(batch_data, f)
            
            self.total_samples_collected += len(batch_data)
            print(f"\n主进程：批次数据成功写入到 {filename}。")
            print(f"主进程：本次收集 {len(batch_data)} 条样本。会话总计: {self.total_samples_collected} 条。")
        except Exception as e:
            print(f"\n主进程文件操作失败: {e}")

    def _update_shared_model(self):
        """从磁盘加载最新的模型权重以更新共享模型"""
        try:
            model_path = COLLECT_CONFIG.PYTORCH_MODEL_PATH
            if os.path.exists(model_path):
                self.policy_value_net.network.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"主进程：共享模型参数已从 {model_path} 更新。")
        except Exception as e:
            print(f"主进程加载模型权重失败: {e}")

    def run(self):
        """主运行循环，负责管理工作进程、收集数据和更新模型"""
        self.policy_value_net.network.share_memory()

        manager = mp.Manager()
        stop_event = manager.Event()
        data_queue = manager.Queue(maxsize=COLLECT_CONFIG.QUEUE_MAX_SIZE)

        # 启动工作进程
        workers = [
            Worker(i, self.policy_value_net, data_queue, stop_event)
            for i in range(COLLECT_CONFIG.NUM_THREADS)
        ]
        for w in workers:
            w.start()

        try:
            while not stop_event.is_set():
                # --- 1. 从队列中收集一个批次的数据 ---
                batch_data_buffer = []
                target_sample_size = COLLECT_CONFIG.MAIN_PROCESS_BATCH_SIZE
                games_collected_since_print = 0
                
                print(f"\n主进程：开始新批次收集，目标样本数: {target_sample_size}")
                
                while len(batch_data_buffer) < target_sample_size:
                    try:
                        game_data = data_queue.get(timeout=1.0)
                        batch_data_buffer.extend(game_data)
                        games_collected_since_print += 1

                        # 每收集10局游戏数据后刷新一次进度显示
                        if games_collected_since_print >= 10:
                            print(f"\r主进程：采集中... 当前批次样本数: {len(batch_data_buffer)}/{target_sample_size}", end="", flush=True)
                            games_collected_since_print = 0
                    except Empty:
                        if stop_event.is_set():
                            break # 如果收到停止信号，则退出收集循环
                        continue
                
                if stop_event.is_set() or not batch_data_buffer:
                    continue

                # --- 2. 保存收集到的数据 ---
                self._save_batch_to_file(batch_data_buffer)

                # --- 3. 更新工作进程使用的模型 ---
                self._update_shared_model()

        except KeyboardInterrupt:
            print('\n接收到中断信号，正在优雅地关闭所有工作进程...')
        except Exception as e:
            print(f"主进程发生严重错误: {e}")
        finally:
            print("\n正在关闭所有子进程...")
            stop_event.set()
            
            for worker in workers:
                worker.join(timeout=5)
                if worker.is_alive():
                    print(f"进程 #{worker.worker_id} 未能在5秒内结束，将强制终止。")
                    worker.terminate()
            print("所有工作进程已关闭。")


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("多进程启动方法已设置为 'spawn'")
    except RuntimeError:
        print("多进程启动方法已经被设置，跳过。")

    # 确保模型目录和数据目录存在
    model_dir = os.path.dirname(COLLECT_CONFIG.PYTORCH_MODEL_PATH)
    data_dir = COLLECT_CONFIG.TRAIN_DATA_DIR
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    pipeline = CollectPipeline(init_model_path=COLLECT_CONFIG.PYTORCH_MODEL_PATH)
    pipeline.run()