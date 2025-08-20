# main_launcher.py
import os
import multiprocessing as mp
import time
import torch

from config import TRAINING_CONFIG, COLLECT_CONFIG, get_device
from net_model import Model
from environment import GameEnvironment
from train import TrainPipeline
from collect import Worker

def main():
    """主启动器，协调数据收集和模型训练"""
    try:
        mp.set_start_method('spawn', force=True)
        print("多进程启动方法已设置为 'spawn'")
    except RuntimeError:
        print("多进程启动方法已经被设置，跳过。")

    # 确保模型目录存在
    os.makedirs(TRAINING_CONFIG.SAVEDIR, exist_ok=True)
    
    device = get_device()
    print(f"主进程启动，使用设备: {device}")
    
    # 共享模型必须在CPU上以支持多进程间的参数共享
    shared_model_device = torch.device("cpu")
    print(f"共享模型使用设备: {shared_model_device} (CPU，用于多进程共享)")

    # --- 1. 初始化共享资源 ---
    # 模拟一个环境以获取观察空间
    env = GameEnvironment()
    
    # 创建一个将在所有进程中共享的模型 - 必须在CPU上
    shared_model = Model(env.observation_space, shared_model_device)
    # 尝试从磁盘加载最新模型
    model_path = TRAINING_CONFIG.MODEL_PATH
    if os.path.exists(model_path):
        try:
            shared_model.network.load_state_dict(torch.load(model_path, map_location=shared_model_device))
            print(f"成功从 {model_path} 加载初始模型。")
        except Exception as e:
            print(f"加载初始模型权重失败: {e}。将使用随机初始化的模型。")
    else:
        print("未找到初始模型，将使用随机初始化的模型。")

    # 将模型置于共享内存中
    shared_model.share_memory()

    # 创建进程间通信队列
    data_queue = mp.Queue(maxsize=COLLECT_CONFIG.QUEUE_MAX_SIZE)
    
    # 创建停止事件
    stop_event = mp.Event()

    # --- 2. 实例化并启动训练器和收集器 ---
    
    # 训练器
    trainer = TrainPipeline(shared_model, data_queue, stop_event)
    trainer_process = mp.Process(target=trainer.run, name="Trainer")
    
    # 收集器
    collectors = [
        Worker(i, shared_model, data_queue, stop_event) 
        for i in range(COLLECT_CONFIG.NUM_THREADS)
    ]

    try:
        print("--- 启动训练进程 ---")
        trainer_process.start()
        time.sleep(2) # 等待训练器初始化

        print(f"--- 启动 {len(collectors)} 个收集进程 ---")
        for c in collectors:
            c.start()
            
        # 等待所有进程结束
        trainer_process.join()
        for c in collectors:
            c.join()

    except KeyboardInterrupt:
        print("\n接收到中断信号，正在优雅地关闭所有进程...")
    except Exception as e:
        print(f"主进程发生严重错误: {e}")
    finally:
        print("\n正在关闭所有子进程...")
        stop_event.set() # 通知所有子进程停止

        # 等待并终止进程
        trainer_process.join(timeout=5)
        if trainer_process.is_alive():
            print("训练进程未能正常结束，将强制终止。")
            trainer_process.terminate()

        for worker in collectors:
            worker.join(timeout=5)
            if worker.is_alive():
                print(f"进程 #{worker.worker_id} 未能正常结束，将强制终止。")
                worker.terminate()
        
        print("所有进程已关闭。")

if __name__ == '__main__':
    main()