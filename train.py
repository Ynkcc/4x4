import os
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np
import torch.nn as nn

# 确保 GameEnvironment 类被正确导入
from Game import GameEnvironment


class ProgressCallback(BaseCallback):
    """自定义回调，用于显示训练进度和统计信息"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_training_start(self) -> None:
        self.start_time = time.time()
    
    def _on_step(self) -> bool:
        if len(self.locals.get('rewards', [])) > 0:
            for reward in self.locals['rewards']:
                self.episode_rewards.append(reward)
        
        # 每1000步打印一次统计信息
        if self.num_timesteps % 1000 == 0:
            elapsed_time = time.time() - self.start_time
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-100:])  # 最近100个episode的平均奖励
                print(f"Steps: {self.num_timesteps:,} | "
                      f"Avg Reward (last 100): {avg_reward:.4f} | "
                      f"Time: {elapsed_time:.1f}s")
        
        return True


def main():
    """
    基于numpy向量版本GameEnvironment的训练函数。
    使用MaskablePPO进行强化学习训练。
    """
    # --- 训练参数 ---
    total_timesteps = 3_000_000  # 总训练步数
    n_envs = 6  # 并行环境数量（适当降低以避免内存问题）
    learning_rate = 3e-4
    batch_size = 64
    n_steps = 2048  # 每次收集的步数

    # --- 路径设置 ---
    log_dir = "./banqi_numpy_ppo_logs/"
    model_save_path = os.path.join(log_dir, "banqi_numpy_ppo_model.zip")
    best_model_save_path = os.path.join(log_dir, "best_model")
    os.makedirs(log_dir, exist_ok=True)

    # --- 创建环境 ---
    print(f"创建 {n_envs} 个并行环境...")
    
    def make_env():
        """创建单个环境的工厂函数"""
        def _init():
            env = GameEnvironment()
            return env
        return _init
    
    # 使用SubprocVecEnv进行真正的多进程并行
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    
    # 创建评估环境（单个环境用于评估）
    eval_env = GameEnvironment()

    # --- 回调函数设置 ---
    # 1. 进度回调
    progress_callback = ProgressCallback(verbose=1)
    
    # 2. 定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000 // n_envs, 1),  # 每20000步保存一次
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,  # 节省磁盘空间
        save_vecnormalize=True,
    )

    # 3. 评估回调
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=log_dir,
        eval_freq=max(25000 // n_envs, 1),  # 每25000步评估一次
        n_eval_episodes=10,  # 减少评估episode数以加快速度
        deterministic=True,  # 评估时使用确定性策略
        render=False
    )
    
    # --- 模型定义与加载 ---
    print("初始化或加载模型...")
    if os.path.exists(model_save_path):
        print(f"发现已存在的模型，从 {model_save_path} 加载")
        # 加载现有模型
        model = MaskablePPO.load(
            model_save_path,
            env=env,
            tensorboard_log=log_dir
        )
        print(f"模型已加载，当前训练步数: {model.num_timesteps}")
    else:
        print("创建新的MaskablePPO模型")
        # 针对当前Game.py的优化参数
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=0.995,  # 稍微提高折扣因子
            n_steps=n_steps,
            ent_coef=0.005,  # 降低熵系数，促进收敛
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=10,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            policy_kwargs=dict(
                net_arch=[512, 512, 256],  # 更深的网络结构
                activation_fn=nn.Tanh  # 使用函数对象而非字符串
            )
        )

    # --- 开始训练 ---
    print("开始训练...")
    print(f"目标训练步数: {total_timesteps:,}")
    print(f"并行环境数: {n_envs}")
    print(f"每次更新收集步数: {n_steps}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_callback, checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False  # 继续从之前的步数开始
        )
    except KeyboardInterrupt:
        print("\n训练被手动中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        
    # --- 保存最终模型 ---
    try:
        model.save(model_save_path.replace('.zip', ''))
        print(f"最终模型已保存至: {model_save_path}")
        print(f"最佳模型保存在: {best_model_save_path}")
        print(f"TensorBoard日志保存在: {log_dir}")
        print("训练完成!")
    except Exception as e:
        print(f"保存模型时出现错误: {e}")
    
    # 清理环境
    env.close()
    if hasattr(eval_env, 'close'):
        eval_env.close()


if __name__ == "__main__":
    main()