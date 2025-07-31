# train.py
import os
import time
# 新增: 导入必要的库
import random
import torch
import numpy as np

import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

# 确保 GameEnvironment 类被正确导入
from Game import GameEnvironment


class ProgressCallback(BaseCallback):
    """自定义回调，用于显示训练进度和统计信息"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.initial_timesteps = 0
    
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.initial_timesteps = self.model.num_timesteps
    
    def _on_step(self) -> bool:
        if len(self.locals.get('rewards', [])) > 0:
            for reward in self.locals['rewards']:
                self.episode_rewards.append(reward)
        
        if self.num_timesteps % 500 == 0:
            elapsed_time = time.time() - self.start_time
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                absolute_steps = self.initial_timesteps + self.num_timesteps
                print(f"Absolute Steps: {absolute_steps:,} | "
                      f"Session Steps: {self.num_timesteps:,} | "
                      f"Avg Reward (last 100): {avg_reward:.6f} | "
                      f"Time: {elapsed_time:.1f}s")
        
        return True

# --- 修改: make_env 工厂函数，使其只使用一个固定的基础种子 ---
def make_env(seed: int = 0):
    """
    创建单个环境实例的工厂函数。
    :param seed: 基础种子。
    """
    def _init():
        # 为每个环境设置完全相同的种子
        env = GameEnvironment()
        env.reset(seed=seed)
        return env
    return _init

def main():
    """
    训练函数。
    修改为：所有环境开局相同，但模型动作具有随机性。
    """
    # --- 新增: 仅为环境设定种子 ---
    # 这个种子将用于所有并行环境，以确保它们的开局完全一致
    env_seed = 42

    # --- 训练参数 ---
    total_timesteps = 15_000_000
    n_envs = 6
    learning_rate = 4e-4
    batch_size = 64
    n_steps = 2048

    # --- 路径设置 ---
    log_dir = "./banqi_ppo_logs/"
    model_save_path = os.path.join(log_dir, "banqi_ppo_model.zip")
    best_model_save_path = os.path.join(log_dir, "best_model")
    os.makedirs(log_dir, exist_ok=True)
    
    # --- 创建环境 ---
    print(f"创建 {n_envs} 个并行环境，所有环境开局将保持一致...")
    
    # 修改: 所有环境都使用同一个`make_env(seed=env_seed)`
    env = SubprocVecEnv([make_env(seed=env_seed) for _ in range(n_envs)])
    
    # 创建评估环境
    eval_env = GameEnvironment()
    # 为评估环境也设定相同的种子，确保评估的起点一致
    eval_env.reset(seed=env_seed)

    # --- 回调函数设置 ---
    progress_callback = ProgressCallback(verbose=1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=log_dir,
        name_prefix="rl_model_fixed_start",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=log_dir,
        eval_freq=max(15000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    # --- 模型定义与加载 ---
    print("初始化或加载模型...")
    if os.path.exists(model_save_path):
        print(f"发现已存在的模型，从 {model_save_path} 加载")
        # 修改: 不再为模型设置seed，以允许动作的随机性
        model = MaskablePPO.load(
            model_save_path,
            env=env,
            tensorboard_log=log_dir
        )
        model.lr_schedule = lambda _: learning_rate
        print(f"模型已加载，当前训练步数: {model.num_timesteps}")
        print(f"学习率已调整为: {learning_rate}")
    else:
        print("未找到已存在的模型，将创建新模型...")
        # 修改: 不再为模型设置seed，以允许动作的随机性
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=0.995,
            n_steps=n_steps,
            ent_coef=0.005,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=10,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            policy_kwargs=dict(
                net_arch=[512, 512, 256],
                activation_fn=nn.Tanh
            )
        )

    # --- 开始训练 ---
    print("开始训练...")
    remaining_steps = max(0, total_timesteps - model.num_timesteps)
    
    if remaining_steps <= 0:
        print("模型已达到目标训练步数，无需继续训练。")
        env.close()
        eval_env.close()
        return
    
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=[progress_callback, checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n训练被手动中断。")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        
    # --- 保存最终模型 ---
    try:
        model.save(os.path.join(log_dir, "final_model"))
        print(f"最终模型已保存。")
    except Exception as e:
        print(f"保存最终模型时出现错误: {e}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()