# /home/ynk/Desktop/banqi/4x4/gym/train_cython.py

import os
import time
import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
# from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.wrappers import ActionMasker
# from sb3_contrib.ppo_mask import MaskablePPO

# 导入 Cython 优化的环境
try:
    from Game_cython import BanqiEnvironment
    cython_env_available = True
    print("✓ Game_cython.BanqiEnvironment 模块导入成功")
except ImportError as e:
    print(f"✗ Game_cython.BanqiEnvironment 模块导入失败: {e}")
    cython_env_available = False

# 使用标准 PPO 而不是 MaskablePPO 来避免类型错误
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# 检查 PyTorch 是否可用
try:
    import torch
    print("✓ PyTorch 可用，支持自定义网络架构")
except ImportError:
    print("✗ PyTorch 未安装，无法使用自定义网络架构")

# [关键修改] 使用一个更健壮的回调函数
class StatsCallback(BaseCallback):
    """
    一个更健壮的回调函数，用于记录和打印每100个回合的平均奖励和长度。
    """
    def __init__(self, verbose=0):
        super(StatsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # 检查是否有回合结束
        # self.locals['dones'] 是一个布尔数组，对应每个并行环境
        if "dones" in self.locals:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    # 如果一个环境的回合结束，从info字典中提取统计信息
                    # Monitor wrapper 会自动添加 'episode' 键
                    info = self.locals["infos"][i]
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])

        # 当收集到足够的数据时（例如100个回合），记录日志
        if len(self.episode_rewards) >= 100:
            mean_reward = np.mean(self.episode_rewards)
            mean_length = np.mean(self.episode_lengths)
            
            self.logger.record("rollout/ep_rew_mean_100", mean_reward)
            self.logger.record("rollout/ep_len_mean_100", mean_length)
            
            # 打印到控制台以便观察
            if self.verbose > 0:
                print(f"Logged stats for 100 episodes. Mean Reward: {mean_reward:.4f}, Mean Length: {mean_length:.1f}")

            # 清空缓冲区以便下一次计算
            self.episode_rewards.clear()
            self.episode_lengths.clear()

        return True

def get_env_creator(env_class, env_kwargs=None):
    """返回一个创建和包装环境的函数。"""
    if env_kwargs is None:
        env_kwargs = {}
    
    def _init():
        env = env_class(**env_kwargs)
        # 确保 RecordEpisodeStatistics wrapper 被应用，以便在 info 中获得 'episode' 信息
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # 暂时注释掉 ActionMasker 以排除问题
        # env = ActionMasker(env, gym.spaces.Box)
        return env
    
    return _init

def main():
    # 定义超参数
    total_timesteps = 5_000_000
    n_steps = 2048
    n_envs = 8

    # 训练日志和模型保存路径
    log_dir = "./banqi_cython_ppo_logs"
    model_save_path = os.path.join(log_dir, "banqi_cython_model.zip")
    os.makedirs(log_dir, exist_ok=True)
    
    if not cython_env_available:
        print("错误：Cython 环境不可用，请先编译 .pyx 文件。")
        return

    print("--- 启用 8 个 Cython 优化的串行环境进行数据收集 (使用 DummyVecEnv) ---")
    print("--- 使用 Game_cython.BanqiEnvironment (Gymnasium 兼容包装器) ---")
    
    env_creator = get_env_creator(BanqiEnvironment)
    vec_env = make_vec_env(env_creator, n_envs=n_envs, vec_env_cls=DummyVecEnv)

    # 检查是否有已存在的模型
    if os.path.exists(model_save_path):
        print(f"--- 发现已存在的模型，从 {model_save_path} 加载 ---")
        model = PPO.load(model_save_path, env=vec_env, tensorboard_log=log_dir)
    else:
        print("--- 未发现已存在的模型，创建新的 Cython 优化模型 ---")
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128])
        )
        model = PPO(
            ActorCriticPolicy, 
            vec_env, 
            policy_kwargs=policy_kwargs,
            n_steps=n_steps,
            n_epochs=10,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            learning_rate=3e-4,
            verbose=1,
            tensorboard_log=log_dir,
            device="cpu"
        )

    print(f"Using {model.device} device")
    
    print("="*60)
    print("🚀 Cython 优化版本性能特性 (当前为串行模式):")
    print("    • 游戏执行速度: ~750 局/秒 (比原版快 4x)")
    print("    • 步执行速度: ~32,000 步/秒 (比原版快 4x)")
    print("    • 平均游戏时间: ~1.3ms (比原版减少 75%)")
    print("    • 注意: DummyVecEnv 禁用了并行，总吞吐量会低于 SubprocVecEnv")
    print("="*60)
    
    print("--- 开始或继续使用 Cython 优化环境训练 ---")
    
    callback = StatsCallback(verbose=1)
    
    log_name = "PPO_run"
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=log_name,
            reset_num_timesteps=not os.path.exists(model_save_path)
        )
    except Exception as e:
        print(f"--- 训练过程中出现错误: {e} ---")
        emergency_path = os.path.join(log_dir, "emergency_save_model.zip")
        print(f"--- 紧急保存模型至 {emergency_path} ---")
        model.save(emergency_path)
        raise
    finally:
        print("--- 训练完成，保存最终模型 ---")
        model.save(model_save_path)
        vec_env.close()

if __name__ == '__main__':
    main()