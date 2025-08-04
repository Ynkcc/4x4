# continue_training.py - 加载预训练模型并进行更长时间的深化训练
import os
from typing import Callable

from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO

# 导入环境和自定义策略，确保加载时能够识别
from Game import GameEnvironment
from custom_policy import CustomActorCriticPolicy

# --- 1. 定义学习率衰减函数 ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    创建一个线性衰减学习率的调度器。
    :param initial_value: 初始学习率 (例如 3e-4)
    :return: 一个函数，该函数接收一个从1.0到0.0的进度，并返回当前的学习率
    """
    def func(progress_remaining: float) -> float:
        """
        progress_remaining 会从 1.0 线性下降到 0.0
        """
        return progress_remaining * initial_value

    return func

if __name__ == "__main__":
    # --- 2. 路径和参数定义 ---
    model_dir = "cnn_curriculum_models"
    log_dir = "cnn_curriculum_logs/"

    # 要加载的现有模型
    load_model_path = os.path.join(model_dir, "final_model_cnn.zip")
    
    # 训练完成后，新模型的保存路径
    save_model_path = os.path.join(model_dir, "final_model_cnn_extended.zip")
    
    # 本次深化训练的总步数
    # 之前的训练是100万步，我们再增加500万步
    total_timesteps = 5_000_000 
    
    # 初始学习率，用于衰减
    initial_learning_rate = 3e-4

    # --- 3. 检查模型文件是否存在 ---
    if not os.path.exists(load_model_path):
        raise FileNotFoundError(f"错误：找不到预训练模型，请确保 '{load_model_path}' 文件存在！")

    print("\n" + "="*50)
    print("开始深化训练流程")
    print(f"加载模型: {load_model_path}")
    print(f"总训练步数: {total_timesteps:,}")
    print(f"初始学习率: {initial_learning_rate} (将线性衰减)")
    print("="*50 + "\n")

    # --- 4. 创建环境 ---
    # 我们将继续在最终的课程阶段（完整游戏）上训练
    env = make_vec_env(GameEnvironment, n_envs=8, env_kwargs={'curriculum_stage': 4})

    # --- 5. 加载模型并应用新配置 ---
    
    # 创建学习率调度器
    lr_schedule = linear_schedule(initial_learning_rate)

    # 加载模型
    # custom_objects 用于确保SB3知道如何处理我们的自定义策略
    # 这在某些复杂场景下是必须的，可以保证加载的健壮性
    model = MaskablePPO.load(
        load_model_path, 
        env=env,
        custom_objects={"policy_class": CustomActorCriticPolicy, "learning_rate": 0.0} # 临时设置lr，下面会覆盖
    )

    # 【重要】为加载的模型设置新的学习率调度器
    # 这是为已加载模型应用新学习率策略的标准方法
    model.learning_rate = lr_schedule
    
    # (可选) 如果你想重置模型的学习步数计数器，可以这样做
    model._total_timesteps = 0
    model.num_timesteps = 0


    # --- 6. 开始训练 ---
    # 我们为这次训练指定一个新的TensorBoard日志名称 "PPO_5_extended"
    # 这样在TensorBoard中可以清晰地看到这次训练的曲线
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name="PPO_5_extended", # 新的日志名称
        reset_num_timesteps=False # 关键：不要重置模型的内部步数，以便日志可以延续
    )

    # --- 7. 保存最终模型 ---
    print(f"\n深化训练完成，保存最终模型到 {save_model_path}...")
    model.save(save_model_path)

    # --- 8. 清理 ---
    env.close()

    print("\n" + "="*50)
    print("所有训练流程已完成！")
    print(f"最终强化模型已保存在: {save_model_path}")
    print("="*50)
