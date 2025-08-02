# train_curriculum.py
import os
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from Game import GameEnvironment # 确保导入我们修改后的环境

def train_stage(stage_name, curriculum_stage, total_timesteps, load_from_model=None, save_to_model=None):
    """
    一个通用的函数，用于训练课程学习的单个阶段。

    :param stage_name: (str) 阶段的名称 (用于打印日志)。
    :param curriculum_stage: (int) 要传递给 GameEnvironment 的阶段编号。
    :param total_timesteps: (int) 此阶段要训练的总步数。
    :param load_from_model: (str, optional) 从哪个模型文件加载权重。如果是 None，则从头开始训练。
    :param save_to_model: (str, optional) 训练完成后，将模型保存到哪个文件。
    """
    print("\n" + "="*50)
    print(f"开始课程学习阶段: {stage_name} (Stage {curriculum_stage})")
    print(f"训练步数: {total_timesteps:,}")
    print("="*50 + "\n")

    # 1. 创建环境
    # 我们使用 make_vec_env 来创建多个并行的环境，以加速训练
    # env_kwargs 字典会将 curriculum_stage 参数传递给 GameEnvironment 的构造函数
    env = make_vec_env(GameEnvironment, n_envs=8, env_kwargs={'curriculum_stage': curriculum_stage})

    # 2. 创建或加载模型
    if load_from_model and os.path.exists(load_from_model):
        print(f"从 {load_from_model} 加载预训练模型...")
        model = MaskablePPO.load(load_from_model, env=env)
    else:
        print("创建新模型...")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=0.99,
            n_steps=2048,
            ent_coef=0.01,
            learning_rate=3e-4,
            batch_size=64,
            n_epochs=10,
            clip_range=0.2,
            tensorboard_log="./curriculum_logs/",
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                activation_fn=nn.ReLU
            )
        )
    
    # 3. 训练模型
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # 4. 保存模型
    if save_to_model:
        print(f"训练完成，保存模型到 {save_to_model}...")
        model.save(save_to_model)

    # 5. 清理环境
    env.close()
    del model, env


if __name__ == "__main__":
    # --- 模型保存路径定义 ---
    model_dir = "curriculum_models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_stage1_path = os.path.join(model_dir, "model_stage1.zip")
    model_stage2_path = os.path.join(model_dir, "model_stage2.zip")
    model_stage3_path = os.path.join(model_dir, "model_stage3.zip")
    final_model_path = os.path.join(model_dir, "final_model.zip")

    # --- 按顺序执行课程学习 ---

    # 阶段一: 吃子入门
    train_stage(
        stage_name="吃子入门 (Capture 101)",
        curriculum_stage=1,
        total_timesteps=100_000,
        load_from_model=None, # 从零开始
        save_to_model=model_stage1_path
    )

    # 阶段二: 简单战斗
    train_stage(
        stage_name="简单战斗 (Simple Combat)",
        curriculum_stage=2,
        total_timesteps=200_000,
        load_from_model=model_stage1_path, # 加载阶段一的模型
        save_to_model=model_stage2_path
    )

    # 阶段三: 探索与决策
    train_stage(
        stage_name="探索与决策 (Flip vs. Move)",
        curriculum_stage=3,
        total_timesteps=500_000,
        load_from_model=model_stage2_path, # 加载阶段二的模型
        save_to_model=model_stage3_path
    )

    # 阶段四: 完整对局
    train_stage(
        stage_name="完整对局 (Full Game)",
        curriculum_stage=4,
        total_timesteps=10_000_000, # 在完整游戏上进行大量训练
        load_from_model=model_stage3_path, # 加载阶段三的模型
        save_to_model=final_model_path
    )

    print("\n" + "="*50)
    print("所有课程学习阶段已完成！")
    print(f"最终模型已保存在: {final_model_path}")
    print("="*50)
