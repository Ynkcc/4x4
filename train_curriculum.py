# train_curriculum.py (已修改以使用自定义CNN策略)
import os
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO

# 【重要修改】导入我们新的环境和自定义策略
from Game import GameEnvironment
from custom_policy import CustomActorCriticPolicy

def train_stage(stage_name, curriculum_stage, total_timesteps, load_from_model=None, save_to_model=None):
    """
    一个通用的函数，用于训练课程学习的单个阶段。
    """
    print("\n" + "="*50)
    print(f"开始课程学习阶段: {stage_name} (Stage {curriculum_stage})")
    print(f"训练步数: {total_timesteps:,}")
    print("="*50 + "\n")

    # 1. 创建环境 (使用新的CNN版GameEnvironment)
    env = make_vec_env(GameEnvironment, n_envs=8, env_kwargs={'curriculum_stage': curriculum_stage})

    # 【重要修改】定义策略网络参数
    # features_extractor_kwargs: 传递给我们自定义的CustomNetwork的参数
    #   - num_res_blocks: 残差块数量，影响CNN的深度和特征提取能力
    #   - num_hidden_channels: CNN中间层通道数，影响网络容量
    # net_arch: 定义在特征提取之后，策略头(pi)和价值头(vf)的网络结构
    #   - CustomNetwork输出1088维特征(1024来自CNN + 64来自FC标量分支)
    #   - 策略头和价值头各使用两层256维的全连接层
    policy_kwargs = dict(
        features_extractor_kwargs=dict(num_res_blocks=4, num_hidden_channels=64),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # 2. 创建或加载模型
    if load_from_model and os.path.exists(load_from_model):
        print(f"从 {load_from_model} 加载预训练模型...")
        # 加载模型时，它会自动使用与保存时相同的策略结构
        # 我们也可以通过 custom_objects 来确保加载时能正确识别自定义策略
        model = MaskablePPO.load(load_from_model, env=env, custom_objects={"policy_class": CustomActorCriticPolicy})
    else:
        print("创建新的CNN模型...")
        model = MaskablePPO(
            # 使用我们自定义的策略！
            policy=CustomActorCriticPolicy,
            env=env,
            verbose=1,
            gamma=0.99,
            n_steps=2048,
            ent_coef=0.01,
            learning_rate=3e-4,
            batch_size=64,
            n_epochs=10,
            clip_range=0.2,
            tensorboard_log="./cnn_curriculum_logs/", # 使用新的日志目录
            policy_kwargs=policy_kwargs
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
    model_dir = "cnn_curriculum_models" # 使用新的模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    model_stage1_path = os.path.join(model_dir, "model_stage1.zip")
    model_stage2_path = os.path.join(model_dir, "model_stage2.zip")
    # 【重要修改】移除阶段3
    final_model_path = os.path.join(model_dir, "final_model_cnn.zip") # 新的模型名称

    # --- 按顺序执行简化的课程学习 ---

    # 阶段一: 吃子入门
    train_stage(
        stage_name="吃子入门 (Capture 101)",
        curriculum_stage=1,
        total_timesteps=200_000, # 可以适当增加简单任务的步数，为CNN打好基础
        load_from_model=None,
        save_to_model=model_stage1_path
    )

    # 阶段二: 简单战斗
    train_stage(
        stage_name="简单战斗 (Simple Combat)",
        curriculum_stage=2,
        total_timesteps=300_000,
        load_from_model=model_stage1_path,
        save_to_model=model_stage2_path
    )

    # 【重要修改】直接进入最终阶段
    # 阶段四: 完整对局
    train_stage(
        stage_name="完整对局 (Full Game with CNN)",
        curriculum_stage=4,
        total_timesteps=1_000_000,
        load_from_model=model_stage2_path, # 从阶段二的模型加载
        save_to_model=final_model_path
    )

    print("\n" + "="*50)
    print("所有CNN课程学习阶段已完成！")
    print(f"最终模型已保存在: {final_model_path}")
    print("="*50)
