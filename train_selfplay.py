# train_selfplay.py - 自我对弈训练脚本
import os
import shutil
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from Game import GameEnvironment
from custom_policy import CustomActorCriticPolicy

def main():
    # 设置和初始化
    CURRICULUM_MODEL_PATH = "cnn_curriculum_models/final_model_cnn.zip"
    SELF_PLAY_MODEL_DIR = "self_play_models"
    LEARNER_MODEL_PATH = os.path.join(SELF_PLAY_MODEL_DIR, "learner.zip")
    OPPONENT_MODEL_PATH = os.path.join(SELF_PLAY_MODEL_DIR, "opponent.zip")
    
    TOTAL_TRAINING_LOOPS = 100  # 总共更新多少次对手
    STEPS_PER_LOOP = 50000      # 每次更新对手之间，学习者训练多少步
    
    print("开始自我对弈训练...")
    print(f"总训练循环数: {TOTAL_TRAINING_LOOPS}")
    print(f"每循环训练步数: {STEPS_PER_LOOP}")
    
    # 创建模型目录
    os.makedirs(SELF_PLAY_MODEL_DIR, exist_ok=True)
    
    # 检查课程学习最终模型是否存在
    if not os.path.exists(CURRICULUM_MODEL_PATH):
        raise FileNotFoundError(f"课程学习最终模型不存在: {CURRICULUM_MODEL_PATH}")
    
    # 关键的初始步骤：使用课程学习的最终模型作为第一个对手
    print(f"复制课程学习最终模型 {CURRICULUM_MODEL_PATH} 作为初始对手...")
    shutil.copy(CURRICULUM_MODEL_PATH, OPPONENT_MODEL_PATH)
    
    # 创建环境 - 包含对手模型
    print("创建自我对弈环境...")
    env = make_vec_env(
        GameEnvironment,
        n_envs=8,
        env_kwargs={
            'curriculum_stage': 4,  # 始终是完整游戏
            'opponent_policy': OPPONENT_MODEL_PATH
        }
    )
    
    # 加载学习者模型 - 从课程学习的最终模型开始
    print(f"加载学习者模型从 {CURRICULUM_MODEL_PATH}...")
    model = MaskablePPO.load(
        CURRICULUM_MODEL_PATH,
        env=env,
        learning_rate=3e-4,
        tensorboard_log="./self_play_tensorboard_logs/"
    )
    
    print("开始自我对弈训练循环...")
    
    # 主训练循环
    for i in range(1, TOTAL_TRAINING_LOOPS + 1):
        print(f"\n=== 训练循环 {i}/{TOTAL_TRAINING_LOOPS} ===")
        
        # (a) 训练学习者
        print(f"训练学习者 {STEPS_PER_LOOP} 步...")
        model.learn(
            total_timesteps=STEPS_PER_LOOP,
            reset_num_timesteps=False,  # 保持连续的时间步计数
            progress_bar=True
        )
        
        # (b) 保存学习者
        print("保存学习者模型...")
        model.save(LEARNER_MODEL_PATH)
        
        # (c) 更新对手 - 用当前学习者覆盖对手
        print("更新对手模型...")
        shutil.copy(LEARNER_MODEL_PATH, OPPONENT_MODEL_PATH)
        
        # (d) 重建环境 - 关键步骤！
        print("重建环境以加载新的对手模型...")
        env.close()
        env = make_vec_env(
            GameEnvironment,
            n_envs=8,
            env_kwargs={
                'curriculum_stage': 4,
                'opponent_policy': OPPONENT_MODEL_PATH
            }
        )
        model.set_env(env)
        
        print(f"循环 {i} 完成！")
        
        # 定期保存检查点
        if i % 10 == 0:
            checkpoint_path = os.path.join(SELF_PLAY_MODEL_DIR, f"checkpoint_loop_{i}.zip")
            model.save(checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")
    
    # 最终收尾
    final_model_path = os.path.join(SELF_PLAY_MODEL_DIR, "final_selfplay_model.zip")
    model.save(final_model_path)
    print(f"\n自我对弈训练完成！最终模型保存到: {final_model_path}")
    
    env.close()

if __name__ == "__main__":
    main()
