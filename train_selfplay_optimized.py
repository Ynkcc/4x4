# train_selfplay_optimized.py - 优化的自我对弈训练脚本
import os
import shutil
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from Game import GameEnvironment
from custom_policy import CustomActorCriticPolicy
from opponent_model_manager import shared_opponent_manager

def main():
    # 设置和初始化
    CURRICULUM_MODEL_PATH = "cnn_curriculum_models/final_model_cnn.zip"
    SELF_PLAY_MODEL_DIR = "self_play_models_optimized"
    LEARNER_MODEL_PATH = os.path.join(SELF_PLAY_MODEL_DIR, "learner.zip")
    OPPONENT_MODEL_PATH = os.path.join(SELF_PLAY_MODEL_DIR, "opponent.zip")
    
    TOTAL_TRAINING_LOOPS = 100  # 总共更新多少次对手
    STEPS_PER_LOOP = 50000      # 每次更新对手之间，学习者训练多少步
    
    print("🚀 开始优化版自我对弈训练...")
    print(f"📊 总训练循环数: {TOTAL_TRAINING_LOOPS}")
    print(f"🔄 每循环训练步数: {STEPS_PER_LOOP}")
    print("⚡ 使用共享对手模型管理器，优化内存使用和性能")
    
    # 创建模型目录
    os.makedirs(SELF_PLAY_MODEL_DIR, exist_ok=True)
    
    # 检查课程学习最终模型是否存在
    if not os.path.exists(CURRICULUM_MODEL_PATH):
        raise FileNotFoundError(f"课程学习最终模型不存在: {CURRICULUM_MODEL_PATH}")
    
    # 关键的初始步骤：使用课程学习的最终模型作为第一个对手
    print(f"📦 复制课程学习最终模型 {CURRICULUM_MODEL_PATH} 作为初始对手...")
    shutil.copy(CURRICULUM_MODEL_PATH, OPPONENT_MODEL_PATH)
    
    # 预加载对手模型到共享管理器（避免每个环境重复加载）
    print("🔧 预加载对手模型到共享管理器...")
    shared_opponent_manager.load_model(OPPONENT_MODEL_PATH)
    model_info = shared_opponent_manager.get_model_info()
    print(f"📋 对手模型信息: {model_info}")
    
    # 创建环境 - 现在使用共享模型管理器
    print("🌍 创建优化的自我对弈环境...")
    env = make_vec_env(
        GameEnvironment,
        n_envs=8,
        env_kwargs={
            'curriculum_stage': 4,  # 始终是完整游戏
            'opponent_policy': OPPONENT_MODEL_PATH
        }
    )
    
    # 加载学习者模型 - 从课程学习的最终模型开始
    print(f"🤖 加载学习者模型从 {CURRICULUM_MODEL_PATH}...")
    model = MaskablePPO.load(
        CURRICULUM_MODEL_PATH,
        env=env,
        learning_rate=3e-4,
        tensorboard_log="./self_play_tensorboard_logs_optimized/"
    )
    
    print("🎯 开始优化版自我对弈训练循环...")
    
    # 主训练循环
    for i in range(1, TOTAL_TRAINING_LOOPS + 1):
        print(f"\n{'='*60}")
        print(f"🔄 训练循环 {i}/{TOTAL_TRAINING_LOOPS}")
        print(f"{'='*60}")
        
        # (a) 训练学习者
        print(f"🏋️  训练学习者 {STEPS_PER_LOOP:,} 步...")
        model.learn(
            total_timesteps=STEPS_PER_LOOP,
            reset_num_timesteps=False,  # 保持连续的时间步计数
            progress_bar=True
        )
        
        # (b) 保存学习者
        print("💾 保存学习者模型...")
        model.save(LEARNER_MODEL_PATH)
        
        # (c) 更新对手 - 用当前学习者覆盖对手
        print("🔄 更新对手模型...")
        shutil.copy(LEARNER_MODEL_PATH, OPPONENT_MODEL_PATH)
        
        # (d) 【关键优化】更新共享模型管理器中的对手模型
        print("⚡ 更新共享模型管理器...")
        shared_opponent_manager.load_model(OPPONENT_MODEL_PATH)
        
        # (e) 重建环境 - 现在更高效，因为模型已经在共享管理器中
        print("🌍 重建环境...")
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
        
        print(f"✅ 循环 {i} 完成！")
        
        # 定期保存检查点
        if i % 10 == 0:
            checkpoint_path = os.path.join(SELF_PLAY_MODEL_DIR, f"checkpoint_loop_{i}.zip")
            model.save(checkpoint_path)
            print(f"📋 保存检查点到 {checkpoint_path}")
    
    # 最终收尾
    final_model_path = os.path.join(SELF_PLAY_MODEL_DIR, "final_selfplay_model_optimized.zip")
    model.save(final_model_path)
    print(f"\n🎉 优化版自我对弈训练完成！")
    print(f"📦 最终模型保存到: {final_model_path}")
    
    # 显示优化效果总结
    model_info = shared_opponent_manager.get_model_info()
    print(f"\n📊 优化总结:")
    print(f"   ✅ 共享模型管理器状态: {model_info}")
    print(f"   ⚡ 避免了 {8 * TOTAL_TRAINING_LOOPS} 次重复模型加载")
    print(f"   💾 节省了大量内存和加载时间")
    
    env.close()

if __name__ == "__main__":
    main()
