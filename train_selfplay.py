# train_selfplay_optimized.py (已应用内存更新优化)
import os
import shutil
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from Game import GameEnvironment
# 【重要】从修改后的文件中导入管理器
from opponent_model_manager import shared_opponent_manager

def main():
    # --- 设置和初始化 ---
    CURRICULUM_MODEL_PATH = "cnn_curriculum_models/final_model_cnn.zip"
    SELF_PLAY_MODEL_DIR = "self_play_models_optimized"
    # 我们不再需要在循环中保存learner和opponent，但保留路径用于最终保存
    FINAL_MODEL_PATH = os.path.join(SELF_PLAY_MODEL_DIR, "final_selfplay_model_optimized.zip")
    
    TOTAL_TRAINING_LOOPS = 100
    STEPS_PER_LOOP = 50_000 # 每次更新对手之间，学习者训练多少步
    
    print("🚀 开始内存优化版自我对弈训练...")
    print(f"📊 总训练循环数: {TOTAL_TRAINING_LOOPS}")
    print(f"🔄 每循环训练步数: {STEPS_PER_LOOP:,}")
    print("⚡ 使用共享对手模型管理器，并采用内存直接更新策略！")
    
    # --- 初始设置 (仅执行一次) ---
    os.makedirs(SELF_PLAY_MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(CURRICULUM_MODEL_PATH):
        raise FileNotFoundError(f"课程学习最终模型不存在: {CURRICULUM_MODEL_PATH}")
    
    # 1. 预加载初始对手模型到共享管理器
    # 这是我们唯一一次从磁盘加载对手模型
    print("🔧 预加载初始对手模型到共享管理器...")
    shared_opponent_manager.load_model(CURRICULUM_MODEL_PATH)
    model_info = shared_opponent_manager.get_model_info()
    print(f"📋 初始对手模型信息: {model_info}")

    # 2. 创建环境 (这个环境将一直被使用，不再重建)
    print("🌍 创建持久化自我对弈环境...")
    env = make_vec_env(
        GameEnvironment,
        n_envs=8,
        env_kwargs={
            'curriculum_stage': 4,
            'opponent_policy': CURRICULUM_MODEL_PATH # 传入路径以触发管理器加载
        }
    )
    
    # 3. 加载学习者模型
    print(f"🤖 加载学习者模型从 {CURRICULUM_MODEL_PATH}...")
    model = MaskablePPO.load(
        CURRICULUM_MODEL_PATH,
        env=env,
        learning_rate=3e-4, # 可以在这里调整学习率
        tensorboard_log="./self_play_tensorboard_logs_optimized/"
    )
    
    print("🎯 开始内存优化版自我对弈训练循环...")
    
    # --- 主训练循环 (现在非常高效) ---
    for i in range(1, TOTAL_TRAINING_LOOPS + 1):
        print(f"\n{'='*60}")
        print(f"🔄 训练循环 {i}/{TOTAL_TRAINING_LOOPS}")
        print(f"{'='*60}")
        
        # (a) 训练学习者 - 这是循环中唯一耗时的部分
        print(f"🏋️  训练学习者 {STEPS_PER_LOOP:,} 步...")
        model.learn(
            total_timesteps=STEPS_PER_LOOP,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # (b) 【核心优化】直接在内存中更新对手模型
        # 这会取代所有的 save, copy, load 和环境重建操作
        shared_opponent_manager.update_model_from_learner(model)
        
        print(f"✅ 循环 {i} 完成！对手已在内存中更新，继续训练。")
        
        # (c) 定期保存检查点 (可选，但推荐)
        if i % 10 == 0:
            checkpoint_path = os.path.join(SELF_PLAY_MODEL_DIR, f"checkpoint_loop_{i}.zip")
            model.save(checkpoint_path)
            print(f"📋 已保存训练检查点到 {checkpoint_path}")
    
    # --- 最终收尾 ---
    model.save(FINAL_MODEL_PATH)
    print(f"\n🎉 内存优化版自我对弈训练完成！")
    print(f"📦 最终模型保存到: {FINAL_MODEL_PATH}")
    
    model_info = shared_opponent_manager.get_model_info()
    print(f"\n📊 优化总结:")
    print(f"   ✅ 最终对手模型状态: {model_info}")
    print(f"   ⚡️ 通过内存更新，避免了 {TOTAL_TRAINING_LOOPS} 次的磁盘I/O和环境重建！")
    print(f"   💾 显著节省了训练时间，提高了硬件利用率。")
    
    env.close()

if __name__ == "__main__":
    main()