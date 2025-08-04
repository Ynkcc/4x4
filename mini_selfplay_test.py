# mini_selfplay_test.py - 小规模自我对弈训练测试
import os
import shutil
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from Game import GameEnvironment
from custom_policy import CustomActorCriticPolicy

def mini_selfplay_test():
    """运行一个小规模的自我对弈训练测试"""
    print("开始小规模自我对弈训练测试...")
    
    # 设置路径
    CURRICULUM_MODEL_PATH = "cnn_curriculum_models/final_model_cnn.zip"
    SELF_PLAY_MODEL_DIR = "test_self_play_models"
    LEARNER_MODEL_PATH = os.path.join(SELF_PLAY_MODEL_DIR, "learner.zip")
    OPPONENT_MODEL_PATH = os.path.join(SELF_PLAY_MODEL_DIR, "opponent.zip")
    
    # 小规模测试参数
    TOTAL_TRAINING_LOOPS = 2    # 只运行2个循环
    STEPS_PER_LOOP = 1000       # 每循环只训练1000步
    
    # 创建测试目录
    os.makedirs(SELF_PLAY_MODEL_DIR, exist_ok=True)
    
    # 检查模型存在
    if not os.path.exists(CURRICULUM_MODEL_PATH):
        print(f"错误: 课程学习最终模型不存在: {CURRICULUM_MODEL_PATH}")
        return False
    
    # 复制初始对手
    print(f"复制初始对手模型...")
    shutil.copy(CURRICULUM_MODEL_PATH, OPPONENT_MODEL_PATH)
    
    # 创建环境
    print("创建环境...")
    env = make_vec_env(
        GameEnvironment,
        n_envs=2,  # 减少环境数量以便快速测试
        env_kwargs={
            'curriculum_stage': 4,
            'opponent_policy': OPPONENT_MODEL_PATH
        }
    )
    
    # 加载模型
    print("加载学习者模型...")
    model = MaskablePPO.load(
        CURRICULUM_MODEL_PATH,
        env=env,
        learning_rate=3e-4
    )
    
    # 训练循环
    for i in range(1, TOTAL_TRAINING_LOOPS + 1):
        print(f"\n=== 测试循环 {i}/{TOTAL_TRAINING_LOOPS} ===")
        
        # 训练
        print(f"训练 {STEPS_PER_LOOP} 步...")
        model.learn(
            total_timesteps=STEPS_PER_LOOP,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # 保存学习者
        model.save(LEARNER_MODEL_PATH)
        
        # 更新对手
        print("更新对手...")
        shutil.copy(LEARNER_MODEL_PATH, OPPONENT_MODEL_PATH)
        
        # 重建环境
        print("重建环境...")
        env.close()
        env = make_vec_env(
            GameEnvironment,
            n_envs=2,
            env_kwargs={
                'curriculum_stage': 4,
                'opponent_policy': OPPONENT_MODEL_PATH
            }
        )
        model.set_env(env)
        
        print(f"测试循环 {i} 完成！")
    
    # 清理
    env.close()
    
    # 清理测试文件
    if os.path.exists(SELF_PLAY_MODEL_DIR):
        shutil.rmtree(SELF_PLAY_MODEL_DIR)
        print("清理测试文件...")
    
    print("小规模自我对弈训练测试完成！")
    return True

if __name__ == "__main__":
    success = mini_selfplay_test()
    if success:
        print("\n✅ 小规模自我对弈训练测试成功！")
    else:
        print("\n❌ 小规模自我对弈训练测试失败！")
