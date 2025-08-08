# train_simple.py
import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO

# 导入本地模块
from game.environment import GameEnvironment
from game.policy import CustomActorCriticPolicy
from training.simple_agent import SimpleAgent
from utils.scheduler import linear_schedule
from utils.constants import N_ENVS, INITIAL_LR

def train_with_simple_opponent():
    """
    使用一个简单的、基于规则的对手来训练主模型。
    这用于验证学习流程是否正常工作。
    """
    print("=================================================")
    print("      开始使用简单Agent进行验证性训练       ")
    print("=================================================")
    
    # --- 配置 ---
    TOTAL_STEPS = 200_000 # 总训练步数
    LOG_DIR = "./tensorboard_logs/simple_agent_test/"
    MODEL_SAVE_PATH = "./models/simple_agent_test/final_model.zip"
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # 1. 创建简单对手的实例
    simple_opponent = SimpleAgent()
    
    # 2. 创建环境，并通过 `env_kwargs` 将对手实例注入
    env_kwargs = {
        'curriculum_stage': 4,       # 使用完整游戏模式
        'opponent_agent': simple_opponent
    }
    
    # 使用 SubprocVecEnv 以实现真正的并行化
    env = make_vec_env(
        GameEnvironment,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs
    )
    
    # 3. 创建或加载学习者模型
    # 在这里，我们总是从头开始训练一个新的模型来进行验证
    model = MaskablePPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=linear_schedule(INITIAL_LR),
        verbose=1,
        tensorboard_log=LOG_DIR,
        gamma=0.99,
        n_steps=2048
    )
    
    print("\n--- [训练开始] ---")
    print(f"学习者模型: MaskablePPO (CustomPolicy)")
    print(f"对手模型: SimpleAgent (随机策略)")
    print(f"并行环境数: {N_ENVS}")
    print(f"总训练步数: {TOTAL_STEPS:,}")
    print(f"日志路径: {LOG_DIR}")
    print(f"模型保存路径: {MODEL_SAVE_PATH}")
    print("---------------------------------\n")

    # 4. 启动训练
    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断。")
        
    # 5. 保存最终模型
    model.save(MODEL_SAVE_PATH)
    print(f"\n🎉 训练完成！最终模型已保存到: {MODEL_SAVE_PATH}")
    
    env.close()

if __name__ == "__main__":
    train_with_simple_opponent()