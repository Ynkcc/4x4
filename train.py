import os
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from Game import GameEnvironment


def main():
    """
    主训练函数。
    """
    # --- 训练参数 ---
    total_timesteps = 2_000_000
    n_envs = 8  # 使用多个环境并行采样
    learning_rate = 3e-4
    batch_size = 64
    n_steps = 2048

    # --- 路径设置 ---
    log_dir = "./banqi_ppo_logs/"
    model_save_path = os.path.join(log_dir, "banqi_ppo_model")
    best_model_save_path = os.path.join(log_dir, "best_model")
    os.makedirs(log_dir, exist_ok=True)

    # --- 创建并行化的环境 ---
    # 使用 SubprocVecEnv 可以利用多核 CPU
    if __name__ == '__main__':
        # 注意：暂时移除 SubprocVecEnv 以进行调试
        env = make_vec_env(GameEnvironment, n_envs=n_envs) #, vec_env_cls=SubprocVecEnv)

        # --- 回调函数设置 ---
        # 1. 定期保存模型的 Checkpoint
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10000 // n_envs, 1),
            save_path=log_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        # 2. 评估并保存最佳模型
        # 注意：对于自我对弈，评估环境也应该是自我对弈。
        # MaskableEvalCallback 会处理动作掩码。
        eval_callback = MaskableEvalCallback(
            env,  # 在同一个环境设置上评估
            best_model_save_path=best_model_save_path,
            log_path=log_dir,
            eval_freq=max(20000 // n_envs, 1),
            n_eval_episodes=20,
            deterministic=False, # 在评估时也使用随机策略
            render=False
        )

        # --- 模型定义 ---
        # MaskablePPO 会自动从环境返回的 info dict 中获取 "action_mask"
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=0.99,
            n_steps=n_steps,
            ent_coef=0.01,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=10,
            clip_range=0.2,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=[256, 256]) # 使用更大的网络
        )

        # --- 开始训练 ---
        print("--- 开始训练 ---")
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("--- 训练被手动中断 ---")

        # --- 保存最终模型 ---
        model.save(model_save_path)
        print(f"--- 训练完成，最终模型已保存至 {model_save_path} ---")
        print(f"--- 最佳模型已保存在 {best_model_save_path} 文件夹中 ---")


if __name__ == "__main__":
    main()
