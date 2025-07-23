import os
from stable_baselines3.common.env_util import make_vec_env
# 新增 SubprocVecEnv 的导入
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

# 确保 GameEnvironment 类被正确导入
from Game import GameEnvironment


def main():
    """
    主训练函数。
    """
    # --- 训练参数 ---
    # 注意：请将此值设置为您希望达到的新的总步数
    # 例如，如果已训练200万步，希望再训练300万步，则设置为 5_000_000
    total_timesteps = 5_000_000
    n_envs = 8  # 使用多个环境并行采样
    learning_rate = 3e-4
    batch_size = 64
    n_steps = 2048

    # --- 路径设置 ---
    log_dir = "./banqi_ppo_logs/"
    # 这是最终模型的保存路径，也将是加载模型的路径
    model_save_path = os.path.join(log_dir, "banqi_ppo_model.zip")
    best_model_save_path = os.path.join(log_dir, "best_model")
    os.makedirs(log_dir, exist_ok=True)

    # --- 创建并行化的环境 ---
    # 注意：在 Windows 或 macOS 的某些情况下，多进程需要放在 if __name__ == '__main__': 块内
    if __name__ == '__main__':
        # 通过 vec_env_cls=SubprocVecEnv 启用多进程并行环境
        print(f"--- 启用 {n_envs} 个并行环境进行数据收集 ---")
        env = make_vec_env(
            GameEnvironment, 
            n_envs=n_envs, 
            vec_env_cls=SubprocVecEnv
        )

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
        eval_callback = MaskableEvalCallback(
            env,
            best_model_save_path=best_model_save_path,
            log_path=log_dir,
            eval_freq=max(20000 // n_envs, 1),
            n_eval_episodes=20,
            deterministic=False,
            render=False
        )
        
        # --- 模型定义与加载 ---
        # 检查是否存在已保存的模型
        if os.path.exists(model_save_path):
            print(f"--- 发现已存在的模型，从 {model_save_path} 加载 ---")
            # 加载现有模型，并确保它使用当前的环境和TensorBoard日志设置
            model = MaskablePPO.load(
                model_save_path,
                env=env,
                tensorboard_log=log_dir
            )
            # 重置模型的 timesteps，以便 `learn` 方法的进度条正确显示
            model.num_timesteps = model._total_timesteps
        else:
            print("--- 未发现已存在的模型，创建新模型 ---")
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

        # --- 开始或继续训练 ---
        print("--- 开始或继续训练 ---")
        try:
            # `learn` 会从加载的步数继续，直到达到新的 total_timesteps
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=True,
                reset_num_timesteps=False # 关键参数：告诉 alearn 不要重置步数计数器
            )
        except KeyboardInterrupt:
            print("--- 训练被手动中断 ---")

        # --- 保存最终模型 ---
        # 这里的路径不需要 .zip 后缀，save 方法会自动添加
        final_model_save_path_base = os.path.join(log_dir, "banqi_ppo_model")
        model.save(final_model_save_path_base)
        print(f"--- 训练完成，最终模型已保存至 {final_model_save_path_base}.zip ---")
        print(f"--- 最佳模型已保存在 {best_model_save_path} 文件夹中 ---")


if __name__ == "__main__":
    main()