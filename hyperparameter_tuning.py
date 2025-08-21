# hyperparameter_tuning.py

import os
import warnings
import optuna
import time
import traceback

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

# 导入项目中的模块
from game.environment import GameEnvironment
from game.policy import CustomActorCriticPolicy
from training.evaluator import evaluate_models
# 导入经过整理的常量
from utils.constants import (
    N_ENVS,
    MAIN_OPPONENT_PATH
)

# --- 调优配置 ---
# 优化试验的总次数
N_TRIALS = 50
# 每次试验中，模型训练的总步数 (为了快速迭代，可以设置得比主训练小)
TRAINING_TIMESTEPS_PER_TRIAL = 2048 * 10
# 评估时使用的基准模型
BASELINE_MODEL_PATH = MAIN_OPPONENT_PATH
# 优化结果存储路径
TUNING_OUTPUT_DIR = "./tuning_results/"
# Optuna数据库文件路径，用于持久化存储研究结果
OPTUNA_DB_PATH = os.path.join(TUNING_OUTPUT_DIR, "tuning_study.db")


def objective(trial: optuna.Trial) -> float:
    """
    Optuna的目标函数，用于定义、训练和评估一组超参数。
    """
    print(f"\n{'='*50}\n trial {trial.number}/{N_TRIALS} is starting...\n{'='*50}")

    # --- 1. 定义超参数的搜索空间 ---
    # PPO 核心参数
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    n_epochs = trial.suggest_int("n_epochs", 5, 20)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    vf_coef = trial.suggest_float("vf_coef", 0.2, 0.8)

    # 自定义网络架构参数
    net_features_dim = trial.suggest_categorical("net_features_dim", [128, 256, 512])
    net_num_res_blocks = trial.suggest_int("net_num_res_blocks", 2, 8)
    net_num_hidden_channels = trial.suggest_categorical("net_num_hidden_channels", [32, 64, 128])

    # 剪枝规则：确保 n_steps >= batch_size
    if n_steps < batch_size:
        raise optuna.exceptions.TrialPruned()

    # --- 2. 使用建议的超参数创建环境和模型 ---
    run_name = f"trial_{trial.number}_{int(time.time())}"
    # 将临时模型存放在子目录中，保持根目录整洁
    temp_model_dir = os.path.join(TUNING_OUTPUT_DIR, "temp_models")
    os.makedirs(temp_model_dir, exist_ok=True)
    trial_model_save_path = os.path.join(temp_model_dir, f"{run_name}.zip")

    try:
        # 创建训练环境 (直接自我对弈)
        env = make_vec_env(GameEnvironment, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

        model = MaskablePPO(
            policy=CustomActorCriticPolicy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            device='auto',
            verbose=0, # 在调优时关闭SB3的详细日志
            policy_kwargs={
                'features_extractor_kwargs': {
                    'features_dim': net_features_dim,
                    'num_res_blocks': net_num_res_blocks,
                    'num_hidden_channels': net_num_hidden_channels
                }
            }
        )

        # --- 3. 训练模型 ---
        print(f"Training trial {trial.number} for {TRAINING_TIMESTEPS_PER_TRIAL} timesteps...")
        model.learn(total_timesteps=TRAINING_TIMESTEPS_PER_TRIAL, progress_bar=False)
        model.save(trial_model_save_path)

        env.close() # 清理环境

        # --- 4. 评估模型 ---
        # 确保基准模型存在
        if not os.path.exists(BASELINE_MODEL_PATH):
            print(f"警告: 基准模型 {BASELINE_MODEL_PATH} 不存在，将跳过评估并返回0.0。") #
            return 0.0

        print(f"Evaluating trial {trial.number} against baseline model...")
        # evaluate_models 返回挑战者的胜率
        win_rate = evaluate_models(
            challenger_path=trial_model_save_path,
            main_opponent_path=BASELINE_MODEL_PATH,
            show_progress=False # 在调优时关闭评估的进度条
        )
        print(f"Trial {trial.number} result: win_rate = {win_rate:.2%}")

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        traceback.print_exc()
        # 如果出错，返回一个很差的结果
        return 0.0
    finally:
        # 清理本次试验产生的模型文件
        if os.path.exists(trial_model_save_path):
            os.remove(trial_model_save_path)

    return win_rate


if __name__ == '__main__':
    # --- 主程序入口 ---
    print("🚀 Starting hyperparameter tuning with Optuna...")

    # 确保输出目录存在
    os.makedirs(TUNING_OUTPUT_DIR, exist_ok=True)

    # 创建一个研究(study)，并使用SQLite进行持久化存储
    # 这样即使程序中断，也可以从上次的进度继续
    storage_name = f"sqlite:///{OPTUNA_DB_PATH}"
    study = optuna.create_study(
        study_name="dark_chess_ppo_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        storage=storage_name,
        load_if_exists=True # 如果数据库文件已存在，则加载之前的研究
    )

    try:
        # 运行优化
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n🛑 Tuning process interrupted by user.")

    # --- 打印优化结果 ---
    print("\n" + "="*50)
    print("          📊 Hyperparameter Tuning Results 📊")
    print("="*50)
    print(f"Study name: {study.study_name}")
    print(f"Storage: {storage_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\n--- Best Trial ---")
    best_trial = study.best_trial
    print(f"  Value (Win Rate): {best_trial.value:.4f}")

    print("  Best Parameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- 保存最佳参数到文件 ---
    best_params_path = os.path.join(TUNING_OUTPUT_DIR, "best_params.txt")
    with open(best_params_path, "w") as f:
        f.write("Best trial:\n")
        f.write(f"  Value: {best_trial.value}\n")
        f.write("  Params:\n")
        for key, value in best_trial.params.items():
            f.write(f"    {key}: {value}\n")
    print(f"\n✅ Best parameters saved to: {best_params_path}")

    print("\n--- All Trials Summary (Top 10) ---")
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df.sort_values("value", ascending=False).head(10))

    print(f"\n✅ Tuning finished! You can find the best parameters above.")
    print("To visualize results, you can use optuna-dashboard:")
    print("  pip install optuna-dashboard")
    print(f"  optuna-dashboard {storage_name}")