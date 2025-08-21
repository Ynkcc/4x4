# src_code/hyperparameter_tuning.py

import os
import warnings
import optuna
import time

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
from utils.constants import MAIN_OPPONENT_PATH, EVALUATION_GAMES, N_ENVS

# --- 配置 ---
# 优化试验的总次数
N_TRIALS = 50
# 每次试验中，模型训练的总步数 (为了快速迭代，可以设置得比主训练小)
TRAINING_TIMESTEPS_PER_TRIAL = 2048 * 8 
# 评估时使用的基准模型
BASELINE_MODEL_PATH = MAIN_OPPONENT_PATH
# 优化结果存储路径
TUNING_OUTPUT_DIR = "./tuning_results/"

def objective(trial: optuna.Trial) -> float:
    """
    Optuna的目标函数，用于定义、训练和评估一组超参数。
    """
    print(f"\n{'='*50}\n trial {trial.number}/{N_TRIALS} is starting...\n{'='*50}")
    
    # --- 1. 定义超参数的搜索空间 ---
    # PPO 核心参数
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    n_epochs = trial.suggest_int("n_epochs", 5, 20)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    vf_coef = trial.suggest_float("vf_coef", 0.2, 0.8)

    # 自定义网络架构参数
    net_features_dim = trial.suggest_categorical("net_features_dim", [128, 256, 512])
    net_num_res_blocks = trial.suggest_int("net_num_res_blocks", 2, 8)
    net_num_hidden_channels = trial.suggest_categorical("net_num_hidden_channels", [32, 64, 128])
    
    # 确保 n_steps > batch_size
    if n_steps < batch_size:
        # Optuna V2. Pruning.
        # Optuna V3. trial.report() is used for pruning and report() returns void.
        # See https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-implement-pruning-in-my-objective-function
        raise optuna.exceptions.TrialPruned()


    # --- 2. 使用建议的超参数创建环境和模型 ---
    run_name = f"trial_{trial.number}_{int(time.time())}"
    trial_model_save_path = os.path.join(TUNING_OUTPUT_DIR, f"{run_name}.zip")

    try:
        # 创建训练环境 (这里为了简单，不使用对手池，直接自我对弈)
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
            verbose=0, # 在调优时关闭详细日志
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
        model.learn(total_timesteps=TRAINING_TIMESTEPS_PER_TRIAL, progress_bar=True)
        model.save(trial_model_save_path)
        
        env.close() # 清理环境

        # --- 4. 评估模型 ---
        # 确保基准模型存在
        if not os.path.exists(BASELINE_MODEL_PATH):
            raise FileNotFoundError(f"基准模型 {BASELINE_MODEL_PATH} 不存在，无法进行评估。")

        print(f"Evaluating trial {trial.number} against baseline model...")
        # evaluate_models 返回挑战者的胜率
        win_rate = evaluate_models(
            challenger_path=trial_model_save_path,
            main_opponent_path=BASELINE_MODEL_PATH,
            show_progress=False # 在调优时关闭进度条
        )
        print(f"Trial {trial.number} result: win_rate = {win_rate:.2%}")

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
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
    
    # 创建一个研究(study)，目标是最大化胜率
    # TPE (Tree-structured Parzen Estimator) 是一种高效的采样算法
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    
    try:
        # 运行优化
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("🛑 Tuning process interrupted by user.")
    
    # --- 打印优化结果 ---
    print("\n" + "="*50)
    print("          📊 Hyperparameter Tuning Results 📊")
    print("="*50)
    print(f"Number of finished trials: {len(study.trials)}")
    
    print("\n--- Best Trial ---")
    best_trial = study.best_trial
    print(f"  Value (Win Rate): {best_trial.value:.4f}")
    
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    print("\n--- All Trials Summary ---")
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
    
    print(f"\n✅ Tuning finished! You can find the best parameters above.")
    print("To visualize results, you can use optuna-dashboard:")
    print("  pip install optuna-dashboard")
    print(f"  optuna-dashboard sqlite:///{study.storage.get_url(study.study_name)}")