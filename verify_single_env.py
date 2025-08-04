# verify_single_env.py
# 目的：在一个完全非并行的设置中（n_envs=1）测试评估函数，以验证“无效动作”警告的根本原因。

import os
import shutil
import sys
import traceback
import warnings
import numpy as np

# --- 环境设置 ---
# 禁用不必要的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# --- 模块导入 ---
# 将项目根目录添加到sys.path以确保可以导入自定义模块
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的模块和类
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from game.environment import GameEnvironment
from utils.constants import (
    MAIN_OPPONENT_PATH, CHALLENGER_PATH, SELF_PLAY_OUTPUT_DIR,
    CURRICULUM_MODEL_PATH, SELF_PLAY_MODEL_PATH, EVALUATION_GAMES
)
from utils.model_compatibility import setup_legacy_imports


def evaluate_models_single_env(challenger_path: str, main_opponent_path: str) -> float:
    """
    【特别验证版】评估函数，强制使用单个环境（n_envs=1）。
    这个版本专门用于验证“无效动作”问题是否在非并行环境下复现。
    """
    challenger_name = os.path.basename(challenger_path)
    main_name = os.path.basename(main_opponent_path)
    print(f"\n--- 单环境评估开始: (挑战者) {challenger_name} vs (主宰者) {main_name} ---")
    
    eval_env = None
    try:
        # 【核心验证点】强制创建单个环境 (n_envs=1)
        # 这里不指定 vec_env_cls，让 stable-baselines3 使用其默认的单环境包装器。
        # 这完全模拟了“不使用任何加速方案”的场景。
        eval_env = make_vec_env(
            GameEnvironment,
            n_envs=1,  # 强制为 1
            env_kwargs={
                'curriculum_stage': 4,
                'opponent_policy': main_opponent_path
            }
        )
        
        # 加载挑战者模型
        challenger_model = MaskablePPO.load(challenger_path, env=eval_env, device='auto')
        
        wins = 0
        draws = 0
        losses = 0
        games_played = 0
        
        obs = eval_env.reset()
        
        # 获取初始动作掩码
        action_masks = np.array(eval_env.env_method('action_masks'))
        
        while games_played < EVALUATION_GAMES:
            action, _ = challenger_model.predict(obs, action_masks=action_masks, deterministic=True)
            
            obs, rewards, dones, infos = eval_env.step(action)
            
            # 从返回的 infos 中为下一步提取新的动作掩码
            action_masks = np.array([info.get('action_mask') for info in infos])
            
            # 由于 n_envs=1, dones 和 infos 都是列表，但只有一个元素
            if dones[0]:
                games_played += 1
                winner = infos[0].get('winner')
                if winner == 1:
                    wins += 1
                elif winner == -1:
                    losses += 1
                else:
                    draws += 1
                
                print(f"  游戏 {games_played}/{EVALUATION_GAMES} 结束。当前战绩: {wins}胜/{losses}负/{draws}平")

                if games_played >= EVALUATION_GAMES:
                    break
        
        total_games_for_winrate = wins + losses
        win_rate = wins / total_games_for_winrate if total_games_for_winrate > 0 else 0.0
        
        print(f"\n--- 单环境评估结束: 共进行了 {games_played} 局游戏 ---")
        print(f"    挑战者战绩: {wins}胜 / {losses}负 / {draws}平")
        print(f"    挑战者胜率 (胜 / (胜+负)): {win_rate:.2%}")
        
        return win_rate

    finally:
        if eval_env:
            eval_env.close()


def run_verification():
    """
    执行验证流程的主函数。
    """
    print("==============================================")
    print("==   验证脚本：单环境评估是否存在无效动作   ==")
    print("==============================================")

    # 1. 确保模型目录存在
    os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)

    # 2. 准备用于测试的模型文件
    source_model_path = MAIN_OPPONENT_PATH
    if not os.path.exists(source_model_path):
        print(f"未找到主宰者模型，尝试从 '{SELF_PLAY_MODEL_PATH}' 或 '{CURRICULUM_MODEL_PATH}' 复制...")
        source_model_path = SELF_PLAY_MODEL_PATH if os.path.exists(SELF_PLAY_MODEL_PATH) else CURRICULUM_MODEL_PATH
        if not os.path.exists(source_model_path):
            print(f"❌ 错误：找不到任何可用的模型进行测试。请确保模型存在。")
            return
    
    print(f"将使用 '{source_model_path}' 作为挑战者和主宰者进行自我对战测试。")



    # 3. 设置模块导入兼容性
    setup_legacy_imports()

    # 4. 调用特别版的评估函数并观察输出
    try:
        print("\n▶️  开始调用单环境评估函数 (evaluate_models_single_env)...")
        print("请密切观察控制台是否出现 '警告：试图执行无效动作' 的信息。")
        
        win_rate = evaluate_models_single_env(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
        
        print("\n--- ✅ 验证成功 ---")
        print(f"评估函数在单环境下成功运行，没有崩溃。")
        print("请检查以上日志，判断是否存在'无效动作'的警告。")
        print("如果警告消失，则证明该问题确实与并行环境有关。")

    except Exception:
        print("\n--- ❌ 验证失败 ---")
        print("评估函数在运行时发生严重错误:")
        traceback.print_exc()


if __name__ == "__main__":
    run_verification()