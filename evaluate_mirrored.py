# evaluate_mirrored.py
"""
镜像对局评估脚本。

【优化】:
- 不再维护独立的 play_one_game 函数，而是直接从 training.evaluator 模块导入，确保评估逻辑统一。
- 对局数现在也由 constants.py 控制。
"""
import os
import warnings

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import numpy as np
import time
from tqdm import tqdm

# 【更新】导入本地模块
from game.environment import GameEnvironment
# 【更新】直接复用训练器中的评估逻辑和Agent
from training.evaluator import EvaluationAgent, _play_one_game
# 【更新】从常量文件中读取配置
from utils.constants import EVALUATION_GAMES


def evaluate_mirrored_matches(model1_path: str, model2_path: str):
    """
    执行镜像对局评估。

    :param model1_path: 模型1的路径。
    :param model2_path: 模型2的路径。
    """
    # 【优化】校验评估局数
    if EVALUATION_GAMES % 2 != 0:
        raise ValueError(f"EVALUATION_GAMES ({EVALUATION_GAMES}) 必须是偶数，才能进行完美的镜像对局。")
    num_groups = EVALUATION_GAMES // 2

    print("=" * 70)
    print("           ⚔️  镜像对局评估系统 ⚔️")
    print("=" * 70)

    # 1. 加载模型
    try:
        model1 = EvaluationAgent(model1_path)
        model2 = EvaluationAgent(model2_path)
        print(f"评测模型 A: {model1.name}")
        print(f"评测模型 B: {model2.name}")
        print(f"对局组数: {num_groups} (总计 {EVALUATION_GAMES} 局游戏)")
        print("-" * 70)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 2. 初始化环境和计分板
    eval_env = GameEnvironment()
    scores = {
        'model1_wins': 0,
        'model2_wins': 0,
        'draws': 0,
        'model1_as_red_wins': 0,
        'model2_as_red_wins': 0,
    }

    # 3. 执行评估循环
    start_time = time.time()
    # 【优化】使用和训练器完全一致的 _play_one_game 函数
    for i in tqdm(range(num_groups), desc="正在进行镜像对局评估"):
        game_seed = int(time.time_ns() + i) % (2**32 - 1)

        # 第一局: 模型A执红 vs 模型B执黑
        winner_1 = _play_one_game(eval_env, red_player=model1, black_player=model2, seed=game_seed)
        if winner_1 == 1:
            scores['model1_wins'] += 1
            scores['model1_as_red_wins'] += 1
        elif winner_1 == -1:
            scores['model2_wins'] += 1
        else:
            scores['draws'] += 1

        # 第二局: 模型B执红 vs 模型A执黑 (镜像)
        winner_2 = _play_one_game(eval_env, red_player=model2, black_player=model1, seed=game_seed)
        if winner_2 == 1:
            scores['model2_wins'] += 1
            scores['model2_as_red_wins'] += 1
        elif winner_2 == -1:
            scores['model1_wins'] += 1
        else:
            scores['draws'] += 1
            
    eval_env.close()
    end_time = time.time()

    # 4. 计算并打印结果
    total_games = num_groups * 2
    total_decisive_games = scores['model1_wins'] + scores['model2_wins']
    win_rate_model1 = scores['model1_wins'] / total_decisive_games if total_decisive_games > 0 else 0.0

    print("\n" + "=" * 70)
    print("           📊 最终评估结果 📊")
    print("=" * 70)
    print(f"总计用时: {end_time - start_time:.2f} 秒")
    print(f"平均每局用时: {(end_time - start_time) / total_games:.2f} 秒\n")
    print(f"--- 总体战绩 (共 {total_games} 局) ---")
    print(f"  - {model1.name} 胜: {scores['model1_wins']} 局")
    print(f"  - {model2.name} 胜: {scores['model2_wins']} 局")
    print(f"  - 平局: {scores['draws']} 局\n")
    print(f"--- 胜率计算 (基于非平局) ---")
    print(f"  - {model1.name} 胜率: {win_rate_model1:.2%}")
    print(f"  - {model2.name} 胜率: {1.0 - win_rate_model1:.2%}\n")
    print(f"--- 先手（红方）表现分析 ---")
    print(f"  - {model1.name} 执红胜局: {scores['model1_as_red_wins']} / {num_groups} ({scores['model1_as_red_wins']/num_groups:.2%})")
    print(f"  - {model2.name} 执红胜局: {scores['model2_as_red_wins']} / {num_groups} ({scores['model2_as_red_wins']/num_groups:.2%})")
    print("=" * 70)


if __name__ == '__main__':
    # --- 在此配置您要评估的模型路径 ---
    MODEL_A_PATH = "./models/self_play_final/main_opponent.zip"
    MODEL_B_PATH = "./models/self_play_final/challenger.zip"
    
    # 【优化】对局组数现在由 constants.py 统一管理，此处无需配置
    # NUM_EVALUATION_GROUPS = 100

    evaluate_mirrored_matches(MODEL_A_PATH, MODEL_B_PATH)