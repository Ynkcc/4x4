# src_code/run_league.py

import os
import json
import itertools
from tqdm import tqdm
import numpy as np

# 确保在运行此脚本时，项目根目录在Python路径中
import sys
# 将 'src_code' 的父目录添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 导入本地模块
from utils.constants import SELF_PLAY_OUTPUT_DIR, OPPONENT_POOL_DIR, MAIN_OPPONENT_PATH
from training.evaluator import evaluate_models

# --- 配置 ---
DEFAULT_ELO = 1200
# 在联赛更新中，K因子通常会设得小一些，因为数据量更大
ELO_K_FACTOR = 16 

def run_league_tournament():
    """
    执行一次完整的联赛，让所有模型互相比赛，并全局更新Elo评分。
    """
    print("=" * 70)
    print("           🏆 全模型循环联赛系统 🏆")
    print("=" * 70)

    # --- 1. 发现并加载所有参赛模型 ---
    print("\n[步骤 1/5] 正在发现所有参赛模型...")
    
    model_paths = []
    model_names = []

    # 添加主宰者模型
    if os.path.exists(MAIN_OPPONENT_PATH):
        model_paths.append(MAIN_OPPONENT_PATH)
        model_names.append(os.path.basename(MAIN_OPPONENT_PATH))
    
    # 添加对手池中的所有模型
    if os.path.exists(OPPONENT_POOL_DIR):
        for filename in os.listdir(OPPONENT_POOL_DIR):
            if filename.endswith('.zip'):
                full_path = os.path.join(OPPONENT_POOL_DIR, filename)
                model_paths.append(full_path)
                model_names.append(filename)

    if len(model_paths) < 2:
        print("错误：参赛模型少于2个，无法举办联赛。")
        return

    print(f"发现 {len(model_paths)} 个参赛模型: {', '.join(model_names)}")


    # --- 2. 加载现有Elo评分 ---
    print("\n[步骤 2/5] 正在加载Elo评分...")
    elo_ratings = {}
    elo_file_path = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
    if os.path.exists(elo_file_path):
        try:
            with open(elo_file_path, 'r') as f:
                elo_ratings = json.load(f)
            print(f"已从 {elo_file_path} 加载Elo数据。")
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告：读取Elo文件失败: {e}。将使用默认值。")

    # 为所有模型确保Elo记录存在
    initial_elos = {}
    for name in model_names:
        initial_elos[name] = elo_ratings.get(name, DEFAULT_ELO)
    
    print("初始Elo评分:")
    for name, elo in initial_elos.items():
        print(f"  - {name}: {elo:.0f}")


    # --- 3. 生成所有对局组合 ---
    matchups = list(itertools.combinations(model_paths, 2))
    num_matches = len(matchups)
    print(f"\n[步骤 3/5] 已生成 {num_matches} 场对局。")

    
    # --- 4. 执行循环赛 ---
    print("\n[步骤 4/5] 联赛开始！正在进行评估...")
    
    # 记录每个模型在联赛中的得分（胜1分，平0.5分，负0分）
    actual_scores = {name: 0.0 for name in model_names}
    
    for model_a_path, model_b_path in tqdm(matchups, desc="联赛进度"):
        model_a_name = os.path.basename(model_a_path)
        model_b_name = os.path.basename(model_b_path)
        
        # evaluate_models 返回模型A的胜率
        win_rate_a = evaluate_models(model_a_path, model_b_path)
        
        # 在两局镜像对局中，A的得分是 win_rate_a * 2
        # （例如，A赢1输1，胜率0.5，得分1；A赢2，胜率1.0，得分2）
        # 这里我们简化为直接使用胜率作为得分比例
        actual_scores[model_a_name] += win_rate_a
        actual_scores[model_b_name] += (1.0 - win_rate_a)


    # --- 5. 全局更新Elo评分 ---
    print("\n[步骤 5/5] 联赛结束，正在计算并更新Elo评分...")

    # 计算期望得分
    expected_scores = {name: 0.0 for name in model_names}
    for model_a_name in model_names:
        for model_b_name in model_names:
            if model_a_name == model_b_name:
                continue
            
            elo_a = initial_elos[model_a_name]
            elo_b = initial_elos[model_b_name]
            
            # 模型A对阵模型B的期望胜率
            expected_win_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
            expected_scores[model_a_name] += expected_win_a

    # 计算新的Elo
    new_elos = {}
    print("\n--- Elo 变更详情 ---")
    print(f"{'模型名称':<25} | {'旧Elo':>8} | {'新Elo':>8} | {'变化':>8} | {'实际得分':>10} | {'期望得分':>10}")
    print("-" * 85)

    sorted_models = sorted(model_names, key=lambda n: initial_elos[n], reverse=True)

    for name in sorted_models:
        old_elo = initial_elos[name]
        # K因子需要乘以比赛场数的一半，因为每个模型都和 N-1 个对手比赛
        # 但我们这里简化处理，使用一个固定的K因子
        score_diff = actual_scores[name] - expected_scores[name]
        new_elo = old_elo + ELO_K_FACTOR * score_diff
        new_elos[name] = new_elo
        
        print(f"{name:<25} | {old_elo:>8.0f} | {new_elo:>8.0f} | {new_elo - old_elo:>+8.1f} | "
              f"{actual_scores[name]:>10.2f} | {expected_scores[name]:>10.2f}")

    # --- 保存最终结果 ---
    try:
        # 合并新旧elo，只更新参赛模型的
        elo_ratings.update(new_elos)
        with open(elo_file_path, 'w') as f:
            json.dump(elo_ratings, f, indent=4)
        print(f"\n✅ 成功将更新后的Elo评分保存至: {elo_file_path}")
    except IOError as e:
        print(f"\n❌ 错误：无法保存Elo评分文件: {e}")

    print("=" * 70)
    print("           🎉 联赛圆满结束! 🎉")
    print("=" * 70)


if __name__ == '__main__':
    run_league_tournament()