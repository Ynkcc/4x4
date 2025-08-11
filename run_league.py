# src_code/run_league.py (内存优化版)

import os
import json
import itertools
import time
from tqdm import tqdm
import numpy as np
import multiprocessing
import sys

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 【重要】直接从 evaluator 导入核心组件，而不是整个函数
from training.evaluator import EvaluationAgent, _play_one_game
from game.environment import GameEnvironment
from utils.constants import (
    SELF_PLAY_OUTPUT_DIR, OPPONENT_POOL_DIR, MAIN_OPPONENT_PATH, EVALUATION_GAMES
)

# --- 配置 ---
DEFAULT_ELO = 1200
ELO_K_FACTOR = 16
# 使用 os.cpu_count() 来动态设置进程数，更具适应性
NUM_WORKERS = max(1, os.cpu_count() - 2) if os.cpu_count() else 1


# --- 并行工作进程专属函数 ---

# 定义一个在工作进程生命周期内持续存在的全局变量（缓存）
_worker_model_cache = {}

def get_model_from_cache(path: str) -> EvaluationAgent:
    """
    工作进程内部的助手函数，用于加载并缓存模型。
    每个进程都有自己的 _worker_model_cache。
    """
    if path not in _worker_model_cache:
        # 如果模型不在缓存中，则加载它并存入
        _worker_model_cache[path] = EvaluationAgent(path)
    return _worker_model_cache[path]

def evaluate_matchup_worker(matchup: tuple) -> tuple:
    """
    【新的工作函数】用于并行处理。
    它负责加载模型（利用缓存）并执行完整的镜像对局评估。
    """
    model_a_path, model_b_path = matchup
    
    # 1. 从缓存中获取模型
    model_a = get_model_from_cache(model_a_path)
    model_b = get_model_from_cache(model_b_path)
    
    # 2. 在工作进程内部创建独立的环境
    eval_env = GameEnvironment()
    
    # 3. 执行评估循环（逻辑从 evaluate_models 移到此处）
    scores = {'model_a_wins': 0, 'model_b_wins': 0, 'draws': 0}
    num_groups = EVALUATION_GAMES // 2

    for i in range(num_groups):
        game_seed = int(time.time_ns() + i) % (2**32 - 1)
        
        # A执红 vs B执黑
        winner_1 = _play_one_game(eval_env, red_player=model_a, black_player=model_b, seed=game_seed)
        if winner_1 == 1: scores['model_a_wins'] += 1
        elif winner_1 == -1: scores['model_b_wins'] += 1
        else: scores['draws'] += 1

        # B执红 vs A执黑 (镜像)
        winner_2 = _play_one_game(eval_env, red_player=model_b, black_player=model_a, seed=game_seed)
        if winner_2 == 1: scores['model_b_wins'] += 1
        elif winner_2 == -1: scores['model_a_wins'] += 1
        else: scores['draws'] += 1
    
    eval_env.close()

    # 4. 计算并返回结果
    total_games = scores['model_a_wins'] + scores['model_b_wins'] + scores['draws']
    win_rate_a = scores['model_a_wins'] / total_games if total_games > 0 else 0.5
    
    return model_a_path, model_b_path, win_rate_a

def run_league_tournament():
    """
    【内存优化版】执行一次完整的联赛，并行处理所有模型的相互比赛，并全局更新Elo评分。
    """
    print("=" * 70)
    print("       🏆 全模型循环联赛系统 (内存优化并行版) 🏆")
    print("=" * 70)

    # ... [步骤 1, 2, 3: 发现模型、加载Elo、生成对局组合] (这部分代码与之前相同) ...
    print("\n[步骤 1/5] 正在发现所有参赛模型...")
    model_paths = []
    if os.path.exists(MAIN_OPPONENT_PATH):
        model_paths.append(MAIN_OPPONENT_PATH)
    if os.path.exists(OPPONENT_POOL_DIR):
        for filename in sorted(os.listdir(OPPONENT_POOL_DIR)):
            if filename.endswith('.zip'):
                full_path = os.path.join(OPPONENT_POOL_DIR, filename)
                if full_path not in model_paths:
                    model_paths.append(full_path)
    
    model_names = [os.path.basename(p) for p in model_paths]
    if len(model_paths) < 2:
        print("错误：参赛模型少于2个，无法举办联赛。")
        return
    print(f"发现 {len(model_paths)} 个参赛模型。")
    
    print("\n[步骤 2/5] 正在加载Elo评分...")
    elo_file_path = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
    elo_ratings = {}
    if os.path.exists(elo_file_path):
        with open(elo_file_path, 'r') as f:
            elo_ratings = json.load(f)
    initial_elos = {name: elo_ratings.get(name, DEFAULT_ELO) for name in model_names}
    
    matchups = list(itertools.combinations(model_paths, 2))
    print(f"\n[步骤 3/5] 已生成 {len(matchups)} 场对局。")

    # --- 4. 并行执行循环赛 ---
    print(f"\n[步骤 4/5] 联赛开始！使用 {NUM_WORKERS} 个进程并行评估...")
    actual_scores = {name: 0.0 for name in model_names}
    
    # 使用进程池执行所有比赛
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # 使用 pool.imap_unordered 可能更快，因为它会立即处理完成的任务
        results_iterator = pool.imap_unordered(evaluate_matchup_worker, matchups)
        
        # 使用 tqdm 显示总体进度
        results = list(tqdm(results_iterator, total=len(matchups), desc="联赛进度"))

    # 处理结果
    for model_a_path, model_b_path, win_rate_a in results:
        model_a_name = os.path.basename(model_a_path)
        model_b_name = os.path.basename(model_b_path)
        # 每场比赛（2局镜像）的总分是1分，按胜率分配
        actual_scores[model_a_name] += win_rate_a
        actual_scores[model_b_name] += (1.0 - win_rate_a)

    # ... [步骤 5: 更新Elo评分] (这部分代码与之前相同) ...
    print("\n[步骤 5/5] 联赛结束，正在计算并更新Elo评分...")
    expected_scores = {name: 0.0 for name in model_names}
    for model_a_name, model_b_name in itertools.permutations(model_names, 2):
        expected_win_a = 1 / (1 + 10 ** ((initial_elos[model_b_name] - initial_elos[model_a_name]) / 400))
        expected_scores[model_a_name] += expected_win_a
    
    new_elos = {}
    print("\n--- Elo 变更详情 ---")
    print(f"{'模型名称':<25} | {'旧Elo':>8} | {'新Elo':>8} | {'变化':>8} | {'实际得分':>10} | {'期望得分':>10}")
    print("-" * 85)
    
    for name in sorted(model_names, key=lambda n: initial_elos.get(n, DEFAULT_ELO), reverse=True):
        old_elo = initial_elos.get(name, DEFAULT_ELO)
        # 每个模型会和其他 N-1 个模型比赛
        score_diff = actual_scores[name] - expected_scores[name]
        new_elo = old_elo + ELO_K_FACTOR * score_diff
        new_elos[name] = new_elo
        print(f"{name:<25} | {old_elo:>8.0f} | {new_elo:>8.0f} | {new_elo - old_elo:>+8.1f} | "
              f"{actual_scores[name]:>10.2f} | {expected_scores[name]:>10.2f}")
              
    elo_ratings.update(new_elos)
    with open(elo_file_path, 'w') as f:
        json.dump(elo_ratings, f, indent=4)
    print(f"\n✅ 成功将更新后的Elo评分保存至: {elo_file_path}")
    print("=" * 70, "\n           🎉 联赛圆满结束! 🎉\n", "=" * 70, sep='')


if __name__ == '__main__':
    # 在Windows和macOS上，'spawn'是更安全的多进程启动方法
    if sys.platform.startswith('win') or sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn', force=True)
    run_league_tournament()