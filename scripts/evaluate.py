# rl_code/rllib_version/scripts/evaluate.py

import os
import sys
import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray.rllib.algorithms.algorithm import Algorithm # 导入 Algorithm
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

# 假设你的环境和常量定义在这里
from core.environment import DarkChessEnv # 改为 MultiAgent 环境的底层
from core.multi_agent_env import RLLibMultiAgentEnv
from utils.constants import *

# 注册环境
register_env("dark_chess_env", lambda config: RLLibMultiAgentEnv(config))

class EvaluationAgent:
    """用于评估的AI代理包装器"""
    def __init__(self, policy: Policy, name: str):
        self.policy = policy
        self.name = name
        # RLModule API 使用不同的方法
        self.is_rl_module = not hasattr(self.policy, 'compute_single_action')

    def predict(self, observation: dict, deterministic: bool = True) -> int:
        if self.is_rl_module:
            # 新版 RLModule API
            import torch
            # RLModule 需要一个批次维度
            obs_tensor = {k: torch.from_numpy(np.expand_dims(v, axis=0)) for k, v in observation.items()}
            
            if deterministic:
                action_dist_inputs = self.policy.forward_inference({"obs": obs_tensor})[Columns.ACTION_DIST_INPUTS]
                action = torch.argmax(action_dist_inputs, dim=1).item()
            else:
                action_dist_inputs = self.policy.forward_exploration({"obs": obs_tensor})[Columns.ACTION_DIST_INPUTS]
                dist_class = self.policy.get_exploration_action_dist_cls()
                action_dist = dist_class.from_logits(action_dist_inputs)
                action = action_dist.sample().item()
        else:
            # 旧版 ModelV2 API
            action, _, _ = self.policy.compute_single_action(
                obs=observation,
                deterministic=deterministic
            )
        return int(action)

def play_one_game(env: DarkChessEnv, red_player: EvaluationAgent, black_player: EvaluationAgent, seed: int) -> int:
    """进行一局完整的游戏，返回获胜方 (1 for red, -1 for black, 0 for draw)"""
    env.reset(seed=seed)
    
    while True:
        current_player_agent = red_player if env.current_player == 1 else black_player
        
        # obs 字典需要包含 action_mask
        obs = env.get_state()
        action_mask = env.action_masks()
        obs['action_mask'] = action_mask
        
        if not np.any(action_mask):
            return -env.current_player

        action = current_player_agent.predict(obs)
        _, terminated, truncated, winner = env.step(action)
        
        if terminated or truncated:
            return winner if winner is not None else 0


def find_latest_checkpoint_from_experiment(experiment_path: str) -> str | None:
    """在指定的实验目录中查找最新的检查点。"""
    try:
        checkpoint_dirs = [
            os.path.join(experiment_path, d)
            for d in os.listdir(experiment_path)
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(experiment_path, d))
        ]
        if not checkpoint_dirs:
            return None
        return max(checkpoint_dirs, key=os.path.getmtime)
    except FileNotFoundError:
        return None

def evaluate_models(model1_checkpoint_path: str, model2_checkpoint_path: str):
    """执行镜像对局评估。"""
    if EVALUATION_GAMES % 2 != 0:
        raise ValueError(f"EVALUATION_GAMES ({EVALUATION_GAMES}) 必须是偶数，才能进行完美的镜像对局。")
    num_groups = EVALUATION_GAMES // 2

    print("=" * 70)
    print("           ⚔️  RLlib 模型镜像对局评估系统 ⚔️")
    print("=" * 70)

    try:
        # --- 标准化模型加载 ---
        print(f"正在从检查点恢复算法 A: {model1_checkpoint_path}")
        algo1 = Algorithm.from_checkpoint(model1_checkpoint_path)
        print(f"正在从检查点恢复算法 B: {model2_checkpoint_path}")
        algo2 = Algorithm.from_checkpoint(model2_checkpoint_path)
        
        # --- 获取主策略 ---
        policy1 = algo1.get_policy(MAIN_POLICY_ID)
        policy2 = algo2.get_policy(MAIN_POLICY_ID)
        
        agent1 = EvaluationAgent(policy1, os.path.basename(os.path.dirname(model1_checkpoint_path)))
        agent2 = EvaluationAgent(policy2, os.path.basename(os.path.dirname(model2_checkpoint_path)))
        
        print(f"\n评测模型 A: {agent1.name}")
        print(f"评测模型 B: {agent2.name}")
        print(f"对局组数: {num_groups} (总计 {EVALUATION_GAMES} 局游戏)")
        print("-" * 70)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 使用底层的单智能体环境进行评估
    eval_env = DarkChessEnv()
    scores = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0, 'model1_as_red_wins': 0, 'model2_as_red_wins': 0}

    start_time = time.time()
    for i in tqdm(range(num_groups), desc="正在进行镜像对局评估"):
        game_seed = int(time.time() * 1000 + i) % (2**32 - 1)

        winner_1 = play_one_game(eval_env, red_player=agent1, black_player=agent2, seed=game_seed)
        if winner_1 == 1: scores['model1_wins'] += 1; scores['model1_as_red_wins'] += 1
        elif winner_1 == -1: scores['model2_wins'] += 1
        else: scores['draws'] += 1

        winner_2 = play_one_game(eval_env, red_player=agent2, black_player=agent1, seed=game_seed)
        if winner_2 == 1: scores['model2_wins'] += 1; scores['model2_as_red_wins'] += 1
        elif winner_2 == -1: scores['model1_wins'] += 1
        else: scores['draws'] += 1
            
    eval_env.close()
    end_time = time.time()
    
    # ... [结果打印部分保持不变] ...
    total_games = num_groups * 2
    total_decisive_games = scores['model1_wins'] + scores['model2_wins']
    win_rate_model1 = scores['model1_wins'] / total_decisive_games if total_decisive_games > 0 else 0.0

    print("\n" + "=" * 70)
    print("           📊 最终评估结果 📊")
    print("=" * 70)
    print(f"总计用时: {end_time - start_time:.2f} 秒")
    print(f"平均每局用时: {(end_time - start_time) / total_games:.2f} 秒\n")
    print(f"--- 总体战绩 (共 {total_games} 局) ---")
    print(f"  - {agent1.name} 胜: {scores['model1_wins']} 局")
    print(f"  - {agent2.name} 胜: {scores['model2_wins']} 局")
    print(f"  - 平局: {scores['draws']} 局\n")
    print(f"--- 胜率计算 (基于非平局) ---")
    print(f"  - {agent1.name} 胜率: {win_rate_model1:.2%}")
    print(f"  - {agent2.name} 胜率: {1.0 - win_rate_model1:.2%}\n")
    print(f"--- 先手（红方）表现分析 ---")
    print(f"  - {agent1.name} 执红胜局: {scores['model1_as_red_wins']} / {num_groups} ({scores['model1_as_red_wins']/num_groups:.2%})")
    print(f"  - {agent2.name} 执红胜局: {scores['model2_as_red_wins']} / {num_groups} ({scores['model2_as_red_wins']/num_groups:.2%})")
    print("=" * 70)


if __name__ == '__main__':
    ray.init(local_mode=True)

    # --- 配置要评估的模型检查点 ---
    # 【重要】请提供实验目录的路径，脚本会自动查找最新的检查点
    # 例如: .../tensorboard_logs/self_play_final/PPO_self_play_experiment_.../
    
    # 示例:
    MODEL_A_EXPERIMENT_PATH = "/path/to/your/project/tensorboard_logs/self_play_final/PPO_self_play_experiment_2023-10-27_10-00-00_.../"
    MODEL_B_EXPERIMENT_PATH = "/path/to/your/project/tensorboard_logs/self_play_final/PPO_self_play_experiment_2023-10-26_18-00-00_.../"

    print("请注意：请在脚本中修改 MODEL_A_EXPERIMENT_PATH 和 MODEL_B_EXPERIMENT_PATH 的路径。")
    
    # checkpoint_a = find_latest_checkpoint_from_experiment(MODEL_A_EXPERIMENT_PATH)
    # checkpoint_b = find_latest_checkpoint_from_experiment(MODEL_B_EXPERIMENT_PATH)
    
    # if checkpoint_a and checkpoint_b:
    #     evaluate_models(checkpoint_a, checkpoint_b)
    # else:
    #     print("错误: 未能在指定的实验目录中找到有效的模型检查点路径。")

    ray.shutdown()