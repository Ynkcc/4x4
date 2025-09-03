# rl_code/rllib_version/scripts/evaluate.py

import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入RLlib和项目模块
import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from core.environment import GameEnvironment
from core.policy import RLLibCustomNetwork
from utils.constants import *

# 注册自定义环境和模型
register_env("dark_chess_env", lambda config: GameEnvironment())
ModelCatalog.register_custom_model("custom_torch_model", RLLibCustomNetwork)


class EvaluationAgent:
    """用于评估的AI代理包装器"""
    def __init__(self, policy: Policy, name: str):
        self.policy = policy
        self.name = name

    def predict(self, observation: dict, deterministic: bool = True) -> int:
        """使用策略进行预测"""
        # RLlib 策略需要一个 observation_space 字典
        # 我们的环境的 observation 已经是字典格式，所以直接传入
        action, _, _ = self.policy.compute_single_action(
            obs=observation,
            deterministic=deterministic
        )
        return int(action)

def play_one_game(env: GameEnvironment, red_player: EvaluationAgent, black_player: EvaluationAgent, seed: int) -> int:
    """进行一局完整的游戏，返回获胜方 (1 for red, -1 for black, 0 for draw)"""
    env.reset(seed=seed)
    
    # 强制红方先手
    env.current_player = 1
    
    while True:
        current_player_agent = red_player if env.current_player == 1 else black_player
        
        obs = env.get_state()
        action_mask = env.action_masks()
        obs['action_mask'] = action_mask # 将掩码添加到观察中
        
        if not np.any(action_mask):
            return -env.current_player # 当前玩家无棋可走，对方获胜

        action = current_player_agent.predict(obs)
        _, terminated, truncated, winner = env._internal_apply_action(action)
        
        if terminated or truncated:
            return winner if winner is not None else 0

        env.current_player *= -1

def find_latest_checkpoint(directory: str) -> str | None:
    """在指定目录中查找最新的RLlib检查点。"""
    try:
        # PPO_xxxx 目录
        ppo_dirs = [os.path.join(directory, d) for d in os.listdir(directory) if d.startswith("PPO_")]
        if not ppo_dirs: return None
        latest_experiment = max(ppo_dirs, key=os.path.getmtime)
        
        # checkpoint_xxxx 目录
        checkpoint_dirs = [os.path.join(latest_experiment, d) for d in os.listdir(latest_experiment) if d.startswith("checkpoint_")]
        if not checkpoint_dirs: return None
        return max(checkpoint_dirs, key=os.path.getmtime)
    except FileNotFoundError:
        return None

def evaluate_models(model1_checkpoint_path: str, model2_checkpoint_path: str):
    """
    执行镜像对局评估。
    """
    if EVALUATION_GAMES % 2 != 0:
        raise ValueError(f"EVALUATION_GAMES ({EVALUATION_GAMES}) 必须是偶数，才能进行完美的镜像对局。")
    num_groups = EVALUATION_GAMES // 2

    print("=" * 70)
    print("           ⚔️  RLlib 模型镜像对局评估系统 ⚔️")
    print("=" * 70)

    try:
        # 从检查点恢复策略
        policy1 = Policy.from_checkpoint(model1_checkpoint_path)
        policy2 = Policy.from_checkpoint(model2_checkpoint_path)
        
        agent1 = EvaluationAgent(policy1, os.path.basename(os.path.dirname(model1_checkpoint_path)))
        agent2 = EvaluationAgent(policy2, os.path.basename(os.path.dirname(model2_checkpoint_path)))
        
        print(f"评测模型 A: {agent1.name}")
        print(f"评测模型 B: {agent2.name}")
        print(f"对局组数: {num_groups} (总计 {EVALUATION_GAMES} 局游戏)")
        print("-" * 70)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return

    eval_env = GameEnvironment()
    scores = {
        'model1_wins': 0, 'model2_wins': 0, 'draws': 0,
        'model1_as_red_wins': 0, 'model2_as_red_wins': 0,
    }

    start_time = time.time()
    for i in tqdm(range(num_groups), desc="正在进行镜像对局评估"):
        game_seed = int(time.time_ns() + i) % (2**32 - 1)

        # 第一局: 模型A执红 vs 模型B执黑
        winner_1 = play_one_game(eval_env, red_player=agent1, black_player=agent2, seed=game_seed)
        if winner_1 == 1:
            scores['model1_wins'] += 1
            scores['model1_as_red_wins'] += 1
        elif winner_1 == -1:
            scores['model2_wins'] += 1
        else:
            scores['draws'] += 1

        # 第二局: 模型B执红 vs 模型A执黑 (镜像)
        winner_2 = play_one_game(eval_env, red_player=agent2, black_player=agent1, seed=game_seed)
        if winner_2 == 1:
            scores['model2_wins'] += 1
            scores['model2_as_red_wins'] += 1
        elif winner_2 == -1:
            scores['model1_wins'] += 1
        else:
            scores['draws'] += 1
            
    eval_env.close()
    end_time = time.time()

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
    # 初始化 Ray
    ray.init(local_mode=True) # 在本地模式下运行，方便调试

    # --- 配置要评估的模型检查点 ---
    # 脚本会自动查找最新的检查点
    # 如果要指定特定模型，请直接提供检查点目录的完整路径
    
    # 示例1: 评估最新的两个训练运行
    # MODEL_A_CHECKPOINT = find_latest_checkpoint(TENSORBOARD_LOG_PATH) # 通常是主宰者
    # MODEL_B_CHECKPOINT = find_latest_checkpoint(...) # 可能是另一个分支的模型
    
    # 示例2: 手动指定路径
    # 注意: 路径必须指向 checkpoint_xxxxx 目录，而不是PPO_...目录
    MODEL_A_CHECKPOINT = "/path/to/your/project/tensorboard_logs/self_play_final/PPO_dark_chess_multi_agent_.../checkpoint_000100"
    MODEL_B_CHECKPOINT = "/path/to/your/project/tensorboard_logs/self_play_final/PPO_dark_chess_multi_agent_.../checkpoint_000090"
    
    print("请注意：请在脚本中修改 MODEL_A_CHECKPOINT 和 MODEL_B_CHECKPOINT 的路径。")
    
    # if MODEL_A_CHECKPOINT and MODEL_B_CHECKPOINT:
    #     evaluate_models(MODEL_A_CHECKPOINT, MODEL_B_CHECKPOINT)
    # else:
    #     print("错误: 未能找到有效的模型检查点路径，请检查 TENSORBOARD_LOG_PATH 或手动指定路径。")

    ray.shutdown()