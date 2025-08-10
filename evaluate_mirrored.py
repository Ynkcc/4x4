# evaluate_mirrored.py
"""
镜像对局评估脚本。

功能:
1. 加载两个指定的PPO模型。
2. 进行N组成对的镜像比赛。
3. 在每组比赛中：
    a. 生成一个固定的随机种子来初始化棋盘。
    b. 第一局：模型A执红（先手），模型B执黑。
    c. 第二局：使用完全相同的种子重置棋盘，模型B执红，模型A执黑。
4. 统计并报告详细的对战结果和胜率。
"""
import os
import warnings

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 禁用INFO和WARNING日志
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import os
import numpy as np
import time
from tqdm import tqdm  # 使用tqdm库来显示美观的进度条

# 导入本地模块
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from game.environment import GameEnvironment

class EvaluationAgent:
    """一个简单的包装类，用于在评估时加载和使用模型。"""
    def __init__(self, model_path: str):
        self.model = MaskablePPO.load(model_path, device='auto')
        self.name = os.path.basename(model_path)

    def predict(self, observation, action_masks, deterministic=True):
        action, _ = self.model.predict(
            observation,
            action_masks=action_masks,
            deterministic=deterministic
        )
        return int(action), None

def play_one_game(env: GameEnvironment, red_player: EvaluationAgent, black_player: EvaluationAgent, seed: int) -> int:
    """
    进行一局完整的游戏。

    :param env: 游戏环境实例。
    :param red_player: 控制红方（玩家1）的Agent。
    :param black_player: 控制黑方（玩家-1）的Agent。
    :param seed: 用于重置环境的随机种子。
    :return: 游戏结果 (1: 红方胜, -1: 黑方胜, 0: 平局)。
    """
    obs, info = env.reset(seed=seed)
    
    while True:
        current_player_agent = red_player if env.current_player == 1 else black_player
        
        action_mask = env.action_masks()
        if not np.any(action_mask):
            # 如果当前玩家无棋可走，则对手获胜
            return -env.current_player

        action, _ = current_player_agent.predict(obs, action_masks=action_mask)
        
        # 使用内部函数手动推进游戏，避免环境自动切换对手
        _, terminated, truncated, winner = env._internal_apply_action(action)
        
        if terminated or truncated:
            return winner
        
        # 手动切换玩家
        env.current_player *= -1
        obs = env.get_state()


def evaluate_mirrored_matches(model1_path: str, model2_path: str, num_groups: int = 100):
    """
    执行镜像对局评估。

    :param model1_path: 模型1的路径。
    :param model2_path: 模型2的路径。
    :param num_groups: 进行多少组镜像对局（每组2局）。
    """
    print("=" * 70)
    print("           ⚔️  镜像对局评估系统 ⚔️")
    print("=" * 70)

    # 1. 加载模型
    try:
        model1 = EvaluationAgent(model1_path)
        model2 = EvaluationAgent(model2_path)
        print(f"评测模型 A: {model1.name}")
        print(f"评测模型 B: {model2.name}")
        print(f"对局组数: {num_groups} (总计 {num_groups * 2} 局游戏)")
        print("-" * 70)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 2. 初始化环境和计分板
    # 注意：这里我们只创建单个环境实例，然后在循环中重复使用它
    eval_env = GameEnvironment()
    
    # 计分板 (从模型1的视角)
    scores = {
        'model1_wins': 0,
        'model2_wins': 0,
        'draws': 0,
        'model1_as_red_wins': 0,
        'model2_as_red_wins': 0,
    }

    # 3. 执行评估循环
    start_time = time.time()
    for i in tqdm(range(num_groups), desc="正在进行镜像对局评估"):
        # 为每组对局生成一个唯一的、固定的种子
        game_seed = int(time.time_ns() + i) % (2**32 - 1)

        # --- 第一局: 模型1执红 vs 模型2执黑 ---
        winner_1 = play_one_game(eval_env, red_player=model1, black_player=model2, seed=game_seed)
        if winner_1 == 1:
            scores['model1_wins'] += 1
            scores['model1_as_red_wins'] += 1
        elif winner_1 == -1:
            scores['model2_wins'] += 1
        else:
            scores['draws'] += 1

        # --- 第二局: 模型2执红 vs 模型1执黑 (镜像对局) ---
        winner_2 = play_one_game(eval_env, red_player=model2, black_player=model1, seed=game_seed)
        if winner_2 == 1: # 此时红方是模型2
            scores['model2_wins'] += 1
            scores['model2_as_red_wins'] += 1
        elif winner_2 == -1: # 此时黑方是模型1
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
    MODEL_B_PATH = "./models/self_play_final/opponent1.zip"
    
    # --- 配置对局组数 ---
    NUM_EVALUATION_GROUPS = 100

    evaluate_mirrored_matches(MODEL_A_PATH, MODEL_B_PATH, num_groups=NUM_EVALUATION_GROUPS)