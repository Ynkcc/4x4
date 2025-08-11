# training/evaluator.py

import os
import time
import numpy as np
from tqdm import tqdm

# 导入本地模块
from sb3_contrib import MaskablePPO
from game.environment import GameEnvironment
from utils.constants import EVALUATION_GAMES

class EvaluationAgent:
    """一个简单的包装类，用于在评估时加载和使用模型。"""
    def __init__(self, model_path: str):
        self.model = MaskablePPO.load(model_path, device='auto')
        self.name = os.path.basename(model_path)

    def predict(self, observation, action_masks, deterministic=True):
        """使用加载的模型进行预测。"""
        action, _ = self.model.predict(
            observation,
            action_masks=action_masks,
            deterministic=deterministic
        )
        return int(action), None

def _play_one_game(env: GameEnvironment, red_player: EvaluationAgent, black_player: EvaluationAgent, seed: int) -> int:
    """
    进行一局完整的游戏。

    Args:
        env: 游戏环境实例。
        red_player: 控制红方（玩家1）的Agent。
        black_player: 控制黑方（玩家-1）的Agent。
        seed: 用于重置环境的随机种子，以确保棋盘布局相同。

    Returns:
        游戏结果 (1: 红方胜, -1: 黑方胜, 0: 平局)。
    """
    obs, info = env.reset(seed=seed)
    
    while True:
        current_player_agent = red_player if env.current_player == 1 else black_player
        action_mask = env.action_masks()
        
        if not np.any(action_mask):
            return -env.current_player

        action, _ = current_player_agent.predict(obs, action_masks=action_mask)
        _, terminated, truncated, winner = env._internal_apply_action(action)
        
        if terminated or truncated:
            return winner
        
        env.current_player *= -1
        obs = env.get_state()

def evaluate_models(challenger_path: str, main_opponent_path: str) -> float:
    """
    【镜像对局评估版】评估挑战者模型对阵主宰者模型的表现。
    通过进行成对的镜像比赛来获得更公平、更可信的胜率。

    Args:
        challenger_path: 挑战者模型的 .zip 文件路径。
        main_opponent_path: 主宰者（当前最强）模型的 .zip 文件路径。

    Returns:
        挑战者模型的胜率 (范围: 0.0 到 1.0)。
    """
    # 【优化】增加对评估局数的校验
    if EVALUATION_GAMES % 2 != 0:
        raise ValueError(f"EVALUATION_GAMES ({EVALUATION_GAMES}) 必须是偶数，才能进行完美的镜像对局。")

    challenger_name = os.path.basename(challenger_path)
    main_name = os.path.basename(main_opponent_path)
    print(f"\n---  ⚔️  镜像对局评估开始: (挑战者) {challenger_name} vs (主宰者) {main_name} ---")

    try:
        challenger = EvaluationAgent(challenger_path)
        main_opponent = EvaluationAgent(main_opponent_path)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return 0.0

    eval_env = GameEnvironment()
    num_groups = EVALUATION_GAMES // 2
    
    scores = {'challenger_wins': 0, 'opponent_wins': 0, 'draws': 0}

    print(f"将进行 {num_groups} 组镜像对局 (总计 {EVALUATION_GAMES} 局游戏)")
    
    for i in tqdm(range(num_groups), desc="镜像评估进度"):
        game_seed = int(time.time_ns() + i) % (2**32 - 1)

        # 第一局: 挑战者执红 vs 主宰者执黑
        winner_1 = _play_one_game(eval_env, red_player=challenger, black_player=main_opponent, seed=game_seed)
        if winner_1 == 1:
            scores['challenger_wins'] += 1
        elif winner_1 == -1:
            scores['opponent_wins'] += 1
        else:
            scores['draws'] += 1

        # 第二局: 主宰者执红 vs 挑战者执黑 (镜像)
        winner_2 = _play_one_game(eval_env, red_player=main_opponent, black_player=challenger, seed=game_seed)
        if winner_2 == 1:
            scores['opponent_wins'] += 1
        elif winner_2 == -1:
            scores['challenger_wins'] += 1
        else:
            scores['draws'] += 1
            
    eval_env.close()

    total_decisive_games = scores['challenger_wins'] + scores['opponent_wins']
    win_rate = scores['challenger_wins'] / total_decisive_games if total_decisive_games > 0 else 0.0

    print(f"\n--- 📊 评估结束: 共进行了 {EVALUATION_GAMES} 局游戏 ---")
    print(f"    挑战者战绩: {scores['challenger_wins']}胜 / {scores['opponent_wins']}负 / {scores['draws']}平")
    print(f"    挑战者胜率 (胜 / (胜+负)): {win_rate:.2%}")
    
    return win_rate