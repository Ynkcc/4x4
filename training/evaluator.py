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
        # 【注意】我们假设模型加载速度足够快，或者在评估期间可以接受这个开销
        # 对于超大规模评估，可能需要更复杂的模型服务化方案
        # 【潜在风险 1 修复】强制使用CPU加载评估模型，避免与训练过程抢占GPU资源
        self.model = MaskablePPO.load(model_path, device='cpu')
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
    此函数保持不变，因为它不涉及模型加载。
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

def evaluate_models(challenger_path: str, main_opponent_path: str, show_progress: bool = True) -> float:
    """
    【单线程版】评估函数。
    此函数现在主要用于非并行的评估任务，例如训练器中的挑战者评估。
    联赛评估将使用下面 run_league.py 中的新函数。
    """
    if EVALUATION_GAMES % 2 != 0:
        raise ValueError(f"EVALUATION_GAMES ({EVALUATION_GAMES}) 必须是偶数，才能进行镜像对局。")

    challenger_name = os.path.basename(challenger_path)
    main_name = os.path.basename(main_opponent_path)
    
    if show_progress:
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
    
    progress_iterator = range(num_groups)
    if show_progress:
        progress_iterator = tqdm(range(num_groups), desc="镜像评估进度", leave=False)
    
    for i in progress_iterator:
        game_seed = int(time.time_ns() + i) % (2**32 - 1)
        winner_1 = _play_one_game(eval_env, red_player=challenger, black_player=main_opponent, seed=game_seed)
        if winner_1 == 1: scores['challenger_wins'] += 1
        elif winner_1 == -1: scores['opponent_wins'] += 1
        else: scores['draws'] += 1
        
        winner_2 = _play_one_game(eval_env, red_player=main_opponent, black_player=challenger, seed=game_seed)
        if winner_2 == 1: scores['opponent_wins'] += 1
        elif winner_2 == -1: scores['challenger_wins'] += 1
        else: scores['draws'] += 1
            
    eval_env.close()

    total_games = scores['challenger_wins'] + scores['opponent_wins'] + scores['draws']
    win_rate = scores['challenger_wins'] / total_games if total_games > 0 else 0.0

    if show_progress:
        print(f"\n--- 📊 评估结束: 共进行了 {EVALUATION_GAMES} 局游戏 ---")
        print(f"    挑战者战绩: {scores['challenger_wins']}胜 / {scores['opponent_wins']}负 / {scores['draws']}平")
        print(f"    挑战者胜率 (胜 / (胜+负+平)): {win_rate:.2%}")
    
    return win_rate