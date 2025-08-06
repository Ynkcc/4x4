# training/evaluator.py

import os
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
# 在评估时，DummyVecEnv 更简单且不易出错，特别是对于状态重置的管理
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

# 导入环境和配置
from game.environment import GameEnvironment, ACTION_SPACE_SIZE
from utils.constants import EVALUATION_GAMES, EVALUATION_N_ENVS
from utils.model_compatibility import setup_legacy_imports

def evaluate_models(challenger_path: str, main_opponent_path: str) -> float:
    """
    【最终修正版】在一次系列赛中评估挑战者模型对阵主宰者模型的表现。
    此版本采用了最健壮的动作掩码同步逻辑。

    :param challenger_path: 挑战者模型的 .zip 文件路径。
    :param main_opponent_path: 主宰者（当前最强）模型的 .zip 文件路径。
    :return: 挑战者模型的胜率 (0.0 到 1.0 之间)。
    """
    challenger_name = os.path.basename(challenger_path)
    main_name = os.path.basename(main_opponent_path)
    print(f"\n---  Elo评估开始: (挑战者) {challenger_name} vs (主宰者) {main_name} ---")
    
    eval_env = None
    try:
        # 创建一个专门用于评估的向量化环境。
        setup_legacy_imports()
        eval_env = make_vec_env(
            GameEnvironment,
            n_envs=EVALUATION_N_ENVS,
            vec_env_cls=DummyVecEnv,
            env_kwargs={
                'curriculum_stage': 4,
                'opponent_policy': main_opponent_path
            }
        )
        
        challenger_model = MaskablePPO.load(challenger_path, env=eval_env, device='auto')
        
        games_played = 0
        wins = 0
        draws = 0
        losses = 0
        
        # 初始化所有并行环境
        obs = eval_env.reset()
        
        # 持续进行游戏，直到完成指定的总局数
        while games_played < EVALUATION_GAMES:
            # 【最终修正核心】: 不再从 info 字典中解析掩码。
            # 而是每次都在模型预测前，直接从环境中获取与当前 obs 完全同步的掩码。
            # 这是最可靠的方法，能完全避免状态异步问题。
            action_masks = np.array(eval_env.env_method("action_masks"), dtype=np.int32)

            # 模型为所有并行环境做出决策
            action, _ = challenger_model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # 环境执行动作
            obs, rewards, dones, infos = eval_env.step(action)
            
            # 检查每个并行环境是否结束
            for i, done in enumerate(dones):
                if done:
                    games_played += 1
                    # 从被 Monitor 包装器扁平化后的 info 字典中直接获取游戏结果
                    winner = infos[i].get('winner')
                    if winner == 1:
                        wins += 1
                    elif winner == -1:
                        losses += 1
                    else:
                        draws += 1
                    
                    # 使用 end="\r" 实现单行动态更新，使日志更整洁
                    print(f"  评估游戏 {games_played}/{EVALUATION_GAMES} 结束。 "
                          f"当前累计战绩: {wins}胜 / {losses}负 / {draws}平", end="\r")

                    # 如果已经完成了所有评估局数，提前退出循环
                    if games_played >= EVALUATION_GAMES:
                        break
        
        print() # 换行，避免最终的统计信息覆盖上面的进度条
        total_games_for_winrate = wins + losses
        win_rate = wins / total_games_for_winrate if total_games_for_winrate > 0 else 0.0
        
        print(f"--- 评估结束: 共进行了 {games_played} 局游戏 ---")
        print(f"    挑战者战绩: {wins}胜 / {losses}负 / {draws}平")
        print(f"    挑战者胜率 (胜 / (胜+负)): {win_rate:.2%}")
        
        return win_rate

    finally:
        if eval_env:
            eval_env.close()