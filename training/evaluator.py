# training/evaluator.py

import os
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO

# 导入环境和配置
from game.environment import GameEnvironment
from utils.constants import EVALUATION_GAMES, EVALUATION_N_ENVS

def evaluate_models(challenger_path: str, main_opponent_path: str) -> float:
    """
    在一次系列赛中评估挑战者模型对阵主宰者模型的表现。

    这个函数会创建一个临时的多进程环境，让两个模型对战指定的局数，
    然后计算并返回挑战者模型的胜率。

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
        # 挑战者将作为 player 1 (学习者)，主宰者将作为 player -1 (对手)。
        eval_env = make_vec_env(
            GameEnvironment,
            n_envs=EVALUATION_N_ENVS,
            vec_env_cls=SubprocVecEnv,  # 使用多进程环境以加速评估
            env_kwargs={
                'curriculum_stage': 4,  # 完整游戏模式
                'opponent_policy': main_opponent_path
            }
        )
        
        # 加载挑战者模型
        # 设置向后兼容性
        from utils.model_compatibility import setup_legacy_imports
        setup_legacy_imports()
            
        challenger_model = MaskablePPO.load(challenger_path, env=eval_env, device='auto')
        
        wins = 0
        draws = 0
        losses = 0
        games_played = 0
        
        obs = eval_env.reset()
        # 初始化时没有动作掩码，需要执行一步或使用默认掩码
        current_action_masks = None
        
        while games_played < EVALUATION_GAMES:
            # 挑战者模型根据观察进行确定性预测
            if current_action_masks is not None:
                action, _ = challenger_model.predict(obs, action_masks=current_action_masks, deterministic=True)
            else:
                action, _ = challenger_model.predict(obs, deterministic=True)
            
            # 环境执行动作
            obs, rewards, dones, infos = eval_env.step(action)
            
            # 提取动作掩码供下一步使用
            current_action_masks = [info.get('action_mask') for info in infos if 'action_mask' in info]
            if len(current_action_masks) == 0:
                current_action_masks = None
            
            # 环境执行动作
            obs, rewards, dones, infos = eval_env.step(action)
            
            for i, done in enumerate(dones):
                # 如果当前环境中的一局游戏结束
                if done:
                    games_played += 1
                    winner = infos[i].get('winner')
                    if winner == 1:  # 玩家1是挑战者
                        wins += 1
                    elif winner == -1: # 玩家-1是主宰者
                        losses += 1
                    else: # winner == 0 是平局
                        draws += 1
                    
                    # 提前结束循环，避免不必要的游戏
                    if games_played >= EVALUATION_GAMES:
                        break
        
        # 计算胜率
        win_rate = wins / games_played if games_played > 0 else 0.0
        
        print(f"--- 评估结束: 共进行了 {games_played} 局游戏 ---")
        print(f"    挑战者战绩: {wins}胜 / {losses}负 / {draws}平")
        print(f"    挑战者胜率: {win_rate:.2%}")
        
        return win_rate

    finally:
        # 确保环境在结束时被正确关闭
        if eval_env:
            eval_env.close()
