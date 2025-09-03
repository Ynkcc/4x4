# rllib_version_complete/callbacks/self_play_callback.py

import os
import torch
import numpy as np
from typing import Dict

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import PolicyID

from utils.constants import *
from utils import elo

class SelfPlayCallback(DefaultCallbacks):
    """
    处理自我对弈中的模型评估、Elo更新和对手池管理的核心回调。
    """
    def __init__(self):
        super().__init__()
        # 存储主策略与每个对手的胜负关系: {opponent_id: [win, loss, draw, ...]}
        self.win_rates_buffer: Dict[str, list] = {}
        self.elo_ratings = elo.load_elo_ratings()

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        """在每局游戏结束时被调用，记录胜负结果。"""
        # 确定主策略和对手策略分别控制哪个agent
        main_policy_agent_id = None
        opponent_agent_id = None
        
        for agent_id, policy_id in episode.policy_for.items():
            if policy_id == MAIN_POLICY_ID:
                main_policy_agent_id = agent_id
            else:
                opponent_agent_id = agent_id
        
        # 确保这是一场主策略 vs 其他策略的对局
        if main_policy_agent_id and opponent_agent_id:
            winner = episode.last_info_for(main_policy_agent_id).get("winner")
            if winner is not None:
                # 获取环境实例以查询agent与player的映射关系
                env = base_env.get_sub_environments()[env_index]
                main_player_id = env._agent_to_player_map[main_policy_agent_id]
                
                opponent_policy_id = episode.policy_for[opponent_agent_id]

                if opponent_policy_id not in self.win_rates_buffer:
                    self.win_rates_buffer[opponent_policy_id] = []

                # 记录结果：1.0代表胜利, 0.5代表平局, 0.0代表失败
                if winner == 0:
                    self.win_rates_buffer[opponent_policy_id].append(0.5)
                elif winner == main_player_id:
                    self.win_rates_buffer[opponent_policy_id].append(1.0)
                else:
                    self.win_rates_buffer[opponent_policy_id].append(0.0)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """在每次 `algo.train()` 后被调用，执行评估和更新逻辑。"""
        main_policy = algorithm.get_policy(MAIN_POLICY_ID)
        
        # --- 1. 计算平均胜率并更新Elo ---
        total_games_played = 0
        challenger_wins = 0

        print("\n--- 评估与Elo更新 ---")
        for opponent_id, results in self.win_rates_buffer.items():
            if not results:
                continue
            
            num_games = len(results)
            total_games_played += num_games
            wins = sum(1 for r in results if r == 1.0)
            challenger_wins += wins
            
            avg_win_rate = np.mean(results)
            self.elo_ratings = elo.update_elo(
                self.elo_ratings, MAIN_POLICY_ID, opponent_id, avg_win_rate
            )
            print(f"  - vs {opponent_id:<30}: "
                  f"胜率 = {avg_win_rate:.2%} ({wins}/{num_games} 局)")

        # --- 2. 检查主策略是否满足晋级条件 ---
        if total_games_played >= EVALUATION_GAMES:
            overall_win_rate = challenger_wins / total_games_played if total_games_played > 0 else 0.0
            
            if overall_win_rate > EVALUATION_THRESHOLD:
                print(f"\n🏆 挑战成功! (总胜率 {overall_win_rate:.2%} > {EVALUATION_THRESHOLD:.2%})！新主宰者诞生！")
                
                # a. 将当前主策略模型存入对手池
                # 注意：保存的是state_dict，而不是整个模型
                new_opponent_name = f"challenger_{algorithm.iteration}.pt"
                new_opponent_path = os.path.join(OPPONENT_POOL_DIR, new_opponent_name)
                torch.save(main_policy.model.state_dict(), new_opponent_path)

                # b. 为新对手设置Elo
                new_opponent_policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{new_opponent_name.replace('.pt', '')}"
                self.elo_ratings[new_opponent_policy_id] = self.elo_ratings.get(MAIN_POLICY_ID, ELO_DEFAULT)
                print(f"  - 新对手 {new_opponent_name} 已存入池中，Elo设置为 {self.elo_ratings[new_opponent_policy_id]:.0f}")

                # c. 动态地将新策略添加到正在运行的算法中
                # 这允许在不重启训练的情况下引入新对手
                new_policy_spec = PolicySpec()
                algorithm.add_policy(
                    policy_id=new_opponent_policy_id,
                    policy_cls=type(main_policy),
                    policy_spec=new_policy_spec,
                )
                
                # d. 为新添加的策略加载权重并锁定
                new_policy = algorithm.get_policy(new_opponent_policy_id)
                new_policy.model.load_state_dict(main_policy.model.state_dict())
                new_policy.lock_weights() # 确保新加入的对手不被训练
                
                print(f"  - 新策略 {new_opponent_policy_id} 已被动态添加到训练器中。")

                # e. 清空胜率记录器，以便下一轮评估
                self.win_rates_buffer.clear()
        
        # --- 3. 保存更新后的Elo评分 ---
        elo.save_elo_ratings(self.elo_ratings)
