# rllib_version_complete/callbacks/self_play_callback.py

import os
import torch
import numpy as np
import json
from typing import Dict, Any, List

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import PolicyID
from ray.rllib.algorithms.algorithm import Algorithm

from utils.constants import *
from utils import elo

class SelfPlayCallback(DefaultCallbacks):
    """
    处理自我对弈中的模型评估、Elo更新和对手池管理的核心回调。
    【V2 更新】: 实现了基于模型代数的长短期对手池管理策略。
    """
    def __init__(self):
        super().__init__()
        # 存储主策略与每个对手的胜负关系: {opponent_id: [win, loss, draw, ...]}
        self.win_rates_buffer: Dict[str, list] = {}
        
        # --- 对手池核心属性 (从旧版 trainer.py 移植) ---
        self.elo_ratings: Dict[str, float] = {}
        self.model_generations: Dict[str, int] = {}
        self.latest_generation: int = 0
        self.long_term_pool_paths: List[str] = []
        self.short_term_pool_paths: List[str] = []
        self.long_term_power_of_2: int = 1
        
        # 启动时加载所有状态
        self._load_state()

    def _load_state(self):
        """从JSON文件加载Elo评分、模型代数和模型池状态。"""
        state_file = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.elo_ratings = data.get("elo", {})
                    self.model_generations = data.get("generations", {})
                    self.latest_generation = data.get("latest_generation", 0)
                    self.long_term_pool_paths = data.get("long_term_pool_paths", [])
                    self.short_term_pool_paths = data.get("short_term_pool_paths", [])
                    self.long_term_power_of_2 = data.get("long_term_power_of_2", 1)
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"警告：读取状态文件失败或格式不完整: {e}。将使用默认值。")
        else:
             print("未找到状态文件，将使用默认值初始化。")

    def _save_state(self):
        """将Elo、模型代数和模型池状态保存到同一个JSON文件。"""
        state_file = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        data = {
            "elo": self.elo_ratings,
            "generations": self.model_generations,
            "latest_generation": self.latest_generation,
            "long_term_pool_paths": self.long_term_pool_paths,
            "short_term_pool_paths": self.short_term_pool_paths,
            "long_term_power_of_2": self.long_term_power_of_2,
        }
        try:
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            print(f"错误：无法保存状态文件: {e}")

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
        main_policy_agent_id = None
        opponent_agent_id = None
        
        for agent_id, policy_id in episode.policy_for.items():
            if policy_id == MAIN_POLICY_ID:
                main_policy_agent_id = agent_id
            else:
                opponent_agent_id = agent_id
        
        if main_policy_agent_id and opponent_agent_id:
            winner = episode.last_info_for(main_policy_agent_id).get("winner")
            if winner is not None:
                env = base_env.get_sub_environments()[env_index]
                main_player_id = env._agent_to_player_map[main_policy_agent_id]
                opponent_policy_id = episode.policy_for[opponent_agent_id]

                if opponent_policy_id not in self.win_rates_buffer:
                    self.win_rates_buffer[opponent_policy_id] = []

                if winner == 0: self.win_rates_buffer[opponent_policy_id].append(0.5)
                elif winner == main_player_id: self.win_rates_buffer[opponent_policy_id].append(1.0)
                else: self.win_rates_buffer[opponent_policy_id].append(0.0)

    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs):
        """在每次 `algo.train()` 后被调用，执行评估和更新逻辑。"""
        main_policy = algorithm.get_policy(MAIN_POLICY_ID)
        total_games_played = sum(len(res) for res in self.win_rates_buffer.values())

        if total_games_played < EVALUATION_GAMES:
            return

        print("\n--- 评估周期结束，开始处理结果 ---")
        
        # --- 1. 计算平均胜率并更新Elo ---
        challenger_wins = 0
        for opponent_id, results in self.win_rates_buffer.items():
            if not results: continue
            
            wins = sum(1 for r in results if r == 1.0)
            challenger_wins += wins
            avg_win_rate = np.mean(results)
            
            self.elo_ratings = elo.update_elo(
                self.elo_ratings, MAIN_POLICY_ID, opponent_id, avg_win_rate
            )
            print(f"  - vs {opponent_id:<30}: "
                  f"胜率 = {avg_win_rate:.2%} ({wins}/{len(results)} 局)")

        # --- 2. 检查主策略是否满足晋级条件 ---
        overall_win_rate = challenger_wins / total_games_played
        
        if overall_win_rate > EVALUATION_THRESHOLD:
            print(f"\n🏆 挑战成功! (总胜率 {overall_win_rate:.2%} > {EVALUATION_THRESHOLD:.2%})！新主宰者诞生！")
            
            # a. 保存当前主策略模型到对手池
            new_opponent_name = f"challenger_{algorithm.iteration}.pt"
            new_opponent_path = os.path.join(OPPONENT_POOL_DIR, new_opponent_name)
            torch.save(main_policy.model.state_dict(), new_opponent_path)
            
            # b. 为新对手设置初始Elo
            new_opponent_policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{new_opponent_name.replace('.pt', '')}"
            self.elo_ratings[new_opponent_policy_id] = self.elo_ratings.get(MAIN_POLICY_ID, ELO_DEFAULT)
            print(f"  - 新对手 {new_opponent_name} 已存入池中，Elo设置为 {self.elo_ratings[new_opponent_policy_id]:.0f}")

            # c. 执行复杂的对手池管理
            self._manage_opponent_pool(new_opponent_name, algorithm)
            
            # d. 动态添加新策略到训练器
            new_policy_spec = PolicySpec()
            algorithm.add_policy(
                policy_id=new_opponent_policy_id,
                policy_cls=type(main_policy),
                policy_spec=new_policy_spec,
            )
            new_policy = algorithm.get_policy(new_opponent_policy_id)
            new_policy.model.load_state_dict(main_policy.model.state_dict())
            new_policy.lock_weights()
            print(f"  - 新策略 {new_opponent_policy_id} 已被动态添加到训练器中。")

        else:
            print(f"\n🛡️  挑战失败 (总胜率 {overall_win_rate:.2%} <= {EVALUATION_THRESHOLD:.2%})。主策略将继续训练。")

        # --- 3. 清空胜率记录器并保存状态 ---
        self.win_rates_buffer.clear()
        self._save_state()

    def _manage_opponent_pool(self, new_opponent_name: str, algorithm: Algorithm):
        """
        【从 trainer.py 移植】管理长期和短期对手池。
        """
        print("\n--- 正在更新对手池 ---")
        self.latest_generation += 1
        self.model_generations[new_opponent_name] = self.latest_generation
        
        added_to_long_term = False
        
        long_term_pool_with_gens = sorted(
            [(p, self.model_generations.get(p, 0)) for p in self.long_term_pool_paths],
            key=lambda x: x[1]
        )
        self.long_term_pool_paths = [p for p, _ in long_term_pool_with_gens]
        long_term_gens = [g for _, g in long_term_pool_with_gens]

        if not self.long_term_pool_paths:
            self.long_term_pool_paths.append(new_opponent_name)
            added_to_long_term = True
        else:
            required_gap = 2 ** self.long_term_power_of_2
            actual_gap = self.latest_generation - long_term_gens[-1]

            if actual_gap >= required_gap:
                if len(self.long_term_pool_paths) >= LONG_TERM_POOL_SIZE:
                    self.long_term_power_of_2 += 1
                    new_required_gap = 2 ** self.long_term_power_of_2
                    
                    retained_pool = [self.long_term_pool_paths[0]]
                    last_kept_gen = long_term_gens[0]
                    
                    for i in range(1, len(long_term_gens)):
                        if (long_term_gens[i] - last_kept_gen) >= new_required_gap:
                            retained_pool.append(self.long_term_pool_paths[i])
                            last_kept_gen = long_term_gens[i]
                    
                    self.long_term_pool_paths = retained_pool
                    
                    new_last_gen = self.model_generations.get(self.long_term_pool_paths[-1], 0)
                    if len(self.long_term_pool_paths) < LONG_TERM_POOL_SIZE and (self.latest_generation - new_last_gen) >= new_required_gap:
                        self.long_term_pool_paths.append(new_opponent_name)
                        added_to_long_term = True
                else:
                    self.long_term_pool_paths.append(new_opponent_name)
                    added_to_long_term = True

        if not added_to_long_term:
            self.short_term_pool_paths.append(new_opponent_name)
            self.short_term_pool_paths.sort(key=lambda p: self.model_generations.get(p, 0), reverse=True)
            if len(self.short_term_pool_paths) > SHORT_TERM_POOL_SIZE:
                self.short_term_pool_paths = self.short_term_pool_paths[:SHORT_TERM_POOL_SIZE]
    
        # --- 清理被淘汰的模型 (文件和RLlib策略) ---
        current_pool_names = set(self.short_term_pool_paths + self.long_term_pool_paths)
        
        # 遍历磁盘上的文件，找出需要删除的
        for filename in os.listdir(OPPONENT_POOL_DIR):
            if filename.endswith('.pt') and filename not in current_pool_names:
                print(f"✂️  清理过时对手: {filename}")
                
                # 1. 从RLlib训练器中移除策略
                policy_id_to_remove = f"{OPPONENT_POLICY_ID_PREFIX}{filename.replace('.pt', '')}"
                try:
                    if algorithm.get_policy(policy_id_to_remove):
                        algorithm.remove_policy(policy_id_to_remove)
                        print(f"    - 成功从RLlib中移除策略: {policy_id_to_remove}")
                except Exception as e:
                    print(f"    - 警告: 移除策略 {policy_id_to_remove} 时出错: {e}")

                # 2. 删除模型文件
                os.remove(os.path.join(OPPONENT_POOL_DIR, filename))
                
                # 3. 从状态记录中移除
                self.elo_ratings.pop(policy_id_to_remove, None)
                self.model_generations.pop(filename, None)
        
        print("\n--- 对手池状态更新完毕 ---")
        print(f"短期池 ({len(self.short_term_pool_paths)}/{SHORT_TERM_POOL_SIZE}): {self.short_term_pool_paths}")
        print(f"长期池 ({len(self.long_term_pool_paths)}/{LONG_TERM_POOL_SIZE}): {self.long_term_pool_paths}")
        print(f"长期池代数差值指数: {self.long_term_power_of_2} (当前要求差值 >= {2**self.long_term_power_of_2})")