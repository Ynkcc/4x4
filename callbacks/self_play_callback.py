# rllib_version_complete/callbacks/self_play_callback.py

import os
import torch
import numpy as np
import json
from typing import Dict, Any, List
import random

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
    处理自我对弈中的模型评估、Elo更新和对手池管理的核心回调 (重构版)。
    职责:
    - 统一管理所有自我对弈状态 (Elo, 模型池, 代数)。
    - 在启动时动态加载对手策略。
    - 在训练过程中更新 Elo 和模型池。
    - 动态更新并广播对手采样分布。
    """
    def __init__(self):
        super().__init__()
        self.win_rates_buffer: Dict[str, list] = {}
        self.elo_ratings: Dict[str, float] = {}
        self.model_generations: Dict[str, int] = {}
        self.latest_generation: int = 0
        self.long_term_pool_paths: List[str] = []
        self.short_term_pool_paths: List[str] = []
        self.long_term_power_of_2: int = 1
        
        # 确保目录存在
        os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        self._load_state()

    # --- 状态管理 ---
    def _load_state(self):
        """从JSON文件加载Elo、模型代数和模型池状态。"""
        state_file = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.elo_ratings = data.get("elo", {MAIN_POLICY_ID: ELO_DEFAULT})
                    self.model_generations = data.get("generations", {})
                    self.latest_generation = data.get("latest_generation", 0)
                    self.long_term_pool_paths = data.get("long_term_pool_paths", [])
                    self.short_term_pool_paths = data.get("short_term_pool_paths", [])
                    self.long_term_power_of_2 = data.get("long_term_power_of_2", 1)
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"警告：读取状态文件失败或格式不完整: {e}。将使用默认值。")
        else:
             self.elo_ratings = {MAIN_POLICY_ID: ELO_DEFAULT}
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

    # --- 对手采样分布 ---
    def _get_opponent_sampling_distribution(self) -> Dict[str, float]:
        """根据Elo差异计算对手的采样概率。"""
        main_elo = self.elo_ratings.get(MAIN_POLICY_ID, ELO_DEFAULT)
        
        weights = {}
        # 1. 包括主策略自身 (用于自我对弈)
        weights[MAIN_POLICY_ID] = 1.0 
        
        # 2. 包括所有池中对手
        all_opponents = self.long_term_pool_paths + self.short_term_pool_paths
        for opp_name in all_opponents:
            opp_policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{opp_name.replace('.pt', '')}"
            opp_elo = self.elo_ratings.get(opp_policy_id, ELO_DEFAULT)
            # 使用温度参数调整采样权重
            weight = np.exp(-abs(main_elo - opp_elo) / ELO_WEIGHT_TEMPERATURE)
            weights[opp_policy_id] = weight

        total_weight = sum(weights.values())
        if total_weight == 0:
             return {MAIN_POLICY_ID: 1.0}
        return {k: v / total_weight for k, v in weights.items()}
    
    def _update_sampler_on_workers(self, worker: RolloutWorker):
        """计算并更新所有 worker 的采样分布。"""
        dist = self._get_opponent_sampling_distribution()
        # 将分布注入到每个 worker，供 policy_mapping_fn 使用
        # setattr(worker, 'sampler_dist', dist)
        worker.sampler_dist = dist

    # --- RLlib 回调钩子 ---
    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs):
        """在算法初始化时，动态添加所有现有的对手策略。"""
        print("--- Callback: on_algorithm_init ---")
        
        def setup_worker(worker: RolloutWorker):
            """在每个 worker 上执行的设置函数"""
            all_opponents = self.long_term_pool_paths + self.short_term_pool_paths
            main_policy = worker.get_policy(MAIN_POLICY_ID)
            
            for opp_name in all_opponents:
                policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{opp_name.replace('.pt', '')}"
                model_path = os.path.join(OPPONENT_POOL_DIR, opp_name)

                if policy_id not in worker.policy_map:
                    print(f"  - Worker {worker.worker_index}: 动态添加历史对手策略: {policy_id}")
                    worker.add_policy(
                        policy_id=policy_id,
                        policy_cls=type(main_policy),
                        policy_spec=PolicySpec(),
                    )

                if os.path.exists(model_path):
                    policy = worker.get_policy(policy_id)
                    if policy:
                        try:
                            state_dict = torch.load(model_path, map_location=policy.device)
                            policy.model.load_state_dict(state_dict)
                            policy.lock_weights() # 确保对手策略不被训练
                        except Exception as e:
                            print(f"  - Worker {worker.worker_index}: 加载模型 {model_path} 失败: {e}")

            # 更新该 worker 的采样分布
            self._update_sampler_on_workers(worker)

        # 在所有 worker 上执行设置
        algorithm.workers.foreach_worker(setup_worker)


    def on_episode_end(
        self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
        episode: EpisodeV2, env_index: int, **kwargs,
    ):
        """在每局游戏结束时被调用，记录胜负结果。"""
        # 确定主策略和对手策略的 agent_id
        main_agent_id, opponent_agent_id = None, None
        opponent_policy_id = None

        for agent_id, policy_id in episode.policy_for.items():
            if policy_id == MAIN_POLICY_ID:
                main_agent_id = agent_id
            else:
                opponent_agent_id = agent_id
                opponent_policy_id = policy_id
        
        # 确保双方都参与了游戏
        if not all([main_agent_id, opponent_agent_id, opponent_policy_id]):
            return

        # 从 info 字典中获取胜利者信息
        # RLLibMultiAgentEnv 会为两个 agent 都提供 info
        last_info = episode.last_info_for(main_agent_id) or episode.last_info_for(opponent_agent_id)
        if not last_info:
            return
            
        winner = last_info.get("winner")
        if winner is None:
            return

        # 获取环境实例以映射 agent_id 到 player number (1 or -1)
        env = base_env.get_sub_environments()[env_index]
        main_player_num = env._agent_to_player_map.get(main_agent_id)

        if main_player_num is None:
            return

        # 记录胜率
        if opponent_policy_id not in self.win_rates_buffer:
            self.win_rates_buffer[opponent_policy_id] = []
        
        if winner == 0: # 平局
            self.win_rates_buffer[opponent_policy_id].append(0.5)
        elif winner == main_player_num: # 主策略获胜
            self.win_rates_buffer[opponent_policy_id].append(1.0)
        else: # 主策略失败
            self.win_rates_buffer[opponent_policy_id].append(0.0)


    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs):
        """在每次 `algo.train()` 后被调用，执行评估和更新逻辑。"""
        total_games_played = sum(len(res) for res in self.win_rates_buffer.values())
        if total_games_played < EVALUATION_GAMES:
            return

        print("\n--- 评估周期结束，开始处理结果 ---")
        main_policy = algorithm.get_policy(MAIN_POLICY_ID)
        
        # 1. 计算平均胜率并更新Elo
        challenger_total_wins = 0
        challenger_total_games = 0
        for opponent_id, results in self.win_rates_buffer.items():
            if not results: continue
            
            wins = sum(r for r in results if r == 1.0)
            num_games = len(results)
            challenger_total_wins += wins
            challenger_total_games += num_games
            
            avg_win_rate = np.mean(results)
            self.elo_ratings = elo.update_elo(self.elo_ratings, MAIN_POLICY_ID, opponent_id, avg_win_rate)
            print(f"  - vs {opponent_id:<30}: 胜率 = {avg_win_rate:.2%} ({int(wins)}/{num_games} 局)")

        # 2. 检查主策略是否满足晋级条件
        if challenger_total_games == 0:
             overall_win_rate = 0
        else:
             overall_win_rate = challenger_total_wins / challenger_total_games

        if overall_win_rate > EVALUATION_THRESHOLD:
            print(f"\n🏆 挑战成功! (总胜率 {overall_win_rate:.2%} > {EVALUATION_THRESHOLD:.2%})！新主宰者诞生！")
            
            new_opponent_name = f"challenger_{algorithm.iteration}.pt"
            new_opponent_path = os.path.join(OPPONENT_POOL_DIR, new_opponent_name)
            torch.save(main_policy.model.state_dict(), new_opponent_path)
            
            new_opponent_policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{new_opponent_name.replace('.pt', '')}"
            self.elo_ratings[new_opponent_policy_id] = self.elo_ratings.get(MAIN_POLICY_ID, ELO_DEFAULT)
            print(f"  - 新对手 {new_opponent_name} 已存入池中，Elo设置为 {self.elo_ratings[new_opponent_policy_id]:.0f}")

            self._manage_opponent_pool(new_opponent_name, algorithm)
            
            # 动态添加新策略到训练器
            print(f"  - 动态添加新策略 {new_opponent_policy_id} 到训练器中...")
            algorithm.add_policy(
                policy_id=new_opponent_policy_id,
                policy_cls=type(main_policy),
                # 从主策略克隆权重
                weights=main_policy.get_weights(),
                policy_state=main_policy.get_state(),
            )
            
            # 锁定新对手策略的权重并更新采样分布
            def setup_new_opponent(worker: RolloutWorker):
                if worker.get_policy(new_opponent_policy_id):
                    worker.get_policy(new_opponent_policy_id).lock_weights()
                self._update_sampler_on_workers(worker)

            algorithm.workers.foreach_worker(setup_new_opponent)
        else:
            print(f"\n🛡️  挑战失败 (总胜率 {overall_win_rate:.2%} <= {EVALUATION_THRESHOLD:.2%})。主策略将继续训练。")

        self.win_rates_buffer.clear()
        self._save_state()

    def _manage_opponent_pool(self, new_opponent_name: str, algorithm: Algorithm):
        """管理长期和短期对手池。"""
        print("\n--- 正在更新对手池 ---")
        self.latest_generation += 1
        self.model_generations[new_opponent_name] = self.latest_generation
        added_to_long_term = False

        # --- 更新长期池 ---
        long_term_pool_with_gens = sorted([(p, self.model_generations.get(p, 0)) for p in self.long_term_pool_paths], key=lambda x: x[1])
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
                    # 池已满，需要根据新的代数差距要求进行筛选
                    self.long_term_power_of_2 += 1
                    new_required_gap = 2 ** self.long_term_power_of_2
                    retained_pool = [self.long_term_pool_paths[0]]
                    last_kept_gen = long_term_gens[0]
                    for i in range(1, len(long_term_gens)):
                        if (long_term_gens[i] - last_kept_gen) >= new_required_gap:
                            retained_pool.append(self.long_term_pool_paths[i])
                            last_kept_gen = long_term_gens[i]
                    self.long_term_pool_paths = retained_pool

                # 在筛选后检查是否还有空间
                if len(self.long_term_pool_paths) < LONG_TERM_POOL_SIZE:
                    self.long_term_pool_paths.append(new_opponent_name)
                    added_to_long_term = True
            
        # --- 更新短期池 ---
        if not added_to_long_term:
            self.short_term_pool_paths.append(new_opponent_name)
            self.short_term_pool_paths.sort(key=lambda p: self.model_generations.get(p, 0), reverse=True)
            if len(self.short_term_pool_paths) > SHORT_TERM_POOL_SIZE:
                # 移除最旧的模型
                removed_model_name = self.short_term_pool_paths.pop()
                policy_id_to_remove = f"{OPPONENT_POLICY_ID_PREFIX}{removed_model_name.replace('.pt', '')}"
                self._remove_policy_and_files(algorithm, policy_id_to_remove, removed_model_name)
        
        # --- 清理不再使用的策略 ---
        current_pool_names = set(self.short_term_pool_paths + self.long_term_pool_paths)
        
        # 获取一个 worker 上的策略列表作为参考
        policies_on_workers = set(algorithm.workers.local_worker().policy_map.keys())
        
        for pid in policies_on_workers:
            if pid.startswith(OPPONENT_POLICY_ID_PREFIX):
                model_name = pid.replace(OPPONENT_POLICY_ID_PREFIX, "") + ".pt"
                if model_name not in current_pool_names:
                    self._remove_policy_and_files(algorithm, pid, model_name)

        print("\n--- 对手池状态更新完毕 ---")
        print(f"短期池 ({len(self.short_term_pool_paths)}/{SHORT_TERM_POOL_SIZE}): {self.short_term_pool_paths}")
        print(f"长期池 ({len(self.long_term_pool_paths)}/{LONG_TERM_POOL_SIZE}): {self.long_term_pool_paths}")
        print(f"长期池代数差值指数: {self.long_term_power_of_2} (当前要求差值 >= {2**self.long_term_power_of_2})")

    def _remove_policy_and_files(self, algorithm: Algorithm, policy_id_to_remove: str, model_filename: str):
        """从算法中移除策略并删除相关文件。"""
        print(f"✂️  清理过时对手策略: {policy_id_to_remove}")
        try:
            if algorithm.workers.local_worker().has_policy(policy_id_to_remove):
                algorithm.remove_policy(policy_id_to_remove, workers=algorithm.workers)
                print(f"    - 成功从RLlib中移除策略: {policy_id_to_remove}")
            
            # 删除模型文件
            model_path = os.path.join(OPPONENT_POOL_DIR, model_filename)
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"    - 成功删除模型文件: {model_filename}")
            
            # 从状态字典中移除
            self.elo_ratings.pop(policy_id_to_remove, None)
            self.model_generations.pop(model_filename, None)
            
        except Exception as e:
            print(f"    - 警告: 移除策略 {policy_id_to_remove} 时出错: {e}")