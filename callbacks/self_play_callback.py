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
        # 1. 包括主策略自身
        weights[MAIN_POLICY_ID] = 1.0 
        
        # 2. 包括所有池中对手
        all_opponents = self.long_term_pool_paths + self.short_term_pool_paths
        for opp_name in all_opponents:
            opp_policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{opp_name.replace('.pt', '')}"
            opp_elo = self.elo_ratings.get(opp_policy_id, ELO_DEFAULT)
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
        worker.foreach_policy(lambda p, pid: setattr(p, 'sampler_dist', dist))
        worker.sampler_dist = dist # 也为 worker 本身设置

    # --- RLlib 回调钩子 ---
    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs):
        """在算法初始化时，动态添加所有现有的对手策略。"""
        print("--- Callback: on_algorithm_init ---")
        all_opponents = self.long_term_pool_paths + self.short_term_pool_paths
        main_policy = algorithm.get_policy(MAIN_POLICY_ID)
        
        for opp_name in all_opponents:
            policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{opp_name.replace('.pt', '')}"
            print(f"  - 动态添加历史对手策略: {policy_id}")
            algorithm.add_policy(
                policy_id=policy_id,
                policy_cls=type(main_policy),
                policy_spec=PolicySpec(), # 使用默认 spec
            )
        
        # 在所有 worker 上加载权重并更新采样器
        algorithm.workers.foreach_worker(self._load_weights_and_update_sampler)
    
    def _load_weights_and_update_sampler(self, worker: RolloutWorker):
        """一个辅助函数，用于在 worker 上加载权重和更新采样器。"""
        all_opponents = self.long_term_pool_paths + self.short_term_pool_paths
        for opp_name in all_opponents:
            policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{opp_name.replace('.pt', '')}"
            model_path = os.path.join(OPPONENT_POOL_DIR, opp_name)
            if os.path.exists(model_path):
                policy = worker.get_policy(policy_id)
                if policy:
                    state_dict = torch.load(model_path, map_location=policy.device)
                    policy.model.load_state_dict(state_dict)
                    policy.lock_weights() # 确保对手策略不被训练
        
        # 更新该 worker 的采样分布
        self._update_sampler_on_workers(worker)

    def on_episode_end(
        self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
        episode: EpisodeV2, env_index: int, **kwargs,
    ):
        """在每局游戏结束时被调用，记录胜负结果。"""
        # ... (此部分逻辑不变) ...
        main_policy_agent_id = None
        opponent_agent_id = None
        for agent_id, policy_id in episode.policy_for.items():
            if policy_id == MAIN_POLICY_ID: main_policy_agent_id = agent_id
            else: opponent_agent_id = agent_id
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
        # ... (此部分逻辑基本不变, 但添加了更新 worker 的步骤) ...
        total_games_played = sum(len(res) for res in self.win_rates_buffer.values())
        if total_games_played < EVALUATION_GAMES: return

        print("\n--- 评估周期结束，开始处理结果 ---")
        main_policy = algorithm.get_policy(MAIN_POLICY_ID)
        
        # 1. 计算平均胜率并更新Elo
        challenger_wins = 0
        for opponent_id, results in self.win_rates_buffer.items():
            if not results: continue
            wins = sum(1 for r in results if r == 1.0)
            challenger_wins += wins
            avg_win_rate = np.mean(results)
            self.elo_ratings = elo.update_elo(self.elo_ratings, MAIN_POLICY_ID, opponent_id, avg_win_rate)
            print(f"  - vs {opponent_id:<30}: 胜率 = {avg_win_rate:.2%} ({wins}/{len(results)} 局)")

        # 2. 检查主策略是否满足晋级条件
        overall_win_rate = challenger_wins / total_games_played
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
                policy_spec=PolicySpec(),
                # 在所有 worker 上同步新策略并加载其权重
                workers=algorithm.workers
            )
            
            # 手动在本地 worker 设置权重，并广播到远程 workers
            new_policy_local = algorithm.get_policy(new_opponent_policy_id)
            new_policy_local.model.load_state_dict(main_policy.model.state_dict())
            new_policy_local.lock_weights()
            algorithm.workers.sync_weights(policies=[new_opponent_policy_id])

            # 在所有 worker 上更新采样分布
            algorithm.workers.foreach_worker(self._update_sampler_on_workers)
        else:
            print(f"\n🛡️  挑战失败 (总胜率 {overall_win_rate:.2%} <= {EVALUATION_THRESHOLD:.2%})。主策略将继续训练。")

        self.win_rates_buffer.clear()
        self._save_state()

    def _manage_opponent_pool(self, new_opponent_name: str, algorithm: Algorithm):
        """【从 trainer.py 移植】管理长期和短期对手池。"""
        # ... (此部分逻辑不变) ...
        print("\n--- 正在更新对手池 ---")
        self.latest_generation += 1
        self.model_generations[new_opponent_name] = self.latest_generation
        added_to_long_term = False
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
    
        current_pool_names = set(self.short_term_pool_paths + self.long_term_pool_paths)
        
        policies_on_workers = set(algorithm.workers.local_worker().policy_map.keys())
        policies_to_remove = []
        
        for pid in policies_on_workers:
            if pid.startswith(OPPONENT_POLICY_ID_PREFIX):
                model_name = pid.replace(OPPONENT_POLICY_ID_PREFIX, "") + ".pt"
                if model_name not in current_pool_names:
                    policies_to_remove.append(pid)

        for policy_id_to_remove in policies_to_remove:
            print(f"✂️  清理过时对手策略: {policy_id_to_remove}")
            try:
                algorithm.remove_policy(policy_id_to_remove, workers=algorithm.workers)
                print(f"    - 成功从RLlib中移除策略: {policy_id_to_remove}")
                model_filename = policy_id_to_remove.replace(OPPONENT_POLICY_ID_PREFIX, "") + ".pt"
                model_path = os.path.join(OPPONENT_POOL_DIR, model_filename)
                if os.path.exists(model_path):
                    os.remove(model_path)
                    print(f"    - 成功删除模型文件: {model_filename}")
                self.elo_ratings.pop(policy_id_to_remove, None)
                self.model_generations.pop(model_filename, None)
            except Exception as e:
                print(f"    - 警告: 移除策略 {policy_id_to_remove} 时出错: {e}")

        print("\n--- 对手池状态更新完毕 ---")
        print(f"短期池 ({len(self.short_term_pool_paths)}/{SHORT_TERM_POOL_SIZE}): {self.short_term_pool_paths}")
        print(f"长期池 ({len(self.long_term_pool_paths)}/{LONG_TERM_POOL_SIZE}): {self.long_term_pool_paths}")
        print(f"长期池代数差值指数: {self.long_term_power_of_2} (当前要求差值 >= {2**self.long_term_power_of_2})")