# src_code/callbacks/self_play_callback.py

import os
import torch
import numpy as np
import json
from typing import Dict, Any, List
import logging

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from utils.constants import *
from utils import elo

logger = logging.getLogger(__name__)

class SelfPlayCallback(DefaultCallbacks):
    """
    处理自我对弈中的模型评估、Elo更新和对手池管理的核心回调 (已适配新版 API)。
    """
    def __init__(self):
        super().__init__()
        self.win_rates_buffer: Dict[str, list] = {}
        self.state_file_path = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        
        self.elo_ratings: Dict[str, float] = {MAIN_POLICY_ID: ELO_DEFAULT}
        self.model_generations: Dict[str, int] = {}
        self.latest_generation: int = 0
        self.long_term_pool_paths: List[str] = []
        self.short_term_pool_paths: List[str] = []
        self.long_term_power_of_2: int = 1

        os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        
        # 修复点：调用 _load_state 前，确保它已定义
        self._load_state()

    # 修复点：添加 _load_state 方法
    def _load_state(self):
        """从状态文件加载 Elo 评级和模型生成数据。"""
        if os.path.exists(self.state_file_path):
            with open(self.state_file_path, 'r') as f:
                state = json.load(f)
                self.elo_ratings = state.get("elo_ratings", {MAIN_POLICY_ID: ELO_DEFAULT})
                self.model_generations = state.get("model_generations", {})
                self.latest_generation = state.get("latest_generation", 0)
                self.long_term_pool_paths = state.get("long_term_pool_paths", [])
                self.short_term_pool_paths = state.get("short_term_pool_paths", [])
                self.long_term_power_of_2 = state.get("long_term_power_of_2", 1)
                logger.info(f"成功从 {self.state_file_path} 加载了 Elo 状态。")
        else:
            logger.info("未找到 Elo 状态文件，将使用默认值初始化。")

    # 修复点：添加 _save_state 方法
    def _save_state(self):
        """保存当前的 Elo 评级和模型生成数据。"""
        state = {
            "elo_ratings": self.elo_ratings,
            "model_generations": self.model_generations,
            "latest_generation": self.latest_generation,
            "long_term_pool_paths": self.long_term_pool_paths,
            "short_term_pool_paths": self.short_term_pool_paths,
            "long_term_power_of_2": self.long_term_power_of_2,
        }
        with open(self.state_file_path, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Elo 状态已成功保存到 {self.state_file_path}。")

    def _get_opponent_sampling_distribution(self) -> Dict[str, float]:
        """根据Elo差异计算对手的采样概率。"""
        main_elo = self.elo_ratings.get(MAIN_POLICY_ID, ELO_DEFAULT)
        
        weights = {}
        weights[MAIN_POLICY_ID] = 1.0 
        
        all_opponents = self.long_term_pool_paths + self.short_term_pool_paths
        for opp_name in all_opponents:
            opp_policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{opp_name.replace('.pt', '')}"
            opp_elo = self.elo_ratings.get(opp_policy_id, ELO_DEFAULT)
            weight = np.exp(-abs(main_elo - opp_elo) / ELO_WEIGHT_TEMPERATURE)
            weights[opp_policy_id] = weight

        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {MAIN_POLICY_ID: 1.0}

    def _update_sampler_on_workers(self, algorithm: Algorithm):
        """计算最新的采样分布并将其广播到所有 EnvRunner。"""
        dist = self._get_opponent_sampling_distribution()
        
        def set_sampler(worker: RolloutWorker):
            worker.sampler_dist = dist
        
        algorithm.env_runner_group.foreach_env_runner(set_sampler)
        logger.info("已将最新的对手采样分布广播到所有 EnvRunners。")

    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs):
        """在算法初始化时，动态添加所有现有的对手模块。"""
        logger.info("--- Callback: on_algorithm_init ---")
        
        all_opponents = self.long_term_pool_paths + self.short_term_pool_paths
        main_module = algorithm.get_module(MAIN_POLICY_ID)

        for opp_name in all_opponents:
            module_id = f"{OPPONENT_POLICY_ID_PREFIX}{opp_name.replace('.pt', '')}"
            if not algorithm.get_module(module_id):
                 algorithm.add_module(
                    module_id=module_id,
                    module_spec=RLModuleSpec.from_module(main_module),
                )

        def setup_worker_modules(worker: RolloutWorker):
            for opp_name in all_opponents:
                module_id = f"{OPPONENT_POLICY_ID_PREFIX}{opp_name.replace('.pt', '')}"
                model_path = os.path.join(OPPONENT_POOL_DIR, opp_name)
                
                module = worker.module.get(module_id)
                if module and os.path.exists(model_path):
                    try:
                        state_dict = torch.load(model_path, map_location=module.device)
                        module.load_state_dict(state_dict)
                        logger.info(f"Worker {worker.worker_index}: 成功加载模块 {module_id}")
                    except Exception as e:
                        logger.error(f"Worker {worker.worker_index}: 加载模型 {model_path} 失败: {e}")

        algorithm.env_runner_group.foreach_env_runner(setup_worker_modules)
        self._update_sampler_on_workers(algorithm)

    def on_episode_end(
        self, *, worker: "RolloutWorker", base_env: BaseEnv,
        episode: EpisodeV2, env_index: int, **kwargs,
    ):
        """在新版 API 中，使用 EpisodeV2 对象。"""
        main_agent_id, opponent_agent_id, opponent_module_id = None, None, None

        for agent_id in episode.agent_ids:
            module_id = episode.module_for(agent_id)
            if module_id == MAIN_POLICY_ID:
                main_agent_id = agent_id
            else:
                opponent_agent_id = agent_id
                opponent_module_id = module_id
        
        if not all([main_agent_id, opponent_agent_id, opponent_module_id]): return

        last_info = episode.last_info_for(main_agent_id) or {}
        winner = last_info.get("winner")
        if winner is None: return

        env = base_env.get_sub_environments()[env_index]
        main_player_num = env._agent_to_player_map.get(main_agent_id)
        if main_player_num is None: return

        if opponent_module_id not in self.win_rates_buffer:
            self.win_rates_buffer[opponent_module_id] = []
        
        if winner == 0: self.win_rates_buffer[opponent_module_id].append(0.5)
        elif winner == main_player_num: self.win_rates_buffer[opponent_module_id].append(1.0)
        else: self.win_rates_buffer[opponent_module_id].append(0.0)

    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs):
        """评估和更新逻辑，适配新 API。"""
        total_games_played = sum(len(res) for res in self.win_rates_buffer.values())
        if total_games_played < EVALUATION_GAMES:
            return

        logger.info("\n--- 评估周期结束，开始处理结果 ---")
        main_module = algorithm.get_module(MAIN_POLICY_ID)
        
        all_results = []
        for opponent, results in self.win_rates_buffer.items():
            win_rate = np.mean(results)
            logger.info(f"  - vs {opponent}: Win Rate = {win_rate:.2f} ({len(results)} games)")
            
            main_elo = self.elo_ratings.get(MAIN_POLICY_ID, ELO_DEFAULT)
            opponent_elo = self.elo_ratings.get(opponent, ELO_DEFAULT)
            
            new_main_elo, new_opponent_elo = elo.rate_1vs1(main_elo, opponent_elo, win_rate)
            self.elo_ratings[MAIN_POLICY_ID] = new_main_elo
            self.elo_ratings[opponent] = new_opponent_elo
            
            logger.info(f"    Elo: {main_elo:.0f} -> {new_main_elo:.0f} | Opponent Elo: {opponent_elo:.0f} -> {new_opponent_elo:.0f}")
            all_results.extend(results)

        overall_win_rate = np.mean(all_results) if all_results else 0.0
        result["overall_win_rate"] = overall_win_rate
        logger.info(f"\n  综合胜率: {overall_win_rate:.2f}")

        if overall_win_rate > EVALUATION_THRESHOLD:
            logger.info(f"🏆 挑战成功! 新主宰者诞生！")
            self.latest_generation += 1
            
            new_opponent_name = f"challenger_{self.latest_generation}.pt"
            new_opponent_path = os.path.join(OPPONENT_POOL_DIR, new_opponent_name)
            torch.save(main_module.state_dict(), new_opponent_path)
            
            new_opponent_module_id = f"{OPPONENT_POLICY_ID_PREFIX}{new_opponent_name.replace('.pt', '')}"
            self.elo_ratings[new_opponent_module_id] = self.elo_ratings.get(MAIN_POLICY_ID, ELO_DEFAULT)
            self.model_generations[new_opponent_module_id] = self.latest_generation

            logger.info(f"  - 动态添加新模块 {new_opponent_module_id} 到训练器中...")
            algorithm.add_module(
                module_id=new_opponent_module_id,
                module_spec=RLModuleSpec.from_module(main_module),
            )
            
            def set_module_trainable(learner):
                if new_opponent_module_id in learner.module:
                    learner.remove_module_to_update(new_opponent_module_id)

            algorithm.learner_group.foreach_learner(set_module_trainable)
            
            self._manage_opponent_pool(new_opponent_name, algorithm)
            self._update_sampler_on_workers(algorithm)
        else:
            logger.info(f"🛡️ 挑战失败。主策略将继续训练。")

        self.win_rates_buffer.clear()
        self._save_state()

    # 修复点：添加 _manage_opponent_pool 的完整逻辑
    def _manage_opponent_pool(self, new_opponent_name: str, algorithm: Algorithm):
        """管理短期和长期对手池。"""
        self.short_term_pool_paths.append(new_opponent_name)
        if len(self.short_term_pool_paths) > SHORT_TERM_POOL_MAX_SIZE:
            to_remove_name = self.short_term_pool_paths.pop(0)
            module_id_to_remove = f"{OPPONENT_POLICY_ID_PREFIX}{to_remove_name.replace('.pt', '')}"
            self._remove_module_from_algorithm(algorithm, module_id_to_remove, to_remove_name)

        if self.latest_generation >= self.long_term_power_of_2:
            self.long_term_pool_paths.append(new_opponent_name)
            self.long_term_power_of_2 *= 2
            logger.info(f"  - 新模型被提升到长期对手池。下一个提升点: 第 {self.long_term_power_of_2} 代。")

        if len(self.long_term_pool_paths) > LONG_TERM_POOL_MAX_SIZE:
            to_remove_name = self.long_term_pool_paths.pop(0)
            module_id_to_remove = f"{OPPONENT_POLICY_ID_PREFIX}{to_remove_name.replace('.pt', '')}"
            self._remove_module_from_algorithm(algorithm, module_id_to_remove, to_remove_name)

    # 修复点：添加 _remove_module_from_algorithm 的完整逻辑
    def _remove_module_from_algorithm(self, algorithm: Algorithm, module_id_to_remove: str, model_filename: str):
        """从算法中移除模块并删除相关文件。"""
        logger.info(f"✂️  清理过时对手模块: {module_id_to_remove}")
        try:
            if algorithm.get_module(module_id_to_remove):
                algorithm.remove_module(module_id_to_remove)
                logger.info(f"    - 成功从 RLlib 训练器中移除模块: {module_id_to_remove}")
            
            model_path = os.path.join(OPPONENT_POOL_DIR, model_filename)
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"    - 成功删除模型文件: {model_path}")

            if module_id_to_remove in self.elo_ratings:
                del self.elo_ratings[module_id_to_remove]
            if module_id_to_remove in self.model_generations:
                del self.model_generations[module_id_to_remove]
        except Exception as e:
            logger.warning(f"    - 移除模块 {module_id_to_remove} 时出错: {e}")