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
# 添加新版API的import
from ray.rllib.env.env_runner import EnvRunner
try:
    from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
except ImportError:
    MultiAgentEpisode = None

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
        
        self._load_state()

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
        # [鲁棒性修复] 始终确保主策略（自我对弈）在采样池中
        weights[MAIN_POLICY_ID] = 1.0 
        
        all_opponents = self.long_term_pool_paths + self.short_term_pool_paths
        for opp_name in all_opponents:
            # 确保文件名和策略ID正确对应
            opp_policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{os.path.splitext(opp_name)[0]}"
            opp_elo = self.elo_ratings.get(opp_policy_id, ELO_DEFAULT)
            weight = np.exp(-abs(main_elo - opp_elo) / ELO_WEIGHT_TEMPERATURE)
            weights[opp_policy_id] = weight

        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {MAIN_POLICY_ID: 1.0}

    def _update_sampler_on_workers(self, algorithm: Algorithm):
        """计算最新的采样分布并将其广播到所有 EnvRunner。"""
        dist = self._get_opponent_sampling_distribution()
        
        def set_sampler(env_runner):
            # 将采样分布附加到 env_runner 对象上，以便 policy_mapping_fn 访问
            env_runner.sampler_dist = dist
            # 兼容旧版API
            if hasattr(env_runner, 'worker_index'):
                logger.debug(f"EnvRunner {env_runner.worker_index} sampler updated: {dist}")
            else:
                logger.debug(f"EnvRunner sampler updated: {dist}")
        
        # 使用 foreach_env_runner 确保在所有 rollout worker 上设置
        algorithm.env_runner_group.foreach_env_runner(set_sampler)
        logger.info("已将最新的对手采样分布广播到所有 EnvRunners。")

    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs):
        """在算法初始化时，动态添加所有现有的对手模块。"""
        logger.info("--- Callback: on_algorithm_init ---")
        
        all_opponents = self.long_term_pool_paths + self.short_term_pool_paths
        main_module = algorithm.get_module(MAIN_POLICY_ID)

        for opp_name in all_opponents:
            module_id = f"{OPPONENT_POLICY_ID_PREFIX}{os.path.splitext(opp_name)[0]}"
            # 只有当模块不存在时才添加
            if not algorithm.get_module(module_id):
                 logger.info(f"动态添加历史对手模块: {module_id}")
                 algorithm.add_module(
                    module_id=module_id,
                    module_spec=RLModuleSpec.from_module(main_module),
                )

        def setup_worker_modules(env_runner):
            # [鲁棒性修复] 确保 env_runner 上有 sampler_dist 属性
            if not hasattr(env_runner, "sampler_dist"):
                env_runner.sampler_dist = {}

            for opp_name in all_opponents:
                module_id = f"{OPPONENT_POLICY_ID_PREFIX}{os.path.splitext(opp_name)[0]}"
                model_path = os.path.join(OPPONENT_POOL_DIR, opp_name)
                
                module = env_runner.module.get(module_id)
                if module and os.path.exists(model_path):
                    try:
                        # 确保在正确的设备上加载模型
                        state_dict = torch.load(model_path, map_location=module.device)
                        module.load_state_dict(state_dict)
                        # 兼容旧版API
                        if hasattr(env_runner, 'worker_index'):
                            logger.info(f"EnvRunner {env_runner.worker_index}: 成功加载模块 {module_id}")
                        else:
                            logger.info(f"EnvRunner: 成功加载模块 {module_id}")
                    except Exception as e:
                        # 兼容旧版API
                        if hasattr(env_runner, 'worker_index'):
                            logger.error(f"EnvRunner {env_runner.worker_index}: 加载模型 {model_path} 失败: {e}")
                        else:
                            logger.error(f"EnvRunner: 加载模型 {model_path} 失败: {e}")

        algorithm.env_runner_group.foreach_env_runner(setup_worker_modules)
        self._update_sampler_on_workers(algorithm)

    def on_episode_end(
        self, *, 
        episode, 
        env_index: int,
        worker: "RolloutWorker" = None,
        base_env: BaseEnv = None, 
        env_runner: "RolloutWorker" = None,
        **kwargs,
    ):
        """在新版 API 中，处理不同类型的episode对象。"""
        # 兼容性处理：使用env_runner如果worker为None
        if worker is None and env_runner is not None:
            worker = env_runner
            
        # 如果没有worker，无法获取环境信息，直接返回
        if worker is None:
            return
            
        main_agent_id, opponent_agent_id, opponent_module_id = None, None, None

        # 处理不同类型的episode对象
        if hasattr(episode, 'agent_for_module_map'):
            # 旧版EpisodeV2对象
            agent_module_map = episode.agent_for_module_map
        elif hasattr(episode, 'agent_ids') and hasattr(episode, 'module_for'):
            # 新版MultiAgentEpisode对象
            agent_module_map = {}
            for agent_id in episode.agent_ids:
                try:
                    module_id = episode.module_for(agent_id)
                    agent_module_map[agent_id] = module_id
                except:
                    # 如果无法获取module_id，跳过这个agent
                    continue
        else:
            # 无法处理的episode类型
            return

        # 确定哪个 agent 是 main_policy
        for agent_id, module_id in agent_module_map.items():
            if module_id == MAIN_POLICY_ID:
                main_agent_id = agent_id
                break
        
        # 如果没有 main_policy 参与（不太可能发生），则退出
        if main_agent_id is None:
            return

        # 找到对手 agent 和 module
        for agent_id, module_id in agent_module_map.items():
            if agent_id != main_agent_id:
                opponent_agent_id = agent_id
                opponent_module_id = module_id
                break
        
        if not all([opponent_agent_id, opponent_module_id]): 
            return

        # 获取游戏结果信息，兼容不同的episode类型
        winner = None
        if hasattr(episode, 'last_info_for'):
            # EpisodeV2类型
            last_info = episode.last_info_for(main_agent_id) or {}
            winner = last_info.get("winner")
        elif hasattr(episode, 'get_infos'):
            # MultiAgentEpisode类型
            try:
                infos = episode.get_infos(agent_ids=[main_agent_id])
                if infos and len(infos) > 0:
                    # 获取最后一个info
                    last_info = infos[-1].get(main_agent_id, {})
                    winner = last_info.get("winner")
            except:
                pass
        
        if winner is None: 
            return

        # 获取环境信息
        if base_env is None:
            return
            
        env = base_env.get_sub_environments()[env_index]
        main_player_num = env._agent_to_player_map.get(main_agent_id)
        if main_player_num is None: 
            return

        if opponent_module_id not in self.win_rates_buffer:
            self.win_rates_buffer[opponent_module_id] = []
        
        if winner == 0: 
            self.win_rates_buffer[opponent_module_id].append(0.5)
        elif winner == main_player_num: 
            self.win_rates_buffer[opponent_module_id].append(1.0)
        else: 
            self.win_rates_buffer[opponent_module_id].append(0.0)

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
            
            # 使用正确的 Elo 更新函数
            updated_elos = elo.update_elo(self.elo_ratings, MAIN_POLICY_ID, opponent, win_rate)
            self.elo_ratings.update(updated_elos)
            
            logger.info(f"    Elo: {main_elo:.0f} -> {self.elo_ratings[MAIN_POLICY_ID]:.0f} | Opponent Elo: {opponent_elo:.0f} -> {self.elo_ratings[opponent]:.0f}")
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
            
            new_opponent_module_id = f"{OPPONENT_POLICY_ID_PREFIX}{os.path.splitext(new_opponent_name)[0]}"
            self.elo_ratings[new_opponent_module_id] = self.elo_ratings.get(MAIN_POLICY_ID, ELO_DEFAULT)
            self.model_generations[new_opponent_module_id] = self.latest_generation

            logger.info(f"  - 动态添加新模块 {new_opponent_module_id} 到训练器中...")
            algorithm.add_module(
                module_id=new_opponent_module_id,
                module_spec=RLModuleSpec.from_module(main_module),
            )
            
            # 确保新添加的模块是不可训练的
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

    def _manage_opponent_pool(self, new_opponent_name: str, algorithm: Algorithm):
        """管理短期和长期对手池。"""
        self.short_term_pool_paths.append(new_opponent_name)
        if len(self.short_term_pool_paths) > SHORT_TERM_POOL_MAX_SIZE:
            to_remove_name = self.short_term_pool_paths.pop(0)
            module_id_to_remove = f"{OPPONENT_POLICY_ID_PREFIX}{os.path.splitext(to_remove_name)[0]}"
            self._remove_module_from_algorithm(algorithm, module_id_to_remove, to_remove_name)

        if self.latest_generation >= self.long_term_power_of_2:
            self.long_term_pool_paths.append(new_opponent_name)
            self.long_term_power_of_2 *= 2
            logger.info(f"  - 新模型被提升到长期对手池。下一个提升点: 第 {self.long_term_power_of_2} 代。")

        if len(self.long_term_pool_paths) > LONG_TERM_POOL_MAX_SIZE:
            to_remove_name = self.long_term_pool_paths.pop(0)
            module_id_to_remove = f"{OPPONENT_POLICY_ID_PREFIX}{os.path.splitext(to_remove_name)[0]}"
            # 确保不会把自己从短期池中移除（如果它同时在长期池中）
            if to_remove_name not in self.short_term_pool_paths:
                 self._remove_module_from_algorithm(algorithm, module_id_to_remove, to_remove_name)

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