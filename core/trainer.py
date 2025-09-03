# src_code/core/trainer.py

import os
import random
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.stopper import MaximumIterationStopper
from ray.tune import RunConfig, CheckpointConfig

from core.multi_agent_env import RLLibMultiAgentEnv
from callbacks.self_play_callback import SelfPlayCallback
from core.custom_model import CustomDarkChessRLModule
from utils.constants import *

class RLLibSelfPlayTrainer:
    """
    使用 RLlib PPO 进行自我对弈训练的主类 (已全面升级至新版 API)。
    """
    def __init__(self):
        print("--- [步骤 1/3] 初始化环境和模型 ---")
        self._register_env()

        print("--- [步骤 2/3] 配置 PPO 算法 ---")
        self.algo_config = self._build_algorithm_config()

    def _register_env(self):
        """注册自定义环境。"""
        tune.register_env("dark_chess_multi_agent", lambda config: RLLibMultiAgentEnv(config))

    def _build_algorithm_config(self) -> PPOConfig:
        """
        使用新版 PPOConfig 对象构建并返回 PPO 算法的配置。
        """
        # 最终修正点: 增加对 agent_id 格式的检查，防止 IndexError
        def policy_mapping_fn(agent_id: str, episode, **kwargs):
            """
            健壮的策略映射函数，能处理 RLlib 内部的非玩家 agent_id。
            """
            # 只有当 agent_id 是我们定义的玩家 ID 格式时，才进行解析和对战逻辑分配
            if agent_id.startswith("player_"):
                player_idx = int(agent_id.split("_")[1])
                # 使用 episode ID 的奇偶性来交替分配主策略，确保模型能学习先手和后手
                if hash(episode.id_) % 2 == player_idx:
                    return MAIN_POLICY_ID
                else:
                    # 在自我对弈中，对手通常也是一个策略
                    # 这里的 "opponent_policy" 会由 SelfPlayCallback 动态管理
                    return OPPONENT_POLICY_ID_PREFIX 
            # 对于其他内部 agent_id (例如 '__env__'), 安全地返回一个默认策略
            else:
                return MAIN_POLICY_ID

        config = (
            PPOConfig()
            .framework("torch")
            .environment(
                "dark_chess_multi_agent",
                env_config={},
                disable_env_checking=True
            )
            .env_runners(
                num_env_runners=NUM_WORKERS,
                rollout_fragment_length="auto",
                num_gpus_per_env_runner=NUM_GPUS_PER_WORKER
            )
            .training(
                lambda_=0.95,
                kl_coeff=0.5,
                clip_param=CLIP_PARAM,
                vf_clip_param=10.0,
                entropy_coeff=ENTROPY_COEFF,
                train_batch_size=TRAIN_BATCH_SIZE,
                minibatch_size=SGD_MINIBATCH_SIZE,
                num_epochs=NUM_SGD_ITER,
                lr=LEARNING_RATE,
            )
            .rl_module(
                rl_module_spec=RLModuleSpec(module_class=CustomDarkChessRLModule)
            )
            .multi_agent(
                policies={MAIN_POLICY_ID, OPPONENT_POLICY_ID_PREFIX}, # 使用常量
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=[MAIN_POLICY_ID],
            )
            .callbacks(SelfPlayCallback)
            .resources(
                num_gpus=NUM_GPUS
            )
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
        )
        return config

    def run(self):
        """
        运行训练过程。
        """
        tuner = tune.Tuner(
            "PPO",
            param_space=self.algo_config.to_dict(),
            run_config=RunConfig(
                name="PPO_self_play_experiment_new_api",
                stop=MaximumIterationStopper(max_iter=TRAINING_ITERATIONS),
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=CHECKPOINT_FREQ,
                    checkpoint_at_end=True
                ),
                storage_path=TENSORBOARD_LOG_DIR,
            ),
        )
        
        results = tuner.fit()
        print("--- [步骤 3/3] 训练完成！ ---")
        return results