# rllib_version_complete/core/trainer.py

import os
import shutil
import torch
import numpy as np
import ray
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from typing import Dict, Any, Optional

from core.multi_agent_env import RLLibMultiAgentEnv
from core.policy import RLLibCustomNetwork
from callbacks.self_play_callback import SelfPlayCallback
from utils.constants import *

class RLLibSelfPlayTrainer:
    """
    使用 Ray RLlib 的自我对弈训练器 (重构版)。
    职责: 仅负责配置和启动训练任务。所有动态逻辑移至 Callback。
    """
    def __init__(self):
        self._setup_directories()

    def _setup_directories(self):
        """创建所有必要的目录。"""
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

    def run(self):
        """执行完整的训练流程。"""
        print("--- [步骤 1/3] 初始化 Ray 和环境 ---")
        ray.init(num_gpus=1 if PPO_DEVICE == 'cuda' else 0, local_mode=False)
        
        ModelCatalog.register_custom_model("custom_torch_model", RLLibCustomNetwork)
        register_env("dark_chess_multi_agent", lambda config: RLLibMultiAgentEnv(config))

        print("--- [步骤 2/3] 配置 PPO 算法 ---")
        
        # 初始策略定义：只有主策略。对手策略将由 Callback 动态加载和管理。
        policies = {MAIN_POLICY_ID: PolicySpec()}

        # policy_mapping_fn 从 worker 获取最新的采样分布，而不是依赖 trainer
        # 这使得对手选择更加动态和分布式
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # 在 worker 初始化后，callback 会为其注入 sampler_dist
            dist = getattr(worker, "sampler_dist", {MAIN_POLICY_ID: 1.0})

            if agent_id == "player1":
                return MAIN_POLICY_ID
            else: # player2
                policies = list(dist.keys())
                probabilities = list(dist.values())
                if not policies or sum(probabilities) == 0:
                    return MAIN_POLICY_ID # 备用方案
                return np.random.choice(policies, p=probabilities)

        config = (
            PPOConfig()
            .environment("dark_chess_multi_agent")
            .framework("torch")
            .rollouts(num_rollout_workers=N_ENVS, rollout_fragment_length="auto")
            .training(
                model={"custom_model": "custom_torch_model"},
                lr=INITIAL_LR,
                clip_param=PPO_CLIP_RANGE,
                train_batch_size=PPO_N_STEPS * N_ENVS,
                sgd_minibatch_size=PPO_BATCH_SIZE,
                num_sgd_iter=PPO_N_EPOCHS,
                lambda_=PPO_GAE_LAMBDA,
                vf_loss_coeff=PPO_VF_COEF,
                entropy_coeff=PPO_ENT_COEF,
            )
            .multi_agent(
                policies=policies, # 初始时只有主策略
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=[MAIN_POLICY_ID],
            )
            .resources(num_gpus=1 if PPO_DEVICE == 'cuda' else 0)
            .callbacks(SelfPlayCallback) # 核心逻辑在 Callback 中
            .rl_module(_enable_rl_module_api=False)
            .training(_enable_learner_api=False)
        )

        # 使用 tune.Tuner 可以更优雅地处理检查点和恢复
        tuner = tune.Tuner(
            "PPO",
            run_config=RunConfig(
                stop={"training_iteration": TOTAL_TRAINING_LOOPS},
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=1, 
                    checkpoint_at_end=True,
                    num_to_keep=3, # 保留最近3个检查点
                ),
                local_dir=TENSORBOARD_LOG_PATH,
                name="PPO_self_play_experiment"
            ),
            param_space=config.to_dict(),
        )
        
        # 尝试从最新的检查点恢复
        # tuner.fit() 会自动处理恢复逻辑
        results = tuner.fit()
        
        print("\n--- [步骤 3/3] 训练完成！ ---")
        best_checkpoint = results.get_best_result(metric="episode_reward_mean", mode="max").checkpoint
        if best_checkpoint:
            print(f"最佳检查点位于: {best_checkpoint.path}")
            
        ray.shutdown()