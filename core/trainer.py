# rllib_version_complete/core/trainer.py

import os
import random
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.stopper import MaximumIterationStopper
from ray.air import RunConfig, CheckpointConfig


from core.multi_agent_env import RLLibMultiAgentEnv
from callbacks.self_play_callback import SelfPlayCallback
from core.custom_model import CustomDarkChessModel
from utils.constants import *

class RLLibSelfPlayTrainer:
    """
    使用 RLlib PPO 进行自我对弈训练的主类 (重构版)。
    """
    def __init__(self):
        print("--- [步骤 1/3] 初始化环境和模型 ---")
        self._register_env_and_model()

        print("--- [步骤 2/3] 配置 PPO 算法 ---")
        self.algo_config = self._build_algorithm_config()

    def _register_env_and_model(self):
        """注册自定义环境和模型。"""
        # 注册自定义环境
        tune.register_env("dark_chess_multi_agent", lambda config: RLLibMultiAgentEnv(config))

        # 注册自定义模型
        ModelCatalog.register_custom_model("custom_dark_chess_model", CustomDarkChessModel)

    def _build_algorithm_config(self) -> PPOConfig:
        """构建并返回 PPO 算法的配置对象。"""
        # 定义策略映射函数
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if episode.last_info_for(agent_id).get('policy_id') == MAIN_POLICY_ID:
                return MAIN_POLICY_ID
            
            # 使用 callback 中注入的采样分布来选择对手
            dist = getattr(worker, 'sampler_dist', {MAIN_POLICY_ID: 1.0})
            opponent_policy_id = random.choices(list(dist.keys()), list(dist.values()), k=1)[0]
            return opponent_policy_id

        # --- [API变更] ---
        # 使用新的 .env_runners() API 替代旧的 .rollouts()
        # 将 .training() 中的参数直接设置到 config 对象上
        config = (
            PPOConfig()
            .environment(
                "dark_chess_multi_agent",
                env_config={},
                disable_env_checking=True
            )
            .framework("torch")
            # 变更点 1: rollouts() -> env_runners()
            # num_rollout_workers -> num_env_runners
            .env_runners(
                num_env_runners=NUM_WORKERS,
                rollout_fragment_length="auto"
            )
            # 变更点 2: .training() 中的模型和训练超参数直接在 config 上设置
            .training(
                model={
                    "custom_model": "custom_dark_chess_model",
                    "vf_share_layers": True,
                },
                lambda_=0.95,
                kl_coeff=0.5,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                train_batch_size=TRAIN_BATCH_SIZE,
                sgd_minibatch_size=SGD_MINIBATCH_SIZE,
                num_sgd_iter=NUM_SGD_ITER,
                lr=LEARNING_RATE,
            )
            .multi_agent(
                policies={MAIN_POLICY_ID}, # 一开始只有主策略
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=[MAIN_POLICY_ID],
            )
            .callbacks(SelfPlayCallback)
            .resources(
                num_gpus=NUM_GPUS,
                num_gpus_per_worker=NUM_GPUS_PER_WORKER,
            )
        )
        return config

    def run(self):
        """
        运行训练过程，包括启动、训练和可选的模型恢复。
        """
        # --- [API变更] ---
        # ray.air.RunConfig 和 ray.air.CheckpointConfig 的使用方式保持不变，但为了代码清晰，
        # 在文件头部明确导入。
        tuner = tune.Tuner(
            "PPO",
            param_space=self.algo_config.to_dict(),
            run_config=RunConfig(
                name="PPO_self_play_experiment",
                stop=MaximumIterationStopper(max_iter=TRAINING_ITERATIONS),
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=CHECKPOINT_FREQ,
                    checkpoint_at_end=True
                ),
                local_dir=TENSORBOARD_LOG_DIR,
            ),
        )
        
        results = tuner.fit()
        print("--- [步骤 3/3] 训练完成！ ---")
        return results