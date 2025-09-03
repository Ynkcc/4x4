# rllib_version_complete/core/trainer.py

import os
import shutil
import torch
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from typing import Dict, Any, Optional

# 导入项目模块
from core.multi_agent_env import RLLibMultiAgentEnv
from core.policy import RLLibCustomNetwork
from callbacks.self_play_callback import SelfPlayCallback
from utils import elo
from utils.constants import *

class RLLibSelfPlayTrainer:
    """
    使用 Ray RLlib 的自我对弈训练器。
    """
    def __init__(self):
        self._setup_directories()
        self.elo_ratings = elo.load_elo_ratings()
        # 初始化一个空的对手池，将在运行时动态填充
        self.opponent_policies_specs: Dict[str, PolicySpec] = {}

    def _setup_directories(self):
        """创建所有必要的目录。"""
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

    def _get_opponent_sampling_distribution(self) -> Dict[str, float]:
        """
        根据Elo差异计算对手的采样概率。
        对手池中除了历史模型，还包括当前的主策略自身。
        """
        opponents = [f for f in os.listdir(OPPONENT_POOL_DIR) if f.endswith('.pt')]
        main_elo = self.elo_ratings.get(MAIN_POLICY_ID, ELO_DEFAULT)
        
        weights = {}
        # 为主策略设置一个基础权重，使其有一定概率与自己对战
        weights[MAIN_POLICY_ID] = 1.0 
        
        for opp_name in opponents:
            opp_policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{opp_name.replace('.pt', '')}"
            opp_elo = self.elo_ratings.get(opp_policy_id, ELO_DEFAULT)
            # 使用温度参数来平滑Elo差异，使得Elo相近的对手更容易被选中
            weight = np.exp(-abs(main_elo - opp_elo) / ELO_WEIGHT_TEMPERATURE)
            weights[opp_policy_id] = weight

        total_weight = sum(weights.values())
        if total_weight == 0:
             return {MAIN_POLICY_ID: 1.0}
        return {k: v / total_weight for k, v in weights.items()}

    def _load_opponent_policies(self):
        """从对手池目录加载所有对手策略，并为它们创建PolicySpec。"""
        self.opponent_policies_specs.clear()
        for filename in os.listdir(OPPONENT_POOL_DIR):
            if filename.endswith('.pt'):
                policy_id = f"{OPPONENT_POLICY_ID_PREFIX}{filename.replace('.pt', '')}"
                self.opponent_policies_specs[policy_id] = PolicySpec()
        print(f"从池中加载了 {len(self.opponent_policies_specs)} 个对手策略。")

    def run(self):
        """执行完整的训练流程。"""
        print("--- [步骤 1/4] 初始化 Ray 和环境 ---")
        ray.init(num_gpus=1 if PPO_DEVICE == 'cuda' else 0, local_mode=False)
        
        ModelCatalog.register_custom_model("custom_torch_model", RLLibCustomNetwork)
        register_env("dark_chess_multi_agent", lambda config: RLLibMultiAgentEnv(config))
        self._load_opponent_policies()

        print("--- [步骤 2/4] 配置 PPO 算法 ---")
        
        # 动态更新的对手采样分布
        opponent_dist = self._get_opponent_sampling_distribution()
        
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id == "player1":
                return MAIN_POLICY_ID
            else: # player2
                # 从对手池（包含主策略）中随机选择一个策略
                policies = list(opponent_dist.keys())
                probabilities = list(opponent_dist.values())
                if not policies or sum(probabilities) == 0:
                    return MAIN_POLICY_ID
                return np.random.choice(policies, p=probabilities)

        config = (
            PPOConfig()
            .environment("dark_chess_multi_agent")
            .framework("torch")
            .env_runners(num_env_runners=N_ENVS, rollout_fragment_length="auto")
            .training(
                model={"custom_model": "custom_torch_model"},
                lr=INITIAL_LR,
                clip_param=PPO_CLIP_RANGE,
                train_batch_size=PPO_N_STEPS * N_ENVS,
                minibatch_size=PPO_BATCH_SIZE,
                num_epochs=PPO_N_EPOCHS,
                lambda_=PPO_GAE_LAMBDA,
                vf_loss_coeff=PPO_VF_COEF,
                entropy_coeff=PPO_ENT_COEF,
            )
            .multi_agent(
                policies={MAIN_POLICY_ID: PolicySpec()} | self.opponent_policies_specs,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=[MAIN_POLICY_ID],
            )
            .resources(num_gpus=1 if PPO_DEVICE == 'cuda' else 0)
            .callbacks(SelfPlayCallback) # 使用类引用
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        )

        latest_checkpoint = self._find_latest_checkpoint(TENSORBOARD_LOG_PATH)
        if latest_checkpoint:
            print(f"--- 发现检查点，从 {latest_checkpoint} 恢复训练 ---")
            algo = config.build()
            algo.restore(latest_checkpoint)
        else:
            print("--- 未发现检查点，开始新的训练 ---")
            algo = config.build()

        # 为池中的非训练策略加载权重
        main_policy = algo.get_policy(MAIN_POLICY_ID)
        for policy_id in self.opponent_policies_specs.keys():
            opponent_model_name = policy_id.replace(OPPONENT_POLICY_ID_PREFIX, "") + ".pt"
            model_path = os.path.join(OPPONENT_POOL_DIR, opponent_model_name)
            if os.path.exists(model_path):
                 print(f"为策略 {policy_id} 加载权重: {model_path}")
                 policy = algo.get_policy(policy_id)
                 # 创建一个与主策略相同类型的模型实例，然后加载状态
                 state_dict = torch.load(model_path, map_location=policy.device)
                 policy.model.load_state_dict(state_dict)
                 # 确保对手策略不训练
                 policy.lock_weights()

        print("--- [步骤 3/4] 开始自我对弈主循环 ---")
        for i in range(1, TOTAL_TRAINING_LOOPS + 1):
            print(f"\n{'='*70}\n🔄 训练循环 {i}/{TOTAL_TRAINING_LOOPS}\n{'='*70}")
            
            # 每轮训练前都更新一次对手采样分布
            opponent_dist = self._get_opponent_sampling_distribution()
            print("对手采样分布:", {k: f"{v:.2%}" for k, v in opponent_dist.items()})

            result = algo.train()
            
            checkpoint_dir = algo.save(checkpoint_dir=TENSORBOARD_LOG_PATH)
            print(f"检查点已保存至: {checkpoint_dir}")

        print("\n--- [步骤 4/4] 训练完成！ ---")
        algo.stop()
        ray.shutdown()

    def _find_latest_checkpoint(self, directory: str) -> Optional[str]:
        """查找最新的RLlib检查点目录。"""
        try:
            checkpoints = [os.path.join(directory, d) for d in os.listdir(directory) if d.startswith("PPO_")]
            if not checkpoints:
                return None
            # 找到最新的实验目录
            latest_experiment = max(checkpoints, key=os.path.getmtime)
            
            # 在实验目录中找到最新的检查点
            checkpoint_dirs = [os.path.join(latest_experiment, d) for d in os.listdir(latest_experiment) if d.startswith("checkpoint_")]
            if not checkpoint_dirs:
                return None
            return max(checkpoint_dirs, key=os.path.getmtime)
        except FileNotFoundError:
            return None
