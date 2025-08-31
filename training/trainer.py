# src_code/training/trainer.py

import os
import shutil
import time
import re
import json
import numpy as np
import sys

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from sb3_contrib import MaskablePPO
from typing import Dict, Any, List, Optional

from utils.constants import *
from game.environment import GameEnvironment
from game.policy import CustomActorCriticPolicy
from training.evaluator import evaluate_models

def create_new_ppo_model(env, tensorboard_log=None):
    """
    创建一个全新的随机初始化的PPO模型。
    """
    # 【V8 修改】输入维度现在是CNN和MLP输出之和
    input_dim = NETWORK_NUM_HIDDEN_CHANNELS + SCALAR_ENCODER_OUTPUT_DIM

    # 推荐方案: 强化价值网络 (已加深加宽)
    policy_net_arch = dict(
        pi=[input_dim, input_dim * 2, input_dim],
        vf=[input_dim, input_dim * 2, input_dim * 2, input_dim]
    )

    model = MaskablePPO(
        policy=CustomActorCriticPolicy,
        env=env,
        learning_rate=INITIAL_LR,
        clip_range=PPO_CLIP_RANGE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gae_lambda=PPO_GAE_LAMBDA,
        vf_coef=PPO_VF_COEF,
        ent_coef=PPO_ENT_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        tensorboard_log=tensorboard_log,
        device=PPO_DEVICE,
        verbose=PPO_VERBOSE,
        policy_kwargs={
            "net_arch": policy_net_arch,
        }
    )
    return model

def load_ppo_model_with_hyperparams(model_path: str, env, tensorboard_log=None):
    """
    加载PPO模型并应用自定义超参数。
    """
    # 【V8 修改】输入维度现在是CNN和MLP输出之和
    input_dim = NETWORK_NUM_HIDDEN_CHANNELS + SCALAR_ENCODER_OUTPUT_DIM
    
    # 推荐方案: 强化价值网络 (已加深加宽)
    policy_net_arch = dict(
        pi=[input_dim, input_dim * 2, input_dim],
        vf=[input_dim, input_dim * 2, input_dim * 2, input_dim]
    )

    model = MaskablePPO.load(
        model_path,
        env=env,
        learning_rate=INITIAL_LR,
        clip_range=PPO_CLIP_RANGE,
        tensorboard_log=tensorboard_log,
        n_steps=PPO_N_STEPS,
        device=PPO_DEVICE,
        custom_objects={
            "policy_class": CustomActorCriticPolicy
        },
        policy_kwargs={
            "net_arch": policy_net_arch,
        }
    )
    # 重新应用超参数
    model.batch_size = PPO_BATCH_SIZE
    model.n_epochs = PPO_N_EPOCHS
    model.gae_lambda = PPO_GAE_LAMBDA
    model.vf_coef = PPO_VF_COEF
    model.ent_coef = PPO_ENT_COEF
    model.max_grad_norm = PPO_MAX_GRAD_NORM
    return model

class SelfPlayTrainer:
    """
    【V7 新规则版】
    - 以 "挑战者" 为核心进行持续训练。
    - 对手池分为 "长期" 和 "短期" 池，采用新的动态差值规则。
    - 实现了更科学的历史模型保留和采样机制。
    """
    def __init__(self):
        self.model: Optional[MaskablePPO] = None
        self.env: Optional[VecEnv] = None
        self.tensorboard_log_run_path = None
        
        # --- 对手池核心属性 (新规则) ---
        self.long_term_pool_paths = []
        self.short_term_pool_paths = []
        self.long_term_power_of_2 = 1 # 记录长期模型中2的指数，初始为1
        self.combined_opponent_data: List[Dict[str, Any]] = []

        # --- Elo与模型管理 ---
        self.elo_ratings = {}
        self.model_generations = {} # 新增: 用于追踪模型代数
        self.latest_generation = 0
        self.default_elo = ELO_DEFAULT
        self.elo_k_factor = ELO_K_FACTOR
        
        self._setup()

    def _setup(self):
        """
        【重构】执行所有启动前的准备工作，管理模型生命周期。
        """
        print("--- [步骤 1/5] 初始化设置 ---")
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

        self._load_elo_and_generations()

        # 核心模型生命周期管理
        if not os.path.exists(CHALLENGER_PATH):
            print(">>> 挑战者模型不存在，视为从零开始训练。")
            self._create_initial_models()
        
        if not os.path.exists(MAIN_OPPONENT_PATH):
            print(">>> 主宰者模型不存在，将从现有挑战者模型复制。")
            shutil.copy(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
            main_opp_name = os.path.basename(MAIN_OPPONENT_PATH)
            challenger_name = os.path.basename(CHALLENGER_PATH)
            if main_opp_name not in self.elo_ratings:
                self.elo_ratings[main_opp_name] = self.elo_ratings.get(challenger_name, self.default_elo)
            if main_opp_name not in self.model_generations:
                 self.model_generations[main_opp_name] = self.model_generations.get(challenger_name, 0)
            self._save_elo_and_generations()

        # 启动时进行一次池管理（主要用于清理无效文件）
        self._manage_opponent_pool()

    def _create_initial_models(self):
        """创建一个全新的随机初始化模型作为训练起点。"""
        print("正在创建临时环境以初始化模型...")
        temp_env = GameEnvironment()
        
        print("正在创建新的PPO模型...")
        new_model = create_new_ppo_model(env=temp_env)
        
        # 保存为挑战者和主宰者
        new_model.save(CHALLENGER_PATH)
        shutil.copy(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
        print(f"✅ 初始模型已创建并保存为 'challenger.zip' 和 'main_opponent.zip'")
        
        # 初始化Elo和代数
        challenger_name = os.path.basename(CHALLENGER_PATH)
        main_opponent_name = os.path.basename(MAIN_OPPONENT_PATH)
        self.elo_ratings[challenger_name] = self.default_elo
        self.elo_ratings[main_opponent_name] = self.default_elo
        self.model_generations[challenger_name] = 0
        self.model_generations[main_opponent_name] = 0
        self.latest_generation = 0
        self._save_elo_and_generations()
        
        temp_env.close()
        print("✅ 临时环境已清理")

    def _load_elo_and_generations(self):
        """从JSON文件加载Elo评分、模型代数和新的模型池状态。"""
        elo_file = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        if os.path.exists(elo_file):
            try:
                with open(elo_file, 'r') as f:
                    data = json.load(f)
                    self.elo_ratings = data.get("elo", {})
                    self.model_generations = data.get("generations", {})
                    self.latest_generation = data.get("latest_generation", 0)
                    self.long_term_pool_paths = data.get("long_term_pool_paths", [])
                    self.short_term_pool_paths = data.get("short_term_pool_paths", [])
                    self.long_term_power_of_2 = data.get("long_term_power_of_2", 1)
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"警告：读取状态文件失败或格式不完整: {e}。将使用默认值。")
                self.elo_ratings = {}
                self.model_generations = {}
                self.latest_generation = 0
                self.long_term_pool_paths = []
                self.short_term_pool_paths = []
                self.long_term_power_of_2 = 1
    
    def _save_elo_and_generations(self):
        """将Elo、模型代数和模型池状态保存到同一个JSON文件。"""
        elo_file = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        data = {
            "elo": self.elo_ratings,
            "generations": self.model_generations,
            "latest_generation": self.latest_generation,
            "long_term_pool_paths": self.long_term_pool_paths,
            "short_term_pool_paths": self.short_term_pool_paths,
            "long_term_power_of_2": self.long_term_power_of_2,
        }
        try:
            with open(elo_file, 'w') as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            print(f"错误：无法保存状态文件: {e}")

    def _manage_opponent_pool(self, new_opponent_path=None):
        """
        【V7 新规则】管理长期和短期对手池。
        """
        if new_opponent_path:
            self.latest_generation += 1
            new_opponent_name = os.path.basename(new_opponent_path)
            self.model_generations[new_opponent_name] = self.latest_generation
            
            added_to_long_term = False
            
            long_term_pool_with_gens = sorted(
                [(p, self.model_generations.get(p, 0)) for p in self.long_term_pool_paths],
                key=lambda x: x[1]
            )
            self.long_term_pool_paths = [p for p, _ in long_term_pool_with_gens]
            long_term_gens = [g for _, g in long_term_pool_with_gens]

            if not self.long_term_pool_paths:
                print(f"长期池为空，新模型 {new_opponent_name} 直接加入。")
                self.long_term_pool_paths.append(new_opponent_name)
                added_to_long_term = True
            else:
                required_gap = 2 ** self.long_term_power_of_2
                actual_gap = self.latest_generation - long_term_gens[-1]

                if actual_gap == required_gap:
                    if len(self.long_term_pool_paths) >= LONG_TERM_POOL_SIZE:
                        print(f"长期池已满且满足差值 {required_gap}，触发指数更新。")
                        self.long_term_power_of_2 += 1
                        new_required_gap = 2 ** self.long_term_power_of_2
                        print(f"2的指数提升至 {self.long_term_power_of_2} (新差值为 {new_required_gap})。")
                        
                        retained_pool = [self.long_term_pool_paths[0]]
                        last_kept_gen = long_term_gens[0]
                        
                        for i in range(1, len(long_term_gens)):
                            if (long_term_gens[i] - last_kept_gen) == new_required_gap:
                                retained_pool.append(self.long_term_pool_paths[i])
                                last_kept_gen = long_term_gens[i]
                        
                        self.long_term_pool_paths = retained_pool
                        print(f"长期池更新后保留 {len(self.long_term_pool_paths)} 个模型。")
                        
                        new_last_gen = self.model_generations.get(self.long_term_pool_paths[-1], 0)
                        if len(self.long_term_pool_paths) < LONG_TERM_POOL_SIZE and (self.latest_generation - new_last_gen) == new_required_gap:
                            self.long_term_pool_paths.append(new_opponent_name)
                            added_to_long_term = True
                            print(f"新模型 {new_opponent_name} 在更新后成功加入长期池。")
                    else:
                        self.long_term_pool_paths.append(new_opponent_name)
                        added_to_long_term = True
                        print(f"长期池未满，新模型 {new_opponent_name} 成功加入。")

            if not added_to_long_term:
                self.short_term_pool_paths.append(new_opponent_name)
                self.short_term_pool_paths.sort(
                    key=lambda p: self.model_generations.get(p, 0),
                    reverse=True
                )
                if len(self.short_term_pool_paths) > SHORT_TERM_POOL_SIZE:
                    self.short_term_pool_paths = self.short_term_pool_paths[:SHORT_TERM_POOL_SIZE]
        
        current_pool_names = set(self.short_term_pool_paths + self.long_term_pool_paths)
        
        for filename in os.listdir(OPPONENT_POOL_DIR):
            if filename.endswith('.zip') and filename not in current_pool_names:
                print(f"✂️ 清理过时对手: {filename}")
                os.remove(os.path.join(OPPONENT_POOL_DIR, filename))
                self.elo_ratings.pop(filename, None)
                self.model_generations.pop(filename, None)
        
        self._save_elo_and_generations()
        self._update_opponent_data()

    def _update_opponent_data(self):
        """
        【新规则】创建一个包含路径、权重和预加载模型实例的字典列表。
        - 从长期+短期池中加权随机采样指定数量的参训模型
        - 长期池模型具有两倍权重
        - 确保主宰者权重不少于51%
        """
        self.combined_opponent_data.clear()
        
        # 第一步：构建池候选者列表（使用字典推导式简化）
        pool_candidates = [
            {'filename': f, 'pool_type': 'short_term', 'sampling_weight': 1.0}
            for f in self.short_term_pool_paths
        ] + [
            {'filename': f, 'pool_type': 'long_term', 'sampling_weight': LONG_TERM_POOL_WEIGHT_MULTIPLIER}
            for f in self.long_term_pool_paths
        ]
        
        # 预先判断：检查可用模型数量
        total_available = len(pool_candidates)
        if total_available < TRAINING_POOL_SAMPLE_SIZE:
            print(f"⚠️  警告：可用模型数量 ({total_available}) 少于所需采样数量 ({TRAINING_POOL_SAMPLE_SIZE})")
            print(f"     将使用所有可用的 {total_available} 个模型进行训练")
        
        # 执行加权随机采样（简化版）
        selected_pool_models = []
        if pool_candidates:
            sample_size = min(TRAINING_POOL_SAMPLE_SIZE, total_available)
            
            # 使用numpy加权采样（一行代码完成）
            weights = np.array([c['sampling_weight'] for c in pool_candidates])
            probs = weights / weights.sum()
            selected_indices = np.random.choice(len(pool_candidates), size=sample_size, replace=False, p=probs)
            selected_pool_models = [pool_candidates[i]['filename'] for i in selected_indices]
            
            print(f"\n--- 对手池采样结果 ---")
            print(f"从 {total_available} 个候选者中采样了 {len(selected_pool_models)} 个模型:")
            for i, filename in enumerate(selected_pool_models):
                candidate = pool_candidates[selected_indices[i]]
                pool_type_cn = "长期池" if candidate['pool_type'] == 'long_term' else "短期池"
                print(f"  {i+1}. {filename} ({pool_type_cn})")
        
        # 第二步：预加载模型并计算Elo权重（使用字典推导式简化）
        main_opponent_name = os.path.basename(MAIN_OPPONENT_PATH)
        if main_opponent_name not in self.elo_ratings:
            self.elo_ratings[main_opponent_name] = self.default_elo
        main_elo = self.elo_ratings[main_opponent_name]
        
        all_model_paths = [os.path.join(OPPONENT_POOL_DIR, f) for f in selected_pool_models] + [MAIN_OPPONENT_PATH]
        
        # 使用字典推导式预加载模型
        try:
            loaded_models = {
                path: MaskablePPO.load(path, device="cpu") 
                for path in set(all_model_paths)
            }
        except Exception as e:
            raise ValueError(f"训练器错误: 预加载模型失败: {e}")
        
        # 使用lambda和列表推导式计算Elo权重
        calculate_elo_weight = lambda path: np.exp(-abs(main_elo - self.elo_ratings.get(os.path.basename(path), self.default_elo)) / ELO_WEIGHT_TEMPERATURE)
        pool_weights = [{'path': path, 'weight': calculate_elo_weight(path)} for path in all_model_paths]
        
        # 第三步：确保主宰者权重不少于MAIN_OPPONENT_MIN_WEIGHT_RATIO
        pool_total_weight = sum(w['weight'] for w in pool_weights if w['path'] != MAIN_OPPONENT_PATH)
        main_current_weight = next((w['weight'] for w in pool_weights if w['path'] == MAIN_OPPONENT_PATH), 0)
        
        # 计算主宰者应有的最小权重
        min_main_weight = pool_total_weight * MAIN_OPPONENT_MIN_WEIGHT_RATIO / (1 - MAIN_OPPONENT_MIN_WEIGHT_RATIO)
        
        if main_current_weight < min_main_weight:
            print(f">>> 主宰者权重调整: {main_current_weight:.3f} -> {min_main_weight:.3f} (确保不少于 {MAIN_OPPONENT_MIN_WEIGHT_RATIO:.1%})")
            # 使用列表推导式更新权重
            pool_weights = [
                {**w, 'weight': min_main_weight} if w['path'] == MAIN_OPPONENT_PATH else w
                for w in pool_weights
            ]
        
        # 第四步：归一化权重并构建最终数据结构（使用列表推导式）
        total_weight = sum(w['weight'] for w in pool_weights)
        
        if total_weight > 0:
            self.combined_opponent_data = [
                {
                    'path': w['path'],
                    'weight': w['weight'] / total_weight,
                    'model': loaded_models[w['path']]
                }
                for w in pool_weights
            ]
        else:
            # 备用方案：平均分配权重
            uniform_weight = 1.0 / len(all_model_paths) if all_model_paths else 0.0
            self.combined_opponent_data = [
                {'path': path, 'weight': uniform_weight, 'model': loaded_models[path]}
                for path in all_model_paths
            ]

        # 打印池状态和权重分布
        print("\n--- 对手池状态 ---")
        print(f"短期池 ({len(self.short_term_pool_paths)}/{SHORT_TERM_POOL_SIZE}): {self.short_term_pool_paths}")
        print(f"长期池 ({len(self.long_term_pool_paths)}/{LONG_TERM_POOL_SIZE}): {self.long_term_pool_paths}")
        print(f"长期池代数差值指数: {self.long_term_power_of_2} (当前要求差值: {2**self.long_term_power_of_2})")
        
        print(f"\n对手池最终权重分布 (参训模型: {len(selected_pool_models)}, 总模型: {len(self.combined_opponent_data)}):")
        # 使用lambda表达式排序
        sorted_data = sorted(self.combined_opponent_data, key=lambda x: os.path.basename(x['path']))
        for item in sorted_data:
            name = os.path.basename(item['path'])
            elo = self.elo_ratings.get(name, self.default_elo)
            is_main = "★主宰者" if item['path'] == MAIN_OPPONENT_PATH else ""
            print(f"  - {name:<25} (Elo: {elo:.0f}, 权重: {item['weight']:.2%}) {is_main}")

    def _prepare_environment_and_models(self):
        """准备用于训练的模型和环境。"""
        print("\n--- [步骤 2/5] 准备环境和模型 ---")
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.tensorboard_log_run_path = os.path.join(TENSORBOARD_LOG_PATH, run_name)
        print(f"TensorBoard 日志将保存到: {self.tensorboard_log_run_path}")

        print(f"创建 {N_ENVS} 个并行的训练环境...")
        vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
        self.env = make_vec_env(
            GameEnvironment, n_envs=N_ENVS, vec_env_cls=vec_env_cls,
            env_kwargs={
                'opponent_data': self.combined_opponent_data,
                'shaping_coef': SHAPING_COEF_INITIAL 
            }
        )
        
        print(f"加载学习者模型: {os.path.basename(CHALLENGER_PATH)}")
        self.model = load_ppo_model_with_hyperparams(
            CHALLENGER_PATH,
            env=self.env,
            tensorboard_log=self.tensorboard_log_run_path
        )
        print("✅ 环境和模型准备完成！")

    def _train_learner(self):
        """训练学习者模型（即挑战者）。"""
        assert self.model is not None, "Model not initialized"
        assert self.env is not None, "Environment not initialized"
        print(f"🏋️  阶段一: 挑战者进行 {STEPS_PER_LOOP:,} 步训练...")
        start_time = time.time()
        self.model.learn(total_timesteps=STEPS_PER_LOOP, reset_num_timesteps=False, progress_bar=PPO_SHOW_PROGRESS)
        self.model.save(CHALLENGER_PATH)
        elapsed_time = time.time() - start_time
        print(f"✅ 训练完成! 用时: {elapsed_time:.1f}秒, 总步数: {self.model.num_timesteps:,}")
        print(f"✅ 挑战者训练完成，新参数已保存至 {os.path.basename(CHALLENGER_PATH)}")

    def _update_elo(self, player_a_name, player_b_name, player_a_win_rate):
        """根据胜率更新Elo。"""
        player_a_elo = self.elo_ratings.get(player_a_name, self.default_elo)
        player_b_elo = self.elo_ratings.get(player_b_name, self.default_elo)

        expected_win_a = 1 / (1 + 10 ** ((player_b_elo - player_a_elo) / 400))
        
        new_player_a_elo = player_a_elo + self.elo_k_factor * (player_a_win_rate - expected_win_a)
        new_player_b_elo = player_b_elo - self.elo_k_factor * (player_a_win_rate - expected_win_a)
        
        self.elo_ratings[player_a_name] = new_player_a_elo
        self.elo_ratings[player_b_name] = new_player_b_elo
        
        print(f"Elo 更新 ({player_a_name} vs {player_b_name}, 基于胜率 {player_a_win_rate:.2%}):")
        print(f"  - {player_a_name}: {player_a_elo:.0f} -> {new_player_a_elo:.0f} (Δ {new_player_a_elo - player_a_elo:+.1f})")
        print(f"  - {player_b_name}: {player_b_elo:.0f} -> {new_player_b_elo:.0f} (Δ {new_player_b_elo - player_b_elo:+.1f})")
        
    def _evaluate_and_update(self) -> bool:
        """评估、决策、更新Elo、轮换对手、同步环境的完整流程。"""
        assert self.model is not None, "Model not initialized"
        assert self.env is not None, "Environment not initialized"
        print(f"\n💾 阶段二: {os.path.basename(CHALLENGER_PATH)} 向 {os.path.basename(MAIN_OPPONENT_PATH)} 发起挑战")
        
        print(f"\n⚔️  阶段三: 启动镜像对局评估...")
        win_rate = evaluate_models(CHALLENGER_PATH, MAIN_OPPONENT_PATH, show_progress=True)
        
        print(f"\n👑 阶段四: 决策...")
        challenger_name = os.path.basename(CHALLENGER_PATH)
        main_opponent_name = os.path.basename(MAIN_OPPONENT_PATH)

        self._update_elo(challenger_name, main_opponent_name, win_rate)
        
        if win_rate > EVALUATION_THRESHOLD:
            print(f"🏆 挑战成功 (胜率 {win_rate:.2%} > {EVALUATION_THRESHOLD:.2%})！新主宰者诞生！")
            
            old_main_gen = self.latest_generation + 1
            new_opponent_name = f"opponent_{old_main_gen}.zip"
            new_opponent_path = os.path.join(OPPONENT_POOL_DIR, new_opponent_name)
            
            shutil.copy(MAIN_OPPONENT_PATH, new_opponent_path)
            self.elo_ratings[new_opponent_name] = self.elo_ratings[main_opponent_name]
            print(f"旧主宰者 {main_opponent_name} 已存入对手池，名为 {new_opponent_name}")
            
            shutil.copy(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
            self.elo_ratings[main_opponent_name] = self.elo_ratings[challenger_name]
            print(f"挑战者已成为新主宰者！")

            self._manage_opponent_pool(new_opponent_path=new_opponent_path)
            
            print(f"🔥 发送指令，在所有 {N_ENVS} 个并行环境中更新对手池...")
            self.env.env_method("reload_opponent_pool", new_opponent_data=self.combined_opponent_data)
            print("✅ 所有环境中的对手池均已成功更新！")
            
            return True
        else:
            print(f"🛡️  挑战失败 (胜率 {win_rate:.2%} <= {EVALUATION_THRESHOLD:.2%})。主宰者保持不变。")
            print("...挑战者将继续训练以发起下一次挑战。")
            self._save_elo_and_generations()
            return False

    def run(self):
        """启动并执行完整的自我对弈训练流程。"""
        assert self.model is not None, "Model not initialized"
        assert self.env is not None, "Environment not initialized"
        try:
            self._prepare_environment_and_models()
            print("\n--- [步骤 3/5] 开始Elo自我对弈主循环 ---")
            successful_challenges = 0
            
            total_decay_loops = min(TOTAL_TRAINING_LOOPS, SHAPING_DECAY_END_LOOP)
            if total_decay_loops > 0:
                decay_per_loop = (SHAPING_COEF_INITIAL - SHAPING_COEF_FINAL) / total_decay_loops
            else:
                decay_per_loop = 0
            
            for i in range(1, TOTAL_TRAINING_LOOPS + 1):
                print(f"\n{'='*70}\n🔄 训练循环 {i}/{TOTAL_TRAINING_LOOPS} | 成功挑战次数: {successful_challenges}\n{'='*70}")
                try:
                    if SHAPING_COEF_INITIAL > SHAPING_COEF_FINAL:
                        if i <= total_decay_loops:
                            current_coef = SHAPING_COEF_INITIAL - (i * decay_per_loop)
                        else:
                            current_coef = SHAPING_COEF_FINAL
                        
                        self.env.set_attr("shaping_coef", current_coef)
                        
                        if PPO_VERBOSE > 0 and (i < total_decay_loops + 1):
                            actual_coef = self.env.get_attr("shaping_coef")[0]
                            print(f"      [INFO] 奖励塑形系数 (shaping_coef) 已更新为: {actual_coef:.4f}")

                    self._train_learner()
                    if self._evaluate_and_update():
                        successful_challenges += 1
                except Exception as e:
                    print(f"⚠️ 训练循环 {i} 出现严重错误: {e}")
                    import traceback
                    traceback.print_exc()
                    print("...继续下一次循环...")
                    continue
            
            self.model.save(FINAL_MODEL_PATH)
            print(f"\n--- [步骤 4/5] 训练完成！ ---")
            
        finally:
            print("\n正在保存最终的状态文件...")
            self._save_elo_and_generations()
            if self.env:
                print("\n--- [步骤 5/5] 清理环境 ---")
                self.env.close()
                print("✅ 资源清理完成")

if __name__ == '__main__':
    trainer = SelfPlayTrainer()
    trainer.run()