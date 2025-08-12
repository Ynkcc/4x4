# training/trainer.py

import os
import shutil
import time
import re
import json
import numpy as np
import sys

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import MaskablePPO

from utils.constants import *
from game.environment import GameEnvironment
from game.policy import CustomActorCriticPolicy
from training.evaluator import evaluate_models

def create_new_ppo_model(env=None, tensorboard_log=None):
    """
    创建一个全新的随机初始化的PPO模型。
    """
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
            'features_extractor_kwargs': {
                'features_dim': NETWORK_FEATURES_DIM,
                'num_res_blocks': NETWORK_NUM_RES_BLOCKS,
                'num_hidden_channels': NETWORK_NUM_HIDDEN_CHANNELS
            }
        }
    )
    return model

def load_ppo_model_with_hyperparams(model_path: str, env=None, tensorboard_log=None):
    """
    加载PPO模型并应用自定义超参数。
    """
    model = MaskablePPO.load(
        model_path,
        env=env,
        learning_rate=INITIAL_LR,
        clip_range=PPO_CLIP_RANGE,
        tensorboard_log=tensorboard_log,
        n_steps=PPO_N_STEPS,
        device=PPO_DEVICE
    )
    model.batch_size = PPO_BATCH_SIZE
    model.n_epochs = PPO_N_EPOCHS
    model.gae_lambda = PPO_GAE_LAMBDA
    model.vf_coef = PPO_VF_COEF
    model.ent_coef = PPO_ENT_COEF
    model.max_grad_norm = PPO_MAX_GRAD_NORM
    return model

class SelfPlayTrainer:
    """
    【V6 重构版】
    - 以 "挑战者" 为核心进行持续训练。
    - 对手池分为 "长期" 和 "短期" 池。
    - 实现了更科学的历史模型保留和采样机制。
    """
    def __init__(self):
        self.model = None
        self.env = None
        self.tensorboard_log_run_path = None
        
        # --- 对手池核心属性 (重构) ---
        self.long_term_pool_paths = []
        self.short_term_pool_paths = []
        self.opponent_pool_for_env = []
        self.opponent_pool_weights = []

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
        """从JSON文件加载Elo评分和模型代数。"""
        elo_file = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        if os.path.exists(elo_file):
            try:
                with open(elo_file, 'r') as f:
                    data = json.load(f)
                    self.elo_ratings = data.get("elo", {})
                    self.model_generations = data.get("generations", {})
                    self.latest_generation = data.get("latest_generation", 0)
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"警告：读取Elo文件失败或格式不完整: {e}。将使用默认值。")
                self.elo_ratings = {}
                self.model_generations = {}
                self.latest_generation = 0
    
    def _save_elo_and_generations(self):
        """将Elo和模型代数保存到同一个JSON文件。"""
        elo_file = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        data = {
            "elo": self.elo_ratings,
            "generations": self.model_generations,
            "latest_generation": self.latest_generation
        }
        try:
            with open(elo_file, 'w') as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            print(f"错误：无法保存Elo评分文件: {e}")

    def _manage_opponent_pool(self, new_opponent_path=None):
        """
        【核心重构】管理长期和短期对手池。
        - 添加新对手（如果提供）。
        - 根据规则对池进行修剪。
        """
        # 如果有新对手加入（即被击败的主宰者），则增加代数并记录
        if new_opponent_path:
            self.latest_generation += 1
            new_opponent_name = os.path.basename(new_opponent_path)
            self.model_generations[new_opponent_name] = self.latest_generation
            self._save_elo_and_generations()

        # 1. 从磁盘加载所有对手，并按代数排序
        all_opponents = []
        for filename in os.listdir(OPPONENT_POOL_DIR):
            if filename.endswith('.zip'):
                gen = self.model_generations.get(filename, 0)
                all_opponents.append((filename, gen))
        
        all_opponents.sort(key=lambda x: x[1], reverse=True) # 按代数降序排，最新的在前

        # 2. 清空现有池
        self.short_term_pool_paths = []
        self.long_term_pool_paths = []

        # 3. 填充短期池
        self.short_term_pool_paths = [os.path.join(OPPONENT_POOL_DIR, name) for name, gen in all_opponents[:SHORT_TERM_POOL_SIZE]]

        # 4. 填充长期池
        # 从短期池之后开始考虑
        candidates_for_long_term = all_opponents[SHORT_TERM_POOL_SIZE:]
        
        # 长期池的保留规则：与最新模型的代数差为2的指数
        for opp_name, opp_gen in candidates_for_long_term:
            if len(self.long_term_pool_paths) >= LONG_TERM_POOL_SIZE:
                break
            
            age = self.latest_generation - opp_gen
            if age > 0 and (age & (age - 1) == 0): # 检查age是否是2的幂
                self.long_term_pool_paths.append(os.path.join(OPPONENT_POOL_DIR, opp_name))
        
        # 5. 清理磁盘上不再需要的模型
        current_pool_names = {os.path.basename(p) for p in self.short_term_pool_paths + self.long_term_pool_paths}
        for filename, _ in all_opponents:
            if filename not in current_pool_names:
                print(f"✂️ 清理过时对手: {filename}")
                os.remove(os.path.join(OPPONENT_POOL_DIR, filename))
                self.elo_ratings.pop(filename, None)
                self.model_generations.pop(filename, None)
        
        self._save_elo_and_generations()
        self._update_opponent_weights()

    def _update_opponent_weights(self):
        """
        【重构】根据Elo评分计算采样权重，合并长短期池。
        """
        self.opponent_pool_for_env = self.short_term_pool_paths + self.long_term_pool_paths
        
        # 确保主宰者有Elo
        main_opponent_name = os.path.basename(MAIN_OPPONENT_PATH)
        if main_opponent_name not in self.elo_ratings:
            self.elo_ratings[main_opponent_name] = self.default_elo
        main_elo = self.elo_ratings[main_opponent_name]
        
        weights = []
        # 计算池中每个对手的权重
        for path in self.opponent_pool_for_env:
            opp_name = os.path.basename(path)
            opp_elo = self.elo_ratings.get(opp_name, self.default_elo)
            elo_diff = abs(main_elo - opp_elo)
            weight = np.exp(-elo_diff / ELO_WEIGHT_TEMPERATURE)
            weights.append(weight)
        
        # 主宰者也有一定概率成为对手
        main_opponent_weight = sum(weights) * 0.3 if weights else 1.0

        final_pool_for_env = self.opponent_pool_for_env + [MAIN_OPPONENT_PATH]
        all_weights = weights + [main_opponent_weight]

        total_weight = sum(all_weights)
        if total_weight > 0:
            self.opponent_pool_weights = [w / total_weight for w in all_weights]
        else:
            num_opps = len(final_pool_for_env)
            self.opponent_pool_weights = [1.0 / num_opps] * num_opps if num_opps > 0 else []

        print("\n--- 对手池状态 ---")
        print(f"短期池 ({len(self.short_term_pool_paths)}/{SHORT_TERM_POOL_SIZE}): {[os.path.basename(p) for p in self.short_term_pool_paths]}")
        print(f"长期池 ({len(self.long_term_pool_paths)}/{LONG_TERM_POOL_SIZE}): {[os.path.basename(p) for p in self.long_term_pool_paths]}")
        print("\n对手池采样权重已更新:")
        for path, weight in zip(final_pool_for_env, self.opponent_pool_weights):
            elo = self.elo_ratings.get(os.path.basename(path), self.default_elo)
            print(f"  - {os.path.basename(path)} (Elo: {elo:.0f}, 权重: {weight:.2%})")

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
                'opponent_pool': self.opponent_pool_for_env,
                'opponent_weights': self.opponent_pool_weights,
                # 使用常量中的初始值
                'shaping_coef': SHAPING_COEF_INITIAL 
            }
        )
        
        # 【核心修改】训练器永远只加载和训练挑战者模型
        print(f"加载学习者模型: {os.path.basename(CHALLENGER_PATH)}")
        self.model = load_ppo_model_with_hyperparams(
            CHALLENGER_PATH,
            env=self.env,
            tensorboard_log=self.tensorboard_log_run_path
        )
        print("✅ 环境和模型准备完成！")

    def _train_learner(self):
        """训练学习者模型（即挑战者）。"""
        print(f"🏋️  阶段一: 挑战者进行 {STEPS_PER_LOOP:,} 步训练...")
        start_time = time.time()
        self.model.learn(total_timesteps=STEPS_PER_LOOP, reset_num_timesteps=False, progress_bar=PPO_SHOW_PROGRESS)
        # 训练后立即保存，覆盖旧的挑战者模型
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
        print(f"\n💾 阶段二: {os.path.basename(CHALLENGER_PATH)} 向 {os.path.basename(MAIN_OPPONENT_PATH)} 发起挑战")
        
        print(f"\n⚔️  阶段三: 启动镜像对局评估...")
        win_rate = evaluate_models(CHALLENGER_PATH, MAIN_OPPONENT_PATH, show_progress=True)
        
        print(f"\n👑 阶段四: 决策...")
        challenger_name = os.path.basename(CHALLENGER_PATH)
        main_opponent_name = os.path.basename(MAIN_OPPONENT_PATH)

        self._update_elo(challenger_name, main_opponent_name, win_rate)
        
        if win_rate > EVALUATION_THRESHOLD:
            print(f"🏆 挑战成功 (胜率 {win_rate:.2%} > {EVALUATION_THRESHOLD:.2%})！新主宰者诞生！")
            
            # 1. 确定旧主宰者入池后的新名字
            old_main_gen = self.latest_generation + 1
            new_opponent_name = f"opponent_{old_main_gen}.zip"
            new_opponent_path = os.path.join(OPPONENT_POOL_DIR, new_opponent_name)
            
            # 2. 将旧主宰者复制到对手池
            shutil.copy(MAIN_OPPONENT_PATH, new_opponent_path)
            self.elo_ratings[new_opponent_name] = self.elo_ratings[main_opponent_name]
            print(f"旧主宰者 {main_opponent_name} 已存入对手池，名为 {new_opponent_name}")
            
            # 3. 挑战者晋升为新主宰者
            shutil.copy(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
            # 主宰者的Elo直接继承晋升后的挑战者Elo
            self.elo_ratings[main_opponent_name] = self.elo_ratings[challenger_name]
            print(f"挑战者已成为新主宰者！")

            # 4. 更新对手池结构并重新计算权重
            self._manage_opponent_pool(new_opponent_path=new_opponent_path)
            
            # 5. 在所有并行环境中更新对手池
            print(f"🔥 发送指令，在所有 {N_ENVS} 个并行环境中更新对手池...")
            self.env.env_method("reload_opponent_pool", new_pool=self.opponent_pool_for_env, new_weights=self.opponent_pool_weights)
            print("✅ 所有环境中的对手池均已成功更新！")
            
            return True
        else:
            print(f"🛡️  挑战失败 (胜率 {win_rate:.2%} <= {EVALUATION_THRESHOLD:.2%})。主宰者保持不变。")
            print("...挑战者将继续训练以发起下一次挑战。")
            self._save_elo_and_generations() # 仅保存Elo和代数更新
            return False

    def run(self):
        """启动并执行完整的自我对弈训练流程。"""
        try:
            self._prepare_environment_and_models()
            print("\n--- [步骤 3/5] 开始Elo自我对弈主循环 ---")
            successful_challenges = 0
            
            # --- 【核心修改】奖励塑形衰减逻辑 ---
            total_decay_loops = min(TOTAL_TRAINING_LOOPS, SHAPING_DECAY_END_LOOP)
            if total_decay_loops > 0:
                # 计算每个循环需要衰减的量
                decay_per_loop = (SHAPING_COEF_INITIAL - SHAPING_COEF_FINAL) / total_decay_loops
            else:
                decay_per_loop = 0
            
            for i in range(1, TOTAL_TRAINING_LOOPS + 1):
                print(f"\n{'='*70}\n🔄 训练循环 {i}/{TOTAL_TRAINING_LOOPS} | 成功挑战次数: {successful_challenges}\n{'='*70}")
                try:
                    # --- 在每个循环开始时更新塑形系数 ---
                    if SHAPING_COEF_INITIAL > SHAPING_COEF_FINAL: # 仅当需要衰减时才执行
                        if i <= total_decay_loops:
                            # 线性衰减
                            current_coef = SHAPING_COEF_INITIAL - (i * decay_per_loop)
                        else:
                            # 衰减结束后，保持最终值
                            current_coef = SHAPING_COEF_FINAL
                        
                        # 使用 set_attr 更新所有并行环境的属性
                        self.env.set_attr("shaping_coef", current_coef)
                        
                        # 定期打印当前系数以供监控
                        if PPO_VERBOSE > 0 and (i < total_decay_loops + 1):
                            # get_attr 会返回一个列表，每个环境一个值，我们只取第一个来显示
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
            print("\n正在保存最终的Elo评分和模型代数...")
            self._save_elo_and_generations()
            if self.env:
                print("\n--- [步骤 5/5] 清理环境 ---")
                self.env.close()
                print("✅ 资源清理完成")

if __name__ == '__main__':
    trainer = SelfPlayTrainer()
    trainer.run()