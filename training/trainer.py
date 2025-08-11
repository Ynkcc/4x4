# training/trainer.py

import os
import shutil
import time
import re
import json
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import MaskablePPO

# 【更新】导入所有需要的常量
from utils.constants import *
from game.environment import GameEnvironment
from training.evaluator import evaluate_models # 使用镜像评估器

class SelfPlayTrainer:
    """
    【V5 最终版】集成了动态Elo评估、对手池轮换和多环境实时同步的训练器。
    """
    def __init__(self):
        """初始化训练器，设置模型和环境为None。"""
        self.model = None
        self.env = None

        # --- 对手池核心属性 ---
        self.opponent_pool_paths = []
        self.opponent_pool_weights = []
        self.opponent_pool_paths_for_env = []

        # Elo评分系统
        self.elo_ratings = {}
        # 【更新】使用常量初始化Elo参数
        self.default_elo = ELO_DEFAULT
        self.elo_k_factor = ELO_K_FACTOR
        
        self._setup()

    def _setup(self):
        """
        执行所有启动前的准备工作。
        """
        print("--- [步骤 1/5] 初始化设置 ---")
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

        if not os.path.exists(MAIN_OPPONENT_PATH):
            print("未找到主宰者模型，将从指定的初始模型开始全新训练。")
            initial_model_candidates = [SELF_PLAY_MODEL_PATH, CURRICULUM_MODEL_PATH]
            initial_model_found = None
            for candidate in initial_model_candidates:
                if os.path.exists(candidate):
                    initial_model_found = candidate
                    print(f"找到初始模型: {candidate}")
                    break
            if not initial_model_found:
                raise FileNotFoundError(f"未找到任何可用的初始模型。尝试过的路径: {initial_model_candidates}")
            
            shutil.copy(initial_model_found, MAIN_OPPONENT_PATH)
            print(f"已将初始模型复制为第一个主宰者: {MAIN_OPPONENT_PATH}")
            
            # 将初始主宰者也加入对手池，作为第一个对手
            initial_opponent_path = os.path.join(OPPONENT_POOL_DIR, "opponent_0.zip")
            if not os.path.exists(initial_opponent_path):
                 shutil.copy(initial_model_found, initial_opponent_path)
                 # 初始化Elo并立即保存
                 self.elo_ratings['opponent_0.zip'] = self.default_elo
                 self._save_elo_ratings()

        self._load_opponent_pool_and_elo()

    def _load_opponent_pool_and_elo(self):
        """
        从磁盘加载所有对手模型，并加载Elo评分。
        """
        print("正在从磁盘加载对手池和Elo评分...")
        self.opponent_pool_paths = []
        
        elo_file = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        if os.path.exists(elo_file):
            try:
                with open(elo_file, 'r') as f:
                    self.elo_ratings = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告：读取Elo文件失败: {e}。将使用默认值。")
                self.elo_ratings = {}
        
        opponent_files = [f for f in os.listdir(OPPONENT_POOL_DIR) if f.endswith('.zip')]
        opponent_files.sort(key=lambda x: int(re.search(r'opponent_(\d+)\.zip', x).group(1)))

        for filename in opponent_files:
            full_path = os.path.join(OPPONENT_POOL_DIR, filename)
            self.opponent_pool_paths.append(full_path)
            if filename not in self.elo_ratings:
                self.elo_ratings[filename] = self.default_elo
        
        print(f"成功加载 {len(self.opponent_pool_paths)} 个对手。")
        self._update_opponent_weights()
    
    def _save_elo_ratings(self):
        """将Elo评分保存到JSON文件。"""
        elo_file = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")
        try:
            with open(elo_file, 'w') as f:
                json.dump(self.elo_ratings, f, indent=4)
        except IOError as e:
            print(f"错误：无法保存Elo评分文件: {e}")

    def _update_elo(self, winner_name, loser_name, win_rate):
        """
        根据镜像对局的实际胜率更新双方的Elo评分。
        """
        winner_elo = self.elo_ratings.get(winner_name, self.default_elo)
        loser_elo = self.elo_ratings.get(loser_name, self.default_elo)

        expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        
        new_winner_elo = winner_elo + self.elo_k_factor * (win_rate - expected_win)
        new_loser_elo = loser_elo - self.elo_k_factor * (win_rate - expected_win)
        
        self.elo_ratings[winner_name] = new_winner_elo
        self.elo_ratings[loser_name] = new_loser_elo
        
        print(f"Elo 更新 (基于胜率 {win_rate:.2%}):")
        print(f"  - {winner_name}: {winner_elo:.0f} -> {new_winner_elo:.0f} (Δ {new_winner_elo - winner_elo:+.1f})")
        print(f"  - {loser_name}: {loser_elo:.0f} -> {new_loser_elo:.0f} (Δ {new_loser_elo - loser_elo:+.1f})")

    def _update_opponent_weights(self):
        """
        根据Elo评分计算采样权重。
        """
        main_opponent_name = "main_opponent.zip"
        if main_opponent_name not in self.elo_ratings:
            self.elo_ratings[main_opponent_name] = self.default_elo
        main_elo = self.elo_ratings[main_opponent_name]
        
        weights = []
        if not self.opponent_pool_paths:
            self.opponent_pool_paths_for_env = [MAIN_OPPONENT_PATH]
            self.opponent_pool_weights = [1.0]
            return

        # 计算池中每个对手的权重
        for path in self.opponent_pool_paths:
            opp_name = os.path.basename(path)
            opp_elo = self.elo_ratings.get(opp_name, self.default_elo)
            elo_diff = abs(main_elo - opp_elo)
            # 【更新】使用常量设置温度参数
            weight = np.exp(-elo_diff / ELO_WEIGHT_TEMPERATURE)
            weights.append(weight)
        
        main_opponent_weight = sum(weights) * 0.5 if weights else 1.0

        self.opponent_pool_paths_for_env = self.opponent_pool_paths + [MAIN_OPPONENT_PATH]
        all_weights = weights + [main_opponent_weight]

        total_weight = sum(all_weights)
        if total_weight == 0:
            num_opps = len(self.opponent_pool_paths_for_env)
            self.opponent_pool_weights = [1.0 / num_opps] * num_opps if num_opps > 0 else []
        else:
            self.opponent_pool_weights = [w / total_weight for w in all_weights]

        print("对手池采样权重已更新:")
        for path, weight in zip(self.opponent_pool_paths_for_env, self.opponent_pool_weights):
            elo = self.elo_ratings.get(os.path.basename(path), self.default_elo)
            print(f"  - {os.path.basename(path)} (Elo: {elo:.0f}, 权重: {weight:.2%})")

    def _add_new_opponent(self, challenger_elo):
        """
        挑战成功后，执行“主宰者降级入池 -> 挑战者晋升为主宰者 -> 池大小管理”的完整流程。
        """
        print("🔄 正在执行对手池轮换...")

        # 1. 确定新对手的文件名
        opponent_files = [f for f in os.listdir(OPPONENT_POOL_DIR) if f.endswith('.zip')]
        max_num = -1
        for f in opponent_files:
            match = re.search(r'opponent_(\d+)\.zip', f)
            if match:
                max_num = max(max_num, int(match.group(1)))
        new_opponent_num = max_num + 1
        new_opponent_name = f"opponent_{new_opponent_num}.zip"
        new_opponent_path = os.path.join(OPPONENT_POOL_DIR, new_opponent_name)

        # 2. 旧主宰者进入对手池
        old_main_name = "main_opponent.zip"
        if os.path.exists(MAIN_OPPONENT_PATH):
            shutil.copy(MAIN_OPPONENT_PATH, new_opponent_path)
            self.elo_ratings[new_opponent_name] = self.elo_ratings.get(old_main_name, self.default_elo)
            self.opponent_pool_paths.append(new_opponent_path)
            print(f"旧主宰者已存入对手池: {new_opponent_name} (Elo: {self.elo_ratings[new_opponent_name]:.0f})")

        # 3. 挑战者成为新主宰者
        shutil.copy(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
        self.elo_ratings[old_main_name] = challenger_elo
        print(f"挑战者已成为新主宰者 (Elo: {self.elo_ratings[old_main_name]:.0f})")

        # 4. 管理对手池大小
        if len(self.opponent_pool_paths) > MAX_OPPONENT_POOL_SIZE:
            pool_with_elo = [(p, self.elo_ratings.get(os.path.basename(p), self.default_elo)) for p in self.opponent_pool_paths]
            pool_with_elo.sort(key=lambda x: x[1])
            
            removed_opponent_path, removed_elo = pool_with_elo[0]
            removed_opponent_name = os.path.basename(removed_opponent_path)
            
            self.opponent_pool_paths.remove(removed_opponent_path)
            os.remove(removed_opponent_path)
            if removed_opponent_name in self.elo_ratings:
                del self.elo_ratings[removed_opponent_name]
            
            print(f"对手池已满，移除Elo最低的对手: {removed_opponent_name} (Elo: {removed_elo:.0f})")
            
        self._save_elo_ratings()

    def _prepare_environment_and_models(self):
        """
        准备用于训练的模型和环境。
        """
        print("\n--- [步骤 2/5] 准备环境和模型 ---")
        print(f"创建 {N_ENVS} 个并行的训练环境...")
        vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
        
        self.env = make_vec_env(
            GameEnvironment,
            n_envs=N_ENVS,
            vec_env_cls=vec_env_cls,
            env_kwargs={
                'opponent_pool': self.opponent_pool_paths_for_env,
                'opponent_weights': self.opponent_pool_weights,
            }
        )
        
        print("加载学习者模型...")
        self.model = MaskablePPO.load(
            MAIN_OPPONENT_PATH,
            env=self.env,
            n_steps=512,
            learning_rate=INITIAL_LR,
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
        
        print("重置模型初始训练步数...")
        self.model.num_timesteps = 0
        self.model._total_timesteps = 0
        
        print("✅ 环境和模型准备完成！")

    def _train_learner(self, loop_number: int):
        """训练学习者模型。"""
        print(f"🏋️  阶段一: 学习者进行 {STEPS_PER_LOOP:,} 步训练...")
        self.model.learn(
            total_timesteps=STEPS_PER_LOOP,
            reset_num_timesteps=False,
            progress_bar=True
        )

    def _evaluate_and_update(self, loop_number: int) -> bool:
        """
        评估、决策、更新Elo、轮换对手、同步环境的完整流程。
        """
        print(f"\n💾 阶段二: 保存学习者为挑战者模型 -> {os.path.basename(CHALLENGER_PATH)}")
        self.model.save(CHALLENGER_PATH)
        time.sleep(0.5)
        
        print(f"\n⚔️  阶段三: 启动镜像对局评估...")
        win_rate = evaluate_models(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
        
        print(f"\n👑 阶段四: 决策...")
        challenger_name = "challenger_temp"
        main_opponent_name = "main_opponent.zip"
        
        if win_rate > EVALUATION_THRESHOLD:
            print(f"🏆 挑战成功 (胜率 {win_rate:.2%} > {EVALUATION_THRESHOLD:.2%})！新主宰者诞生！")
            
            self._update_elo(challenger_name, main_opponent_name, win_rate)
            challenger_final_elo = self.elo_ratings.pop(challenger_name)
            self._add_new_opponent(challenger_final_elo)
            self._update_opponent_weights()
            
            print(f"🔥 发送指令，在所有 {N_ENVS} 个并行环境中更新对手池...")
            try:
                results = self.env.env_method(
                    "reload_opponent_pool",
                    new_pool=self.opponent_pool_paths_for_env,
                    new_weights=self.opponent_pool_weights
                )
                if all(results):
                    print("✅ 所有环境中的对手池均已成功更新！")
                else:
                    print("⚠️ 部分环境未能成功更新对手池。")

                print("🧠 正在将学习者重置为新主宰者的状态...")
                old_logger = self.model.logger
                old_total_timesteps = self.model._total_timesteps
                
                self.model = MaskablePPO.load(MAIN_OPPONENT_PATH, env=self.env)
                
                self.model.set_logger(old_logger)
                self.model.num_timesteps = 0
                self.model._total_timesteps = old_total_timesteps
                
                return True

            except Exception as e:
                raise RuntimeError(f"在更新并行环境中的对手池时发生严重错误: {e}")

        else:
            print(f"🛡️  挑战失败 (胜率 {win_rate:.2%} <= {EVALUATION_THRESHOLD:.2%})。主宰者与对手池保持不变。")
            self._update_elo(main_opponent_name, challenger_name, 1.0 - win_rate)
            self.elo_ratings.pop(challenger_name)
            self._save_elo_ratings()
            return False

    def run(self):
        """
        启动并执行完整的自我对弈训练流程。
        """
        try:
            self._prepare_environment_and_models()
            print("\n--- [步骤 3/5] 开始Elo自我对弈主循环 ---")
            successful_challenges = 0
            
            for i in range(1, TOTAL_TRAINING_LOOPS + 1):
                print(f"\n{'='*70}\n🔄 训练循环 {i}/{TOTAL_TRAINING_LOOPS} | 成功挑战次数: {successful_challenges}\n{'='*70}")
                try:
                    self._train_learner(i)
                    if self._evaluate_and_update(i):
                        successful_challenges += 1
                except Exception as e:
                    print(f"⚠️ 训练循环 {i} 出现严重错误: {e}")
                    import traceback
                    traceback.print_exc()
                    print("...继续下一次循环...")
                    continue
            
            self.model.save(FINAL_MODEL_PATH)
            print(f"\n--- [步骤 4/5] 训练完成！ ---")
            print(f"🎉 最终模型已保存到: {FINAL_MODEL_PATH}")
            print(f"📈 总计成功挑战: {successful_challenges}/{TOTAL_TRAINING_LOOPS}")
            
        finally:
            if hasattr(self, 'env') and self.env:
                print("\n--- [步骤 5/5] 清理环境 ---")
                self.env.close()
                print("✅ 资源清理完成")

if __name__ == '__main__':
    trainer = SelfPlayTrainer()
    trainer.run()