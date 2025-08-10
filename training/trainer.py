# training/trainer.py

import os
import shutil
import time
import re
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
# 【修复】明确导入 SubprocVecEnv 和 DummyVecEnv 以提高代码清晰度
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import MaskablePPO

# 导入本地模块
from utils.constants import *
# 【移除】不再需要学习率调度器
from game.environment import GameEnvironment
from training.evaluator import evaluate_models
# 【修复】不再需要 NeuralAgent，因为它的单例模式在多进程下有问题

class SelfPlayTrainer:
    """
    Elo自我对弈训练器。
    【V3 增强版】引入了对手池机制，并实现了周期性的步数重置。
    """
    def __init__(self):
        """初始化训练器，设置模型和环境为None。"""
        self.model = None
        self.env = None
        # 【新增】对手池相关属性
        self.opponent_pool_paths = []
        self.opponent_pool_weights = []
        self.opponent_counter = 0  # 对手编号计数器
        self.max_recent_opponents = 3  # 固定选取最近的3个对手模型
        self._setup()

    def _setup(self):
        """
        执行所有启动前的准备工作。
        """
        print("--- [步骤 1/5] 初始化设置 ---")
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

        # 确保主宰者模型存在，如果不存在则从初始模型复制
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
        else:
            print(f"检测到已存在的主宰者模型: {MAIN_OPPONENT_PATH}")

        # 更新对手池
        self._initialize_opponent_counter()
        self._update_opponent_pool()

    def _initialize_opponent_counter(self):
        """
        【新增】初始化对手计数器，使用正则表达式从现有文件中恢复计数。
        """
        # 使用正则表达式扫描已存在的对手文件，找到最大的编号
        max_num = 0
        opponent_pattern = re.compile(r'^opponent(\d+)\.zip$')
        
        for filename in os.listdir(SELF_PLAY_OUTPUT_DIR):
            match = opponent_pattern.match(filename)
            if match:
                try:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
                except ValueError:
                    continue
        
        self.opponent_counter = max_num
        print(f"对手计数器初始化为: {self.opponent_counter}")

    def _get_opponent_paths(self):
        """
        【修改】获取当前对手池的路径，包含主宰者和最近的3个对手模型。
        """
        paths = [MAIN_OPPONENT_PATH]  # 主宰者
        
        # 添加最近的3个对手模型
        recent_opponents = []
        for i in range(max(1, self.opponent_counter - self.max_recent_opponents + 1), self.opponent_counter + 1):
            opp_path = os.path.join(SELF_PLAY_OUTPUT_DIR, f"opponent{i}.zip")
            if os.path.exists(opp_path):
                recent_opponents.append(opp_path)
        
        # 按编号降序排列，确保最新的在前面
        recent_opponents.sort(key=lambda x: int(re.search(r'opponent(\d+)\.zip', x).group(1)), reverse=True)
        
        # 只取最近的3个
        paths.extend(recent_opponents[:self.max_recent_opponents])
        
        return paths

    def _update_opponent_pool(self):
        """
        【修改】根据磁盘上的文件更新当前的对手池和权重。
        """
        print("正在更新对手池...")
        
        # 获取所有现有的对手模型路径
        self.opponent_pool_paths = self._get_opponent_paths()
        
        # 设置权重：主宰者权重为2，其余为1
        if len(self.opponent_pool_paths) == 1:
            # 只有主宰者
            self.opponent_pool_weights = [1.0]
            print("对手池仅包含主宰者模型。")
        else:
            # 主宰者 + 最近的对手
            weights = [2.0] + [1.0] * (len(self.opponent_pool_paths) - 1)
            # 归一化权重
            total_weight = sum(weights)
            self.opponent_pool_weights = [w / total_weight for w in weights]
            print(f"对手池已更新 (1个主宰者 + {len(self.opponent_pool_paths)-1}个最近对手，总计: {self.opponent_counter}个历史对手)。")
            
        for path, weight in zip(self.opponent_pool_paths, self.opponent_pool_weights):
            print(f"  - {os.path.basename(path)} (权重: {weight:.2f})")

    def _add_new_opponent(self):
        """
        【修改】添加新对手到对手池。
        当挑战成功时：
        1. 旧主宰者保存为新的对手模型（递增编号）
        2. 挑战者成为新主宰者
        3. 保留所有历史对手模型，但对手池只选取最近的3个
        """
        print("🔄 正在添加新对手...")
        
        # 增加对手计数器
        self.opponent_counter += 1
        
        # 保存旧主宰者为新对手
        new_opponent_path = os.path.join(SELF_PLAY_OUTPUT_DIR, f"opponent{self.opponent_counter}.zip")
        if os.path.exists(MAIN_OPPONENT_PATH):
            shutil.copy(MAIN_OPPONENT_PATH, new_opponent_path)
            print(f"旧主宰者已保存为: {os.path.basename(new_opponent_path)}")
        
        # 挑战者成为新主宰者
        shutil.copy(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
        print(f"挑战者已成为新主宰者: {os.path.basename(MAIN_OPPONENT_PATH)}")
        
        print(f"✅ 对手池更新完成！当前共有 {self.opponent_counter} 个历史对手模型")

    def _prepare_environment_and_models(self):
        """
        【修改】准备用于训练的模型和环境。
        现在将整个对手池信息传递给环境。
        """
        print("\n--- [步骤 2/5] 准备环境和模型 ---")
        
        print(f"创建 {N_ENVS} 个并行的训练环境...")
        vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
        
        self.env = make_vec_env(
            GameEnvironment,
            n_envs=N_ENVS,
            vec_env_cls=vec_env_cls,
            env_kwargs={
                # 【关键点】将对手池和权重注入每个环境
                'opponent_pool': self.opponent_pool_paths,
                'opponent_weights': self.opponent_pool_weights,
            }
        )
        
        print("加载学习者模型...")
        # 学习者总是从当前最强的主宰者模型开始学习
        learner_start_path = MAIN_OPPONENT_PATH
        self.model = MaskablePPO.load(
            learner_start_path,
            env=self.env,
            n_steps=512,
            learning_rate=INITIAL_LR,
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
        
        # 注释掉重置步数，保持训练统计信息的连续性
        print("重置模型初始训练步数...")
        self.model.num_timesteps = 0
        self.model._total_timesteps = 0
        
        print("✅ 环境和模型准备完成！")

    def _train_learner(self, loop_number: int):
        """
        训练学习者模型。
        """
        print(f"🏋️  阶段一: 学习者进行 {STEPS_PER_LOOP:,} 步训练...")
        self.model.learn(
            total_timesteps=STEPS_PER_LOOP,
            reset_num_timesteps=False,
            progress_bar=True
        )

    def _evaluate_and_update(self, loop_number: int) -> bool:
        """
        【修改】评估挑战者，如果成功，则轮换对手池并重置步数。
        """
        print(f"\n💾 阶段二: 保存学习者为挑战者模型 -> {os.path.basename(CHALLENGER_PATH)}")
        self.model.save(CHALLENGER_PATH)
        time.sleep(0.5)
        
        print(f"\n⚔️  阶段三: 启动Elo评估...")
        win_rate = evaluate_models(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
        
        print(f"\n👑 阶段四: 决策...")
        if win_rate > EVALUATION_THRESHOLD:
            print(f"🏆 挑战成功 (胜率 {win_rate:.2%} > {EVALUATION_THRESHOLD:.2%})！新主宰者诞生！")
            
            # 步骤1: 添加新对手并更新主宰者
            self._add_new_opponent()
            
            # 步骤2: 更新内部的对手池配置
            self._update_opponent_pool()
            
            # 步骤3: 命令所有并行环境重新加载新的对手池
            print(f"🔥 发送指令，在所有 {N_ENVS} 个并行环境中更新对手池...")
            try:
                self.env.env_method(
                    "reload_opponent_pool",
                    new_pool=self.opponent_pool_paths,
                    new_weights=self.opponent_pool_weights
                )
                print("✅ 所有环境中的对手池均已更新！")
                
                # 步骤4: 将学习者模型重置为新主宰者的状态
                print("🧠 为了学习的连续性，将学习者模型重置为新主宰者的状态...")
                # ...existing code...
                old_model = self.model
                new_model = MaskablePPO.load(MAIN_OPPONENT_PATH, env=self.env)
                # 迁移日志与步数，确保训练与可视化连续
                if hasattr(old_model, "logger") and hasattr(new_model, "set_logger"):
                    new_model.set_logger(old_model.logger)
                new_model.num_timesteps = getattr(old_model, "num_timesteps", 0)
                if hasattr(old_model, "_total_timesteps"):
                    new_model._total_timesteps = old_model._total_timesteps
                self.model = new_model
                # ...existing code...
                
                # 【核心修改】步骤5: 注释掉重置训练步数，保持TensorBoard日志连续性
                # 重置步数会导致训练指标丢失，影响训练过程的监控
                # print("🔄 一个新时代开始，重置训练步数...")
                # self.model.num_timesteps = 0
                # self.model._total_timesteps = 0

                return True
            except Exception as e:
                print(f"⚠️ 对手模型更新失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"🛡️  挑战失败 (胜率 {win_rate:.2%} <= {EVALUATION_THRESHOLD:.2%})。主宰者与对手池保持不变。")
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
                print(f"\n{'='*70}")
                print(f"🔄 训练循环 {i}/{TOTAL_TRAINING_LOOPS} | 成功挑战次数: {successful_challenges}")
                print(f"{'='*70}")
                
                try:
                    self._train_learner(i)
                    if self._evaluate_and_update(i):
                        successful_challenges += 1
                    
                except Exception as e:
                    print(f"⚠️ 训练循环 {i} 出现错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.model.save(FINAL_MODEL_PATH)
            print(f"\n--- [步骤 4/5] 训练完成！ ---")
            print(f"🎉 最终模型已保存到: {FINAL_MODEL_PATH}")
            print(f"📈 总计成功挑战: {successful_challenges}/{TOTAL_TRAINING_LOOPS}")
            
        except Exception as e:
            print(f"❌ 训练过程中发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            if hasattr(self, 'env') and self.env:
                print("\n--- [步骤 5/5] 清理环境 ---")
                self.env.close()
                print("✅ 资源清理完成")