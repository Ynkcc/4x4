# training/trainer.py

import os
import shutil
import time
from stable_baselines3.common.env_util import make_vec_env
# 【修复】明确导入 SubprocVecEnv 以提高代码清晰度
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO

# 导入本地模块
from utils.constants import *
from utils.scheduler import linear_schedule
from game.environment import GameEnvironment
from training.evaluator import evaluate_models
# 【修复】不再需要 NeuralAgent，因为它的单例模式在多进程下有问题

class SelfPlayTrainer:
    """
    Elo自我对弈训练器。
    【V2 修复版】该类封装了整个"训练-评估-更新"的循环逻辑。
    修复了在多进程环境下对手模型无法更新的核心BUG。
    """
    def __init__(self):
        """初始化训练器，设置模型和环境为None。"""
        self.model = None
        self.env = None
        # 【修复】移除了 self.opponent_agent 属性
        self._setup()

    def _setup(self):
        """
        执行所有启动前的准备工作 (逻辑无变化)。
        """
        print("--- [步骤 1/4] 初始化设置 ---")
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

        if os.path.exists(MAIN_OPPONENT_PATH):
            print(f"检测到已存在的主宰者模型: {MAIN_OPPONENT_PATH}")
            print("将从该模型恢复训练。")
            self.start_model_path = MAIN_OPPONENT_PATH
        else:
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
            self.start_model_path = MAIN_OPPONENT_PATH
    
    def _prepare_environment_and_models(self):
        """
        【修复】准备用于训练的模型和环境。
        现在将对手模型的路径直接传递给环境，由环境自行加载。
        """
        print("\n--- [步骤 2/4] 准备环境和模型 ---")
        
        # 1. 【修复】创建持久化训练环境，将初始对手模型路径注入
        print(f"创建 {N_ENVS} 个并行的训练环境...")
        print(f"每个环境将自行加载对手模型: {os.path.basename(self.start_model_path)}")
        
        # 确保 N_ENVS > 1 时使用 SubprocVecEnv
        vec_env_cls = SubprocVecEnv if N_ENVS > 1 else 'auto'
        
        self.env = make_vec_env(
            GameEnvironment,
            n_envs=N_ENVS,
            vec_env_cls=vec_env_cls,
            env_kwargs={
                # 将路径传递给环境的构造函数
                'opponent_model_path': self.start_model_path
            }
        )
        
        # 2. 加载学习者模型
        print(f"加载学习者模型，它将挑战当前主宰者...")
        
        from utils.model_compatibility import setup_legacy_imports
        setup_legacy_imports()
        
        self.model = MaskablePPO.load(
            self.start_model_path,
            env=self.env,
            learning_rate=linear_schedule(INITIAL_LR), 
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
        
        print("✅ 环境和模型准备完成！")

    def _train_learner(self, loop_number: int):
        """
        训练学习者模型 (逻辑无变化)。
        """
        print(f"🏋️  阶段一: 学习者进行 {STEPS_PER_LOOP:,} 步训练...")
        self.model.learn(
            total_timesteps=STEPS_PER_LOOP,
            reset_num_timesteps=False,
            progress_bar=True
        )

    def _evaluate_and_update(self, loop_number: int) -> bool:
        """
        【修复】评估挑战者并决定是否更新主宰者。
        使用 env_method 在所有子进程中更新对手模型。
        """
        print(f"\n💾 阶段二: 保存学习者为挑战者模型 -> {os.path.basename(CHALLENGER_PATH)}")
        self.model.save(CHALLENGER_PATH)
        time.sleep(0.5)
        
        print(f"\n⚔️  阶段三: 启动Elo评估...")
        # 注意：evaluate_models 内部使用独立的 DummyVecEnv，它的逻辑是正确的，无需修改。
        win_rate = evaluate_models(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
        
        print(f"\n👑 阶段四: 决策...")
        if win_rate > EVALUATION_THRESHOLD:
            print(f"🏆 挑战成功 (胜率 {win_rate:.2%} > {EVALUATION_THRESHOLD:.2%})！新主宰者诞生！")
            
            # 步骤1: 更新磁盘上的主宰者模型文件
            print(f"💾 持久化新主宰者模型到磁盘: {os.path.basename(MAIN_OPPONENT_PATH)}")
            shutil.copy(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
            time.sleep(0.5)
            
            # 【核心修正】步骤2: 命令所有并行环境从磁盘重新加载最新的主宰者模型
            print(f"🔥 发送指令，在所有 {N_ENVS} 个并行环境中更新对手模型...")
            try:
                # 调用在 GameEnvironment 中新增的 `update_opponent` 方法
                self.env.env_method("update_opponent", new_model_path=MAIN_OPPONENT_PATH)
                print("✅ 所有环境中的对手模型均已更新！")
                
                # 【重要】在对手更新后，学习者模型也应该从新的主宰者权重开始下一轮学习，
                # 而不是继续基于旧的权重。这能确保学习的连续性。
                print("🧠 为了学习的连续性，将学习者模型重置为新主宰者的状态...")
                self.model.load(MAIN_OPPONENT_PATH, env=self.env)
                
                return True
            except Exception as e:
                print(f"⚠️ 对手模型更新失败: {e}")
                import traceback
                traceback.print_exc()
                # 这是一个严重错误，可能导致训练发散，可以选择停止或继续
                return False
        else:
            print(f"🛡️  挑战失败 (胜率 {win_rate:.2%} <= {EVALUATION_THRESHOLD:.2%})。主宰者保持不变。")
            return False

    def run(self):
        """
        启动并执行完整的自我对弈训练流程 (逻辑无变化)。
        """
        try:
            self._prepare_environment_and_models()

            print("\n--- [步骤 3/4] 开始Elo自我对弈主循环 ---")
            successful_challenges = 0
            
            for i in range(1, TOTAL_TRAINING_LOOPS + 1):
                # 【修复】获取当前模型信息的方式略有改变，但更清晰
                current_opponent_path = self.env.get_attr("opponent_model_path", indices=0)[0]
                current_opponent_name = os.path.basename(current_opponent_path)
                
                print(f"\n{'='*70}")
                print(f"🔄 训练循环 {i}/{TOTAL_TRAINING_LOOPS} | 当前对手: {current_opponent_name}")
                print(f"{'='*70}")
                
                try:
                    self._train_learner(i)
                    if self._evaluate_and_update(i):
                        successful_challenges += 1
                    
                    print(f"📊 统计: 成功挑战次数 {successful_challenges}/{i}")
                    
                except Exception as e:
                    print(f"⚠️ 训练循环 {i} 出现错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.model.save(FINAL_MODEL_PATH)
            print(f"\n--- [步骤 4/4] 训练完成！ ---")
            print(f"🎉 最终模型已保存到: {FINAL_MODEL_PATH}")
            print(f"📈 总计成功挑战: {successful_challenges}/{TOTAL_TRAINING_LOOPS}")
            
        except Exception as e:
            print(f"❌ 训练过程中发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            if hasattr(self, 'env') and self.env:
                print("🧹 清理训练环境...")
                self.env.close()
            print("✅ 资源清理完成")