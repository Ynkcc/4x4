# training/trainer.py

import os
import shutil
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO

# 导入本地模块
from utils.constants import *
from utils.scheduler import linear_schedule
from game.environment import GameEnvironment
from training.evaluator import evaluate_models
# 【修改】导入新的 NeuralAgent
from training.neural_agent import NeuralAgent

class SelfPlayTrainer:
    """
    Elo自我对弈训练器。
    该类封装了整个"训练-评估-更新"的循环逻辑。
    """
    def __init__(self):
        """初始化训练器，设置模型和环境为None。"""
        self.model = None
        self.env = None
        # 【修改】持有对手 agent 的实例
        self.opponent_agent = None
        self._setup()

    def _setup(self):
        """
        执行所有启动前的准备工作：
        1. 创建输出目录。
        2. 实现灵活的模型加载逻辑，支持从头开始或恢复训练。
        """
        print("--- [步骤 1/4] 初始化设置 ---")
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

        # 智能加载逻辑：检查是否存在主宰者模型以恢复训练
        if os.path.exists(MAIN_OPPONENT_PATH):
            print(f"检测到已存在的主宰者模型: {MAIN_OPPONENT_PATH}")
            print("将从该模型恢复训练。")
            self.start_model_path = MAIN_OPPONENT_PATH
        else:
            print("未找到主宰者模型，将从指定的初始模型开始全新训练。")
            
            # 尝试多个可能的初始模型位置
            initial_model_candidates = [
                SELF_PLAY_MODEL_PATH,
                CURRICULUM_MODEL_PATH
            ]
            
            initial_model_found = None
            for candidate in initial_model_candidates:
                if os.path.exists(candidate):
                    initial_model_found = candidate
                    print(f"找到初始模型: {candidate}")
                    break
            
            if not initial_model_found:
                raise FileNotFoundError(f"未找到任何可用的初始模型。尝试过的路径: {initial_model_candidates}")
            
            # 将初始模型复制为主宰者，启动训练
            shutil.copy(initial_model_found, MAIN_OPPONENT_PATH)
            print(f"已将初始模型复制为第一个主宰者: {MAIN_OPPONENT_PATH}")
            self.start_model_path = MAIN_OPPONENT_PATH
    
    def _prepare_environment_and_models(self):
        """
        准备用于训练的模型和环境。
        """
        print("\n--- [步骤 2/4] 准备环境和模型 ---")
        # 1. 【修改】创建并初始化对手 Agent (单例)
        print(f"加载当前主宰者 '{os.path.basename(self.start_model_path)}' 到共享 NeuralAgent...")
        self.opponent_agent = NeuralAgent(model_path=self.start_model_path)

        # 2. 创建持久化训练环境，注入对手 agent
        print(f"创建 {N_ENVS} 个并行的训练环境...")
        self.env = make_vec_env(
            GameEnvironment,
            n_envs=N_ENVS,
            env_kwargs={
                'curriculum_stage': 4, 
                'opponent_agent': self.opponent_agent
            }
        )
        
        # 3. 加载学习者模型
        print(f"加载学习者模型，它将挑战当前主宰者...")
        
        # 设置向后兼容性
        from utils.model_compatibility import setup_legacy_imports
        setup_legacy_imports()
        
        self.model = MaskablePPO.load(
            self.start_model_path,
            env=self.env,
            learning_rate=linear_schedule(INITIAL_LR), 
            tensorboard_log=TENSORBOARD_LOG_PATH
        )

    def _train_learner(self, loop_number: int):
        """
        训练学习者模型。
        
        :param loop_number: 当前训练循环编号
        """
        print(f"🏋️  阶段一: 学习者进行 {STEPS_PER_LOOP:,} 步训练...")
        self.model.learn(
            total_timesteps=STEPS_PER_LOOP,
            reset_num_timesteps=False,  # 保持TensorBoard的时间步连续
            progress_bar=True
        )

    def _evaluate_and_update(self, loop_number: int) -> bool:
        """
        评估挑战者并决定是否更新主宰者。
        
        :param loop_number: 当前训练循环编号
        :return: 是否成功更新了主宰者
        """
        # 保存学习者为挑战者
        print(f"\n💾 阶段二: 保存学习者为挑战者模型 -> {os.path.basename(CHALLENGER_PATH)}")
        self.model.save(CHALLENGER_PATH)
        
        # 进行评估
        print(f"\n⚔️  阶段三: 启动Elo评估...")
        win_rate = evaluate_models(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
        
        # 决策是否更新主宰者
        print(f"\n👑 阶段四: 决策...")
        if win_rate > EVALUATION_THRESHOLD:
            print(f"🏆 挑战成功 (胜率 {win_rate:.2%} > {EVALUATION_THRESHOLD:.2%})！新主宰者诞生！")
            shutil.copy(CHALLENGER_PATH, MAIN_OPPONENT_PATH)
            # 【修改】让共享的对手 agent 重新加载模型，这会更新所有环境的对手
            print(f"🔥 更新共享 NeuralAgent 以使用新的主宰者模型...")
            self.opponent_agent.load_model(MAIN_OPPONENT_PATH)
            return True
        else:
            print(f"🛡️  挑战失败 (胜率 {win_rate:.2%} <= {EVALUATION_THRESHOLD:.2%})。主宰者保持不变。")
            return False

    def run(self):
        """
        启动并执行完整的自我对弈训练流程。
        """
        # 准备环境和模型
        self._prepare_environment_and_models()

        print("\n--- [步骤 3/4] 开始Elo自我对弈主循环 ---")
        successful_challenges = 0
        
        for i in range(1, TOTAL_TRAINING_LOOPS + 1):
            # 【修改】从 agent 获取当前模型信息
            current_opponent_name = os.path.basename(self.opponent_agent.get_model_path())
            print(f"\n{'='*70}")
            print(f"🔄 训练循环 {i}/{TOTAL_TRAINING_LOOPS} | 当前对手: {current_opponent_name}")
            print(f"{'='*70}")
            
            # 训练学习者
            self._train_learner(i)
            
            # 评估并可能更新主宰者
            if self._evaluate_and_update(i):
                successful_challenges += 1
            
            print(f"📊 统计: 成功挑战次数 {successful_challenges}/{i}")
        
        # 最终保存
        self.model.save(FINAL_MODEL_PATH)
        print(f"\n--- [步骤 4/4] 训练完成！ ---")
        print(f"🎉 最终模型已保存到: {FINAL_MODEL_PATH}")
        print(f"📈 总计成功挑战: {successful_challenges}/{TOTAL_TRAINING_LOOPS}")
        
        if self.env:
            self.env.close()
