# train_and_eval_simple.py
import os
import warnings

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 禁用INFO和WARNING日志
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# continuous_training.py
"""
统一的持续训练脚本 - 整合评估、训练、分析功能
每次运行训练81920步，支持从任何阶段继续训练
"""
import os
import numpy as np
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO

# 导入本地模块
from game.environment import (
    GameEnvironment, PieceType, PIECE_VALUES,
    REVEAL_ACTIONS_COUNT, REGULAR_MOVE_ACTIONS_COUNT
)
from game.policy import CustomActorCriticPolicy
from training.simple_agent import SimpleAgent
from utils.scheduler import linear_schedule
from utils.constants import INITIAL_LR, N_ENVS

class ContinuousTrainer:
    """持续训练管理器"""

    def __init__(self):
        # 训练配置
        self.STEPS_PER_SESSION = 81920  # 每次训练步数
        self.EVALUATION_GAMES = 1000    # 评估局数
        # --- 【修改一】: 大幅降低学习率，从 5e-4 降至 5e-5 ---
        self.ENHANCED_LR = 5e-5

        # 路径配置
        self.BASE_MODEL_PATH = "./models/self_play_final/main_opponent.zip"
        self.CURRENT_MODEL_PATH = "./models/continuous_train/current_model.zip"
        self.BACKUP_MODEL_PATH = "./models/continuous_train/backup_model.zip"
        self.LOG_DIR = "./tensorboard_logs/continuous_train/"
        self.PROGRESS_FILE = "./models/continuous_train/training_progress.txt"

        # 创建目录
        os.makedirs(os.path.dirname(self.CURRENT_MODEL_PATH), exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

        # 初始化SimpleAgent
        self._setup_opponent()

        # 加载训练进度
        self.session_count = 0
        self.best_winrate = 0.0
        self._load_progress()

    def _setup_opponent(self):
        """设置智能对手"""
        print("正在初始化智能对手...")
        temp_env = GameEnvironment()
        action_to_coords = temp_env.action_to_coords
        temp_env.close()

        self.simple_opponent = SimpleAgent(
            action_to_coords=action_to_coords,
            piece_values=PIECE_VALUES,
            piece_types=PieceType,
            reveal_actions_count=REVEAL_ACTIONS_COUNT,
            regular_move_actions_count=REGULAR_MOVE_ACTIONS_COUNT
        )
        print("✓ 智能对手初始化完成")

    def _load_progress(self):
        """加载训练进度"""
        if os.path.exists(self.PROGRESS_FILE):
            try:
                with open(self.PROGRESS_FILE, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("session_count:"):
                            self.session_count = int(line.split(":")[1].strip())
                        elif line.startswith("best_winrate:"):
                            self.best_winrate = float(line.split(":")[1].strip())
                print(f"✓ 加载进度: 会话{self.session_count}, 最佳胜率{self.best_winrate:.2%}")
            except Exception as e:
                print(f"⚠️ 加载进度失败: {e}")

    def _save_progress(self, winrate):
        """保存训练进度"""
        try:
            with open(self.PROGRESS_FILE, 'w') as f:
                f.write(f"session_count: {self.session_count}\n")
                f.write(f"best_winrate: {winrate}\n")
                f.write(f"last_update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"⚠️ 保存进度失败: {e}")

    def _get_current_model_path(self):
        """获取当前应该使用的模型路径"""
        if os.path.exists(self.CURRENT_MODEL_PATH):
            return self.CURRENT_MODEL_PATH
        elif os.path.exists(self.BASE_MODEL_PATH):
            return self.BASE_MODEL_PATH
        else:
            raise FileNotFoundError("找不到可用的模型文件")

    def evaluate_model(self, model_path: str, n_games: int = None) -> dict:
        """
        评估模型性能
        :param model_path: 模型路径
        :param n_games: 评估局数，默认使用配置值
        :return: 评估结果字典
        """
        if n_games is None:
            n_games = self.EVALUATION_GAMES

        print(f"\n--- [评估阶段] ---")
        print(f"模型: {os.path.basename(model_path)}")
        print(f"对手: SimpleAgent (智能策略)")
        print(f"评估局数: {n_games}")

        eval_env = None
        try:
            eval_env = make_vec_env(
                GameEnvironment,
                n_envs=N_ENVS,
                vec_env_cls=DummyVecEnv,
                env_kwargs={
                    'curriculum_stage': 4,
                    'opponent_agent': self.simple_opponent
                }
            )

            model = MaskablePPO.load(model_path, env=eval_env, device='auto')

            games_played = 0
            wins = 0
            draws = 0
            losses = 0
            all_rewards = []

            obs = eval_env.reset()

            while games_played < n_games:
                action_masks = np.array(eval_env.env_method("action_masks"), dtype=np.int32)
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, rewards, dones, infos = eval_env.step(action)

                for i, done in enumerate(dones):
                    if done:
                        games_played += 1
                        winner = infos[i].get('winner')
                        episode_reward = rewards[i]
                        all_rewards.append(episode_reward)

                        if winner == 1:
                            wins += 1
                        elif winner == -1:
                            losses += 1
                        else:
                            draws += 1

                        if games_played % 100 == 0:
                            print(f"  评估进度: {games_played}/{n_games} | 当前战绩: {wins}胜/{losses}负/{draws}平", end="\r")

                        if games_played >= n_games:
                            break

            print("\n--- 评估完成 ---")
            total_decisive = wins + losses
            winrate = wins / total_decisive if total_decisive > 0 else 0.0
            avg_reward = np.mean(all_rewards) if all_rewards else 0.0

            result = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'winrate': winrate,
                'avg_reward': avg_reward,
                'total_games': games_played
            }

            print(f"最终战绩: {wins}胜 / {losses}负 / {draws}平")
            print(f"胜率: {winrate:.2%}")
            print(f"平均奖励: {avg_reward:.3f}")

            return result

        finally:
            if eval_env:
                eval_env.close()

    def train_session(self, model_path: str) -> str:
        """
        执行一次训练会话
        :param model_path: 起始模型路径
        :return: 训练后模型路径
        """
        print(f"\n--- [训练会话 #{self.session_count + 1}] ---")
        print(f"起始模型: {os.path.basename(model_path)}")
        print(f"训练步数: {self.STEPS_PER_SESSION:,}")
        print(f"学习率: {self.ENHANCED_LR}") # 已降低
        print(f"并行环境数: {N_ENVS}")
        print("-" * 50)

        train_env = None
        try:
            # 创建训练环境
            train_env = make_vec_env(
                GameEnvironment,
                n_envs=N_ENVS,
                vec_env_cls=DummyVecEnv,  # 使用DummyVecEnv避免段错误
                env_kwargs={
                    'curriculum_stage': 4,
                    'opponent_agent': self.simple_opponent
                }
            )

            # 加载或创建模型
            if os.path.exists(model_path):
                model = MaskablePPO.load(
                    model_path,
                    env=train_env,
                    learning_rate=self.ENHANCED_LR,
                    tensorboard_log=self.LOG_DIR,
                    # --- 【修改二】: 增加 n_steps 使更新更稳定 ---
                    n_steps=4096,
                    # --- 【修改三】: 明确 gamma 值 ---
                    gamma=0.99
                )
                print(f"✓ 从现有模型继续训练 (已应用新的稳定化超参数)")
            else:
                model = MaskablePPO(
                    CustomActorCriticPolicy,
                    train_env,
                    learning_rate=linear_schedule(self.ENHANCED_LR),
                    verbose=1,
                    tensorboard_log=self.LOG_DIR,
                    gamma=0.99,
                    n_steps=4096 # 同样应用于新模型
                )
                print(f"✓ 创建新模型开始训练")

            # 设置模型参数
            model.verbose = 1
            model.tensorboard_log = self.LOG_DIR

            # 开始训练
            print(f"\n🚀 开始训练...")
            model.learn(
                total_timesteps=self.STEPS_PER_SESSION,
                progress_bar=True,
                reset_num_timesteps=False
            )

            # 保存模型
            # 备份当前模型
            if os.path.exists(self.CURRENT_MODEL_PATH):
                # 【优化】使用 shutil.move 代替 os.rename, 更健壮
                import shutil
                shutil.move(self.CURRENT_MODEL_PATH, self.BACKUP_MODEL_PATH)


            # 保存新训练的模型
            model.save(self.CURRENT_MODEL_PATH)
            print(f"✓ 模型已保存到: {self.CURRENT_MODEL_PATH}")

            self.session_count += 1
            return self.CURRENT_MODEL_PATH

        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
            return model_path
        except Exception as e:
            print(f"\n❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return model_path
        finally:
            if train_env:
                train_env.close()

    def run_continuous_training(self):
        """运行持续训练流程"""
        print("=" * 60)
        print("           🤖 持续训练系统 🤖")
        print("=" * 60)
        print(f"每次训练步数: {self.STEPS_PER_SESSION:,}")
        print(f"当前会话: #{self.session_count}")
        print(f"历史最佳胜率: {self.best_winrate:.2%}")
        print("=" * 60)

        # 1. 获取当前模型
        try:
            current_model = self._get_current_model_path()
            print(f"✓ 使用模型: {os.path.basename(current_model)}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

        # 2. 训练前评估
        print(f"\n📊 训练前评估...")
        pre_result = self.evaluate_model(current_model)
        pre_winrate = pre_result['winrate']

        # 3. 执行训练
        trained_model = self.train_session(current_model)

        # 4. 训练后评估
        print(f"\n📊 训练后评估...")
        post_result = self.evaluate_model(trained_model)
        post_winrate = post_result['winrate']

        # 5. 结果分析
        improvement = post_winrate - pre_winrate
        relative_improvement = (improvement / pre_winrate * 100) if pre_winrate > 0 else 0

        print(f"\n" + "=" * 60)
        print(f"           📈 训练会话 #{self.session_count} 结果")
        print("=" * 60)
        print(f"训练前胜率: {pre_winrate:.2%}")
        print(f"训练后胜率: {post_winrate:.2%}")
        print(f"绝对提升: {improvement:+.2%}")
        if pre_winrate > 0:
            print(f"相对提升: {relative_improvement:+.1f}%")

        # 更新最佳记录
        if post_winrate > self.best_winrate:
            self.best_winrate = post_winrate
            print(f"🎉 新的最佳胜率记录！")

            # 创建最佳模型备份
            best_model_path = f"./models/continuous_train/best_model_session_{self.session_count}.zip"
            if os.path.exists(trained_model):
                # 【优化】使用 shutil.copy2 保证元数据也被复制
                import shutil
                shutil.copy2(trained_model, best_model_path)
                print(f"✓ 最佳模型已备份到: {best_model_path}")
        
        # 保存进度
        self._save_progress(post_winrate)

        # 训练建议
        print(f"\n💡 建议:")
        if improvement > 0.001: # 设置一个小的阈值，避免噪音
            print(f"   ✓ 训练有效！建议继续训练")
        else:
            print(f"   ⚠️ 本次训练胜率未提升，但参数已调整，建议再观察一轮。")

        print(f"\n🔄 要继续下一轮训练，请再次运行此脚本")
        print("=" * 60)

def main():
    """主函数"""
    trainer = ContinuousTrainer()
    trainer.run_continuous_training()

if __name__ == "__main__":
    main()