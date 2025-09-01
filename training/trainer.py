# src_code/training/trainer.py

import os
import shutil
import time
import re
import json
import numpy as np
import sys
import multiprocessing as mp
from queue import Empty
from functools import partial

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from sb3_contrib import MaskablePPO
from typing import Dict, Any, List, Optional

from utils.constants import *
from game.environment import GameEnvironment
from game.policy import CustomActorCriticPolicy
from training.evaluator import evaluate_models

# --- 【新增】预测器工作进程函数 ---
def _predictor_worker(
    prediction_q: mp.Queue,
    result_q: mp.Queue,
    initial_opponent_data: List[Dict[str, Any]]
):
    """
    一个在独立进程中运行的工作函数，用于加载模型并处理预测请求。
    """
    print(f"[Predictor Worker, PID: {os.getpid()}] 启动...")

    # 在工作进程内部加载模型
    loaded_models: Dict[str, MaskablePPO] = {}

    def load_models(opponent_data: List[Dict[str, Any]]):
        """加载或重新加载模型"""
        loaded_models.clear()
        for item in opponent_data:
            path = item['path']
            if path not in loaded_models:
                try:
                    loaded_models[path] = MaskablePPO.load(path, device="auto")
                except Exception as e:
                    print(f"[Predictor Worker] 错误：加载模型 {path} 失败: {e}")
        print(f"[Predictor Worker] 已加载/更新 {len(loaded_models)} 个模型。")

    # 初始加载
    load_models(initial_opponent_data)

    while True:
        try:
            # 使用带超时的get，允许定期检查特殊指令
            request = prediction_q.get(timeout=0.1)

            # --- 处理特殊指令 ---
            if isinstance(request, str) and request == "STOP":
                print(f"[Predictor Worker, PID: {os.getpid()}] 收到停止信号，正在退出。")
                break

            if isinstance(request, dict) and request.get("command") == "RELOAD_MODELS":
                print("[Predictor Worker] 收到重新加载模型的指令。")
                load_models(request["data"])
                continue

            # --- 处理预测请求 ---
            worker_id, model_path, observation, action_masks = request

            model = loaded_models.get(model_path)
            if model:
                action, _ = model.predict(observation, action_masks=action_masks, deterministic=True)
                result_q.put((worker_id, int(action)))
            else:
                print(f"[Predictor Worker] 警告：找不到请求的模型 {model_path}，将返回随机动作。")
                valid_actions = np.where(action_masks)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
                result_q.put((worker_id, action))

        except Empty:
            # 队列为空时继续循环，这很正常
            continue
        except (KeyboardInterrupt, EOFError):
            print(f"[Predictor Worker, PID: {os.getpid()}] 进程被中断。")
            break
        except Exception as e:
            print(f"[Predictor Worker] 处理请求时发生未知错误: {e}")

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

def _init_env(env_class, **kwargs):
    """一个简单的辅助函数，用于实例化带参数的环境。"""
    return env_class(**kwargs)

class SelfPlayTrainer:
    """
    【V7 新规则版 + 进程间预测优化】
    - 以 "挑战者" 为核心进行持续训练。
    - 对手池分为 "长期" 和 "短期" 池，采用新的动态差值规则。
    - 实现了更科学的历史模型保留和采样机制。
    - 使用独立的预测器进程来处理所有对手的动作预测，大幅降低内存占用。
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

        # --- 【新增】进程间通信 ---
        self.prediction_q: Optional[mp.Queue] = None
        self.result_q: Optional[mp.Queue] = None
        self.predictor_process: Optional[mp.Process] = None

        self._setup()

    def _setup(self):
        """
        【重构】执行所有启动前的准备工作，管理模型生命周期。
        """
        print("--- [步骤 1/5] 初始化设置 ---")
        os.makedirs(SELF_PLAY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(OPPONENT_POOL_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

        # 设置多进程启动方法 (对Windows/macOS很重要)
        if sys.platform.startswith('win') or sys.platform == 'darwin':
            mp.set_start_method('spawn', force=True)

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
        self._update_opponent_data_structure()

    def _update_opponent_data_structure(self):
        """
        【IPC版】仅创建包含路径和权重的字典列表，不再预加载模型。
        实际的模型加载由预测器进程完成。
        """
        self.combined_opponent_data.clear()

        pool_candidates = [
            {'filename': f, 'pool_type': 'short_term', 'sampling_weight': 1.0}
            for f in self.short_term_pool_paths
        ] + [
            {'filename': f, 'pool_type': 'long_term', 'sampling_weight': LONG_TERM_POOL_WEIGHT_MULTIPLIER}
            for f in self.long_term_pool_paths
        ]

        total_available = len(pool_candidates)
        if total_available < TRAINING_POOL_SAMPLE_SIZE:
            print(f"⚠️  警告：可用模型数量 ({total_available}) 少于所需采样数量 ({TRAINING_POOL_SAMPLE_SIZE})")

        selected_pool_models = []
        if pool_candidates:
            sample_size = min(TRAINING_POOL_SAMPLE_SIZE, total_available)
            weights = np.array([c['sampling_weight'] for c in pool_candidates])
            probs = weights / weights.sum()
            selected_indices = np.random.choice(len(pool_candidates), size=sample_size, replace=False, p=probs)
            selected_pool_models = [pool_candidates[i]['filename'] for i in selected_indices]

        main_opponent_name = os.path.basename(MAIN_OPPONENT_PATH)
        if main_opponent_name not in self.elo_ratings:
            self.elo_ratings[main_opponent_name] = self.default_elo
        main_elo = self.elo_ratings[main_opponent_name]

        # 【修改】这里只收集模型路径
        all_model_names = selected_pool_models + [main_opponent_name]

        calculate_elo_weight = lambda name: np.exp(-abs(main_elo - self.elo_ratings.get(name, self.default_elo)) / ELO_WEIGHT_TEMPERATURE)
        pool_weights = [{'name': name, 'weight': calculate_elo_weight(name)} for name in all_model_names]

        pool_total_weight = sum(w['weight'] for w in pool_weights if w['name'] != main_opponent_name)
        main_current_weight = next((w['weight'] for w in pool_weights if w['name'] == main_opponent_name), 0)

        min_main_weight = pool_total_weight * MAIN_OPPONENT_MIN_WEIGHT_RATIO / (1 - MAIN_OPPONENT_MIN_WEIGHT_RATIO) if (1 - MAIN_OPPONENT_MIN_WEIGHT_RATIO) > 0 else float('inf')

        if main_current_weight < min_main_weight:
            pool_weights = [
                {**w, 'weight': min_main_weight} if w['name'] == main_opponent_name else w
                for w in pool_weights
            ]

        total_weight = sum(w['weight'] for w in pool_weights)

        if total_weight > 0:
            # 【修改】最终数据结构只包含路径和权重，不再包含模型实例
            self.combined_opponent_data = [
                {
                    'path': os.path.join(OPPONENT_POOL_DIR, w['name']) if w['name'] != main_opponent_name else MAIN_OPPONENT_PATH,
                    'weight': w['weight'] / total_weight
                }
                for w in pool_weights
            ]
        else:
            uniform_weight = 1.0 / len(all_model_names) if all_model_names else 0.0
            self.combined_opponent_data = [
                {
                    'path': os.path.join(OPPONENT_POOL_DIR, name) if name != main_opponent_name else MAIN_OPPONENT_PATH,
                    'weight': uniform_weight
                }
                for name in all_model_names
            ]

        print("\n--- 对手池状态 (IPC模式) ---")
        print(f"短期池 ({len(self.short_term_pool_paths)}/{SHORT_TERM_POOL_SIZE}): {self.short_term_pool_paths}")
        print(f"长期池 ({len(self.long_term_pool_paths)}/{LONG_TERM_POOL_SIZE}): {self.long_term_pool_paths}")
        print(f"长期池代数差值指数: {self.long_term_power_of_2} (当前要求差值: {2**self.long_term_power_of_2})")

        print(f"\n对手池最终权重分布 (总模型: {len(self.combined_opponent_data)}):")
        sorted_data = sorted(self.combined_opponent_data, key=lambda x: os.path.basename(x['path']))
        for item in sorted_data:
            name = os.path.basename(item['path'])
            elo = self.elo_ratings.get(name, self.default_elo)
            is_main = "★主宰者" if item['path'] == MAIN_OPPONENT_PATH else ""
            print(f"  - {name:<25} (Elo: {elo:.0f}, 权重: {item['weight']:.2%}) {is_main}")

    def _prepare_environment_and_models(self):
        """【IPC版】准备模型、环境，并启动预测器进程。"""
        print("\n--- [步骤 2/5] 准备环境和模型 (IPC模式) ---")

        # --- 1. 启动预测器进程 ---
        print(">>> 正在启动中央预测器进程...")
        self.prediction_q = mp.Queue()
        self.result_q = mp.Queue()

        # 初始化的对手数据（仅路径和权重）
        initial_opponent_data_for_worker = self.combined_opponent_data.copy()

        self.predictor_process = mp.Process(
            target=_predictor_worker,
            args=(self.prediction_q, self.result_q, initial_opponent_data_for_worker),
            daemon=True # 设置为守护进程，主进程退出时它也会退出
        )
        self.predictor_process.start()
        print(f"✅ 中央预测器进程已启动 (PID: {self.predictor_process.pid})")

        # --- 2. 准备训练环境 ---
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.tensorboard_log_run_path = os.path.join(TENSORBOARD_LOG_PATH, run_name)
        print(f"TensorBoard 日志将保存到: {self.tensorboard_log_run_path}")

        print(f"创建 {N_ENVS} 个并行的训练环境...")
        vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv

        # 【修复】为每个并行环境提供不同的参数
        env_kwargs_list = [{
            'opponent_data': self.combined_opponent_data,
            'shaping_coef': SHAPING_COEF_INITIAL,
            'prediction_q': self.prediction_q,
            'result_q': self.result_q,
            'worker_id': i
        } for i in range(N_ENVS)]
        
        if vec_env_cls == SubprocVecEnv:
            # 手动创建 env_fns 列表
            env_fns = [partial(_init_env, GameEnvironment, **kwargs) for kwargs in env_kwargs_list]
            self.env = SubprocVecEnv(env_fns)
        else: # DummyVecEnv
            # make_vec_env 对 DummyVecEnv 的处理是正确的
            self.env = make_vec_env(
                GameEnvironment, n_envs=N_ENVS, vec_env_cls=DummyVecEnv,
                env_kwargs=env_kwargs_list[0]
            )

        # --- 3. 加载学习者模型 ---
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

            # 【修改】通过队列向预测器和环境广播更新
            print(f"🔥 发送指令，更新中央预测器和所有 {N_ENVS} 个并行环境...")
            if self.prediction_q:
                self.prediction_q.put({
                    "command": "RELOAD_MODELS",
                    "data": self.combined_opponent_data
                })
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
        try:
            self._prepare_environment_and_models()
            assert self.model is not None, "Model not initialized"
            assert self.env is not None, "Environment not initialized"
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

            # --- 【新增】清理IPC资源 ---
            if self.predictor_process and self.predictor_process.is_alive():
                 print("\n--- [步骤 5/5] 清理资源 ---")
                 print(">>> 正在向预测器进程发送停止信号...")
                 if self.prediction_q: self.prediction_q.put("STOP")
                 self.predictor_process.join(timeout=5) # 等待最多5秒
                 if self.predictor_process.is_alive():
                     print(">>> 预测器进程未能正常关闭，强制终止。")
                     self.predictor_process.terminate()
                 else:
                     print(">>> 预测器进程已成功关闭。")

            if self.env:
                print(">>> 正在关闭环境...")
                self.env.close()
            print("✅ 资源清理完成")

if __name__ == '__main__':
    trainer = SelfPlayTrainer()
    trainer.run()