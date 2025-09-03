# src_code/utils/constants.py

import os
import torch

# ==============================================================================
# --- 1. 路径与目录配置 (通常固定不变) ---
# ==============================================================================
# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 训练输出的主目录
SELF_PLAY_OUTPUT_DIR = os.path.join(ROOT_DIR, "models", "self_play_final")

# 对手池专用目录
OPPONENT_POOL_DIR = os.path.join(SELF_PLAY_OUTPUT_DIR, "opponent_pool")

# 核心模型文件路径
MAIN_OPPONENT_PATH = os.path.join(SELF_PLAY_OUTPUT_DIR, "main_opponent.zip")
CHALLENGER_PATH = os.path.join(SELF_PLAY_OUTPUT_DIR, "challenger.zip")
FINAL_MODEL_PATH = os.path.join(SELF_PLAY_OUTPUT_DIR, "final_model.zip")

# TensorBoard 日志路径
TENSORBOARD_LOG_PATH = os.path.join(ROOT_DIR, "tensorboard_logs", "self_play_final")


# ==============================================================================
# --- 2. 游戏环境超参数 (可调优) ---
# ==============================================================================
# 奖励塑形(Reward Shaping)系数的初始值，用于引导模型学习
SHAPING_COEF_INITIAL = 0.001
# 奖励塑形系数衰减后的最终值
SHAPING_COEF_FINAL = 0
# 奖励塑形系数衰减完成所需的训练循环次数
SHAPING_DECAY_END_LOOP = 0


# ==============================================================================
# --- 3. PPO 算法超参数 (主要调优目标) ---
# ==============================================================================
# 学习率
INITIAL_LR = 1e-4
# PPO 裁剪范围 (Clip Range)
PPO_CLIP_RANGE = 0.2
# 每次更新前，每个环境收集的步数
PPO_N_STEPS = 512
# 训练时每个 mini-batch 的大小
PPO_BATCH_SIZE = 512
# 每次更新时，对采集到的数据进行优化的轮数
PPO_N_EPOCHS = 20
# GAE (Generalized Advantage Estimation) 的 lambda 参数
PPO_GAE_LAMBDA = 0.95
# 价值函数损失的权重系数
PPO_VF_COEF = 0.5
# 熵损失的权重系数，用于鼓励探索
PPO_ENT_COEF = 0.01
# 设备选择: 'cuda' 或 'cpu'
PPO_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# --- 4. 自我对弈超参数 (核心调优目标) ---
# ==============================================================================
# 总训练循环次数
TOTAL_TRAINING_LOOPS = 100
# 评估游戏局数 (每轮训练后进行评估)
EVALUATION_GAMES = 100
# 晋级阈值 (胜率超过此值则晋级)
EVALUATION_THRESHOLD = 0.55
# Elo 评分相关
ELO_DEFAULT = 1500  # 默认Elo评分
ELO_K_FACTOR = 32   # Elo K因子
ELO_WEIGHT_TEMPERATURE = 1.0  # 对手采样温度参数
# 主策略ID
MAIN_POLICY_ID = "main_policy"
# 对手策略ID前缀
OPPONENT_POLICY_ID_PREFIX = "opponent_"
# 环境数量
N_ENVS = 4


# ==============================================================================
# --- 5. 网络架构超参数 (可调优) ---
# ==============================================================================
# 残差块数量
NETWORK_NUM_RES_BLOCKS = 4
# 隐藏通道数
NETWORK_NUM_HIDDEN_CHANNELS = 128
# 标量编码器输出维度
SCALAR_ENCODER_OUTPUT_DIM = 64


# ==============================================================================
# --- 6. 调试与开发相关 ---
# ==============================================================================
# 是否使用固定种子进行训练 (用于重现结果)
USE_FIXED_SEED_FOR_TRAINING = False
# 固定种子值
FIXED_SEED_VALUE = 42
# 状态堆叠大小
STATE_STACK_SIZE = 4

# ==============================================================================
# --- 7. 对手池大小配置 ---
# ==============================================================================
# 长期对手池大小
LONG_TERM_POOL_SIZE = 5
# 短期对手池大小
SHORT_TERM_POOL_SIZE = 10
