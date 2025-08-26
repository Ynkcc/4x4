# utils/constants.py

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
INITIAL_LR = 1e-6
# PPO 裁剪范围 (Clip Range)
PPO_CLIP_RANGE = 0.2
# 每次更新前，每个环境收集的步数
PPO_N_STEPS = 512
# 训练时每个 mini-batch 的大小
PPO_BATCH_SIZE = 512
# 每次更新时，对采集到的数据进行优化的轮数
PPO_N_EPOCHS = 20
# GAE (Generalized Advantage Estimation) 的 lambda 参数
PPO_GAE_LAMBDA = 0.999
# 【修改】价值函数 (Value Function) 在总损失中的系数 - 适当降低以平衡策略和价值学习，防止价值损失主导更新
PPO_VF_COEF = 0.6 # 增加价值函数的权重，鼓励更强的价值学习
# 熵 (Entropy) 在总损失中的系数，鼓励探索
PPO_ENT_COEF = 0.08
# 【修改】梯度裁剪的最大范数 - 放宽限制，允许价值函数进行更大幅度的更新以适应变化的策略
PPO_MAX_GRAD_NORM = 0.5
# PPO模型训练时使用的设备 ('auto', 'cpu', 'cuda')
PPO_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 训练过程的日志详细程度 (0=无, 1=信息, 2=调试)
PPO_VERBOSE = 2
# 是否在训练时显示进度条
PPO_SHOW_PROGRESS = True


# ==============================================================================
# --- 4. 神经网络架构超参数 (主要调优目标) ---
# ==============================================================================
# 特征提取器最终输出的特征维度
NETWORK_FEATURES_DIM = 256
# CNN中的残差块数量
NETWORK_NUM_RES_BLOCKS = 5
# CNN中的隐藏层通道数
NETWORK_NUM_HIDDEN_CHANNELS = 128


# ==============================================================================
# --- 5. Elo系统与对手池配置 (通常固定不变) ---
# ==============================================================================
# 对手池最大容量
MAX_OPPONENT_POOL_SIZE = 20
# 短期池：存储最近生成的模型
SHORT_TERM_POOL_SIZE = MAX_OPPONENT_POOL_SIZE // 2
# 长期池：按特定策略从旧模型中采样
LONG_TERM_POOL_SIZE = MAX_OPPONENT_POOL_SIZE - SHORT_TERM_POOL_SIZE
# Elo 评分的默认初始值
ELO_DEFAULT = 1200
# Elo K因子，控制评分更新的幅度
ELO_K_FACTOR = 32
# Elo 权重温度参数，用于在选择对手时平衡Elo差异
ELO_WEIGHT_TEMPERATURE = 100

# --- 新增：对手池采样配置 ---
# 从长期+短期池中随机选择的参训模型数量
TRAINING_POOL_SAMPLE_SIZE = 6
# 主宰者在最终对手池中的最小权重比例 (确保不少于51%)
MAIN_OPPONENT_MIN_WEIGHT_RATIO = 0.51
# 长期池模型相对于短期池模型的权重倍数 (用于加权随机采样)
LONG_TERM_POOL_WEIGHT_MULTIPLIER = 2.0


# ==============================================================================
# --- 6. 训练循环配置 (通常固定不变) ---
# ==============================================================================
# 总共进行多少次 "训练 -> 评估" 的大循环
TOTAL_TRAINING_LOOPS = 1000
# 用于训练的并行环境数量
N_ENVS = 8
# 在每次大循环中，学习者模型训练的总步数
STEPS_PER_LOOP = PPO_N_STEPS * N_ENVS * 8


# ==============================================================================
# --- 7. 评估配置 (通常固定不变) ---
# ==============================================================================
# 每次评估时进行的游戏局数 (必须是偶数，以进行镜像对局)
EVALUATION_GAMES = 50
# 挑战者胜率需要超过此阈值才能取代主宰者
EVALUATION_THRESHOLD = 0.51
# 评估时使用的并行环境数量 (通常为1)
EVALUATION_N_ENVS = 1


# ==============================================================================
# --- 8. 调试与实验配置 (新增) ---
# ==============================================================================
# 是否在训练和评估初期使用固定的随机种子，以确保棋盘布局不变。
# 这对于调试和验证模型在特定情况下的行为非常有用。
# 在正式、长期的训练中，应将其设置为 False 以确保棋局多样性。
USE_FIXED_SEED_FOR_TRAINING = False

# 如果 USE_FIXED_SEED_FOR_TRAINING 为 True，将使用此种子值。
FIXED_SEED_VALUE = 42