# utils/constants.py

import os

# --- 路径配置 ---
# 根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

# 初始模型路径
# 支持从多个位置加载初始模型
CURRICULUM_MODEL_PATH = os.path.join(ROOT_DIR, "cnn_curriculum_models", "final_model_cnn.zip")
SELF_PLAY_MODEL_PATH = os.path.join(ROOT_DIR, "self_play_models", "final_selfplay_model_optimized.zip")

# 本次训练的输出目录
SELF_PLAY_OUTPUT_DIR = os.path.join(ROOT_DIR, "models", "self_play_final")

# 对手池专用目录
OPPONENT_POOL_DIR = os.path.join(SELF_PLAY_OUTPUT_DIR, "opponent_pool")

# Elo评估中使用的模型路径
MAIN_OPPONENT_PATH = os.path.join(SELF_PLAY_OUTPUT_DIR, "main_opponent.zip")
CHALLENGER_PATH = os.path.join(SELF_PLAY_OUTPUT_DIR, "challenger.zip")
FINAL_MODEL_PATH = os.path.join(SELF_PLAY_OUTPUT_DIR, "final_model.zip")

# TensorBoard 日志路径
TENSORBOARD_LOG_PATH = os.path.join(ROOT_DIR, "tensorboard_logs", "self_play_final")


# --- 训练超参数 ---
# 总共进行多少次 "训练 -> 评估" 的循环
TOTAL_TRAINING_LOOPS = 1000
# 在每次评估之间，学习者训练多少步
STEPS_PER_LOOP = 16384 * 2
# 并行环境数量
N_ENVS = 8


# --- PPO 模型超参数 (基于 stable-baselines3 默认值) ---
# 学习率 (默认: 3e-4)
INITIAL_LR = 3e-4
# PPO 裁剪范围 (默认: 0.2)
PPO_CLIP_RANGE = 0.2
# 每次更新收集的步数 (默认: 2048)
PPO_N_STEPS = 2048
# 批次大小 (默认: 64)
PPO_BATCH_SIZE = 64
# 训练轮数 (默认: 10)
PPO_N_EPOCHS = 10
# GAE lambda参数 (默认: 0.95)
PPO_GAE_LAMBDA = 0.95
# 价值函数损失系数 (默认: 0.5)
PPO_VF_COEF = 0.5
# 熵损失系数 (默认: 0.01)
PPO_ENT_COEF = 0.01
# 最大梯度范数 (默认: 0.5)
PPO_MAX_GRAD_NORM = 0.5
# 设备选择 ('auto', 'cpu', 'cuda')
PPO_DEVICE = 'auto'
# 日志详细程度 (0=无输出, 1=信息, 2=调试)
PPO_VERBOSE = 1
# 是否显示训练进度条
PPO_SHOW_PROGRESS = True


# --- 神经网络架构超参数 ---
# 特征提取器输出维度 (默认: 256)
NETWORK_FEATURES_DIM = 256
# 残差块数量 (默认: 4)
NETWORK_NUM_RES_BLOCKS = 4
# 隐藏层通道数 (默认: 64)
NETWORK_NUM_HIDDEN_CHANNELS = 64


# --- 对手池与Elo系统超参数 ---
# 对手池最大容量
MAX_OPPONENT_POOL_SIZE = 20
# Elo 默认初始值
ELO_DEFAULT = 1200
# Elo K因子 (用于更新评分)
ELO_K_FACTOR = 32
# Elo 权重计算的温度参数，用于决定与主宰者Elo差距多大的对手被选中的概率
ELO_WEIGHT_TEMPERATURE = 100


# --- 评估超参数 ---
# 评估时进行多少局游戏 (必须是偶数，以进行镜像对局)
EVALUATION_GAMES = 100
# 挑战者胜率需要超过多少才能取代主宰者
EVALUATION_THRESHOLD = 0.53 
# 评估时使用的并行环境数量 (评估时通常用1个)
EVALUATION_N_ENVS = 1