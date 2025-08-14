# src_code/constants.py

import os

# --- 游戏和环境常量 (部分来自 environment.py, 统一管理) ---
WINNING_SCORE = 60
MAX_CONSECUTIVE_MOVES_FOR_DRAW = 12
MAX_STEPS_PER_EPISODE = 100
ACTION_SPACE_SIZE = 112 # 16(翻棋) + 48(普通移动) + 48(炮攻击)

# --- 网络结构超参数 ---
NETWORK_NUM_HIDDEN_CHANNELS = 64
NETWORK_NUM_RES_BLOCKS = 5
LSTM_HIDDEN_SIZE = 128
# 网络最终输出的融合特征维度，这里是动态计算的，但我们在这里定义一个常量来确保一致性
# CNN展平后大小 (4x4xNETWORK_NUM_HIDDEN_CHANNELS) + FC输出大小 (64) + LSTM输出大小 (LSTM_HIDDEN_SIZE)
NETWORK_FEATURES_DIM = 4 * 4 * NETWORK_NUM_HIDDEN_CHANNELS + 64 + LSTM_HIDDEN_SIZE

# --- 历史动作编码超参数 ---
# 历史动作序列的窗口大小
HISTORY_WINDOW_SIZE = 15

# --- 训练超参数 ---
XPID = 'dark_chess_self_play'
SAVEDIR = 'training_checkpoints'
SAVE_INTERVAL_MIN = 30 # 每30分钟保存一次模型
TOTAL_FRAMES = 1000000000 # 训练的总步数
EXP_EPSILON = 0.01 # 探索概率
BATCH_SIZE = 32
UNROLL_LENGTH = 100 # Rollout的展开长度
NUM_BUFFERS = 50 # 共享内存缓冲区的数量
NUM_THREADS = 4 # 学习者线程数
MAX_GRAD_NORM = 40.0
LEARNING_RATE = 0.0001
ALPHA = 0.99 # RMSProp平滑常数
MOMENTUM = 0
EPSILON = 1e-5 # RMSProp epsilon

# --- 硬件配置 ---
ACTOR_DEVICE_CPU = False # 是否使用CPU作为Actor设备
GPU_DEVICES = '0' # GPU设备ID
NUM_ACTOR_DEVICES = 1 # 用于模拟的设备数量
NUM_ACTORS = 5 # 每个设备的Actor进程数
TRAINING_DEVICE = '0' # 用于训练的GPU设备ID, 'cpu'表示使用CPU

# --- 蒙特卡洛 Rollout 超参数 ---
# 每次Rollout时，对所有暗棋进行重新随机化的次数
NUM_IMPERFECT_INFO_ROLLOUTS = 5