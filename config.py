# config.py
"""
配置文件 - 统一管理所有项目配置
"""
import torch

# ============================================================================
# 游戏相关配置
# ============================================================================
WINNING_SCORE = 60
MAX_CONSECUTIVE_MOVES_FOR_DRAW = 12
MAX_STEPS_PER_EPISODE = 100
ACTION_SPACE_SIZE = 112  # 16个翻棋动作 + 48个移动动作 + 48个炮攻击动作
HISTORY_WINDOW_SIZE = 15

# 动作空间细分
REVEAL_ACTIONS_COUNT = 16  # 翻棋动作数量
REGULAR_MOVE_ACTIONS_COUNT = 48  # 移动动作数量
CANNON_ATTACK_ACTIONS_COUNT = 48  # 炮攻击动作数量

# ============================================================================
# 神经网络相关配置
# ============================================================================
NETWORK_NUM_HIDDEN_CHANNELS = 64
NETWORK_NUM_RES_BLOCKS = 5
LSTM_HIDDEN_SIZE = 128  # 保留LSTM相关配置以备不时之需
GRU_HIDDEN_SIZE = 128
EXP_EPSILON = 0.01

# ============================================================================
# 设备配置
# ============================================================================
def get_device():
    """自动选择最佳可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# ============================================================================
# 训练相关配置
# ============================================================================
class TrainingConfig:
    # 设备配置
    DEVICE = get_device()
    
    # 模型保存路径
    SAVEDIR = 'saved_models'
    MODEL_PATH = 'saved_models/model.pt'
    TRAIN_DATA_DIR = 'train_data'  # 训练数据缓存路径
    
    # 优化器参数
    LEARNING_RATE = 0.0001
    MOMENTUM = 0
    EPSILON = 1e-5
    ALPHA = 0.99
    MAX_GRAD_NORM = 40.0
    
    # DMC 训练参数
    # 【优化】显著增大 BATCH_SIZE，这是提高GPU利用率最关键的参数之一
    BATCH_SIZE = 4096  
    # 【优化】增加每次更新的 EPOCHS，让GPU对同一批数据进行更多次训练
    EPOCHS = 8  
    # 【优化】极大增加经验池大小，确保有足够多样性的数据可供采样
    BUFFER_SIZE = 200000 
    GAME_BATCH_NUM = 1500  # 训练更新的总次数
    TRAIN_UPDATE_INTERVAL = 10  # 每次更新的间隔时间(秒)
    
    # 经验池采样参数
    # 【优化】相应地增加开始训练的最小经验池大小
    MIN_BUFFER_SIZE = 20000  
    # 保持不变或适当增加，确保训练之间有足够的新数据
    TRAINING_FREQUENCY = 1000  
    # 【优化】显著增加每次训练的采样批次数，这是增加单次训练负载最有效的方法
    SAMPLES_PER_UPDATE = 32 

# ============================================================================
# 数据收集相关配置
# ============================================================================
class CollectConfig:
    # 并发与批处理
    # 【优化】增加并发收集进程数，如果你的CPU核心数足够多（例如8核或以上），可以设为8或更高
    NUM_THREADS = 4 
    MAIN_PROCESS_BATCH_SIZE = 8192 
    QUEUE_MAX_SIZE = 3 * MAIN_PROCESS_BATCH_SIZE 
    # 【优化】增大每个进程的本地缓冲区，减少向主队列put的频率，降低进程间通信开销
    LOCAL_BUFFER_SIZE = 4096  

    # 探索参数
    EPSILON = 0.1  # Epsilon for epsilon-greedy exploration
    
    # 数据缓冲
    BUFFER_SIZE = 10000  # 经验池大小
    
    # 文件路径
    PYTORCH_MODEL_PATH = 'saved_models/model.pt'  # 模型路径
    TRAIN_DATA_DIR = 'train_data'  # 训练数据缓存路径

# ============================================================================
# 人机对战相关配置
# ============================================================================
class HumanVsAIConfig:
    # 模型路径
    MODEL_PATH = 'saved_models/model.pt'
    
    # AI配置
    AI_EPSILON = 0.0  # AI的探索率，通常设为0使其采用最优策略

# ============================================================================
# 向后兼容 - 提供全局变量访问方式
# ============================================================================
# 将配置类的属性暴露为模块级变量，以保持向后兼容性
TRAINING_CONFIG = TrainingConfig()
COLLECT_CONFIG = CollectConfig()
HUMAN_VS_AI_CONFIG = HumanVsAIConfig()

# 设备相关
DEVICE = get_device()
TRAINING_DEVICE = DEVICE

# 训练相关
LEARNING_RATE = TrainingConfig.LEARNING_RATE
MOMENTUM = TrainingConfig.MOMENTUM
EPSILON = TrainingConfig.EPSILON
ALPHA = TrainingConfig.ALPHA
MAX_GRAD_NORM = TrainingConfig.MAX_GRAD_NORM
BATCH_SIZE = TrainingConfig.BATCH_SIZE
EPOCHS = TrainingConfig.EPOCHS
BUFFER_SIZE = TrainingConfig.BUFFER_SIZE
GAME_BATCH_NUM = TrainingConfig.GAME_BATCH_NUM
TRAIN_UPDATE_INTERVAL = TrainingConfig.TRAIN_UPDATE_INTERVAL
SAVEDIR = TrainingConfig.SAVEDIR

# 数据收集相关
COLLECT_EPSILON = CollectConfig.EPSILON
PYTORCH_MODEL_PATH = CollectConfig.PYTORCH_MODEL_PATH
TRAIN_DATA_DIR = CollectConfig.TRAIN_DATA_DIR
NUM_COLLECT_THREADS = CollectConfig.NUM_THREADS
MAIN_PROCESS_BATCH_SIZE = CollectConfig.MAIN_PROCESS_BATCH_SIZE