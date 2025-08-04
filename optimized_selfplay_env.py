# optimized_selfplay_env.py - 优化的自我对弈环境管理器
import os
import numpy as np
from typing import Optional, Dict, Any
from sb3_contrib import MaskablePPO
from Game import GameEnvironment
from gymnasium import spaces  # 添加此行以导入spaces

class SharedOpponentModel:
    """共享的对手模型管理器，避免重复加载"""
    _instance = None
    _model = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str) -> Optional[MaskablePPO]:
        """加载或重用对手模型"""
        if model_path != self._model_path or self._model is None:
            if os.path.exists(model_path):
                print(f"共享对手模型管理器：加载模型 {model_path}")
                try:
                    self._model = MaskablePPO.load(model_path)
                    self._model_path = model_path
                    print(f"成功加载对手模型，将被 {8} 个环境共享")
                except Exception as e:
                    print(f"警告：无法加载对手模型 {model_path}: {e}")
                    self._model = None
                    self._model_path = None
            else:
                print(f"警告：对手模型文件不存在: {model_path}")
                self._model = None
                self._model_path = None
        
        return self._model
    
    def predict_batch(self, observations: list, action_masks: list, deterministic: bool = True):
        """批量预测，提高效率"""
        if self._model is None:
            return [None] * len(observations)
        
        try:
            # 如果只有一个观察，直接预测
            if len(observations) == 1:
                action, _ = self._model.predict(
                    observations[0], 
                    action_masks=action_masks[0], 
                    deterministic=deterministic
                )
                return [int(action)]
            
            # 批量预测（如果模型支持）
            actions = []
            for obs, mask in zip(observations, action_masks):
                action, _ = self._model.predict(obs, action_masks=mask, deterministic=deterministic)
                actions.append(int(action))
            return actions
            
        except Exception as e:
            print(f"警告：对手模型批量预测失败: {e}")
            return [None] * len(observations)

class OptimizedGameEnvironment(GameEnvironment):
    """优化的游戏环境，使用共享对手模型"""
    
    def __init__(self, render_mode=None, curriculum_stage=4, opponent_policy=None):
        # 先调用父类构造函数，但跳过对手模型加载
        super(GameEnvironment, self).__init__()
        self.render_mode = render_mode
        self.curriculum_stage = curriculum_stage
        
        # 自我对弈相关属性
        self.learning_player_id = 1  # 学习者始终是玩家1（红方）
        self.opponent_model = None
        self.shared_model_manager = SharedOpponentModel()
        
        # 使用共享模型管理器
        if opponent_policy:
            self.opponent_model = self.shared_model_manager.load_model(opponent_policy)
            self.opponent_policy_path = opponent_policy
        
        # 继续原有的初始化代码
        self._initialize_spaces_and_data()
    
    def _initialize_spaces_and_data(self):
        """初始化状态空间和数据结构"""
        # --- 状态空间定义 ---
        num_channels = 7 * 2 + 2  # 我方7种棋子 + 敌方7种棋子 + 暗棋 + 空位 = 16个通道
        board_shape = (num_channels, 4, 4)  # BOARD_ROWS, BOARD_COLS
        scalar_shape = (3 + 8 + 8,)  # 基础标量 + 我方存活 + 敌方存活
        
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0.0, high=1.0, shape=board_shape, dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=scalar_shape, dtype=np.float32)
        })
        
        self.action_space = spaces.Discrete(112)  # ACTION_SPACE_SIZE
        
        # --- 核心数据结构 ---
        self.board = np.empty(16, dtype=object)  # TOTAL_POSITIONS
        self.piece_vectors = {p: [np.zeros(16, dtype=bool) for _ in range(7)] for p in [1, -1]}
        self.revealed_vectors = {p: np.zeros(16, dtype=bool) for p in [1, -1]}
        self.hidden_vector = np.zeros(16, dtype=bool)
        self.empty_vector = np.zeros(16, dtype=bool)
        
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}
        
        self.attack_tables = {}
        self.action_to_coords = {}
        self.coords_to_action = {}
        self._initialize_lookup_tables()
    
    def predict_opponent_action(self, observation, action_mask):
        """预测对手动作，使用共享模型"""
        if self.opponent_model is None:
            return None
        
        # 使用共享模型管理器进行预测
        actions = self.shared_model_manager.predict_batch([observation], [action_mask])
        return actions[0] if actions and actions[0] is not None else None
