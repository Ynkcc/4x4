# opponent_model_manager.py (修改后)
import os
import numpy as np
from typing import Optional, List, Dict, Any
from sb3_contrib import MaskablePPO

class SharedOpponentModelManager:
    """
    共享的对手模型管理器
    解决多环境中重复加载同一模型的问题
    【优化】增加了从内存直接更新模型的功能
    """
    _instance = None
    _model = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str) -> Optional[MaskablePPO]:
        """
        加载或重用对手模型
        只有当模型路径变化时才重新加载
        """
        if model_path != self._model_path or self._model is None:
            if os.path.exists(model_path):
                print(f"📦 共享模型管理器：从磁盘加载对手模型 {model_path}")
                try:
                    # 我们为模型指定一个设备，以确保一致性
                    self._model = MaskablePPO.load(model_path, device='auto')
                    self._model_path = model_path
                    print(f"✅ 成功加载对手模型，将被多个环境共享使用")
                except Exception as e:
                    print(f"❌ 警告：无法加载对手模型 {model_path}: {e}")
                    self._model = None
                    self._model_path = None
            else:
                print(f"⚠️  警告：对手模型文件不存在: {model_path}")
                self._model = None
                self._model_path = None
        
        return self._model
        
    # 【新增的优化方法】
    def update_model_from_learner(self, learner_model: MaskablePPO):
        """
        直接从内存中用learner的权重更新当前持有的opponent模型。
        这是一个非常高效的操作，避免了磁盘I/O。
        """
        if self._model is None:
            print("⚠️ 警告: 对手模型尚未初始化，无法从内存更新。")
            return

        print("🧠 正在从内存直接更新对手模型权重...")
        # 从learner的策略网络中提取最新的权重
        learner_weights = learner_model.policy.state_dict()
        # 将权重加载到opponent的策略网络中
        self._model.policy.load_state_dict(learner_weights)
        print("✅ 对手模型权重已在内存中更新完毕！")


    def predict_single(self, observation: Dict, action_mask: np.ndarray, deterministic: bool = True) -> Optional[int]:
        """
        单个预测
        """
        if self._model is None:
            return None
        
        try:
            action, _ = self._model.predict(
                observation, 
                action_masks=action_mask, 
                deterministic=deterministic
            )
            return int(action)
        except Exception as e:
            print(f"⚠️  警告：对手模型预测失败: {e}")
            return None
    
    def predict_batch(self, observations: List[Dict], action_masks: List[np.ndarray], 
                     deterministic: bool = True) -> List[Optional[int]]:
        """
        批量预测（如果需要优化多个环境同时请求的情况）
        """
        if self._model is None:
            return [None] * len(observations)
        
        try:
            actions = []
            for obs, mask in zip(observations, action_masks):
                action, _ = self._model.predict(obs, action_masks=mask, deterministic=deterministic)
                actions.append(int(action))
            return actions
        except Exception as e:
            print(f"⚠️  警告：对手模型批量预测失败: {e}")
            return [None] * len(observations)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        return {
            "model_loaded": self._model is not None,
            "model_path": self._model_path,
            "model_type": type(self._model).__name__ if self._model else None
        }

# 全局单例实例
shared_opponent_manager = SharedOpponentModelManager()