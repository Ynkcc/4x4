# training/neural_agent.py
import os
from typing import Optional, Dict, Tuple
from sb3_contrib import MaskablePPO
from utils.model_compatibility import setup_legacy_imports

class NeuralAgent:
    """
    一个基于神经网络的Agent，使用单例模式来确保模型只被加载一次并被共享。
    这取代了原有的 SharedOpponentModelManager。
    """
    _instance = None
    _model: Optional[MaskablePPO] = None
    _model_path: Optional[str] = None

    def __new__(cls, model_path: Optional[str] = None):
        if cls._instance is None:
            print("🧠 创建 NeuralAgent 单例...")
            cls._instance = super(NeuralAgent, cls).__new__(cls)
            if model_path:
                cls._instance.load_model(model_path)
        elif model_path and model_path != cls._model_path:
            # 如果实例已存在但请求了不同的模型路径，则加载新模型
            print(f"🧠 NeuralAgent 单例已存在，但请求了新模型。正在加载 {model_path}...")
            cls._instance.load_model(model_path)
            
        return cls._instance

    def load_model(self, model_path: str) -> None:
        """
        加载或重载神经网络模型。
        只有当模型路径变化时才重新从磁盘加载。
        """
        if model_path == self._model_path and self._model is not None:
            print(f"📦 NeuralAgent: 模型 {os.path.basename(model_path)} 已加载，无需重复操作。")
            return

        if os.path.exists(model_path):
            print(f"📦 NeuralAgent: 从磁盘加载模型 {model_path}...")
            try:
                # 确保旧模型文件的兼容性
                setup_legacy_imports()
                self._model = MaskablePPO.load(model_path, device='auto')
                self._model_path = model_path
                print(f"✅ 成功加载模型，将在多个环境中共享使用。")
            except Exception as e:
                self._model = None
                self._model_path = None
                raise RuntimeError(f"❌ 无法加载模型 {model_path}: {e}")
        else:
            self._model = None
            self._model_path = None
            raise FileNotFoundError(f"⚠️ 模型文件不存在: {model_path}")

    def update_model_from_learner(self, learner_model: MaskablePPO) -> None:
        """
        直接从内存中用learner的权重更新当前持有的模型。
        这是一个非常高效的操作，避免了磁盘I/O。
        """
        if self._model is None:
            raise ValueError("⚠️ 对手模型尚未初始化，无法从内存更新。")

        print("🧠 正在从内存直接更新对手模型权重...")
        learner_weights = learner_model.policy.state_dict()
        self._model.policy.load_state_dict(learner_weights)
        print("✅ 对手模型权重已在内存中更新完毕！")

    def predict(self, observation: Dict, action_masks: Dict, deterministic: bool = True) -> Tuple[int, None]:
        """
        使用加载的模型进行预测。
        """
        if self._model is None:
            raise ValueError("❌ 模型未加载，无法进行预测。")
        
        try:
            action, _ = self._model.predict(
                observation, 
                action_masks=action_masks, 
                deterministic=deterministic
            )
            return int(action), None
        except Exception as e:
            raise RuntimeError(f"⚠️ 模型预测失败: {e}")
            
    def get_model_path(self) -> Optional[str]:
        """获取当前模型的路径。"""
        return self._model_path
