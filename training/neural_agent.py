# training/neural_agent.py

# ==============================================================================
# 警告：关于在多进程环境中使用本类的说明
#
# 这个 NeuralAgent 类使用了单例模式（Singleton）和线程锁（threading.Lock），
# 这使得它在单进程、多线程的环境下是安全的。
#
# 然而，这个模式在与 stable-baselines3 的 SubprocVecEnv 一起使用时是【无效且错误的】。
# SubprocVecEnv 会为每个环境创建一个独立的子进程。在创建子进程时，Python 会
# 序列化（pickle）父进程的对象并发送给子进程，而不是共享内存。
#
# 这会导致每个子进程都拥有一个独立的、互不相关的 NeuralAgent 实例副本。
# 在主进程中对这个单例进行的任何修改（例如加载新模型），都【不会】反映到
# 子进程中去。
#
# 正确的做法是，像修复后的 `training/trainer.py` 和 `game/environment.py`
# 那样，让环境自身负责加载模型，并通过 `VecEnv.env_method()` 从主进程发送
# 更新指令。
#
# 因此，请【不要】在基于 SubprocVecEnv 的多进程训练循环中直接共享此类实例。
# ==============================================================================

import os
import threading
from typing import Optional, Dict, Tuple
from sb3_contrib import MaskablePPO
from utils.model_compatibility import setup_legacy_imports

class NeuralAgent:
    """
    一个基于神经网络的Agent，简化的线程安全单例模式。
    避免使用可重入锁，减少复杂性和死锁风险。
    """
    _instance = None
    _lock = threading.Lock()  # 使用简单的互斥锁

    def __new__(cls, model_path: Optional[str] = None):
        # 快速检查，避免不必要的锁获取
        if cls._instance is not None and model_path is None:
            return cls._instance
            
        with cls._lock:
            if cls._instance is None:
                print("🧠 创建 NeuralAgent 单例...")
                cls._instance = super(NeuralAgent, cls).__new__(cls)
                cls._instance._model = None
                cls._instance._model_path = None
                cls._instance._initialized = False
                
            # 如果请求了新模型路径，在锁外进行加载
            instance = cls._instance
            
        # 在锁外处理模型加载，避免死锁
        if model_path is not None:
            if not instance._initialized:
                instance._initialize_model(model_path)
            elif model_path != instance._model_path:
                print(f"🧠 NeuralAgent 请求加载新模型: {model_path}")
                instance._load_model_safe(model_path)
                
        return instance
    
    def _initialize_model(self, model_path: str):
        """初始化模型（首次）"""
        if not self._initialized:
            print(f"📦 首次初始化 NeuralAgent 模型: {model_path}")
            self._load_model_internal(model_path)
            self._initialized = True

    def _load_model_safe(self, model_path: str):
        """线程安全的模型加载"""
        with self._lock:
            self._load_model_internal(model_path)

    def _load_model_internal(self, model_path: str) -> None:
        """
        内部模型加载方法，不使用锁（调用者负责线程安全）
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

    def load_model(self, model_path: str) -> None:
        """
        公共模型加载接口，线程安全
        """
        self._load_model_safe(model_path)

    def update_model_from_learner(self, learner_model: MaskablePPO) -> None:
        """
        直接从内存中用learner的权重更新当前持有的模型。
        这是一个非常高效的操作，避免了磁盘I/O。
        """
        with self._lock:
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
        # 快速检查，避免不必要的锁
        if self._model is None:
            raise ValueError("❌ 模型未加载，无法进行预测。")
            
        # 创建模型的本地引用，减少锁持有时间
        with self._lock:
            model = self._model
            
        if model is None:
            raise ValueError("❌ 模型未加载，无法进行预测。")
        
        try:
            action, _ = model.predict(
                observation, 
                action_masks=action_masks, 
                deterministic=deterministic
            )
            return int(action), None
        except Exception as e:
            raise RuntimeError(f"⚠️ 模型预测失败: {e}")
                
    def get_model_path(self) -> Optional[str]:
        """获取当前模型的路径。"""
        return self._model_path  # 读取操作，通常不需要锁

    @classmethod
    def reset_instance(cls):
        """重置单例实例，主要用于测试或特殊情况。"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._model = None
                cls._instance._model_path = None
                cls._instance._initialized = False
            cls._instance = None
            print("🔄 NeuralAgent 单例已重置")
