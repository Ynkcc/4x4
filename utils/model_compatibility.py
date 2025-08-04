# utils/model_compatibility.py

import sys
import types

def setup_legacy_imports():
    """
    设置向后兼容的模块导入，支持从旧版本的模型文件加载。
    这个函数确保旧的custom_policy模块能够正确映射到新的game.policy模块。
    """
    if 'custom_policy' not in sys.modules:
        # 导入新的策略类
        from game.policy import CustomActorCriticPolicy
        
        # 创建虚拟的custom_policy模块
        custom_policy_module = types.ModuleType('custom_policy')
        custom_policy_module.CustomActorCriticPolicy = CustomActorCriticPolicy
        
        # 注册到sys.modules中，这样pickle.loads可以找到它
        sys.modules['custom_policy'] = custom_policy_module
        
        return True
    return False
