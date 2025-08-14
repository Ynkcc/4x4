# src_code/model_utils.py
import torch
import io
from redis_utils import get_redis_connection
from constants import REDIS_MODEL_KEY, REDIS_MODEL_VERSION_KEY

def save_model_to_redis(model, version):
    """将模型参数和版本号保存到 Redis"""
    redis_conn = get_redis_connection()
    # 使用 BytesIO 将 state_dict 序列化为字节流
    buffer = io.BytesIO()
    torch.save(model.network.state_dict(), buffer)
    buffer.seek(0)
    
    # 存入模型参数和版本号
    redis_conn.set(REDIS_MODEL_KEY, buffer.getvalue())
    redis_conn.set(REDIS_MODEL_VERSION_KEY, str(version))
    print(f"模型已保存到 Redis，版本号: {version}")

def get_latest_model_from_redis(model):
    """从 Redis 获取最新模型参数和版本号"""
    redis_conn = get_redis_connection()
    model_data = redis_conn.get(REDIS_MODEL_KEY)
    version = redis_conn.get(REDIS_MODEL_VERSION_KEY)
    
    if model_data and version:
        buffer = io.BytesIO(model_data)
        buffer.seek(0)
        model.network.load_state_dict(torch.load(buffer))
        print(f"已从 Redis 加载模型，版本号: {version.decode()}")
        return int(version.decode())
    return None