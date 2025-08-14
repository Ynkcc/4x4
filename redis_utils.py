import redis
import pickle
import numpy as np

from constants import (REDIS_HOST, REDIS_PORT, REDIS_DB, 
                       REDIS_FREE_QUEUE_KEY, REDIS_FULL_QUEUE_KEY, REDIS_BUFFER_KEY)

def get_redis_connection():
    """获取 Redis 连接实例"""
    return redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=False # 保持字节格式，以便正确处理pickle数据
    )

def serialize_data(data):
    """使用 pickle 序列化数据"""
    return pickle.dumps(data)

def deserialize_data(data):
    """使用 pickle 反序列化数据"""
    return pickle.loads(data)

def push_to_free_queue(redis_conn, index):
    """将缓冲区索引推入空闲队列"""
    redis_conn.rpush(REDIS_FREE_QUEUE_KEY, str(index))

def pop_from_free_queue(redis_conn, timeout=0):
    """从空闲队列中阻塞式弹出缓冲区索引"""
    item = redis_conn.blpop(REDIS_FREE_QUEUE_KEY, timeout=timeout)
    return int(item[1]) if item else None

def push_to_full_queue(redis_conn, index):
    """将缓冲区索引推入已满队列"""
    redis_conn.rpush(REDIS_FULL_QUEUE_KEY, str(index))

def pop_from_full_queue(redis_conn, timeout=0):
    """从已满队列中阻塞式弹出缓冲区索引"""
    item = redis_conn.blpop(REDIS_FULL_QUEUE_KEY, timeout=timeout)
    return int(item[1]) if item else None

def set_buffer_data(redis_conn, index, data):
    """将序列化数据存入 Redis 哈希表"""
    redis_conn.hset(REDIS_BUFFER_KEY, str(index), serialize_data(data))

def get_buffer_data(redis_conn, index):
    """从 Redis 哈希表获取并反序列化数据"""
    data = redis_conn.hget(REDIS_BUFFER_KEY, str(index))
    if data:
        return deserialize_data(data)
    return None

def clear_redis_queues_and_buffers(redis_conn):
    """清空 Redis 中与训练相关的键"""
    redis_conn.delete(REDIS_FREE_QUEUE_KEY, REDIS_FULL_QUEUE_KEY, REDIS_BUFFER_KEY)