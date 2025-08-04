# utils/scheduler.py

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    创建一个线性学习率衰减的调度器函数。
    Stable-Baselines3 会在每次更新时调用这个函数，并传入一个从1.0逐渐衰减到0.0的进度值。

    :param initial_value: 初始学习率 (例如 3e-4)
    :return: 一个可调用的调度器函数
    """
    def func(progress_remaining: float) -> float:
        """
        根据剩余的训练进度计算当前的学习率。
        :param progress_remaining: 训练进度，从 1.0 线性下降到 0.0
        """
        return progress_remaining * initial_value
        
    return func
