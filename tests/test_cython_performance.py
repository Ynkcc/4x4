#!/usr/bin/env python3
"""
性能测试脚本 - 比较原版和 Cython 优化版的性能
"""

import time
import numpy as np
import random
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 导入两个版本
from banqi_gym.Game import GameEnvironment as OriginalGameEnvironment
from banqi_gym.Game_cython import GameEnvironment as CythonGameEnvironment

def run_game_benchmark(env_class, num_games=100, max_steps_per_game=200):
    """运行游戏性能测试"""
    print(f"\n测试 {env_class.__name__}...")
    
    total_steps = 0
    total_time = 0
    game_lengths = []
    
    start_time = time.time()
    
    for game_idx in range(num_games):
        env = env_class()
        state, info = env.reset(seed=42 + game_idx)
        
        steps = 0
        done = False
        
        game_start = time.time()
        
        while not done and steps < max_steps_per_game:
            action_mask = info['action_mask']
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) == 0:
                break
                
            action = np.random.choice(valid_actions)
            state, reward, terminated, truncated, info = env.step(action)
            
            steps += 1
            done = terminated or truncated
        
        game_time = time.time() - game_start
        total_time += game_time
        total_steps += steps
        game_lengths.append(steps)
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    print(f"总用时: {total_elapsed:.3f}秒")
    print(f"平均每局游戏: {total_elapsed/num_games:.4f}秒")
    print(f"总步数: {total_steps}")
    print(f"平均每步用时: {total_time/total_steps*1000:.3f}毫秒")
    print(f"平均游戏长度: {np.mean(game_lengths):.1f}步")
    print(f"每秒执行步数: {total_steps/total_elapsed:.1f}")
    
    return {
        'total_time': total_elapsed,
        'total_steps': total_steps,
        'avg_game_time': total_elapsed/num_games,
        'avg_step_time': total_time/total_steps,
        'steps_per_second': total_steps/total_elapsed,
        'avg_game_length': np.mean(game_lengths)
    }

def run_action_mask_benchmark(env_class, num_calls=1000):
    """测试 action_masks 方法的性能"""
    print(f"\n测试 {env_class.__name__} action_masks 方法...")
    
    env = env_class()
    env.reset(seed=42)
    
    start_time = time.time()
    
    for _ in range(num_calls):
        action_mask = env.action_masks()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"调用 {num_calls} 次 action_masks: {elapsed:.3f}秒")
    print(f"平均每次调用: {elapsed/num_calls*1000:.3f}毫秒")
    print(f"每秒调用次数: {num_calls/elapsed:.1f}")
    
    return {
        'total_time': elapsed,
        'avg_call_time': elapsed/num_calls,
        'calls_per_second': num_calls/elapsed
    }

def main():
    print("暗棋环境性能测试 - 原版 vs Cython 优化版")
    print("=" * 50)
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 游戏性能测试
    print("\n1. 游戏执行性能测试 (100局游戏)")
    original_results = run_game_benchmark(OriginalGameEnvironment, num_games=100)
    cython_results = run_game_benchmark(CythonGameEnvironment, num_games=100)
    
    # action_masks 性能测试
    print("\n2. action_masks 方法性能测试 (1000次调用)")
    original_mask_results = run_action_mask_benchmark(OriginalGameEnvironment, num_calls=1000)
    cython_mask_results = run_action_mask_benchmark(CythonGameEnvironment, num_calls=1000)
    
    # 性能对比总结
    print("\n" + "=" * 50)
    print("性能对比总结:")
    print("=" * 50)
    
    speedup_game = original_results['total_time'] / cython_results['total_time']
    speedup_step = original_results['avg_step_time'] / cython_results['avg_step_time']
    speedup_mask = original_mask_results['avg_call_time'] / cython_mask_results['avg_call_time']
    
    print(f"游戏执行总体加速: {speedup_game:.2f}x")
    print(f"单步执行加速: {speedup_step:.2f}x")
    print(f"action_masks 加速: {speedup_mask:.2f}x")
    
    print(f"\n原版执行速度: {original_results['steps_per_second']:.1f} 步/秒")
    print(f"Cython版执行速度: {cython_results['steps_per_second']:.1f} 步/秒")
    
    print(f"\n原版 action_masks: {original_mask_results['calls_per_second']:.1f} 调用/秒")
    print(f"Cython版 action_masks: {cython_mask_results['calls_per_second']:.1f} 调用/秒")

if __name__ == "__main__":
    main()
