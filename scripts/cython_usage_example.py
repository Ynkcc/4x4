#!/usr/bin/env python3
"""
Cython 优化版本使用示例
演示如何使用高性能的 Cython 版本进行游戏仿真
"""

import time
import numpy as np
from Game_cython_simple import GameEnvironment

def basic_usage_example():
    """基本使用示例"""
    print("=== Cython 优化版本基本使用示例 ===")
    
    # 创建环境
    env = GameEnvironment()
    print(f"✓ 环境创建成功")
    
    # 重置环境
    state, info = env.reset(seed=42)
    print(f"✓ 环境重置成功，状态维度: {state.shape}")
    
    # 检查动作掩码
    action_mask = info['action_mask']
    valid_actions = np.where(action_mask)[0]
    print(f"✓ 可用动作数量: {len(valid_actions)}")
    
    # 执行几步游戏
    steps = 0
    while steps < 10:
        action = np.random.choice(valid_actions)
        state, reward, terminated, truncated, info = env.step(action)
        
        steps += 1
        print(f"步骤 {steps}: 动作={action}, 奖励={reward:.3f}, 结束={terminated or truncated}")
        
        if terminated or truncated:
            print(f"游戏结束，获胜者: {info.get('winner', 'None')}")
            break
            
        action_mask = info['action_mask']
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            print("无可用动作，游戏结束")
            break

def performance_demo():
    """性能演示"""
    print("\n=== 性能演示 ===")
    
    num_games = 100
    print(f"运行 {num_games} 局游戏...")
    
    start_time = time.time()
    total_steps = 0
    
    for i in range(num_games):
        env = GameEnvironment()
        state, info = env.reset(seed=i)
        
        steps = 0
        while steps < 200:  # 限制最大步数
            action_mask = info['action_mask']
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) == 0:
                break
                
            action = np.random.choice(valid_actions)
            state, reward, terminated, truncated, info = env.step(action)
            
            steps += 1
            total_steps += 1
            
            if terminated or truncated:
                break
    
    elapsed = time.time() - start_time
    
    print(f"✓ 完成 {num_games} 局游戏")
    print(f"✓ 总用时: {elapsed:.3f} 秒")
    print(f"✓ 平均每局: {elapsed/num_games*1000:.3f} 毫秒")
    print(f"✓ 总步数: {total_steps}")
    print(f"✓ 执行速度: {total_steps/elapsed:.1f} 步/秒")
    print(f"✓ 游戏速度: {num_games/elapsed:.1f} 局/秒")

def action_mask_performance():
    """动作掩码性能测试"""
    print("\n=== 动作掩码性能测试 ===")
    
    env = GameEnvironment()
    env.reset(seed=42)
    
    num_calls = 1000
    print(f"调用 action_masks() {num_calls} 次...")
    
    start_time = time.time()
    for _ in range(num_calls):
        action_mask = env.action_masks()
    elapsed = time.time() - start_time
    
    print(f"✓ 总用时: {elapsed:.3f} 秒")
    print(f"✓ 平均每次: {elapsed/num_calls*1000:.3f} 毫秒")
    print(f"✓ 调用频率: {num_calls/elapsed:.1f} 次/秒")

def state_extraction_demo():
    """状态提取演示"""
    print("\n=== 状态提取演示 ===")
    
    env = GameEnvironment()
    state, info = env.reset(seed=42)
    
    print(f"状态向量维度: {state.shape}")
    print(f"状态向量类型: {state.dtype}")
    print(f"非零元素数量: {np.count_nonzero(state)}")
    print(f"状态向量范围: [{state.min():.3f}, {state.max():.3f}]")
    
    # 执行几步并观察状态变化
    print("\n执行几步并观察状态变化:")
    for step in range(3):
        action_mask = info['action_mask']
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            break
            
        action = valid_actions[0]  # 选择第一个可用动作
        state, reward, terminated, truncated, info = env.step(action)
        
        print(f"步骤 {step+1}: 非零元素数量 = {np.count_nonzero(state)}")
        
        if terminated or truncated:
            break

def comparison_with_original():
    """与原版的快速对比"""
    print("\n=== 与原版的快速对比 ===")
    
    try:
        from Game import GameEnvironment as OriginalEnv
        
        # 测试参数
        num_games = 50
        
        # 测试 Cython 版本
        print("测试 Cython 版本...")
        start_time = time.time()
        for i in range(num_games):
            env = GameEnvironment()
            state, info = env.reset(seed=i)
            for _ in range(20):  # 执行20步
                action_mask = info['action_mask']
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    break
                action = np.random.choice(valid_actions)
                state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        cython_time = time.time() - start_time
        
        # 测试原版
        print("测试原版...")
        start_time = time.time()
        for i in range(num_games):
            env = OriginalEnv()
            state, info = env.reset(seed=i)
            for _ in range(20):  # 执行20步
                action_mask = info['action_mask']
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    break
                action = np.random.choice(valid_actions)
                state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        original_time = time.time() - start_time
        
        # 对比结果
        speedup = original_time / cython_time
        print(f"\n对比结果:")
        print(f"✓ Cython 版本: {cython_time:.3f} 秒")
        print(f"✓ 原版: {original_time:.3f} 秒")
        print(f"✓ 加速比: {speedup:.2f}x")
        
    except ImportError:
        print("原版环境不可用，跳过对比测试")

def main():
    """主函数"""
    print("Cython 优化版本暗棋环境使用示例")
    print("=" * 50)
    
    # 运行各种示例
    basic_usage_example()
    performance_demo()
    action_mask_performance()
    state_extraction_demo()
    comparison_with_original()
    
    print("\n" + "=" * 50)
    print("所有示例执行完毕！")
    print("Cython 优化版本运行正常，性能优异！")

if __name__ == "__main__":
    main()
