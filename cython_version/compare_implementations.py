#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较 Game_cython.pyx 和 bitboard_version/Game_bitboard.py 的实现
确保在相同种子下生成相同的 state 和 action_mask
"""

import sys
import os
import numpy as np
import random

# 添加路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bitboard_version'))

def compare_arrays(arr1, arr2, name, tolerance=1e-6):
    """比较两个数组，如果不同则抛出错误"""
    if arr1.shape != arr2.shape:
        raise ValueError(f"{name} 形状不同: {arr1.shape} vs {arr2.shape}")
    
    if not np.allclose(arr1, arr2, atol=tolerance):
        diff_indices = np.where(~np.isclose(arr1, arr2, atol=tolerance))
        print(f"\n❌ {name} 不一致!")
        print(f"形状: {arr1.shape}")
        print(f"不同元素数量: {len(diff_indices[0])}")
        print(f"前10个不同的索引和值:")
        for i in range(min(10, len(diff_indices[0]))):
            idx = diff_indices[0][i]
            print(f"  索引 {idx}: Cython={arr1[idx]:.6f}, Bitboard={arr2[idx]:.6f}")
        
        print(f"\nCython 版本 {name}:")
        print(arr1)
        print(f"\nBitboard 版本 {name}:")
        print(arr2)
        
        raise ValueError(f"{name} 不一致")
    else:
        print(f"✓ {name} 一致")

def test_initialization():
    """测试初始化是否一致"""
    print("=== 测试初始化 ===")
    
    # 导入两个版本
    try:
        from Game_cython import GameEnvironment as CythonGame
        print("✓ 成功导入 Cython 版本")
    except ImportError as e:
        print(f"✗ 导入 Cython 版本失败: {e}")
        return False
    
    try:
        from Game_bitboard import GameEnvironment as BitboardGame
        print("✓ 成功导入 Bitboard 版本")
    except ImportError as e:
        print(f"✗ 导入 Bitboard 版本失败: {e}")
        return False
    
    # 创建环境
    cython_env = CythonGame()
    bitboard_env = BitboardGame()
    
    # 使用相同种子重置
    seed = 42
    cython_state, cython_info = cython_env.reset(seed=seed)
    bitboard_state, bitboard_info = bitboard_env.reset(seed=seed)
    
    # 比较初始状态
    compare_arrays(cython_state, bitboard_state, "初始状态")
    compare_arrays(cython_info['action_mask'], bitboard_info['action_mask'], "初始动作掩码")
    
    return cython_env, bitboard_env

def test_step_by_step(cython_env, bitboard_env, num_steps=20):
    """逐步测试游戏进程"""
    print(f"\n=== 逐步测试 {num_steps} 步 ===")
    
    # 设置相同的随机种子用于选择动作
    random.seed(42)
    np.random.seed(42)
    
    for step in range(num_steps):
        print(f"\n--- 步骤 {step + 1} ---")
        
        # 获取当前状态和动作掩码
        cython_state = cython_env.get_state()
        bitboard_state = bitboard_env.get_state()
        cython_mask = cython_env.action_masks()
        bitboard_mask = bitboard_env.action_masks()
        
        # 比较状态和动作掩码
        compare_arrays(cython_state, bitboard_state, f"步骤{step+1}状态")
        compare_arrays(cython_mask, bitboard_mask, f"步骤{step+1}动作掩码")
        
        # 选择相同的有效动作
        valid_actions = np.where(cython_mask)[0]
        if len(valid_actions) == 0:
            print(f"步骤 {step + 1}: 没有有效动作，游戏结束")
            break
        
        # 使用相同的随机选择
        action = np.random.choice(valid_actions)
        print(f"选择动作: {action}")
        
        # 执行动作
        cython_result = cython_env.step(action)
        bitboard_result = bitboard_env.step(action)
        
        cython_state, cython_reward, cython_term, cython_trunc, cython_info = cython_result
        bitboard_state, bitboard_reward, bitboard_term, bitboard_trunc, bitboard_info = bitboard_result
        
        # 比较结果
        compare_arrays(cython_state, bitboard_state, f"步骤{step+1}执行后状态")
        
        if abs(cython_reward - bitboard_reward) > 1e-6:
            raise ValueError(f"步骤{step+1}奖励不一致: Cython={cython_reward}, Bitboard={bitboard_reward}")
        
        if cython_term != bitboard_term:
            raise ValueError(f"步骤{step+1}终止状态不一致: Cython={cython_term}, Bitboard={bitboard_term}")
        
        if cython_trunc != bitboard_trunc:
            raise ValueError(f"步骤{step+1}截断状态不一致: Cython={cython_trunc}, Bitboard={bitboard_trunc}")
        
        compare_arrays(cython_info['action_mask'], bitboard_info['action_mask'], f"步骤{step+1}执行后动作掩码")
        
        print(f"✓ 步骤 {step + 1} 一致 (奖励: {cython_reward:.6f})")
        
        # 如果游戏结束，停止测试
        if cython_term or cython_trunc:
            print(f"游戏在步骤 {step + 1} 结束")
            break
    
    return True

def test_multiple_games(num_games=5):
    """测试多局游戏"""
    print(f"\n=== 测试 {num_games} 局游戏 ===")
    
    from Game_cython import GameEnvironment as CythonGame
    from Game_bitboard import GameEnvironment as BitboardGame
    
    for game_num in range(num_games):
        print(f"\n--- 游戏 {game_num + 1} ---")
        
        # 创建新环境
        cython_env = CythonGame()
        bitboard_env = BitboardGame()
        
        # 使用不同种子
        seed = 42 + game_num * 10
        cython_state, cython_info = cython_env.reset(seed=seed)
        bitboard_state, bitboard_info = bitboard_env.reset(seed=seed)
        
        # 比较初始状态
        compare_arrays(cython_state, bitboard_state, f"游戏{game_num+1}初始状态")
        compare_arrays(cython_info['action_mask'], bitboard_info['action_mask'], f"游戏{game_num+1}初始动作掩码")
        
        # 设置相同的随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 进行几步测试
        for step in range(5):  # 每局测试5步
            cython_mask = cython_env.action_masks()
            bitboard_mask = bitboard_env.action_masks()
            
            compare_arrays(cython_mask, bitboard_mask, f"游戏{game_num+1}步骤{step+1}动作掩码")
            
            valid_actions = np.where(cython_mask)[0]
            if len(valid_actions) == 0:
                break
            
            action = np.random.choice(valid_actions)
            
            cython_result = cython_env.step(action)
            bitboard_result = bitboard_env.step(action)
            
            # 只比较状态，不比较其他返回值以节省时间
            compare_arrays(cython_result[0], bitboard_result[0], f"游戏{game_num+1}步骤{step+1}状态")
            
            if cython_result[2] or cython_result[3]:  # terminated or truncated
                break
        
        print(f"✓ 游戏 {game_num + 1} 一致")

def test_specific_scenarios():
    """测试特定场景"""
    print("\n=== 测试特定场景 ===")
    
    from Game_cython import GameEnvironment as CythonGame
    from Game_bitboard import GameEnvironment as BitboardGame
    
    # 测试场景1: 翻棋动作
    print("\n--- 场景1: 仅翻棋动作 ---")
    cython_env = CythonGame()
    bitboard_env = BitboardGame()
    
    seed = 100
    cython_env.reset(seed=seed)
    bitboard_env.reset(seed=seed)
    
    # 执行几次翻棋动作
    random.seed(seed)
    np.random.seed(seed)
    
    for i in range(3):
        cython_mask = cython_env.action_masks()
        bitboard_mask = bitboard_env.action_masks()
        
        # 找到翻棋动作（前16个动作）
        reveal_actions = []
        for j in range(16):
            if cython_mask[j]:
                reveal_actions.append(j)
        
        if reveal_actions:
            action = np.random.choice(reveal_actions)
            cython_env.step(action)
            bitboard_env.step(action)
            
            # 比较执行后的状态
            compare_arrays(cython_env.get_state(), bitboard_env.get_state(), f"翻棋{i+1}后状态")
    
    print("✓ 翻棋场景测试通过")

def main():
    """主测试函数"""
    print("开始比较 Cython 和 Bitboard 实现...")
    print("="*60)
    
    try:
        # 测试初始化
        cython_env, bitboard_env = test_initialization()
        
        # 逐步测试
        test_step_by_step(cython_env, bitboard_env, num_steps=30)
        
        # 测试多局游戏
        test_multiple_games(num_games=3)
        
        # 测试特定场景
        test_specific_scenarios()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！两个实现完全一致。")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
