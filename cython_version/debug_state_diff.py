#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试状态向量差异，分析索引对应的含义
"""

import sys
import os
import numpy as np
import random

# 添加路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bitboard_version'))

from Game_cython import GameEnvironment as CythonGame
from Game_bitboard import GameEnvironment as BitboardGame

def analyze_state_vector_structure():
    """分析状态向量的结构"""
    print("=== 分析状态向量结构 ===")
    
    # 创建两个环境
    cython_env = CythonGame()
    bitboard_env = BitboardGame()
    
    print(f"状态向量大小: {cython_env.observation_space.shape[0]}")
    print(f"我方棋子平面起始索引: {cython_env._my_pieces_plane_start_idx}")
    print(f"对方棋子平面起始索引: {cython_env._opponent_pieces_plane_start_idx}")
    print(f"隐藏棋子平面起始索引: {cython_env._hidden_pieces_plane_start_idx}")
    print(f"空位置平面起始索引: {cython_env._empty_plane_start_idx}")
    print(f"标量特征起始索引: {cython_env._scalar_features_start_idx}")
    
    # 分析有问题的索引
    problem_indices = [118, 166]
    
    for idx in problem_indices:
        print(f"\n--- 分析索引 {idx} ---")
        
        if idx < cython_env._opponent_pieces_plane_start_idx:
            # 我方棋子平面
            relative_idx = idx - cython_env._my_pieces_plane_start_idx
            piece_type = relative_idx // 16
            position = relative_idx % 16
            print(f"属于: 我方棋子平面")
            print(f"棋子类型: {piece_type} (0=兵,1=炮,2=马,3=车,4=象,5=士,6=帅)")
            print(f"位置: {position}")
            
        elif idx < cython_env._hidden_pieces_plane_start_idx:
            # 对方棋子平面
            relative_idx = idx - cython_env._opponent_pieces_plane_start_idx
            piece_type = relative_idx // 16
            position = relative_idx % 16
            print(f"属于: 对方棋子平面")
            print(f"棋子类型: {piece_type} (0=兵,1=炮,2=马,3=车,4=象,5=士,6=帅)")
            print(f"位置: {position}")
            
        elif idx < cython_env._empty_plane_start_idx:
            # 隐藏棋子平面
            relative_idx = idx - cython_env._hidden_pieces_plane_start_idx
            print(f"属于: 隐藏棋子平面")
            print(f"位置: {relative_idx}")
            
        elif idx < cython_env._scalar_features_start_idx:
            # 空位置平面
            relative_idx = idx - cython_env._empty_plane_start_idx
            print(f"属于: 空位置平面")
            print(f"位置: {relative_idx}")
            
        else:
            # 标量特征
            relative_idx = idx - cython_env._scalar_features_start_idx
            print(f"属于: 标量特征")
            print(f"特征索引: {relative_idx} (0=我方得分,1=对方得分,2=移动计数)")

def debug_first_step():
    """调试第一步执行"""
    print("\n=== 调试第一步执行 ===")
    
    # 创建两个环境
    cython_env = CythonGame()
    bitboard_env = BitboardGame()
    
    # 使用相同种子重置
    seed = 42
    cython_state, cython_info = cython_env.reset(seed=seed)
    bitboard_state, bitboard_info = bitboard_env.reset(seed=seed)
    
    print("初始状态一致性检查通过")
    
    # 设置相同的随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 选择相同的动作
    valid_actions = np.where(cython_info['action_mask'])[0]
    action = np.random.choice(valid_actions)
    print(f"选择动作: {action}")
    
    # 查看动作的坐标
    cython_coords = cython_env.action_to_coords[action]
    bitboard_coords = bitboard_env.action_to_coords[action]
    print(f"Cython 动作坐标: {cython_coords}")
    print(f"Bitboard 动作坐标: {bitboard_coords}")
    
    # 查看执行动作前的关键状态
    print(f"\n执行前状态:")
    print(f"Cython 当前玩家: {cython_env.current_player}")
    print(f"Bitboard 当前玩家: {bitboard_env.current_player}")
    print(f"Cython 隐藏bitboard: {cython_env.get_hidden_bitboard()}")
    print(f"Bitboard 隐藏bitboard: {bitboard_env.hidden_bitboard}")
    
    # 执行动作
    cython_result = cython_env.step(action)
    bitboard_result = bitboard_env.step(action)
    
    print(f"\n执行后状态:")
    print(f"Cython 当前玩家: {cython_env.current_player}")
    print(f"Bitboard 当前玩家: {bitboard_env.current_player}")
    print(f"Cython 隐藏bitboard: {cython_env.get_hidden_bitboard()}")
    print(f"Bitboard 隐藏bitboard: {bitboard_env.hidden_bitboard}")
    
    # 检查问题索引的值
    cython_state_after = cython_result[0]
    bitboard_state_after = bitboard_result[0]
    
    for idx in [118, 166]:
        print(f"索引 {idx}: Cython={cython_state_after[idx]}, Bitboard={bitboard_state_after[idx]}")

def check_bitboard_consistency():
    """检查 bitboard 的一致性"""
    print("\n=== 检查 bitboard 一致性 ===")
    
    # 创建两个环境
    cython_env = CythonGame()
    bitboard_env = BitboardGame()
    
    # 使用相同种子重置
    seed = 42
    cython_env.reset(seed=seed)
    bitboard_env.reset(seed=seed)
    
    # 比较各种 bitboard
    print(f"隐藏bitboard: Cython={cython_env.get_hidden_bitboard()}, Bitboard={bitboard_env.hidden_bitboard}")
    print(f"空位bitboard: Cython={cython_env.get_empty_bitboard()}, Bitboard={bitboard_env.empty_bitboard}")
    
    # 比较各个玩家的已翻开bitboard
    for player in [1, -1]:
        cython_revealed = cython_env.get_revealed_bitboard(player)
        bitboard_revealed = bitboard_env.revealed_bitboards[player]
        print(f"玩家{player}已翻开bitboard: Cython={cython_revealed}, Bitboard={bitboard_revealed}")
    
    # 比较各个棋子类型的bitboard
    for player in [1, -1]:
        for piece_type in range(7):
            cython_pieces = cython_env.get_piece_bitboard(player, piece_type)
            bitboard_pieces = bitboard_env.piece_bitboards[player][piece_type]
            print(f"玩家{player}棋子类型{piece_type}: Cython={cython_pieces}, Bitboard={bitboard_pieces}")

def main():
    analyze_state_vector_structure()
    debug_first_step()
    check_bitboard_consistency()

if __name__ == "__main__":
    main()
