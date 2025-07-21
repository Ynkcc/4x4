#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的调试脚本，专注于分析第一步的差异
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

def analyze_specific_differences():
    """分析具体差异"""
    print("=== 分析具体差异 ===")
    
    # 创建两个环境
    cython_env = CythonGame()
    bitboard_env = BitboardGame()
    
    # 使用相同种子重置
    seed = 42
    cython_env.reset(seed=seed)
    bitboard_env.reset(seed=seed)
    
    print("状态向量结构分析:")
    print(f"我方棋子平面: 0-{cython_env._opponent_pieces_plane_start_idx-1}")
    print(f"对方棋子平面: {cython_env._opponent_pieces_plane_start_idx}-{cython_env._hidden_pieces_plane_start_idx-1}")
    print(f"隐藏棋子平面: {cython_env._hidden_pieces_plane_start_idx}-{cython_env._empty_plane_start_idx-1}")
    print(f"空位置平面: {cython_env._empty_plane_start_idx}-{cython_env._scalar_features_start_idx-1}")
    
    # 分析有问题的索引
    problem_indices = [118, 166]
    
    for idx in problem_indices:
        print(f"\n索引 {idx} 分析:")
        
        if idx < cython_env._opponent_pieces_plane_start_idx:
            # 我方棋子平面
            relative_idx = idx - cython_env._my_pieces_plane_start_idx
            piece_type = relative_idx // 16
            position = relative_idx % 16
            print(f"  属于: 我方棋子平面")
            print(f"  棋子类型: {piece_type}")
            print(f"  位置: {position}")
            
        elif idx < cython_env._hidden_pieces_plane_start_idx:
            # 对方棋子平面
            relative_idx = idx - cython_env._opponent_pieces_plane_start_idx
            piece_type = relative_idx // 16
            position = relative_idx % 16
            print(f"  属于: 对方棋子平面")
            print(f"  棋子类型: {piece_type}")
            print(f"  位置: {position}")
            
        elif idx < cython_env._empty_plane_start_idx:
            # 隐藏棋子平面
            relative_idx = idx - cython_env._hidden_pieces_plane_start_idx
            print(f"  属于: 隐藏棋子平面")
            print(f"  位置: {relative_idx}")
            
        elif idx < cython_env._scalar_features_start_idx:
            # 空位置平面
            relative_idx = idx - cython_env._empty_plane_start_idx
            print(f"  属于: 空位置平面")
            print(f"  位置: {relative_idx}")
    
    # 设置相同的随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 选择相同的动作 (翻棋动作6)
    action = 6
    print(f"\n执行动作: {action}")
    
    # 查看动作执行前后的状态
    print("\n执行前:")
    print(f"Cython 当前玩家: {cython_env.current_player}")
    print(f"Bitboard 当前玩家: {bitboard_env.current_player}")
    
    # 执行动作
    cython_result = cython_env.step(action)
    bitboard_result = bitboard_env.step(action)
    
    cython_state_after = cython_result[0]
    bitboard_state_after = bitboard_result[0]
    
    print("\n执行后:")
    print(f"Cython 当前玩家: {cython_env.current_player}")
    print(f"Bitboard 当前玩家: {bitboard_env.current_player}")
    
    # 检查问题索引的值
    print("\n问题索引值:")
    for idx in [118, 166]:
        print(f"索引 {idx}: Cython={cython_state_after[idx]}, Bitboard={bitboard_state_after[idx]}")
    
    # 检查相关的 bitboard 状态
    print("\n相关 bitboard 状态:")
    
    # 检查索引118 (对方棋子平面的某个位置)
    if 118 < cython_env._hidden_pieces_plane_start_idx:
        relative_idx = 118 - cython_env._opponent_pieces_plane_start_idx
        piece_type = relative_idx // 16
        position = relative_idx % 16
        print(f"索引118 - 对方棋子类型{piece_type}位置{position}:")
        print(f"  Cython piece_bitboard[0][{piece_type}]: {cython_env.get_piece_bitboard(0, piece_type) if hasattr(cython_env, 'get_piece_bitboard') else 'N/A'}")
        print(f"  Bitboard piece_bitboard[-1][{piece_type}]: {bitboard_env.piece_bitboards[-1][piece_type]}")
        
    # 检查索引166 (隐藏棋子平面的某个位置)
    if 166 >= cython_env._hidden_pieces_plane_start_idx and 166 < cython_env._empty_plane_start_idx:
        relative_idx = 166 - cython_env._hidden_pieces_plane_start_idx
        print(f"索引166 - 隐藏棋子位置{relative_idx}:")
        print(f"  Cython hidden_bitboard: {cython_env.get_hidden_bitboard()}")
        print(f"  Bitboard hidden_bitboard: {bitboard_env.hidden_bitboard}")
        print(f"  位置{relative_idx}是否隐藏: Cython={(cython_env.get_hidden_bitboard() >> relative_idx) & 1}, Bitboard={(bitboard_env.hidden_bitboard >> relative_idx) & 1}")

if __name__ == "__main__":
    analyze_specific_differences()
