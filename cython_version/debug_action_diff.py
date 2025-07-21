#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析动作掩码差异
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

def analyze_action_difference():
    """分析动作掩码差异"""
    print("=== 分析动作掩码差异 ===")
    
    # 创建两个环境并重现前7步
    cython_env = CythonGame()
    bitboard_env = BitboardGame()
    
    # 使用相同种子重置
    seed = 42
    cython_env.reset(seed=seed)
    bitboard_env.reset(seed=seed)
    
    # 设置相同的随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 重现前6步
    actions = []
    for step in range(6):
        valid_actions = np.where(cython_env.action_masks())[0]
        action = np.random.choice(valid_actions)
        actions.append(action)
        cython_env.step(action)
        bitboard_env.step(action)
        print(f"步骤 {step+1}: 动作 {action}")
    
    # 第7步：翻棋动作7
    action = 7
    actions.append(action)
    print(f"步骤 7: 动作 {action}")
    
    cython_env.step(action)
    bitboard_env.step(action)
    
    # 获取动作掩码
    cython_mask = cython_env.action_masks()
    bitboard_mask = bitboard_env.action_masks()
    
    # 分析差异的动作索引
    print(f"\n分析差异动作:")
    
    for action_idx in [37, 39]:
        print(f"\n动作索引 {action_idx}:")
        
        if action_idx < 16:
            print(f"  类型: 翻棋动作")
            pos = cython_env.action_to_coords[action_idx]
            print(f"  位置: {pos}")
        elif action_idx < 64:
            print(f"  类型: 普通移动动作")
            from_pos, to_pos = cython_env.action_to_coords[action_idx]
            print(f"  从 {from_pos} 到 {to_pos}")
        else:
            print(f"  类型: 炮攻击动作")
            from_pos, to_pos = cython_env.action_to_coords[action_idx]
            print(f"  从 {from_pos} 攻击 {to_pos}")
        
        print(f"  Cython 有效: {cython_mask[action_idx]}")
        print(f"  Bitboard 有效: {bitboard_mask[action_idx]}")
    
    # 打印当前棋盘状态
    print(f"\n当前棋盘状态:")
    print(f"当前玩家: {cython_env.current_player}")
    
    # 检查特定位置的棋子
    for sq in range(16):
        cython_piece = cython_env.board[sq]
        bitboard_piece = bitboard_env.board[sq]
        if cython_piece != bitboard_piece or (cython_piece and cython_piece.revealed):
            r, c = sq // 4, sq % 4
            print(f"  位置 ({r},{c}): Cython={cython_piece}, Bitboard={bitboard_piece}")

def debug_action_masks_details():
    """详细调试动作掩码生成"""
    print("\n=== 详细调试动作掩码生成 ===")
    
    # 重现场景
    cython_env = CythonGame()
    bitboard_env = BitboardGame()
    
    # 重置并执行前7步
    seed = 42
    cython_env.reset(seed=seed)
    bitboard_env.reset(seed=seed)
    
    random.seed(42)
    np.random.seed(42)
    
    # 执行前7步
    for step in range(7):
        if step < 6:
            valid_actions = np.where(cython_env.action_masks())[0]
            action = np.random.choice(valid_actions)
        else:
            action = 7  # 第7步固定为动作7
        
        cython_env.step(action)
        bitboard_env.step(action)
    
    # 打印相关的bitboard状态
    print(f"\nbitboard状态:")
    print(f"Cython 隐藏bitboard: {cython_env.get_hidden_bitboard()}")
    print(f"Bitboard 隐藏bitboard: {bitboard_env.hidden_bitboard}")
    print(f"Cython 空位bitboard: {cython_env.get_empty_bitboard()}")
    print(f"Bitboard 空位bitboard: {bitboard_env.empty_bitboard}")
    
    # 检查当前玩家的棋子bitboard
    current_player = cython_env.current_player
    print(f"\n当前玩家 {current_player} 的棋子:")
    for piece_type in range(7):
        cython_bb = cython_env.get_piece_bitboard(current_player, piece_type)
        bitboard_bb = bitboard_env.piece_bitboards[current_player][piece_type]
        if cython_bb != bitboard_bb:
            print(f"  类型{piece_type}: Cython={cython_bb}, Bitboard={bitboard_bb} <- 不同!")
        elif cython_bb > 0:
            print(f"  类型{piece_type}: {cython_bb}")

if __name__ == "__main__":
    analyze_action_difference()
    debug_action_masks_details()
