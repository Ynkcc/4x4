#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析target_bbs计算差异
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

def analyze_target_bbs():
    """分析target_bbs计算差异"""
    print("=== 分析target_bbs计算差异 ===")
    
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
    
    current_player = cython_env.current_player  # -1 (黑方)
    opponent_player = -current_player  # 1 (红方)
    
    print(f"当前玩家: {current_player}, 对手: {opponent_player}")
    
    # 分析Cython版本的target_bbs计算
    print(f"\nCython版本的对手棋子bitboard:")
    cython_opponent_idx = 1 if opponent_player == 1 else 0
    cumulative_targets = cython_env.get_empty_bitboard()
    
    for pt_val in range(7-1, -1, -1):  # 6到0
        bb = cython_env.get_piece_bitboard(opponent_player, pt_val)
        cumulative_targets |= bb
        print(f"  棋子类型{pt_val}: {bb}, 累积目标: {cumulative_targets}")
    
    # 分析特殊规则
    print(f"\n特殊规则处理:")
    soldier_targets = cumulative_targets | cython_env.get_piece_bitboard(opponent_player, 6)  # SOLDIER gets GENERAL
    general_targets = cumulative_targets & ~cython_env.get_piece_bitboard(opponent_player, 0)  # GENERAL excludes SOLDIER
    
    print(f"兵的目标 (包含将): {soldier_targets}")
    print(f"将的目标 (排除兵): {general_targets}")
    
    # 马的目标
    horse_targets = cumulative_targets  # 假设马使用pt_val=2的累积目标
    print(f"马的目标: {horse_targets}")
    
    # 分析Bitboard版本的target_bbs计算
    print(f"\nBitboard版本的对手棋子bitboard:")
    from Game_bitboard import PieceType
    
    bitboard_cumulative = bitboard_env.empty_bitboard
    for pt in PieceType:
        bb = bitboard_env.piece_bitboards[opponent_player][pt.value]
        bitboard_cumulative |= bb
        print(f"  棋子类型{pt.value}({pt.name}): {bb}, 累积目标: {bitboard_cumulative}")
    
    # 检查特定位置
    print(f"\n检查特定位置:")
    print(f"位置 (0,3) [位3]: 目标棋子为象(ELEPHANT=4)")
    print(f"位置 (1,2) [位6]: 目标棋子为兵(SOLDIER=0)")
    
    # 检查这两个位置在累积目标中的表示
    pos_3_bit = 1 << 3  # 位置(0,3)对应位3
    pos_6_bit = 1 << 6  # 位置(1,2)对应位6
    
    print(f"\n位置(0,3)在各cumulative_targets中:")
    print(f"  Cython累积(马的目标): {(horse_targets >> 3) & 1}")
    print(f"  Bitboard累积: {(bitboard_cumulative >> 3) & 1}")
    
    print(f"\n位置(1,2)在各cumulative_targets中:")
    print(f"  Cython累积(马的目标): {(horse_targets >> 6) & 1}")
    print(f"  Bitboard累积: {(bitboard_cumulative >> 6) & 1}")
    
    # 详细分析马能攻击什么
    print(f"\n马的攻击规则分析:")
    print(f"按暗棋规则，马(HORSE=2)应该能攻击:")
    print(f"  象(ELEPHANT=4): {(cumulative_targets >> 3) & 1} (Cython), {(bitboard_cumulative >> 3) & 1} (Bitboard)")
    print(f"  士(ADVISOR=5): 需要检查士的位置")
    print(f"  将(GENERAL=6): 需要检查将的位置")
    print(f"  不能攻击兵(SOLDIER=0): {(cumulative_targets >> 6) & 1} (Cython), {(bitboard_cumulative >> 6) & 1} (Bitboard)")

if __name__ == "__main__":
    analyze_target_bbs()
