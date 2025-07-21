#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门调试玩家索引问题
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

def debug_player_mapping():
    """调试玩家映射问题"""
    print("=== 调试玩家映射问题 ===")
    
    # 创建两个环境
    cython_env = CythonGame()
    bitboard_env = BitboardGame()
    
    # 使用相同种子重置
    seed = 42
    cython_env.reset(seed=seed)
    bitboard_env.reset(seed=seed)
    
    print("初始状态:")
    print(f"Cython 当前玩家: {cython_env.current_player}")
    print(f"Bitboard 当前玩家: {bitboard_env.current_player}")
    
    # 检查位置6的棋子
    print(f"\n位置6的棋子:")
    print(f"Cython: {cython_env.board[6]}")
    print(f"Bitboard: {bitboard_env.board[6]}")
    
    # 执行翻棋动作6
    cython_env.step(6)
    bitboard_env.step(6)
    
    print(f"\n翻棋后状态:")
    print(f"Cython 当前玩家: {cython_env.current_player}")
    print(f"Bitboard 当前玩家: {bitboard_env.current_player}")
    
    print(f"\n位置6的棋子（翻开后）:")
    print(f"Cython: {cython_env.board[6]} (玩家: {cython_env.board[6].player}, 类型: {cython_env.board[6].piece_type.value})")
    print(f"Bitboard: {bitboard_env.board[6]} (玩家: {bitboard_env.board[6].player}, 类型: {bitboard_env.board[6].piece_type.value})")
    
    # 检查状态构建时的玩家定义
    print(f"\n状态构建时的玩家定义:")
    
    # Cython版本的玩家定义
    cython_my_player = cython_env.current_player
    cython_opponent_player = -cython_env.current_player
    cython_my_player_idx = 1 if cython_my_player == 1 else 0
    cython_opponent_player_idx = 1 if cython_opponent_player == 1 else 0
    
    print(f"Cython: 当前玩家={cython_my_player}, 对手={cython_opponent_player}")
    print(f"Cython: 当前玩家索引={cython_my_player_idx}, 对手索引={cython_opponent_player_idx}")
    
    # Bitboard版本的玩家定义
    bitboard_my_player = bitboard_env.current_player
    bitboard_opponent_player = -bitboard_env.current_player
    
    print(f"Bitboard: 当前玩家={bitboard_my_player}, 对手={bitboard_opponent_player}")
    
    # 检查位置6的棋子在两个版本的piece_bitboards中的表示
    piece_at_6 = cython_env.board[6]  # 应该与bitboard_env.board[6]相同
    piece_player = piece_at_6.player
    piece_type_val = piece_at_6.piece_type.value
    
    print(f"\n位置6棋子 (玩家={piece_player}, 类型={piece_type_val}) 在bitboards中的表示:")
    
    # Cython版本
    cython_player_idx = 1 if piece_player == 1 else 0
    cython_bb = cython_env.get_piece_bitboard(piece_player, piece_type_val)
    print(f"Cython piece_bitboards[{cython_player_idx}][{piece_type_val}]: {cython_bb}")
    print(f"位置6是否设置: {(cython_bb >> 6) & 1}")
    
    # Bitboard版本
    bitboard_bb = bitboard_env.piece_bitboards[piece_player][piece_type_val]
    print(f"Bitboard piece_bitboards[{piece_player}][{piece_type_val}]: {bitboard_bb}")
    print(f"位置6是否设置: {(bitboard_bb >> 6) & 1}")
    
    # 分析状态向量中的表示
    print(f"\n在状态向量中的表示:")
    
    # 检查索引118对应的棋子类型0位置6
    print(f"索引118 (对方兵类型位置6):")
    print(f"  应该查看对手玩家的兵bitboard")
    print(f"  Cython对手: 玩家{cython_opponent_player} (索引{cython_opponent_player_idx})")
    print(f"  Bitboard对手: 玩家{bitboard_opponent_player}")
    print(f"  Cython对手兵bitboard: {cython_env.get_piece_bitboard(cython_opponent_player, 0)}")
    print(f"  Bitboard对手兵bitboard: {bitboard_env.piece_bitboards[bitboard_opponent_player][0]}")
    
    # 检查索引166对应的棋子类型3位置6
    print(f"索引166 (对方车类型位置6):")
    print(f"  Cython对手车bitboard: {cython_env.get_piece_bitboard(cython_opponent_player, 3)}")
    print(f"  Bitboard对手车bitboard: {bitboard_env.piece_bitboards[bitboard_opponent_player][3]}")

if __name__ == "__main__":
    debug_player_mapping()
