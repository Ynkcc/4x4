#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度调试随机种子问题
"""

import sys
import os
import numpy as np
import random

# 添加路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bitboard_version'))

from Game_cython import GameEnvironment as CythonGame, PieceType, Piece
from Game_bitboard import GameEnvironment as BitboardGame

def test_random_seed_behavior():
    """测试随机种子行为"""
    print("=== 测试随机种子行为 ===")
    
    # 测试1: 基本随机数生成
    print("\n测试1: 基本随机数生成")
    
    seed = 42
    
    # 使用相同种子生成随机数
    random.seed(seed)
    np.random.seed(seed)
    print(f"random.random(): {random.random()}")
    print(f"np.random.random(): {np.random.random()}")
    
    # 重新设置种子
    random.seed(seed)
    np.random.seed(seed)
    print(f"重新设置后 random.random(): {random.random()}")
    print(f"重新设置后 np.random.random(): {np.random.random()}")

def test_piece_shuffling():
    """测试棋子洗牌"""
    print("\n=== 测试棋子洗牌 ===")
    
    # 创建相同的棋子列表
    def create_pieces():
        pieces = []
        PIECE_MAX_COUNTS = [2, 1, 1, 1, 1, 1, 1]
        for pt in PieceType:
            count = PIECE_MAX_COUNTS[pt.value]
            for p in [1, -1]:
                for _ in range(count):
                    pieces.append(Piece(pt, p))
        return pieces
    
    seed = 42
    
    # 测试Cython版本的洗牌方式
    print("\nCython版本洗牌:")
    pieces1 = create_pieces()
    print("洗牌前:", [f"{p.piece_type.name}_{p.player}" for p in pieces1[:8]])
    
    # 模拟Cython的随机种子设置
    np_random = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    np_random.shuffle(pieces1)
    print("洗牌后:", [f"{p.piece_type.name}_{p.player}" for p in pieces1[:8]])
    
    # 测试Bitboard版本的洗牌方式
    print("\nBitboard版本洗牌:")
    pieces2 = create_pieces()
    print("洗牌前:", [f"{p.piece_type.name}_{p.player}" for p in pieces2[:8]])
    
    # 模拟Bitboard的随机种子设置（使用gym的方式）
    # 创建一个临时的BitboardGame来获取正确的np_random
    temp_game = BitboardGame()
    temp_game.reset(seed=seed)
    if hasattr(temp_game, 'np_random') and temp_game.np_random is not None:
        temp_game.np_random.shuffle(pieces2)
    else:
        random.shuffle(pieces2)
    
    print("洗牌后:", [f"{p.piece_type.name}_{p.player}" for p in pieces2[:8]])
    
    # 比较结果
    print("\n比较洗牌结果:")
    for i in range(len(pieces1)):
        p1, p2 = pieces1[i], pieces2[i]
        if p1.piece_type != p2.piece_type or p1.player != p2.player:
            print(f"位置 {i}: Cython={p1.piece_type.name}_{p1.player}, Bitboard={p2.piece_type.name}_{p2.player} <- 不同!")
        else:
            print(f"位置 {i}: 一致")

def test_gym_random_behavior():
    """测试Gym随机行为"""
    print("\n=== 测试Gym随机行为 ===")
    
    # 测试Bitboard版本的实际随机种子设置
    bitboard_game = BitboardGame()
    print(f"重置前的np_random: {getattr(bitboard_game, 'np_random', 'None')}")
    
    bitboard_game.reset(seed=42)
    print(f"重置后的np_random: {getattr(bitboard_game, 'np_random', 'None')}")
    
    if hasattr(bitboard_game, 'np_random') and bitboard_game.np_random is not None:
        print(f"np_random类型: {type(bitboard_game.np_random)}")
        print(f"np_random生成的随机数: {bitboard_game.np_random.random()}")

if __name__ == "__main__":
    test_random_seed_behavior()
    test_piece_shuffling()
    test_gym_random_behavior()
