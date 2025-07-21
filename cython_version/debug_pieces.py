#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试棋子创建顺序
"""

import sys
import os

# 添加路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bitboard_version'))

from Game_cython import GameEnvironment as CythonGame, PieceType as CythonPieceType, Piece as CythonPiece
from Game_bitboard import GameEnvironment as BitboardGame, PieceType as BitboardPieceType, Piece as BitboardPiece

def compare_piece_creation():
    """比较棋子创建方式"""
    print("=== 比较棋子创建方式 ===")
    
    # Cython版本的棋子创建
    print("Cython版本的棋子创建:")
    cython_pieces = []
    PIECE_MAX_COUNTS = [2, 1, 1, 1, 1, 1, 1]  # 与Cython中的定义一致
    for pt in CythonPieceType:
        count = PIECE_MAX_COUNTS[pt.value]
        for p in [1, -1]:
            for _ in range(count):
                piece = CythonPiece(pt, p)
                cython_pieces.append(piece)
                print(f"  {piece}")
    
    print(f"\nCython总共创建了 {len(cython_pieces)} 个棋子")
    
    # Bitboard版本的棋子创建
    print("\nBitboard版本的棋子创建:")
    bitboard_game = BitboardGame()
    PIECE_MAX_COUNTS_DICT = bitboard_game.PIECE_MAX_COUNTS
    print(f"PIECE_MAX_COUNTS_DICT: {PIECE_MAX_COUNTS_DICT}")
    
    bitboard_pieces = []
    for pt, count in PIECE_MAX_COUNTS_DICT.items():
        for p in [1, -1]:
            for _ in range(count):
                piece = BitboardPiece(pt, p)
                bitboard_pieces.append(piece)
                print(f"  {piece}")
    
    print(f"\nBitboard总共创建了 {len(bitboard_pieces)} 个棋子")
    
    # 比较棋子顺序
    print(f"\n比较棋子顺序:")
    for i, (c_piece, b_piece) in enumerate(zip(cython_pieces, bitboard_pieces)):
        if c_piece.piece_type.value != b_piece.piece_type.value or c_piece.player != b_piece.player:
            print(f"位置 {i}: Cython={c_piece}, Bitboard={b_piece} <- 不同!")
        else:
            print(f"位置 {i}: {c_piece} = {b_piece}")

def debug_enum_order():
    """调试枚举顺序"""
    print("\n=== 调试枚举顺序 ===")
    
    print("Cython PieceType:")
    for pt in CythonPieceType:
        print(f"  {pt.name} = {pt.value}")
    
    print("\nBitboard PieceType:")
    for pt in BitboardPieceType:
        print(f"  {pt.name} = {pt.value}")

if __name__ == "__main__":
    debug_enum_order()
    compare_piece_creation()
