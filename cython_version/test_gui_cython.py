#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Cython 优化版本的 GUI 是否能正常工作
"""

import sys
import os

# 确保能找到模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from Game_cython import GameEnvironment, PieceType
    print("✓ 成功导入 Game_cython 模块")
except ImportError as e:
    print(f"✗ 导入 Game_cython 失败: {e}")
    sys.exit(1)

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    # 创建游戏环境
    try:
        game = GameEnvironment()
        print("✓ 成功创建 GameEnvironment")
    except Exception as e:
        print(f"✗ 创建 GameEnvironment 失败: {e}")
        return False
    
    # 测试重置
    try:
        state, info = game.reset()
        print("✓ 成功重置游戏")
    except Exception as e:
        print(f"✗ 重置游戏失败: {e}")
        return False
    
    # 测试访问属性
    try:
        print(f"✓ 当前玩家: {game.current_player}")
        print(f"✓ 动作空间大小: {game.ACTION_SPACE_SIZE}")
        print(f"✓ 翻棋动作数量: {game.REVEAL_ACTIONS_COUNT}")
    except Exception as e:
        print(f"✗ 访问属性失败: {e}")
        return False
    
    # 测试 bitboard 访问方法
    try:
        hidden_bb = game.get_hidden_bitboard()
        empty_bb = game.get_empty_bitboard()
        revealed_bb = game.get_revealed_bitboard(1)
        piece_bb = game.get_piece_bitboard(1, 0)
        print(f"✓ 隐藏棋子 bitboard: {hidden_bb}")
        print(f"✓ 空位置 bitboard: {empty_bb}")
        print(f"✓ 红方已翻开 bitboard: {revealed_bb}")
        print(f"✓ 红方兵 bitboard: {piece_bb}")
    except Exception as e:
        print(f"✗ 访问 bitboard 失败: {e}")
        return False
    
    return True

def test_game_play():
    """测试游戏操作"""
    print("\n=== 测试游戏操作 ===")
    
    game = GameEnvironment()
    state, info = game.reset()
    
    # 测试动作掩码
    try:
        action_mask = game.action_masks()
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        print(f"✓ 有效动作数量: {len(valid_actions)}")
    except Exception as e:
        print(f"✗ 获取动作掩码失败: {e}")
        return False
    
    # 测试执行动作
    try:
        if valid_actions:
            action = valid_actions[0]
            state, reward, terminated, truncated, info = game.step(action)
            print(f"✓ 成功执行动作 {action}，奖励: {reward}")
            print(f"✓ 游戏状态: terminated={terminated}, truncated={truncated}")
        else:
            print("✗ 没有有效动作可执行")
            return False
    except Exception as e:
        print(f"✗ 执行动作失败: {e}")
        return False
    
    return True

def test_gui_compatibility():
    """测试GUI兼容性"""
    print("\n=== 测试GUI兼容性 ===")
    
    try:
        # 导入GUI所需的模块
        from PySide6.QtWidgets import QApplication
        print("✓ PySide6 可用")
        
        # 测试能否创建应用（不显示窗口）
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # 导入GUI脚本
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
        from game_gui import MainWindow
        print("✓ 成功导入 GUI 模块")
        
        return True
        
    except ImportError as e:
        print(f"✗ GUI依赖缺失: {e}")
        return False
    except Exception as e:
        print(f"✗ GUI兼容性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试 Cython 优化版本...")
    
    success = True
    
    # 运行各项测试
    if not test_basic_functionality():
        success = False
    
    if not test_game_play():
        success = False
    
    if not test_gui_compatibility():
        success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 所有测试通过！Cython 版本可以正常工作。")
        print("现在可以运行 GUI：")
        print("  cd scripts && python game_gui.py")
    else:
        print("❌ 部分测试失败，需要修复问题。")
    
    return success

if __name__ == "__main__":
    main()
