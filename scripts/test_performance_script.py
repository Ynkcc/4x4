#!/usr/bin/env python3
"""
测试性能测试脚本是否正常工作
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入是否正常"""
    print("=== 测试导入 ===")
    
    try:
        from performance_test import PerformanceTester
        print("✓ 成功导入 PerformanceTester")
    except Exception as e:
        print(f"✗ 导入 PerformanceTester 失败: {e}")
        return False
    
    # 测试各个版本的导入
    try:
        from performance_test import ORIGINAL_VERSION_AVAILABLE, BITBOARD_VERSION_AVAILABLE, CYTHON_VERSION_AVAILABLE
        print(f"✓ 原版可用: {ORIGINAL_VERSION_AVAILABLE}")
        print(f"✓ Bitboard版本可用: {BITBOARD_VERSION_AVAILABLE}")
        print(f"✓ Cython版本可用: {CYTHON_VERSION_AVAILABLE}")
    except Exception as e:
        print(f"✗ 导入版本标志失败: {e}")
        return False
    
    return True

def test_small_performance():
    """运行小规模性能测试"""
    print("\n=== 小规模性能测试 ===")
    
    try:
        from performance_test import PerformanceTester
        tester = PerformanceTester(random_seed=42)
        
        # 运行10局游戏的测试
        print("运行10局游戏的快速测试...")
        results = tester.run_comprehensive_test(num_games=10, max_steps_per_game=100)
        
        if results:
            print("✓ 性能测试成功完成")
            print(f"✓ 测试了 {len(results)} 个版本")
            return True
        else:
            print("✗ 性能测试返回空结果")
            return False
            
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试性能测试脚本...")
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_small_performance():
        success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 所有测试通过！性能测试脚本可以正常工作。")
        print("\n现在可以运行完整的性能测试：")
        print("  python performance_test.py all")
        print("  python performance_test.py cython")
        print("  python performance_test.py profile")
    else:
        print("❌ 部分测试失败，需要修复问题。")
    
    return success

if __name__ == "__main__":
    main()
