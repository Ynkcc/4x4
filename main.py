import sys
import os

# 将项目根目录添加到Python路径中，以确保可以正确导入其他模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.trainer import RLLibSelfPlayTrainer

def main():
    """
    项目主入口函数。
    """
    print("=" * 70)
    print("      欢迎使用暗棋强化学习 Elo 自我对弈训练框架 (RLlib版)      ")
    print("=" * 70)

    try:
        # 实例化并运行训练器
        trainer = RLLibSelfPlayTrainer()
        trainer.run()

    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断。")

    except Exception as e:
        print(f"\n❌ 训练过程中发生严重错误: {e}")
        # 打印异常的详细信息，有助于调试
        import traceback
        traceback.print_exc()

    finally:
        print("\n程序执行结束。")

if __name__ == "__main__":
    # 当该脚本被直接执行时，调用main函数
    main()
