# main.py

import os
import sys
import warnings

# 设置标准输出编码为UTF-8（解决Windows下的编码问题）
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 禁用INFO和WARNING日志
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# 【移除】不再需要设置模型兼容性

from training.trainer import SelfPlayTrainer

def main():
    """
    项目主入口函数。
    """
    print("=================================================")
    print("      欢迎使用暗棋强化学习 Elo 自我对弈训练框架      ")
    print("=================================================")
    
    try:
        # 实例化并运行训练器
        trainer = SelfPlayTrainer()
        trainer.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        
    except Exception as e:
        print("\n❌ 训练过程中发生严重错误:")
        # 打印异常的详细信息，有助于调试
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n程序执行结束。")

if __name__ == "__main__":
    # 当该脚本被直接执行时，调用main函数
    main()