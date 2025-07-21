#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速启动人机对弈的脚本
"""

import os
import sys
from play_with_human import main

def quick_start():
    """快速启动，使用默认设置"""
    print("快速启动人机对弈...")
    
    # 检查是否有可用的模型
    log_dir = "./banqi_ppo_logs/"
    
    # 优先使用最佳模型
    best_model = os.path.join(log_dir, "best_model.zip")
    final_model = os.path.join(log_dir, "banqi_ppo_model.zip")
    
    if os.path.exists(best_model):
        print(f"找到最佳模型: {best_model}")
        model_path = best_model
    elif os.path.exists(final_model):
        print(f"找到最终模型: {final_model}")
        model_path = final_model
    else:
        print("未找到训练好的模型！")
        print("请先运行 python train.py 进行训练")
        return
    
    # 运行游戏
    print("启动游戏...")
    main()

if __name__ == "__main__":
    quick_start()
