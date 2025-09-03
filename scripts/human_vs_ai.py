# scripts/human_vs_ai.py - 人机对战脚本

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import GameEnvironment
from core.policy import RLLibCustomNetwork
from utils.constants import *

class HumanPlayer:
    """人类玩家类。"""
    def __init__(self, player_id: int):
        self.player_id = player_id

    def get_action(self, env: GameEnvironment) -> int:
        """获取人类玩家的动作。"""
        legal_actions = env.get_legal_actions()
        while True:
            try:
                print(f"\n当前玩家 {self.player_id} 的回合")
                print("棋盘状态:")
                env.render()
                print(f"合法动作: {legal_actions}")
                action = int(input("请输入动作编号: "))
                if action in legal_actions:
                    return action
                else:
                    print("非法动作，请重新输入")
            except ValueError:
                print("请输入有效的数字")

class AIPlayer:
    """AI玩家类。"""
    def __init__(self, model_path: str, player_id: int):
        self.player_id = player_id
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载AI模型。"""
        # 这里需要实现模型加载逻辑
        pass

    def get_action(self, env: GameEnvironment) -> int:
        """获取AI的动作。"""
        # 这里需要实现AI决策逻辑
        legal_actions = env.get_legal_actions()
        return np.random.choice(legal_actions)  # 临时随机选择

def play_game(human_player_id: int, ai_model_path: str):
    """进行一场人机对战。"""
    env = GameEnvironment()
    env.reset()

    human = HumanPlayer(human_player_id)
    ai = AIPlayer(ai_model_path, -human_player_id)

    current_player = 1  # 假设人类先手

    while not env.is_game_over():
        if current_player == human_player_id:
            action = human.get_action(env)
        else:
            action = ai.get_action(env)

        _, _, terminated, truncated, info = env.step(action)
        current_player *= -1

        if terminated or truncated:
            break

    winner = env.get_winner()
    if winner == human_player_id:
        print("人类获胜！")
    elif winner == -human_player_id:
        print("AI获胜！")
    else:
        print("平局！")

if __name__ == "__main__":
    # 示例用法
    human_id = 1  # 人类玩家ID
    ai_model = "path/to/ai/model"
    play_game(human_id, ai_model)
