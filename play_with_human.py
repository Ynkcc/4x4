#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sb3_contrib import MaskablePPO
from Game import GameEnvironment

class HumanPlayer:
    """人类玩家类，处理人机交互"""
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.name = "红方" if player_id == 1 else "黑方"
        self.color = "\033[91m" if player_id == 1 else "\033[94m"
        self.color_end = "\033[0m"
    
    def get_action(self, env):
        """获取人类玩家的动作输入"""
        print(f"\n{self.color}轮到{self.name}下棋{self.color_end}")
        
        # 显示棋盘
        self.display_board(env)
        
        # 获取合法动作
        valid_actions = env.action_masks()
        valid_action_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_action_indices) == 0:
            print(f"{self.color}{self.name}无合法动作！{self.color_end}")
            return None
        
        # 显示合法动作
        print(f"\n{self.color}合法动作：{self.color_end}")
        action_descriptions = []
        for i, action_idx in enumerate(valid_action_indices):
            pos_idx = action_idx // 5
            action_sub_idx = action_idx % 5
            row = pos_idx // 4
            col = pos_idx % 4
            
            if action_sub_idx == 4:
                desc = f"翻开 ({row},{col})"
            else:
                directions = ["上", "下", "左", "右"]
                desc = f"从 ({row},{col}) 向{directions[action_sub_idx]}移动/攻击"
            
            action_descriptions.append(desc)
            print(f"{i+1}. {desc}")
        
        # 获取用户输入
        while True:
            try:
                choice = input(f"\n{self.color}请选择动作 (1-{len(valid_action_indices)}): {self.color_end}")
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(valid_action_indices):
                    selected_action = valid_action_indices[choice_idx]
                    print(f"{self.color}您选择了: {action_descriptions[choice_idx]}{self.color_end}")
                    return selected_action
                else:
                    print(f"{self.color}无效选择，请输入 1-{len(valid_action_indices)} 之间的数字{self.color_end}")
            except ValueError:
                print(f"{self.color}请输入有效的数字{self.color_end}")
            except KeyboardInterrupt:
                print(f"\n{self.color}游戏被中断{self.color_end}")
                return None
    
    def display_board(self, env):
        """显示当前棋盘状态"""
        print("\n" + "="*50)
        print("当前棋盘状态:")
        env.render()
        print("="*50)

class AIPlayer:
    """AI玩家类"""
    
    def __init__(self, player_id, model_path):
        self.player_id = player_id
        self.name = "红方AI" if player_id == 1 else "黑方AI"
        self.color = "\033[91m" if player_id == 1 else "\033[94m"
        self.color_end = "\033[0m"
        
        # 加载训练好的模型
        try:
            # 创建临时环境用于获取环境规格
            temp_env = GameEnvironment()
            self.model = MaskablePPO.load(model_path, env=temp_env)
            print(f"{self.color}AI模型加载成功: {model_path}{self.color_end}")
        except Exception as e:
            print(f"{self.color}AI模型加载失败: {e}{self.color_end}")
            raise
    
    def get_action(self, env):
        """获取AI的动作"""
        print(f"\n{self.color}{self.name}正在思考...{self.color_end}")
        
        # 获取当前状态和动作掩码
        obs = env.get_state()
        action_mask = env.action_masks()
        
        # 让AI选择动作
        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=False)
        
        # 解释AI的动作
        pos_idx = action // 5
        action_sub_idx = action % 5
        row = pos_idx // 4
        col = pos_idx % 4
        
        if action_sub_idx == 4:
            desc = f"翻开 ({row},{col})"
        else:
            directions = ["上", "下", "左", "右"]
            desc = f"从 ({row},{col}) 向{directions[action_sub_idx]}移动/攻击"
        
        print(f"{self.color}{self.name}选择: {desc}{self.color_end}")
        return action

def select_model():
    """选择AI模型"""
    log_dir = "./banqi_ppo_logs/"
    
    # 查找可用的模型文件
    model_files = []
    
    # 检查最终模型
    final_model = os.path.join(log_dir, "banqi_ppo_model.zip")
    if os.path.exists(final_model):
        model_files.append(("最终模型", final_model))
    
    # 检查最佳模型
    best_model = os.path.join(log_dir, "best_model.zip")
    if os.path.exists(best_model):
        model_files.append(("最佳模型", best_model))
    
    # 检查检查点模型
    if os.path.exists(log_dir):
        for file in os.listdir(log_dir):
            if file.startswith("rl_model_") and file.endswith("_steps.zip"):
                steps = file.replace("rl_model_", "").replace("_steps.zip", "")
                model_files.append((f"检查点模型 ({steps} steps)", os.path.join(log_dir, file)))
    
    if not model_files:
        print("未找到训练好的模型！请先运行训练脚本。")
        return None
    
    # 显示可用模型
    print("\n可用的AI模型:")
    for i, (name, path) in enumerate(model_files):
        print(f"{i+1}. {name}")
    
    # 让用户选择
    while True:
        try:
            choice = input(f"\n请选择AI模型 (1-{len(model_files)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(model_files):
                selected_model = model_files[choice_idx][1]
                print(f"选择了: {model_files[choice_idx][0]}")
                return selected_model
            else:
                print(f"无效选择，请输入 1-{len(model_files)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n退出选择")
            return None

def select_player_side():
    """选择人类玩家的一方"""
    print("\n选择您要扮演的一方:")
    print("1. 红方 (先手)")
    print("2. 黑方 (后手)")
    
    while True:
        try:
            choice = input("请选择 (1-2): ")
            if choice == "1":
                return 1  # 红方
            elif choice == "2":
                return -1  # 黑方
            else:
                print("无效选择，请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n退出选择")
            return None

def main():
    """主函数"""
    print("="*60)
    print("欢迎来到 4x4 暗棋人机对弈！")
    print("="*60)
    
    # 选择AI模型
    model_path = select_model()
    if model_path is None:
        return
    
    # 选择人类玩家一方
    human_side = select_player_side()
    if human_side is None:
        return
    
    ai_side = -human_side
    
    try:
        # 创建玩家
        human_player = HumanPlayer(human_side)
        ai_player = AIPlayer(ai_side, model_path)
        
        # 创建游戏环境
        env = GameEnvironment(render_mode="human")
        obs, info = env.reset()
        
        print(f"\n游戏开始！")
        print(f"人类玩家: {human_player.name}")
        print(f"AI玩家: {ai_player.name}")
        print(f"红方先手")
        
        # 游戏主循环
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            current_player = env.current_player
            
            # 根据当前玩家选择行动者
            if current_player == human_side:
                action = human_player.get_action(env)
            else:
                action = ai_player.get_action(env)
            
            if action is None:
                break
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 显示游戏结果
            if terminated or truncated:
                print("\n" + "="*60)
                print("游戏结束！")
                
                # 最终棋盘状态
                env.render()
                
                winner = info.get('winner')
                if winner == 1:
                    print("\033[91m红方获胜！\033[0m")
                elif winner == -1:
                    print("\033[94m黑方获胜！\033[0m")
                else:
                    print("平局！")
                
                if winner == human_side:
                    print("🎉 恭喜您获胜！")
                elif winner == ai_side:
                    print("😔 AI获胜，继续努力！")
                else:
                    print("🤝 平局，势均力敌！")
                
                print("="*60)
        
    except Exception as e:
        print(f"游戏过程中发生错误: {e}")
    except KeyboardInterrupt:
        print("\n游戏被中断")
    
    print("感谢游玩！")

if __name__ == "__main__":
    main()
