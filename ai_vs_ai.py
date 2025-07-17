#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI vs AI 对弈脚本，用于观察AI的水平
"""

import os
import numpy as np
from sb3_contrib import MaskablePPO
from Game import GameEnvironment
import time

class AIPlayer:
    """AI玩家类"""
    
    def __init__(self, player_id, model_path, name="AI"):
        self.player_id = player_id
        self.name = name
        self.color = "\033[91m" if player_id == 1 else "\033[94m"
        self.color_end = "\033[0m"
        
        # 加载训练好的模型
        try:
            # 创建临时环境用于获取环境规格
            temp_env = GameEnvironment()
            self.model = MaskablePPO.load(model_path, env=temp_env)
            print(f"{self.color}{self.name}模型加载成功: {model_path}{self.color_end}")
        except Exception as e:
            print(f"{self.color}{self.name}模型加载失败: {e}{self.color_end}")
            raise
    
    def get_action(self, env, deterministic=False):
        """获取AI的动作"""
        # 获取当前状态和动作掩码
        obs = env.get_state()
        action_mask = env.action_masks()
        
        # 让AI选择动作
        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=deterministic)
        
        return action
    
    def describe_action(self, action):
        """描述动作"""
        pos_idx = action // 5
        action_sub_idx = action % 5
        row = pos_idx // 4
        col = pos_idx % 4
        
        if action_sub_idx == 4:
            return f"翻开 ({row},{col})"
        else:
            directions = ["上", "下", "左", "右"]
            return f"从 ({row},{col}) 向{directions[action_sub_idx]}移动/攻击"

def select_models():
    """选择两个AI模型"""
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
    
    # 检查检查点模型（只显示一些关键的）
    if os.path.exists(log_dir):
        checkpoint_files = []
        for file in os.listdir(log_dir):
            if file.startswith("rl_model_") and file.endswith("_steps.zip"):
                steps = int(file.replace("rl_model_", "").replace("_steps.zip", ""))
                checkpoint_files.append((steps, file))
        
        # 排序并只显示一些关键的检查点
        checkpoint_files.sort(key=lambda x: x[0])
        for steps, file in checkpoint_files[-10:]:  # 只显示最后10个检查点
            model_files.append((f"检查点模型 ({steps} steps)", os.path.join(log_dir, file)))
    
    if not model_files:
        print("未找到训练好的模型！请先运行训练脚本。")
        return None, None
    
    # 显示可用模型
    print("\n可用的AI模型:")
    for i, (name, path) in enumerate(model_files):
        print(f"{i+1}. {name}")
    
    # 选择红方AI
    print("\n选择红方AI:")
    while True:
        try:
            choice = input(f"请选择红方AI模型 (1-{len(model_files)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(model_files):
                red_model = model_files[choice_idx][1]
                print(f"红方AI选择了: {model_files[choice_idx][0]}")
                break
            else:
                print(f"无效选择，请输入 1-{len(model_files)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n退出选择")
            return None, None
    
    # 选择黑方AI
    print("\n选择黑方AI:")
    while True:
        try:
            choice = input(f"请选择黑方AI模型 (1-{len(model_files)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(model_files):
                black_model = model_files[choice_idx][1]
                print(f"黑方AI选择了: {model_files[choice_idx][0]}")
                break
            else:
                print(f"无效选择，请输入 1-{len(model_files)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n退出选择")
            return None, None
    
    return red_model, black_model

def run_single_game(red_model_path, black_model_path, game_num=1, show_details=True, delay=0.5):
    """运行单局游戏"""
    
    # 创建AI玩家
    red_ai = AIPlayer(1, red_model_path, "红方AI")
    black_ai = AIPlayer(-1, black_model_path, "黑方AI")
    
    # 创建游戏环境
    env = GameEnvironment(render_mode="human" if show_details else None)
    obs, info = env.reset()
    
    if show_details:
        print(f"\n{'='*60}")
        print(f"第 {game_num} 局游戏开始！")
        print(f"{'='*60}")
    
    # 游戏主循环
    terminated = False
    truncated = False
    move_count = 0
    
    while not terminated and not truncated:
        current_player = env.current_player
        move_count += 1
        
        # 根据当前玩家选择AI
        if current_player == 1:
            action = red_ai.get_action(env)
            if show_details:
                print(f"\n回合 {move_count}: {red_ai.color}{red_ai.name}{red_ai.color_end} - {red_ai.describe_action(action)}")
        else:
            action = black_ai.get_action(env)
            if show_details:
                print(f"\n回合 {move_count}: {black_ai.color}{black_ai.name}{black_ai.color_end} - {black_ai.describe_action(action)}")
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 显示棋盘状态
        if show_details:
            env.render()
            if delay > 0:
                time.sleep(delay)
    
    # 游戏结果
    winner = info.get('winner')
    if show_details:
        print(f"\n{'='*60}")
        print(f"第 {game_num} 局结束！总回合数: {move_count}")
        
        if winner == 1:
            print(f"{red_ai.color}红方AI获胜！{red_ai.color_end}")
        elif winner == -1:
            print(f"{black_ai.color}黑方AI获胜！{black_ai.color_end}")
        else:
            print("平局！")
        
        print(f"最终分数: 红方 {env.scores[1]} vs 黑方 {env.scores[-1]}")
        print(f"{'='*60}")
    
    return winner, move_count, env.scores[1], env.scores[-1]

def run_multiple_games(red_model_path, black_model_path, num_games=10):
    """运行多局游戏并统计结果"""
    
    results = {'red_wins': 0, 'black_wins': 0, 'draws': 0}
    total_moves = 0
    red_total_score = 0
    black_total_score = 0
    
    print(f"\n开始运行 {num_games} 局AI对弈...")
    
    for i in range(num_games):
        winner, move_count, red_score, black_score = run_single_game(
            red_model_path, black_model_path, i+1, show_details=False, delay=0
        )
        
        if winner == 1:
            results['red_wins'] += 1
        elif winner == -1:
            results['black_wins'] += 1
        else:
            results['draws'] += 1
        
        total_moves += move_count
        red_total_score += red_score
        black_total_score += black_score
        
        print(f"第 {i+1} 局完成 - 胜者: {'红方' if winner == 1 else '黑方' if winner == -1 else '平局'}")
    
    # 显示统计结果
    print(f"\n{'='*60}")
    print(f"AI对弈统计结果 ({num_games} 局)")
    print(f"{'='*60}")
    print(f"红方胜利: {results['red_wins']} 局 ({results['red_wins']/num_games*100:.1f}%)")
    print(f"黑方胜利: {results['black_wins']} 局 ({results['black_wins']/num_games*100:.1f}%)")
    print(f"平局: {results['draws']} 局 ({results['draws']/num_games*100:.1f}%)")
    print(f"平均回合数: {total_moves/num_games:.1f}")
    print(f"平均分数: 红方 {red_total_score/num_games:.1f}, 黑方 {black_total_score/num_games:.1f}")
    print(f"{'='*60}")

def main():
    """主函数"""
    print("="*60)
    print("AI vs AI 对弈观察系统")
    print("="*60)
    
    # 选择模型
    red_model, black_model = select_models()
    if red_model is None or black_model is None:
        return
    
    # 选择运行模式
    print("\n选择运行模式:")
    print("1. 单局对弈 (显示详细过程)")
    print("2. 多局对弈 (统计结果)")
    
    while True:
        try:
            choice = input("请选择模式 (1-2): ")
            if choice == "1":
                run_single_game(red_model, black_model)
                break
            elif choice == "2":
                num_games = int(input("请输入对弈局数 (建议10-100): ") or "10")
                run_multiple_games(red_model, black_model, num_games)
                break
            else:
                print("无效选择，请输入 1 或 2")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n退出")
            return
    
    print("\n观察完成！")

if __name__ == "__main__":
    main()
