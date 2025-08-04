# evaluate_self_play.py
import os
from tqdm import tqdm
from sb3_contrib import MaskablePPO
from Game import GameEnvironment

# --- 配置 ---
# 模型路径，指向您训练好的最终模型
MODEL_PATH = os.path.join("curriculum_models", "final_model.zip")
# 对弈总局数
NUM_GAMES = 200
# 评估时是否使用确定性动作（通常推荐True，以测试模型的最佳策略）
DETERMINISTIC_ACTION = True

def evaluate_self_play():
    """
    执行自我对弈评估，并打印结果。
    """
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 at '{MODEL_PATH}'")
        print("请先确保已成功训练并保存了 final_model.zip")
        return

    print("="*50)
    print("开始自我对弈评估...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"对弈局数: {NUM_GAMES}")
    print(f"确定性动作: {DETERMINISTIC_ACTION}")
    print("="*50)

    # 1. 加载环境
    # 必须使用 curriculum_stage=4 来加载完整的游戏环境
    env = GameEnvironment(curriculum_stage=4)

    # 2. 加载训练好的模型
    model = MaskablePPO.load(MODEL_PATH)

    # 3. 初始化统计数据
    stats = {
        'red_wins': 0,      # 红方（先手）胜场
        'black_wins': 0,    # 黑方（后手）胜场
        'draws': 0          # 平局场次
    }

    # 4. 循环进行对弈
    for _ in tqdm(range(NUM_GAMES), desc="自我对弈评估"):
        # 重置环境，开始新的一局
        obs, info = env.reset()
        terminated, truncated = False, False

        # 在一局游戏结束前，持续进行
        while not terminated and not truncated:
            # 获取合法的动作掩码
            action_mask = info['action_mask']
            
            # 模型根据当前观察值和动作掩码来预测动作
            action, _states = model.predict(
                obs,
                action_masks=action_mask,
                deterministic=DETERMINISTIC_ACTION
            )
            
            # 在环境中执行动作
            obs, reward, terminated, truncated, info = env.step(int(action))

        # 游戏结束后，根据info字典中的'winner'信息更新统计数据
        # Game.py 中的逻辑: winner=1 (红胜), winner=-1 (黑胜), winner=0 (平局)
        winner = info.get('winner')
        if winner == 1:
            stats['red_wins'] += 1
        elif winner == -1:
            stats['black_wins'] += 1
        else: # 包括 winner == 0 或 winner is None (在truncated情况下)
            stats['draws'] += 1

    # 5. 计算并打印最终结果
    total_games = sum(stats.values())
    red_win_rate = (stats['red_wins'] / total_games) * 100
    black_win_rate = (stats['black_wins'] / total_games) * 100
    draw_rate = (stats['draws'] / total_games) * 100

    print("\n" + "="*50)
    print("自我对弈评估完成！")
    print(f"总对弈局数: {total_games}")
    print("-" * 20)
    print(f"红方 (先手) 胜场: {stats['red_wins']} ({red_win_rate:.2f}%)")
    print(f"黑方 (后手) 胜场: {stats['black_wins']} ({black_win_rate:.2f}%)")
    print(f"平局: {stats['draws']} ({draw_rate:.2f}%)")
    print("="*50)

    # 清理环境
    env.close()

if __name__ == "__main__":
    evaluate_self_play()