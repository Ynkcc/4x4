# test_selfplay.py - 测试自我对弈功能
import os
import numpy as np
from Game import GameEnvironment
from sb3_contrib import MaskablePPO

def test_selfplay_environment():
    """测试自我对弈环境是否能正常工作"""
    print("测试自我对弈环境...")
    
    # 检查课程学习最终模型是否存在
    curriculum_model_path = "cnn_curriculum_models/final_model_cnn.zip"
    if not os.path.exists(curriculum_model_path):
        print(f"错误: 课程学习最终模型不存在: {curriculum_model_path}")
        return False
    
    # 创建没有对手的环境 (传统模式)
    print("1. 测试传统环境 (无对手模型)...")
    env_no_opponent = GameEnvironment(curriculum_stage=4)
    obs, info = env_no_opponent.reset()
    print(f"   状态形状: board={obs['board'].shape}, scalars={obs['scalars'].shape}")
    print(f"   合法动作数量: {np.sum(info['action_mask'])}")
    
    # 创建带对手的环境
    print("2. 测试自我对弈环境 (带对手模型)...")
    env_with_opponent = GameEnvironment(
        curriculum_stage=4, 
        opponent_policy=curriculum_model_path
    )
    obs, info = env_with_opponent.reset()
    print(f"   状态形状: board={obs['board'].shape}, scalars={obs['scalars'].shape}")
    print(f"   合法动作数量: {np.sum(info['action_mask'])}")
    print(f"   对手模型加载状态: {'成功' if env_with_opponent.opponent_model is not None else '失败'}")
    
    if env_with_opponent.opponent_model is None:
        print("   警告: 对手模型未能正确加载")
        return False
    
    # 执行几步测试
    print("3. 执行测试步骤...")
    for step in range(5):
        # 获取合法动作
        action_mask = info['action_mask']
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            print(f"   步骤 {step}: 没有合法动作，游戏结束")
            break
        
        # 随机选择一个合法动作
        action = np.random.choice(valid_actions)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env_with_opponent.step(action)
        
        print(f"   步骤 {step}: 动作={action}, 奖励={reward:.4f}, 结束={terminated}, 截断={truncated}")
        
        if terminated or truncated:
            print(f"   游戏结束，获胜者: {info.get('winner', '无')}")
            break
    
    print("测试完成！")
    env_no_opponent.close()
    env_with_opponent.close()
    return True

if __name__ == "__main__":
    success = test_selfplay_environment()
    if success:
        print("\n✅ 自我对弈环境测试通过！")
    else:
        print("\n❌ 自我对弈环境测试失败！")
