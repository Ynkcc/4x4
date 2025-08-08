# 暗棋强化学习训练框架

一个基于强化学习的4x4暗棋训练项目，包含Elo自我对弈训练、持续学习、人机对战和可视化分析等完整功能。

## 项目结构

```
gym/
├── main.py                     # Elo自我对弈训练主入口
├── continuous_training.py      # 持续训练脚本
├── human_vs_ai.py             # 人机对战GUI界面
├── export_plots.py            # 训练数据可视化导出
├── README.md                   # 项目说明文档
├── game/                       # 游戏核心模块
│   ├── __init__.py
│   ├── environment.py          # 暗棋游戏环境(Gymnasium兼容)
│   └── policy.py               # 自定义CNN策略网络
├── training/                   # 训练相关模块
│   ├── __init__.py
│   ├── evaluator.py            # 模型评估器
│   ├── simple_agent.py         # 简单规则代理
│   └── trainer.py              # Elo自我对弈训练器
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── constants.py            # 全局配置和常量
│   ├── model_compatibility.py  # 模型兼容性处理
│   └── scheduler.py            # 学习率调度器
├── models/                     # 模型存储目录
│   ├── train_simple.zip        # 基础训练模型
│   ├── train_simple_v2.zip     # 改进版基础模型
│   ├── continuous_train/       # 持续训练模型
│   │   ├── current_model.zip   # 当前训练模型
│   │   ├── backup_model.zip    # 备份模型
│   │   └── best_model_session_*.zip  # 各阶段最佳模型
│   └── self_play_final/        # Elo自我对弈最终模型
│       ├── main_opponent.zip   # 主要对手模型
│       └── challenger.zip      # 挑战者模型
├── tensorboard_logs/           # TensorBoard日志
│   └── self_play_final/        # 训练监控日志
└── training_plots/            # 训练图表
    ├── rollout_ep_*.png        # 回合统计图表
    ├── train_*.png             # 训练损失图表
    └── time_fps.png            # 性能监控图表
```
