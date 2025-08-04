# 暗棋强化学习 Elo 自我对弈训练框架

## 项目结构

```
gym/
├── models/                     # 所有模型输出的根目录
│   └── self_play_final/        # 本次训练输出的目录
│       ├── main_opponent.zip   # 当前最强的模型
│       ├── challenger.zip      # 临时的挑战者模型
│       └── final_model.zip     # 最终训练完成的模型
├── game/
│   ├── __init__.py
│   ├── environment.py          # 包含 GameEnvironment 类
│   └── policy.py               # 包含自定义的CNN策略网络
├── training/
│   ├── __init__.py
│   ├── manager.py              # 共享对手模型管理器
│   ├── evaluator.py            # 独立的模型评估器
│   └── trainer.py              # 核心的Elo自我对弈训练器
├── utils/
│   ├── __init__.py
│   ├── constants.py            # 存放所有路径和超参数
│   └── scheduler.py            # 学习率调度函数
├── main.py                     # 整个项目的入口点
├── human_vs_ai.py             # 人机对战脚本
└── export_plots.py            # 图表导出脚本
```

## 特性

### 🚀 最先进的训练流程
- 采用Elo评估机制，确保只有更强的模型才能成为基准对手
- 实现稳定、高质量的迭代训练过程

### 🏗️ 清晰的项目结构
- 代码被拆分到不同的目录和文件中，职责分明
- 易于维护和扩展

### 🔄 灵活的模型加载
- 智能启动逻辑：如果存在main_opponent.zip，自动恢复训练
- 否则从课程学习或自我对弈的最终模型开始全新训练

### 📝 详尽的中文注释
- 对核心模块、类、函数和关键代码段都添加了注释
- 解释功能和设计思想

### ⚡ 性能优化
- 使用共享模型管理器避免重复加载
- 支持多进程并行训练和评估

## 使用方法

### 1. 准备环境
确保已安装必要的库：
```bash
pip install stable-baselines3 sb3-contrib gymnasium torch numpy
```

### 2. 启动训练
在项目根目录下执行：
```bash
python main.py
```

### 3. 恢复训练
如果训练中断，只需再次运行 `python main.py`。程序会自动检测并从上次的主宰者模型继续训练。

### 4. 监控训练
可以使用TensorBoard监控训练过程：
```bash
tensorboard --logdir=tensorboard_logs/self_play_final
```

## 配置参数

主要配置在 `utils/constants.py` 中：

- `TOTAL_TRAINING_LOOPS`: 总训练循环数（默认100）
- `STEPS_PER_LOOP`: 每循环训练步数（默认50,000）
- `EVALUATION_GAMES`: 评估对战局数（默认20）
- `EVALUATION_THRESHOLD`: 挑战成功胜率阈值（默认0.55）
- `N_ENVS`: 并行环境数量（默认8）

## 训练流程

1. **初始化**: 检查是否存在已有模型，支持恢复训练
2. **训练**: 学习者模型训练指定步数
3. **评估**: 挑战者与主宰者对战评估
4. **更新**: 根据胜率决定是否更新主宰者
5. **循环**: 重复上述过程直到完成所有训练循环

## 输出文件

- `models/self_play_final/main_opponent.zip`: 当前最强模型
- `models/self_play_final/final_model.zip`: 最终训练完成的模型
- `tensorboard_logs/self_play_final/`: TensorBoard日志
