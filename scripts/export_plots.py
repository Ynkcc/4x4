# src_code/scripts/export_plots.py

import os
import warnings
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from tbparse import SummaryReader

# --- 新增：从项目中导入常量 ---
# 这使得路径配置更加一致和健壮
try:
    from utils.constants import TENSORBOARD_LOG_PATH, ROOT_DIR
except ImportError:
    # 如果直接运行此脚本，可能需要手动设置路径
    print("警告：无法从 utils.constants 导入路径，将使用默认相对路径。")
    # 获取当前脚本所在目录 (scripts)
    _SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (src_code)
    ROOT_DIR = os.path.dirname(_SCRIPTS_DIR)
    TENSORBOARD_LOG_PATH = os.path.join(ROOT_DIR, "tensorboard_logs", "self_play_final")


# 尝试导入scipy，如果失败则使用numpy实现
try:
    from scipy.ndimage import uniform_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    uniform_filter1d = None  # type: ignore
    SCIPY_AVAILABLE = False


def setup_chinese_fonts():
    """根据操作系统设置中文字体"""
    system = platform.system()
    
    # 定义不同操作系统的常用中文字体列表
    font_map = {
        "Windows": ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif'],
        "Linux": ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif'],
        "Darwin": ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'sans-serif']
    }
    chinese_fonts = font_map.get(system, ['DejaVu Sans', 'sans-serif'])
    
    # 查找系统中可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_found = False
    
    print(f"检测到操作系统: {system}")
    
    for font in chinese_fonts:
        if font in available_fonts or font == 'sans-serif':
            plt.rcParams['font.sans-serif'] = [font]
            font_found = True
            print(f"使用字体: {font}")
            break
    
    if not font_found:
        print("警告：未找到推荐的中文字体，将使用默认字体。图表中的中文可能无法正常显示。")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 解决负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

def smooth_data(data, smoothing_window=1):
    """对数据进行平滑处理"""
    if smoothing_window <= 1 or len(data) < smoothing_window:
        return data
    
    if SCIPY_AVAILABLE:
        return uniform_filter1d(data.astype(float), size=smoothing_window)  # type: ignore
    else:
        # 使用numpy的卷积实现移动平均
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(data, kernel, mode='valid')
        # 为了保持长度一致，在数据前端进行填充
        padding_length = len(data) - len(smoothed)
        padding = data[:padding_length]
        return np.concatenate([padding, smoothed])

# --- 脚本执行入口 ---
if __name__ == "__main__":
    
    # 设置中文字体
    setup_chinese_fonts() 
    
    # --- 配置 ---
    # 1. 设置 TensorBoard 日志目录 (从常量导入)
    log_dir = TENSORBOARD_LOG_PATH

    # 2. 设置保存图像的输出目录
    output_dir = os.path.join(ROOT_DIR, "training_plots_rllib")

    # 3. 设置步数限制（只导出最近的N步，设为0表示导出全部）
    # RLlib的步数通常非常大，这里设为最近200万步
    max_steps = 2_000_000  

    # 4. 设置数据平滑参数（平滑窗口大小，1表示不平滑）
    smoothing_window = 100 

    # 5. 【RLlib 修改】定义你想要导出的训练指标（Tags）
    #    RLlib的指标名称与SB3不同，通常包含 "info/learner/main_policy/" 前缀
    tags_to_plot = [
        # --- 环境回报与长度 ---
        "episode_reward_mean",          # 平均每回合奖励
        "episode_len_mean",             # 平均每回合长度

        # --- 核心损失函数 ---
        "info/learner/main_policy/total_loss",      # 总损失
        "info/learner/main_policy/policy_loss",     # 策略损失
        "info/learner/main_policy/vf_loss",         # 价值损失
        "info/learner/main_policy/entropy",         # 熵

        # --- PPO特定指标 ---
        "info/learner/main_policy/kl",              # KL散度 (衡量策略更新幅度)
        "info/learner/main_policy/clip_frac",       # 裁剪比例
        "info/learner/main_policy/vf_explained_var",# 价值函数解释方差
    ]

    # --- 脚本执行 ---
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n正在从 '{log_dir}' 读取 RLlib 日志...")
    if not os.path.exists(log_dir) or not os.listdir(log_dir):
        print(f"错误：日志目录 '{log_dir}' 不存在或为空。请先运行训练。")
        exit()

    # 使用 tbparse 读取日志
    # `log_dir` 指向包含一个或多个 PPO_... 子文件夹的根目录
    reader = SummaryReader(log_dir, extra_columns={"dir_name"})
    df = reader.scalars

    if df.empty:
        print(f"错误：在 '{log_dir}' 中没有找到任何 scalar 数据。请确认训练已开始并生成了日志。")
        exit()

    print("成功读取数据，开始生成图像...")
    print("可用的指标 (Tags):", df['tag'].unique())

    # 遍历每一个需要绘图的指标
    for tag in tags_to_plot:
        # 筛选出当前指标的数据
        tag_df = df[df['tag'] == tag]

        if tag_df.empty:
            print(f"\n警告：未找到指标 '{tag}' 的数据，跳过绘图。")
            continue

        print(f"\n正在为 '{tag}' 生成图像...")

        # 获取所有训练运行的数据 (例如 PPO_dark_chess_multi_agent_...)
        runs = tag_df['dir_name'].unique()
        plt.figure(figsize=(12, 7))

        for run in runs:
            # RLlib的run name可能很长，只取最后一部分
            short_run_name = os.path.basename(run)
            run_df = tag_df[tag_df['dir_name'] == run].copy().sort_values('step')
            
            if max_steps > 0:
                max_step_in_data = run_df['step'].max()
                min_step_threshold = max_step_in_data - max_steps
                run_df = run_df[run_df['step'] >= min_step_threshold]
            
            if run_df.empty:
                print(f"  警告：运行 '{short_run_name}' 在步数限制内没有数据")
                continue
            
            steps = run_df['step'].values
            values = run_df['value'].values
            
            if smoothing_window > 1:
                smoothed_values = smooth_data(values, smoothing_window)
                label = f"{short_run_name} (平滑窗口: {smoothing_window})"
            else:
                smoothed_values = values
                label = short_run_name
            
            plt.plot(steps, smoothed_values, label=label, linewidth=2)

        # --- 美化图表 ---
        # 规范化标题和文件名
        clean_title = tag.replace('info/learner/main_policy/', '').replace('/', ' - ').replace('_', ' ').title()
        clean_filename = tag.replace('/', '_') + ".png"
        
        if max_steps > 0:
            clean_title += f" (最近 {max_steps/1_000_000:.1f}M 步)"

        plt.title(clean_title, fontsize=14, fontweight='bold')
        plt.xlabel("训练总步数 (Total Timesteps)", fontsize=12)
        plt.ylabel("值 (Value)", fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()

        # --- 保存图像 ---
        output_path = os.path.join(output_dir, clean_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"图像已保存至: {output_path}")

    print(f"\n✅ 所有图像导出完成！")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 步数限制: {'全部数据' if max_steps <= 0 else f'最近 {max_steps:,} 步'}")
    print(f"  - 数据平滑: {'无平滑' if smoothing_window <= 1 else f'窗口大小 {smoothing_window}'}")