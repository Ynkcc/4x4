import os
import warnings

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 禁用INFO和WARNING日志
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import os
import platform
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import matplotlib.font_manager as fm
import numpy as np

try:
    from scipy.ndimage import uniform_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    uniform_filter1d = None  # type: ignore
    SCIPY_AVAILABLE = False


def setup_chinese_fonts():
    """根据操作系统设置中文字体"""
    system = platform.system()
    
    if system == "Windows":
        # Windows常用中文字体
        chinese_fonts = [
            'SimHei',           # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',           # 宋体
            'KaiTi',            # 楷体
            'FangSong',         # 仿宋
            'sans-serif'        # 系统默认
        ]
    elif system == "Linux":
        # Linux常用中文字体
        chinese_fonts = [
            'Noto Sans CJK SC',     # Google Noto字体
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Droid Sans Fallback',  # Android字体
            'DejaVu Sans',          # 通用字体，支持Unicode
            'Liberation Sans',      # Linux常见字体
            'sans-serif'            # 系统默认
        ]
    elif system == "Darwin":  # macOS
        # macOS常用中文字体
        chinese_fonts = [
            'PingFang SC',          # 苹方简体
            'Hiragino Sans GB',     # 冬青黑体简体中文
            'STHeiti',              # 华文黑体
            'Arial Unicode MS',     # Arial Unicode
            'sans-serif'            # 系统默认
        ]
    else:
        # 其他系统使用通用字体
        chinese_fonts = ['DejaVu Sans', 'sans-serif']
    
    # 查找可用的字体
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
        print("警告：未找到合适的中文字体，使用默认字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 解决负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

def smooth_data(data, smoothing_window=1):
    """对数据进行平滑处理"""
    if smoothing_window <= 1:
        return data
    
    if SCIPY_AVAILABLE:
        return uniform_filter1d(data.astype(float), size=smoothing_window)  # type: ignore
    else:
        # 使用numpy实现简单的移动平均
        if len(data) < smoothing_window:
            return data
        
        # 使用卷积实现移动平均
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(data, kernel, mode='valid')
        
        # 为了保持长度一致，在开头填充原始数据
        padding_length = len(data) - len(smoothed)
        padding = data[:padding_length]
        
        return np.concatenate([padding, smoothed])

# 设置中文字体
setup_chinese_fonts() 


# --- 配置 ---
# 1. 设置你的 TensorBoard 日志目录 (与你的模型训练配置一致)
#    这个路径是相对于你运行此脚本的位置的。
log_dir = "./tensorboard_logs/self_play_final/"

# 2. 设置保存图像的输出目录
output_dir = "./training_plots/"

# 3. 设置步数限制（只导出最近的N步，设为0表示导出全部）
max_steps = 500000  # 最近500K步

# 4. 设置数据平滑参数（平滑窗口大小，1表示不平滑）
smoothing_window = 50  # 可以根据需要调整，数值越大越平滑

# 5. 定义你想要导出的训练指标（Tags）
tags_to_plot = [
    "train/approx_kl",
    "train/clip_fraction",
    "train/clip_range",
    "train/entropy_loss",
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "train/explained_variance"
]

# --- 脚本执行 ---

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

print(f"正在从 '{log_dir}' 读取日志...")

# 检查日志目录是否存在
if not os.path.exists(log_dir):
    raise ValueError(f"错误：日志目录 '{log_dir}' 不存在。请确保路径正确。")

# 使用 tbparse 读取日志
# `log_dir` 指向包含一个或多个 PPO_1, PPO_2 等子文件夹的根目录
reader = SummaryReader(log_dir, extra_columns={"dir_name"})
df = reader.scalars

# 检查是否成功读取到数据
if df.empty:
    raise ValueError(f"错误：在 '{log_dir}' 中没有找到任何 scalar 数据。请确认是否已开始训练并生成了日志。")

print("成功读取数据，开始生成图像...")
print("可用的指标 (Tags):", df['tag'].unique())


# 遍历每一个需要绘图的指标
for tag in tags_to_plot:
    # 从DataFrame中筛选出当前指标的数据
    tag_df = df[df['tag'] == tag]

    if tag_df.empty:
        print(f"\n警告：未找到指标 '{tag}' 的数据，跳过绘图。")
        continue

    print(f"\n正在为 '{tag}' 生成图像...")

    # 获取所有训练运行的数据 (例如 PPO_1, PPO_2...)
    runs = tag_df['dir_name'].unique()

    plt.figure(figsize=(12, 7))

    # 在一张图上绘制每次运行的曲线
    for run in runs:
        run_df = tag_df[tag_df['dir_name'] == run].copy()
        
        # 如果设置了步数限制，只保留最近的数据
        if max_steps > 0:
            max_step_in_data = run_df['step'].max()
            min_step_threshold = max_step_in_data - max_steps
            run_df = run_df[run_df['step'] >= min_step_threshold]
        
        if run_df.empty:
            print(f"  警告：运行 '{run}' 在步数限制 {max_steps} 内没有数据")
            continue
        
        # 排序确保数据按步数顺序
        run_df = run_df.sort_values('step')
        
        # 应用数据平滑
        steps = run_df['step'].values
        values = run_df['value'].values
        
        if smoothing_window > 1:
            smoothed_values = smooth_data(values, smoothing_window)
            # 创建标签来区分原始数据和平滑数据
            label = f"{run} (平滑窗口: {smoothing_window})"
        else:
            smoothed_values = values
            label = run
        
        # 使用 'step' 作为 x 轴, 平滑后的数据作为 y 轴
        plt.plot(steps, smoothed_values, label=label, linewidth=2)

    # --- 美化图表 ---
    # 规范化标题和文件名
    clean_title = tag.replace('/', ' - ').replace('_', ' ').title()
    clean_filename = tag.replace('/', '_') + ".png"
    
    # 在标题中添加步数限制信息
    if max_steps > 0:
        clean_title += f" (最近 {max_steps//1000}K 步)"

    plt.title(clean_title, fontsize=14, fontweight='bold')
    plt.xlabel("训练步数 (Timesteps)", fontsize=12)
    plt.ylabel("值 (Value)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # --- 保存图像 ---
    output_path = os.path.join(output_dir, clean_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # 关闭当前图像，释放内存

    print(f"图像已保存至: {output_path}")

print(f"\n所有图像导出完成！")
print(f"配置信息：")
print(f"  - 步数限制: {'全部数据' if max_steps <= 0 else f'最近 {max_steps} 步'}")
print(f"  - 数据平滑: {'无平滑' if smoothing_window <= 1 else f'窗口大小 {smoothing_window}'}")
print(f"  - 字体系统: {platform.system()}")
print(f"  - 输出目录: {output_dir}")