import os
import warnings

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 禁用INFO和WARNING日志
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import os
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import matplotlib.font_manager as fm


# 设置中文字体 - Arch Linux优化版本
chinese_fonts = [
    'Noto Sans CJK SC',  # 刚安装的Noto字体
    'WenQuanYi Micro Hei',  # 文泉驿微米黑
    'DejaVu Sans',  # 通用字体，支持Unicode
    'Liberation Sans',  # Linux常见字体
    'sans-serif'  # 系统默认
]

# 查找可用的字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
font_found = False

print("可用字体中包含的中文字体:")
for font_name in available_fonts:
    if any(cjk in font_name.lower() for cjk in ['noto', 'cjk', 'wqy', 'micro', 'hei']):
        print(f"  - {font_name}")

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


# --- 配置 ---
# 1. 设置你的 TensorBoard 日志目录 (与你的模型训练配置一致)
#    这个路径是相对于你运行此脚本的位置的。
log_dir = "./tensorboard_logs/continuous_train/"

# 2. 设置保存图像的输出目录
output_dir = "./training_plots/"

# 3. 定义你想要导出的训练指标（Tags）
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
    "time/fps"
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
        run_df = tag_df[tag_df['dir_name'] == run]
        # 使用 'step' 作为 x 轴, 'value' 作为 y 轴
        plt.plot(run_df['step'], run_df['value'], label=run)

    # --- 美化图表 ---
    # 规范化标题和文件名
    clean_title = tag.replace('/', ' - ').replace('_', ' ').title()
    clean_filename = tag.replace('/', '_') + ".png"

    plt.title(clean_title)
    plt.xlabel("训练步数 (Timesteps)")
    plt.ylabel("值 (Value)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --- 保存图像 ---
    output_path = os.path.join(output_dir, clean_filename)
    plt.savefig(output_path)
    plt.close() # 关闭当前图像，释放内存

    print(f"图像已保存至: {output_path}")

print("\n所有图像导出完成！")