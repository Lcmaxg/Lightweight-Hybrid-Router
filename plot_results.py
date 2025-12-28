import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局设置：学术风格 (字体、字号、清晰度)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # 优先使用 Arial
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300  # 高分辨率，适合打印
plt.rcParams['savefig.dpi'] = 300


# ==========================================
# 图 1：检索命中率与延迟对比 (Hit Rate & Latency)
# ==========================================
def plot_performance_comparison():
    # 数据准备
    methods = ['Naive\n(BM25)', 'Baseline\n(BGE)', 'Ours\n(Hybrid)', 'Oracle\n(Ideal)']
    hit_rates = [63.15, 58.60, 69.65, 80.17]  # 命中率 (%)
    latencies = [55.31, 11.55, 12.37, 0]  # 延迟 (ms), Oracle设为0或不显示

    # 创建画布 (双轴图：左轴命中率，右轴延迟)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- 绘制柱状图 (命中率) ---
    bar_width = 0.5
    x = np.arange(len(methods))
    # 使用不同颜色区分 Ours
    colors = ['#A9A9A9', '#A9A9A9', '#1f77b4', '#D3D3D3']  # 灰色基调，Ours用蓝色高亮
    bars = ax1.bar(x, hit_rates, bar_width, color=colors, alpha=0.9, label='Hit Rate', zorder=10)

    # 设置左轴 (Hit Rate)
    ax1.set_ylabel('Hit Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylim(40, 90)  # 设置Y轴范围，让差异更明显
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.tick_params(axis='y', labelsize=10)

    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # --- 绘制折线图 (延迟) - 可选 ---
    # 如果想把延迟也画在同一张图里，取消下面注释
    """
    ax2 = ax1.twinx()  # 创建右轴
    ax2.plot(x[:-1], latencies[:-1], color='#ff7f0e', marker='o', linewidth=2, linestyle='--', label='Latency', zorder=20)
    ax2.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0, 70)
    # 标注延迟数值
    for i, v in enumerate(latencies[:-1]):
        ax2.text(i, v + 2, f'{v:.1f}ms', color='#ff7f0e', ha='center', fontweight='bold')
    """

    # 标题与布局
    plt.title('Comparison of Retrieval Hit Rate Across Methods', fontsize=14, pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()

    # 保存图片
    plt.savefig('result_hit_rate.png')
    print("[√] 已生成: result_hit_rate.png")
    plt.show()


# ==========================================
# 图 2：路由决策分布 (Decision Distribution)
# ==========================================
def plot_decision_distribution():
    # 数据准备
    labels = ['Fast Mode (BM25)', 'Precise Mode (Vector)']
    sizes = [67.8, 32.2]  # 占比 (%)
    colors = ['#66c2a5', '#fc8d62']  # 学术常用配色 (蓝绿/橙红)
    explode = (0.05, 0)  # 将 Fast Mode 稍微分离出来，强调多数

    # 创建画布
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制饼图
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=140, pctdistance=0.85,
                                      shadow=True, textprops={'fontsize': 12})

    # 设置字体样式
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # 添加中心圆 (做成甜甜圈图 Donut Chart，更现代)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # 标题
    plt.title('Router Decision Distribution\n(Simple vs. Complex Queries)', fontsize=14)
    plt.tight_layout()

    # 保存图片
    plt.savefig('result_distribution.png')
    print("[√] 已生成: result_distribution.png")
    plt.show()


if __name__ == "__main__":
    plot_performance_comparison()
    plot_decision_distribution()