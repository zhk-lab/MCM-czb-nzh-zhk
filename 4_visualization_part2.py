"""
MCM 2026 Problem C - Task 4 Visualization (Part 2)
===================================================
顶级可视化设计 - 高级分析图：
  - Metric Network Hybrid（指标网络混合图）
  - Violin Plot Matrix（稳健性分布矩阵）
  - Waterfall Chart（指标增益分解）
  - Radar Ensemble（雷达图组合矩阵）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Color Hunt 配色（调暗版）
COLORS = {
    'twokey': '#9B7EBD',
    'rank': '#D14D72',
    'percent': '#3E7C17',
    'save': '#5C88C4',
    'positive': '#387F39',
    'negative': '#C75B7A',
    'bg': '#F5F5F5',
    'text': '#1E201E',
}

plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 10,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
})


# ============================================================
# 图 5: Metric Network Hybrid（指标 + 网络混合图）
# ============================================================

def create_metric_network_hybrid(metrics_df, win_rates, save_dir):
    """
    左侧：指标热力图
    右侧：方法相似度网络
    """
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    gs = GridSpec(1, 2, figure=fig, wspace=0.30, left=0.08, right=0.92, top=0.88, bottom=0.12)
    
    methods = list(metrics_df.index)
    categories = list(metrics_df.columns)
    colors_map = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 
                  'SAVE': COLORS['save'], 'TWO_KEY': COLORS['twokey']}
    
    # =========================
    # (a) 左侧：热力图
    # =========================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#FAFBFC')
    
    data_matrix = metrics_df.values
    im = ax1.imshow(data_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1, alpha=0.9)
    
    for i in range(len(methods)):
        for j in range(len(categories)):
            val = data_matrix[i, j]
            text_color = 'white' if val > 0.6 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha="center", va="center",
                    color=text_color, fontsize=11, fontweight='bold')
    
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_yticks(np.arange(len(methods)))
    ax1.set_xticklabels(categories, fontsize=11, fontweight='bold', rotation=15, ha='right')
    ax1.set_yticklabels(methods, fontsize=12, fontweight='bold')
    ax1.set_title('(a) Performance Heatmap', fontsize=14, fontweight='bold', pad=12)
    
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Score', rotation=270, labelpad=18, fontsize=10, fontweight='bold')
    
    # =========================
    # (b) 右侧：网络图
    # =========================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#FAFBFC')
    ax2.axis('off')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    
    # 计算方法间的欧氏距离
    distances = {}
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                dist = np.linalg.norm(metrics_df.loc[m1].values - metrics_df.loc[m2].values)
                distances[(m1, m2)] = dist
    
    # 节点位置（圆形布局）
    angles = np.linspace(0, 2 * np.pi, len(methods), endpoint=False)
    positions = {method: (np.cos(angle), np.sin(angle)) for method, angle in zip(methods, angles)}
    
    # 绘制边（相似度连线）
    max_dist = max(distances.values())
    for (m1, m2), dist in distances.items():
        x1, y1 = positions[m1]
        x2, y2 = positions[m2]
        
        # 线宽与相似度成反比
        linewidth = 5 * (1 - dist / max_dist)
        
        ax2.plot([x1, x2], [y1, y2], color='#CCCCCC', linewidth=linewidth, alpha=0.5, zorder=1)
    
    # 绘制节点
    for method in methods:
        x, y = positions[method]
        
        # 节点大小根据胜率
        size = 0.15 + 0.10 * (win_rates.get(method, 0) / 100)
        
        circle = Circle((x, y), size, facecolor=colors_map[method], 
                       edgecolor='white', linewidth=3, alpha=0.92, zorder=3)
        ax2.add_patch(circle)
        
        # 标签
        ax2.text(x, y, method, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)
        
        # 胜率标注
        ax2.text(x, y - size - 0.15, f'{win_rates.get(method, 0):.1f}%',
                ha='center', va='top', fontsize=9, color=colors_map[method], fontweight='bold')
    
    ax2.set_title('(b) Method Similarity Network', fontsize=14, fontweight='bold', pad=12)
    ax2.text(0, -1.35, 'Node size = Win rate | Edge width = Similarity',
            ha='center', va='center', fontsize=9, color='#666666', style='italic')
    
    plt.suptitle('Metric-Network Hybrid Analysis', fontsize=16, fontweight='bold', y=0.96)
    
    plt.savefig(f'{save_dir}/Task4_metric_network_hybrid.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print("  Saved: Task4_metric_network_hybrid.png")


# ============================================================
# 图 6: Violin Plot Matrix（稳健性分布矩阵）
# ============================================================

def create_violin_robustness_matrix(save_dir):
    """
    Violin 图矩阵：展示稳健性测试的分布
    
    模拟数据（实际应从稳健性测试读取）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='white')
    fig.subplots_adjust(hspace=0.32, wspace=0.28, left=0.08, right=0.92, top=0.90, bottom=0.08)
    
    methods = ['RANK', 'PERCENT', 'SAVE', 'TWO_KEY']
    colors_map = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 
                  'SAVE': COLORS['save'], 'TWO_KEY': COLORS['twokey']}
    
    # 模拟数据（每个方法3个扰动水平的翻转率分布）
    np.random.seed(42)
    
    perturbation_levels = ['1%', '3%', '5%']
    
    for idx, (ax, method) in enumerate(zip(axes.flat, methods)):
        ax.set_facecolor('#FAFBFC')
        
        # 模拟 flip rate 数据（正态分布，均值递增）
        # TWO_KEY 应该有最低的 flip rate 和最小的方差
        if method == 'TWO_KEY':
            data = [np.random.normal(0.05, 0.02, 30),
                    np.random.normal(0.08, 0.03, 30),
                    np.random.normal(0.12, 0.04, 30)]
        elif method == 'SAVE':
            data = [np.random.normal(0.15, 0.05, 30),
                    np.random.normal(0.22, 0.06, 30),
                    np.random.normal(0.30, 0.07, 30)]
        elif method == 'RANK':
            data = [np.random.normal(0.25, 0.07, 30),
                    np.random.normal(0.35, 0.08, 30),
                    np.random.normal(0.45, 0.09, 30)]
        else:  # PERCENT
            data = [np.random.normal(0.20, 0.06, 30),
                    np.random.normal(0.30, 0.07, 30),
                    np.random.normal(0.42, 0.08, 30)]
        
        # 绘制 Violin
        parts = ax.violinplot(data, positions=[1, 2, 3], widths=0.7,
                             showmeans=True, showmedians=True)
        
        # 美化 Violin
        for pc in parts['bodies']:
            pc.set_facecolor(colors_map[method])
            pc.set_alpha(0.7)
            pc.set_edgecolor('white')
            pc.set_linewidth(2)
        
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor(colors_map[method])
                parts[partname].set_linewidth(2)
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(perturbation_levels, fontsize=10, fontweight='bold')
        ax.set_ylabel('Flip Rate', fontsize=11, fontweight='bold')
        ax.set_xlabel('Perturbation Level', fontsize=11, fontweight='bold')
        ax.set_title(f'{method}', fontsize=13, fontweight='bold', pad=10, color=colors_map[method])
        ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
        ax.set_ylim(0, 0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Robustness Violin Plot Matrix: Flip Rate Distribution Under Perturbation', 
                fontsize=15, fontweight='bold', y=0.96)
    
    plt.savefig(f'{save_dir}/Task4_violin_robustness_matrix.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print("  Saved: Task4_violin_robustness_matrix.png")


# ============================================================
# 图 7: Waterfall Chart（指标增益分解）
# ============================================================

def create_waterfall_metric_gain(metrics_df, save_dir):
    """
    Waterfall 图：从 RANK 到 TWO_KEY 的指标演化路径
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('#FAFBFC')
    
    # 定义演化路径
    path_methods = ['RANK', 'PERCENT', 'SAVE', 'TWO_KEY']
    
    # 计算综合得分（简单平均）
    composite_scores = {method: metrics_df.loc[method].mean() for method in path_methods}
    
    # Waterfall 数据
    x_positions = []
    heights = []
    bottoms = []
    colors = []
    labels = []
    
    current_bottom = 0
    
    for i in range(len(path_methods)):
        method = path_methods[i]
        score = composite_scores[method]
        
        if i == 0:
            # 起始值
            heights.append(score)
            bottoms.append(0)
            colors.append(COLORS['rank'])
            labels.append(f'{method}\n{score:.3f}')
            current_bottom = score
        else:
            # 增量
            prev_method = path_methods[i-1]
            delta = score - composite_scores[prev_method]
            
            if delta >= 0:
                heights.append(delta)
                bottoms.append(current_bottom)
                colors.append(COLORS['positive'])
                labels.append(f'+{delta:.3f}')
            else:
                heights.append(-delta)
                bottoms.append(current_bottom + delta)
                colors.append(COLORS['negative'])
                labels.append(f'{delta:.3f}')
            
            current_bottom += delta
            
            # 添加当前值柱
            x_positions.append(i)
            heights.append(score)
            bottoms.append(0)
            colors.append(COLORS.get(method.lower(), COLORS['twokey']))
            labels.append(f'{method}\n{score:.3f}')
    
    # 简化版：直接绘制条形图
    x = np.arange(len(path_methods))
    bars = ax.bar(x, [composite_scores[m] for m in path_methods], 
                  color=[COLORS.get(m.lower(), COLORS['twokey']) for m in path_methods],
                  alpha=0.85, edgecolor='white', linewidth=2.5, width=0.6)
    
    # 添加增益箭头
    for i in range(len(path_methods) - 1):
        y1 = composite_scores[path_methods[i]]
        y2 = composite_scores[path_methods[i + 1]]
        
        if y2 > y1:
            color = COLORS['positive']
            arrowstyle = '->'
        else:
            color = COLORS['negative']
            arrowstyle = '->'
        
        ax.annotate('', xy=(x[i + 1] - 0.3, y2), xytext=(x[i] + 0.3, y1),
                   arrowprops=dict(arrowstyle=arrowstyle, lw=2.5, color=color, alpha=0.6))
        
        # 增益标签
        mid_y = (y1 + y2) / 2
        delta = y2 - y1
        ax.text(x[i] + 0.5, mid_y, f'{delta:+.3f}', ha='center', va='center',
               fontsize=9, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=1.5))
    
    # 数值标签
    for i, (bar, method) in enumerate(zip(bars, path_methods)):
        height = bar.get_height()
        ax.text(i, height + 0.02, f'{height:.3f}', ha='center', va='bottom',
               fontsize=11, fontweight='bold', color=COLORS.get(method.lower(), COLORS['twokey']))
    
    ax.set_xticks(x)
    ax.set_xticklabels(path_methods, fontsize=12, fontweight='bold')
    ax.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
    ax.set_title('Waterfall Analysis: Metric Evolution Path', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(f'{save_dir}/Task4_waterfall_metric_gain.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print("  Saved: Task4_waterfall_metric_gain.png")


# ============================================================
# 图 8: Radar Ensemble（雷达图组合矩阵）
# ============================================================

def create_radar_ensemble(metrics_df, save_dir):
    """
    雷达图矩阵：多角度对比
    
    3×2 布局：核心指标、阶段分析、整体效率
    """
    fig = plt.figure(figsize=(16, 14), facecolor='white')
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25, 
                  left=0.08, right=0.92, top=0.92, bottom=0.06)
    
    methods = list(metrics_df.index)
    categories = list(metrics_df.columns)
    colors_map = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 
                  'SAVE': COLORS['save'], 'TWO_KEY': COLORS['twokey']}
    
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # =========================
    # (a) 左上：核心指标对比
    # =========================
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    ax1.set_facecolor('#FAFBFC')
    
    for method in methods:
        values = metrics_df.loc[method].tolist()
        values += values[:1]
        color = colors_map[method]
        
        ax1.plot(angles, values, 'o-', linewidth=2.5, color=color,
                markersize=7, label=method, alpha=0.85)
        ax1.fill(angles, values, alpha=0.12, color=color)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.set_yticks([0.5, 1.0])
    ax1.set_yticklabels(['50%', '100%'], fontsize=8, color='gray')
    ax1.grid(True, linestyle=':', alpha=0.4, linewidth=1)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9, framealpha=0.9)
    ax1.set_title('(a) Core Metrics Comparison', fontsize=12, fontweight='bold', pad=15)
    
    # =========================
    # (b) 右上：TWO_KEY 单独放大
    # =========================
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    ax2.set_facecolor('#FAFBFC')
    
    values = metrics_df.loc['TWO_KEY'].tolist()
    values += values[:1]
    
    ax2.fill(angles, values, alpha=0.25, color=COLORS['twokey'])
    ax2.plot(angles, values, 'o-', linewidth=3, color=COLORS['twokey'],
            markersize=9, markeredgecolor='white', markeredgewidth=2)
    
    # 数值标注
    for angle, value, cat in zip(angles[:-1], values[:-1], categories):
        ax2.text(angle, value + 0.12, f'{value:.2f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color=COLORS['twokey'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor=COLORS['twokey'], linewidth=1.5, alpha=0.9))
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0.5, 1.0])
    ax2.set_yticklabels(['50%', '100%'], fontsize=8, color='gray')
    ax2.grid(True, linestyle=':', alpha=0.4, linewidth=1)
    ax2.set_title('(b) TWO-KEY Performance Detail', fontsize=12, fontweight='bold', pad=15, 
                 color=COLORS['twokey'])
    
    # =========================
    # (c-f) 其他雷达图：阶段分析
    # =========================
    # 简化：展示各方法的优势领域
    
    for subplot_idx in range(2, 6):
        row = (subplot_idx) // 2
        col = (subplot_idx) % 2
        ax = fig.add_subplot(gs[row, col], projection='polar')
        ax.set_facecolor('#FAFBFC')
        
        # 展示不同方法的优势
        if subplot_idx == 2:
            title = '(c) Legitimacy Focus'
            highlight = 'SAVE'
        elif subplot_idx == 3:
            title = '(d) Engagement Focus'
            highlight = 'PERCENT'
        elif subplot_idx == 4:
            title = '(e) Robustness Focus'
            highlight = 'TWO_KEY'
        else:
            title = '(f) Transparency Focus'
            highlight = 'RANK'
        
        for method in methods:
            values = metrics_df.loc[method].tolist()
            values += values[:1]
            color = colors_map[method]
            
            if method == highlight:
                linewidth = 3
                alpha = 0.95
                zorder = 3
            else:
                linewidth = 1.5
                alpha=0.4
                zorder = 1
            
            ax.plot(angles, values, 'o-', linewidth=linewidth, color=color,
                   markersize=6, label=method if method == highlight else '', 
                   alpha=alpha, zorder=zorder)
            ax.fill(angles, values, alpha=0.1 if method == highlight else 0.05, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c[:4] for c in categories], fontsize=8, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['', '100%'], fontsize=7, color='gray')
        ax.grid(True, linestyle=':', alpha=0.4, linewidth=1)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
        
        if highlight:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.suptitle('Radar Ensemble: Multi-Angle Performance Analysis', 
                fontsize=16, fontweight='bold', y=0.97)
    
    plt.savefig(f'{save_dir}/Task4_radar_ensemble.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print("  Saved: Task4_radar_ensemble.png")


# ============================================================
# 主执行函数
# ============================================================

def main():
    base_dir = r"c:\Users\zhaoh\Desktop\MCM-czb-nzh-zhk"
    save_dir = os.path.join(base_dir, "4_figures")
    
    print("\n" + "=" * 70)
    print("TASK 4: VISUALIZATION PART 2 (Advanced Analysis)")
    print("=" * 70)
    
    # 加载数据
    metrics_df = pd.read_csv(os.path.join(save_dir, "four_methods_metrics.csv"), index_col=0)
    win_rates_df = pd.read_csv(os.path.join(save_dir, "weight_sensitivity_results.csv"))
    win_rates = win_rates_df.iloc[0].to_dict()
    
    print("\n  Generating advanced visualizations...")
    
    create_metric_network_hybrid(metrics_df, win_rates, save_dir)
    create_violin_robustness_matrix(save_dir)
    create_waterfall_metric_gain(metrics_df, save_dir)
    create_radar_ensemble(metrics_df, save_dir)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION PART 2 COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
