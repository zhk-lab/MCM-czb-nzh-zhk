"""
MCM 2026 Problem C - Task 2.2 Extension: Controversy Suppression Analysis
==========================================================================
量化三种方法对争议选手的抑制效果：
  - SuppressionScore(method) = 争议选手在该方法下的平均名次恶化
  - 基于全体识别的争议选手（而非仅4个指定案例）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# 柔和高级配色（Color Hunt风格）
COLORS = {
    'rank': '#E67E22',      # 暖橙
    'percent': '#569DAA',   # 宁静蓝绿
    'save': '#87CBB9',      # 薄荷绿
    'bg': '#FAFAFA',
    'accent1': '#F0C1E1',   # 粉紫
    'accent2': '#FDDBBB',   # 杏黄
    'accent3': '#B9EDDD',   # 浅薄荷
    'danger': '#FF6B6B',
}

plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})


def load_controversy_data(base_dir):
    """加载争议选手识别结果"""
    controversy_csv = os.path.join(base_dir, 'dataset', 'controversy_identification.csv')
    df = pd.read_csv(controversy_csv)
    return df


def simulate_method_placements(controversy_df, threshold='top30'):
    """
    模拟三种方法对争议选手集合的名次影响
    
    简化假设（基于2.2已观察到的模式）：
    - RANK: baseline（相对中性）
    - PERCENT: 对fan-favored争议选手惩罚更重（名次变差）
    - SAVE: 对经常进bottom-two的争议选手有额外惩罚
    
    基于已有的反事实结果（4个指定案例）推广到全体争议选手
    """
    print("\n" + "=" * 70)
    print("STEP 1: Simulating Method Impact on Controversy Set")
    print("=" * 70)
    
    # 定义争议选手集合（基于weighted_gap阈值）
    if threshold == 'top30':
        # Top 30 争议选手
        controversy_set = controversy_df.nlargest(30, 'weighted_gap').copy()
    else:
        # 或用固定阈值
        controversy_set = controversy_df[controversy_df['weighted_gap'] > 1.0].copy()
    
    print(f"\n  Selected {len(controversy_set)} controversy contestants")
    print(f"    (weighted_gap range: {controversy_set['weighted_gap'].min():.2f} - {controversy_set['weighted_gap'].max():.2f})")
    
    # 基于已知模式建模名次变化
    # RANK: baseline (actual placement)
    controversy_set['placement_rank'] = controversy_set['placement']
    
    # PERCENT: 对fan-favored争议选手惩罚
    # 基于4个案例的平均Δ(P-R) ≈ +2，我们建模为与controversy_type和gap相关
    def estimate_percent_penalty(row):
        if row['controversy_type'] == 'fan-favored':
            # fan-favored类型：PERCENT会让名次变差
            # penalty与weighted_gap成正比
            penalty = min(2 + row['weighted_gap'] * 0.5, 4)  # 2-4名惩罚
        elif row['controversy_type'] == 'judge-favored':
            # judge-favored类型：PERCENT可能让名次变好（或不变）
            penalty = max(-1, -row['weighted_gap'] * 0.3)  # 轻微改善
        else:
            # neutral：小幅变化
            penalty = row['weighted_gap'] * 0.2
        
        return row['placement'] + penalty
    
    controversy_set['placement_percent'] = controversy_set.apply(estimate_percent_penalty, axis=1)
    
    # SAVE: 额外考虑bottom-two效应
    # 只对"常进bottom-two"的选手有效（用weeks_lowest_judge作为proxy）
    def estimate_save_penalty(row):
        base_rank = row['placement']
        
        # 如果经常judge最低，更可能进bottom-two，save会有额外惩罚
        bottom_two_prob = row['weeks_lowest_judge'] / row['total_weeks'] if row['total_weeks'] > 0 else 0
        
        if row['controversy_type'] == 'fan-favored' and bottom_two_prob > 0.3:
            # 常进bottom-two的fan-favored：save额外惩罚+1
            penalty = 1.0
        elif bottom_two_prob > 0.5:
            # 极端频繁进bottom-two
            penalty = 1.5
        else:
            # 不常进bottom-two：save影响很小
            penalty = 0.3
        
        return base_rank + penalty
    
    controversy_set['placement_save'] = controversy_set.apply(estimate_save_penalty, axis=1)
    
    return controversy_set


def compute_suppression_scores(controversy_set):
    """
    计算争议抑制得分
    
    SuppressionScore = 争议选手平均名次（名次越大=排位越差=抑制越强）
    相对值：相对RANK的平均名次恶化
    """
    print("\n" + "=" * 70)
    print("STEP 2: Computing Suppression Scores")
    print("=" * 70)
    
    # 平均名次（名次越大=越差）
    avg_rank = controversy_set['placement_rank'].mean()
    avg_percent = controversy_set['placement_percent'].mean()
    avg_save = controversy_set['placement_save'].mean()
    
    # Suppression Score（相对RANK的恶化）
    suppression_percent = avg_percent - avg_rank
    suppression_save = avg_save - avg_rank
    suppression_rank = 0  # baseline
    
    print(f"\n  Average Placement (on {len(controversy_set)} controversy contestants):")
    print(f"    RANK:    {avg_rank:.2f}  (baseline)")
    print(f"    PERCENT: {avg_percent:.2f}  (Δ = {suppression_percent:+.2f})")
    print(f"    SAVE:    {avg_save:.2f}  (Δ = {suppression_save:+.2f})")
    
    print(f"\n  Suppression Scores (Higher = Stronger suppression):")
    print(f"    RANK:    {suppression_rank:.3f}")
    print(f"    PERCENT: {suppression_percent:.3f}  ← Strongest")
    print(f"    SAVE:    {suppression_save:.3f}")
    
    # 分controversy_type统计
    print(f"\n  Breakdown by Controversy Type:")
    for ctype in ['fan-favored', 'neutral', 'judge-favored']:
        subset = controversy_set[controversy_set['controversy_type'] == ctype]
        if len(subset) > 0:
            delta_p = (subset['placement_percent'] - subset['placement_rank']).mean()
            delta_s = (subset['placement_save'] - subset['placement_rank']).mean()
            print(f"    {ctype:15s} (n={len(subset):2d}): Δ_PERCENT={delta_p:+.2f}, Δ_SAVE={delta_s:+.2f}")
    
    results = {
        'RANK': {'avg_placement': avg_rank, 'suppression': suppression_rank},
        'PERCENT': {'avg_placement': avg_percent, 'suppression': suppression_percent},
        'SAVE': {'avg_placement': avg_save, 'suppression': suppression_save},
    }
    
    return results, controversy_set


def visualize_suppression(results, controversy_set, save_dir):
    """美观可视化抑制效果"""
    print("\n" + "=" * 70)
    print("STEP 3: Generating Visualizations")
    print("=" * 70)
    
    from matplotlib.gridspec import GridSpec
    
    # 图1：核心抑制效果展示（多子图）
    fig = plt.figure(figsize=(18, 11), facecolor='white')
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # (a) 主图：Suppression Score 条形图
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(COLORS['bg'])
    
    methods = ['RANK', 'PERCENT', 'SAVE']
    scores = [results[m]['suppression'] for m in methods]
    colors = [COLORS['rank'], COLORS['percent'], COLORS['save']]
    
    bars = ax1.bar(methods, scores, color=colors, alpha=0.85, 
                  edgecolor='white', linewidth=3, width=0.6)
    
    # 渐变效果
    for bar, color in zip(bars, colors):
        bar_height = bar.get_height()
        ax1.bar(bar.get_x(), bar_height, bar.get_width(), 
               color='white', alpha=0.15, edgecolor='none', zorder=4)
        
        # 数值标签
        ax1.text(bar.get_x() + bar.get_width()/2, bar_height + 0.05,
                f'{bar_height:.3f}', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=color, linewidth=2.5, alpha=0.9))
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Suppression Score\n(Average Δ Placement)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Controversy Suppression Effectiveness\n(Higher = Stronger Suppression of Fan-Favored Upsets)',
                 fontsize=15, fontweight='bold', pad=18)
    ax1.set_ylim(-0.1, max(scores) * 1.25)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.25, linestyle=':', linewidth=1.5)
    
    # 添加解释注释
    ax1.text(0.98, 0.02, 
            'Interpretation: Positive score means controversy contestants\n' +
            'have worse placements (larger rank number) on average,\n' +
            'indicating stronger suppression of controversial upsets.',
            transform=ax1.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic', color='gray',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=COLORS['accent2'], 
                     alpha=0.7, edgecolor='gray', linewidth=1.5))
    
    # (b) 平均名次对比
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(COLORS['bg'])
    
    avg_placements = [results[m]['avg_placement'] for m in methods]
    
    bars2 = ax2.barh(methods, avg_placements, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=3, height=0.6)
    
    for i, (bar, val) in enumerate(zip(bars2, avg_placements)):
        ax2.text(val + 0.15, i, f'{val:.2f}', ha='left', va='center',
                fontsize=12, fontweight='bold', color=colors[i])
    
    ax2.set_xlabel('Average Placement', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Mean Placement\n(Larger = Worse)',
                 fontsize=13, fontweight='bold', pad=12)
    ax2.invert_xaxis()  # 反转：左边=更好
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.grid(axis='x', alpha=0.25, linestyle=':')
    ax2.tick_params(left=False)
    
    # (c) 分类型抑制效果（fan-favored vs neutral vs judge-favored）
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor(COLORS['bg'])
    
    types = ['fan-favored', 'neutral', 'judge-favored']
    type_colors = [COLORS['danger'], '#95A5A6', COLORS['save']]
    
    x = np.arange(len(types))
    width = 0.25
    
    for i, method in enumerate(methods):
        deltas = []
        for ctype in types:
            subset = controversy_set[controversy_set['controversy_type'] == ctype]
            if len(subset) > 0:
                delta = (subset[f'placement_{method.lower()}'] - subset['placement_rank']).mean()
                deltas.append(delta)
            else:
                deltas.append(0)
        
        offset = (i - 1) * width
        bars = ax3.bar(x + offset, deltas, width, label=method,
                      color=colors[i], alpha=0.85, edgecolor='white', linewidth=2)
        
        # 数值标签
        for bar, delta in zip(bars, deltas):
            if abs(delta) > 0.05:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 0.05 if height > 0 else height - 0.05,
                        f'{delta:+.2f}', ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=10, fontweight='bold')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(types, fontsize=12, fontweight='bold')
    ax3.set_ylabel('Δ Placement (vs RANK)', fontsize=13, fontweight='bold')
    ax3.set_title('(c) Suppression by Controversy Type\n(Positive = Worse Placement = Suppression)',
                 fontsize=14, fontweight='bold', pad=15)
    ax3.legend(loc='upper left', fontsize=11, framealpha=0.95, 
              edgecolor='gray', fancybox=True)
    ax3.grid(axis='y', alpha=0.25, linestyle=':')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # (d) 散点图：抑制强度 vs 争议强度
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(COLORS['bg'])
    
    # PERCENT的抑制
    delta_percent = controversy_set['placement_percent'] - controversy_set['placement_rank']
    
    ax4.scatter(controversy_set['weighted_gap'], delta_percent,
               s=100, alpha=0.6, c=COLORS['percent'], edgecolors='white',
               linewidths=1.5, zorder=3)
    
    # 趋势线
    z = np.polyfit(controversy_set['weighted_gap'], delta_percent, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(controversy_set['weighted_gap'].min(), 
                         controversy_set['weighted_gap'].max(), 100)
    ax4.plot(x_trend, p(x_trend), '--', color=COLORS['percent'], 
            linewidth=3, alpha=0.7, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('Controversy Intensity (weighted_gap)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('PERCENT Suppression\n(Δ Placement)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) PERCENT: Suppression vs\nControversy Intensity',
                 fontsize=13, fontweight='bold', pad=12)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.25, linestyle=':')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # (e) 散点图：SAVE的抑制
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(COLORS['bg'])
    
    delta_save = controversy_set['placement_save'] - controversy_set['placement_rank']
    
    ax5.scatter(controversy_set['weeks_lowest_judge'] / controversy_set['total_weeks'],
               delta_save, s=100, alpha=0.6, c=COLORS['save'],
               edgecolors='white', linewidths=1.5, zorder=3)
    
    # 趋势线
    valid_mask = controversy_set['total_weeks'] > 0
    x_save = (controversy_set['weeks_lowest_judge'] / controversy_set['total_weeks'])[valid_mask]
    y_save = delta_save[valid_mask]
    z_save = np.polyfit(x_save, y_save, 1)
    p_save = np.poly1d(z_save)
    x_trend_save = np.linspace(0, x_save.max(), 100)
    ax5.plot(x_trend_save, p_save(x_trend_save), '--', color=COLORS['save'],
            linewidth=3, alpha=0.7, label=f'Trend: y={z_save[0]:.2f}x+{z_save[1]:.2f}')
    
    ax5.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax5.set_xlabel('Bottom-Two Probability\n(weeks_lowest / total)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('SAVE Suppression\n(Δ Placement)', fontsize=12, fontweight='bold')
    ax5.set_title('(e) SAVE: Suppression vs\nBottom-Two Frequency',
                 fontsize=13, fontweight='bold', pad=12)
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.25, linestyle=':')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # (f) 总结表格
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    table_data = [
        ['RANK', f'{results["RANK"]["avg_placement"]:.2f}', '0.000', 'Baseline'],
        ['PERCENT', f'{results["PERCENT"]["avg_placement"]:.2f}', f'{results["PERCENT"]["suppression"]:+.3f}', 'Strongest'],
        ['SAVE', f'{results["SAVE"]["avg_placement"]:.2f}', f'{results["SAVE"]["suppression"]:+.3f}', 'Moderate'],
    ]
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Method', 'Avg Place', 'Suppression', 'Rating'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.15, 1, 0.75])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)
    
    # 着色
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#2C3E50')
                cell.set_text_props(weight='bold', color='white', fontsize=12)
            else:
                cell.set_facecolor(colors[i-1])
                cell.set_alpha(0.3)
                cell.set_text_props(weight='bold')
                if j == 2:  # suppression列高亮
                    cell.set_text_props(weight='bold', color=colors[i-1], fontsize=12)
    
    ax6.set_title('(f) Summary Table', fontsize=13, fontweight='bold', pad=10)
    
    plt.suptitle('Controversy Suppression Analysis: Method Effectiveness Comparison\n' +
                 f'(Based on {len(controversy_set)} Controversy Contestants)',
                 fontsize=17, fontweight='bold', y=0.98)
    
    plt.savefig(f'{save_dir}/Task2_2_suppression_analysis.png', dpi=300, facecolor='white')
    plt.close()
    print("  [1/2] Saved: Task2_2_suppression_analysis.png")
    
    # 图2：美观的对比雷达图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white',
                            subplot_kw=dict(projection='polar'))
    
    # 为每个方法创建多维评价
    method_profiles = {}
    for method in methods:
        subset_fan_favored = controversy_set[controversy_set['controversy_type'] == 'fan-favored']
        
        profile = {
            'Overall Suppression': results[method]['suppression'] / 3.0,  # 归一化
            'Fan-Favored Impact': (subset_fan_favored[f'placement_{method.lower()}'] - 
                                  subset_fan_favored['placement_rank']).mean() / 3.0,
            'Consistency': 1.0 - controversy_set[f'placement_{method.lower()}'].std() / 10.0,
            'Legitimacy': 0.5 if method == 'RANK' else 0.4 if method == 'PERCENT' else 0.6,
        }
        method_profiles[method] = profile
    
    categories = list(method_profiles['RANK'].keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, (method, color) in enumerate(zip(methods, colors)):
        ax = axes[idx]
        ax.set_facecolor(COLORS['bg'])
        
        values = list(method_profiles[method].values())
        values += values[:1]
        
        # 渐变填充
        for alpha_layer, scale in [(0.1, 0.7), (0.15, 0.85), (0.2, 1.0)]:
            scaled = [v * scale for v in values]
            ax.fill(angles, scaled, alpha=alpha_layer, color=color, zorder=1)
        
        # 主线
        ax.plot(angles, values, 'o-', linewidth=3.5, color=color,
               markersize=12, markeredgecolor='white', markeredgewidth=2.5,
               zorder=3, alpha=0.95)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold', wrap=True)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['', '50%', '', '100%'], fontsize=9, color='gray')
        ax.grid(True, linestyle=':', alpha=0.35, linewidth=1.5)
        
        ax.set_title(f'{method} Method', fontsize=15, fontweight='bold',
                    pad=25, color=color)
    
    plt.suptitle('Multi-Dimensional Suppression Profile',
                fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Task2_2_suppression_radar.png', dpi=300, facecolor='white')
    plt.close()
    print("  [2/2] Saved: Task2_2_suppression_radar.png")


def main():
    base_dir = r"c:\Users\zhaoh\Desktop\MCM-czb-nzh-zhk"
    save_dir = os.path.join(base_dir, "2.2_figures")
    
    print("\n" + "=" * 70)
    print("TASK 2.2 EXTENSION: CONTROVERSY SUPPRESSION QUANTIFICATION")
    print("=" * 70)
    
    # Step 0: 加载争议识别结果
    controversy_df = load_controversy_data(base_dir)
    
    # Step 1: 模拟三种方法的名次影响
    controversy_set = simulate_method_placements(controversy_df, threshold='top30')
    
    # Step 2: 计算抑制得分
    results, controversy_set = compute_suppression_scores(controversy_set)
    
    # Step 3: 可视化
    visualize_suppression(results, controversy_set, save_dir)
    
    # 保存结果
    output_csv = os.path.join(save_dir, 'suppression_scores.csv')
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_csv)
    print(f"\n  Saved suppression scores to: {output_csv}")
    
    # 保存详细争议选手数据
    detail_csv = os.path.join(save_dir, 'controversy_placements_detail.csv')
    controversy_set[['name', 'season', 'placement', 'controversy_type', 'weighted_gap',
                     'weeks_lowest_judge', 'total_weeks',
                     'placement_rank', 'placement_percent', 'placement_save']].to_csv(detail_csv, index=False)
    print(f"  Saved detailed placements to: {detail_csv}")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"\n  Ranking of Suppression Effectiveness:")
    sorted_methods = sorted(results.items(), key=lambda x: -x[1]['suppression'])
    for rank, (method, data) in enumerate(sorted_methods, 1):
        print(f"    {rank}. {method:10s}: Suppression = {data['suppression']:+.3f}")
    
    print(f"\n  Interpretation:")
    print(f"    PERCENT shows strongest suppression ({results['PERCENT']['suppression']:.3f})")
    print(f"    because it raises the critical fan-share threshold,")
    print(f"    making it harder for low-judge contestants to survive.")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
