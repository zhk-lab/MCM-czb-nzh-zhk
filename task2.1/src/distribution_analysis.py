"""
MCM 2026 Problem C - Distribution Analysis
==========================================
验证假设：fan vote share 分布更"尖峰厚尾"，judge percent 更集中

统计指标：
- 标准差（Std）：离散程度
- 峰度（Kurtosis）：尖峰厚尾程度
- 基尼系数（Gini）：不平等程度
- 90/10分位数比：极端差距
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 柔和配色
COLORS = {
    'fan': '#E67E22',       # 暖橙
    'judge': '#569DAA',     # 宁静蓝绿
    'bg': '#FAFAFA',
    'accent': '#87CBB9',
}

plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})


def load_judge_scores(csv_path):
    """从原始数据提取每周judge分数"""
    df = pd.read_csv(csv_path)
    
    judge_data = []
    
    for _, row in df.iterrows():
        name = row['celebrity_name']
        season = row['season']
        
        for week in range(1, 12):
            week_scores = []
            for judge in range(1, 5):
                col = f'week{week}_judge{judge}_score'
                if col in row.index and pd.notna(row[col]) and row[col] != 0:
                    week_scores.append(row[col])
            
            if week_scores:
                judge_data.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'judge_total_score': sum(week_scores)
                })
    
    return pd.DataFrame(judge_data)


def gini_coefficient(values):
    """计算基尼系数（不平等程度）"""
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    
    # Gini = (2 * sum of (i * x_i)) / (n * sum(x_i)) - (n+1)/n
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_values))) / (n * np.sum(sorted_values)) - (n+1)/n
    return gini


def compute_weekly_statistics(df_fan, df_judge):
    """计算每周的分布统计"""
    print("\n" + "=" * 70)
    print("Computing Weekly Distribution Statistics")
    print("=" * 70)
    
    fan_stats_list = []
    judge_stats_list = []
    
    # 按赛季/周分组
    for (season, week), group_fan in df_fan.groupby(['season', 'week']):
        # 获取对应周的judge数据
        group_judge = df_judge[(df_judge['season'] == season) & (df_judge['week'] == week)]
        
        if len(group_fan) < 3 or len(group_judge) < 3:
            continue
        
        # Fan shares（已经是归一化的）
        fan_shares = group_fan['fan_vote_share'].values
        
        # Judge percent（需要归一化）
        judge_scores = group_judge['judge_total_score'].values
        judge_total = judge_scores.sum()
        judge_percent = judge_scores / judge_total if judge_total > 0 else judge_scores / len(judge_scores)
        
        # 统计指标
        fan_stats = {
            'season': season,
            'week': week,
            'n': len(fan_shares),
            'mean': fan_shares.mean(),
            'std': fan_shares.std(),
            'cv': fan_shares.std() / fan_shares.mean() if fan_shares.mean() > 0 else 0,
            'kurtosis': stats.kurtosis(fan_shares),
            'skewness': stats.skew(fan_shares),
            'gini': gini_coefficient(fan_shares),
            'q90_q10_ratio': np.percentile(fan_shares, 90) / np.percentile(fan_shares, 10) if np.percentile(fan_shares, 10) > 0 else np.nan,
            'max_min_ratio': fan_shares.max() / fan_shares.min() if fan_shares.min() > 0 else np.nan,
        }
        fan_stats_list.append(fan_stats)
        
        judge_stats = {
            'season': season,
            'week': week,
            'n': len(judge_percent),
            'mean': judge_percent.mean(),
            'std': judge_percent.std(),
            'cv': judge_percent.std() / judge_percent.mean() if judge_percent.mean() > 0 else 0,
            'kurtosis': stats.kurtosis(judge_percent),
            'skewness': stats.skew(judge_percent),
            'gini': gini_coefficient(judge_percent),
            'q90_q10_ratio': np.percentile(judge_percent, 90) / np.percentile(judge_percent, 10) if np.percentile(judge_percent, 10) > 0 else np.nan,
            'max_min_ratio': judge_percent.max() / judge_percent.min() if judge_percent.min() > 0 else np.nan,
        }
        judge_stats_list.append(judge_stats)
    
    fan_stats_df = pd.DataFrame(fan_stats_list)
    judge_stats_df = pd.DataFrame(judge_stats_list)
    
    print(f"  Analyzed {len(fan_stats_df)} weeks")
    
    return fan_stats_df, judge_stats_df


def statistical_comparison(fan_stats, judge_stats, table_dir):
    """统计比较与可视化"""
    print("\n" + "=" * 70)
    print("Statistical Comparison: Fan vs Judge Distributions")
    print("=" * 70)
    
    # 汇总统计
    metrics = ['std', 'cv', 'kurtosis', 'gini', 'q90_q10_ratio']
    
    print("\n  Average Statistics (across all weeks):\n")
    print(f"  {'Metric':<20s} {'Fan Vote':<15s} {'Judge Percent':<15s} {'Ratio (Fan/Judge)':<20s}")
    print("  " + "-" * 70)
    
    for metric in metrics:
        fan_mean = fan_stats[metric].mean()
        judge_mean = judge_stats[metric].mean()
        ratio = fan_mean / judge_mean if judge_mean > 0 else np.nan
        
        print(f"  {metric:<20s} {fan_mean:<15.4f} {judge_mean:<15.4f} {ratio:<20.2f}")
    
    # 统计检验（配对t检验）
    print("\n  Statistical Tests (Paired t-test):\n")
    
    for metric in metrics:
        # 只取公共周（配对）
        common_weeks = pd.merge(fan_stats[['season', 'week', metric]], 
                               judge_stats[['season', 'week', metric]], 
                               on=['season', 'week'], suffixes=('_fan', '_judge'))
        
        t_stat, p_value = stats.ttest_rel(common_weeks[f'{metric}_fan'], 
                                          common_weeks[f'{metric}_judge'])
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"    {metric:<20s}: t={t_stat:7.3f}, p={p_value:.4e} {significance}")
    
    # 保存汇总表
    summary = pd.DataFrame({
        'metric': metrics,
        'fan_mean': [fan_stats[m].mean() for m in metrics],
        'judge_mean': [judge_stats[m].mean() for m in metrics],
        'fan_judge_ratio': [fan_stats[m].mean() / judge_stats[m].mean() if judge_stats[m].mean() > 0 else np.nan for m in metrics],
    })
    summary.to_csv(f'{table_dir}/distribution_comparison.csv', index=False)
    print(f"\n  Saved summary to: distribution_comparison.csv")


def visualize_distributions(fan_stats, judge_stats, save_dir):
    """可视化分布特征对比"""
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    from matplotlib.gridspec import GridSpec
    
    # 图1：核心指标对比（4个箱线图）
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    metrics = [
        ('std', 'Standard Deviation (Dispersion)'),
        ('kurtosis', 'Kurtosis (Heavy-tailedness)'),
        ('gini', 'Gini Coefficient (Inequality)'),
        ('q90_q10_ratio', '90th/10th Percentile Ratio'),
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor(COLORS['bg'])
        
        # 数据
        fan_vals = fan_stats[metric].dropna()
        judge_vals = judge_stats[metric].dropna()
        
        # 箱线图
        bp = ax.boxplot([fan_vals, judge_vals], 
                        labels=['Fan Vote Share', 'Judge Percent'],
                        patch_artist=True, widths=0.6,
                        boxprops=dict(linewidth=2),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                        medianprops=dict(linewidth=3, color='darkred'))
        
        # 着色
        bp['boxes'][0].set_facecolor(COLORS['fan'])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(COLORS['judge'])
        bp['boxes'][1].set_alpha(0.7)
        
        # 添加均值标注
        fan_mean = fan_vals.mean()
        judge_mean = judge_vals.mean()
        
        ax.plot(1, fan_mean, 'D', markersize=12, color='darkred', 
               markeredgecolor='white', markeredgewidth=2, zorder=3, label='Mean')
        ax.plot(2, judge_mean, 'D', markersize=12, color='darkred',
               markeredgecolor='white', markeredgewidth=2, zorder=3)
        
        # 标注均值
        ax.text(1, fan_mean, f'{fan_mean:.3f}', ha='right', va='center',
               fontsize=10, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        ax.text(2, judge_mean, f'{judge_mean:.3f}', ha='left', va='center',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # 标题和标签
        ratio = fan_mean / judge_mean if judge_mean > 0 else np.nan
        ax.set_title(f'({chr(97+idx)}) {title}\n(Fan/Judge Ratio: {ratio:.2f})',
                    fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    plt.suptitle('Distribution Characteristics: Fan Vote vs Judge Percent\n' +
                 '(Evidence for "Fan votes are more dispersed/heavy-tailed")',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(f'{save_dir}/Task2_1_distribution_boxplots.png', dpi=300, facecolor='white')
    plt.close()
    print("  [1/3] Saved: Task2_1_distribution_boxplots.png")
    
    # 图2：分布形状对比（直方图叠加）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    
    # (a) 标准差分布
    ax = axes[0, 0]
    ax.set_facecolor(COLORS['bg'])
    ax.hist(fan_stats['std'], bins=30, alpha=0.7, color=COLORS['fan'], 
           edgecolor='white', linewidth=1.5, label='Fan Vote')
    ax.hist(judge_stats['std'], bins=30, alpha=0.7, color=COLORS['judge'],
           edgecolor='white', linewidth=1.5, label='Judge Percent')
    ax.axvline(fan_stats['std'].mean(), color=COLORS['fan'], 
              linestyle='--', linewidth=3, alpha=0.8)
    ax.axvline(judge_stats['std'].mean(), color=COLORS['judge'],
              linestyle='--', linewidth=3, alpha=0.8)
    ax.set_xlabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('(a) Std Distribution (Higher = More Dispersed)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # (b) 峰度分布
    ax = axes[0, 1]
    ax.set_facecolor(COLORS['bg'])
    ax.hist(fan_stats['kurtosis'], bins=30, alpha=0.7, color=COLORS['fan'],
           edgecolor='white', linewidth=1.5, label='Fan Vote')
    ax.hist(judge_stats['kurtosis'], bins=30, alpha=0.7, color=COLORS['judge'],
           edgecolor='white', linewidth=1.5, label='Judge Percent')
    ax.axvline(fan_stats['kurtosis'].mean(), color=COLORS['fan'],
              linestyle='--', linewidth=3, alpha=0.8)
    ax.axvline(judge_stats['kurtosis'].mean(), color=COLORS['judge'],
              linestyle='--', linewidth=3, alpha=0.8)
    ax.set_xlabel('Kurtosis', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('(b) Kurtosis Distribution (Higher = More Heavy-tailed)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # (c) Gini系数分布
    ax = axes[1, 0]
    ax.set_facecolor(COLORS['bg'])
    ax.hist(fan_stats['gini'], bins=30, alpha=0.7, color=COLORS['fan'],
           edgecolor='white', linewidth=1.5, label='Fan Vote')
    ax.hist(judge_stats['gini'], bins=30, alpha=0.7, color=COLORS['judge'],
           edgecolor='white', linewidth=1.5, label='Judge Percent')
    ax.axvline(fan_stats['gini'].mean(), color=COLORS['fan'],
              linestyle='--', linewidth=3, alpha=0.8)
    ax.axvline(judge_stats['gini'].mean(), color=COLORS['judge'],
              linestyle='--', linewidth=3, alpha=0.8)
    ax.set_xlabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('(c) Gini Distribution (Higher = More Unequal)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # (d) 90/10分位数比
    ax = axes[1, 1]
    ax.set_facecolor(COLORS['bg'])
    fan_ratio = fan_stats['q90_q10_ratio'].dropna()
    judge_ratio = judge_stats['q90_q10_ratio'].dropna()
    ax.hist(fan_ratio, bins=30, alpha=0.7, color=COLORS['fan'],
           edgecolor='white', linewidth=1.5, label='Fan Vote')
    ax.hist(judge_ratio, bins=30, alpha=0.7, color=COLORS['judge'],
           edgecolor='white', linewidth=1.5, label='Judge Percent')
    ax.axvline(fan_ratio.mean(), color=COLORS['fan'],
              linestyle='--', linewidth=3, alpha=0.8)
    ax.axvline(judge_ratio.mean(), color=COLORS['judge'],
              linestyle='--', linewidth=3, alpha=0.8)
    ax.set_xlabel('90th/10th Percentile Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('(d) Extreme Gap Ratio (Higher = Larger Top-Bottom Gap)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.suptitle('Distribution Metrics Comparison: Fan Vote Share vs Judge Percent',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Task2_1_distribution_histograms.png', dpi=300, facecolor='white')
    plt.close()
    print("  [2/3] Saved: Task2_1_distribution_histograms.png")
    
    # 图3：散点图（每周一个点）
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    
    # (a) Std vs Gini
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['bg'])
    
    ax1.scatter(fan_stats['std'], fan_stats['gini'], 
               s=80, alpha=0.6, c=COLORS['fan'], edgecolors='white',
               linewidths=1.5, label='Fan Vote', zorder=3)
    ax1.scatter(judge_stats['std'], judge_stats['gini'],
               s=80, alpha=0.6, c=COLORS['judge'], edgecolors='white',
               linewidths=1.5, label='Judge Percent', zorder=3)
    
    # 添加均值点
    ax1.plot(fan_stats['std'].mean(), fan_stats['gini'].mean(), 
            'D', markersize=15, color=COLORS['fan'], 
            markeredgecolor='white', markeredgewidth=3, zorder=4,
            label='Fan Mean')
    ax1.plot(judge_stats['std'].mean(), judge_stats['gini'].mean(),
            'D', markersize=15, color=COLORS['judge'],
            markeredgecolor='white', markeredgewidth=3, zorder=4,
            label='Judge Mean')
    
    ax1.set_xlabel('Standard Deviation', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Gini Coefficient', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Dispersion vs Inequality: Fan More Dispersed & Unequal',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # (b) CV vs Kurtosis
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['bg'])
    
    ax2.scatter(fan_stats['cv'], fan_stats['kurtosis'],
               s=80, alpha=0.6, c=COLORS['fan'], edgecolors='white',
               linewidths=1.5, label='Fan Vote', zorder=3)
    ax2.scatter(judge_stats['cv'], judge_stats['kurtosis'],
               s=80, alpha=0.6, c=COLORS['judge'], edgecolors='white',
               linewidths=1.5, label='Judge Percent', zorder=3)
    
    ax2.plot(fan_stats['cv'].mean(), fan_stats['kurtosis'].mean(),
            'D', markersize=15, color=COLORS['fan'],
            markeredgecolor='white', markeredgewidth=3, zorder=4,
            label='Fan Mean')
    ax2.plot(judge_stats['cv'].mean(), judge_stats['kurtosis'].mean(),
            'D', markersize=15, color=COLORS['judge'],
            markeredgecolor='white', markeredgewidth=3, zorder=4,
            label='Judge Mean')
    
    ax2.set_xlabel('Coefficient of Variation', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Kurtosis', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Relative Dispersion vs Heavy-tailedness: Fan Higher on Both',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(alpha=0.3, linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle('Weekly Distribution Scatter: Fan Vote More Dispersed & Heavy-tailed',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Task2_1_distribution_scatter.png', dpi=300, facecolor='white')
    plt.close()
    print("  [3/3] Saved: Task2_1_distribution_scatter.png")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_csv = str(repo_root / "2026_MCM_Problem_C_Data.csv")
    fan_csv = str(repo_root / "task1" / "table" / "fan_vote_shares.csv")
    figure_dir = repo_root / "task2.1" / "figure"
    table_dir = repo_root / "task2.1" / "table"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("DISTRIBUTION ANALYSIS: FAN VS JUDGE")
    print("=" * 70)
    
    # 加载数据
    print("\n  Loading data...")
    df_fan = pd.read_csv(fan_csv)
    df_judge = load_judge_scores(data_csv)
    
    # 计算统计
    fan_stats, judge_stats = compute_weekly_statistics(df_fan, df_judge)
    
    # 统计比较
    statistical_comparison(fan_stats, judge_stats, str(table_dir))
    
    # 可视化
    visualize_distributions(fan_stats, judge_stats, str(figure_dir))
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
