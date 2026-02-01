"""
MCM 2026 Problem C - Task 2.1: Voting Method Comparison
========================================================
比较 RANK 和 PERCENT 两种合并方法的差异及对 fan votes 的偏向程度
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fan_vote_estimation_v2 import (
    load_and_preprocess_data,
    sample_season_feasible_region_v2,
    build_hmm_and_find_map_path_v2,
    evaluate_consistency_v2
)

# O奖级配色
COLORS = {
    'primary': '#2E86AB',
    'accent': '#F18F01',
    'success': '#4CAF50',
    'warning': '#FF6B6B',
    'rank': '#E67E22',
    'percent': '#3498DB',
    'bg': '#FAFAFA',
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


def collect_fan_vote_shares(seasons_data, valid_seasons, output_csv):
    """
    步骤1: 运行主模型，收集所有 season/week/person 的 fan vote shares
    保存到 CSV 表格中
    """
    print("\n" + "=" * 60)
    print("STEP 1: Collecting Fan Vote Shares")
    print("=" * 60)
    
    all_records = []
    
    for season in sorted(valid_seasons):
        print(f"\n  Processing Season {season}...", end=" ")
        
        season_data = seasons_data[season]
        method = 'rank' if (season <= 2 or season >= 28) else 'percent'
        
        # 使用快速参数（足够用于方法比较）
        n_samples = 800 if method == 'rank' else 500
        n_states = 60 if method == 'rank' else 40
        
        # 采样 + HMM
        sampled_data = sample_season_feasible_region_v2(season_data, season, n_samples=n_samples)
        map_result = build_hmm_and_find_map_path_v2(sampled_data, n_states=n_states)
        
        # 提取每周每人的 share
        path = map_result['path']
        n_weeks = len(path)
        
        for week_info in path:
            fan_shares = week_info['fan_shares']
            active = week_info['active']
            
            for i, contestant in enumerate(active):
                all_records.append({
                    'season': season,
                    'week': week_info['week'],
                    'celebrity_name': contestant,
                    'fan_vote_share': fan_shares[i],
                    'method': method
                })
        
        print(f"Done ({n_weeks} weeks)")
    
    # 保存到 CSV
    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)
    print(f"\n  Saved fan vote shares to: {output_csv}")
    print(f"  Total records: {len(df)}")
    
    return df


def load_judge_scores(csv_path, seasons_data):
    """加载评委分数数据"""
    judge_scores = {}
    
    for season, data in seasons_data.items():
        judge_scores[season] = {}
        weeks = data['weeks']
        
        for week_num, week_info in weeks.items():
            if week_info['eliminated'] is None:
                continue
            
            scores = week_info['scores']
            if scores:
                judge_scores[season][week_num] = scores
    
    return judge_scores


def compute_rankings(scores_dict, method='descending'):
    """
    计算排名（处理并列，使用 mid-rank）
    method: 'descending' (高分排前) 或 'ascending' (低分排前)
    """
    values = np.array(list(scores_dict.values()))
    names = list(scores_dict.keys())
    
    if method == 'descending':
        # 高分排前（fan votes, judge scores）
        sorted_indices = np.argsort(-values)
    else:
        # 低分排前（combined score for elimination）
        sorted_indices = np.argsort(values)
    
    ranks = np.zeros(len(values))
    
    i = 0
    while i < len(values):
        # 找到所有与当前值相同的索引
        current_value = values[sorted_indices[i]]
        tied_indices = []
        j = i
        
        while j < len(values) and np.abs(values[sorted_indices[j]] - current_value) < 1e-9:
            tied_indices.append(sorted_indices[j])
            j += 1
        
        # 分配平均排名
        avg_rank = (i + 1 + j) / 2
        for idx in tied_indices:
            ranks[idx] = avg_rank
        
        i = j
    
    return {names[i]: ranks[i] for i in range(len(names))}


def apply_rank_method(judge_scores, fan_shares):
    """应用 RANK 合并方法"""
    judge_ranks = compute_rankings(judge_scores, 'descending')
    fan_ranks = compute_rankings(fan_shares, 'descending')
    
    combined = {}
    for name in judge_ranks:
        if name in fan_ranks:
            combined[name] = judge_ranks[name] + fan_ranks[name]
    
    # 排名和最大者被淘汰（rank 越大越差）
    final_ranks = compute_rankings(combined, 'ascending')
    return final_ranks, combined


def apply_percent_method(judge_scores, fan_shares):
    """应用 PERCENT 合并方法"""
    total_judge = sum(judge_scores.values())
    total_fan = sum(fan_shares.values())
    
    judge_pct = {name: score / total_judge for name, score in judge_scores.items()}
    fan_pct = {name: share / total_fan for name, share in fan_shares.items()}
    
    combined = {}
    for name in judge_pct:
        if name in fan_pct:
            combined[name] = judge_pct[name] + fan_pct[name]
    
    # 百分比和最小者被淘汰（percent 越小越差）
    final_ranks = compute_rankings(combined, 'descending')
    return final_ranks, combined


def kendall_distance(rank1, rank2):
    """
    计算 Kendall tau 距离（处理并列）
    返回: (tau, distance)
    """
    common_names = set(rank1.keys()) & set(rank2.keys())
    if len(common_names) < 2:
        return 0, 1
    
    r1 = [rank1[name] for name in common_names]
    r2 = [rank2[name] for name in common_names]
    
    tau, _ = kendalltau(r1, r2)
    
    # 转换为距离: distance = (1 - tau) / 2 ∈ [0, 1]
    distance = (1 - tau) / 2
    
    return tau, distance


def footrule_distance(rank1, rank2):
    """计算 Spearman footrule 距离（归一化）"""
    common_names = set(rank1.keys()) & set(rank2.keys())
    n = len(common_names)
    
    if n < 2:
        return 0
    
    total_diff = sum(abs(rank1[name] - rank2[name]) for name in common_names)
    
    # 归一化: 最大可能距离是 n^2/2 (for even n)
    max_distance = n * n / 2 if n % 2 == 0 else (n * n - 1) / 2
    
    return total_diff / max_distance if max_distance > 0 else 0


def analyze_season(season, judge_scores_season, fan_shares_df):
    """分析单个赛季的所有周"""
    season_fan = fan_shares_df[fan_shares_df['season'] == season]
    
    results = []
    
    for week in sorted(season_fan['week'].unique()):
        if week not in judge_scores_season:
            continue
        
        week_fan = season_fan[season_fan['week'] == week]
        
        judge_scores = judge_scores_season[week]
        fan_shares = dict(zip(week_fan['celebrity_name'], week_fan['fan_vote_share']))
        
        # 确保两者有共同选手
        common = set(judge_scores.keys()) & set(fan_shares.keys())
        if len(common) < 2:
            continue
        
        judge_scores = {k: v for k, v in judge_scores.items() if k in common}
        fan_shares = {k: v for k, v in fan_shares.items() if k in common}
        
        # 四种排序
        R_judge = compute_rankings(judge_scores, 'descending')
        R_fan = compute_rankings(fan_shares, 'descending')
        R_rank, _ = apply_rank_method(judge_scores, fan_shares)
        R_percent, _ = apply_percent_method(judge_scores, fan_shares)
        
        # 计算距离（使用 Kendall）
        _, d_rank_percent = kendall_distance(R_rank, R_percent)
        _, d_rank_judge = kendall_distance(R_rank, R_judge)
        _, d_rank_fan = kendall_distance(R_rank, R_fan)
        _, d_percent_judge = kendall_distance(R_percent, R_judge)
        _, d_percent_fan = kendall_distance(R_percent, R_fan)
        
        # FFI
        ffi_rank = d_rank_judge - d_rank_fan
        ffi_percent = d_percent_judge - d_percent_fan
        
        # 淘汰者
        # NOTE:
        # `compute_rankings()` assigns rank=1 to the best (highest score / highest combined),
        # and larger ranks to worse contestants. Therefore the eliminated contestant is the
        # one with the *largest* rank value, not the smallest.
        elim_rank = max(R_rank, key=R_rank.get)
        elim_percent = max(R_percent, key=R_percent.get)
        
        results.append({
            'season': season,
            'week': week,
            'd_rank_percent': d_rank_percent,
            'ffi_rank': ffi_rank,
            'ffi_percent': ffi_percent,
            'elim_agree': elim_rank == elim_percent,
            'elim_rank': elim_rank,
            'elim_percent': elim_percent
        })
    
    return results


def generate_visualizations(analysis_df, save_dir):
    """生成多种 O 奖级别的可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STEP 3: Generating Visualizations")
    print("=" * 60)
    
    # 图1: FFI 热力图（跨赛季）
    print("\n  [1/5] FFI Heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    
    pivot_rank = analysis_df.pivot_table(values='ffi_rank', index='week', columns='season', aggfunc='mean')
    pivot_percent = analysis_df.pivot_table(values='ffi_percent', index='week', columns='season', aggfunc='mean')
    
    im1 = axes[0].imshow(pivot_rank.values, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[0].set_title('(a) FFI - RANK Method', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Season', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Week', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(pivot_rank.columns)))
    axes[0].set_xticklabels(pivot_rank.columns, fontsize=8)
    axes[0].set_yticks(range(len(pivot_rank.index)))
    axes[0].set_yticklabels(pivot_rank.index)
    plt.colorbar(im1, ax=axes[0], label='FFI (>0: favor fans)')
    
    im2 = axes[1].imshow(pivot_percent.values, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[1].set_title('(b) FFI - PERCENT Method', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Season', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Week', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(pivot_percent.columns)))
    axes[1].set_xticklabels(pivot_percent.columns, fontsize=8)
    axes[1].set_yticks(range(len(pivot_percent.index)))
    axes[1].set_yticklabels(pivot_percent.index)
    plt.colorbar(im2, ax=axes[1], label='FFI (>0: favor fans)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Task2_1_FFI_heatmap.png', dpi=300, facecolor='white')
    plt.close()
    
    # 图2: FFI 对比箱线图
    print("  [2/5] FFI Comparison Boxplot...")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor(COLORS['bg'])
    
    data_to_plot = [analysis_df['ffi_rank'].dropna(), analysis_df['ffi_percent'].dropna()]
    bp = ax.boxplot(data_to_plot, labels=['RANK Method', 'PERCENT Method'],
                    patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], [COLORS['rank'], COLORS['percent']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('FFI (Fan-Favor Index)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Fan-Favor Index Between Two Methods', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Task2_1_FFI_boxplot.png', dpi=300, facecolor='white')
    plt.close()
    
    # 图3: 方法差异度分布
    print("  [3/5] Method Disagreement Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    
    # 左: Kendall 距离分布
    axes[0].set_facecolor(COLORS['bg'])
    axes[0].hist(analysis_df['d_rank_percent'].dropna(), bins=30, 
                 color=COLORS['primary'], alpha=0.7, edgecolor='white')
    axes[0].axvline(x=analysis_df['d_rank_percent'].mean(), color=COLORS['warning'], 
                    linestyle='--', linewidth=2, label=f'Mean: {analysis_df["d_rank_percent"].mean():.3f}')
    axes[0].set_xlabel('Kendall Distance', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('(a) Distance Between RANK and PERCENT Rankings', fontsize=13, fontweight='bold', pad=10)
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # 右: 淘汰一致性
    axes[1].set_facecolor(COLORS['bg'])
    agree_rate = analysis_df['elim_agree'].mean() * 100
    disagree_rate = 100 - agree_rate
    
    bars = axes[1].bar(['Agree', 'Disagree'], [agree_rate, disagree_rate],
                       color=[COLORS['success'], COLORS['warning']], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('(b) Elimination Agreement Between Methods', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_ylim(0, 110)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Task2_1_disagreement.png', dpi=300, facecolor='white')
    plt.close()
    
    # 图4: FFI 散点图（RANK vs PERCENT）
    print("  [4/5] FFI Scatter Plot...")
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_facecolor(COLORS['bg'])
    
    scatter = ax.scatter(analysis_df['ffi_rank'], analysis_df['ffi_percent'],
                        c=analysis_df['season'], cmap='viridis', alpha=0.6, s=50, edgecolors='white')
    
    ax.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5, linewidth=1.5, label='y=x')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('FFI (RANK Method)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FFI (PERCENT Method)', fontsize=12, fontweight='bold')
    ax.set_title('FFI Comparison: RANK vs PERCENT Methods', fontsize=14, fontweight='bold', pad=15)
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Season', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Task2_1_FFI_scatter.png', dpi=300, facecolor='white')
    plt.close()
    
    # 图5: 季度汇总统计
    print("  [5/5] Season Summary Statistics...")
    season_stats = analysis_df.groupby('season').agg({
        'ffi_rank': 'mean',
        'ffi_percent': 'mean',
        'd_rank_percent': 'mean',
        'elim_agree': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    
    # (a) 平均 FFI by season
    axes[0, 0].set_facecolor(COLORS['bg'])
    axes[0, 0].plot(season_stats['season'], season_stats['ffi_rank'], 
                    'o-', color=COLORS['rank'], label='RANK', linewidth=2, markersize=6)
    axes[0, 0].plot(season_stats['season'], season_stats['ffi_percent'], 
                    's-', color=COLORS['percent'], label='PERCENT', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Season', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Mean FFI', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('(a) Average FFI by Season', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # (b) 方法距离 by season
    axes[0, 1].set_facecolor(COLORS['bg'])
    axes[0, 1].plot(season_stats['season'], season_stats['d_rank_percent'], 
                    'D-', color=COLORS['primary'], linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Season', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Mean Kendall Distance', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('(b) Method Disagreement by Season', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # (c) 淘汰一致率 by season
    axes[1, 0].set_facecolor(COLORS['bg'])
    axes[1, 0].bar(season_stats['season'], season_stats['elim_agree'] * 100, 
                   color=COLORS['success'], alpha=0.7, edgecolor='white')
    axes[1, 0].set_xlabel('Season', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Elimination Agreement (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('(c) Elimination Consistency by Season', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(0, 110)
    
    # (d) FFI 分布对比
    axes[1, 1].set_facecolor(COLORS['bg'])
    axes[1, 1].hist(analysis_df['ffi_rank'], bins=20, alpha=0.6, 
                    color=COLORS['rank'], label='RANK', edgecolor='white')
    axes[1, 1].hist(analysis_df['ffi_percent'], bins=20, alpha=0.6, 
                    color=COLORS['percent'], label='PERCENT', edgecolor='white')
    axes[1, 1].axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    axes[1, 1].set_xlabel('FFI', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('(d) FFI Distribution Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    
    for ax_row in axes:
        for ax in ax_row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Task2_1_season_summary.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\n  All figures saved to: {save_dir}")


def main():
    # Resolve paths relative to the repo root for reproducibility.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset", "2026_MCM_Problem_C_Data.csv")
    fan_shares_csv = os.path.join(base_dir, "dataset", "fan_vote_shares.csv")
    save_dir = os.path.join(base_dir, "2.1_figures")
    
    print("\n" + "=" * 60)
    print("TASK 2.1: VOTING METHOD COMPARISON ANALYSIS")
    print("=" * 60)
    
    # 加载数据
    print("\nLoading data...")
    seasons_data, valid_seasons = load_and_preprocess_data(data_path)
    
    # 步骤1: 收集 fan vote shares（如果还没有）
    if not os.path.exists(fan_shares_csv):
        fan_shares_df = collect_fan_vote_shares(seasons_data, valid_seasons, fan_shares_csv)
    else:
        print(f"\n  Loading existing fan vote shares from: {fan_shares_csv}")
        fan_shares_df = pd.read_csv(fan_shares_csv)
    
    # 加载评委分数
    judge_scores = load_judge_scores(data_path, seasons_data)
    
    # 步骤2: 分析所有赛季
    print("\n" + "=" * 60)
    print("STEP 2: Analyzing All Seasons")
    print("=" * 60)
    
    all_results = []
    for season in sorted(valid_seasons):
        if season in judge_scores:
            print(f"  Season {season}...", end=" ")
            results = analyze_season(season, judge_scores[season], fan_shares_df)
            all_results.extend(results)
            print(f"Done ({len(results)} weeks)")
    
    analysis_df = pd.DataFrame(all_results)
    
    # 保存分析结果
    analysis_csv = fan_shares_csv.replace('.csv', '_analysis.csv')
    analysis_df.to_csv(analysis_csv, index=False)
    print(f"\n  Saved analysis results to: {analysis_csv}")
    
    # 步骤3: 生成可视化
    generate_visualizations(analysis_df, save_dir)
    
    # 打印总结统计
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\n  Total weeks analyzed: {len(analysis_df)}")
    print(f"  Elimination agreement rate: {analysis_df['elim_agree'].mean()*100:.1f}%")
    print(f"\n  Mean FFI (RANK):    {analysis_df['ffi_rank'].mean():+.4f}")
    print(f"  Mean FFI (PERCENT): {analysis_df['ffi_percent'].mean():+.4f}")
    print(f"\n  Mean Kendall Distance (RANK vs PERCENT): {analysis_df['d_rank_percent'].mean():.4f}")
    
    # 判断哪个更 favor fans
    if analysis_df['ffi_rank'].mean() > analysis_df['ffi_percent'].mean():
        print(f"\n  ** RANK method favors fan votes MORE (FFI difference: {analysis_df['ffi_rank'].mean() - analysis_df['ffi_percent'].mean():+.4f}) **")
    else:
        print(f"\n  ** PERCENT method favors fan votes MORE (FFI difference: {analysis_df['ffi_percent'].mean() - analysis_df['ffi_rank'].mean():+.4f}) **")
    
    print("\n" + "=" * 60)
    print("TASK 2.1 COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
