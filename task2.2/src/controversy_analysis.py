"""
MCM 2026 Problem C - Task 2.2: Controversy Case Analysis (Full Data-Driven)
============================================================================
完整数据驱动的反事实分析，对所有争议选手进行模拟

四步骤：
  Step 1: 争议选手识别（数据驱动）
  Step 2: RANK vs PERCENT 全员反事实模拟
  Step 3: RANK + Save 机制全员模拟
  Step 4: 三种方法的Suppression效果统计分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Color Hunt 风格配色 - 柔和高端大气
# ============================================================
PALETTE = {
    # 主色调
    'rank': '#E76F51',       # 珊瑚红橙
    'percent': '#5C469C',    # 深紫蓝
    'save': '#2D9596',       # 青绿色
    
    # 四个指定案例
    'jerry': '#E76F51',
    'billy': '#569DAA', 
    'bristol': '#87CBB9',
    'bobby': '#7C3AED',
    
    # 辅助色
    'positive': '#E76F51',   # 抑制（名次变差）
    'negative': '#4CAF50',   # 帮助（名次变好）
    'neutral': '#95A5A6',
    
    # 背景
    'bg': '#FAFBFC',
    'grid': '#E8E8E8',
    'text': '#2C3E50',
    
    # 渐变色板
    'grad1': '#DFF2EB',
    'grad2': '#B9E5E8',
    'grad3': '#7AB2D3',
    'grad4': '#4A628A',
}

plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': PALETTE['bg'],
    'figure.facecolor': 'white',
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# 四个指定案例
SPECIFIED_CASES = [
    ('Jerry Rice', 2),
    ('Billy Ray Cyrus', 4),
    ('Bristol Palin', 11),
    ('Bobby Bones', 27)
]


# ============================================================
# 数据加载
# ============================================================
def load_all_data(base_dir):
    """加载所有数据"""
    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)
    
    base_dir = Path(base_dir)
    fan_df = pd.read_csv(str(base_dir / "task1" / "table" / "fan_vote_shares.csv"))
    print(f"  Fan data: {len(fan_df)} records")
    
    data_df = pd.read_csv(str(base_dir / "2026_MCM_Problem_C_Data.csv"))
    
    judge_data = []
    for _, row in data_df.iterrows():
        name = row['celebrity_name']
        season = row['season']
        placement = row['placement']
        
        for week in range(1, 12):
            week_scores = []
            for judge in range(1, 5):
                col = f'week{week}_judge{judge}_score'
                if col in row.index:
                    val = row[col]
                    if pd.notna(val) and val not in ['N/A', 'n/a', ''] and val != 0:
                        try:
                            week_scores.append(float(val))
                        except:
                            pass
            
            if week_scores and sum(week_scores) > 0:
                judge_data.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'judge_total': sum(week_scores),
                    'actual_placement': placement
                })
    
    judge_df = pd.DataFrame(judge_data)
    print(f"  Judge data: {len(judge_df)} records")
    
    return fan_df, judge_df, data_df


# ============================================================
# STEP 1: 争议选手识别
# ============================================================
def step1_identify_controversy(fan_df, judge_df, save_dir):
    """Step 1: 识别所有争议选手"""
    print("\n" + "=" * 70)
    print("STEP 1: CONTROVERSY IDENTIFICATION")
    print("=" * 70)
    
    contestants_data = []
    
    for (season, name), group in fan_df.groupby(['season', 'celebrity_name']):
        if len(group) < 2:
            continue
        
        person_judge = judge_df[(judge_df['season'] == season) & 
                                (judge_df['celebrity_name'] == name)]
        if len(person_judge) == 0:
            continue
        
        placement = person_judge['actual_placement'].iloc[0]
        
        weekly_gaps = []
        weeks_lowest_judge = 0
        
        for week in group['week'].unique():
            week_all_fan = fan_df[(fan_df['season'] == season) & (fan_df['week'] == week)]
            week_all_judge = judge_df[(judge_df['season'] == season) & (judge_df['week'] == week)]
            
            if len(week_all_fan) < 2 or len(week_all_judge) < 2:
                continue
            
            judge_ranks = week_all_judge.set_index('celebrity_name')['judge_total'].rank(
                ascending=False, method='average')
            fan_ranks = week_all_fan.set_index('celebrity_name')['fan_vote_share'].rank(
                ascending=False, method='average')
            
            if name in judge_ranks.index and name in fan_ranks.index:
                judge_rank = judge_ranks[name]
                fan_rank = fan_ranks[name]
                gap = judge_rank - fan_rank  # positive = judge低fan高 = fan-favored
                weekly_gaps.append(gap)
                
                if judge_rank == judge_ranks.max():
                    weeks_lowest_judge += 1
        
        if not weekly_gaps:
            continue
        
        avg_abs_gap = np.mean([abs(g) for g in weekly_gaps])
        weighted_gap = np.mean([abs(g) * (i+1)/len(weekly_gaps) for i, g in enumerate(weekly_gaps)])
        mean_delta = np.mean(weekly_gaps)
        
        contestants_data.append({
            'name': name,
            'season': season,
            'placement': int(placement),
            'avg_abs_gap': avg_abs_gap,
            'max_gap': max([abs(g) for g in weekly_gaps]),
            'weighted_gap': weighted_gap,
            'weeks_lowest_judge': weeks_lowest_judge,
            'total_weeks': len(weekly_gaps),
            'mean_delta': mean_delta,
            'controversy_type': 'fan-favored' if mean_delta > 1 else 'judge-favored' if mean_delta < -1 else 'neutral',
            'is_specified': (name, season) in SPECIFIED_CASES
        })
    
    controversy_df = pd.DataFrame(contestants_data)
    
    # 定义争议阈值（top 25%）
    threshold = controversy_df['weighted_gap'].quantile(0.75)
    controversy_df['is_controversial'] = controversy_df['weighted_gap'] >= threshold
    
    controversy_df.to_csv(f'{save_dir}/controversy_all_contestants.csv', index=False)
    
    n_controversial = controversy_df['is_controversial'].sum()
    print(f"\n  Total contestants analyzed: {len(controversy_df)}")
    print(f"  Controversy threshold (Q75): {threshold:.3f}")
    print(f"  Controversial contestants: {n_controversial}")
    print(f"\n  Breakdown by type:")
    for ct in ['fan-favored', 'neutral', 'judge-favored']:
        n = len(controversy_df[(controversy_df['is_controversial']) & 
                               (controversy_df['controversy_type'] == ct)])
        print(f"    {ct:15s}: {n}")
    
    return controversy_df


# ============================================================
# STEP 2 & 3: 全员反事实模拟
# ============================================================
def simulate_season_elimination(fan_df, judge_df, season, method='rank'):
    """模拟单季淘汰过程"""
    season_fan = fan_df[fan_df['season'] == season].copy()
    season_judge = judge_df[judge_df['season'] == season].copy()
    
    if season_fan.empty or season_judge.empty:
        return {}
    
    all_contestants = set(season_fan['celebrity_name'].unique()) & \
                     set(season_judge['celebrity_name'].unique())
    
    if not all_contestants:
        return {}
    
    remaining = set(all_contestants)
    elimination_order = []
    
    max_week = int(max(season_fan['week'].max(), season_judge['week'].max()))
    
    for week in range(1, max_week + 1):
        if len(remaining) <= 1:
            break
        
        week_fan = season_fan[
            (season_fan['week'] == week) & 
            (season_fan['celebrity_name'].isin(remaining))
        ]
        week_judge = season_judge[
            (season_judge['week'] == week) & 
            (season_judge['celebrity_name'].isin(remaining))
        ]
        
        if week_fan.empty or week_judge.empty:
            continue
        
        merged = pd.merge(
            week_fan[['celebrity_name', 'fan_vote_share']],
            week_judge[['celebrity_name', 'judge_total']],
            on='celebrity_name'
        )
        
        if len(merged) < 2:
            continue
        
        total_judge = merged['judge_total'].sum()
        merged['judge_percent'] = merged['judge_total'] / total_judge if total_judge > 0 else 0
        
        if method == 'rank':
            merged['rank_fan'] = merged['fan_vote_share'].rank(ascending=False, method='average')
            merged['rank_judge'] = merged['judge_total'].rank(ascending=False, method='average')
            merged['combined'] = merged['rank_fan'] + merged['rank_judge']
            eliminated_name = merged.loc[merged['combined'].idxmax(), 'celebrity_name']
        else:  # percent
            merged['combined'] = merged['fan_vote_share'] + merged['judge_percent']
            eliminated_name = merged.loc[merged['combined'].idxmin(), 'celebrity_name']
        
        elimination_order.append(eliminated_name)
        remaining.remove(eliminated_name)
    
    n_total = len(all_contestants)
    results = {}
    
    for i, name in enumerate(elimination_order):
        results[name] = n_total - i
    
    for i, name in enumerate(list(remaining)):
        results[name] = len(remaining) - i
    
    return results


def simulate_season_with_save(fan_df, judge_df, season):
    """模拟 RANK + Save 机制"""
    season_fan = fan_df[fan_df['season'] == season].copy()
    season_judge = judge_df[judge_df['season'] == season].copy()
    
    if season_fan.empty or season_judge.empty:
        return {}
    
    all_contestants = set(season_fan['celebrity_name'].unique()) & \
                     set(season_judge['celebrity_name'].unique())
    
    if not all_contestants:
        return {}
    
    remaining = set(all_contestants)
    elimination_order = []
    
    max_week = int(max(season_fan['week'].max(), season_judge['week'].max()))
    
    for week in range(1, max_week + 1):
        if len(remaining) <= 1:
            break
        
        week_fan = season_fan[
            (season_fan['week'] == week) & 
            (season_fan['celebrity_name'].isin(remaining))
        ]
        week_judge = season_judge[
            (season_judge['week'] == week) & 
            (season_judge['celebrity_name'].isin(remaining))
        ]
        
        if week_fan.empty or week_judge.empty:
            continue
        
        merged = pd.merge(
            week_fan[['celebrity_name', 'fan_vote_share']],
            week_judge[['celebrity_name', 'judge_total']],
            on='celebrity_name'
        )
        
        if len(merged) < 2:
            continue
        
        # RANK找bottom-two
        merged['rank_fan'] = merged['fan_vote_share'].rank(ascending=False, method='average')
        merged['rank_judge'] = merged['judge_total'].rank(ascending=False, method='average')
        merged['combined_rank'] = merged['rank_fan'] + merged['rank_judge']
        
        merged_sorted = merged.sort_values('combined_rank', ascending=False)
        bottom_two = merged_sorted.head(2)
        
        # Judges选择淘汰judge分数更低者
        if len(bottom_two) == 2:
            eliminated_idx = bottom_two['judge_total'].idxmin()
            eliminated_name = merged.loc[eliminated_idx, 'celebrity_name']
        else:
            eliminated_name = bottom_two.iloc[0]['celebrity_name']
        
        elimination_order.append(eliminated_name)
        remaining.remove(eliminated_name)
    
    n_total = len(all_contestants)
    results = {}
    
    for i, name in enumerate(elimination_order):
        results[name] = n_total - i
    
    for i, name in enumerate(list(remaining)):
        results[name] = len(remaining) - i
    
    return results


def step2_3_full_simulation(fan_df, judge_df, controversy_df, save_dir):
    """对所有争议选手进行完整反事实模拟"""
    print("\n" + "=" * 70)
    print("STEP 2-3: FULL COUNTERFACTUAL SIMULATION (All Controversial)")
    print("=" * 70)
    
    # 获取所有争议选手（包括四个指定案例，即使它们不在top 25%）
    controversial = controversy_df[
        (controversy_df['is_controversial']) | (controversy_df['is_specified'])
    ].copy()
    seasons_to_simulate = controversial['season'].unique()
    
    print(f"\n  Simulating {len(seasons_to_simulate)} seasons...")
    
    all_results = []
    
    for season in seasons_to_simulate:
        # 三种方法模拟
        rank_results = simulate_season_elimination(fan_df, judge_df, season, 'rank')
        percent_results = simulate_season_elimination(fan_df, judge_df, season, 'percent')
        save_results = simulate_season_with_save(fan_df, judge_df, season)
        
        # 记录该季所有争议选手的结果
        season_controversial = controversial[controversial['season'] == season]
        
        for _, row in season_controversial.iterrows():
            name = row['name']
            if name in rank_results and name in percent_results and name in save_results:
                all_results.append({
                    'name': name,
                    'season': season,
                    'controversy_type': row['controversy_type'],
                    'weighted_gap': row['weighted_gap'],
                    'is_specified': row['is_specified'],
                    'placement_rank': rank_results[name],
                    'placement_percent': percent_results[name],
                    'placement_save': save_results[name],
                    'delta_percent': percent_results[name] - rank_results[name],
                    'delta_save': save_results[name] - rank_results[name]
                })
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{save_dir}/counterfactual_all_controversial.csv', index=False)
    
    print(f"\n  Simulated {len(results_df)} controversial contestants")
    print(f"\n  Sample results (first 5):")
    for _, r in results_df.head(5).iterrows():
        print(f"    {r['name'][:18]:18s} (S{r['season']:2d}): "
              f"R={r['placement_rank']}, P={r['placement_percent']}, S={r['placement_save']}, "
              f"dP={r['delta_percent']:+d}, dS={r['delta_save']:+d}")
    
    return results_df


# ============================================================
# STEP 4: Suppression统计分析
# ============================================================
def step4_suppression_analysis(results_df, save_dir):
    """统计三种方法的Suppression效果"""
    print("\n" + "=" * 70)
    print("STEP 4: SUPPRESSION STATISTICAL ANALYSIS")
    print("=" * 70)
    
    # 计算三种方法的平均效果
    n_total = len(results_df)
    
    # PERCENT相对RANK的效果
    avg_delta_percent = results_df['delta_percent'].mean()
    std_delta_percent = results_df['delta_percent'].std()
    
    # 统计检验：是否显著不为0
    t_stat_p, p_value_p = stats.ttest_1samp(results_df['delta_percent'], 0)
    
    # SAVE相对RANK的效果
    avg_delta_save = results_df['delta_save'].mean()
    std_delta_save = results_df['delta_save'].std()
    t_stat_s, p_value_s = stats.ttest_1samp(results_df['delta_save'], 0)
    
    print(f"\n  [PERCENT vs RANK]")
    print(f"    N = {n_total}")
    print(f"    Mean delta = {avg_delta_percent:+.3f}")
    print(f"    Std = {std_delta_percent:.3f}")
    print(f"    t-stat = {t_stat_p:.3f}, p-value = {p_value_p:.4f}")
    print(f"    Significant (p<0.05)? {'Yes' if p_value_p < 0.05 else 'No'}")
    
    print(f"\n  [SAVE vs RANK]")
    print(f"    Mean delta = {avg_delta_save:+.3f}")
    print(f"    Std = {std_delta_save:.3f}")
    print(f"    t-stat = {t_stat_s:.3f}, p-value = {p_value_s:.4f}")
    print(f"    Significant (p<0.05)? {'Yes' if p_value_s < 0.05 else 'No'}")
    
    # 分类统计
    print(f"\n  [By Controversy Type]")
    for ct in ['fan-favored', 'neutral', 'judge-favored']:
        subset = results_df[results_df['controversy_type'] == ct]
        if len(subset) > 0:
            print(f"\n    {ct} (N={len(subset)}):")
            print(f"      PERCENT effect: {subset['delta_percent'].mean():+.3f}")
            print(f"      SAVE effect: {subset['delta_save'].mean():+.3f}")
    
    # 计算Suppression Score（正值=抑制效果）
    # 这里我们定义：名次变大 = 被抑制
    suppression_scores = {
        'RANK': 0.0,  # baseline
        'PERCENT': avg_delta_percent,
        'RANK+Save': avg_delta_save
    }
    
    # 排序
    ranking = sorted(suppression_scores.items(), key=lambda x: -x[1])
    
    print(f"\n  [SUPPRESSION RANKING]")
    print(f"  (Higher score = Stronger suppression of controversial contestants)")
    for i, (method, score) in enumerate(ranking, 1):
        print(f"    {i}. {method:12s}: {score:+.3f}")
    
    # 保存统计结果
    stats_summary = {
        'metric': ['mean_delta', 'std', 't_stat', 'p_value', 'n_suppressed', 'n_helped'],
        'PERCENT_vs_RANK': [
            avg_delta_percent,
            std_delta_percent,
            t_stat_p,
            p_value_p,
            (results_df['delta_percent'] > 0).sum(),
            (results_df['delta_percent'] < 0).sum()
        ],
        'SAVE_vs_RANK': [
            avg_delta_save,
            std_delta_save,
            t_stat_s,
            p_value_s,
            (results_df['delta_save'] > 0).sum(),
            (results_df['delta_save'] < 0).sum()
        ]
    }
    pd.DataFrame(stats_summary).to_csv(f'{save_dir}/suppression_stats.csv', index=False)
    
    return suppression_scores, ranking


# ============================================================
# CRITICAL VOTE & SAFETY MARGIN (针对指定案例)
# ============================================================
def compute_critical_vote(fan_df, judge_df, season, week, target_name, method='rank'):
    """使用二分搜索计算临界票"""
    week_fan = fan_df[(fan_df['season'] == season) & (fan_df['week'] == week)].copy()
    week_judge = judge_df[(judge_df['season'] == season) & (judge_df['week'] == week)].copy()
    
    if week_fan.empty or week_judge.empty:
        return None, None
    
    merged = pd.merge(
        week_fan[['celebrity_name', 'fan_vote_share']],
        week_judge[['celebrity_name', 'judge_total']],
        on='celebrity_name'
    )
    
    if target_name not in merged['celebrity_name'].values or len(merged) < 2:
        return None, None
    
    actual_share = merged.loc[merged['celebrity_name'] == target_name, 'fan_vote_share'].values[0]
    
    def would_survive(test_share):
        """判断是否能存活"""
        test_merged = merged.copy()
        test_merged.loc[test_merged['celebrity_name'] == target_name, 'fan_vote_share'] = test_share
        
        total_fan = test_merged['fan_vote_share'].sum()
        test_merged['fan_vote_share'] = test_merged['fan_vote_share'] / total_fan
        
        total_judge = test_merged['judge_total'].sum()
        test_merged['judge_percent'] = test_merged['judge_total'] / total_judge if total_judge > 0 else 0
        
        if method == 'rank':
            test_merged['rank_fan'] = test_merged['fan_vote_share'].rank(ascending=False, method='average')
            test_merged['rank_judge'] = test_merged['judge_total'].rank(ascending=False, method='average')
            test_merged['combined'] = test_merged['rank_fan'] + test_merged['rank_judge']
            eliminated_name = test_merged.loc[test_merged['combined'].idxmax(), 'celebrity_name']
        else:
            test_merged['combined'] = test_merged['fan_vote_share'] + test_merged['judge_percent']
            eliminated_name = test_merged.loc[test_merged['combined'].idxmin(), 'celebrity_name']
        
        return eliminated_name != target_name
    
    if not would_survive(actual_share):
        return actual_share, 0
    
    low, high = 0.001, actual_share
    if would_survive(low):
        return low, actual_share - low
    
    for _ in range(50):
        mid = (low + high) / 2
        if would_survive(mid):
            high = mid
        else:
            low = mid
        if high - low < 0.0001:
            break
    
    return high, actual_share - high


def critical_vote_analysis(fan_df, judge_df, save_dir):
    """针对四个指定案例计算临界票"""
    print("\n" + "=" * 70)
    print("CRITICAL VOTE & SAFETY MARGIN ANALYSIS")
    print("=" * 70)
    
    critical_data = {}
    
    for name, season in SPECIFIED_CASES:
        person_weeks = fan_df[(fan_df['season'] == season) & 
                              (fan_df['celebrity_name'] == name)]['week'].unique()
        
        weekly_data = []
        for week in sorted(person_weeks):
            actual = fan_df[(fan_df['season'] == season) & 
                           (fan_df['week'] == week) & 
                           (fan_df['celebrity_name'] == name)]['fan_vote_share'].values
            
            if len(actual) == 0:
                continue
            
            actual_share = actual[0]
            crit_rank, margin_rank = compute_critical_vote(fan_df, judge_df, season, week, name, 'rank')
            crit_percent, margin_percent = compute_critical_vote(fan_df, judge_df, season, week, name, 'percent')
            
            weekly_data.append({
                'week': week,
                'actual': actual_share * 100,
                'critical_rank': crit_rank * 100 if crit_rank else None,
                'critical_percent': crit_percent * 100 if crit_percent else None,
                'margin_rank': margin_rank * 100 if margin_rank else None,
                'margin_percent': margin_percent * 100 if margin_percent else None
            })
        
        critical_data[name] = {'season': season, 'weekly': weekly_data}
        
        valid_margins = [w['margin_rank'] for w in weekly_data if w['margin_rank'] is not None]
        if valid_margins:
            print(f"\n  {name} (S{season}):")
            print(f"    Min margin: {min(valid_margins):.2f}%")
            print(f"    Danger weeks (<5%): {sum(1 for m in valid_margins if m < 5)}")
    
    return critical_data


# ============================================================
# 可视化 - Color Hunt配色 + 多样化展示
# ============================================================
def create_step1_radar_chart(controversy_df, save_dir):
    """Step 1: 争议选手雷达图"""
    print("\n  Creating Step 1 visualization (Radar Chart)...")
    
    specified = controversy_df[controversy_df['is_specified']]
    
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    categories = ['Avg Gap', 'Max Gap', 'Weeks\nLowest', 'Final\nPlace', 'Weighted\nGap']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors_spec = [PALETTE['jerry'], PALETTE['billy'], PALETTE['bristol'], PALETTE['bobby']]
    
    for idx, (_, case) in enumerate(specified.iterrows()):
        ax = fig.add_subplot(gs[0, idx], projection='polar')
        ax.set_facecolor('#FAFBFC')
        
        values = [
            min(case['avg_abs_gap'] / 8, 1),
            min(case['max_gap'] / 12, 1),
            min(case['weeks_lowest_judge'] / max(case['total_weeks'], 1), 1),
            min((10 - case['placement']) / 9, 1),
            min(case['weighted_gap'] / 6, 1)
        ]
        values += values[:1]
        
        ax.fill(angles, values, alpha=0.25, color=colors_spec[idx])
        ax.plot(angles, values, 'o-', linewidth=2.5, color=colors_spec[idx], markersize=6)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['25%', '50%', '75%'], fontsize=7, color='gray')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        ax.set_title(f"{case['name']}\n(S{case['season']}, #{int(case['placement'])})", 
                    fontsize=11, fontweight='bold', pad=10, color=colors_spec[idx])
    
    # 下方：所有争议选手分布
    ax_dist = fig.add_subplot(gs[1, :2])
    ax_dist.set_facecolor(PALETTE['bg'])
    
    controversial = controversy_df[controversy_df['is_controversial']]
    
    for ct, color in [('fan-favored', PALETTE['positive']), 
                      ('neutral', PALETTE['neutral']),
                      ('judge-favored', PALETTE['grad4'])]:
        subset = controversial[controversial['controversy_type'] == ct]
        ax_dist.scatter(subset['weighted_gap'], subset['placement'], 
                       s=80, c=color, alpha=0.6, label=f'{ct} ({len(subset)})',
                       edgecolors='white', linewidths=1)
    
    # 标记四个指定案例
    for _, case in specified.iterrows():
        ax_dist.scatter(case['weighted_gap'], case['placement'], 
                       s=200, marker='*', c='gold', edgecolors='black', linewidths=1.5, zorder=5)
    
    ax_dist.set_xlabel('Weighted Controversy Index', fontsize=11, fontweight='bold')
    ax_dist.set_ylabel('Final Placement', fontsize=11, fontweight='bold')
    ax_dist.set_title('All Controversial Contestants Distribution', fontsize=12, fontweight='bold')
    ax_dist.legend(loc='upper right', fontsize=9)
    ax_dist.invert_yaxis()
    ax_dist.grid(alpha=0.3, linestyle='--')
    
    # 右下：类型饼图
    ax_pie = fig.add_subplot(gs[1, 2:])
    type_counts = controversial['controversy_type'].value_counts()
    colors_pie = [PALETTE['positive'], PALETTE['neutral'], PALETTE['grad4']]
    
    wedges, texts, autotexts = ax_pie.pie(
        type_counts.values, 
        labels=type_counts.index, 
        autopct='%1.1f%%',
        colors=colors_pie[:len(type_counts)],
        explode=[0.05] * len(type_counts),
        startangle=90,
        textprops={'fontsize': 10}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax_pie.set_title(f'Controversy Type Distribution\n(N={len(controversial)})', 
                    fontsize=12, fontweight='bold')
    
    plt.suptitle('Step 1: Controversy Identification (All Contestants)', 
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(f'{save_dir}/Step1_controversy_radar.png', dpi=300, facecolor='white')
    plt.close()
    print(f"    Saved: Step1_controversy_radar.png")


def create_step2_bump_chart(results_df, save_dir):
    """Step 2: Bump Chart - 方法对比"""
    print("\n  Creating Step 2 visualization (Bump Chart)...")
    
    # 找出四个指定案例（通过名字和赛季匹配）
    specified_names = {(n, s) for n, s in SPECIFIED_CASES}
    specified = results_df[
        results_df.apply(lambda r: (r['name'], r['season']) in specified_names, axis=1)
    ].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    
    # 左图：四个指定案例的Bump Chart
    ax1 = axes[0]
    ax1.set_facecolor(PALETTE['bg'])
    
    methods = ['RANK', 'PERCENT', 'RANK+Save']
    x_positions = np.arange(len(methods))
    
    colors_spec = [PALETTE['jerry'], PALETTE['billy'], PALETTE['bristol'], PALETTE['bobby']]
    
    for i, (_, row) in enumerate(specified.iterrows()):
        y_data = [row['placement_rank'], row['placement_percent'], row['placement_save']]
        
        ax1.plot(x_positions, y_data, 'o-', linewidth=3, markersize=16,
                color=colors_spec[i], alpha=0.8, label=f"{row['name'][:12]} (S{row['season']})",
                markeredgecolor='white', markeredgewidth=2)
        
        for j, y in enumerate(y_data):
            ax1.text(j, y, f'{int(y)}', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
    
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Placement', fontsize=11, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_title('(a) Four Specified Cases: Method Comparison', fontsize=12, fontweight='bold')
    
    # 右图：所有争议选手的Delta分布
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE['bg'])
    
    # Violin plot
    data_violin = [results_df['delta_percent'].values, results_df['delta_save'].values]
    parts = ax2.violinplot(data_violin, positions=[1, 2], showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([PALETTE['percent'], PALETTE['save']][i])
        pc.set_alpha(0.6)
    
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('black')
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['PERCENT - RANK', 'SAVE - RANK'], fontsize=11, fontweight='bold')
    ax2.set_ylabel('Placement Change', fontsize=11, fontweight='bold')
    ax2.set_title(f'(b) All Controversial: Delta Distribution (N={len(results_df)})', 
                 fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 标注均值
    ax2.text(1.2, results_df['delta_percent'].mean(), 
            f'Mean: {results_df["delta_percent"].mean():+.2f}',
            fontsize=10, fontweight='bold', color=PALETTE['percent'])
    ax2.text(2.2, results_df['delta_save'].mean(), 
            f'Mean: {results_df["delta_save"].mean():+.2f}',
            fontsize=10, fontweight='bold', color=PALETTE['save'])
    
    plt.suptitle('Step 2-3: Counterfactual Simulation Results', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/Step2_3_counterfactual.png', dpi=300, facecolor='white')
    plt.close()
    print(f"    Saved: Step2_3_counterfactual.png")


def create_critical_vote_viz(critical_data, save_dir):
    """临界票可视化（两个典型案例）"""
    print("\n  Creating Critical Vote visualization...")
    
    cases_to_show = ['Jerry Rice', 'Bristol Palin']
    colors_case = {'Jerry Rice': PALETTE['jerry'], 'Bristol Palin': PALETTE['bristol']}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    
    for idx, name in enumerate(cases_to_show):
        if name not in critical_data:
            continue
        
        data = critical_data[name]
        weekly = data['weekly']
        
        if not weekly:
            continue
        
        weeks = [w['week'] for w in weekly]
        actual = [w['actual'] for w in weekly]
        crit_rank = [w['critical_rank'] if w['critical_rank'] else 0 for w in weekly]
        crit_percent = [w['critical_percent'] if w['critical_percent'] else 0 for w in weekly]
        margin_rank = [w['margin_rank'] if w['margin_rank'] else 0 for w in weekly]
        margin_percent = [w['margin_percent'] if w['margin_percent'] else 0 for w in weekly]
        
        color = colors_case[name]
        
        # 左图：Actual vs Critical
        ax_left = axes[idx, 0]
        ax_left.set_facecolor(PALETTE['bg'])
        
        ax_left.plot(weeks, actual, 'o-', label='Actual', linewidth=3, markersize=10, color=color, zorder=3)
        ax_left.plot(weeks, crit_rank, 's--', label='Critical (RANK)', linewidth=2, markersize=7, 
                    color=PALETTE['rank'], alpha=0.8, zorder=2)
        ax_left.plot(weeks, crit_percent, 'd--', label='Critical (PERCENT)', linewidth=2, markersize=7,
                    color=PALETTE['percent'], alpha=0.8, zorder=2)
        
        ax_left.fill_between(weeks, crit_rank, actual,
                            where=[a >= c for a, c in zip(actual, crit_rank)],
                            alpha=0.15, color=PALETTE['negative'], label='Safe Zone')
        
        # 标注最小margin
        if margin_rank:
            valid_margins = [(i, m) for i, m in enumerate(margin_rank) if m > 0]
            if valid_margins:
                min_i, min_m = min(valid_margins, key=lambda x: x[1])
                ax_left.annotate(f'Min: {min_m:.1f}%',
                               xy=(weeks[min_i], actual[min_i]),
                               xytext=(15, -20), textcoords='offset points',
                               fontsize=9, fontweight='bold', color=PALETTE['positive'],
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', lw=1.5, color=PALETTE['positive']))
        
        ax_left.set_xlabel('Week', fontsize=11, fontweight='bold')
        ax_left.set_ylabel('Fan Vote Share (%)', fontsize=11, fontweight='bold')
        ax_left.set_title(f'({chr(97+idx*2)}) {name}: Actual vs Critical', fontsize=12, fontweight='bold')
        ax_left.legend(loc='best', fontsize=9)
        ax_left.grid(alpha=0.3, linestyle='--')
        
        # 右图：Safety Margin
        ax_right = axes[idx, 1]
        ax_right.set_facecolor(PALETTE['bg'])
        
        x_pos = np.arange(len(weeks))
        width = 0.35
        
        colors_rank = [PALETTE['negative'] if m > 5 else PALETTE['positive'] if m < 3 else '#FFA500' 
                      for m in margin_rank]
        colors_percent = [PALETTE['negative'] if m > 5 else PALETTE['positive'] if m < 3 else '#FFA500'
                         for m in margin_percent]
        
        bars1 = ax_right.bar(x_pos - width/2, margin_rank, width, label='RANK',
                            color=colors_rank, alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = ax_right.bar(x_pos + width/2, margin_percent, width, label='PERCENT',
                            color=colors_percent, alpha=0.8, edgecolor='white', linewidth=1)
        
        ax_right.axhline(y=5, color=PALETTE['positive'], linestyle='--', linewidth=2,
                        alpha=0.7, label='Danger Threshold')
        ax_right.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        for i, (mr, mp) in enumerate(zip(margin_rank, margin_percent)):
            if 0 < mr < 5:
                ax_right.text(i - width/2, mr + 0.5, f'{mr:.1f}', ha='center',
                             fontsize=8, color='darkred', fontweight='bold')
            if 0 < mp < 5:
                ax_right.text(i + width/2, mp + 0.5, f'{mp:.1f}', ha='center',
                             fontsize=8, color='darkred', fontweight='bold')
        
        ax_right.set_xlabel('Week', fontsize=11, fontweight='bold')
        ax_right.set_ylabel('Safety Margin (%)', fontsize=11, fontweight='bold')
        ax_right.set_title(f'({chr(98+idx*2)}) {name}: Safety Margin', fontsize=12, fontweight='bold')
        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels(weeks, fontsize=10)
        ax_right.legend(loc='best', fontsize=9)
        ax_right.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Critical Fan Vote & Safety Margin Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/Step_Critical_Vote.png', dpi=300, facecolor='white')
    plt.close()
    print(f"    Saved: Step_Critical_Vote.png")


def create_suppression_dashboard(results_df, suppression_scores, ranking, save_dir):
    """创建Suppression分析仪表板"""
    print("\n  Creating Suppression Dashboard...")
    
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # (a) 方法排名条形图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PALETTE['bg'])
    
    methods = [r[0] for r in ranking]
    scores = [r[1] for r in ranking]
    colors_bar = [PALETTE['rank'] if m == 'RANK' else PALETTE['percent'] if m == 'PERCENT' else PALETTE['save']
                 for m in methods]
    
    bars = ax1.barh(range(len(methods)), scores, color=colors_bar, alpha=0.85,
                   edgecolor='white', linewidth=2, height=0.6)
    
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    
    for i, (method, score) in enumerate(zip(methods, scores)):
        ax1.text(-0.3, i, method, ha='right', va='center', fontsize=12, fontweight='bold')
        ax1.text(score + 0.05 if score >= 0 else score - 0.05, i, f'{score:+.3f}',
                ha='left' if score >= 0 else 'right', va='center', fontsize=11, fontweight='bold')
    
    ax1.set_yticks([])
    ax1.set_xlabel('Suppression Score\n(+ = Stronger Suppression)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Method Suppression Ranking', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # (b) 按类型分组的效果
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PALETTE['bg'])
    
    types = ['fan-favored', 'neutral', 'judge-favored']
    x_pos = np.arange(len(types))
    width = 0.35
    
    percent_by_type = [results_df[results_df['controversy_type'] == t]['delta_percent'].mean() 
                       for t in types]
    save_by_type = [results_df[results_df['controversy_type'] == t]['delta_save'].mean() 
                    for t in types]
    
    ax2.bar(x_pos - width/2, percent_by_type, width, label='PERCENT', 
           color=PALETTE['percent'], alpha=0.85, edgecolor='white')
    ax2.bar(x_pos + width/2, save_by_type, width, label='SAVE', 
           color=PALETTE['save'], alpha=0.85, edgecolor='white')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([t.replace('-', '\n') for t in types], fontsize=10)
    ax2.set_ylabel('Mean Placement Change', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Effect by Controversy Type', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # (c) 散点图：争议强度 vs 效果
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PALETTE['bg'])
    
    ax3.scatter(results_df['weighted_gap'], results_df['delta_percent'],
               s=60, c=PALETTE['percent'], alpha=0.6, label='PERCENT effect', edgecolors='white')
    
    # 趋势线
    z = np.polyfit(results_df['weighted_gap'], results_df['delta_percent'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(results_df['weighted_gap'].min(), results_df['weighted_gap'].max(), 100)
    ax3.plot(x_line, p(x_line), '--', color=PALETTE['percent'], linewidth=2, alpha=0.8)
    
    # 相关系数
    corr, p_val = stats.pearsonr(results_df['weighted_gap'], results_df['delta_percent'])
    ax3.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', transform=ax3.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_xlabel('Controversy Intensity', fontsize=11, fontweight='bold')
    ax3.set_ylabel('PERCENT Effect', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Controversy vs PERCENT Impact', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')
    
    # (d) 堆叠条形图：抑制/帮助比例
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(PALETTE['bg'])
    
    n_suppressed_p = (results_df['delta_percent'] > 0).sum()
    n_helped_p = (results_df['delta_percent'] < 0).sum()
    n_unchanged_p = (results_df['delta_percent'] == 0).sum()
    
    n_suppressed_s = (results_df['delta_save'] > 0).sum()
    n_helped_s = (results_df['delta_save'] < 0).sum()
    n_unchanged_s = (results_df['delta_save'] == 0).sum()
    
    methods_stack = ['PERCENT', 'SAVE']
    suppressed = [n_suppressed_p, n_suppressed_s]
    unchanged = [n_unchanged_p, n_unchanged_s]
    helped = [n_helped_p, n_helped_s]
    
    x_stack = np.arange(len(methods_stack))
    
    ax4.bar(x_stack, suppressed, 0.5, label='Suppressed (+)', color=PALETTE['positive'], alpha=0.85)
    ax4.bar(x_stack, unchanged, 0.5, bottom=suppressed, label='Unchanged (0)', color=PALETTE['neutral'], alpha=0.85)
    ax4.bar(x_stack, helped, 0.5, bottom=[s+u for s,u in zip(suppressed, unchanged)], 
           label='Helped (-)', color=PALETTE['negative'], alpha=0.85)
    
    ax4.set_xticks(x_stack)
    ax4.set_xticklabels(methods_stack, fontsize=11, fontweight='bold')
    ax4.set_ylabel('Number of Contestants', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Outcome Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    
    # (e) Top 10 最受影响选手表格
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    # 按|delta_percent|排序，取top 10最受影响的
    results_df['abs_delta'] = results_df['delta_percent'].abs()
    top_affected = results_df.nlargest(10, 'abs_delta')
    
    table_data = []
    for _, r in top_affected.iterrows():
        effect_p = 'Suppressed' if r['delta_percent'] > 0 else 'Helped' if r['delta_percent'] < 0 else 'Unchanged'
        table_data.append([
            r['name'][:15],
            f"S{int(r['season'])}",
            r['controversy_type'][:10],
            f"#{int(r['placement_rank'])}",
            f"#{int(r['placement_percent'])}",
            f"#{int(r['placement_save'])}",
            f"{int(r['delta_percent']):+d}",
            effect_p
        ])
    
    if len(table_data) > 0:
        table = ax5.table(
            cellText=table_data,
            colLabels=['Name', 'Season', 'Type', 'RANK', 'PCT', 'SAVE', 'Delta', 'Effect'],
            cellLoc='center', loc='center',
            bbox=[0, 0.05, 1, 0.85]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(len(table_data) + 1):
            for j in range(8):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor(PALETTE['grad4'])
                    cell.set_text_props(weight='bold', color='white')
                elif j == 7:  # Effect列
                    if table_data[i-1][7] == 'Suppressed':
                        cell.set_facecolor('#FFCCCC')
                    elif table_data[i-1][7] == 'Helped':
                        cell.set_facecolor('#CCFFCC')
    
    ax5.set_title('(e) Top 10 Most Affected by PERCENT', fontsize=12, fontweight='bold', y=0.95)
    
    plt.suptitle('Task 2.2: Controversy Suppression Analysis Dashboard\n' +
                f'(Full Data-Driven, N={len(results_df)} Controversial Contestants)',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(f'{save_dir}/Step4_suppression_dashboard.png', dpi=300, facecolor='white')
    plt.close()
    print(f"    Saved: Step4_suppression_dashboard.png")


# ============================================================
# MAIN
# ============================================================
def main():
    repo_root = Path(__file__).resolve().parents[2]
    figure_dir = repo_root / "task2.2" / "figure"
    table_dir = repo_root / "task2.2" / "table"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("TASK 2.2: CONTROVERSY ANALYSIS (FULL DATA-DRIVEN)")
    print("=" * 70)
    
    # 加载数据
    fan_df, judge_df, data_df = load_all_data(str(repo_root))
    
    # Step 1: 识别所有争议选手
    controversy_df = step1_identify_controversy(fan_df, judge_df, str(table_dir))
    
    # Step 2-3: 全员反事实模拟
    results_df = step2_3_full_simulation(fan_df, judge_df, controversy_df, str(table_dir))
    
    # Step 4: Suppression统计分析
    suppression_scores, ranking = step4_suppression_analysis(results_df, str(table_dir))
    
    # Critical Vote分析（针对指定案例）
    critical_data = critical_vote_analysis(fan_df, judge_df, str(table_dir))
    
    # 生成可视化
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    create_step1_radar_chart(controversy_df, str(figure_dir))
    create_step2_bump_chart(results_df, str(figure_dir))
    create_critical_vote_viz(critical_data, str(figure_dir))
    create_suppression_dashboard(results_df, suppression_scores, ranking, str(figure_dir))
    
    # 最终结论
    print("\n" + "=" * 70)
    print("FINAL CONCLUSIONS")
    print("=" * 70)
    
    print(f"\n  [DATA SUMMARY]")
    print(f"    Total contestants: {len(controversy_df)}")
    print(f"    Controversial (top 25%): {len(results_df)}")
    
    print(f"\n  [SUPPRESSION RANKING] (based on {len(results_df)} controversial contestants)")
    for i, (method, score) in enumerate(ranking, 1):
        sig = ''
        if method == 'PERCENT':
            t, p = stats.ttest_1samp(results_df['delta_percent'], 0)
            sig = ' ***' if p < 0.001 else ' **' if p < 0.01 else ' *' if p < 0.05 else ''
        elif method == 'RANK+Save':
            t, p = stats.ttest_1samp(results_df['delta_save'], 0)
            sig = ' ***' if p < 0.001 else ' **' if p < 0.01 else ' *' if p < 0.05 else ''
        print(f"    {i}. {method:12s}: {score:+.3f}{sig}")
    
    print(f"\n  [KEY FINDINGS]")
    if ranking[0][1] > 0:
        print(f"    - {ranking[0][0]} shows STRONGEST suppression effect on controversial contestants")
    else:
        print(f"    - All methods show weak or negative suppression effect")
    
    n_helped_p = (results_df['delta_percent'] < 0).sum()
    n_suppressed_p = (results_df['delta_percent'] > 0).sum()
    print(f"    - PERCENT: {n_suppressed_p} suppressed, {n_helped_p} helped")
    
    n_helped_s = (results_df['delta_save'] < 0).sum()
    n_suppressed_s = (results_df['delta_save'] > 0).sum()
    print(f"    - SAVE: {n_suppressed_s} suppressed, {n_helped_s} helped")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED!")
    print(f"All figures saved to: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
