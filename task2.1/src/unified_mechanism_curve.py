"""
MCM 2026 Problem C - Unified Mechanism Analysis (Corrected Version)
====================================================================
核心修正：使用整季淘汰模拟（Counterfactual Simulation）代替逐周排名差

分析目标：
  - X轴：Δ = mean(fan_share - judge_percent) 表示选手的粉丝偏好程度
  - Y轴：placement_PERCENT - placement_RANK 表示PERCENT方法对该选手的"打压/提升"效果
  
预期曲线形态（如果PERCENT抑制极端fan-favored）：
  - Δ < 0（judge-favored）: Y < 0（PERCENT帮助）
  - 0 < Δ < threshold: Y ≈ 0（中立区）
  - Δ > threshold（extreme fan-favored）: Y > 0（PERCENT抑制）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from pathlib import Path
from scipy.stats import linregress, spearmanr
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配色方案 - 借鉴 Color Hunt 柔和高端配色
# ============================================================
PALETTE = {
    # 主色调 - 冷暖对比
    'primary': '#569DAA',       # 宁静蓝绿
    'secondary': '#E67E22',     # 暖橙
    'accent': '#87CBB9',        # 薄荷绿
    
    # 区域填充 - 柔和渐变
    'zone_judge': '#C8ACD6',    # 淡紫（judge-favored区）
    'zone_neutral': '#F5EFFF',  # 极淡紫白（中立区）
    'zone_fan': '#FFE9D0',      # 浅杏（fan-favored区）
    
    # 散点颜色 - 按类型
    'dot_judge': '#5C469C',     # 深紫
    'dot_neutral': '#93B1A6',   # 灰绿
    'dot_fan': '#E76F51',       # 珊瑚橙
    
    # 背景与辅助
    'bg': '#FAFBFC',
    'grid': '#E5E5E5',
    'text_dark': '#2D3436',
    'text_light': '#636E72',
    
    # 强调色
    'highlight': '#FF6B6B',
    'success': '#26DE81',
    'warning': '#F7B731',
}

# 全局样式
plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'Segoe UI', 'sans-serif'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#CCCCCC',
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.3,
})


# ============================================================
# STEP 1: 数据加载与预处理
# ============================================================
def load_all_data(base_dir):
    """加载所有必需数据"""
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)
    
    # Fan vote shares
    fan_df = pd.read_csv(str(Path(base_dir) / "task1" / "table" / "fan_vote_shares.csv"))
    print(f"  Fan vote data: {len(fan_df)} records")
    
    # Original data with judge scores
    data_df = pd.read_csv(str(Path(base_dir) / "2026_MCM_Problem_C_Data.csv"))
    print(f"  Original data: {len(data_df)} contestants")
    
    # 提取每周judge scores
    judge_weekly = []
    for _, row in data_df.iterrows():
        name = row['celebrity_name']
        season = row['season']
        actual_placement = row['placement']
        
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
                judge_weekly.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'judge_total': sum(week_scores),
                    'actual_placement': actual_placement
                })
    
    judge_df = pd.DataFrame(judge_weekly)
    print(f"  Judge weekly scores: {len(judge_df)} records")
    
    return fan_df, judge_df, data_df


# ============================================================
# STEP 2: 整季淘汰模拟 (Counterfactual Simulation)
# ============================================================
def simulate_season_eliminations(fan_df, judge_df, season, method='rank'):
    """
    模拟单季淘汰过程
    
    method: 'rank' 或 'percent'
    返回: dict {contestant_name: simulated_placement}
    """
    # 获取本季数据
    season_fan = fan_df[fan_df['season'] == season].copy()
    season_judge = judge_df[judge_df['season'] == season].copy()
    
    if season_fan.empty or season_judge.empty:
        return {}
    
    # 获取所有选手
    all_contestants = set(season_fan['celebrity_name'].unique()) & \
                     set(season_judge['celebrity_name'].unique())
    
    if not all_contestants:
        return {}
    
    # 初始化：所有人都在
    remaining = set(all_contestants)
    elimination_order = []  # 按淘汰顺序记录
    
    # 获取最大周数
    max_week = max(season_fan['week'].max(), season_judge['week'].max())
    
    for week in range(1, int(max_week) + 1):
        if len(remaining) <= 1:
            break
        
        # 本周数据
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
        
        # 合并
        merged = pd.merge(
            week_fan[['celebrity_name', 'fan_vote_share']],
            week_judge[['celebrity_name', 'judge_total']],
            on='celebrity_name'
        )
        
        if len(merged) < 2:
            continue
        
        # 计算 judge percent
        total_judge = merged['judge_total'].sum()
        merged['judge_percent'] = merged['judge_total'] / total_judge if total_judge > 0 else 0
        
        # 计算排名/得分
        if method == 'rank':
            # RANK方法：名次相加（小更好）
            merged['rank_fan'] = merged['fan_vote_share'].rank(ascending=False, method='average')
            merged['rank_judge'] = merged['judge_total'].rank(ascending=False, method='average')
            merged['combined'] = merged['rank_fan'] + merged['rank_judge']
            # 最大combined = 最后名 = 被淘汰
            eliminated_name = merged.loc[merged['combined'].idxmax(), 'celebrity_name']
        else:
            # PERCENT方法：百分比相加（大更好）
            merged['combined'] = merged['fan_vote_share'] + merged['judge_percent']
            # 最小combined = 最后名 = 被淘汰
            eliminated_name = merged.loc[merged['combined'].idxmin(), 'celebrity_name']
        
        # 记录淘汰
        elimination_order.append(eliminated_name)
        remaining.remove(eliminated_name)
    
    # 剩余的按实际顺序排（如果还有多人）
    remaining_list = list(remaining)
    
    # 构建placement字典
    # 先淘汰的placement大，后淘汰的小，剩下的最小
    n_total = len(all_contestants)
    placements = {}
    
    # 被淘汰的
    for i, name in enumerate(elimination_order):
        placements[name] = n_total - i
    
    # 剩下的（并列第一）
    if remaining_list:
        for i, name in enumerate(remaining_list):
            placements[name] = len(remaining_list) - i
    
    return placements


def run_all_simulations(fan_df, judge_df):
    """对所有季进行双方法模拟"""
    print("\n" + "=" * 70)
    print("STEP 2: Running Counterfactual Simulations")
    print("=" * 70)
    
    all_seasons = sorted(fan_df['season'].unique())
    
    results = []
    
    for season in all_seasons:
        # 模拟两种方法
        placements_rank = simulate_season_eliminations(fan_df, judge_df, season, 'rank')
        placements_percent = simulate_season_eliminations(fan_df, judge_df, season, 'percent')
        
        # 计算每个选手的delta
        season_fan = fan_df[fan_df['season'] == season]
        season_judge = judge_df[judge_df['season'] == season]
        
        for name in placements_rank.keys():
            if name not in placements_percent:
                continue
            
            # 计算该选手的平均delta
            contestant_fan = season_fan[season_fan['celebrity_name'] == name]
            contestant_judge = season_judge[season_judge['celebrity_name'] == name]
            
            deltas = []
            for week in contestant_fan['week'].unique():
                wf = contestant_fan[contestant_fan['week'] == week]
                wj = contestant_judge[contestant_judge['week'] == week]
                
                if wf.empty or wj.empty:
                    continue
                
                # 该周所有选手的judge总分
                week_all_judge = season_judge[season_judge['week'] == week]
                total_judge = week_all_judge['judge_total'].sum()
                
                if total_judge > 0:
                    fan_share = wf['fan_vote_share'].values[0]
                    judge_percent = wj['judge_total'].values[0] / total_judge
                    deltas.append(fan_share - judge_percent)
            
            if deltas:
                mean_delta = np.mean(deltas)
                
                # 获取实际placement
                actual_place = contestant_judge['actual_placement'].values[0] \
                              if 'actual_placement' in contestant_judge.columns else None
                
                results.append({
                    'season': season,
                    'name': name,
                    'mean_delta': mean_delta,
                    'placement_rank': placements_rank[name],
                    'placement_percent': placements_percent[name],
                    'Y_diff': placements_percent[name] - placements_rank[name],
                    'actual_placement': actual_place,
                })
    
    df_results = pd.DataFrame(results)
    
    # 使用百分位数分类（更合理）
    q20 = df_results['mean_delta'].quantile(0.20)
    q80 = df_results['mean_delta'].quantile(0.80)
    
    df_results['category'] = pd.cut(
        df_results['mean_delta'],
        bins=[-np.inf, q20, q80, np.inf],
        labels=['Judge-Favored', 'Neutral', 'Fan-Favored']
    )
    
    print(f"  Simulated {len(df_results)} contestant-season pairs")
    print(f"  Delta quantiles: Q20={q20:.4f}, Q80={q80:.4f}")
    print(f"  Categories: {df_results['category'].value_counts().to_dict()}")
    
    return df_results


# ============================================================
# STEP 3: 可视化 - 主曲线
# ============================================================
def plot_main_curve(df_results, save_dir):
    """绘制核心机制曲线（大幅优化版）"""
    print("\n" + "=" * 70)
    print("STEP 3: Generating Main Visualization")
    print("=" * 70)
    
    fig = plt.figure(figsize=(16, 11), facecolor='white')
    gs = GridSpec(2, 3, figure=fig, 
                  height_ratios=[2.5, 1], 
                  width_ratios=[1, 1, 1],
                  hspace=0.35, wspace=0.25)
    
    # ========================================
    # (A) 主散点图 + 趋势线
    # ========================================
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor(PALETTE['bg'])
    
    # 绘制分区背景（使用动态百分位阈值）
    x_min, x_max = df_results['mean_delta'].min() - 0.01, df_results['mean_delta'].max() + 0.01
    q20 = df_results['mean_delta'].quantile(0.20)
    q80 = df_results['mean_delta'].quantile(0.80)
    
    # Judge-favored区 (Δ < Q20)
    ax_main.axvspan(x_min, q20, alpha=0.12, color=PALETTE['zone_judge'], zorder=0)
    # Neutral区 (Q20 < Δ < Q80)
    ax_main.axvspan(q20, q80, alpha=0.06, color=PALETTE['zone_neutral'], zorder=0)
    # Fan-favored区 (Δ > Q80)
    ax_main.axvspan(q80, x_max, alpha=0.12, color=PALETTE['zone_fan'], zorder=0)
    
    # 绘制散点（按类别着色）
    color_map = {
        'Judge-Favored': PALETTE['dot_judge'],
        'Neutral': PALETTE['dot_neutral'],
        'Fan-Favored': PALETTE['dot_fan']
    }
    
    for cat, color in color_map.items():
        subset = df_results[df_results['category'] == cat]
        ax_main.scatter(
            subset['mean_delta'], subset['Y_diff'],
            s=80, alpha=0.65, c=color, edgecolors='white', 
            linewidths=0.8, label=f'{cat} (n={len(subset)})', zorder=3
        )
    
    # 分箱趋势线
    n_bins = 20
    df_results['delta_bin'] = pd.cut(df_results['mean_delta'], bins=n_bins)
    binned = df_results.groupby('delta_bin', observed=True).agg({
        'mean_delta': 'mean',
        'Y_diff': ['mean', 'std', 'count']
    }).reset_index()
    binned.columns = ['bin', 'delta_mean', 'Y_mean', 'Y_std', 'count']
    binned = binned[binned['count'] >= 3].copy()
    
    # 标准误
    binned['Y_se'] = binned['Y_std'] / np.sqrt(binned['count'])
    binned['Y_ci_lower'] = binned['Y_mean'] - 1.96 * binned['Y_se']
    binned['Y_ci_upper'] = binned['Y_mean'] + 1.96 * binned['Y_se']
    
    # 绘制趋势线与置信带
    if len(binned) > 3:
        # 平滑
        x_smooth = binned['delta_mean'].values
        y_smooth = gaussian_filter1d(binned['Y_mean'].values, sigma=1.2)
        
        ax_main.fill_between(
            x_smooth, 
            gaussian_filter1d(binned['Y_ci_lower'].values, sigma=1),
            gaussian_filter1d(binned['Y_ci_upper'].values, sigma=1),
            alpha=0.2, color=PALETTE['primary'], zorder=2
        )
        
        ax_main.plot(
            x_smooth, y_smooth, '-', 
            linewidth=4, color=PALETTE['primary'],
            zorder=4, label='Binned Trend (smoothed)'
        )
    
    # 参考线（使用动态阈值）
    ax_main.axhline(y=0, color=PALETTE['text_dark'], linestyle='-', linewidth=2, alpha=0.8, zorder=2)
    ax_main.axvline(x=0, color=PALETTE['text_light'], linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
    ax_main.axvline(x=q20, color=PALETTE['dot_judge'], linestyle=':', linewidth=1.2, alpha=0.5, zorder=1)
    ax_main.axvline(x=q80, color=PALETTE['dot_fan'], linestyle=':', linewidth=1.2, alpha=0.5, zorder=1)
    
    # 线性回归
    slope, intercept, r_value, p_value, _ = linregress(
        df_results['mean_delta'], df_results['Y_diff']
    )
    rho, p_spearman = spearmanr(df_results['mean_delta'], df_results['Y_diff'])
    
    # 回归线
    x_reg = np.array([x_min, x_max])
    y_reg = slope * x_reg + intercept
    ax_main.plot(x_reg, y_reg, '--', linewidth=2.5, color=PALETTE['secondary'], 
                alpha=0.8, zorder=4, label=f'Linear Fit (R={r_value:.3f})')
    
    # 标注统计信息框
    stats_text = (
        f"Linear Regression:\n"
        f"  Y = {slope:.2f} * Delta + {intercept:.2f}\n"
        f"  R = {r_value:.3f}, p = {p_value:.2e}\n\n"
        f"Spearman rho = {rho:.3f}, p = {p_spearman:.2e}"
    )
    
    text_box = ax_main.text(
        0.02, 0.98, stats_text,
        transform=ax_main.transAxes, ha='left', va='top',
        fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor=PALETTE['grid'], linewidth=1.5, alpha=0.95)
    )
    
    # 区域标签（动态位置）
    y_lim = ax_main.get_ylim()
    y_label_pos = y_lim[1] - (y_lim[1] - y_lim[0]) * 0.08
    
    judge_center = (x_min + q20) / 2
    neutral_center = (q20 + q80) / 2  
    fan_center = (q80 + x_max) / 2
    
    ax_main.text(judge_center, y_label_pos, 'JUDGE\nFAVORED', ha='center', va='top',
                fontsize=9, fontweight='bold', color=PALETTE['dot_judge'], alpha=0.7)
    ax_main.text(neutral_center, y_label_pos, 'NEUTRAL\n(60%)', ha='center', va='top',
                fontsize=9, fontweight='bold', color=PALETTE['text_light'], alpha=0.7)
    ax_main.text(fan_center, y_label_pos, 'FAN\nFAVORED', ha='center', va='top',
                fontsize=9, fontweight='bold', color=PALETTE['dot_fan'], alpha=0.7)
    
    ax_main.set_xlabel(
        r'$\Delta$ = Mean(Fan Share - Judge Percent)' + '\n' +
        '(Negative = Judge-Favored, Positive = Fan-Favored)',
        fontsize=11, fontweight='bold', labelpad=10
    )
    ax_main.set_ylabel(
        'Y = Placement(PERCENT) - Placement(RANK)\n' +
        '(Positive = PERCENT ranks WORSE, Negative = PERCENT ranks BETTER)',
        fontsize=11, fontweight='bold', labelpad=10
    )
    ax_main.set_title(
        'Unified Mechanism Curve: PERCENT vs RANK Effect on Final Placement\n' +
        '(Counterfactual Simulation Across All Seasons)',
        fontsize=14, fontweight='bold', pad=15, color=PALETTE['text_dark']
    )
    
    ax_main.legend(loc='upper right', fontsize=9, framealpha=0.95,
                  edgecolor=PALETTE['grid'], fancybox=True)
    ax_main.grid(alpha=0.3, linestyle='-', linewidth=0.8, color=PALETTE['grid'])
    
    # ========================================
    # (B) 分类箱线图
    # ========================================
    ax_box = fig.add_subplot(gs[1, 0])
    ax_box.set_facecolor(PALETTE['bg'])
    
    categories = ['Judge-Favored', 'Neutral', 'Fan-Favored']
    box_colors = [PALETTE['dot_judge'], PALETTE['dot_neutral'], PALETTE['dot_fan']]
    
    box_data = [df_results[df_results['category'] == cat]['Y_diff'].values for cat in categories]
    
    bp = ax_box.boxplot(
        box_data, positions=[1, 2, 3], widths=0.6,
        patch_artist=True, showfliers=False
    )
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    
    for whisker in bp['whiskers']:
        whisker.set_color(PALETTE['text_light'])
    for cap in bp['caps']:
        cap.set_color(PALETTE['text_light'])
    
    # 添加均值点
    means = [np.mean(d) for d in box_data]
    ax_box.scatter([1, 2, 3], means, s=80, color='white', edgecolors='black', 
                  linewidths=1.5, zorder=5, marker='D')
    
    # 标注均值
    for i, (cat, m) in enumerate(zip(categories, means)):
        ax_box.annotate(f'{m:+.2f}', xy=(i+1, m), xytext=(0, 10),
                       textcoords='offset points', ha='center', fontsize=9,
                       fontweight='bold', color=box_colors[i])
    
    ax_box.axhline(y=0, color=PALETTE['text_dark'], linestyle='-', linewidth=1.5, alpha=0.7)
    ax_box.set_xticks([1, 2, 3])
    ax_box.set_xticklabels(['Judge\nFavored', 'Neutral', 'Fan\nFavored'], fontsize=9)
    ax_box.set_ylabel('Y (Placement Diff)', fontsize=10, fontweight='bold')
    ax_box.set_title('(B) Effect by Category', fontsize=11, fontweight='bold', pad=10)
    ax_box.grid(axis='y', alpha=0.3, linestyle='-', color=PALETTE['grid'])
    
    # ========================================
    # (C) Delta分布直方图
    # ========================================
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.set_facecolor(PALETTE['bg'])
    
    # 分层直方图
    for cat, color in zip(categories, box_colors):
        subset = df_results[df_results['category'] == cat]['mean_delta']
        ax_hist.hist(subset, bins=20, alpha=0.5, color=color, 
                    edgecolor='white', linewidth=0.8, label=cat)
    
    ax_hist.axvline(x=0, color=PALETTE['text_dark'], linestyle='-', linewidth=1.5, alpha=0.7)
    ax_hist.axvline(x=df_results['mean_delta'].mean(), color=PALETTE['secondary'],
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Mean: {df_results["mean_delta"].mean():.4f}')
    
    ax_hist.set_xlabel(r'$\Delta$ (Fan - Judge)', fontsize=10, fontweight='bold')
    ax_hist.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax_hist.set_title(r'(C) Distribution of $\Delta$', fontsize=11, fontweight='bold', pad=10)
    ax_hist.legend(fontsize=8, loc='upper right')
    ax_hist.grid(alpha=0.3, linestyle='-', color=PALETTE['grid'])
    
    # ========================================
    # (D) Y分布直方图
    # ========================================
    ax_hist_y = fig.add_subplot(gs[1, 2])
    ax_hist_y.set_facecolor(PALETTE['bg'])
    
    ax_hist_y.hist(df_results['Y_diff'], bins=25, alpha=0.6,
                  color=PALETTE['primary'], edgecolor='white', linewidth=0.8)
    
    ax_hist_y.axvline(x=0, color=PALETTE['text_dark'], linestyle='-', linewidth=1.5, alpha=0.7)
    ax_hist_y.axvline(x=df_results['Y_diff'].mean(), color=PALETTE['secondary'],
                     linestyle='--', linewidth=2, alpha=0.8,
                     label=f'Mean: {df_results["Y_diff"].mean():.3f}')
    
    # 标注正负比例
    n_positive = (df_results['Y_diff'] > 0).sum()
    n_negative = (df_results['Y_diff'] < 0).sum()
    n_zero = (df_results['Y_diff'] == 0).sum()
    total = len(df_results)
    
    ax_hist_y.text(0.98, 0.95, 
                  f'Y > 0: {n_positive} ({n_positive/total*100:.1f}%)\n'
                  f'Y = 0: {n_zero} ({n_zero/total*100:.1f}%)\n'
                  f'Y < 0: {n_negative} ({n_negative/total*100:.1f}%)',
                  transform=ax_hist_y.transAxes, ha='right', va='top',
                  fontsize=9, family='monospace',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                           edgecolor=PALETTE['grid'], alpha=0.9))
    
    ax_hist_y.set_xlabel('Y (PERCENT - RANK Placement)', fontsize=10, fontweight='bold')
    ax_hist_y.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax_hist_y.set_title('(D) Distribution of Y', fontsize=11, fontweight='bold', pad=10)
    ax_hist_y.legend(fontsize=9, loc='upper left')
    ax_hist_y.grid(alpha=0.3, linestyle='-', color=PALETTE['grid'])
    
    plt.savefig(f'{save_dir}/Task2_1_unified_mechanism_curve.png', dpi=300, facecolor='white')
    plt.close()
    print("  [1/2] Saved: Task2_1_unified_mechanism_curve.png")
    
    return slope, intercept, r_value, p_value, rho


# ============================================================
# STEP 4: 补充可视化 - 热力图 + 3D
# ============================================================
def plot_supplementary(df_results, save_dir):
    """补充可视化：2D密度 + 分季热力图"""
    print("  [2/2] Generating supplementary visualizations...")
    
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)
    
    # ========================================
    # (A) 2D核密度估计
    # ========================================
    ax_kde = fig.add_subplot(gs[0, 0])
    ax_kde.set_facecolor(PALETTE['bg'])
    
    from scipy.stats import gaussian_kde
    
    x = df_results['mean_delta'].values
    y = df_results['Y_diff'].values
    
    # KDE
    xy = np.vstack([x, y])
    try:
        kde = gaussian_kde(xy)
        
        x_grid = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
        y_grid = np.linspace(y.min() - 1, y.max() + 1, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        # 自定义colormap
        colors_cmap = ['#FAFBFC', '#B7C9F2', '#569DAA', '#2D3436']
        custom_cmap = LinearSegmentedColormap.from_list('custom', colors_cmap)
        
        cf = ax_kde.contourf(X, Y, Z, levels=15, cmap=custom_cmap, alpha=0.85)
        ax_kde.contour(X, Y, Z, levels=8, colors='white', linewidths=0.6, alpha=0.5)
        
        plt.colorbar(cf, ax=ax_kde, shrink=0.8, label='Density')
    except:
        ax_kde.scatter(x, y, s=50, alpha=0.5, c=PALETTE['primary'])
    
    # 散点叠加
    ax_kde.scatter(x, y, s=25, alpha=0.3, c='white', edgecolors=PALETTE['text_dark'], linewidths=0.5)
    
    # 参考线
    ax_kde.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.8)
    ax_kde.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax_kde.set_xlabel(r'$\Delta$ = Mean(Fan - Judge)', fontsize=11, fontweight='bold')
    ax_kde.set_ylabel('Y = Placement Difference', fontsize=11, fontweight='bold')
    ax_kde.set_title('(A) 2D Density Estimation', fontsize=13, fontweight='bold', pad=15)
    ax_kde.grid(alpha=0.2, linestyle='-', color=PALETTE['grid'])
    
    # ========================================
    # (B) 极端案例标注
    # ========================================
    ax_cases = fig.add_subplot(gs[0, 1])
    ax_cases.set_facecolor(PALETTE['bg'])
    
    # 找出Y_diff绝对值最大的前20个案例
    df_results['Y_abs'] = df_results['Y_diff'].abs()
    extreme_cases = df_results.nlargest(20, 'Y_abs').copy()
    extreme_cases = extreme_cases.sort_values('Y_diff', ascending=True)
    
    # 绘制水平条形图
    colors = [PALETTE['dot_fan'] if y < 0 else PALETTE['dot_judge'] for y in extreme_cases['Y_diff']]
    
    y_pos = range(len(extreme_cases))
    bars = ax_cases.barh(
        y_pos, 
        extreme_cases['Y_diff'],
        color=colors, alpha=0.75, edgecolor='white', linewidth=0.8, height=0.7
    )
    
    ax_cases.axvline(x=0, color=PALETTE['text_dark'], linestyle='-', linewidth=2)
    
    # 标签
    labels = [f"{row['name'][:15]} (S{int(row['season'])})" for _, row in extreme_cases.iterrows()]
    ax_cases.set_yticks(y_pos)
    ax_cases.set_yticklabels(labels, fontsize=8)
    
    ax_cases.set_xlabel('Y (PERCENT - RANK Placement)', fontsize=10, fontweight='bold')
    ax_cases.set_title('(B) Top 20 Most Affected Cases\n(Left = PERCENT Helps, Right = PERCENT Hurts)', 
                      fontsize=11, fontweight='bold', pad=10)
    ax_cases.grid(axis='x', alpha=0.3, linestyle='-', color=PALETTE['grid'])
    
    # 标注数值
    for i, (idx, row) in enumerate(extreme_cases.iterrows()):
        x_pos = row['Y_diff']
        offset = 0.3 if x_pos >= 0 else -0.3
        ha = 'left' if x_pos >= 0 else 'right'
        ax_cases.text(x_pos + offset, i, f'{x_pos:+.0f}', 
                    va='center', ha=ha, fontsize=8, fontweight='bold',
                    color=PALETTE['text_dark'])
    
    plt.suptitle(
        'Supplementary Analysis: Density & Seasonal Patterns',
        fontsize=15, fontweight='bold', y=1.02, color=PALETTE['text_dark']
    )
    
    plt.savefig(f'{save_dir}/Task2_1_unified_supplementary.png', dpi=300, facecolor='white')
    plt.close()
    print("  [2/2] Saved: Task2_1_unified_supplementary.png")


# ============================================================
# STEP 5: 统计总结与解释
# ============================================================
def statistical_summary(df_results, slope, intercept, r_value, p_value, rho, table_dir):
    """统计总结与发现解读"""
    print("\n" + "=" * 70)
    print("STEP 4: Statistical Summary")
    print("=" * 70)
    
    # 分类统计
    categories = ['Judge-Favored', 'Neutral', 'Fan-Favored']
    
    print("\n  Regional Analysis:")
    summary_data = []
    
    for cat in categories:
        subset = df_results[df_results['category'] == cat]
        n = len(subset)
        mean_y = subset['Y_diff'].mean()
        std_y = subset['Y_diff'].std()
        median_y = subset['Y_diff'].median()
        pct_positive = (subset['Y_diff'] > 0).mean() * 100
        
        print(f"\n    {cat} (n={n}):")
        print(f"      Mean Y: {mean_y:+.3f} (std: {std_y:.3f})")
        print(f"      Median Y: {median_y:+.3f}")
        print(f"      Percent Y > 0: {pct_positive:.1f}%")
        
        summary_data.append({
            'category': cat,
            'n': n,
            'mean_Y': mean_y,
            'std_Y': std_y,
            'median_Y': median_y,
            'pct_positive': pct_positive
        })
    
    # 回归分析
    print(f"\n  Regression Analysis:")
    print(f"    Y = {slope:.3f} * Delta + {intercept:.3f}")
    print(f"    R = {r_value:.4f}, R^2 = {r_value**2:.4f}")
    print(f"    p-value = {p_value:.2e}")
    print(f"    Spearman rho = {rho:.4f}")
    
    # 零点
    if slope != 0:
        zero_crossing = -intercept / slope
        print(f"    Critical Point (Y=0): Delta ~ {zero_crossing:.4f}")
    
    # 保存
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{table_dir}/unified_mechanism_summary.csv', index=False)
    
    # 保存详细数据
    df_results.to_csv(f'{table_dir}/unified_mechanism_detailed.csv', index=False)
    
    print(f"\n  Saved summary to: unified_mechanism_summary.csv")
    print(f"  Saved detailed data to: unified_mechanism_detailed.csv")
    
    return summary_df


def interpret_results(df_results, slope, r_value):
    """结果解读"""
    print("\n" + "=" * 70)
    print("KEY FINDINGS & INTERPRETATION")
    print("=" * 70)
    
    # 判断曲线方向
    if slope > 0.5 and r_value > 0.1:
        interpretation = "CONFIRMED"
        detail = (
            "The positive slope indicates that PERCENT method systematically\n"
            "penalizes extreme fan-favored contestants (higher Delta -> higher Y).\n"
            "This confirms the 'controversy suppression' hypothesis."
        )
    elif slope < -0.5 and r_value < -0.1:
        interpretation = "REVERSED"
        detail = (
            "The negative slope suggests PERCENT actually HELPS fan-favored\n"
            "contestants achieve better placements, contradicting the\n"
            "'suppression' hypothesis. PERCENT amplifies fan advantage."
        )
    else:
        interpretation = "MIXED/WEAK"
        detail = (
            "The relationship is weak or mixed. PERCENT's effect may depend\n"
            "on other factors not captured in this simple model."
        )
    
    print(f"\n  Hypothesis Status: {interpretation}")
    print(f"\n  {detail}")
    
    # 分区效应
    judge_favored = df_results[df_results['category'] == 'Judge-Favored']['Y_diff'].mean()
    fan_favored = df_results[df_results['category'] == 'Fan-Favored']['Y_diff'].mean()
    
    print(f"\n  Category Effects:")
    print(f"    Judge-Favored: Mean Y = {judge_favored:+.3f}")
    print(f"      -> {'PERCENT helps' if judge_favored < 0 else 'PERCENT hurts'} these contestants")
    print(f"    Fan-Favored: Mean Y = {fan_favored:+.3f}")
    print(f"      -> {'PERCENT suppresses' if fan_favored > 0 else 'PERCENT helps'} these contestants")
    
    # 核心发现
    print("\n" + "-" * 70)
    print("UNIFIED EXPLANATION:")
    print("-" * 70)
    
    if slope > 0:
        print("""
  The curve reconciles two seemingly contradictory findings:
  
  1. FFI shows PERCENT 'favors fans' (+0.093 Kendall distance)
     -> This is a WEEKLY ranking metric; PERCENT gives better RANKS
        to fan-favored contestants in most individual weeks.
  
  2. This curve shows PERCENT 'suppresses controversy'
     -> Over a FULL SEASON simulation, extreme fan-favored contestants
        end up with WORSE final placements under PERCENT.
  
  RESOLUTION: PERCENT gives better weekly ranks but SMALLER safety margins.
  Over many weeks, smaller margins accumulate into higher elimination risk.
  The 'favor' is illusory at the individual week level; the 'suppression'
  emerges at the season level through cumulative probability effects.
""")
    else:
        print("""
  The curve shows PERCENT CONSISTENTLY helps fan-favored contestants:
  
  - Weekly: Better ranks (confirmed by FFI)
  - Seasonal: Better final placements (Y < 0 for high Delta)
  
  This suggests the 'suppression' effect observed in specific cases
  (e.g., Bobby Bones) may be anomalous rather than systematic.
  The aggregate data shows PERCENT amplifies fan advantage at ALL levels.
""")


# ============================================================
# MAIN
# ============================================================
def main():
    repo_root = Path(__file__).resolve().parents[2]
    figure_dir = repo_root / "task2.1" / "figure"
    table_dir = repo_root / "task2.1" / "table"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("UNIFIED MECHANISM ANALYSIS (CORRECTED VERSION)")
    print("Using Full-Season Counterfactual Simulation")
    print("=" * 70)
    
    # Step 1: 加载数据
    fan_df, judge_df, data_df = load_all_data(str(repo_root))
    
    # Step 2: 运行模拟
    df_results = run_all_simulations(fan_df, judge_df)
    
    # Step 3: 主可视化
    slope, intercept, r_value, p_value, rho = plot_main_curve(df_results, str(figure_dir))
    
    # Step 4: 补充可视化
    plot_supplementary(df_results, str(figure_dir))
    
    # Step 5: 统计总结
    summary = statistical_summary(df_results, slope, intercept, r_value, p_value, rho, str(table_dir))
    
    # Step 6: 解读
    interpret_results(df_results, slope, r_value)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
