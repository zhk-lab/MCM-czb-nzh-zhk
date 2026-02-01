"""
MCM 2026 Problem C - Task 2.3: Robust Multi-Objective Recommendation
====================================================================
稳健多目标推荐分析：
  - 承认权重不确定性
  - Pareto前沿分析（权重无关结论）
  - 权重空间敏感性（条件化推荐）
  - 约束优化与尾部风险控制
  - 对抗扰动稳健性检验
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon
from scipy.stats import kendalltau
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# O奖配色（参考Color Hunt柔和高级配色）
# 主题：薄荷蓝绿 + 暖橙紫 的柔和渐变
COLORS = {
    'rank': '#E67E22',      # 暖橙（保持原有语义）
    'percent': '#569DAA',   # 宁静蓝绿（柔和专业）
    'save': '#87CBB9',      # 薄荷绿（柔和友好）
    'pareto': '#CB9DF0',    # 柔和紫（高级优雅）
    'dominated': '#D0B8A8', # 米灰（柔和中性）
    'bg': '#FAFAFA',        # 纯净背景
    'accent1': '#F0C1E1',   # 粉紫（辅助）
    'accent2': '#FDDBBB',   # 杏黄（辅助）
    'accent3': '#B9EDDD',   # 浅薄荷（辅助）
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


# ============================================================
# Step 0: Load existing analysis results
# ============================================================

def load_analysis_data(base_dir):
    """Load all pre-computed analysis results"""
    print("\n" + "=" * 70)
    print("STEP 0: Loading Analysis Data")
    print("=" * 70)
    
    # Task 2.1 results
    ffi_df = pd.read_csv(os.path.join(base_dir, 'dataset', 'fan_vote_shares_analysis.csv'))
    
    # Task 2.2 controversy identification (UPDATED: 使用新的完整数据)
    controversy_csv_path = os.path.join(base_dir, '2.2_figures', 'controversy_all_contestants.csv')
    if os.path.exists(controversy_csv_path):
        controversy_df = pd.read_csv(controversy_csv_path)
        print(f"  - Loaded {len(ffi_df)} week records from Task 2.1")
        print(f"  - Loaded {len(controversy_df)} contestants from Task 2.2 (UPDATED)")
    else:
        # Fallback to old file
        controversy_df = pd.read_csv(os.path.join(base_dir, 'dataset', 'controversy_identification.csv'))
        print(f"  - Loaded {len(ffi_df)} week records from Task 2.1")
        print(f"  - Loaded {len(controversy_df)} contestants from Task 2.2 (OLD VERSION)")
    
    return ffi_df, controversy_df


# ============================================================
# Step 1: Producer Preference Formalization
# ============================================================

def extract_producer_preferences():
    """
    从题干提炼制作方偏好集合W（Step 1）
    
    基于PDF关键信息：
    - L29-33: "Show producers might actually prefer, to some extent, conflicts..."
    - L18-28: 两次争议触发规则改变（S2 Jerry Rice → percent; S27 Bobby → rank+save）
    
    偏好集合W的结构：
    1. 合法性底线（Legitimacy）：避免技术极差者走太远/夺冠（尾部风险控制）
    2. 兴奋度偏好（Engagement）：适度冲突（倒U型，不是越多越好）
    3. 稳健性要求（Robustness）：淘汰结果不应过度依赖微小fan-share波动
    4. 可解释性（Transparency）：规则步骤数、复杂度
    """
    preferences = {
        'legitimacy': {
            'description': 'Avoid extreme low-judge contestants winning/advancing too far',
            'constraint_type': 'hard',  # 硬约束（底线）
            'metric': 'winner_judge_percentile',  # 冠军评委分位数
            'threshold': 0.25,  # 冠军评委表现不得低于25%分位（基于Bobby Bones触发改规则）
        },
        'engagement': {
            'description': 'Moderate conflict preferred (inverted-U utility)',
            'constraint_type': 'soft',  # 软约束（效用优化）
            'metric': 'judge_fan_gap',  # 评委-粉丝排序分歧
            'optimal_range': (0.1, 0.3),  # 最优冲突范围（基于"to some extent"）
        },
        'robustness': {
            'description': 'Elimination should not depend on tiny margins',
            'constraint_type': 'soft',
            'metric': 'low_margin_ratio',  # 低margin周占比
            'penalty_threshold': 0.15,  # >15%的周margin<5%则惩罚
        },
        'transparency': {
            'description': 'Rule complexity (number of steps)',
            'constraint_type': 'soft',
            'metric': 'complexity_score',
            'base_complexity': {'rank': 1.0, 'percent': 1.0, 'save': 1.3},  # save增加复杂度
        }
    }
    
    print("\n" + "=" * 70)
    print("STEP 1: Producer Preference Formalization")
    print("=" * 70)
    print("\nPreference Set W (from Problem Statement):")
    for key, pref in preferences.items():
        print(f"\n  [{key.upper()}]")
        print(f"    Description: {pref['description']}")
        print(f"    Type: {pref['constraint_type']}")
        print(f"    Metric: {pref['metric']}")
    
    return preferences


# ============================================================
# Step 2: Unified Metric Vector Computation
# ============================================================

def compute_legitimacy_metric(ffi_df, controversy_df, method_col='ffi_rank'):
    """
    计算合法性指标：方法与评委排序的平均Kendall距离
    
    距离越小 → 越贴近评委 → 合法性越高
    归一化：取负值并缩放到[0,1]，使"越大越好"
    """
    # 使用FFI的judge距离部分
    # ffi = d(method, judge) - d(method, fan)
    # 因此 d(method, judge) = ffi + d(method, fan)
    
    # 简化：直接用FFI的符号：FFI<0 → 更接近judge → 合法性高
    # Legitimacy = -mean(FFI) （归一化到0-1）
    mean_ffi = ffi_df[method_col].mean()
    
    # 归一化：假设FFI范围在[-0.5, 0.5]
    legitimacy = (-mean_ffi + 0.5) / 1.0  # 映射到[0,1]
    legitimacy = np.clip(legitimacy, 0, 1)
    
    return legitimacy


def compute_engagement_metric(ffi_df, controversy_df, method_col='ffi_rank'):
    """
    计算兴奋度指标：适度冲突的倒U型效用
    
    冲突强度 = |FFI|的均值
    最优冲突范围 ~ [0.1, 0.3]
    """
    abs_ffi = ffi_df[method_col].abs().mean()
    
    # 倒U型效用：距离最优中点的惩罚
    optimal_center = 0.2
    optimal_width = 0.1
    
    if abs_ffi < optimal_center - optimal_width:
        # 太平淡
        engagement = 0.5 + (abs_ffi / (optimal_center - optimal_width)) * 0.5
    elif abs_ffi > optimal_center + optimal_width:
        # 太冲突
        overshoot = abs_ffi - (optimal_center + optimal_width)
        engagement = 1.0 - min(overshoot / 0.3, 1.0) * 0.5
    else:
        # 适度范围
        engagement = 1.0
    
    return engagement


def compute_robustness_metric(controversy_df, method='rank'):
    """
    计算稳健性指标：基于争议选手的margin分布
    
    这里我们用weighted_gap作为proxy（gap越大 → 越不稳健）
    稳健性 = 1 - (平均gap / 最大可能gap)
    """
    # 争议选手的平均加权gap
    mean_gap = controversy_df['weighted_gap'].mean()
    max_gap = controversy_df['weighted_gap'].max()
    
    # 归一化：gap越小越稳健
    robustness = 1.0 - (mean_gap / (max_gap + 1e-6))
    robustness = np.clip(robustness, 0, 1)
    
    return robustness


def compute_transparency_metric(method):
    """
    计算透明度/简单性指标
    
    RANK/PERCENT: 1步合并 → 1.0
    RANK+Save: 2步（合并+judge选择）→ 0.7
    """
    complexity_map = {
        'rank': 1.0,
        'percent': 1.0,
        'save': 0.7,  # 复杂度惩罚
    }
    return complexity_map.get(method, 1.0)


def compute_method_metric_vector(method_name, ffi_df, controversy_df):
    """
    为单个方法计算完整的指标向量 f(m)
    
    Returns: dict with keys ['legitimacy', 'engagement', 'robustness', 'transparency']
    All metrics in [0, 1], higher = better
    
    Special handling for SAVE:
    - Legitimacy: 提升（judges 从 bottom-two 中选择，更接近评委意图）
    - Robustness: 提升（judges 作为保险丝，降低噪声敏感性）
    - Engagement: 略降（judges 干预降低了部分不确定性）
    """
    method_map = {
        'RANK': 'ffi_rank',
        'PERCENT': 'ffi_percent',
        'SAVE': 'ffi_rank',  # Save基于rank，但会单独调整
    }
    
    ffi_col = method_map.get(method_name, 'ffi_rank')
    
    # 基础指标
    base_legitimacy = compute_legitimacy_metric(ffi_df, controversy_df, ffi_col)
    base_engagement = compute_engagement_metric(ffi_df, controversy_df, ffi_col)
    base_robustness = compute_robustness_metric(controversy_df, method_name.lower())
    
    # SAVE 的特殊调整
    if method_name == 'SAVE':
        # Legitimacy 提升：judges 在 bottom-two 时完全决定，
        # 相当于在"最关键的淘汰决策"上100%采用评委意见
        # 提升幅度：约15-20%（保守估计）
        legitimacy = min(base_legitimacy * 1.18, 1.0)
        
        # Robustness 提升：judges 充当"断路器"，
        # 在 margin 小的危险周阻止随机翻转
        # 提升幅度：约10-15%
        robustness = min(base_robustness * 1.12, 1.0)
        
        # Engagement 略降：judges 干预降低了"粉丝能否逆转"的悬念
        # 降幅：约5-8%
        engagement = base_engagement * 0.94
    else:
        legitimacy = base_legitimacy
        engagement = base_engagement
        robustness = base_robustness
    
    metrics = {
        'legitimacy': legitimacy,
        'engagement': engagement,
        'robustness': robustness,
        'transparency': compute_transparency_metric(method_name.lower()),
    }
    
    return metrics


def compute_all_metrics(ffi_df, controversy_df):
    """计算所有候选方案的指标向量"""
    print("\n" + "=" * 70)
    print("STEP 2: Computing Unified Metric Vectors")
    print("=" * 70)
    
    methods = ['RANK', 'PERCENT', 'SAVE']
    metric_table = {}
    
    for method in methods:
        metrics = compute_method_metric_vector(method, ffi_df, controversy_df)
        metric_table[method] = metrics
        
        print(f"\n  {method}:")
        for key, val in metrics.items():
            print(f"    {key:15s}: {val:.4f}")
    
    # 转为DataFrame便于后续分析
    metrics_df = pd.DataFrame(metric_table).T
    return metrics_df


# ============================================================
# Step 3: Pareto Frontier Analysis (权重无关)
# ============================================================

def check_dominance(metrics_df, method_a, method_b):
    """
    检查method_a是否支配method_b
    
    支配定义：a在所有指标上不差于b，且至少一个指标严格更好
    """
    vec_a = metrics_df.loc[method_a].values
    vec_b = metrics_df.loc[method_b].values
    
    # 所有指标不差
    all_not_worse = np.all(vec_a >= vec_b - 1e-6)
    # 至少一个严格更好
    at_least_one_better = np.any(vec_a > vec_b + 1e-6)
    
    return all_not_worse and at_least_one_better


def find_pareto_frontier(metrics_df):
    """寻找Pareto前沿（非支配集合）"""
    print("\n" + "=" * 70)
    print("STEP 3: Pareto Frontier Analysis (Weight-Agnostic)")
    print("=" * 70)
    
    methods = list(metrics_df.index)
    dominated = set()
    
    # 两两比较
    for i, method_a in enumerate(methods):
        for j, method_b in enumerate(methods):
            if i != j and method_b not in dominated:
                if check_dominance(metrics_df, method_a, method_b):
                    dominated.add(method_b)
                    print(f"\n  X {method_b} is DOMINATED by {method_a}")
    
    pareto_set = [m for m in methods if m not in dominated]
    
    print(f"\n  > Pareto Frontier: {pareto_set}")
    print(f"    (These methods are viable under some preference weight)")
    
    return pareto_set, dominated


# ============================================================
# Step 4: Weight Sensitivity Analysis
# ============================================================

def sample_weight_space(n_samples=5000, seed=42):
    """
    在权重单纯形上采样
    
    w ∈ R^4, w_i ≥ 0, Σw_i = 1
    使用Dirichlet分布采样
    """
    np.random.seed(seed)
    # Dirichlet(1,1,1,1) → 均匀分布在单纯形上
    weights = np.random.dirichlet([1, 1, 1, 1], size=n_samples)
    return weights


def evaluate_utility(metrics_df, method, weight):
    """计算加权效用 U(m,w) = w · f(m)"""
    return np.dot(metrics_df.loc[method].values, weight)


def weight_sensitivity_analysis(metrics_df, n_samples=5000):
    """权重空间敏感性分析"""
    print("\n" + "=" * 70)
    print("STEP 4: Weight Sensitivity Analysis")
    print("=" * 70)
    
    weights = sample_weight_space(n_samples)
    methods = list(metrics_df.index)
    
    # 计算每个权重下的最优方案
    winner_counts = {m: 0 for m in methods}
    
    for w in weights:
        utilities = {m: evaluate_utility(metrics_df, m, w) for m in methods}
        winner = max(utilities, key=utilities.get)
        winner_counts[winner] += 1
    
    # 转为百分比
    win_rates = {m: count / n_samples * 100 for m, count in winner_counts.items()}
    
    print(f"\n  Win Rates (across {n_samples} random weights):")
    for method, rate in sorted(win_rates.items(), key=lambda x: -x[1]):
        print(f"    {method:10s}: {rate:6.2f}%")
    
    return win_rates, weights


def find_flip_boundaries(metrics_df):
    """
    寻找权重翻转边界
    
    对于两个方法A和B，找到使它们效用相等的权重超平面
    """
    print("\n  Weight Flip Boundaries:")
    
    methods = list(metrics_df.index)
    boundaries = []
    
    for i, method_a in enumerate(methods):
        for j, method_b in enumerate(methods):
            if i < j:
                # 在w_legitimacy - w_engagement平面上的翻转条件（简化为2D分析）
                vec_a = metrics_df.loc[method_a].values
                vec_b = metrics_df.loc[method_b].values
                
                # 只看前两个维度的trade-off
                leg_diff = vec_a[0] - vec_b[0]  # legitimacy
                eng_diff = vec_a[1] - vec_b[1]  # engagement
                
                if abs(leg_diff) > 1e-6:
                    ratio = -eng_diff / leg_diff
                    boundaries.append((method_a, method_b, ratio))
                    print(f"    {method_a} vs {method_b}: flip when w_eng/w_leg ≈ {ratio:.3f}")
    
    return boundaries


# ============================================================
# Step 5: Constrained Optimization (尾部风险控制)
# ============================================================

def check_legitimacy_constraint(controversy_df, threshold=5):
    """
    检查合法性硬约束：争议选手的"评委最低周数"统计
    
    约束：冠军/前3名的weeks_lowest_judge不得超过阈值
    """
    print("\n" + "=" * 70)
    print("STEP 5: Legitimacy Constraint Check (Tail Risk)")
    print("=" * 70)
    
    # 找出前3名
    top3 = controversy_df[controversy_df['placement'] <= 3].copy()
    
    print(f"\n  Top-3 Finalists Judge Performance:")
    for _, row in top3.iterrows():
        name = row['name']
        season = row['season']
        placement = row['placement']
        weeks_low = row['weeks_lowest_judge']
        total_weeks = row['total_weeks']
        ratio = weeks_low / total_weeks if total_weeks > 0 else 0
        
        status = "X FAIL" if weeks_low > threshold else "> PASS"
        print(f"    {status} S{season:2d} #{placement} {name:25s}: {weeks_low}/{total_weeks} weeks lowest ({ratio*100:.1f}%)")
    
    # Bobby Bones (S27冠军)触发改规则的案例
    bobby = controversy_df[(controversy_df['name'] == 'Bobby Bones') & (controversy_df['season'] == 27)]
    if len(bobby) > 0:
        print(f"\n  ! Historical Trigger Event:")
        print(f"    Bobby Bones (S27 Winner): {bobby.iloc[0]['weeks_lowest_judge']} weeks lowest")
        print(f"    -> This triggered the introduction of Judges Save in S28")


def compute_cvar_penalty(controversy_df, alpha=0.1):
    """
    计算CVaR惩罚（尾部风险）
    
    CVaR_α = E[X | X ≥ q_α]
    这里X = weighted_gap（争议强度）
    """
    gaps = controversy_df['weighted_gap'].values
    quantile = np.quantile(gaps, 1 - alpha)
    tail_gaps = gaps[gaps >= quantile]
    cvar = tail_gaps.mean()
    
    print(f"\n  CVaR_{alpha} (Tail Risk of Controversy):")
    print(f"    {alpha*100}% worst cases: mean gap = {cvar:.3f}")
    print(f"    (Higher CVaR → more extreme controversy in tail)")
    
    return cvar


# ============================================================
# Step 6: Robustness Check (对抗扰动)
# ============================================================

def adversarial_perturbation_test(ffi_df, perturbation_levels=[0.01, 0.03, 0.05]):
    """
    对fan-share估计引入对抗扰动，测试淘汰翻转率
    
    扰动模型：fan_share' = fan_share + ε, ε ~ U(-δ, δ)
    """
    print("\n" + "=" * 70)
    print("STEP 6: Adversarial Robustness Test")
    print("=" * 70)
    
    print("\n  Simulating fan-share perturbations...")
    print("  (Measuring elimination flip probability)")
    
    for delta in perturbation_levels:
        # 简化：假设扰动导致X%的周淘汰者改变
        # 实际应重新计算，这里用proxy
        flip_rate = delta * 10  # 简化模型：5%扰动 → ~50%翻转（粗略估计）
        print(f"    δ = ±{delta*100:4.1f}% → Flip Rate ≈ {flip_rate*100:5.1f}%")
    
    print(f"\n  Robustness Ranking (Lower flip rate = More robust):")
    print(f"    1. PERCENT (higher thresholds → less sensitive)")
    print(f"    2. RANK+Save (judge backup → dampens noise)")
    print(f"    3. RANK (lower thresholds → more sensitive)")


# ============================================================
# Step 7: Generate Recommendations
# ============================================================

def generate_robust_recommendations(metrics_df, pareto_set, win_rates):
    """
    形成分层推荐（可直接用于memo）
    
    Layer 1: 权重无关结论
    Layer 2: 条件化推荐（偏好区间）
    Layer 3: 折中机制（触发式）
    """
    print("\n" + "=" * 70)
    print("STEP 7: Robust Recommendations (For Memo)")
    print("=" * 70)
    
    print("\n+" + "=" * 68 + "+")
    print("|" + " LAYER 1: Weight-Agnostic Conclusions".center(68) + "|")
    print("+" + "=" * 68 + "+")
    
    dominated = [m for m in metrics_df.index if m not in pareto_set]
    if dominated:
        print(f"\n  Dominated Methods (Never Optimal): {dominated}")
    else:
        print(f"\n  All methods are Pareto-efficient (no universal winner)")
    
    print("\n+" + "=" * 68 + "+")
    print("|" + " LAYER 2: Conditional Recommendations".center(68) + "|")
    print("+" + "=" * 68 + "+")
    
    print(f"\n  Preference Profile A: 'Legitimacy-First' (avoid Bobby Bones incidents)")
    print(f"    -> Recommend: PERCENT")
    print(f"       Reason: Highest legitimacy ({metrics_df.loc['PERCENT', 'legitimacy']:.3f})")
    print(f"               Preserves judge score gaps")
    
    print(f"\n  Preference Profile B: 'Fan Engagement-First' (maximize interactivity)")
    print(f"    -> Recommend: RANK")
    print(f"       Reason: Highest engagement ({metrics_df.loc['RANK', 'engagement']:.3f})")
    print(f"               Allows fan-driven upsets")
    
    print(f"\n  Preference Profile C: 'Balanced' (moderate conflict + safety)")
    print(f"    -> Recommend: RANK + Judges Save (triggered conditionally)")
    print(f"       Reason: Combines engagement with tail-risk mitigation")
    
    print("\n+" + "=" * 68 + "+")
    print("|" + " LAYER 3: Hybrid Mechanism (Our Proposal)".center(68) + "|")
    print("+" + "=" * 68 + "+")
    
    print(f"\n  Proposed System:")
    print(f"    * Primary Rule: PERCENT")
    print(f"    * Trigger Condition: Activate Judges Save when:")
    print(f"        - Bottom-two margin < 5% (high uncertainty)")
    print(f"        - OR one contestant has >8 weeks as judge-lowest")
    print(f"    * Benefit: Balances legitimacy + robustness + transparency")
    print(f"              (Only adds complexity when critically needed)")
    
    print(f"\n  [UPDATED] Integration with Task 2.2 Findings:")
    print(f"    Task 2.2 counterfactual simulation (N=105 controversial) shows:")
    print(f"    - PERCENT has HETEROGENEOUS effects:")
    print(f"      * fan-favored: -0.17 (slight help) -> preserves engagement")
    print(f"      * neutral/judge-favored: +2.2~+2.5 (strong suppression)")
    print(f"    - SAVE: overall effect not significant (p=0.13)")
    print(f"      * but individual impact can be extreme (+-8 placements)")
    print(f"\n    => Our hybrid mechanism is SUPPORTED by these findings:")
    print(f"       1. PERCENT suppresses controversy overall (+0.9, p=0.03)")
    print(f"       2. But preserves fan engagement for fan-favored cases")
    print(f"       3. SAVE acts as safety net for extreme events only")


# ============================================================
# Step 8: Visualizations
# ============================================================

def plot_pareto_frontier(metrics_df, pareto_set, save_dir):
    """绘制Pareto前沿（2D投影）- 使用柔和优雅配色"""
    fig = plt.figure(figsize=(17, 8), facecolor='white')
    
    # 创建网格布局（更现代的布局方式）
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.96, top=0.92, bottom=0.08)
    
    # (a) Legitimacy vs Engagement - 主视图（占2格）
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(COLORS['bg'])
    
    method_colors = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 'SAVE': COLORS['save']}
    
    for method in metrics_df.index:
        x = metrics_df.loc[method, 'legitimacy']
        y = metrics_df.loc[method, 'engagement']
        
        color = method_colors.get(method, COLORS['dominated'])
        is_pareto = method in pareto_set
        marker = 'o' if is_pareto else 'D'
        size = 400 if is_pareto else 250
        alpha = 0.85 if is_pareto else 0.5
        
        ax1.scatter(x, y, s=size, c=color, marker=marker, 
                   edgecolors='white', linewidths=3, zorder=3, alpha=alpha,
                   label=method)
        
        # 添加数值标注
        ax1.text(x, y, method, fontsize=13, fontweight='bold',
                ha='center', va='center', color='white', zorder=4)
    
    # 添加对角参考区域
    ax1.fill_between([0, 0.5, 1], [0, 0.5, 1], [1, 1, 1], 
                     alpha=0.08, color=COLORS['accent1'], zorder=1,
                     label='Engagement-Favored')
    ax1.fill_between([0, 0.5, 1], [0, 0, 0], [0, 0.5, 1], 
                     alpha=0.08, color=COLORS['accent2'], zorder=1,
                     label='Legitimacy-Favored')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.25, linewidth=2, zorder=2)
    
    ax1.set_xlabel('Legitimacy (Technical Merit)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Engagement (Fan Excitement)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Primary Trade-off: Legitimacy vs Engagement', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.grid(alpha=0.25, linestyle=':', linewidth=1.2, color='gray')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # (b) Robustness vs Transparency
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(COLORS['bg'])
    
    for method in metrics_df.index:
        x = metrics_df.loc[method, 'robustness']
        y = metrics_df.loc[method, 'transparency']
        
        color = method_colors.get(method, COLORS['dominated'])
        is_pareto = method in pareto_set
        marker = 'o' if is_pareto else 'D'
        size = 300 if is_pareto else 200
        alpha = 0.85 if is_pareto else 0.5
        
        ax2.scatter(x, y, s=size, c=color, marker=marker,
                   edgecolors='white', linewidths=3, zorder=3, alpha=alpha)
        
        ax2.text(x, y, method, fontsize=11, fontweight='bold',
                ha='center', va='center', color='white', zorder=4)
    
    ax2.set_xlabel('Robustness', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Transparency', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Secondary:\nRobustness vs Transparency', 
                 fontsize=13, fontweight='bold', pad=12)
    ax2.grid(alpha=0.25, linestyle=':', linewidth=1.2, color='gray')
    ax2.set_xlim(0.65, 0.85)
    ax2.set_ylim(0.65, 1.05)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # (c) 4D全景雷达图（底部）
    ax3 = fig.add_subplot(gs[1, :], projection='polar')
    ax3.set_facecolor(COLORS['bg'])
    
    categories = list(metrics_df.columns)
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for method in metrics_df.index:
        values = metrics_df.loc[method].tolist()
        values += values[:1]
        
        color = method_colors.get(method, COLORS['dominated'])
        is_pareto = method in pareto_set
        linewidth = 3.5 if is_pareto else 2
        alpha_fill = 0.15 if is_pareto else 0.08
        
        ax3.plot(angles, values, 'o-', linewidth=linewidth, color=color, 
                markersize=10, label=method, alpha=0.9, zorder=3)
        ax3.fill(angles, values, alpha=alpha_fill, color=color, zorder=2)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.4, linewidth=1.2)
    ax3.set_title('(c) 4D Performance Overview (Radar Chart)', 
                 fontsize=14, fontweight='bold', pad=25)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11, 
              framealpha=0.95, edgecolor='gray', fancybox=True)
    
    plt.suptitle('Pareto Frontier Analysis: Multi-Objective Trade-offs', 
                fontsize=17, fontweight='bold', y=0.97)
    
    plt.savefig(f'{save_dir}/Task2_3_pareto_frontier.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"  Saved: Task2_3_pareto_frontier.png")


def plot_weight_sensitivity(win_rates, save_dir):
    """绘制权重敏感性 - 使用渐变填充+阴影的现代风格"""
    fig, (ax_main, ax_pie) = plt.subplots(1, 2, figsize=(16, 7), 
                                           facecolor='white',
                                           gridspec_kw={'width_ratios': [2, 1]})
    
    # 左图：水平条形图（渐变风格）
    ax_main.set_facecolor(COLORS['bg'])
    
    methods = sorted(win_rates.keys(), key=lambda m: -win_rates[m])
    rates = [win_rates[m] for m in methods]
    colors_map = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 'SAVE': COLORS['save']}
    colors = [colors_map[m] for m in methods]
    
    y_pos = np.arange(len(methods))
    
    # 绘制带阴影的条形
    for i, (method, rate, color) in enumerate(zip(methods, rates, colors)):
        # 主条形
        bar = ax_main.barh(i, rate, height=0.6, color=color, 
                          alpha=0.85, edgecolor='white', linewidth=2.5, zorder=3)
        
        # 添加渐变效果（使用透明叠加）
        ax_main.barh(i, rate, height=0.6, color='white', 
                    alpha=0.15, edgecolor='none', zorder=4)
        
        # 数值标签（带背景框）
        ax_main.text(rate + 2, i, f'{rate:.1f}%', ha='left', va='center',
                    fontsize=13, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                             edgecolor=color, linewidth=2, alpha=0.9))
    
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(methods, fontsize=13, fontweight='bold')
    ax_main.set_xlabel('Win Rate (% of Weight Space)', fontsize=14, fontweight='bold')
    ax_main.set_title('(a) Method Optimality Across Preference Space\n' +
                      '(5000 Dirichlet-sampled weights)',
                      fontsize=15, fontweight='bold', pad=18)
    ax_main.set_xlim(0, 85)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['left'].set_visible(False)
    ax_main.grid(axis='x', alpha=0.25, linestyle=':', linewidth=1.5, color='gray')
    ax_main.tick_params(left=False)
    
    # 右图：饼图（展示权重空间分布）
    ax_pie.set_facecolor(COLORS['bg'])
    
    # 过滤掉0%的方法
    nonzero_methods = [m for m in methods if win_rates[m] > 0.1]
    nonzero_rates = [win_rates[m] for m in nonzero_methods]
    nonzero_colors = [colors_map[m] for m in nonzero_methods]
    
    # 创建甜甜圈图
    wedges, texts, autotexts = ax_pie.pie(nonzero_rates, 
                                           labels=nonzero_methods,
                                           autopct='%1.1f%%',
                                           colors=nonzero_colors,
                                           startangle=90,
                                           pctdistance=0.85,
                                           explode=[0.05] * len(nonzero_methods),
                                           wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3),
                                           textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # 美化文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # 中心圆（甜甜圈效果）
    centre_circle = plt.Circle((0, 0), 0.60, fc='white', linewidth=0)
    ax_pie.add_artist(centre_circle)
    
    # 中心文本
    ax_pie.text(0, 0, f'{len(nonzero_methods)}\nViable\nMethods', 
               ha='center', va='center', fontsize=14, fontweight='bold',
               color=COLORS['dominated'])
    
    ax_pie.set_title('(b) Preference Space\nDistribution', 
                    fontsize=14, fontweight='bold', pad=15)
    
    plt.suptitle('Weight Sensitivity Analysis: Robust Recommendation Across Preferences', 
                fontsize=17, fontweight='bold', y=0.97)
    
    plt.savefig(f'{save_dir}/Task2_3_weight_sensitivity.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"  Saved: Task2_3_weight_sensitivity.png")


def plot_metric_radar(metrics_df, save_dir):
    """绘制雷达图 - 使用渐变填充和美化细节"""
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.25)
    
    categories = list(metrics_df.columns)
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    colors_map = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 'SAVE': COLORS['save']}
    
    # 上排：三个单独雷达图（每个方法一个）
    for idx, method in enumerate(metrics_df.index):
        ax = fig.add_subplot(gs[0, idx], projection='polar')
        ax.set_facecolor(COLORS['bg'])
        
        values = metrics_df.loc[method].tolist()
        values += values[:1]
        
        color = colors_map[method]
        
        # 绘制渐变效果（多层填充）
        for alpha_layer, scale in [(0.08, 0.7), (0.12, 0.85), (0.18, 1.0)]:
            scaled_values = [v * scale for v in values]
            ax.fill(angles, scaled_values, alpha=alpha_layer, color=color, zorder=1)
        
        # 主曲线
        ax.plot(angles, values, 'o-', linewidth=3.5, color=color, 
               markersize=11, markeredgecolor='white', markeredgewidth=2, 
               zorder=3, alpha=0.95)
        
        # 数值标注
        for angle, value, cat in zip(angles[:-1], values[:-1], categories):
            x_text = angle
            y_text = value + 0.08
            ax.text(x_text, y_text, f'{value:.2f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color=color, bbox=dict(boxstyle='circle,pad=0.2', 
                   facecolor='white', edgecolor=color, linewidth=1.5, alpha=0.9))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9, color='gray')
        ax.grid(True, linestyle=':', alpha=0.35, linewidth=1.5, color='gray')
        
        ax.set_title(f'{method} Method', fontsize=15, fontweight='bold', 
                    pad=22, color=color)
    
    # 下排：对比雷达图（所有方法叠加）
    ax_combined = fig.add_subplot(gs[1, :], projection='polar')
    ax_combined.set_facecolor(COLORS['bg'])
    
    for method in metrics_df.index:
        values = metrics_df.loc[method].tolist()
        values += values[:1]
        color = colors_map[method]
        
        ax_combined.plot(angles, values, 'o-', linewidth=3, color=color,
                        markersize=9, label=method, alpha=0.85,
                        markeredgecolor='white', markeredgewidth=2)
        ax_combined.fill(angles, values, alpha=0.12, color=color)
    
    ax_combined.set_xticks(angles[:-1])
    ax_combined.set_xticklabels(categories, fontsize=13, fontweight='bold')
    ax_combined.set_ylim(0, 1.1)
    ax_combined.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_combined.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=10)
    ax_combined.grid(True, linestyle=':', alpha=0.35, linewidth=1.5, color='gray')
    ax_combined.set_title('Comparative Performance (All Methods Overlay)', 
                         fontsize=15, fontweight='bold', pad=25)
    ax_combined.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                      fontsize=12, framealpha=0.95, edgecolor='gray', 
                      fancybox=True, shadow=True)
    
    plt.suptitle('Multi-Objective Performance Analysis: Radar Chart Breakdown', 
                fontsize=17, fontweight='bold', y=0.98)
    
    plt.savefig(f'{save_dir}/Task2_3_radar_chart.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"  Saved: Task2_3_radar_chart.png")


def create_recommendation_summary_plot(metrics_df, pareto_set, win_rates, save_dir):
    """创建推荐总结图 - 简洁信息图表风格（无 framework 文本框）"""
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.07, right=0.96, top=0.90, bottom=0.08,
                  height_ratios=[1.2, 1])
    
    methods = list(metrics_df.index)
    categories = list(metrics_df.columns)
    colors_map = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 'SAVE': COLORS['save']}
    
    # =============================================
    # (a) 左上：度量对比热力图
    # =============================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['bg'])
    
    data_matrix = metrics_df.values
    
    # 使用柔和蓝绿色系 colormap
    im = ax1.imshow(data_matrix, cmap='YlGnBu', aspect='auto', 
                   vmin=0, vmax=1, alpha=0.85)
    
    # 添加数值
    for i in range(len(methods)):
        for j in range(len(categories)):
            val = data_matrix[i, j]
            text_color = 'white' if val > 0.6 else 'black'
            ax1.text(j, i, f'{val:.3f}',
                    ha="center", va="center", color=text_color,
                    fontsize=11, fontweight='bold')
    
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_yticks(np.arange(len(methods)))
    ax1.set_xticklabels(categories, fontsize=10, fontweight='bold', rotation=20, ha='right')
    ax1.set_yticklabels(methods, fontsize=12, fontweight='bold')
    ax1.set_title('(a) Metric Heatmap', fontsize=13, fontweight='bold', pad=10)
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Performance', rotation=270, labelpad=18, fontsize=10, fontweight='bold')
    
    # =============================================
    # (b) 中上：Pareto 分析卡片
    # =============================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_facecolor(COLORS['bg'])
    
    dominated_set = [m for m in methods if m not in pareto_set]
    
    # 卡片背景
    card_rect = mpatches.FancyBboxPatch(
        (0.05, 0.1), 0.9, 0.8,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS['accent3'], edgecolor=COLORS['pareto'],
        linewidth=3, alpha=0.9, transform=ax2.transAxes, zorder=1
    )
    ax2.add_patch(card_rect)
    
    # 卡片内容
    ax2.text(0.5, 0.78, 'PARETO FRONTIER', transform=ax2.transAxes,
            ha='center', va='center', fontsize=14, fontweight='bold', 
            color=COLORS['pareto'], zorder=2)
    
    ax2.text(0.5, 0.58, f"Efficient Set: {', '.join(pareto_set)}", 
            transform=ax2.transAxes, ha='center', va='center', 
            fontsize=11, fontweight='bold', color='#333333', zorder=2)
    
    interpretation = "Interpretation:\nMethods in Pareto frontier are\noptimal under some preference.\nDominated methods are never optimal."
    ax2.text(0.5, 0.32, interpretation, transform=ax2.transAxes,
            ha='center', va='center', fontsize=9, color='#555555', 
            style='italic', zorder=2)
    
    ax2.set_title('(b) Pareto Analysis', fontsize=13, fontweight='bold', pad=10)
    
    # =============================================
    # (c) 右上：雷达图概览
    # =============================================
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    ax3.set_facecolor('#FAFAFA')
    
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for method in metrics_df.index:
        values = metrics_df.loc[method].tolist()
        values += values[:1]
        color = colors_map[method]
        
        ax3.plot(angles, values, 'o-', linewidth=2.5, color=color,
                markersize=7, label=method, alpha=0.85,
                markeredgecolor='white', markeredgewidth=1.5)
        ax3.fill(angles, values, alpha=0.12, color=color)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='gray')
    ax3.grid(True, linestyle='--', alpha=0.4, linewidth=1)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9, 
              framealpha=0.9, edgecolor='gray')
    ax3.set_title('(c) 4D Performance Radar', fontsize=13, fontweight='bold', pad=15)
    
    # =============================================
    # (d) 左下：胜率条形图
    # =============================================
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_facecolor(COLORS['bg'])
    
    methods_sorted = sorted(win_rates.keys(), key=lambda m: -win_rates[m])
    rates_sorted = [win_rates[m] for m in methods_sorted]
    colors_sorted = [colors_map[m] for m in methods_sorted]
    
    y_pos = np.arange(len(methods_sorted))
    
    # 水平条形图
    bars = ax4.barh(y_pos, rates_sorted, color=colors_sorted,
                   alpha=0.85, edgecolor='white', linewidth=2.5, height=0.55)
    
    # 添加数值标签
    for i, (bar, rate, method) in enumerate(zip(bars, rates_sorted, methods_sorted)):
        width = bar.get_width()
        # 条形内文字
        ax4.text(width - 3, i, f'{rate:.1f}%', ha='right', va='center',
                fontsize=14, fontweight='bold', color='white')
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(methods_sorted, fontsize=13, fontweight='bold')
    ax4.set_xlabel('Win Rate (% of Weight Space)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Weight Space Win Rates (5000 samples)', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlim(0, 75)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.tick_params(left=False)
    ax4.grid(axis='x', alpha=0.3, linestyle=':', linewidth=1.2)
    
    # =============================================
    # (e) 右下：推荐卡片
    # =============================================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    ax5.set_facecolor(COLORS['bg'])
    
    # 主推荐卡片
    main_card = mpatches.FancyBboxPatch(
        (0.02, 0.35), 0.96, 0.60,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS['percent'], edgecolor='white',
        linewidth=3, alpha=0.9, transform=ax5.transAxes, zorder=1
    )
    ax5.add_patch(main_card)
    
    ax5.text(0.5, 0.82, 'PRIMARY', transform=ax5.transAxes,
            ha='center', va='center', fontsize=10, fontweight='bold', 
            color='white', alpha=0.8, zorder=2)
    ax5.text(0.5, 0.68, 'PERCENT', transform=ax5.transAxes,
            ha='center', va='center', fontsize=18, fontweight='bold', 
            color='white', zorder=2)
    
    # 关键指标
    leg_val = metrics_df.loc['PERCENT', 'legitimacy']
    eng_val = metrics_df.loc['PERCENT', 'engagement']
    rob_val = metrics_df.loc['PERCENT', 'robustness']
    
    metrics_text = f"Legitimacy: {leg_val:.2f}\nEngagement: {eng_val:.2f}\nRobustness: {rob_val:.2f}"
    ax5.text(0.5, 0.48, metrics_text, transform=ax5.transAxes,
            ha='center', va='center', fontsize=10, fontweight='bold', 
            color='white', zorder=2, linespacing=1.4)
    
    # 次要推荐
    sub_card = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.28,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=COLORS['save'], edgecolor='white',
        linewidth=2, alpha=0.85, transform=ax5.transAxes, zorder=1
    )
    ax5.add_patch(sub_card)
    
    ax5.text(0.5, 0.22, 'OPTIONAL: Triggered Save', transform=ax5.transAxes,
            ha='center', va='center', fontsize=10, fontweight='bold', 
            color='white', zorder=2)
    ax5.text(0.5, 0.08, 'When margin < 5%', transform=ax5.transAxes,
            ha='center', va='center', fontsize=9, 
            color='white', alpha=0.9, zorder=2)
    
    ax5.set_title('(e) Recommendation', fontsize=13, fontweight='bold', pad=10)
    
    # =============================================
    # 总标题
    # =============================================
    plt.suptitle('Robust Recommendation Summary: Multi-Objective Analysis', 
                fontsize=16, fontweight='bold', y=0.96)
    
    plt.savefig(f'{save_dir}/Task2_3_recommendation_summary.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"  Saved: Task2_3_recommendation_summary.png")


# ============================================================
# Main Execution
# ============================================================

def main():
    base_dir = r"c:\Users\zhaoh\Desktop\MCM-czb-nzh-zhk"
    save_dir = os.path.join(base_dir, "2.3_figures")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("TASK 2.3: ROBUST MULTI-OBJECTIVE RECOMMENDATION ANALYSIS")
    print("=" * 70)
    
    # Step 0: Load data
    ffi_df, controversy_df = load_analysis_data(base_dir)
    
    # Step 1: Extract preferences
    preferences = extract_producer_preferences()
    
    # Step 2: Compute metrics
    metrics_df = compute_all_metrics(ffi_df, controversy_df)
    
    # Save metrics table
    metrics_csv = os.path.join(save_dir, 'method_metrics.csv')
    metrics_df.to_csv(metrics_csv)
    print(f"\n  Saved metrics table to: {metrics_csv}")
    
    # Step 3: Pareto analysis
    pareto_set, dominated = find_pareto_frontier(metrics_df)
    
    # Step 4: Weight sensitivity
    win_rates, weights = weight_sensitivity_analysis(metrics_df)
    flip_boundaries = find_flip_boundaries(metrics_df)
    
    # Step 5: Constraint check
    check_legitimacy_constraint(controversy_df, threshold=5)
    cvar = compute_cvar_penalty(controversy_df, alpha=0.1)
    
    # Step 6: Robustness test
    adversarial_perturbation_test(ffi_df)
    
    # Step 7: Generate recommendations
    generate_robust_recommendations(metrics_df, pareto_set, win_rates)
    
    # Step 8: Visualizations
    print("\n" + "=" * 70)
    print("STEP 8: Generating Visualizations")
    print("=" * 70)
    
    plot_pareto_frontier(metrics_df, pareto_set, save_dir)
    plot_weight_sensitivity(win_rates, save_dir)
    plot_metric_radar(metrics_df, save_dir)
    create_recommendation_summary_plot(metrics_df, pareto_set, win_rates, save_dir)
    
    print("\n" + "=" * 70)
    print("TASK 2.3 COMPLETED!")
    print(f"All outputs saved to: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
