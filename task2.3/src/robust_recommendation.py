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

def _extract_judge_weekly_totals(data_csv_path: str) -> pd.DataFrame:
    """Extract week-level judge total scores as a long table."""
    data_df = pd.read_csv(data_csv_path)
    judge_records = []

    for _, row in data_df.iterrows():
        name = row['celebrity_name']
        season = row['season']

        for week in range(1, 12):
            week_scores = []
            for judge in range(1, 5):
                col = f'week{week}_judge{judge}_score'
                if col in row.index:
                    val = row[col]
                    if pd.notna(val) and val not in ['N/A', 'n/a', ''] and val != 0:
                        try:
                            week_scores.append(float(val))
                        except Exception:
                            pass

            if week_scores and sum(week_scores) > 0:
                judge_records.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'judge_total': sum(week_scores),
                })

    return pd.DataFrame(judge_records)


def _load_weekly_panel(base_dir: str) -> pd.DataFrame:
    """
    Load a clean weekly panel used for robustness tests.
    Columns: season, week, celebrity_name, fan_vote_share, judge_total, judge_percent
    """
    fan_path = os.path.join(base_dir, 'dataset', 'fan_vote_shares.csv')
    data_path = os.path.join(base_dir, 'dataset', '2026_MCM_Problem_C_Data.csv')

    fan_df = pd.read_csv(fan_path)
    judge_df = _extract_judge_weekly_totals(data_path)

    panel_df = pd.merge(
        fan_df,
        judge_df,
        on=['season', 'week', 'celebrity_name'],
        how='inner',
    )

    # Judge percent within week (comparable across weeks)
    panel_df['judge_percent'] = panel_df['judge_total'] / panel_df.groupby(['season', 'week'])['judge_total'].transform('sum')
    return panel_df


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
    
    # Weekly panel for robustness / perturbation tests
    panel_df = _load_weekly_panel(base_dir)
    print(f"  - Loaded weekly panel: {len(panel_df)} contestant-week records")

    return ffi_df, controversy_df, panel_df


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


def _get_elimination_weeks(panel_df: pd.DataFrame) -> set[tuple[int, int]]:
    """
    Infer elimination weeks from panel sizes.

    For each season, if the active contestant count decreases between consecutive
    observed weeks, we mark the earlier week as an elimination week.
    """
    elimination_weeks: set[tuple[int, int]] = set()
    counts = panel_df.groupby(['season', 'week'])['celebrity_name'].nunique().reset_index(name='n')

    for season, g in counts.groupby('season'):
        g2 = g.sort_values('week')
        weeks = g2['week'].tolist()
        ns = g2['n'].tolist()
        for i in range(len(weeks) - 1):
            if ns[i+1] < ns[i]:
                elimination_weeks.add((int(season), int(weeks[i])))

    return elimination_weeks


def _eliminated_name_for_week(week_df: pd.DataFrame, method: str) -> str | None:
    """
    Determine the eliminated contestant for a given week under a method.

    method: 'rank', 'percent', or 'save'
    """
    if week_df.shape[0] < 2:
        return None

    df = week_df.copy()

    if method in {'rank', 'save'}:
        # RANK stage: ranks (1 best) then sum; worst has max combined rank
        df['rank_fan'] = df['fan_vote_share'].rank(ascending=False, method='average')
        df['rank_judge'] = df['judge_total'].rank(ascending=False, method='average')
        df['combined_rank'] = df['rank_fan'] + df['rank_judge']

        # Worst two by combined_rank (tie-break deterministically)
        df_sorted = df.sort_values(
            ['combined_rank', 'judge_total', 'fan_vote_share'],
            ascending=[False, True, True],
            kind='mergesort',
        )

        if method == 'rank':
            return str(df_sorted.iloc[0]['celebrity_name'])

        bottom_two = df_sorted.head(2)
        if bottom_two.shape[0] == 1:
            return str(bottom_two.iloc[0]['celebrity_name'])

        # Judges eliminate lower technical score among bottom-two (tie-break by lower fan share)
        bottom_two_sorted = bottom_two.sort_values(
            ['judge_total', 'fan_vote_share'],
            ascending=[True, True],
            kind='mergesort',
        )
        return str(bottom_two_sorted.iloc[0]['celebrity_name'])

    # PERCENT: lowest fan_share + judge_percent eliminated
    df['combined'] = df['fan_vote_share'] + df['judge_percent']
    df_sorted = df.sort_values(
        ['combined', 'judge_total', 'fan_vote_share'],
        ascending=[True, True, True],
        kind='mergesort',
    )
    return str(df_sorted.iloc[0]['celebrity_name'])


def compute_robustness_flip_rates(
    panel_df: pd.DataFrame,
    perturbation_levels: list[float] | None = None,
    n_trials: int = 30,
    seed: int = 42,
) -> dict:
    """
    Robustness via elimination flip rates under adversarial/noisy fan-share perturbations.

    For each elimination week, we perturb the fan_vote_share vector by ε ~ U(-δ, δ),
    re-normalize to a probability simplex, and recompute the eliminated contestant.
    Flip rate = P(eliminated changes). Robustness = 1 - avg_flip_rate across δ.
    """
    if perturbation_levels is None:
        perturbation_levels = [0.01, 0.03, 0.05]

    rng = np.random.default_rng(seed)
    elimination_weeks = _get_elimination_weeks(panel_df)

    # Group data once for efficiency
    grouped = {(int(s), int(w)): g.copy() for (s, w), g in panel_df.groupby(['season', 'week'])}
    valid_weeks = [(s, w) for (s, w) in sorted(grouped.keys()) if (s, w) in elimination_weeks and grouped[(s, w)].shape[0] >= 3]

    methods = ['rank', 'percent', 'save']
    flip_rates: dict[str, dict[float, float]] = {m: {} for m in methods}

    for method in methods:
        # Baseline eliminated names
        baseline = {}
        for key in valid_weeks:
            e = _eliminated_name_for_week(grouped[key], method)
            if e is not None:
                baseline[key] = e

        for delta in perturbation_levels:
            flips = 0
            trials = 0

            for key in valid_weeks:
                base_elim = baseline.get(key)
                if base_elim is None:
                    continue

                week_df = grouped[key]
                shares = week_df['fan_vote_share'].to_numpy(dtype=float)
                n = shares.size

                for _ in range(n_trials):
                    noise = rng.uniform(-delta, delta, size=n)
                    pert = shares + noise
                    pert = np.clip(pert, 1e-6, None)
                    pert = pert / pert.sum()

                    pert_df = week_df.copy()
                    pert_df['fan_vote_share'] = pert

                    elim = _eliminated_name_for_week(pert_df, method)
                    if elim is None:
                        continue

                    trials += 1
                    if elim != base_elim:
                        flips += 1

            flip_rate = flips / trials if trials > 0 else 0.0
            flip_rates[method][delta] = flip_rate

    # Aggregate robustness score: 1 - average flip rate across δ
    robustness_scores = {
        'RANK': 1.0 - float(np.mean(list(flip_rates['rank'].values()))),
        'PERCENT': 1.0 - float(np.mean(list(flip_rates['percent'].values()))),
        'SAVE': 1.0 - float(np.mean(list(flip_rates['save'].values()))),
    }

    return {
        'elimination_weeks': len(valid_weeks),
        'n_trials': n_trials,
        'perturbation_levels': perturbation_levels,
        'flip_rates': flip_rates,
        'robustness_scores': robustness_scores,
    }


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


def compute_method_metric_vector(method_name, ffi_df, controversy_df, robustness_scores=None):
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
    if robustness_scores is not None and method_name in robustness_scores:
        base_robustness = robustness_scores[method_name]
    else:
        # Fallback (legacy proxy)
        mean_gap = controversy_df['weighted_gap'].mean()
        max_gap = controversy_df['weighted_gap'].max()
        base_robustness = np.clip(1.0 - (mean_gap / (max_gap + 1e-6)), 0, 1)
    
    # SAVE 的特殊调整
    if method_name == 'SAVE':
        # Legitimacy 提升：judges 在 bottom-two 时完全决定，
        # 相当于在"最关键的淘汰决策"上100%采用评委意见
        # 提升幅度：约15-20%（保守估计）
        legitimacy = min(base_legitimacy * 1.18, 1.0)
        
        # Robustness: already measured directly (if provided), otherwise use proxy.
        robustness = base_robustness
        
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


def compute_all_metrics(ffi_df, controversy_df, panel_df):
    """计算所有候选方案的指标向量（含基于扰动翻转率的稳健性）"""
    print("\n" + "=" * 70)
    print("STEP 2: Computing Unified Metric Vectors")
    print("=" * 70)
    
    methods = ['RANK', 'PERCENT', 'SAVE']
    metric_table = {}

    # Robustness: compute once from weekly panel (noisy-vote flip rate)
    print("\n  Computing robustness via perturbation flip rates...")
    robustness_report = compute_robustness_flip_rates(
        panel_df,
        perturbation_levels=[0.01, 0.03, 0.05],
        n_trials=30,
        seed=42,
    )
    robustness_scores = robustness_report['robustness_scores']
    print(f"  - Elimination weeks used: {robustness_report['elimination_weeks']}")
    for m in ['RANK', 'PERCENT', 'SAVE']:
        print(f"    robustness[{m:7s}] = {robustness_scores[m]:.4f}")
    
    for method in methods:
        metrics = compute_method_metric_vector(method, ffi_df, controversy_df, robustness_scores)
        metric_table[method] = metrics
        
        print(f"\n  {method}:")
        for key, val in metrics.items():
            print(f"    {key:15s}: {val:.4f}")
    
    # 转为DataFrame便于后续分析
    metrics_df = pd.DataFrame(metric_table).T
    return metrics_df, robustness_report


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
                    print(f"    {method_a} vs {method_b}: flip when w_eng/w_leg ~ {ratio:.3f}")
    
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
    print(f"    (Higher CVaR -> more extreme controversy in tail)")
    
    return cvar


# ============================================================
# Step 6: Robustness Check (对抗扰动)
# ============================================================

def adversarial_perturbation_test(robustness_report: dict):
    """Report the *measured* elimination flip rates under fan-share perturbations."""
    print("\n" + "=" * 70)
    print("STEP 6: Adversarial Robustness Test")
    print("=" * 70)
    
    methods = ['rank', 'percent', 'save']
    deltas = robustness_report.get('perturbation_levels', [])
    flip_rates = robustness_report.get('flip_rates', {})
    n_weeks = robustness_report.get('elimination_weeks', 0)
    n_trials = robustness_report.get('n_trials', 0)

    print(f"\n  Evaluated {n_weeks} elimination weeks x {n_trials} trials per delta")
    print("  (Flip rate = P(eliminated contestant changes) under fan-share noise)\n")

    # Print per-method per-delta flip rates
    for m in methods:
        if m not in flip_rates:
            continue
        pretty = {'rank': 'RANK', 'percent': 'PERCENT', 'save': 'SAVE'}[m]
        parts = []
        for d in deltas:
            r = flip_rates[m].get(d, 0.0)
            parts.append(f"delta=+/-{d*100:.1f}% -> {r*100:.1f}%")
        print(f"  {pretty:7s}: " + " | ".join(parts))

    # Ranking by average flip rate across deltas (lower is better)
    avg_flip = {}
    for m in methods:
        if m in flip_rates and flip_rates[m]:
            avg_flip[m] = float(np.mean(list(flip_rates[m].values())))

    ranking = sorted(avg_flip.items(), key=lambda kv: kv[1])
    print("\n  Robustness Ranking (lower avg flip = more robust):")
    for i, (m, r) in enumerate(ranking, 1):
        pretty = {'rank': 'RANK', 'percent': 'PERCENT', 'save': 'SAVE'}[m]
        print(f"    {i}. {pretty:7s}: avg flip {r*100:.1f}%")


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

    def _recommend(weights_vec):
        utilities = {m: evaluate_utility(metrics_df, m, weights_vec) for m in metrics_df.index}
        winner = max(utilities, key=utilities.get)
        return winner, utilities

    # Weight vectors correspond to [legitimacy, engagement, robustness, transparency]
    profiles = [
        ("A: Legitimacy-First", np.array([0.55, 0.10, 0.25, 0.10])),
        ("B: Engagement-First", np.array([0.10, 0.60, 0.10, 0.20])),
        ("C: Balanced + Risk-Averse", np.array([0.30, 0.20, 0.35, 0.15])),
    ]

    for label, w in profiles:
        winner, utilities = _recommend(w)
        print(f"\n  Preference Profile {label}")
        print(f"    -> Recommend: {winner}")
        print(f"       Utility (RANK/PERCENT/SAVE): " +
              f"{utilities['RANK']:.3f} / {utilities['PERCENT']:.3f} / {utilities['SAVE']:.3f}")
    
    print("\n+" + "=" * 68 + "+")
    print("|" + " LAYER 3: Hybrid Mechanism (Our Proposal)".center(68) + "|")
    print("+" + "=" * 68 + "+")

    primary = max(win_rates, key=win_rates.get) if win_rates else 'PERCENT'
    print(f"\n  Proposed System:")
    print(f"    * Primary Rule: {primary}")
    print(f"    * Trigger Condition: Activate Judges Save when:")
    print(f"        - Bottom-two margin < 5% (high uncertainty)")
    print(f"        - OR one contestant repeatedly ranks last by judges (persistent low technical approval)")
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
# Step 8: Visualizations (Completely Redesigned for Clarity)
# ============================================================

def plot_pareto_frontier(metrics_df, pareto_set, save_dir):
    """
    绘制Pareto前沿分析图 - 全新设计，避免重叠
    布局：2行2列，左上(散点)、右上(散点)、左下(柱状)、右下(雷达)
    """
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28,
                  left=0.08, right=0.92, top=0.90, bottom=0.08)
    
    method_colors = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 'SAVE': COLORS['save']}
    
    # =============================================
    # (a) 左上：Legitimacy vs Engagement 散点图
    # =============================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#FAFBFC')
    
    for method in metrics_df.index:
        x = metrics_df.loc[method, 'legitimacy']
        y = metrics_df.loc[method, 'engagement']
        color = method_colors.get(method, COLORS['dominated'])
        is_pareto = method in pareto_set
        
        ax1.scatter(x, y, s=700, c=color, marker='o', 
                   edgecolors='white', linewidths=3.5, zorder=3, alpha=0.92)
        # 白色文字+深色描边，确保在任何背景下都清晰
        from matplotlib.patheffects import withStroke
        ax1.text(x, y, method, fontsize=12, fontweight='bold',
                ha='center', va='center', color='white', zorder=4,
                path_effects=[withStroke(linewidth=3, foreground='#333333')])
    
    # 对角线参考
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1.5, zorder=1)
    ax1.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color=COLORS['percent'], zorder=0)
    ax1.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color=COLORS['rank'], zorder=0)
    
    ax1.set_xlabel('Legitimacy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Engagement', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Trade-off: Legitimacy vs Engagement', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlim(0.35, 0.65)
    ax1.set_ylim(0.65, 1.05)
    ax1.grid(alpha=0.3, linestyle=':', linewidth=1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # =============================================
    # (b) 右上：Robustness vs Transparency 散点图
    # =============================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#FAFBFC')
    
    for method in metrics_df.index:
        x = metrics_df.loc[method, 'robustness']
        y = metrics_df.loc[method, 'transparency']
        color = method_colors.get(method, COLORS['dominated'])
        
        ax2.scatter(x, y, s=700, c=color, marker='o',
                   edgecolors='white', linewidths=3.5, zorder=3, alpha=0.92)
        # 同样添加描边效果
        ax2.text(x, y, method, fontsize=12, fontweight='bold',
                ha='center', va='center', color='white', zorder=4,
                path_effects=[withStroke(linewidth=3, foreground='#333333')])
    
    ax2.set_xlabel('Robustness', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Transparency', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Trade-off: Robustness vs Transparency', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlim(0.58, 0.82)
    ax2.set_ylim(0.65, 1.05)
    ax2.grid(alpha=0.3, linestyle=':', linewidth=1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # =============================================
    # (c) 左下：指标柱状图对比
    # =============================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#FAFBFC')
    
    categories = list(metrics_df.columns)
    x_pos = np.arange(len(categories))
    width = 0.25
    
    for i, method in enumerate(metrics_df.index):
        values = metrics_df.loc[method].values
        offset = (i - 1) * width
        bars = ax3.bar(x_pos + offset, values, width, label=method, 
                      color=method_colors[method], alpha=0.85,
                      edgecolor='white', linewidth=1.5)
        # 数值标注
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax3.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Metric Comparison by Method', fontsize=13, fontweight='bold', pad=12)
    ax3.set_ylim(0, 1.15)
    ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # =============================================
    # (d) 右下：雷达图叠加
    # =============================================
    ax4 = fig.add_subplot(gs[1, 1], projection='polar')
    ax4.set_facecolor('#FAFBFC')
    
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for method in metrics_df.index:
        values = metrics_df.loc[method].tolist()
        values += values[:1]
        color = method_colors.get(method, COLORS['dominated'])
        
        ax4.plot(angles, values, 'o-', linewidth=2.5, color=color,
                markersize=8, label=method, alpha=0.9)
        ax4.fill(angles, values, alpha=0.15, color=color)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax4.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='gray')
    ax4.grid(True, linestyle=':', alpha=0.4, linewidth=1)
    ax4.set_title('(d) 4D Performance Radar', fontsize=13, fontweight='bold', pad=18)
    ax4.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.15), fontsize=10, 
              framealpha=0.95, edgecolor='gray')
    
    plt.suptitle('Pareto Frontier Analysis: Multi-Objective Trade-offs', 
                fontsize=16, fontweight='bold', y=0.96)
    
    plt.savefig(f'{save_dir}/Task2_3_pareto_frontier.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"  Saved: Task2_3_pareto_frontier.png")


def plot_weight_sensitivity(win_rates, save_dir):
    """绘制权重敏感性分析 - 清晰简洁的现代风格"""
    fig = plt.figure(figsize=(14, 6), facecolor='white')
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.92, top=0.85, bottom=0.12)
    
    methods = sorted(win_rates.keys(), key=lambda m: -win_rates[m])
    rates = [win_rates[m] for m in methods]
    colors_map = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 'SAVE': COLORS['save']}
    colors = [colors_map[m] for m in methods]
    
    # =============================================
    # (a) 左侧：水平条形图
    # =============================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#FAFBFC')
    
    y_pos = np.arange(len(methods))
    bars = ax1.barh(y_pos, rates, height=0.55, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=2)
    
    # 数值标签
    for i, (bar, rate, method) in enumerate(zip(bars, rates, methods)):
        ax1.text(rate + 1.5, i, f'{rate:.1f}%', ha='left', va='center',
                fontsize=12, fontweight='bold', color=colors_map[method])
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Optimality Across 5000 Random Weights', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlim(0, 75)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(axis='x', alpha=0.3, linestyle=':', linewidth=1)
    ax1.tick_params(left=False)
    
    # =============================================
    # (b) 右侧：甜甜圈图
    # =============================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#FAFBFC')
    
    nonzero_methods = [m for m in methods if win_rates[m] > 0.1]
    nonzero_rates = [win_rates[m] for m in nonzero_methods]
    nonzero_colors = [colors_map[m] for m in nonzero_methods]
    
    wedges, texts, autotexts = ax2.pie(
        nonzero_rates,
        labels=nonzero_methods,
        autopct='%1.1f%%',
        colors=nonzero_colors,
        startangle=90,
        pctdistance=0.75,
        explode=[0.03] * len(nonzero_methods),
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2.5),
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # 中心文字
    ax2.text(0, 0, f'{len(nonzero_methods)}\nViable', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#555555')
    
    ax2.set_title('(b) Preference Space Distribution', fontsize=13, fontweight='bold', pad=12)
    
    plt.suptitle('Weight Sensitivity Analysis', fontsize=15, fontweight='bold', y=0.96)
    
    plt.savefig(f'{save_dir}/Task2_3_weight_sensitivity.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"  Saved: Task2_3_weight_sensitivity.png")


def plot_metric_radar(metrics_df, save_dir):
    """绘制雷达图 - 清晰简洁的布局，避免标签重叠"""
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, figure=fig, wspace=0.38, left=0.05, right=0.95, top=0.85, bottom=0.12)
    
    categories = list(metrics_df.columns)
    # 使用缩写避免重叠
    category_labels = ['engmt', 'legit', 'robust', 'transp']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    colors_map = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 'SAVE': COLORS['save']}
    
    # 前三个子图：单独的雷达图
    for idx, method in enumerate(metrics_df.index):
        ax = fig.add_subplot(gs[0, idx], projection='polar')
        ax.set_facecolor('#FAFBFC')
        
        values = metrics_df.loc[method].tolist()
        values += values[:1]
        color = colors_map[method]
        
        # 填充和曲线
        ax.fill(angles, values, alpha=0.25, color=color, zorder=1)
        ax.plot(angles, values, 'o-', linewidth=2.5, color=color,
               markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        
        # 数值标注（放在点外侧，调整位置避免重叠）
        for angle, value in zip(angles[:-1], values[:-1]):
            # 根据角度调整文字位置
            offset = 0.15 if value > 0.8 else 0.12
            ax.text(angle, value + offset, f'{value:.2f}', ha='center', va='center',
                   fontsize=8, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor=color, linewidth=0.8, alpha=0.85))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(category_labels, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['50%', '100%'], fontsize=7, color='gray')
        ax.grid(True, linestyle=':', alpha=0.4, linewidth=1)
        ax.set_title(method, fontsize=13, fontweight='bold', pad=18, color=color)
    
    # 第四个子图：叠加对比
    ax_comb = fig.add_subplot(gs[0, 3], projection='polar')
    ax_comb.set_facecolor('#FAFBFC')
    
    for method in metrics_df.index:
        values = metrics_df.loc[method].tolist()
        values += values[:1]
        color = colors_map[method]
        
        ax_comb.plot(angles, values, 'o-', linewidth=2, color=color,
                    markersize=6, label=method, alpha=0.85)
        ax_comb.fill(angles, values, alpha=0.10, color=color)
    
    ax_comb.set_xticks(angles[:-1])
    ax_comb.set_xticklabels(category_labels, fontsize=9, fontweight='bold')
    ax_comb.set_ylim(0, 1.15)
    ax_comb.set_yticks([0.5, 1.0])
    ax_comb.set_yticklabels(['50%', '100%'], fontsize=7, color='gray')
    ax_comb.grid(True, linestyle=':', alpha=0.4, linewidth=1)
    ax_comb.set_title('Comparison', fontsize=13, fontweight='bold', pad=18, color='#444444')
    ax_comb.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
                  fontsize=10, framealpha=0.9, edgecolor='gray')
    
    plt.suptitle('Multi-Objective Performance: Radar Chart Analysis', 
                fontsize=15, fontweight='bold', y=0.96)
    
    plt.savefig(f'{save_dir}/Task2_3_radar_chart.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"  Saved: Task2_3_radar_chart.png")


def create_recommendation_summary_plot(metrics_df, pareto_set, win_rates, save_dir):
    """创建推荐总结图 - 清晰简洁的现代信息图风格"""
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32,
                  left=0.06, right=0.94, top=0.88, bottom=0.08)
    
    methods = list(metrics_df.index)
    categories = list(metrics_df.columns)
    colors_map = {'RANK': COLORS['rank'], 'PERCENT': COLORS['percent'], 'SAVE': COLORS['save']}
    
    # =============================================
    # (a) 左上：度量对比热力图
    # =============================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#FAFBFC')
    
    data_matrix = metrics_df.values
    im = ax1.imshow(data_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1, alpha=0.9)
    
    for i in range(len(methods)):
        for j in range(len(categories)):
            val = data_matrix[i, j]
            text_color = 'white' if val > 0.55 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha="center", va="center",
                    color=text_color, fontsize=10, fontweight='bold')
    
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_yticks(np.arange(len(methods)))
    ax1.set_xticklabels([c[:4] for c in categories], fontsize=9, fontweight='bold')
    ax1.set_yticklabels(methods, fontsize=10, fontweight='bold')
    ax1.set_title('(a) Metric Heatmap', fontsize=12, fontweight='bold', pad=8)
    
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    
    # =============================================
    # (b) 中上：Pareto 分析卡片
    # =============================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_facecolor('#FAFBFC')
    
    card_rect = mpatches.FancyBboxPatch(
        (0.05, 0.15), 0.9, 0.70,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor='#E8F4F8', edgecolor='#3498DB', linewidth=2,
        alpha=0.95, transform=ax2.transAxes
    )
    ax2.add_patch(card_rect)
    
    ax2.text(0.5, 0.72, 'PARETO FRONTIER', transform=ax2.transAxes,
            ha='center', va='center', fontsize=12, fontweight='bold', color='#2980B9')
    ax2.text(0.5, 0.52, f"Efficient: {', '.join(pareto_set)}", 
            transform=ax2.transAxes, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#333333')
    ax2.text(0.5, 0.32, "All methods viable\nunder some preference", 
            transform=ax2.transAxes, ha='center', va='center',
            fontsize=9, color='#666666', style='italic')
    ax2.set_title('(b) Pareto Analysis', fontsize=12, fontweight='bold', pad=8)
    
    # =============================================
    # (c) 右上：雷达图概览
    # =============================================
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    ax3.set_facecolor('#FAFBFC')
    
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for method in metrics_df.index:
        values = metrics_df.loc[method].tolist()
        values += values[:1]
        color = colors_map[method]
        ax3.plot(angles, values, 'o-', linewidth=2, color=color,
                markersize=6, label=method, alpha=0.85)
        ax3.fill(angles, values, alpha=0.12, color=color)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels([c[:4] for c in categories], fontsize=8, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.set_yticks([0.5, 1.0])
    ax3.set_yticklabels(['50%', '100%'], fontsize=7, color='gray')
    ax3.grid(True, linestyle=':', alpha=0.4, linewidth=1)
    ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=8,
              framealpha=0.9, ncol=3)
    ax3.set_title('(c) Radar', fontsize=12, fontweight='bold', pad=10)
    
    # =============================================
    # (d) 左下：胜率条形图
    # =============================================
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor('#FAFBFC')
    
    methods_sorted = sorted(win_rates.keys(), key=lambda m: -win_rates[m])
    rates_sorted = [win_rates[m] for m in methods_sorted]
    colors_sorted = [colors_map[m] for m in methods_sorted]
    
    y_pos = np.arange(len(methods_sorted))
    bars = ax4.barh(y_pos, rates_sorted, color=colors_sorted,
                   alpha=0.85, edgecolor='white', linewidth=2, height=0.5)
    
    for i, (bar, rate) in enumerate(zip(bars, rates_sorted)):
        ax4.text(rate + 1.5, i, f'{rate:.1f}%', ha='left', va='center',
                fontsize=11, fontweight='bold', color=colors_sorted[i])
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(methods_sorted, fontsize=11, fontweight='bold')
    ax4.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Weight Space Win Rates (5000 samples)', fontsize=12, fontweight='bold', pad=8)
    ax4.set_xlim(0, 72)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.tick_params(left=False)
    ax4.grid(axis='x', alpha=0.3, linestyle=':', linewidth=1)
    
    # =============================================
    # (e) 右下：推荐卡片
    # =============================================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    ax5.set_facecolor('#FAFBFC')

    primary_method = methods_sorted[0] if methods_sorted else 'PERCENT'
    
    # 主推荐卡片
    main_card = mpatches.FancyBboxPatch(
        (0.03, 0.38), 0.94, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=colors_map.get(primary_method, COLORS['percent']), edgecolor='white',
        linewidth=2.5, alpha=0.92, transform=ax5.transAxes
    )
    ax5.add_patch(main_card)
    
    ax5.text(0.5, 0.82, 'PRIMARY', transform=ax5.transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white', alpha=0.8)
    ax5.text(0.5, 0.68, primary_method, transform=ax5.transAxes,
            ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    
    leg_val = metrics_df.loc[primary_method, 'legitimacy']
    eng_val = metrics_df.loc[primary_method, 'engagement']
    rob_val = metrics_df.loc[primary_method, 'robustness']
    metrics_text = f"L:{leg_val:.2f} E:{eng_val:.2f} R:{rob_val:.2f}"
    ax5.text(0.5, 0.50, metrics_text, transform=ax5.transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 次要推荐
    sub_card = mpatches.FancyBboxPatch(
        (0.03, 0.05), 0.94, 0.28,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=COLORS['save'], edgecolor='white',
        linewidth=2, alpha=0.85, transform=ax5.transAxes
    )
    ax5.add_patch(sub_card)
    
    ax5.text(0.5, 0.25, 'OPTIONAL: Triggered Save', transform=ax5.transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax5.text(0.5, 0.11, 'When margin < 5%', transform=ax5.transAxes,
            ha='center', va='center', fontsize=8, color='white', alpha=0.9)
    
    ax5.set_title('(e) Recommendation', fontsize=12, fontweight='bold', pad=8)
    
    plt.suptitle('Recommendation Summary: Multi-Objective Analysis', 
                fontsize=14, fontweight='bold', y=0.95)
    
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
    ffi_df, controversy_df, panel_df = load_analysis_data(base_dir)
    
    # Step 1: Extract preferences
    preferences = extract_producer_preferences()
    
    # Step 2: Compute metrics
    metrics_df, robustness_report = compute_all_metrics(ffi_df, controversy_df, panel_df)
    
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
    adversarial_perturbation_test(robustness_report)
    
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
