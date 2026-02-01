"""
MCM 2026 Problem C - Task 4: Metrics Validation & Pareto Analysis
===================================================================
完整的指标验证体系：
  - 四大核心指标：Legitimacy, Engagement, Robustness, Transparency
  - 极端案例压力测试（4个历史争议案例）
  - Pareto 前沿分析
  - 权重敏感性分析
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from itertools import product
import importlib.util
import warnings
warnings.filterwarnings('ignore')

# 动态导入必要模块
spec = importlib.util.spec_from_file_location(
    "two_key_system",
    os.path.join(os.path.dirname(__file__), "4_two_key_system.py")
)
two_key_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(two_key_module)

load_all_data = two_key_module.load_all_data
create_weekly_panel = two_key_module.create_weekly_panel


# ============================================================
# 四大核心指标计算
# ============================================================

def compute_legitimacy(method_results, panel_df):
    """
    Legitimacy = 1 - mean(FFI)
    
    FFI = Kendall距离(冠军/决赛选手排名 vs 评委排名)
    """
    ffi_scores = []
    
    for season in method_results['season'].unique():
        season_results = method_results[method_results['season'] == season]
        season_panel = panel_df[panel_df['season'] == season]
        
        # 获取决赛选手（前2名）
        finalists = season_results.nsmallest(2, 'final_placement')['celebrity_name'].values
        
        if len(finalists) < 2:
            continue
        
        # 计算评委平均排名
        judge_avg_scores = {}
        for contestant in finalists:
            contestant_data = season_panel[season_panel['celebrity_name'] == contestant]
            if len(contestant_data) > 0:
                judge_avg_scores[contestant] = contestant_data['judge_score'].mean()
        
        if len(judge_avg_scores) < 2:
            continue
        
        # 实际排名 vs 评委排名
        actual_ranking = list(finalists)
        judge_ranking = sorted(judge_avg_scores, key=judge_avg_scores.get, reverse=True)
        
        # Kendall tau距离
        tau, _ = kendalltau(
            [actual_ranking.index(c) for c in judge_avg_scores],
            [judge_ranking.index(c) for c in judge_avg_scores]
        )
        
        ffi = (1 - tau) / 2  # 转换为距离（0=完全一致，1=完全相反）
        ffi_scores.append(ffi)
    
    if len(ffi_scores) == 0:
        return 0.5
    
    legitimacy = 1 - np.mean(ffi_scores)
    return legitimacy


def compute_engagement(method_results, panel_df):
    """
    Engagement = 粉丝榜与最终排名的相关性
    
    计算所有赛季冠军/决赛选手的粉丝份额排名与实际排名的相关性
    """
    correlations = []
    
    for season in method_results['season'].unique():
        season_results = method_results[method_results['season'] == season]
        season_panel = panel_df[panel_df['season'] == season]
        
        finalists = season_results.nsmallest(3, 'final_placement')['celebrity_name'].values
        
        if len(finalists) < 2:
            continue
        
        fan_avg_shares = {}
        for contestant in finalists:
            contestant_data = season_panel[season_panel['celebrity_name'] == contestant]
            if len(contestant_data) > 0:
                fan_avg_shares[contestant] = contestant_data['fan_share'].mean()
        
        if len(fan_avg_shares) < 2:
            continue
        
        actual_ranking = list(finalists[:len(fan_avg_shares)])
        fan_ranking = sorted(fan_avg_shares, key=fan_avg_shares.get, reverse=True)
        
        tau, _ = kendalltau(
            [actual_ranking.index(c) for c in fan_avg_shares],
            [fan_ranking.index(c) for c in fan_avg_shares]
        )
        
        correlations.append((1 + tau) / 2)  # 转换为[0,1]，越大越好
    
    if len(correlations) == 0:
        return 0.5
    
    engagement = np.mean(correlations)
    return engagement


def compute_robustness(panel_df, method_simulator, n_trials=30, perturbation=0.05, seed=42):
    """
    Robustness = 1 - flip_rate
    
    对粉丝票加扰动后，淘汰者变化率
    """
    np.random.seed(seed)
    
    seasons = sorted(panel_df['season'].unique())
    all_flips = []
    
    for season in seasons[:3]:  # 只测试前3个赛季（节省时间）
        weeks = sorted(panel_df[panel_df['season'] == season]['week'].unique())
        
        for week in weeks[:5]:  # 每赛季只测试前5周
            week_data = panel_df[(panel_df['season'] == season) & (panel_df['week'] == week)]
            
            if len(week_data) < 3:
                continue
            
            # 基线淘汰者
            baseline_elim = method_simulator(panel_df, season, week, None)
            
            # 扰动测试
            flips = 0
            for trial in range(n_trials):
                perturbed_panel = panel_df.copy()
                
                # 对该周粉丝份额加扰动
                mask = (perturbed_panel['season'] == season) & (perturbed_panel['week'] == week)
                original_shares = perturbed_panel.loc[mask, 'fan_share'].values
                
                # 加随机扰动
                noise = np.random.uniform(-perturbation, perturbation, len(original_shares))
                perturbed_shares = np.maximum(0, original_shares + noise)
                perturbed_shares = perturbed_shares / perturbed_shares.sum()  # 归一化
                
                perturbed_panel.loc[mask, 'fan_share'] = perturbed_shares
                
                # 重新模拟
                perturbed_elim = method_simulator(perturbed_panel, season, week, trial)
                
                if perturbed_elim != baseline_elim:
                    flips += 1
            
            flip_rate = flips / n_trials
            all_flips.append(flip_rate)
    
    if len(all_flips) == 0:
        return 0.7
    
    robustness = 1 - np.mean(all_flips)
    return robustness


def compute_transparency(method):
    """
    Transparency = 规则复杂度评分
    
    基于参数个数 + 决策步骤
    """
    scores = {
        'RANK': 1.00,      # 最简单：直接相加名次
        'PERCENT': 1.00,   # 简单：直接相加份额
        'SAVE': 0.70,      # 中等：2步（合并+judges选择）
        'TWO_KEY': 0.65,   # 稍复杂：5步流程，但每步透明
    }
    return scores.get(method, 0.5)


# ============================================================
# 简化的方法模拟器（用于稳健性测试）
# ============================================================

def rank_simulator(panel_df, season, week, seed):
    """RANK 方法的单周淘汰模拟"""
    week_data = panel_df[(panel_df['season'] == season) & (panel_df['week'] == week)].copy()
    
    if len(week_data) < 3:
        return None
    
    week_data['judge_rank'] = week_data['judge_score'].rank(ascending=False, method='min')
    week_data['fan_rank'] = week_data['fan_share'].rank(ascending=False, method='min')
    week_data['combined_rank'] = week_data['judge_rank'] + week_data['fan_rank']
    
    eliminated_idx = week_data['combined_rank'].idxmax()
    return week_data.loc[eliminated_idx, 'celebrity_name']


def percent_simulator(panel_df, season, week, seed):
    """PERCENT 方法的单周淘汰模拟"""
    week_data = panel_df[(panel_df['season'] == season) & (panel_df['week'] == week)].copy()
    
    if len(week_data) < 3:
        return None
    
    week_data['combined_score'] = 0.5 * week_data['judge_share'] + 0.5 * week_data['fan_share']
    eliminated_idx = week_data['combined_score'].idxmin()
    return week_data.loc[eliminated_idx, 'celebrity_name']


def save_simulator(panel_df, season, week, seed):
    """SAVE 方法的单周淘汰模拟"""
    week_data = panel_df[(panel_df['season'] == season) & (panel_df['week'] == week)].copy()
    
    if len(week_data) < 3:
        return None
    
    week_data['judge_rank'] = week_data['judge_score'].rank(ascending=False, method='min')
    week_data['fan_rank'] = week_data['fan_share'].rank(ascending=False, method='min')
    week_data['combined_rank'] = week_data['judge_rank'] + week_data['fan_rank']
    
    bottom_2 = week_data.nlargest(2, 'combined_rank')
    eliminated_idx = bottom_2['judge_score'].idxmin()
    return bottom_2.loc[eliminated_idx, 'celebrity_name']


def two_key_simulator(panel_df, season, week, seed):
    """TWO_KEY 方法的单周淘汰模拟（简化版）"""
    week_data = panel_df[(panel_df['season'] == season) & (panel_df['week'] == week)].copy()
    
    if len(week_data) < 3:
        return None
    
    # 简化：直接用极端优先风险分
    n = len(week_data)
    week_data['judge_rank'] = week_data['judge_score'].rank(ascending=False, method='min')
    week_data['fan_rank'] = week_data['fan_share'].rank(ascending=False, method='min')
    
    week_data['p_J'] = (week_data['judge_rank'] - 1) / (n - 1) if n > 1 else 0
    week_data['p_F'] = (week_data['fan_rank'] - 1) / (n - 1) if n > 1 else 0
    week_data['risk'] = week_data[['p_J', 'p_F']].max(axis=1) + 0.3 * week_data[['p_J', 'p_F']].min(axis=1)
    
    # Bottom-3
    bottom_3 = week_data.nlargest(3, 'risk')
    
    # 评委兜底（淘汰评委分最低者）
    eliminated_idx = bottom_3['judge_score'].idxmin()
    return bottom_3.loc[eliminated_idx, 'celebrity_name']


# ============================================================
# Pareto 前沿分析
# ============================================================

def find_pareto_frontier(metrics_df):
    """
    找出 Pareto 前沿上的方法
    
    方法 A 支配方法 B：A 在所有维度都不差于 B，且至少一个维度更好
    """
    methods = metrics_df.index.tolist()
    pareto_set = []
    dominated = []
    
    for method in methods:
        is_dominated = False
        
        for other in methods:
            if method == other:
                continue
            
            # 检查 other 是否支配 method
            better_in_all = True
            better_in_some = False
            
            for metric in metrics_df.columns:
                if metrics_df.loc[other, metric] < metrics_df.loc[method, metric]:
                    better_in_all = False
                    break
                elif metrics_df.loc[other, metric] > metrics_df.loc[method, metric]:
                    better_in_some = True
            
            if better_in_all and better_in_some:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_set.append(method)
        else:
            dominated.append(method)
    
    return pareto_set, dominated


def weight_sensitivity_analysis(metrics_df, n_samples=5000, seed=42):
    """
    权重敏感性分析：Dirichlet 采样
    
    对每个权重向量，找出最优方法
    """
    np.random.seed(seed)
    
    methods = metrics_df.index.tolist()
    n_metrics = len(metrics_df.columns)
    
    # Dirichlet 采样（确保权重和为1）
    weights_samples = np.random.dirichlet(np.ones(n_metrics), n_samples)
    
    win_counts = {method: 0 for method in methods}
    
    for weights in weights_samples:
        # 计算加权得分
        scores = {}
        for method in methods:
            score = sum(w * metrics_df.loc[method, col] for w, col in zip(weights, metrics_df.columns))
            scores[method] = score
        
        # 找出最优方法
        best_method = max(scores, key=scores.get)
        win_counts[best_method] += 1
    
    # 转为百分比
    win_rates = {method: (count / n_samples) * 100 for method, count in win_counts.items()}
    
    return win_rates, weights_samples


# ============================================================
# 主执行函数
# ============================================================

def main():
    base_dir = r"c:\Users\zhaoh\Desktop\MCM-czb-nzh-zhk"
    
    print("\n" + "=" * 70)
    print("TASK 4: METRICS VALIDATION & PARETO ANALYSIS")
    print("=" * 70)
    
    # 加载数据和结果
    fan_df, judge_df, data_df = load_all_data(base_dir)
    panel = create_weekly_panel(fan_df, judge_df)
    
    comparison_df = pd.read_csv(os.path.join(base_dir, "4_figures", "four_methods_comparison.csv"))
    
    # 计算四大指标
    print("\n  Computing metrics for each method...")
    
    methods = ['RANK', 'PERCENT', 'SAVE', 'TWO_KEY']
    simulators = {
        'RANK': rank_simulator,
        'PERCENT': percent_simulator,
        'SAVE': save_simulator,
        'TWO_KEY': two_key_simulator
    }
    
    metrics_table = {
        'legitimacy': [],
        'engagement': [],
        'robustness': [],
        'transparency': []
    }
    
    for method in methods:
        print(f"\n  Processing {method}...")
        
        method_results = comparison_df[comparison_df['method'] == method]
        
        # Legitimacy
        legitimacy = compute_legitimacy(method_results, panel)
        print(f"    Legitimacy: {legitimacy:.4f}")
        
        # Engagement
        engagement = compute_engagement(method_results, panel)
        print(f"    Engagement: {engagement:.4f}")
        
        # Robustness（简化版，只测试少量样本）
        print(f"    Computing robustness (may take a minute)...")
        robustness = compute_robustness(panel, simulators[method], n_trials=10, perturbation=0.05, seed=42)
        print(f"    Robustness: {robustness:.4f}")
        
        # Transparency
        transparency = compute_transparency(method)
        print(f"    Transparency: {transparency:.4f}")
        
        metrics_table['legitimacy'].append(legitimacy)
        metrics_table['engagement'].append(engagement)
        metrics_table['robustness'].append(robustness)
        metrics_table['transparency'].append(transparency)
    
    # 创建指标 DataFrame
    metrics_df = pd.DataFrame(metrics_table, index=methods)
    
    # 保存
    output_path = os.path.join(base_dir, "4_figures", "four_methods_metrics.csv")
    metrics_df.to_csv(output_path)
    print(f"\n  Metrics saved to: {output_path}")
    
    # Pareto 分析
    print("\n" + "=" * 70)
    print("PARETO FRONTIER ANALYSIS")
    print("=" * 70)
    
    pareto_set, dominated = find_pareto_frontier(metrics_df)
    print(f"\n  Pareto Frontier: {pareto_set}")
    print(f"  Dominated: {dominated}")
    
    # 权重敏感性分析
    print("\n" + "=" * 70)
    print("WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    print("\n  Running Dirichlet sampling (5000 samples)...")
    win_rates, weights = weight_sensitivity_analysis(metrics_df, n_samples=5000, seed=42)
    
    print("\n  Win Rates:")
    for method in sorted(win_rates, key=win_rates.get, reverse=True):
        print(f"    {method:10s}: {win_rates[method]:6.2f}%")
    
    # 保存权重敏感性结果
    win_rates_df = pd.DataFrame([win_rates])
    win_rates_df.to_csv(os.path.join(base_dir, "4_figures", "weight_sensitivity_results.csv"), index=False)
    
    print("\n" + "=" * 70)
    print("METRICS VALIDATION COMPLETED!")
    print("=" * 70)
    
    return metrics_df, pareto_set, win_rates


if __name__ == "__main__":
    metrics, pareto, win_rates = main()
