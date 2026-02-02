"""
MCM 2026 Problem C - Task 4: Metrics Validation (Unified Source)
================================================================
Task4 指标的唯一真源，确保与 Tasks 1-3 完全一致：
  - 复用 Task2.3 的 legitimacy/engagement/robustness 计算逻辑
  - 按 Task2.1 的 Kendall 距离口径逐周计算 ffi_two_key
  - 输出统一指标表 + 权重敏感性 + 效用采样（用于 regime-shift 可视化）
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from pathlib import Path
import importlib.util
import warnings
warnings.filterwarnings('ignore')

# 路径设置
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "task2.3" / "src"))
sys.path.insert(0, str(repo_root / "task4" / "src"))

# 动态导入
from robust_recommendation import compute_legitimacy_metric, compute_engagement_metric
from two_key_system import load_all_data, create_weekly_panel


# ============================================================
# 为 TWO_KEY 按 Task2.1 口径逐周计算 FFI
# ============================================================

def compute_ffi_two_key_by_week(panel_df, seed=42, params=None):
    """
    按 Task2.1 的 Kendall 距离口径逐周计算 ffi_two_key
    
    对每个 (season, week)：
    1. 构造 TWO_KEY 的完整排名 R_two_key（基于风险分，从安全到危险）
    2. 计算 d(R_two_key, R_judge) 和 d(R_two_key, R_fan)
    3. FFI = d_judge - d_fan
    
    Parameters:
    -----------
    params : dict, optional
        TWO_KEY 参数（使用优化后的参数，包括 judge_weight）
    
    Returns:
    --------
    DataFrame with columns: season, week, ffi_two_key
    """
    if params is None:
        params = {'alpha': 0.4, 'beta': 0.6, 'judge_weight': 1.2}
    
    print(f"\n  Computing ffi_two_key by week (Task2.1 style, alpha={params['alpha']}, beta={params['beta']})...")
    
    seasons = sorted(panel_df['season'].unique())
    ffi_records = []
    
    for season in seasons:
        season_data = panel_df[panel_df['season'] == season]
        weeks = sorted(season_data['week'].unique())
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week].copy()
            
            if len(week_data) < 3:
                continue
            
            n = len(week_data)
            
            # 构造 TWO_KEY 的风险优先级排名（使用参数化的 alpha 和 judge_weight）
            week_data['judge_rank'] = week_data['judge_score'].rank(ascending=False, method='min')
            week_data['fan_rank'] = week_data['fan_share'].rank(ascending=False, method='min')
            week_data['p_J'] = (week_data['judge_rank'] - 1) / (n - 1) if n > 1 else 0
            week_data['p_F'] = (week_data['fan_rank'] - 1) / (n - 1) if n > 1 else 0
            
            judge_weight = params.get('judge_weight', 1.2)
            week_data['risk'] = np.maximum(judge_weight * week_data['p_J'], week_data['p_F']) + params['alpha'] * np.minimum(week_data['p_J'], week_data['p_F'])
            
            # 排序：risk 越小越安全（排名越好）
            week_data = week_data.sort_values('risk')
            two_key_ranking = list(week_data['celebrity_name'].values)
            
            # 评委排名
            judge_scores = dict(zip(week_data['celebrity_name'], week_data['judge_score']))
            judge_ranking = sorted(judge_scores, key=judge_scores.get, reverse=True)
            
            # 粉丝排名
            fan_shares = dict(zip(week_data['celebrity_name'], week_data['fan_share']))
            fan_ranking = sorted(fan_shares, key=fan_shares.get, reverse=True)
            
            # 计算 Kendall 距离（与 Task2.1 完全一致）
            common = set(two_key_ranking) & set(judge_ranking) & set(fan_ranking)
            if len(common) < 2:
                continue
            
            # d(R_two_key, R_judge)
            tau_judge, _ = kendalltau(
                [two_key_ranking.index(c) for c in common],
                [judge_ranking.index(c) for c in common]
            )
            d_judge = (1 - tau_judge) / 2
            
            # d(R_two_key, R_fan)
            tau_fan, _ = kendalltau(
                [two_key_ranking.index(c) for c in common],
                [fan_ranking.index(c) for c in common]
            )
            d_fan = (1 - tau_fan) / 2
            
            # FFI = d_judge - d_fan
            ffi = d_judge - d_fan
            
            ffi_records.append({
                'season': season,
                'week': week,
                'ffi_two_key': ffi
            })
    
    ffi_df = pd.DataFrame(ffi_records)
    print(f"    Computed FFI for {len(ffi_df)} weeks")
    
    return ffi_df


# ============================================================
# 统一指标计算（复用 Task2.3 逻辑 + TWO_KEY 扩展）
# ============================================================

def compute_all_metrics_unified(use_tuned_params=True):
    """
    统一计算四种方法的四大指标（使用真实测量的 TWO_KEY robustness）
    
    策略：
    1. RANK/PERCENT/SAVE: 直接从 Task2.3 的 method_metrics.csv 读取（确保一致）
    2. TWO_KEY: 基于真实 robustness 测试结果计算
    
    Parameters:
    -----------
    use_tuned_params : bool
        是否使用优化后的参数
    """
    print("\n" + "=" * 70)
    print("STEP 1: Computing Unified Metrics (Measured Robustness)")
    print("=" * 70)
    
    # 加载 Task2.3 已有的 RANK/PERCENT/SAVE 指标（作为基准）
    task23_metrics_path = repo_root / "task2.3" / "table" / "method_metrics.csv"
    task23_metrics = pd.read_csv(str(task23_metrics_path), index_col=0)
    print("\n  Loaded Task2.3 baseline metrics:")
    print(task23_metrics)
    
    # 加载 Task2.3 的 FFI 数据（用于 TWO_KEY 对比）
    ffi_base_path = repo_root / "task1" / "table" / "fan_vote_shares_analysis.csv"
    ffi_base = pd.read_csv(str(ffi_base_path))
    
    # 加载 Task2 的 controversy 数据
    controversy_path = repo_root / "task2.2" / "table" / "controversy_all_contestants.csv"
    if controversy_path.exists():
        controversy_df = pd.read_csv(str(controversy_path))
    else:
        controversy_df = ffi_base  # fallback
    
    # 加载 panel 数据
    fan_df, judge_df, data_df = load_all_data(str(repo_root))
    panel = create_weekly_panel(fan_df, judge_df)
    
    # 尝试加载优化参数
    twokey_params = {'alpha': 0.4, 'beta': 0.6, 'ban_consecutive_weeks': 2, 'save_eligibility_threshold': 0.20, 'judge_weight': 1.2}
    if use_tuned_params:
        params_path = repo_root / "task4" / "table" / "two_key_best_params.json"
        if params_path.exists():
            import json
            with open(str(params_path), 'r', encoding='utf-8') as f:
                tuned_data = json.load(f)
                twokey_params = tuned_data['params']
            print(f"\n  Using tuned params: {twokey_params}")
    
    # 为 TWO_KEY 计算 FFI
    ffi_two_key_df = compute_ffi_two_key_by_week(panel, seed=42, params=twokey_params)
    
    # 构造一个临时 FFI DataFrame 用于 Task2.3 函数
    ffi_for_twokey = ffi_two_key_df.rename(columns={'ffi_two_key': 'ffi_rank'})
    
    # 计算 TWO_KEY 的 legitimacy & engagement（复用 Task2.3 函数）
    twokey_legitimacy = compute_legitimacy_metric(ffi_for_twokey, controversy_df, 'ffi_rank')
    twokey_engagement = compute_engagement_metric(ffi_for_twokey, controversy_df, 'ffi_rank')
    
    # 计算 TWO_KEY 的 robustness（使用真实测量结果）
    print("\n  Loading TWO_KEY measured robustness...")
    
    robustness_test_path = repo_root / "task4" / "table" / "two_key_robustness_tests.json"
    if robustness_test_path.exists():
        import json
        with open(str(robustness_test_path), 'r', encoding='utf-8') as f:
            rob_data = json.load(f)
        twokey_robustness = rob_data['combined_robustness']
        print(f"    Measured robustness (flip-rate + attack): {twokey_robustness:.4f}")
        print(f"      - Flip-rate component: {rob_data['flip_rate_robustness']:.4f}")
        print(f"      - Attack component: {rob_data['attack_robustness']:.4f}")
        print(f"      - Mean extra weeks under attack: {rob_data['mean_extra_weeks']:.2f}")
        print(f"      - Attack success rate: {rob_data['attack_success_rate']:.1%}")
    else:
        # Fallback（但应该不会到这里）
        print("    WARNING: Robustness test results not found, using fallback")
        twokey_robustness = 0.75
    
    # Transparency（基于复杂度）
    twokey_transparency = 0.65
    
    # 汇总 TWO_KEY 指标
    twokey_metrics = {
        'legitimacy': twokey_legitimacy,
        'engagement': twokey_engagement,
        'robustness': twokey_robustness,
        'transparency': twokey_transparency
    }
    
    # 合并为完整指标表
    full_metrics = task23_metrics.copy()
    full_metrics.loc['TWO_KEY'] = twokey_metrics
    
    print("\n  Full Metrics (Task2.3 + TWO_KEY):")
    print(full_metrics)
    
    return full_metrics, rob_data if robustness_test_path.exists() else {}


# ============================================================
# 权重敏感性分析 + 效用采样（用于 regime-shift 可视化）
# ============================================================

def weight_sensitivity_and_utility_samples(metrics_df, n_samples=5000, seed=42):
    """
    权重敏感性分析 + 保存完整效用采样结果
    
    Returns:
    --------
    win_rates: dict, 各方法胜率
    utility_samples_df: DataFrame with columns [sample_id, method, w_legitimacy, w_engagement, w_robustness, w_transparency, utility]
    """
    print("\n" + "=" * 70)
    print("STEP 2: Weight Sensitivity + Utility Sampling")
    print("=" * 70)
    
    np.random.seed(seed)
    
    methods = list(metrics_df.index)
    n_metrics = len(metrics_df.columns)
    metric_names = list(metrics_df.columns)
    
    # Dirichlet 采样
    weights_samples = np.random.dirichlet(np.ones(n_metrics), n_samples)
    
    # 计算每个样本下各方法的效用
    utility_records = []
    win_counts = {m: 0 for m in methods}
    
    for sample_id, weights in enumerate(weights_samples):
        utilities = {}
        for method in methods:
            utility = sum(w * metrics_df.loc[method, col] for w, col in zip(weights, metric_names))
            utilities[method] = utility
            
            utility_records.append({
                'sample_id': sample_id,
                'method': method,
                'w_legitimacy': weights[0],
                'w_engagement': weights[1],
                'w_robustness': weights[2],
                'w_transparency': weights[3],
                'utility': utility
            })
        
        # 找出胜者
        winner = max(utilities, key=utilities.get)
        win_counts[winner] += 1
    
    # 胜率
    win_rates = {m: (count / n_samples) * 100 for m, count in win_counts.items()}
    
    print(f"\n  Win Rates (across {n_samples} samples):")
    for method in sorted(win_rates, key=win_rates.get, reverse=True):
        print(f"    {method:10s}: {win_rates[method]:6.2f}%")
    
    utility_samples_df = pd.DataFrame(utility_records)
    
    return win_rates, utility_samples_df


# ============================================================
# 主执行函数
# ============================================================

def main(use_tuned_params=True):
    print("\n" + "=" * 70)
    print("TASK 4: UNIFIED METRICS VALIDATION (Measured Robustness)")
    print("=" * 70)
    
    # 计算统一指标（使用优化参数）
    metrics_df, robustness_report = compute_all_metrics_unified(use_tuned_params)
    
    # 权重敏感性 + 效用采样
    win_rates, utility_samples_df = weight_sensitivity_and_utility_samples(
        metrics_df,
        n_samples=5000,
        seed=42
    )
    
    # 保存结果
    output_dir = repo_root / "task4" / "table"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / "four_methods_metrics.csv"
    metrics_df.to_csv(str(metrics_path))
    print(f"\n  Saved: {metrics_path}")
    
    win_rates_df = pd.DataFrame([win_rates])
    win_rates_path = output_dir / "weight_sensitivity_results.csv"
    win_rates_df.to_csv(str(win_rates_path), index=False)
    print(f"  Saved: {win_rates_path}")
    
    utility_path = output_dir / "utility_samples.csv"
    utility_samples_df.to_csv(str(utility_path), index=False)
    print(f"  Saved: {utility_path} ({len(utility_samples_df)} records)")
    
    print("\n" + "=" * 70)
    print("UNIFIED METRICS VALIDATION COMPLETED!")
    print("=" * 70)
    
    return metrics_df, win_rates, utility_samples_df


if __name__ == "__main__":
    metrics, win_rates, utility_samples = main()
