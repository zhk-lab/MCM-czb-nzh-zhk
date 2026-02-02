"""
MCM 2026 Problem C - Task 4: TWO_KEY Parameter Tuning (Measured Robustness)
===========================================================================
在 regime-shift 目标下优化 TWO_KEY 参数：
  - 目标1：最大化后争议阶段效用（legitimacy+robustness 高权重）
  - 约束：engagement >= 0.75（避免变成纯评委制）
  - **Robustness 使用真实测量（flip-rate）**，不是 β 的启发式
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "task4" / "src"))
sys.path.insert(0, str(repo_root / "task2.3" / "src"))

from two_key_system import load_all_data, create_weekly_panel
from robustness_testing import compute_flip_rate_two_key
from robust_recommendation import compute_legitimacy_metric, compute_engagement_metric


# ============================================================
# 参数空间定义
# ============================================================

def generate_param_grid():
    """生成参数网格（包含新参数）"""
    param_grid = {
        'alpha': [0.3, 0.4, 0.5],  # 风险分系数
        'beta': [0.5, 0.6, 0.7],   # 双钥匙区 bonus
        'ban_consecutive_weeks': [2, 3],  # 禁止救援触发周数
        'save_eligibility_threshold': [0.15, 0.20],  # 救援资格底线
        'judge_weight': [1.2, 1.3],  # 评委权重加成
    }
    
    # 生成所有组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    param_combos = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        param_combos.append(params)
    
    return param_combos


# ============================================================
# 计算参数组合的指标（快速版）
# ============================================================

def compute_metrics_for_params(params, panel_df, controversy_df):
    """
    对给定参数计算 TWO_KEY 的四大指标（使用真实 robustness 测试）
    
    Returns:
    --------
    dict with keys: legitimacy, engagement, robustness, transparency
    """
    from scipy.stats import kendalltau
    
    # 计算 FFI（使用参数化的 TWO_KEY）
    seasons = sorted(panel_df['season'].unique())
    ffi_records = []
    
    for season in seasons[:10]:  # 只测试前10个赛季，加速
        season_data = panel_df[panel_df['season'] == season]
        weeks = sorted(season_data['week'].unique())
        
        for week in weeks[:6]:  # 每赛季只测试前6周
            week_data = season_data[season_data['week'] == week].copy()
            
            if len(week_data) < 3:
                continue
            
            n = len(week_data)
            
            # 构造 TWO_KEY 的风险优先级排名（使用参数化风险分）
            week_data['judge_rank'] = week_data['judge_score'].rank(ascending=False, method='min')
            week_data['fan_rank'] = week_data['fan_share'].rank(ascending=False, method='min')
            week_data['p_J'] = (week_data['judge_rank'] - 1) / (n - 1) if n > 1 else 0
            week_data['p_F'] = (week_data['fan_rank'] - 1) / (n - 1) if n > 1 else 0
            
            # 使用参数化的 alpha 和 judge_weight
            judge_weight = params.get('judge_weight', 1.2)
            week_data['risk'] = np.maximum(judge_weight * week_data['p_J'], week_data['p_F']) + params['alpha'] * np.minimum(week_data['p_J'], week_data['p_F'])
            
            # 排序
            week_data = week_data.sort_values('risk')
            two_key_ranking = list(week_data['celebrity_name'].values)
            
            # 评委/粉丝排名
            judge_scores = dict(zip(week_data['celebrity_name'], week_data['judge_score']))
            judge_ranking = sorted(judge_scores, key=judge_scores.get, reverse=True)
            
            fan_shares = dict(zip(week_data['celebrity_name'], week_data['fan_share']))
            fan_ranking = sorted(fan_shares, key=fan_shares.get, reverse=True)
            
            # 计算 Kendall 距离
            common = set(two_key_ranking) & set(judge_ranking) & set(fan_ranking)
            if len(common) < 2:
                continue
            
            tau_judge, _ = kendalltau(
                [two_key_ranking.index(c) for c in common],
                [judge_ranking.index(c) for c in common]
            )
            d_judge = (1 - tau_judge) / 2
            
            tau_fan, _ = kendalltau(
                [two_key_ranking.index(c) for c in common],
                [fan_ranking.index(c) for c in common]
            )
            d_fan = (1 - tau_fan) / 2
            
            ffi = d_judge - d_fan
            ffi_records.append({'ffi_rank': ffi})
    
    if len(ffi_records) == 0:
        return None
    
    ffi_df = pd.DataFrame(ffi_records)
    
    # 计算指标
    legitimacy = compute_legitimacy_metric(ffi_df, controversy_df, 'ffi_rank')
    engagement = compute_engagement_metric(ffi_df, controversy_df, 'ffi_rank')
    
    # Robustness 使用真实 flip-rate 测试（快速版：少量周）
    print(".", end='', flush=True)
    flip_result = compute_flip_rate_two_key(
        panel_df,
        perturbation_levels=[0.03],  # 只测中等扰动，加速
        n_trials=10,  # 减少试验次数
        seed=42,
        params=params
    )
    robustness = flip_result['robustness_score']
    
    # Transparency
    transparency = 0.65
    
    return {
        'legitimacy': legitimacy,
        'engagement': engagement,
        'robustness': robustness,
        'transparency': transparency
    }


# ============================================================
# 目标函数：regime-shift 效用
# ============================================================

def evaluate_params(params, metrics):
    """
    评估参数组合的综合表现
    
    目标：
    1. 后争议阶段效用最大化（legitimacy 0.5, robustness 0.4, engagement 0.05, transparency 0.05）
    2. 跨阶段最差效用最大化（min-max 稳健）
    3. engagement >= 0.75（硬约束）
    
    Returns:
    --------
    score : float（越大越好）
    """
    if metrics is None:
        return -999
    
    # 约束检查
    if metrics['engagement'] < 0.75:
        return -999  # 违反约束
    
    # 后争议权重（高 legitimacy + robustness）
    w_post = np.array([0.50, 0.05, 0.40, 0.05])  # [legitimacy, engagement, robustness, transparency]
    metric_vec = np.array([metrics['legitimacy'], metrics['engagement'], 
                           metrics['robustness'], metrics['transparency']])
    
    utility_post = np.dot(w_post, metric_vec)
    
    # 前争议权重（高 engagement）
    w_pre = np.array([0.10, 0.60, 0.10, 0.20])
    utility_pre = np.dot(w_pre, metric_vec)
    
    # 目标：后争议效用高 + 前争议也不能太差
    score = utility_post + 0.3 * utility_pre
    
    return score


# ============================================================
# 网格搜索
# ============================================================

def grid_search_best_params():
    """网格搜索最优参数（使用真实 robustness 测试）"""
    print("\n" + "=" * 70)
    print("TWO_KEY PARAMETER TUNING (Measured Robustness)")
    print("=" * 70)
    
    # 加载数据
    fan_df, judge_df, data_df = load_all_data(str(repo_root))
    panel = create_weekly_panel(fan_df, judge_df)
    
    controversy_path = repo_root / "task2.2" / "table" / "controversy_all_contestants.csv"
    if controversy_path.exists():
        controversy_df = pd.read_csv(str(controversy_path))
    else:
        controversy_df = pd.read_csv(str(repo_root / "task1" / "table" / "fan_vote_shares_analysis.csv"))
    
    # 生成参数网格
    param_combos = generate_param_grid()
    print(f"\n  Testing {len(param_combos)} parameter combinations...")
    
    # 测试所有组合
    results = []
    
    for idx, params in enumerate(param_combos):
        print(f"\r  Progress: {idx+1}/{len(param_combos)}", end='')
        
        # 计算指标
        metrics = compute_metrics_for_params(params, panel, controversy_df)
        
        if metrics is None:
            continue
        
        # 计算得分
        score = evaluate_params(params, metrics)
        
        results.append({
            'alpha': params['alpha'],
            'beta': params['beta'],
            'ban_consecutive_weeks': params['ban_consecutive_weeks'],
            'save_eligibility_threshold': params.get('save_eligibility_threshold', 0.20),
            'judge_weight': params.get('judge_weight', 1.2),
            'legitimacy': metrics['legitimacy'],
            'engagement': metrics['engagement'],
            'robustness': metrics['robustness'],
            'transparency': metrics['transparency'],
            'score': score
        })
    
    print()  # 换行
    
    results_df = pd.DataFrame(results)
    
    # 找出最优参数
    best_idx = results_df['score'].idxmax()
    best_params = {
        'alpha': results_df.loc[best_idx, 'alpha'],
        'beta': results_df.loc[best_idx, 'beta'],
        'ban_consecutive_weeks': int(results_df.loc[best_idx, 'ban_consecutive_weeks']),
        'save_eligibility_threshold': results_df.loc[best_idx, 'save_eligibility_threshold'],
        'judge_weight': results_df.loc[best_idx, 'judge_weight'],
    }
    best_metrics = {
        'legitimacy': results_df.loc[best_idx, 'legitimacy'],
        'engagement': results_df.loc[best_idx, 'engagement'],
        'robustness': results_df.loc[best_idx, 'robustness'],
        'transparency': results_df.loc[best_idx, 'transparency'],
    }
    best_score = results_df.loc[best_idx, 'score']
    
    print("\n  Best Parameters Found:")
    print(f"    alpha (risk coef):             {best_params['alpha']}")
    print(f"    beta (two-key bonus):          {best_params['beta']}")
    print(f"    ban_consecutive_weeks:         {best_params['ban_consecutive_weeks']}")
    print(f"    save_eligibility_threshold:    {best_params.get('save_eligibility_threshold', 0.20)}")
    print(f"    judge_weight:                  {best_params.get('judge_weight', 1.2)}")
    print(f"\n  Best Metrics:")
    print(f"    Legitimacy:   {best_metrics['legitimacy']:.4f}")
    print(f"    Engagement:   {best_metrics['engagement']:.4f}")
    print(f"    Robustness:   {best_metrics['robustness']:.4f} (MEASURED)")
    print(f"    Transparency: {best_metrics['transparency']:.4f}")
    print(f"\n  Best Score (regime-shift objective): {best_score:.4f}")
    
    # 保存结果
    output_dir = repo_root / "task4" / "table"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "two_key_tuning_results.csv"
    results_df.to_csv(str(results_path), index=False)
    print(f"\n  Saved tuning results: {results_path}")
    
    params_path = output_dir / "two_key_best_params.json"
    with open(str(params_path), 'w', encoding='utf-8') as f:
        json.dump({'params': best_params, 'metrics': best_metrics, 'score': best_score}, f, indent=2, ensure_ascii=False)
    print(f"  Saved best params: {params_path}")
    
    return best_params, best_metrics, results_df


# ============================================================
# 主执行函数
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("TASK 4: TWO_KEY PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    best_params, best_metrics, results_df = grid_search_best_params()
    
    print("\n" + "=" * 70)
    print("PARAMETER TUNING COMPLETED!")
    print("=" * 70)
    
    return best_params, best_metrics, results_df


if __name__ == "__main__":
    best_params, best_metrics, results_df = main()
