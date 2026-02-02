"""
MCM 2026 Problem C - Task 4: Robustness Testing for TWO_KEY
==========================================================
实现可测量的 Robustness（与 Task2.3 口径一致）：
  1. Flip-rate test（随机噪声扰动下的淘汰翻转率）
  2. Adversarial attack test（定向攻击下的额外存活周数）
  
定义：Robustness 衡量"抗有组织刷票/操纵的结构能力"
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 导入 TWO_KEY 系统
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "task4" / "src"))

from two_key_system import simulate_two_key_week, load_all_data, create_weekly_panel


# ============================================================
# 1. Flip-Rate Test（扩展 Task2.3 框架支持 TWO_KEY）
# ============================================================

def compute_flip_rate_two_key(panel_df, perturbation_levels=None, n_trials=30, seed=42, params=None):
    """
    为 TWO_KEY 计算 flip-rate robustness（与 Task2.3 口径一致）
    
    对每个淘汰周：
      1. 记录 baseline 淘汰者
      2. 多次扰动 fan_vote_share（加噪声并重新归一化）
      3. 重跑 TWO_KEY 淘汰逻辑
      4. 统计淘汰者翻转概率
    
    Parameters:
    -----------
    params : dict
        TWO_KEY 参数（alpha, beta, ban_consecutive_weeks）
    
    Returns:
    --------
    dict with flip_rates, robustness_score
    """
    if perturbation_levels is None:
        perturbation_levels = [0.01, 0.03, 0.05]
    
    if params is None:
        params = {'alpha': 0.4, 'beta': 0.6, 'ban_consecutive_weeks': 2}
    
    rng = np.random.default_rng(seed)
    
    # 获取所有淘汰周（≥3 人）
    valid_weeks = []
    for (season, week), group in panel_df.groupby(['season', 'week']):
        if len(group) >= 3:
            valid_weeks.append((int(season), int(week)))
    
    valid_weeks = sorted(valid_weeks)
    
    print(f"\n  Testing TWO_KEY flip-rate on {len(valid_weeks)} elimination weeks...")
    print(f"    Perturbation levels: {perturbation_levels}")
    print(f"    Trials per week: {n_trials}")
    
    flip_rates = {}
    
    for delta in perturbation_levels:
        flips = 0
        trials = 0
        
        for season, week in valid_weeks:
            # Baseline（未扰动）
            history = panel_df[panel_df['season'] == season].copy()
            baseline_result = simulate_two_key_week(season, week, panel_df, history, seed=seed, params=params)
            baseline_elim = baseline_result.get('eliminated')
            
            if baseline_elim is None:
                continue
            
            # 扰动测试
            week_data = panel_df[(panel_df['season'] == season) & (panel_df['week'] == week)].copy()
            shares = week_data['fan_vote_share'].to_numpy(dtype=float)
            n = shares.size
            
            for trial in range(n_trials):
                # 加噪声
                noise = rng.uniform(-delta, delta, size=n)
                pert = shares + noise
                pert = np.clip(pert, 1e-6, None)
                pert = pert / pert.sum()
                
                # 构造扰动后的 panel
                panel_pert = panel_df.copy()
                week_mask = (panel_pert['season'] == season) & (panel_pert['week'] == week)
                panel_pert.loc[week_mask, 'fan_vote_share'] = pert
                panel_pert.loc[week_mask, 'fan_share'] = pert
                
                # 重跑淘汰
                history_pert = panel_pert[panel_pert['season'] == season].copy()
                pert_result = simulate_two_key_week(season, week, panel_pert, history_pert, seed=seed, params=params)
                pert_elim = pert_result.get('eliminated')
                
                if pert_elim is None:
                    continue
                
                trials += 1
                if pert_elim != baseline_elim:
                    flips += 1
        
        flip_rate = flips / trials if trials > 0 else 0.0
        flip_rates[delta] = flip_rate
        print(f"    δ={delta:.2f}: flip_rate={flip_rate:.3f} ({flips}/{trials} flipped)")
    
    # Robustness = 1 - 平均 flip_rate
    avg_flip = np.mean(list(flip_rates.values()))
    robustness = 1.0 - avg_flip
    
    print(f"\n  TWO_KEY Robustness (flip-rate): {robustness:.3f}")
    
    return {
        'flip_rates': flip_rates,
        'robustness_score': robustness,
        'n_weeks': len(valid_weeks),
        'n_trials': n_trials,
    }


# ============================================================
# 2. Adversarial Attack Test（定向对抗测试）
# ============================================================

def run_adversarial_attack_test(panel_df, seed=42, params=None):
    """
    对抗攻击测试：每赛季选一个"评委累计分最低者"作为攻击目标，
    攻击者每周把该目标的 fan_share 拉到当周最高。
    
    度量：
      - 额外存活周数（相对 baseline）
      - 攻击成功率（目标存活超过 baseline 的赛季比例）
    
    Returns:
    --------
    dict with attack_results, robustness_score
    """
    if params is None:
        params = {'alpha': 0.4, 'beta': 0.6, 'ban_consecutive_weeks': 2}
    
    print("\n" + "=" * 70)
    print("ADVERSARIAL ATTACK TEST for TWO_KEY")
    print("=" * 70)
    
    seasons = sorted(panel_df['season'].unique())
    attack_results = []
    
    for season in seasons:
        season_panel = panel_df[panel_df['season'] == season].copy()
        
        # 找到累计评委分最低者（作为攻击目标）
        cumulative_judge = season_panel.groupby('celebrity_name')['judge_score'].sum()
        if len(cumulative_judge) == 0:
            continue
        
        target = cumulative_judge.idxmin()
        
        # Baseline（无攻击）
        baseline_weeks = simulate_season_survival(season, target, season_panel, seed=seed, params=params, attack=False)
        
        # Attacked（每周拉高 fan_share）
        attack_weeks = simulate_season_survival(season, target, season_panel, seed=seed, params=params, attack=True)
        
        extra_weeks = attack_weeks - baseline_weeks
        
        attack_results.append({
            'season': season,
            'target': target,
            'baseline_weeks': baseline_weeks,
            'attack_weeks': attack_weeks,
            'extra_weeks': extra_weeks,
            'attack_success': 1 if extra_weeks > 0 else 0,
        })
        
        print(f"  S{season:2d} | Target: {target:25s} | Baseline: {baseline_weeks} weeks | Attack: {attack_weeks} weeks | Extra: +{extra_weeks}")
    
    results_df = pd.DataFrame(attack_results)
    
    # 统计
    mean_extra = results_df['extra_weeks'].mean()
    success_rate = results_df['attack_success'].mean()
    
    print(f"\n  Summary:")
    print(f"    Mean extra survival weeks: {mean_extra:.2f}")
    print(f"    Attack success rate: {success_rate:.1%}")
    
    # 映射为 0-1 分数（越少额外周数 = 越 robust）
    # 参考：RANK/PERCENT 可能让目标多活 2-3 周，SAVE 约 0.5-1 周
    # 映射公式：robustness = max(0, 1 - extra_weeks / 3.0)
    robustness_from_attack = float(np.clip(1.0 - mean_extra / 3.0, 0, 1))
    
    print(f"    Robustness (attack-based): {robustness_from_attack:.3f}")
    
    return {
        'attack_results': results_df,
        'mean_extra_weeks': mean_extra,
        'success_rate': success_rate,
        'robustness_score': robustness_from_attack,
    }


def simulate_season_survival(season, target, season_panel, seed, params, attack=False):
    """
    模拟目标选手在某赛季的存活周数
    
    Parameters:
    -----------
    attack : bool
        若为 True，每周把目标的 fan_share 拉到当周最高
    
    Returns:
    --------
    int: 存活周数（被淘汰前的周数）
    """
    weeks = sorted(season_panel['week'].unique())
    remaining = set(season_panel['celebrity_name'].unique())
    
    if target not in remaining:
        return 0
    
    survival_weeks = 0
    
    for week in weeks:
        week_data = season_panel[season_panel['week'] == week].copy()
        week_contestants = set(week_data['celebrity_name'].values) & remaining
        
        if len(week_contestants) <= 2:
            break
        
        if target not in week_contestants:
            break
        
        # 如果开启攻击，拉高目标的 fan_share
        if attack:
            week_idx = week_data.index
            shares = week_data['fan_vote_share'].to_numpy(dtype=float)
            
            # 找到目标的位置
            target_idx_in_week = week_data[week_data['celebrity_name'] == target].index
            if len(target_idx_in_week) > 0:
                target_pos = list(week_data.index).index(target_idx_in_week[0])
                
                # 把目标拉到当周最高（+5% margin）
                max_share = shares.max()
                shares[target_pos] = max_share + 0.05
                
                # 重新归一化
                shares = np.clip(shares, 1e-6, None)
                shares = shares / shares.sum()
                
                # 更新 panel
                season_panel.loc[week_idx, 'fan_vote_share'] = shares
                season_panel.loc[week_idx, 'fan_share'] = shares
        
        # 模拟本周淘汰
        history = season_panel.copy()
        result = simulate_two_key_week(season, week, season_panel, history, seed=seed, params=params)
        eliminated = result.get('eliminated')
        
        if eliminated is None:
            survival_weeks += 1
            continue
        
        # 检查目标是否被淘汰
        if eliminated == target:
            return survival_weeks
        
        # 移除被淘汰者
        remaining.discard(eliminated)
        survival_weeks += 1
    
    # 目标存活到赛季结束
    return survival_weeks


# ============================================================
# 3. 综合 Robustness 评估（结合两种测试）
# ============================================================

def compute_combined_robustness(flip_result, attack_result, weight_flip=0.6, weight_attack=0.4):
    """
    综合两种测试给出最终 Robustness
    
    Parameters:
    -----------
    weight_flip : float
        flip-rate 测试的权重
    weight_attack : float
        对抗攻击测试的权重
    
    Returns:
    --------
    float: 综合 robustness 分数
    """
    rob_flip = flip_result['robustness_score']
    rob_attack = attack_result['robustness_score']
    
    combined = weight_flip * rob_flip + weight_attack * rob_attack
    
    print(f"\n  Combined Robustness:")
    print(f"    Flip-rate component: {rob_flip:.3f} (weight {weight_flip})")
    print(f"    Attack-test component: {rob_attack:.3f} (weight {weight_attack})")
    print(f"    Final: {combined:.3f}")
    
    return combined


# ============================================================
# Main
# ============================================================

def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    # 加载数据
    fan_df, judge_df, data_df = load_all_data(str(repo_root))
    panel = create_weekly_panel(fan_df, judge_df)
    
    # 加载优化参数
    params_path = repo_root / "task4" / "table" / "two_key_best_params.json"
    if params_path.exists():
        import json
        with open(params_path, 'r', encoding='utf-8') as f:
            best_params_data = json.load(f)
            params = best_params_data['params']
        print(f"\n  Loaded params: {params}")
    else:
        params = {'alpha': 0.4, 'beta': 0.6, 'ban_consecutive_weeks': 2}
        print(f"\n  Using default params: {params}")
    
    # 测试 1：Flip-rate
    print("\n" + "=" * 70)
    print("TEST 1: Flip-Rate under Random Perturbations")
    print("=" * 70)
    flip_result = compute_flip_rate_two_key(
        panel,
        perturbation_levels=[0.01, 0.03, 0.05],
        n_trials=30,
        seed=42,
        params=params
    )
    
    # 测试 2：对抗攻击
    print("\n" + "=" * 70)
    print("TEST 2: Adversarial Attack (Boost Weakest Judge-Score Contestant)")
    print("=" * 70)
    attack_result = run_adversarial_attack_test(panel, seed=42, params=params)
    
    # 综合 Robustness
    combined_robustness = compute_combined_robustness(flip_result, attack_result, weight_flip=0.6, weight_attack=0.4)
    
    # 保存结果
    output_dir = repo_root / "task4" / "table"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存对抗测试详情
    attack_result['attack_results'].to_csv(
        output_dir / "two_key_attack_test.csv",
        index=False
    )
    
    # 保存汇总
    summary = {
        'flip_rate_robustness': flip_result['robustness_score'],
        'attack_robustness': attack_result['robustness_score'],
        'combined_robustness': combined_robustness,
        'mean_extra_weeks': attack_result['mean_extra_weeks'],
        'attack_success_rate': attack_result['success_rate'],
        'flip_rates': flip_result['flip_rates'],
    }
    
    import json
    with open(output_dir / "two_key_robustness_tests.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("ROBUSTNESS TESTING COMPLETED!")
    print("=" * 70)
    print(f"  Results saved to: task4/table/")
    
    return summary


if __name__ == "__main__":
    main()
