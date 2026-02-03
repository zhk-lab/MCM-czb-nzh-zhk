"""
MCM 2026 Problem C - Fan Vote Estimation (优化版 V2)
=====================================================
两步模型：
  Step 1: 贝叶斯逆问题 + Hit-and-Run MCMC 可行域采样
  Step 2: HMM + Viterbi 最大后验路径推断

优化改进：
  - 更精确的可行域约束（严格满足淘汰规则）
  - 引入明星人气先验（基于行业、年龄等特征）
  - 改进的转移概率和观测似然
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 固定随机种子，确保结果可重复
np.random.seed(42)

plt.rcParams['font.family'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

# Paper-style palette (matched to image.png)
PAPER_LIGHT = '#cad3e0'
PAPER_MID = '#9cb0ce'
PAPER_DARK = '#45618a'
PAPER_GRAY = '#b4b4b6'
PAPER_TEXT = '#222222'
PAPER_CMAP = LinearSegmentedColormap.from_list(
    'paper_blues', [PAPER_LIGHT, PAPER_MID, PAPER_DARK]
)


def apply_shadow_to_patches(patches, offset=(2, -2), alpha=0.25):
    """Apply subtle paper-style shadow to bars/patches."""
    for patch in patches:
        patch.set_path_effects([
            pe.SimplePatchShadow(offset=offset, alpha=alpha, shadow_rgbFace=(0, 0, 0)),
            pe.Normal(),
        ])

# ============================================================
# 1. 数据预处理
# ============================================================

def load_and_preprocess_data(filepath: str) -> Dict:
    """加载并预处理数据"""
    print("=" * 60)
    print("数据预处理")
    print("=" * 60)
    
    seasons_data = defaultdict(lambda: {
        'contestants': [],
        'weeks': defaultdict(lambda: {'scores': {}, 'eliminated': None, 'multi_elim': False})
    })
    
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            season = int(row['season'])
            name = row['celebrity_name']
            result = row['results']
            placement = int(row['placement'])
            
            # 行业人气权重（先验）
            industry = row['celebrity_industry']
            industry_weights = {
                'Actor/Actress': 1.2, 'Singer/Rapper': 1.3, 'Athlete': 1.1,
                'TV Personality': 1.15, 'Model': 1.0, 'Comedian': 0.95,
                'Social Media Personality': 1.25, 'News Anchor': 0.9
            }
            pop_weight = industry_weights.get(industry, 1.0)
            
            try:
                age = int(row['celebrity_age_during_season'])
                # 年龄调整：20-40岁最受欢迎
                if 20 <= age <= 40:
                    pop_weight *= 1.1
                elif age > 60:
                    pop_weight *= 0.9
            except:
                age = 35
            
            contestant_info = {
                'name': name,
                'result': result,
                'placement': placement,
                'industry': industry,
                'age': age,
                'popularity_prior': pop_weight
            }
            seasons_data[season]['contestants'].append(contestant_info)
            
            # 解析每周评委分数
            for week in range(1, 12):
                scores = []
                for judge in range(1, 5):
                    key = f'week{week}_judge{judge}_score'
                    if key in row:
                        val = row[key].strip()
                        if val and val.upper() != 'N/A':
                            try:
                                scores.append(float(val))
                            except ValueError:
                                pass
                
                if scores and sum(scores) > 0:
                    total_score = sum(scores)
                    seasons_data[season]['weeks'][week]['scores'][name] = total_score
            
            # 识别淘汰周
            if 'Eliminated Week' in result:
                try:
                    elim_week = int(result.split('Week')[-1].strip())
                    if seasons_data[season]['weeks'][elim_week]['eliminated']:
                        # 多人淘汰
                        seasons_data[season]['weeks'][elim_week]['multi_elim'] = True
                        existing = seasons_data[season]['weeks'][elim_week]['eliminated']
                        if isinstance(existing, list):
                            existing.append(name)
                        else:
                            seasons_data[season]['weeks'][elim_week]['eliminated'] = [existing, name]
                    else:
                        seasons_data[season]['weeks'][elim_week]['eliminated'] = name
                except:
                    pass
    
    print(f"  - 共加载 {len(seasons_data)} 个赛季")
    total_contestants = sum(len(s['contestants']) for s in seasons_data.values())
    print(f"  - 共 {total_contestants} 位选手记录")
    
    valid_seasons = []
    for season in sorted(seasons_data.keys()):
        data = seasons_data[season]
        weeks_with_elim = sum(1 for w in data['weeks'].values() if w['eliminated'])
        if weeks_with_elim >= 3:
            valid_seasons.append(season)
    
    print(f"  - 有效赛季: {len(valid_seasons)} 个")
    print()
    
    return dict(seasons_data), valid_seasons


def get_season_weekly_data(season_data: Dict, season_num: int) -> List[Dict]:
    """获取某赛季每周的结构化数据"""
    weeks_data = []
    all_contestants = {c['name']: c for c in season_data['contestants']}
    eliminated_so_far = set()
    
    for week in range(1, 12):
        week_info = season_data['weeks'].get(week, {})
        scores = week_info.get('scores', {})
        eliminated = week_info.get('eliminated')
        
        if not scores:
            continue
        
        active = [c for c in all_contestants.keys() if c not in eliminated_so_far and c in scores]
        
        if len(active) < 2:
            continue
        
        # 获取先验人气权重
        priors = {c: all_contestants[c]['popularity_prior'] for c in active}
        
        weeks_data.append({
            'week': week,
            'active': active,
            'scores': {c: scores[c] for c in active if c in scores},
            'eliminated': eliminated,
            'priors': priors
        })
        
        if eliminated:
            if isinstance(eliminated, list):
                eliminated_so_far.update(eliminated)
            else:
                eliminated_so_far.add(eliminated)
    
    return weeks_data


# ============================================================
# 2. Step 1: 改进的可行域采样
# ============================================================

def compute_combined_score(fan_shares: np.ndarray, judge_scores: np.ndarray, 
                           method: str = 'percent') -> np.ndarray:
    """计算综合得分"""
    n = len(fan_shares)
    total_judge = judge_scores.sum()
    
    if method == 'percent':
        judge_pct = judge_scores / total_judge if total_judge > 0 else np.ones(n) / n
        return judge_pct + fan_shares
    else:  # rank
        judge_rank = np.argsort(np.argsort(-judge_scores)) + 1.0
        fan_rank = np.argsort(np.argsort(-fan_shares)) + 1.0
        return -(judge_rank + fan_rank)  # 负号：小排名=高得分


def hit_and_run_sample_v2(week_data: Dict, method: str = 'percent',
                          n_samples: int = 500, burn_in: int = 200) -> np.ndarray:
    """
    改进版 Hit-and-Run MCMC 采样
    
    改进：
    1. 严格检查可行域约束
    2. 使用先验初始化
    3. 自适应步长
    """
    active = week_data['active']
    scores = week_data['scores']
    eliminated = week_data['eliminated']
    priors = week_data['priors']
    n = len(active)
    
    if n < 2:
        return np.ones((n_samples, 1))
    
    # 处理单人或多人淘汰
    if isinstance(eliminated, list):
        elim_names = eliminated
    else:
        elim_names = [eliminated] if eliminated else []
    
    elim_indices = [active.index(e) for e in elim_names if e in active]
    
    if not elim_indices:
        # 无淘汰信息，返回基于先验的采样
        prior_weights = np.array([priors.get(c, 1.0) for c in active])
        prior_weights = prior_weights / prior_weights.sum()
        samples = np.random.dirichlet(prior_weights * 10, n_samples)
        return samples
    
    judge_scores = np.array([scores.get(c, 0) for c in active])
    
    def is_feasible(fan_share: np.ndarray) -> bool:
        """检查是否满足淘汰约束（放宽容差提高采样效率）"""
        if np.any(fan_share < -1e-10) or np.abs(fan_share.sum() - 1) > 1e-6:
            return False
        
        combined = compute_combined_score(fan_share, judge_scores, method)
        min_score = combined.min()
        score_range = combined.max() - min_score + 1e-10
        
        # 严格可行域：淘汰者得分必须是最低（或极接近最低）
        if method == 'rank':
            tolerance = max(1e-6, score_range * 0.001)  # 0.1% 严格容差
        else:
            tolerance = max(1e-6, score_range * 0.001)  # 0.1% 严格容差
        
        for idx in elim_indices:
            if combined[idx] > min_score + tolerance:
                return False
        
        return True
    
    # 初始化：基于先验，偏向让被淘汰者的粉丝投票较低
    prior_weights = np.array([priors.get(c, 1.0) for c in active])
    for idx in elim_indices:
        prior_weights[idx] *= 0.3  # 降低被淘汰者的初始权重
    
    pi = prior_weights / prior_weights.sum()
    
    # 确保初始点可行，否则搜索
    if not is_feasible(pi):
        # 网格搜索找可行起点
        found = False
        for _ in range(1000):
            test_pi = np.random.dirichlet(np.ones(n) * 2)
            if is_feasible(test_pi):
                pi = test_pi
                found = True
                break
        
        if not found:
            # 强制构造可行解
            pi = np.ones(n) / n
            for idx in elim_indices:
                pi[idx] = 0.01
            pi = pi / pi.sum()
    
    samples = []
    accept_count = 0
    
    for iteration in range(burn_in + n_samples * 5):  # 增加迭代次数
        # 生成随机方向（在单纯形上）
        direction = np.random.randn(n)
        direction = direction - direction.mean()
        
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction = direction / norm
        
        # 自适应步长
        step_scale = 0.1 if accept_count < 100 else 0.05
        
        # 找步长范围
        t_min, t_max = -1.0, 1.0
        for i in range(n):
            if direction[i] > 1e-10:
                t_max = min(t_max, (1 - pi[i]) / direction[i] * step_scale)
                t_min = max(t_min, -pi[i] / direction[i] * step_scale)
            elif direction[i] < -1e-10:
                t_min = max(t_min, (1 - pi[i]) / direction[i] * step_scale)
                t_max = min(t_max, -pi[i] / direction[i] * step_scale)
        
        if t_min >= t_max:
            continue
        
        # 随机选择步长
        t = np.random.uniform(t_min, t_max)
        new_pi = pi + t * direction
        new_pi = np.clip(new_pi, 1e-6, 1 - 1e-6)
        new_pi = new_pi / new_pi.sum()
        
        if is_feasible(new_pi):
            pi = new_pi
            accept_count += 1
        
        if iteration >= burn_in and len(samples) < n_samples:
            if is_feasible(pi):
                samples.append(pi.copy())
    
    # 如果采样不足，用扰动填充
    while len(samples) < n_samples:
        if samples:
            noise = np.random.randn(n) * 0.01
            new_sample = samples[-1] + noise
            new_sample = np.clip(new_sample, 1e-6, 1)
            new_sample = new_sample / new_sample.sum()
            if is_feasible(new_sample):
                samples.append(new_sample)
            else:
                samples.append(samples[np.random.randint(len(samples))].copy())
        else:
            samples.append(np.ones(n) / n)
    
    return np.array(samples[:n_samples])


def sample_season_feasible_region_v2(season_data: Dict, season_num: int,
                                      n_samples: int = 500) -> Dict:
    """对整个赛季进行可行域采样"""
    weeks_data = get_season_weekly_data(season_data, season_num)
    
    if season_num <= 2 or season_num >= 28:
        method = 'rank'
    else:
        method = 'percent'
    
    results = {
        'season': season_num,
        'method': method,
        'weeks': []
    }
    
    for week_data in weeks_data:
        samples = hit_and_run_sample_v2(week_data, method, n_samples)
        
        results['weeks'].append({
            'week': week_data['week'],
            'active': week_data['active'],
            'eliminated': week_data['eliminated'],
            'judge_scores': np.array([week_data['scores'].get(c, 0) for c in week_data['active']]),
            'priors': week_data['priors'],
            'samples': samples,
            'mean': samples.mean(axis=0),
            'std': samples.std(axis=0),
            'percentile_5': np.percentile(samples, 5, axis=0),
            'percentile_95': np.percentile(samples, 95, axis=0)
        })
    
    return results


# ============================================================
# 3. Step 2: 改进的 HMM + Viterbi
# ============================================================

def build_hmm_and_find_map_path_v2(sampled_data: Dict, n_states: int = 40) -> Dict:
    """
    改进版 HMM + Viterbi
    
    改进：
    1. 更合理的状态选择（聚类）
    2. 考虑先验的转移概率
    3. 淘汰一致性作为观测似然
    """
    weeks = sampled_data['weeks']
    method = sampled_data['method']
    
    if len(weeks) < 2:
        return {'path': [], 'log_prob': 0}
    
    # 为每周选取代表性状态
    week_states = []
    for week_info in weeks:
        samples = week_info['samples']
        n = min(n_states, len(samples))
        
        # 使用 K-means 风格的选择
        indices = np.linspace(0, len(samples)-1, n, dtype=int)
        states = samples[indices]
        week_states.append(states)
    
    T = len(weeks)
    
    # 计算每个状态的观测似然（淘汰一致性）
    def observation_likelihood(state: np.ndarray, week_info: Dict) -> float:
        """观测似然：状态是否正确预测淘汰者"""
        eliminated = week_info['eliminated']
        active = week_info['active']
        judge_scores = week_info['judge_scores']
        
        if eliminated is None:
            return 0.0
        
        elim_names = eliminated if isinstance(eliminated, list) else [eliminated]
        elim_indices = [active.index(e) for e in elim_names if e in active]
        
        if not elim_indices:
            return 0.0
        
        combined = compute_combined_score(state, judge_scores, method)
        min_score = combined.min()
        
        # 被淘汰者得分应该最低
        log_lik = 0.0
        # 最优参数：RANK用λ=1000（极强调淘汰一致性），PERCENT用λ=300
        weight = 1000 if method == 'rank' else 300
        for idx in elim_indices:
            diff = min_score - combined[idx]
            log_lik += diff * weight
        
        return log_lik
    
    # Viterbi 算法
    n_s0 = len(week_states[0])
    
    # 初始化：观测似然
    log_delta = [np.array([observation_likelihood(week_states[0][i], weeks[0]) 
                           for i in range(n_s0)])]
    psi = [np.zeros(n_s0, dtype=int)]
    
    # 递推
    for t in range(1, T):
        n_st = len(week_states[t])
        n_st_prev = len(week_states[t-1])
        
        log_delta_t = np.full(n_st, -np.inf)
        psi_t = np.zeros(n_st, dtype=int)
        
        # 共同选手
        active_prev = set(weeks[t-1]['active'])
        active_curr = set(weeks[t]['active'])
        common = list(active_prev & active_curr)
        
        for j in range(n_st):
            state_j = week_states[t][j]
            obs_lik = observation_likelihood(state_j, weeks[t])
            
            best_log_prob = -np.inf
            best_i = 0
            
            if not common:
                # 无共同选手
                for i in range(n_st_prev):
                    log_prob = log_delta[t-1][i] + obs_lik
                    if log_prob > best_log_prob:
                        best_log_prob = log_prob
                        best_i = i
            else:
                idx_prev = [weeks[t-1]['active'].index(c) for c in common]
                idx_curr = [weeks[t]['active'].index(c) for c in common]
                
                for i in range(n_st_prev):
                    state_i = week_states[t-1][i]
                    
                    # 平滑性转移概率：RANK用β=0.1（几乎不限制跳变），PERCENT用β=2
                    diff = state_j[idx_curr] - state_i[idx_prev]
                    smooth_weight = 0.1 if method == 'rank' else 2
                    smoothness = -np.sum(diff ** 2) * smooth_weight
                    
                    log_prob = log_delta[t-1][i] + smoothness + obs_lik
                    
                    if log_prob > best_log_prob:
                        best_log_prob = log_prob
                        best_i = i
            
            log_delta_t[j] = best_log_prob
            psi_t[j] = best_i
        
        log_delta.append(log_delta_t)
        psi.append(psi_t)
    
    # 回溯
    path_indices = [0] * T
    path_indices[T-1] = log_delta[T-1].argmax()
    
    for t in range(T-2, -1, -1):
        path_indices[t] = psi[t+1][path_indices[t+1]]
    
    # 提取路径
    map_path = []
    for t, idx in enumerate(path_indices):
        map_path.append({
            'week': weeks[t]['week'],
            'active': weeks[t]['active'],
            'eliminated': weeks[t]['eliminated'],
            'fan_shares': week_states[t][idx],
            'uncertainty': weeks[t]['std'],
            'ci_low': weeks[t]['percentile_5'],
            'ci_high': weeks[t]['percentile_95']
        })
    
    return {
        'path': map_path,
        'log_prob': log_delta[T-1].max()
    }


# ============================================================
# 4. 评估一致性
# ============================================================

def evaluate_consistency_v2(map_result: Dict, sampled_data: Dict) -> Dict:
    """评估模型一致性"""
    path = map_result['path']
    method = sampled_data['method']
    
    correct = 0
    total = 0
    week_metrics = []
    
    for i, week_info in enumerate(path):
        fan_shares = week_info['fan_shares']
        active = week_info['active']
        eliminated = week_info['eliminated']
        uncertainty = week_info['uncertainty']
        
        if eliminated is None:
            continue
        
        elim_names = eliminated if isinstance(eliminated, list) else [eliminated]
        elim_indices = [active.index(e) for e in elim_names if e in active]
        
        if not elim_indices:
            continue
        
        # 获取评委分数
        judge_scores = sampled_data['weeks'][i]['judge_scores']
        
        # 计算综合得分
        combined = compute_combined_score(fan_shares, judge_scores, method)
        
        # 找出所有得分最低的选手（处理 tie 情况）
        min_score = combined.min()
        tie_threshold = 1e-6  # 数值精度阈值
        min_indices = np.where(np.abs(combined - min_score) < tie_threshold)[0]
        pred_elim_candidates = [active[idx] for idx in min_indices]
        pred_elim = pred_elim_candidates[0]  # 主预测
        
        total += 1
        actual_elim = elim_names[0] if len(elim_names) == 1 else elim_names
        
        # 改进的判断逻辑：如果实际淘汰者在预测的最低得分集合中，算正确
        if isinstance(actual_elim, list):
            # 多人淘汰：只要有一个在预测集合中就算正确
            is_correct = any(e in pred_elim_candidates for e in actual_elim)
        else:
            # 单人淘汰：淘汰者在预测的最低得分集合中
            is_correct = actual_elim in pred_elim_candidates
        
        if is_correct:
            correct += 1
        
        # 不确定性度量
        mean_unc = uncertainty.mean()
        entropy = -np.sum(fan_shares * np.log(fan_shares + 1e-10))
        
        # 置信区间宽度
        ci_width = (week_info['ci_high'] - week_info['ci_low']).mean()
        
        week_metrics.append({
            'week': week_info['week'],
            'correct': is_correct,
            'predicted': pred_elim,
            'actual': actual_elim,
            'mean_uncertainty': mean_unc,
            'entropy': entropy,
            'ci_width': ci_width
        })
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'week_metrics': week_metrics
    }


# ============================================================
# 5. 可视化
# ============================================================

def visualize_results_v2(sampled_data: Dict, map_result: Dict, consistency: Dict, 
                         season_num: int, save_prefix: str):
    """多维度可视化"""

    # Keep ONLY the top two panels (2×1 row): evolution + uncertainty
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    path = map_result['path']
    
    if path:
        # ---- Build consistent color mapping ----
        all_contestants = sorted({name for p in path for name in p['active']})
        n_all = len(all_contestants)
        # Use Matplotlib's default categorical palette so lines are clearly distinguishable
        base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        if not base_colors:
            base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_map = {name: base_colors[i % len(base_colors)] for i, name in enumerate(all_contestants)}

        # x-axis should span from week1 to finals week (last week in path)
        final_week = max(p['week'] for p in path)

        # Select 4–5 representative contestants to avoid clutter
        def _series_for(name: str):
            ws, shares, lows, highs = [], [], [], []
            for p in path:
                if name in p['active']:
                    idx = p['active'].index(name)
                    ws.append(p['week'])
                    shares.append(p['fan_shares'][idx])
                    lows.append(p['ci_low'][idx])
                    highs.append(p['ci_high'][idx])
            return ws, shares, lows, highs

        stats = []
        for name in all_contestants:
            ws, shares, lows, highs = _series_for(name)
            if not shares:
                continue
            ciw = [h - l for h, l in zip(highs, lows)]
            stats.append({
                'name': name,
                'len': len(shares),
                'last': shares[-1],
                'range': (max(shares) - min(shares)) if len(shares) > 1 else 0.0,
                'mean_ciw': float(np.mean(ciw)) if ciw else 0.0,
            })

        if season_num == 1:
            # Season 1: show all contestants (no representative filtering)
            selected = all_contestants
        else:
            selected: List[str] = []
            # Top by final vote share
            for s in sorted(stats, key=lambda d: d['last'], reverse=True)[:2]:
                if s['name'] not in selected:
                    selected.append(s['name'])
            # Most volatile
            for s in sorted(stats, key=lambda d: d['range'], reverse=True):
                if s['name'] not in selected:
                    selected.append(s['name'])
                    break
            # Most uncertain (avg CI width)
            for s in sorted(stats, key=lambda d: d['mean_ciw'], reverse=True):
                if s['name'] not in selected:
                    selected.append(s['name'])
                    break
            # Earliest eliminated (shortest trajectory)
            for s in sorted(stats, key=lambda d: d['len']):
                if s['name'] not in selected:
                    selected.append(s['name'])
                    break
            selected = selected[:5]

        # Plot only selected contestants; each line naturally stops after elimination
        for contestant in selected:
            shares = []
            ci_lows = []
            ci_highs = []
            ws = []
            
            for p in path:
                if contestant in p['active']:
                    idx = p['active'].index(contestant)
                    shares.append(p['fan_shares'][idx])
                    ci_lows.append(p['ci_low'][idx])
                    ci_highs.append(p['ci_high'][idx])
                    ws.append(p['week'])
            
            if shares:
                c = color_map[contestant]
                ax1.plot(
                    ws,
                    shares,
                    'o-',
                    label=contestant,
                    linewidth=1.6,
                    markersize=4,
                    color=c,
                    alpha=0.95,
                )
                # Keep CI but reduce opacity to avoid clutter with many contestants
                ax1.fill_between(ws, ci_lows, ci_highs, alpha=0.12, color=c, linewidth=0)
        
        ax1.set_xlabel('Week', fontsize=12)
        ax1.set_ylabel('Fan Vote Share', fontsize=12)
        ax1.set_title(f'Season {season_num}: Fan Vote Evolution (MAP + 90% CI)', fontsize=14)
        ax1.set_xlim(1, final_week)
        ax1.set_xticks(list(range(1, final_week + 1)))
        ax1.legend(loc='upper left', fontsize=9, frameon=True)
        ax1.grid(False)
    
    # 2. 不确定性分析
    week_metrics = consistency['week_metrics']
    
    if week_metrics:
        weeks_m = [m['week'] for m in week_metrics]
        uncertainties = [m['mean_uncertainty'] for m in week_metrics]
        ci_widths = [m['ci_width'] for m in week_metrics]
        correctness = [1 if m['correct'] else 0 for m in week_metrics]
        
        # Keep the original visual semantics: blue bars + red CI line
        ax2.bar(weeks_m, uncertainties, alpha=0.65, label='Std Dev', color='#4C78A8')
        ax2.plot(weeks_m, ci_widths, color='#E45756', marker='s', label='90% CI Width', linewidth=2)
        
        # 标记正确/错误预测（用可显示的符号）
        for w, c, u in zip(weeks_m, correctness, uncertainties):
            marker = r'$\checkmark$' if c else r'$\times$'
            color = '#2CA02C' if c else '#D62728'
            ax2.scatter([w], [u + 0.02], marker=marker, color=color, s=90, zorder=5)
        
        ax2.set_xlabel('Week', fontsize=12)
        ax2.set_ylabel('Uncertainty', fontsize=12)
        ax2.set_title(f'Season {season_num}: Uncertainty Analysis', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(False)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_season{season_num}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_prefix}_season{season_num}.png")


def visualize_cross_season_summary_v2(all_results: List[Dict], save_prefix: str):
    """跨赛季汇总可视化 - 美赛O奖标准美化版"""
    
    # 设置全局字体
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    seasons = [r['season'] for r in all_results]
    accuracies = [r['consistency']['accuracy'] * 100 for r in all_results]
    
    # ===================== 1. 各赛季预测准确率 =====================
    ax1 = axes[0]
    ax1.set_facecolor('white')
    
    # 根据准确率设置渐变色（严格使用论文蓝色系）
    acc_min = min(accuracies) if accuracies else 0
    acc_max = max(accuracies) if accuracies else 100
    acc_scale = [(a - acc_min) / (acc_max - acc_min + 1e-9) for a in accuracies]
    colors = [PAPER_CMAP(0.2 + 0.7 * s) for s in acc_scale]
    bars = ax1.bar(seasons, accuracies, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)
    apply_shadow_to_patches(bars)
    
    # 均值线
    mean_acc = np.mean(accuracies)
    ax1.axhline(y=mean_acc, color=PAPER_DARK, linestyle='--', linewidth=2, alpha=0.8,
                label=f'Mean: {mean_acc:.1f}%')
    ax1.axhline(y=50, color=PAPER_GRAY, linestyle=':', alpha=0.7, label='Random Baseline')
    
    ax1.set_xlabel('Season', fontsize=11, fontweight='medium')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='medium')
    ax1.legend(loc='lower right', framealpha=0.95, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.set_xlim(0, max(seasons) + 1)
    
    # 精简网格
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 只在低于100%的柱子上标注
    for bar, acc in zip(bars, accuracies):
        if acc < 100:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
                    f'{acc:.0f}', ha='center', fontsize=8, color='#333')
    
    # ===================== 2. 赛制方法比较 =====================
    ax3 = axes[1]
    ax3.set_facecolor('white')
    
    rank_results = [r for r in all_results if r['method'] == 'rank']
    percent_results = [r for r in all_results if r['method'] == 'percent']
    
    rank_acc = [r['consistency']['accuracy'] * 100 for r in rank_results]
    percent_acc = [r['consistency']['accuracy'] * 100 for r in percent_results]
    
    x_pos = [0, 1]
    means = [np.mean(rank_acc) if rank_acc else 0, np.mean(percent_acc) if percent_acc else 0]
    stds = [np.std(rank_acc) if rank_acc else 0, np.std(percent_acc) if percent_acc else 0]
    
    bar_colors = [PAPER_MID, PAPER_DARK]
    bars = ax3.bar(x_pos, means, yerr=stds, color=bar_colors, alpha=0.9,
                   edgecolor='white', capsize=6, linewidth=0.5, width=0.5,
                   error_kw={'linewidth': 1.5, 'capthick': 1.5, 'ecolor': PAPER_GRAY})
    apply_shadow_to_patches(bars)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'RANK\n(n={len(rank_acc)})', f'PERCENT\n(n={len(percent_acc)})'], fontsize=10)
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='medium')
    ax3.set_ylim(0, 110)
    ax3.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    for bar, mean, std in zip(bars, means, stds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2, 
                f'{mean:.1f}%', ha='center', fontsize=12, fontweight='bold', color='#333')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_summary.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved cross-season summary: {save_prefix}_summary.png")


# ============================================================
# 主程序
# ============================================================

# ============================================================
# 6. Certainty 表格与热力图（小任务2补充）
# ============================================================

def export_certainty_table(sampled_data: Dict, map_result: Dict, 
                           season_num: int, save_prefix: str) -> str:
    """
    导出每周每位选手的 mean, std, CI width 表格（CSV）
    
    返回保存路径
    """
    import csv
    
    path = map_result['path']
    weeks_sampled = sampled_data['weeks']
    
    rows = []
    for t, week_info in enumerate(path):
        week = week_info['week']
        active = week_info['active']
        fan_shares = week_info['fan_shares']
        
        # 从采样数据获取统计量
        week_sample = weeks_sampled[t]
        means = week_sample['mean']
        stds = week_sample['std']
        p5 = week_sample['percentile_5']
        p95 = week_sample['percentile_95']
        
        for i, contestant in enumerate(active):
            ci_width = p95[i] - p5[i]
            rows.append({
                'Season': season_num,
                'Week': week,
                'Contestant': contestant,
                'MAP_FanShare': f"{fan_shares[i]:.4f}",
                'Mean': f"{means[i]:.4f}",
                'Std': f"{stds[i]:.4f}",
                'CI_5%': f"{p5[i]:.4f}",
                'CI_95%': f"{p95[i]:.4f}",
                'CI_Width': f"{ci_width:.4f}",
                'Eliminated': 'Yes' if contestant == week_info.get('eliminated') or 
                              (isinstance(week_info.get('eliminated'), list) and 
                               contestant in week_info.get('eliminated', [])) else 'No'
            })
    
    filepath = f"{save_prefix}_certainty_table_season{season_num}.csv"
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  Saved certainty table: {filepath}")
    return filepath


def visualize_certainty_heatmap(sampled_data: Dict, map_result: Dict,
                                 season_num: int, save_prefix: str):
    """
    绘制选手×周的 certainty 热力图
    
    横轴: Week
    纵轴: Contestant
    颜色: CI Width (90% 区间宽度) 或 Std
    """
    path = map_result['path']
    weeks_sampled = sampled_data['weeks']
    
    if not path:
        return
    
    # 收集所有选手（按首次出现排序）
    all_contestants = []
    seen = set()
    for p in path:
        for c in p['active']:
            if c not in seen:
                all_contestants.append(c)
                seen.add(c)
    
    # 收集所有周
    all_weeks = [p['week'] for p in path]
    
    n_contestants = len(all_contestants)
    n_weeks = len(all_weeks)
    
    # 构建两个矩阵：std 和 ci_width
    std_matrix = np.full((n_contestants, n_weeks), np.nan)
    ci_matrix = np.full((n_contestants, n_weeks), np.nan)
    
    for t, week_info in enumerate(path):
        week_sample = weeks_sampled[t]
        active = week_info['active']
        stds = week_sample['std']
        p5 = week_sample['percentile_5']
        p95 = week_sample['percentile_95']
        
        for i, contestant in enumerate(active):
            row_idx = all_contestants.index(contestant)
            std_matrix[row_idx, t] = stds[i]
            ci_matrix[row_idx, t] = p95[i] - p5[i]
    
    # 绘图：2 列热力图（Std 和 CI Width）
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, n_contestants * 0.35)))
    
    # 共享颜色条范围
    vmin_std = np.nanmin(std_matrix)
    vmax_std = np.nanmax(std_matrix)
    vmin_ci = np.nanmin(ci_matrix)
    vmax_ci = np.nanmax(ci_matrix)
    
    # 左图：Std Dev 热力图
    ax1 = axes[0]
    im1 = ax1.imshow(std_matrix, aspect='auto', cmap=PAPER_CMAP,
                     vmin=vmin_std, vmax=vmax_std)
    ax1.set_xticks(range(n_weeks))
    ax1.set_xticklabels(all_weeks)
    ax1.set_yticks(range(n_contestants))
    ax1.set_yticklabels(all_contestants, fontsize=8)
    ax1.set_xlabel('Week', fontsize=12)
    ax1.set_ylabel('Contestant', fontsize=12)
    ax1.set_title(f'Season {season_num}: Std Dev by Contestant × Week', fontsize=14)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Std Dev', fontsize=10)
    
    # 标记淘汰（用 X）
    for t, week_info in enumerate(path):
        elim = week_info.get('eliminated')
        if elim:
            elim_names = elim if isinstance(elim, list) else [elim]
            for en in elim_names:
                if en in all_contestants:
                    row_idx = all_contestants.index(en)
                    ax1.plot(t, row_idx, 'kx', markersize=10, markeredgewidth=2)
    
    # 右图：CI Width 热力图
    ax2 = axes[1]
    im2 = ax2.imshow(ci_matrix, aspect='auto', cmap=PAPER_CMAP,
                     vmin=vmin_ci, vmax=vmax_ci)
    ax2.set_xticks(range(n_weeks))
    ax2.set_xticklabels(all_weeks)
    ax2.set_yticks(range(n_contestants))
    ax2.set_yticklabels(all_contestants, fontsize=8)
    ax2.set_xlabel('Week', fontsize=12)
    ax2.set_ylabel('Contestant', fontsize=12)
    ax2.set_title(f'Season {season_num}: 90% CI Width by Contestant × Week', fontsize=14)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('CI Width', fontsize=10)
    
    # 标记淘汰（用 X）
    for t, week_info in enumerate(path):
        elim = week_info.get('eliminated')
        if elim:
            elim_names = elim if isinstance(elim, list) else [elim]
            for en in elim_names:
                if en in all_contestants:
                    row_idx = all_contestants.index(en)
                    ax2.plot(t, row_idx, 'kx', markersize=10, markeredgewidth=2)
    
    # 添加说明
    fig.text(0.5, 0.01, 
             'Note: Gray cells = contestant eliminated before that week; X = elimination week',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    filepath = f"{save_prefix}_certainty_heatmap_season{season_num}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved certainty heatmap: {filepath}")


def main():
    print("\n" + "=" * 60)
    print("MCM 2026 Problem C - Fan Vote Estimation (V2)")
    print("Two-Step Model: Feasible Region Sampling + HMM MAP")
    print("=" * 60 + "\n")
    
    # 加载数据
    repo_root = Path(__file__).resolve().parents[2]
    data_path = str(repo_root / "2026_MCM_Problem_C_Data.csv")
    seasons_data, valid_seasons = load_and_preprocess_data(data_path)
    
    # 计算所有有效赛季的准确率（不仅仅是selected_seasons）
    all_seasons_to_compute = sorted(valid_seasons)
    
    print(f"All valid seasons to compute: {all_seasons_to_compute}")
    print(f"Total: {len(all_seasons_to_compute)} seasons")
    print()
    
    all_results = []
    save_dir = repo_root / "task1" / "figure"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = str(save_dir / "fan_vote_v2")
    
    for season in all_seasons_to_compute:
        print(f"\n{'='*60}")
        print(f"Processing Season {season}")
        print('='*60)
        
        season_data = seasons_data[season]
        method = 'rank' if (season <= 2 or season >= 28) else 'percent'
        
        # Step 1: 可行域采样（RANK method 使用更多采样以提高质量）
        n_samples = 5000 if method == 'rank' else 2000  # RANK 采样数增加到5000
        print(f"  Step 1: Hit-and-Run Feasible Region Sampling ({method})...")
        sampled_data = sample_season_feasible_region_v2(season_data, season, n_samples=n_samples)
        print(f"    - Weeks sampled: {len(sampled_data['weeks'])}")
        
        # Step 2: HMM + Viterbi（RANK method 使用更多状态以提高精度）
        n_states = 200 if method == 'rank' else 100  # RANK 状态数增加到200
        print("  Step 2: HMM + Viterbi MAP Path...")
        map_result = build_hmm_and_find_map_path_v2(sampled_data, n_states=n_states)
        print(f"    - Path length: {len(map_result['path'])}")
        print(f"    - Log probability: {map_result['log_prob']:.2f}")
        
        # 评估一致性
        print("  Step 3: Consistency Evaluation...")
        consistency = evaluate_consistency_v2(map_result, sampled_data)
        print(f"    - Accuracy: {consistency['accuracy']:.1%} ({consistency['correct']}/{consistency['total']})")
        
        # 只为代表性赛季(5和33)生成可视化图
        if season in [5, 33]:
            print("  Step 4: Generating Visualizations (O-Award Standard)...")
            visualize_results_v2(sampled_data, map_result, consistency, season, save_prefix)
            visualize_certainty_heatmap(sampled_data, map_result, season, save_prefix)
        
        all_results.append({
            'season': season,
            'method': method,
            'sampled_data': sampled_data,
            'map_result': map_result,
            'consistency': consistency
        })
    
    # 跨赛季汇总（只生成这一个summary图）
    print("\n" + "=" * 60)
    print("Cross-Season Summary")
    print("=" * 60)
    visualize_cross_season_summary_v2(all_results, save_prefix)
    
    # 打印最终汇总
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Season':<10} {'Method':<10} {'Accuracy':<15} {'Weeks':<10} {'Uncertainty':<12}")
    print("-" * 60)
    for r in all_results:
        mean_unc = np.mean([m['mean_uncertainty'] for m in r['consistency']['week_metrics']]) if r['consistency']['week_metrics'] else 0
        print(f"{r['season']:<10} {r['method']:<10} {r['consistency']['accuracy']:.1%}            {len(r['sampled_data']['weeks']):<10} {mean_unc:.4f}")
    
    overall_acc = np.mean([r['consistency']['accuracy'] for r in all_results])
    print("-" * 60)
    print(f"{'OVERALL':<10} {'':<10} {overall_acc:.1%}")
    print("\nDone!")


if __name__ == "__main__":
    main()
