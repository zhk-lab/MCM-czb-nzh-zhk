"""
生成放大镜图的两个独立部分：
1. 左上角图：Fan Vote Evolution (整体趋势图)
2. 左下角图：Posterior Distribution (放大的细节箱线图)

颜色完全匹配，便于PPT中创建放大镜效果
选择 Season 1（只有6个选手，线数量适中）
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据加载和预处理（复用原代码逻辑）
# ============================================================

def load_and_preprocess_data(filepath: str) -> Dict:
    """加载并预处理数据"""
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
            
            industry = row['celebrity_industry']
            industry_weights = {
                'Actor/Actress': 1.2, 'Singer/Rapper': 1.3, 'Athlete': 1.1,
                'TV Personality': 1.15, 'Model': 1.0, 'Comedian': 0.95,
                'Social Media Personality': 1.25, 'News Anchor': 0.9
            }
            pop_weight = industry_weights.get(industry, 1.0)
            
            try:
                age = int(row['celebrity_age_during_season'])
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
            
            if 'Eliminated Week' in result:
                try:
                    elim_week = int(result.split('Week')[-1].strip())
                    if seasons_data[season]['weeks'][elim_week]['eliminated']:
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
    
    return dict(seasons_data)


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


def compute_combined_score(fan_shares: np.ndarray, judge_scores: np.ndarray, 
                           method: str = 'percent') -> np.ndarray:
    """计算综合得分"""
    n = len(fan_shares)
    total_judge = judge_scores.sum()
    
    if method == 'percent':
        judge_pct = judge_scores / total_judge if total_judge > 0 else np.ones(n) / n
        return judge_pct + fan_shares
    else:
        judge_rank = np.argsort(np.argsort(-judge_scores)) + 1.0
        fan_rank = np.argsort(np.argsort(-fan_shares)) + 1.0
        return -(judge_rank + fan_rank)


def hit_and_run_sample(week_data: Dict, method: str = 'percent',
                       n_samples: int = 500, burn_in: int = 200) -> np.ndarray:
    """Hit-and-Run MCMC 采样"""
    active = week_data['active']
    scores = week_data['scores']
    eliminated = week_data['eliminated']
    priors = week_data['priors']
    n = len(active)
    
    if n < 2:
        return np.ones((n_samples, 1))
    
    if isinstance(eliminated, list):
        elim_names = eliminated
    else:
        elim_names = [eliminated] if eliminated else []
    
    elim_indices = [active.index(e) for e in elim_names if e in active]
    
    if not elim_indices:
        prior_weights = np.array([priors.get(c, 1.0) for c in active])
        prior_weights = prior_weights / prior_weights.sum()
        samples = np.random.dirichlet(prior_weights * 10, n_samples)
        return samples
    
    judge_scores = np.array([scores.get(c, 0) for c in active])
    
    def is_feasible(fan_share: np.ndarray) -> bool:
        if np.any(fan_share < -1e-10) or np.abs(fan_share.sum() - 1) > 1e-6:
            return False
        combined = compute_combined_score(fan_share, judge_scores, method)
        min_score = combined.min()
        for idx in elim_indices:
            if combined[idx] > min_score + 1e-6:
                return False
        return True
    
    prior_weights = np.array([priors.get(c, 1.0) for c in active])
    for idx in elim_indices:
        prior_weights[idx] *= 0.3
    
    pi = prior_weights / prior_weights.sum()
    
    if not is_feasible(pi):
        found = False
        for _ in range(1000):
            test_pi = np.random.dirichlet(np.ones(n) * 2)
            if is_feasible(test_pi):
                pi = test_pi
                found = True
                break
        if not found:
            pi = np.ones(n) / n
            for idx in elim_indices:
                pi[idx] = 0.01
            pi = pi / pi.sum()
    
    samples = []
    accept_count = 0
    
    for iteration in range(burn_in + n_samples * 2):
        direction = np.random.randn(n)
        direction = direction - direction.mean()
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction = direction / norm
        
        step_scale = 0.1 if accept_count < 100 else 0.05
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


def sample_season(season_data: Dict, season_num: int, n_samples: int = 500) -> Dict:
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
        samples = hit_and_run_sample(week_data, method, n_samples)
        
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


def build_hmm_path(sampled_data: Dict, n_states: int = 40) -> Dict:
    """HMM + Viterbi 获取MAP路径"""
    weeks = sampled_data['weeks']
    method = sampled_data['method']
    
    if len(weeks) < 2:
        return {'path': [], 'log_prob': 0}
    
    week_states = []
    for week_info in weeks:
        samples = week_info['samples']
        n = min(n_states, len(samples))
        indices = np.linspace(0, len(samples)-1, n, dtype=int)
        states = samples[indices]
        week_states.append(states)
    
    T = len(weeks)
    
    def observation_likelihood(state: np.ndarray, week_info: Dict) -> float:
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
        
        log_lik = 0.0
        for idx in elim_indices:
            diff = min_score - combined[idx]
            log_lik += diff * 10
        
        return log_lik
    
    n_s0 = len(week_states[0])
    log_delta = [np.array([observation_likelihood(week_states[0][i], weeks[0]) 
                           for i in range(n_s0)])]
    psi = [np.zeros(n_s0, dtype=int)]
    
    for t in range(1, T):
        n_st = len(week_states[t])
        n_st_prev = len(week_states[t-1])
        
        log_delta_t = np.full(n_st, -np.inf)
        psi_t = np.zeros(n_st, dtype=int)
        
        active_prev = set(weeks[t-1]['active'])
        active_curr = set(weeks[t]['active'])
        common = list(active_prev & active_curr)
        
        for j in range(n_st):
            state_j = week_states[t][j]
            obs_lik = observation_likelihood(state_j, weeks[t])
            
            best_log_prob = -np.inf
            best_i = 0
            
            if not common:
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
                    diff = state_j[idx_curr] - state_i[idx_prev]
                    smoothness = -np.sum(diff ** 2) * 20
                    log_prob = log_delta[t-1][i] + smoothness + obs_lik
                    
                    if log_prob > best_log_prob:
                        best_log_prob = log_prob
                        best_i = i
            
            log_delta_t[j] = best_log_prob
            psi_t[j] = best_i
        
        log_delta.append(log_delta_t)
        psi.append(psi_t)
    
    path_indices = [0] * T
    path_indices[T-1] = log_delta[T-1].argmax()
    
    for t in range(T-2, -1, -1):
        path_indices[t] = psi[t+1][path_indices[t+1]]
    
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
    
    return {'path': map_path, 'log_prob': log_delta[T-1].max()}


# ============================================================
# 绘制放大镜所需的两个独立图
# ============================================================

def create_magnifier_plots(sampled_data: Dict, map_result: Dict, season_num: int, 
                           save_dir: str, detail_week_idx: int = None):
    """
    生成放大镜图的两个独立部分
    
    Args:
        sampled_data: 采样数据
        map_result: HMM MAP结果
        season_num: 赛季号
        save_dir: 保存目录
        detail_week_idx: 要放大显示的周的索引（None则自动选择中间周）
    """
    path = map_result['path']
    weeks_sampled = sampled_data['weeks']
    
    if not path:
        print("No path data available!")
        return
    
    # 建立一致的颜色映射
    all_contestants = sorted({name for p in path for name in p['active']})
    n_all = len(all_contestants)
    
    # 使用更鲜艳的颜色方案
    if n_all <= 10:
        # 手动定义鲜艳且区分度高的颜色
        colors = [
            '#1f77b4',  # 蓝色
            '#ff7f0e',  # 橙色
            '#2ca02c',  # 绿色
            '#d62728',  # 红色
            '#9467bd',  # 紫色
            '#8c564b',  # 棕色
            '#e377c2',  # 粉色
            '#7f7f7f',  # 灰色
            '#bcbd22',  # 黄绿色
            '#17becf',  # 青色
        ]
        color_map = {name: colors[i % len(colors)] for i, name in enumerate(all_contestants)}
    else:
        palette = plt.cm.tab20(np.linspace(0, 1, 20))
        color_map = {name: palette[i % len(palette)] for i, name in enumerate(all_contestants)}
    
    # 选择要放大显示的周
    if detail_week_idx is None:
        detail_week_idx = len(weeks_sampled) // 2
    
    detail_week = weeks_sampled[detail_week_idx]
    detail_path = path[detail_week_idx]
    
    print(f"\n=== Season {season_num} ===")
    print(f"总选手数: {n_all}")
    print(f"选手: {all_contestants}")
    print(f"放大显示的周: Week {detail_week['week']}")
    
    # ============================================================
    # 图1: 左上角 - Fan Vote Evolution (整体趋势图)
    # ============================================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    final_week = max(p['week'] for p in path)
    
    for contestant in all_contestants:
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
            ax1.plot(ws, shares, 'o-', label=contestant, linewidth=2.5, 
                    markersize=8, color=c, alpha=0.95)
            ax1.fill_between(ws, ci_lows, ci_highs, alpha=0.15, color=c, linewidth=0)
    
    ax1.set_xlabel('Week', fontsize=14)
    ax1.set_ylabel('Fan Vote Share', fontsize=14)
    ax1.set_title(f'Season {season_num}: Fan Vote Evolution (MAP + 90% CI)', fontsize=16, fontweight='bold')
    ax1.set_xlim(0.5, final_week + 0.5)
    ax1.set_xticks(list(range(1, final_week + 1)))
    ax1.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#fafafa')
    
    # 在详细周位置添加虚线框标记（用于放大镜指示）
    ax1.axvline(x=detail_week['week'], color='gray', linestyle=':', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    path1 = f'{save_dir}/magnifier_overview_season{season_num}.png'
    plt.savefig(path1, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"保存整体图: {path1}")
    
    # ============================================================
    # 图2: 左下角 - Posterior Distribution (放大的细节箱线图)
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    samples = detail_week['samples']
    active = detail_week['active']
    n_show = len(active)
    
    box_data = [samples[:, i] for i in range(n_show)]
    labels = active
    
    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False,
                     widths=0.6,
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # 为每个箱线图设置匹配的颜色
    for patch, name in zip(bp['boxes'], active):
        patch.set_facecolor(color_map.get(name, '#cccccc'))
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # 标记被淘汰者
    elim = detail_week['eliminated']
    elim_names = elim if isinstance(elim, list) else [elim] if elim else []
    for en in elim_names:
        if en in active:
            idx = active.index(en)
            ax2.axvline(x=idx + 1, color='red', linestyle='--', alpha=0.8, linewidth=2.5)
    
    ax2.set_ylabel('Fan Vote Share', fontsize=14)
    ax2.set_title(f'Week {detail_week["week"]}: Posterior Distribution (Red=Eliminated)', 
                  fontsize=16, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=11)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_facecolor('#fafafa')
    
    plt.tight_layout()
    path2 = f'{save_dir}/magnifier_detail_season{season_num}_week{detail_week["week"]}.png'
    plt.savefig(path2, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"保存细节图: {path2}")
    
    # ============================================================
    # 额外：创建颜色对照表（方便PPT制作）
    # ============================================================
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.axis('off')
    
    for i, contestant in enumerate(all_contestants):
        y_pos = 1 - (i + 1) / (len(all_contestants) + 1)
        ax3.add_patch(plt.Rectangle((0.1, y_pos - 0.03), 0.1, 0.06, 
                                     color=color_map[contestant], ec='black'))
        ax3.text(0.25, y_pos, contestant, fontsize=12, va='center')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title(f'Season {season_num} Color Legend', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    path3 = f'{save_dir}/magnifier_legend_season{season_num}.png'
    plt.savefig(path3, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"保存颜色对照: {path3}")
    
    return path1, path2, path3


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("生成放大镜图（Season 1）")
    print("=" * 60)
    
    # 加载数据
    data_path = r"c:\Users\zhaoh\Desktop\MCM-czb-nzh-zhk\2026_MCM_Problem_C_Data.csv"
    seasons_data = load_and_preprocess_data(data_path)
    
    # 选择 Season 1（6个选手，线数量适中）
    season_num = 1
    season_data = seasons_data[season_num]
    
    print(f"\n处理 Season {season_num}...")
    
    # 采样
    print("Step 1: 可行域采样...")
    sampled_data = sample_season(season_data, season_num, n_samples=400)
    print(f"  - 周数: {len(sampled_data['weeks'])}")
    
    # HMM MAP
    print("Step 2: HMM + Viterbi...")
    map_result = build_hmm_path(sampled_data, n_states=35)
    print(f"  - 路径长度: {len(map_result['path'])}")
    
    # 生成放大镜图
    print("Step 3: 生成放大镜图...")
    save_dir = r"c:\Users\zhaoh\Desktop\MCM-czb-nzh-zhk\magnifier_output"
    
    # 创建输出目录
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择要放大的周（选择中间的周，Week 4 或最合适的周）
    # Season 1 有约4-5周数据，选择中间周
    detail_week_idx = len(sampled_data['weeks']) // 2
    
    create_magnifier_plots(sampled_data, map_result, season_num, save_dir, detail_week_idx)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n输出文件保存在: {save_dir}")
    print("\n使用说明:")
    print("1. magnifier_overview_*.png - 整体趋势图（对应原图左上角）")
    print("2. magnifier_detail_*.png - 细节箱线图（对应原图左下角）")
    print("3. magnifier_legend_*.png - 颜色对照表")
    print("\n在PPT中:")
    print("- 将整体图作为背景")
    print("- 将细节图放在放大镜框内")
    print("- 用连接线将两图关联起来")


if __name__ == "__main__":
    main()
