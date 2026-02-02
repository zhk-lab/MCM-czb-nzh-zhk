"""
MCM 2026 Problem C - Task 4: Two-Key + Live Save System
========================================================
改进版双钥匙赛制完整实现：
  - Step A: 双榜单排序（评委 + 粉丝）
  - Step B: 动态 Bottom 集合（并集 + 交集）
  - Step C: Bottom-3 提名（修复温吞水漏洞）
  - Step D: 直播救援（修复 Jerry Rice 漏洞）
  - Step E: 评委兜底淘汰
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 数据加载（复用 Task 2 框架）
# ============================================================

def load_all_data(base_dir):
    """加载粉丝票份额和评委分数数据"""
    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)
    
    from pathlib import Path
    base_path = Path(base_dir)
    
    # 加载粉丝票份额（Task 1 估计结果）
    fan_df = pd.read_csv(str(base_path / "task1" / "table" / "fan_vote_shares.csv"))
    print(f"  Fan vote shares: {len(fan_df)} records")
    
    # 加载原始数据（评委分数）
    data_df = pd.read_csv(str(base_path / "2026_MCM_Problem_C_Data.csv"))
    
    # 构建评委周分数表（合并各评委分数）
    judge_data = []
    for _, row in data_df.iterrows():
        name = row['celebrity_name']
        season = row['season']
        placement = row['placement']
        
        for week in range(1, 12):
            # 获取该周所有评委的分数
            scores = []
            for judge_num in range(1, 5):  # 最多4个评委
                col = f'week{week}_judge{judge_num}_score'
                if col in data_df.columns and pd.notna(row[col]):
                    try:
                        score = float(row[col])
                        if score > 0:  # 过滤0分（表示未参赛）
                            scores.append(score)
                    except:
                        pass
            
            # 如果有有效分数，计算总分
            if len(scores) > 0:
                total_score = sum(scores)
                judge_data.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': name,
                    'judge_score': total_score,
                    'placement': placement
                })
    
    judge_df = pd.DataFrame(judge_data)
    print(f"  Judge scores: {len(judge_df)} records")
    
    return fan_df, judge_df, data_df


def create_weekly_panel(fan_df, judge_df):
    """创建每周选手面板（包含评委分和粉丝票）"""
    panel = pd.merge(
        fan_df[fan_df['method'] == 'rank'],  # 使用 rank 方法的估计
        judge_df,
        on=['season', 'week', 'celebrity_name'],
        how='inner'
    )
    
    # 计算每周的份额（归一化）
    panel['judge_share'] = panel.groupby(['season', 'week'])['judge_score'].transform(
        lambda x: x / x.sum()
    )
    panel['fan_share'] = panel['fan_vote_share']  # 已经是份额
    
    return panel


# ============================================================
# Two-Key 核心逻辑
# ============================================================

def get_bottom_k(season, week, panel, dimension='judge', k=3):
    """
    获取某个维度的 Bottom-k 选手
    
    Parameters:
    -----------
    dimension : str
        'judge' 或 'fan'
    k : int
        底部人数
    """
    week_data = panel[(panel['season'] == season) & (panel['week'] == week)].copy()
    
    if len(week_data) == 0:
        return set()
    
    share_col = 'judge_share' if dimension == 'judge' else 'fan_share'
    
    # 排序（升序，最小的在前）
    week_data = week_data.sort_values(share_col)
    
    # 返回底部 k 个（或全部，如果不足 k 个）
    bottom_names = set(week_data.head(min(k, len(week_data)))['celebrity_name'].values)
    
    return bottom_names


def get_dynamic_k(n_contestants):
    """
    根据当周人数动态确定 k_J 和 k_F
    
    Returns:
    --------
    (k_judge, k_fan)
    """
    if n_contestants >= 10:
        return 3, 5  # 早期：评委主导
    elif 7 <= n_contestants <= 9:
        return 3, 3  # 中期：平衡
    elif 4 <= n_contestants <= 6:
        return 2, 3  # 后期：粉丝主导
    else:
        # 极端情况（决赛）
        return min(2, n_contestants), min(2, n_contestants)


def compute_risk_score(contestant, week_data, bottom_judge, bottom_fan, params=None):
    """
    计算极端优先风险分：R_i = max(p_J, p_F) + alpha * min(p_J, p_F)
    
    p 为归一化差分位（0=最好，1=最差）
    
    改进：加大 judge 权重以降低被操纵性
    
    Parameters:
    -----------
    params : dict, optional
        {'alpha': 风险分系数, 'beta': 双钥匙区 bonus, 'judge_weight': 评委权重加成}
    """
    if params is None:
        params = {'alpha': 0.3, 'beta': 0.5, 'judge_weight': 1.2}
    
    n = len(week_data)
    
    # 获取该选手在两个维度的排名（1=最好）
    judge_rank = (week_data['judge_share'] < week_data[week_data['celebrity_name'] == contestant]['judge_share'].values[0]).sum() + 1
    fan_rank = (week_data['fan_share'] < week_data[week_data['celebrity_name'] == contestant]['fan_share'].values[0]).sum() + 1
    
    # 转为分位数（0=最好，1=最差）
    p_J = (judge_rank - 1) / (n - 1) if n > 1 else 0
    p_F = (fan_rank - 1) / (n - 1) if n > 1 else 0
    
    # 极端优先风险分（参数化，加大评委权重）
    judge_weight = params.get('judge_weight', 1.2)
    risk = max(judge_weight * p_J, p_F) + params['alpha'] * min(p_J, p_F)
    
    # 双钥匙成员额外风险（参数化）
    if contestant in bottom_judge and contestant in bottom_fan:
        risk += params['beta']
    
    return risk


def select_bottom_3(season, week, panel, history_df=None, params=None):
    """
    Step C: 提名 Bottom-3（修复温吞水漏洞）
    
    Parameters:
    -----------
    params : dict, optional
        {'alpha', 'beta', 'k_table': ...}
    
    Returns:
    --------
    list of 3 contestant names
    """
    if params is None:
        params = {'alpha': 0.3, 'beta': 0.5, 'k_table': None}
    
    week_data = panel[(panel['season'] == season) & (panel['week'] == week)].copy()
    
    if len(week_data) < 3:
        # 不足3人，全部进入
        return list(week_data['celebrity_name'].values)
    
    n = len(week_data)
    
    # 使用参数化的 k 表（如果提供）
    if params.get('k_table') is not None:
        k_J, k_F = params['k_table'].get(n, get_dynamic_k(n))
    else:
        k_J, k_F = get_dynamic_k(n)
    
    # Step B: 获取 Bottom 集合
    bottom_judge = get_bottom_k(season, week, panel, 'judge', k_J)
    bottom_fan = get_bottom_k(season, week, panel, 'fan', k_F)
    
    # 危险池（并集）和双钥匙区（交集）
    danger_pool = bottom_judge | bottom_fan
    two_key_zone = bottom_judge & bottom_fan
    
    bottom_3 = []
    
    # 优先填充双钥匙区
    bottom_3.extend(list(two_key_zone))
    
    # 如果交集为空，强制包含两个极端
    if len(two_key_zone) == 0:
        # 评委绝对倒数第1
        judge_worst = week_data.nsmallest(1, 'judge_share')['celebrity_name'].values[0]
        # 粉丝绝对倒数第1
        fan_worst = week_data.nsmallest(1, 'fan_share')['celebrity_name'].values[0]
        
        if judge_worst not in bottom_3:
            bottom_3.append(judge_worst)
        if fan_worst not in bottom_3 and len(bottom_3) < 3:
            bottom_3.append(fan_worst)
    
    # 用风险分补足到 3 人
    if len(bottom_3) < 3:
        remaining = danger_pool - set(bottom_3)
        
        # 计算风险分
        risk_scores = {}
        for contestant in remaining:
            risk_scores[contestant] = compute_risk_score(contestant, week_data, bottom_judge, bottom_fan, params)
        
        # 按风险分降序排序
        sorted_by_risk = sorted(risk_scores.items(), key=lambda x: -x[1])
        
        # 补足
        for contestant, _ in sorted_by_risk:
            if len(bottom_3) >= 3:
                break
            bottom_3.append(contestant)
    
    return bottom_3[:3]


def check_save_ban(contestant, season, week, history_df, params=None):
    """
    检查是否禁止救援（连续两周评委绝对倒数第1）
    
    Parameters:
    -----------
    params : dict, optional
        {'ban_consecutive_weeks': 连续最差周数阈值}
    """
    if params is None:
        params = {'ban_consecutive_weeks': 2}
    
    if history_df is None or week < params['ban_consecutive_weeks']:
        return False
    
    # 检查前 N 周是否连续都是评委倒数第1
    for prev_offset in range(1, params['ban_consecutive_weeks'] + 1):
        prev_week = week - prev_offset
        if prev_week < 1:
            return False
        
        prev_week_data = history_df[(history_df['season'] == season) & (history_df['week'] == prev_week)]
        
        if len(prev_week_data) == 0:
            return False
        
        prev_judge_worst = prev_week_data.nsmallest(1, 'judge_share')['celebrity_name'].values
        
        if len(prev_judge_worst) == 0 or prev_judge_worst[0] != contestant:
            return False
    
    # 如果连续 N 周都是倒数第1，禁止救援
    return True


def live_save_simulation(bottom_3, season, week, panel, history_df, seed=42, params=None):
    """
    Step D: 直播救援（修复 Jerry Rice 漏洞 + 抗操纵强化）
    
    模拟直播窗口新增票，救 ΔV 最高者
    禁止条款（双重）：
      1. 连续 N 周评委倒数第1者
      2. 当周 judge_share < 全周 20% 分位者（救援资格底线）
    
    Parameters:
    -----------
    params : dict, optional
        参数集合（新增 'save_eligibility_threshold'）
    
    Returns:
    --------
    saved_contestant : str or None
    """
    if params is None:
        params = {'ban_consecutive_weeks': 2, 'save_eligibility_threshold': 0.20}
    
    week_data = panel[(panel['season'] == season) & (panel['week'] == week)]
    
    np.random.seed(seed + season * 100 + week)
    
    # 检查禁止救援条款 1：连续最差
    ban_list = []
    for contestant in bottom_3:
        if check_save_ban(contestant, season, week, history_df, params):
            ban_list.append(contestant)
    
    # 检查禁止救援条款 2：救援资格底线（judge_share 过低）
    save_threshold = params.get('save_eligibility_threshold', 0.20)
    if save_threshold > 0:
        # 计算当周所有选手的 judge_share 分位数
        week_judge_shares = week_data['judge_share'].values
        threshold_value = np.percentile(week_judge_shares, save_threshold * 100)
        
        for contestant in bottom_3:
            c_judge_share = week_data[week_data['celebrity_name'] == contestant]['judge_share'].values[0]
            if c_judge_share < threshold_value:
                if contestant not in ban_list:
                    ban_list.append(contestant)
    
    eligible = [c for c in bottom_3 if c not in ban_list]
    
    if len(eligible) == 0:
        return None  # 全部禁止救援（罕见）
    
    # 模拟直播窗口新增票（基于当周份额 + 随机扰动）
    delta_votes = {}
    for contestant in eligible:
        base_share = week_data[week_data['celebrity_name'] == contestant]['fan_share'].values[0]
        # 模拟窗口动员效应（正态分布扰动）
        mobilization = np.random.normal(base_share * 0.5, base_share * 0.2)
        delta_votes[contestant] = max(0, mobilization)
    
    # 救 ΔV 最高者
    if len(delta_votes) > 0:
        saved = max(delta_votes, key=delta_votes.get)
        return saved
    
    return None


def judges_eliminate(bottom_3, saved, season, week, panel, history_df):
    """
    Step E: 评委兜底淘汰
    
    从 Bottom-3 中移除被救者，在剩余2人中淘汰评委分更低者
    
    Returns:
    --------
    eliminated_contestant : str
    """
    remaining = [c for c in bottom_3 if c != saved]
    
    if len(remaining) == 0:
        return None  # 异常情况
    
    if len(remaining) == 1:
        return remaining[0]
    
    # 获取评委分数
    week_data = panel[(panel['season'] == season) & (panel['week'] == week)]
    
    scores = {}
    for contestant in remaining:
        scores[contestant] = week_data[week_data['celebrity_name'] == contestant]['judge_score'].values[0]
    
    # 淘汰评委分更低者
    eliminated = min(scores, key=scores.get)
    
    # Tie-break（如果分数相同）
    if len(set(scores.values())) < len(scores):
        # 先看粉丝份额
        fan_shares = {}
        for contestant in remaining:
            fan_shares[contestant] = week_data[week_data['celebrity_name'] == contestant]['fan_share'].values[0]
        
        # 粉丝份额低者淘汰
        eliminated = min(fan_shares, key=fan_shares.get)
    
    return eliminated


def simulate_two_key_week(season, week, panel, history_df, seed=42, params=None):
    """
    模拟 Two-Key 系统的单周淘汰流程
    
    Parameters:
    -----------
    params : dict, optional
        参数集合（传递给各子函数）
    
    Returns:
    --------
    dict with keys: bottom_3, saved, eliminated
    """
    week_data = panel[(panel['season'] == season) & (panel['week'] == week)]
    
    if len(week_data) <= 2:
        # 决赛周或特殊情况，不淘汰
        return {'bottom_3': [], 'saved': None, 'eliminated': None}
    
    # Step C: 提名 Bottom-3
    bottom_3 = select_bottom_3(season, week, panel, history_df, params)
    
    # Step D: 直播救援
    saved = live_save_simulation(bottom_3, season, week, panel, history_df, seed, params)
    
    # Step E: 评委兜底淘汰
    eliminated = judges_eliminate(bottom_3, saved, season, week, panel, history_df)
    
    return {
        'bottom_3': bottom_3,
        'saved': saved,
        'eliminated': eliminated
    }


# ============================================================
# 全季模拟框架
# ============================================================

def simulate_full_season_two_key(season, panel, seed=42):
    """
    模拟整个赛季的 Two-Key 系统
    
    Returns:
    --------
    DataFrame with columns: week, bottom_3, saved, eliminated
    """
    season_data = panel[panel['season'] == season].copy()
    weeks = sorted(season_data['week'].unique())
    
    records = []
    history = panel[panel['season'] == season].copy()
    
    remaining_contestants = set(season_data['celebrity_name'].unique())
    
    for week in weeks:
        week_contestants = set(season_data[season_data['week'] == week]['celebrity_name'].values)
        
        # 只保留仍在赛的选手
        week_contestants = week_contestants & remaining_contestants
        
        if len(week_contestants) <= 2:
            break  # 决赛周
        
        # 模拟本周淘汰
        result = simulate_two_key_week(season, week, history, history, seed)
        
        records.append({
            'season': season,
            'week': week,
            'n_contestants': len(week_contestants),
            'bottom_3': ','.join(result['bottom_3']) if result['bottom_3'] else '',
            'saved': result['saved'] if result['saved'] else '',
            'eliminated': result['eliminated'] if result['eliminated'] else ''
        })
        
        # 移除被淘汰者
        if result['eliminated']:
            remaining_contestants.discard(result['eliminated'])
    
    return pd.DataFrame(records)


def simulate_all_seasons_two_key(panel, seed=42):
    """
    模拟所有赛季的 Two-Key 系统
    """
    print("\n" + "=" * 70)
    print("TWO-KEY SYSTEM: Full Season Simulation")
    print("=" * 70)
    
    all_records = []
    seasons = sorted(panel['season'].unique())
    
    for season in seasons:
        print(f"  Simulating Season {season}...")
        season_records = simulate_full_season_two_key(season, panel, seed)
        all_records.append(season_records)
    
    full_df = pd.concat(all_records, ignore_index=True)
    print(f"\n  Total elimination weeks simulated: {len(full_df)}")
    
    return full_df


# ============================================================
# 主执行函数
# ============================================================

def main():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    
    # 加载数据
    fan_df, judge_df, data_df = load_all_data(str(repo_root))
    
    # 创建周面板
    panel = create_weekly_panel(fan_df, judge_df)
    print(f"\n  Weekly panel created: {len(panel)} contestant-week records")
    
    # 模拟所有赛季
    two_key_results = simulate_all_seasons_two_key(panel, seed=42)
    
    # 保存结果
    output_path = repo_root / "task4" / "table" / "two_key_elimination_records.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    two_key_results.to_csv(str(output_path), index=False)
    print(f"\n  Results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("TWO-KEY SYSTEM SIMULATION COMPLETED!")
    print("=" * 70)
    
    return two_key_results


if __name__ == "__main__":
    results = main()
