"""
MCM 2026 Problem C - Task 4: Full Comparison Simulation
=========================================================
对比四种方法的完整反事实模拟：
  - RANK（基线）
  - PERCENT（Task 2 推荐）
  - SAVE（Judges Save）
  - TWO_KEY（新方案）

复用 Task 2.2 框架并扩展
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import importlib.util
import warnings
warnings.filterwarnings('ignore')

# 动态导入 Two-Key 系统函数
spec = importlib.util.spec_from_file_location(
    "two_key_system",
    os.path.join(os.path.dirname(__file__), "4_two_key_system.py")
)
two_key_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(two_key_module)

# 导入所需函数
load_all_data = two_key_module.load_all_data
create_weekly_panel = two_key_module.create_weekly_panel
select_bottom_3 = two_key_module.select_bottom_3
live_save_simulation = two_key_module.live_save_simulation
judges_eliminate = two_key_module.judges_eliminate


# ============================================================
# 复用 Task 2.2 的 RANK/PERCENT/SAVE 模拟函数
# ============================================================

def simulate_rank_method(panel_df):
    """RANK 方法：按名次相加（50% judge rank + 50% fan rank）"""
    seasons = sorted(panel_df['season'].unique())
    all_placements = []
    
    for season in seasons:
        season_data = panel_df[panel_df['season'] == season].copy()
        weeks = sorted(season_data['week'].unique())
        
        remaining = set(season_data['celebrity_name'].unique())
        placements = {}
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week]
            week_data = week_data[week_data['celebrity_name'].isin(remaining)]
            
            if len(week_data) <= 2:
                break
            
            # 计算 rank（1=best）
            week_data['judge_rank'] = week_data['judge_score'].rank(ascending=False, method='min')
            week_data['fan_rank'] = week_data['fan_share'].rank(ascending=False, method='min')
            
            # Combined rank
            week_data['combined_rank'] = week_data['judge_rank'] + week_data['fan_rank']
            
            # 淘汰 combined_rank 最大者
            eliminated_idx = week_data['combined_rank'].idxmax()
            eliminated = week_data.loc[eliminated_idx, 'celebrity_name']
            
            remaining.remove(eliminated)
        
        # 剩余选手排名
        for i, contestant in enumerate(sorted(remaining)):
            placements[contestant] = i + 1
        
        for contestant, placement in placements.items():
            all_placements.append({
                'season': season,
                'method': 'RANK',
                'celebrity_name': contestant,
                'final_placement': placement
            })
    
    return pd.DataFrame(all_placements)


def simulate_percent_method(panel_df):
    """PERCENT 方法：按份额相加（50% judge % + 50% fan %）"""
    seasons = sorted(panel_df['season'].unique())
    all_placements = []
    
    for season in seasons:
        season_data = panel_df[panel_df['season'] == season].copy()
        weeks = sorted(season_data['week'].unique())
        
        remaining = set(season_data['celebrity_name'].unique())
        placements = {}
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week]
            week_data = week_data[week_data['celebrity_name'].isin(remaining)]
            
            if len(week_data) <= 2:
                break
            
            # Combined score
            week_data['combined_score'] = 0.5 * week_data['judge_share'] + 0.5 * week_data['fan_share']
            
            # 淘汰 combined_score 最小者
            eliminated_idx = week_data['combined_score'].idxmin()
            eliminated = week_data.loc[eliminated_idx, 'celebrity_name']
            
            remaining.remove(eliminated)
        
        for i, contestant in enumerate(sorted(remaining)):
            placements[contestant] = i + 1
        
        for contestant, placement in placements.items():
            all_placements.append({
                'season': season,
                'method': 'PERCENT',
                'celebrity_name': contestant,
                'final_placement': placement
            })
    
    return pd.DataFrame(all_placements)


def simulate_save_method(panel_df, seed=42):
    """SAVE 方法：Bottom-2 + Judges choose"""
    seasons = sorted(panel_df['season'].unique())
    all_placements = []
    
    np.random.seed(seed)
    
    for season in seasons:
        season_data = panel_df[panel_df['season'] == season].copy()
        weeks = sorted(season_data['week'].unique())
        
        remaining = set(season_data['celebrity_name'].unique())
        placements = {}
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week]
            week_data = week_data[week_data['celebrity_name'].isin(remaining)]
            
            if len(week_data) <= 2:
                break
            
            # Combined rank
            week_data['judge_rank'] = week_data['judge_score'].rank(ascending=False, method='min')
            week_data['fan_rank'] = week_data['fan_share'].rank(ascending=False, method='min')
            week_data['combined_rank'] = week_data['judge_rank'] + week_data['fan_rank']
            
            # Bottom-2
            bottom_2 = week_data.nlargest(2, 'combined_rank')
            
            # Judges choose 淘汰分数低者
            eliminated_idx = bottom_2['judge_score'].idxmin()
            eliminated = bottom_2.loc[eliminated_idx, 'celebrity_name']
            
            remaining.remove(eliminated)
        
        for i, contestant in enumerate(sorted(remaining)):
            placements[contestant] = i + 1
        
        for contestant, placement in placements.items():
            all_placements.append({
                'season': season,
                'method': 'SAVE',
                'celebrity_name': contestant,
                'final_placement': placement
            })
    
    return pd.DataFrame(all_placements)


def simulate_two_key_method(panel_df, seed=42):
    """TWO_KEY 方法：使用前面实现的完整 Two-Key 系统"""
    seasons = sorted(panel_df['season'].unique())
    all_placements = []
    
    for season in seasons:
        season_data = panel_df[panel_df['season'] == season].copy()
        weeks = sorted(season_data['week'].unique())
        
        remaining = set(season_data['celebrity_name'].unique())
        placements = {}
        
        for week in weeks:
            week_contestants = set(season_data[season_data['week'] == week]['celebrity_name'].values)
            week_contestants = week_contestants & remaining
            
            if len(week_contestants) <= 2:
                break
            
            # 调用 Two-Key 系统
            bottom_3 = select_bottom_3(season, week, panel_df, panel_df)
            saved = live_save_simulation(bottom_3, season, week, panel_df, panel_df, seed)
            eliminated = judges_eliminate(bottom_3, saved, season, week, panel_df, panel_df)
            
            if eliminated and eliminated in remaining:
                remaining.remove(eliminated)
        
        for i, contestant in enumerate(sorted(remaining)):
            placements[contestant] = i + 1
        
        for contestant, placement in placements.items():
            all_placements.append({
                'season': season,
                'method': 'TWO_KEY',
                'celebrity_name': contestant,
                'final_placement': placement
            })
    
    return pd.DataFrame(all_placements)


# ============================================================
# 主执行函数
# ============================================================

def main():
    base_dir = r"c:\Users\zhaoh\Desktop\MCM-czb-nzh-zhk"
    
    print("\n" + "=" * 70)
    print("TASK 4: FULL COMPARISON SIMULATION (4 METHODS)")
    print("=" * 70)
    
    # 加载数据
    fan_df, judge_df, data_df = load_all_data(base_dir)
    panel = create_weekly_panel(fan_df, judge_df)
    
    # 模拟四种方法
    print("\n  Simulating RANK method...")
    rank_results = simulate_rank_method(panel)
    
    print("  Simulating PERCENT method...")
    percent_results = simulate_percent_method(panel)
    
    print("  Simulating SAVE method...")
    save_results = simulate_save_method(panel, seed=42)
    
    print("  Simulating TWO_KEY method...")
    two_key_results = simulate_two_key_method(panel, seed=42)
    
    # 合并结果
    all_results = pd.concat([rank_results, percent_results, save_results, two_key_results], ignore_index=True)
    
    # 保存
    output_path = os.path.join(base_dir, "4_figures", "four_methods_comparison.csv")
    all_results.to_csv(output_path, index=False)
    
    print(f"\n  Results saved to: {output_path}")
    print(f"  Total records: {len(all_results)}")
    
    # 统计汇总
    print("\n" + "=" * 70)
    print("SUMMARY BY METHOD")
    print("=" * 70)
    
    for method in ['RANK', 'PERCENT', 'SAVE', 'TWO_KEY']:
        method_data = all_results[all_results['method'] == method]
        n_seasons = method_data['season'].nunique()
        n_contestants = method_data['celebrity_name'].nunique()
        print(f"  {method:10s}: {n_seasons} seasons, {n_contestants} contestants")
    
    print("\n" + "=" * 70)
    print("FULL COMPARISON COMPLETED!")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = main()
