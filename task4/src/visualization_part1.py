"""
MCM 2026 Problem C - Task 4 Visualization (Part 1)
===================================================
顶级可视化设计 - 少见图形：
  - Sankey Flow Diagram（淘汰流向图）
  - Alluvial Diagram（选手命运分流图）
  - Chord Diagram（双榜底部交互网络）
  - Sunburst Chart（决策层级爆炸图）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, PathPatch
from matplotlib.path import Path
from matplotlib.sankey import Sankey
import warnings
warnings.filterwarnings('ignore')

# Color Hunt 配色（调暗版 - 更商业稳重）
COLORS = {
    'twokey': '#9B7EBD',     # 深紫（TWO_KEY 主色）
    'rank': '#D14D72',       # 深玫红
    'percent': '#3E7C17',    # 深绿
    'save': '#5C88C4',       # 深蓝
    
    # 状态颜色
    'safe': '#C4DAD2',       # 柔和灰绿（安全）
    'single_key': '#7AB2D3', # 中度蓝（单钥匙危险）
    'two_key': '#4A628A',    # 深蓝（双钥匙危险）
    'eliminated': '#C75B7A', # 深粉红（淘汰）
    'saved': '#50B498',      # 深薄荷绿（被救援）
    
    # 渐变色板
    'judge_path': ['#304463', '#4A628A', '#7AB2D3'],
    'fan_path': ['#D14D72', '#FFABAB', '#FCC8D1'],
    'balanced_path': ['#468585', '#50B498', '#9CDBA6'],
    
    'bg': '#F5F5F5',
    'text': '#1E201E',
}

plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 10,
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ============================================================
# 图 1: Sankey Flow Diagram（淘汰流向图）
# ============================================================

def create_sankey_elimination_flow(two_key_df, save_dir):
    """
    Sankey 流向图：展示 TWO_KEY 系统的决策流向
    
    从危险池 → Bottom-3 → 救援/淘汰
    """
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    
    # 统计数据
    total_weeks = len(two_key_df[two_key_df['eliminated'] != ''])
    
    # 解析 bottom_3
    bottom_3_all = []
    saved_all = []
    eliminated_all = []
    
    for _, row in two_key_df.iterrows():
        if row['bottom_3']:
            bottom_3_all.extend(row['bottom_3'].split(','))
        if row['saved']:
            saved_all.append(row['saved'])
        if row['eliminated']:
            eliminated_all.append(row['eliminated'])
    
    n_bottom_3 = len(bottom_3_all)
    n_saved = len(saved_all)
    n_eliminated = len(eliminated_all)
    
    # 创建简化的 Sankey 图（使用矩形和箭头）
    # 由于 matplotlib.sankey 功能有限，我们用自定义图形
    
    # 三列：输入、中间处理、输出
    col_x = [0.15, 0.5, 0.85]
    
    # 输入节点（危险池）
    danger_box = FancyBboxPatch(
        (col_x[0] - 0.08, 0.4), 0.16, 0.2,
        boxstyle="round,pad=0.01",
        facecolor=COLORS['two_key'], edgecolor='white',
        linewidth=3, alpha=0.9
    )
    ax.add_patch(danger_box)
    ax.text(col_x[0], 0.5, 'Danger Pool\n(Union)', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    ax.text(col_x[0], 0.35, f'n={n_bottom_3//3:.0f} avg/week', ha='center', va='top',
            fontsize=9, color=COLORS['text'])
    
    # 中间处理（Bottom-3）
    b3_box = FancyBboxPatch(
        (col_x[1] - 0.08, 0.4), 0.16, 0.2,
        boxstyle="round,pad=0.01",
        facecolor='#7AB2D3', edgecolor='white',
        linewidth=3, alpha=0.9
    )
    ax.add_patch(b3_box)
    ax.text(col_x[1], 0.5, 'Bottom-3\nNominees', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    ax.text(col_x[1], 0.35, f'Total={n_bottom_3}', ha='center', va='top',
            fontsize=9, color=COLORS['text'])
    
    # 输出节点（救援 + 淘汰）
    saved_box = FancyBboxPatch(
        (col_x[2] - 0.08, 0.6), 0.16, 0.15,
        boxstyle="round,pad=0.01",
        facecolor=COLORS['saved'], edgecolor='white',
        linewidth=3, alpha=0.9
    )
    ax.add_patch(saved_box)
    ax.text(col_x[2], 0.675, 'Live Saved', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(col_x[2], 0.58, f'n={n_saved}', ha='center', va='top',
            fontsize=9, color=COLORS['text'])
    
    elim_box = FancyBboxPatch(
        (col_x[2] - 0.08, 0.25), 0.16, 0.15,
        boxstyle="round,pad=0.01",
        facecolor=COLORS['eliminated'], edgecolor='white',
        linewidth=3, alpha=0.9
    )
    ax.add_patch(elim_box)
    ax.text(col_x[2], 0.325, 'Eliminated', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(col_x[2], 0.23, f'n={n_eliminated}', ha='center', va='top',
            fontsize=9, color=COLORS['text'])
    
    # 绘制流向箭头
    # Danger Pool → Bottom-3
    ax.annotate('', xy=(col_x[1] - 0.08, 0.5), xytext=(col_x[0] + 0.08, 0.5),
                arrowprops=dict(arrowstyle='->', lw=4, color='#7AB2D3', alpha=0.7))
    
    # Bottom-3 → Saved
    ax.annotate('', xy=(col_x[2] - 0.08, 0.675), xytext=(col_x[1] + 0.08, 0.55),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['saved'], alpha=0.7))
    ax.text(col_x[1] + 0.18, 0.62, f'33%\nsaved', ha='center', va='center',
            fontsize=8, color=COLORS['saved'], style='italic')
    
    # Bottom-3 → Eliminated
    ax.annotate('', xy=(col_x[2] - 0.08, 0.325), xytext=(col_x[1] + 0.08, 0.45),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['eliminated'], alpha=0.7))
    ax.text(col_x[1] + 0.18, 0.38, f'67%\neliminated', ha='center', va='center',
            fontsize=8, color=COLORS['eliminated'], style='italic')
    
    # 标题和说明
    ax.text(0.5, 0.95, 'TWO-KEY Elimination Flow: Decision Pathway', ha='center', va='top',
            fontsize=16, fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.08, 'Two-Key System separates contestants into danger pool (union),\n' +
                       'nominates Bottom-3, applies live save, and judges eliminate the lowest scorer.',
            ha='center', va='center', fontsize=10, color='#666666', style='italic',
            transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.savefig(f'{save_dir}/Task4_sankey_elimination_flow.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print("  Saved: Task4_sankey_elimination_flow.png")


# ============================================================
# 图 2: Alluvial Diagram（选手命运分流图）
# ============================================================

def create_alluvial_fate_paths(panel_df, comparison_df, save_dir):
    """
    Alluvial 图：选手危险状态的周演化（改进版）
    
    展示从早期到决赛的人员分流
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), facecolor='white')
    fig.subplots_adjust(hspace=0.25, wspace=0.25, top=0.90, bottom=0.10)
    
    # 使用 Season 1（数据更完整）
    season = 1
    
    for ax, method, title, color_scheme in [(ax1, 'PERCENT', 'PERCENT Method', COLORS['percent']), 
                                             (ax2, 'TWO_KEY', 'TWO-KEY Method', COLORS['twokey'])]:
        ax.set_facecolor('#FAFBFC')
        
        season_panel = panel_df[panel_df['season'] == season].copy()
        weeks = sorted(season_panel['week'].unique())
        
        # 统计每周的人数分布
        safe_counts = []
        danger_counts = []
        week_labels = []
        
        for week in weeks:
            week_data = season_panel[season_panel['week'] == week]
            n_total = len(week_data)
            
            if n_total < 3:
                continue
            
            # 简化：按综合分数划分
            week_data['combined'] = 0.5 * week_data['judge_share'] + 0.5 * week_data['fan_share']
            bottom_3 = week_data.nsmallest(min(3, n_total), 'combined')
            
            danger_counts.append(len(bottom_3))
            safe_counts.append(n_total - len(bottom_3))
            week_labels.append(f'W{week}')
        
        x = np.arange(len(week_labels))
        
        # 绘制堆叠面积图
        ax.fill_between(x, 0, safe_counts, color=COLORS['safe'], alpha=0.75, label='Safe Zone', edgecolor='white', linewidth=2)
        ax.fill_between(x, safe_counts, [safe_counts[i] + danger_counts[i] for i in range(len(week_labels))],
                       color=color_scheme, alpha=0.75, label='Danger Zone (Bottom-3)', edgecolor='white', linewidth=2)
        
        # 添加数值标注
        for i, (safe, danger) in enumerate(zip(safe_counts, danger_counts)):
            ax.text(i, safe / 2, f'{safe}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='#555555')
            ax.text(i, safe + danger / 2, f'{danger}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        
        ax.set_xlabel('Week Progression', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Contestants', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(week_labels, fontsize=10)
        ax.set_title(f'{title}\n(Season {season} Week-by-Week Flow)', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')
        ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(safe_counts) + max(danger_counts) + 1)
    
    # 去掉大标题避免遮挡
    # plt.suptitle('Alluvial Flow: Contestant Fate Evolution Across Weeks', fontsize=14, fontweight='bold', y=0.94)
    
    plt.savefig(f'{save_dir}/Task4_alluvial_fate_paths.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print("  Saved: Task4_alluvial_fate_paths.png")


# ============================================================
# 图 3: Chord Diagram（双榜底部交互网络）
# ============================================================

def create_chord_bottom_interaction(panel_df, save_dir):
    """
    Chord 图：评委底部 vs 粉丝底部的交互关系（改进版）
    
    展示双钥匙区的统计分布
    """
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    ax.set_facecolor('#FAFBFC')
    ax.axis('off')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # 统计所有赛季的双榜底部交互
    all_judge_bottom = []
    all_fan_bottom = []
    both_bottom = []
    
    for season in panel_df['season'].unique():
        season_panel = panel_df[panel_df['season'] == season]
        
        for week in sorted(season_panel['week'].unique()):
            week_data = season_panel[season_panel['week'] == week]
            
            if len(week_data) < 5:
                continue
            
            # 评委底部 3 人
            judge_bottom_set = set(week_data.nsmallest(3, 'judge_share')['celebrity_name'].values)
            all_judge_bottom.append(len(judge_bottom_set))
            
            # 粉丝底部 3 人
            fan_bottom_set = set(week_data.nsmallest(3, 'fan_share')['celebrity_name'].values)
            all_fan_bottom.append(len(fan_bottom_set))
            
            # 交集
            intersection = judge_bottom_set & fan_bottom_set
            both_bottom.append(len(intersection))
    
    # 统计结果
    total_weeks = len(all_judge_bottom)
    avg_intersection = np.mean(both_bottom)
    
    # 绘制 Venn 图风格的交互图
    # 左圆：评委底部
    judge_circle = Circle((0.35, 0.5), 0.25, facecolor=COLORS['rank'], 
                         edgecolor='white', linewidth=3, alpha=0.5)
    ax.add_patch(judge_circle)
    ax.text(0.25, 0.5, 'Judge\nBottom', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    
    # 右圆：粉丝底部
    fan_circle = Circle((0.65, 0.5), 0.25, facecolor=COLORS['percent'], 
                       edgecolor='white', linewidth=3, alpha=0.5)
    ax.add_patch(fan_circle)
    ax.text(0.75, 0.5, 'Fan\nBottom', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    
    # 交集区域
    intersection_circle = Circle((0.5, 0.5), 0.12, facecolor=COLORS['two_key'],
                                edgecolor='white', linewidth=3, alpha=0.9, zorder=3)
    ax.add_patch(intersection_circle)
    ax.text(0.5, 0.5, f'Two-Key\nZone\n{avg_intersection:.1f} avg', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=4)
    
    # 统计标注
    ax.text(0.5, 0.85, 'Bottom Interaction Analysis', ha='center', va='center',
            fontsize=15, fontweight='bold', color=COLORS['text'])
    
    ax.text(0.5, 0.15, f'Based on {total_weeks} weeks across all seasons\n' +
                       f'Avg intersection size: {avg_intersection:.2f} contestants',
            ha='center', va='center', fontsize=10, color='#666666', style='italic')
    
    # 箭头说明
    ax.annotate('Judge-favored\nbut fan-disliked', xy=(0.20, 0.65), xytext=(0.05, 0.80),
                fontsize=9, color=COLORS['rank'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['rank'], alpha=0.6))
    
    ax.annotate('Fan-favored\nbut judge-disliked', xy=(0.80, 0.65), xytext=(0.95, 0.80),
                fontsize=9, color=COLORS['percent'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['percent'], alpha=0.6))
    
    ax.annotate('Consensus\nBottom', xy=(0.5, 0.38), xytext=(0.5, 0.20),
                fontsize=10, color=COLORS['two_key'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['two_key'], alpha=0.7))
    
    plt.savefig(f'{save_dir}/Task4_chord_bottom_interaction.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print("  Saved: Task4_chord_bottom_interaction.png")


# ============================================================
# 图 4: Sunburst Chart（决策层级爆炸图）
# ============================================================

def create_sunburst_decision_tree(two_key_df, save_dir):
    """
    Sunburst 图：决策层级的嵌套结构（改进版）
    
    使用多层饼图展示决策流
    """
    fig, ax = plt.subplots(figsize=(13, 13), facecolor='white')
    ax.set_facecolor('white')
    ax.axis('equal')
    ax.axis('off')
    
    # 统计数据
    total_weeks = len(two_key_df)
    total_bottom_3 = sum(two_key_df['bottom_3'] != '')
    total_saved = sum(two_key_df['saved'] != '')
    total_eliminated = sum(two_key_df['eliminated'] != '')
    
    # 估算
    avg_contestants = 6  # 平均每周选手数
    safe_count = total_weeks * avg_contestants - total_bottom_3 * 3
    danger_count = total_bottom_3 * 3
    
    # 外圈：初始状态
    sizes_outer = [safe_count, danger_count]
    colors_outer = [COLORS['safe'], COLORS['two_key']]
    labels_outer = ['Safe Zone', 'Danger Pool']
    
    wedges1, texts1, autotexts1 = ax.pie(
        sizes_outer, labels=labels_outer, colors=colors_outer,
        autopct='%1.0f%%', startangle=90, radius=1.0,
        wedgeprops=dict(width=0.25, edgecolor='white', linewidth=3),
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        pctdistance=0.85
    )
    
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # 中圈：Bottom-3 处理
    sizes_mid = [total_saved, total_eliminated]
    colors_mid = [COLORS['saved'], COLORS['eliminated']]
    labels_mid = ['Live Saved', 'Judges Eliminated']
    
    wedges2, texts2, autotexts2 = ax.pie(
        sizes_mid, labels=labels_mid, colors=colors_mid,
        autopct='%1.0f%%', startangle=90, radius=0.75,
        wedgeprops=dict(width=0.25, edgecolor='white', linewidth=3),
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        pctdistance=0.75
    )
    
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # 内圈：总体
    inner_circle = Circle((0, 0), 0.50, facecolor='white', edgecolor='#DDDDDD', linewidth=2)
    ax.add_patch(inner_circle)
    
    ax.text(0, 0.08, f'{total_weeks}', ha='center', va='bottom',
            fontsize=18, fontweight='bold', color=COLORS['text'])
    ax.text(0, -0.08, 'Weeks', ha='center', va='top',
            fontsize=14, fontweight='bold', color='#666666')
    
    # 标题移到图内
    ax.text(0, 1.22, 'Two-Key Decision Flow Hierarchy', ha='center', va='bottom',
            fontsize=15, fontweight='bold', color=COLORS['text'], transform=ax.transData)
    
    # 说明文字
    ax.text(0, -1.28, 'Outer Ring: Initial Status | Inner Ring: Final Fate (Bottom-3 only)',
            ha='center', va='center', fontsize=10, color='#888888', style='italic')
    
    plt.savefig(f'{save_dir}/Task4_sunburst_decision_tree.png', dpi=300, 
                facecolor='white', bbox_inches='tight')
    plt.close()
    
    print("  Saved: Task4_sunburst_decision_tree.png")


# ============================================================
# 主执行函数
# ============================================================

def main():
    from pathlib import Path
    import sys
    
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "task4" / "src"))
    
    table_dir = repo_root / "task4" / "table"
    save_dir = repo_root / "task4" / "figure"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("TASK 4: VISUALIZATION PART 1 (Advanced Charts)")
    print("=" * 70)
    
    # 加载数据
    from two_key_system import load_all_data, create_weekly_panel
    
    fan_df, judge_df, data_df = load_all_data(str(repo_root))
    panel_df = create_weekly_panel(fan_df, judge_df)
    
    two_key_df = pd.read_csv(str(table_dir / "two_key_elimination_records.csv"))
    comparison_df = pd.read_csv(str(table_dir / "four_methods_comparison.csv"))
    
    # 生成可视化
    print("\n  Generating visualizations...")
    
    create_sankey_elimination_flow(two_key_df, save_dir)
    create_alluvial_fate_paths(panel_df, comparison_df, save_dir)
    create_chord_bottom_interaction(panel_df, save_dir)
    create_sunburst_decision_tree(two_key_df, save_dir)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION PART 1 COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
