# Task 2.1 报告：RANK vs PERCENT 的偏向性比较与数据机制解释

本报告从“**现象 → 机制**”的链条组织 Task 2.1：

1. **先做方法比较**：用 `task2.1/src/voting_method_comparison.py` 量化 RANK 与 PERCENT 的差异度与偏向性（FFI），并生成可视化结果。  
2. **再解释为什么会偏向**：发现偏向性的根源不在“算法写法”，而在于 **fan_share 与 judge_percent 的分布形态差异**。  
3. **最后用分布实验验证**：用 `task2.1/src/distribution_analysis.py` 对比两种数据的离散性/不平等/厚尾程度，用统计检验与图表给出证据。

---

## 1. 方法比较：RANK vs PERCENT（先回答题目要求）

### 1.1 输入与数据表

- **粉丝票份额表**：`task1/table/fan_vote_shares.csv`  
  结构：`season, week, celebrity_name, fan_vote_share, method`（每周归一化求和为 1）
- **评委打分原始数据**：`2026_MCM_Problem_C_Data.csv`

### 1.2 度量与定义（与 README 保持一致）

对每个赛季、每一周（当周 active 选手集合上）构造四种排序：

- \(R_{\text{judge}}\)：只用评委分数（或评委 percent）得到的排序
- \(R_{\text{fan}}\)：只用估计的 fan share 得到的排序
- \(R_{\text{rank}}\)：按 rank 合并规则得到的排序/淘汰
- \(R_{\text{percent}}\)：按 percent 合并规则得到的排序/淘汰

选定排序距离 \(D(\cdot, \cdot)\)（主报告用 Kendall，tie 用 mid-rank）。

- **差异度**：\(D(R_{\text{rank}}, R_{\text{percent}})\)（以及淘汰者是否一致）
- **偏向 fan 的程度（FFI）**：

\[
\text{FFI}(m)=D(R_m,R_{\text{judge}})-D(R_m,R_{\text{fan}}),\quad m\in\{\text{rank},\text{percent}\}
\]

### 1.3 主要输出（图表）

图表均在 `task2.1/figure/`：

- `Task2_1_FFI_boxplot.png`：两种方法的 FFI 分布对比  
- `Task2_1_disagreement.png`：Kendall 距离分布 + 淘汰一致率  
- `Task2_1_FFI_scatter.png`：逐周 FFI(RANK) vs FFI(PERCENT)  
- `Task2_1_FFI_heatmap.png`：Week×Season 的 FFI 热力图  
- `Task2_1_season_summary.png`：综合仪表盘（趋势/距离/一致率/分布）

### 1.4 核心结论（现象层）

- **PERCENT 更偏向 fan votes**：\(\text{FFI}(\text{percent})\) 显著高于 \(\text{FFI}(\text{rank})\)  
- **RANK 更接近“平衡”**：FFI 接近 0（更像“压缩后”的折中）

---

## 2. 机制解释：为什么 PERCENT 会更偏 fan（追溯“本质原因”）

直觉上，PERCENT 的合成是“幅度相加”，而 RANK 是“名次相加（压缩差距）”：

\[
\text{PERCENT}: \; C_i = \underbrace{\frac{S_i}{\sum S}}_{\text{judge percent}} + \underbrace{p_i}_{\text{fan share}}
\]

关键在于：如果 \(p_i\) 的波动/不平等程度系统性大于 judge percent，那么在 **线性相加** 下，fan 端会更强地“拉动合成分”，从而把排序推向更接近 fan 的结果；而 RANK 会把这种“极端差距”压扁成有限的名次差。

---

## 3. 分布验证实验：fan_share 是否更“尖峰厚尾”（数据证据）

对应脚本：`task2.1/src/distribution_analysis.py`  
核心目标：在周级面板上比较 fan_share 与 judge_percent 的离散性/厚尾/不平等指标，并给出显著性检验。

### 3.1 关键数值（沿用 README 的表述口径）

跨周统计显示，**fan vote share 在所有离散性/不平等指标上都显著高于 judge percent**：

| 指标 | Fan 均值 | Judge 均值 | 比值（Fan/Judge） | 含义 |
|---|---:|---:|---:|---|
| Std | 0.0264 | 0.0128 | 2.05× | Fan 离散度约为 Judge 的 2 倍 |
| CV | 0.207 | 0.105 | 1.98× | 标准化后仍约 2 倍 |
| Gini | 0.112 | 0.057 | 1.97× | Fan “多的特多、少的特少”更明显 |
| 90/10 分位比 | 2.23 | 1.29 | 1.73× | Fan 的头尾差距更极端 |

### 3.2 图表输出

图表均在 `task2.1/figure/`：

- `Task2_1_distribution_boxplots.png`
- `Task2_1_distribution_histograms.png`
- `Task2_1_distribution_scatter.png`

表格在 `task2.1/table/`：

- `distribution_comparison.csv`

---

## 4. 结论回扣（现象 → 机制）

- **现象**：PERCENT 的 FFI 更大（更偏 fan）。  
- **机制**：fan_share 的分布更离散、更不平等、更厚尾；PERCENT 以幅度相加，因而更容易被 fan 端的“尖峰厚尾”牵引；RANK 则把差距压扁为名次差，因此更接近中性。

