# Task 2.2 报告：争议案例反事实模拟 → 异质性机制（fan_share 极端性）

本报告将 Task 2.2 的两部分工作按“**先完成题目要求 → 再解释深层原因**”合并为一条逻辑链：

1. **按题目要求进行争议分析与反事实模拟**：`task2.2/src/controversy_analysis.py`  
2. **解释 PERCENT 抑制效应的异质性来源**：`task2.2/src/extremeness_heterogeneity_experiment.py`

---

## 1. Part A：争议案例分析（题目要求主线）

### 1.1 研究问题

在评委与粉丝意见存在冲突（controversy）时：

- RANK vs PERCENT 的合并方法是否会改变淘汰链与最终结果？
- 引入 Judges Save（Bottom-two 后由评委决策）会如何影响？
- 四个历史争议案例（Jerry Rice / Billy Ray Cyrus / Bristol Palin / Bobby Bones）如何解释？

### 1.2 方法框架（四步数据驱动反事实模拟）

对应实现：`task2.2/src/controversy_analysis.py`

- **Step 1：争议选手识别**  
  基于每周评委排名与粉丝排名差异构造争议强度，并给出可复现的阈值/分组（fan-favored / judge-favored / neutral）。

- **Step 2：RANK vs PERCENT 整季淘汰链反事实模拟（核心）**  
  固定评委打分与 fan_share，逐周更新 active 集合，跑完整淘汰链得到最终 placement，再比较两规则下的差异。

- **Step 3：RANK + Save 机制模拟**  
  先用 RANK 选 bottom-two，再由评委在 bottom-two 中决定淘汰者（以 judge_total 更低者淘汰为可复现规则）。

- **Step 4：统计检验与可视化汇总**  
  在争议集合上计算 \(\Delta\)（相对 RANK 的名次变化）并做显著性检验，给出总体与分类型结论。

### 1.3 输出文件

#### 图表（`task2.2/figure/`）

- `Step1_controversy_radar.png`
- `Step2_3_counterfactual.png`
- `Step_Critical_Vote.png`
- `Step4_suppression_dashboard.png`

#### 表格（`task2.2/table/`）

- `controversy_all_contestants.csv`
- `counterfactual_all_controversial.csv`
- `suppression_stats.csv`
- `controversy_identification.csv`
- `controversy_placements_detail.csv`
- `suppression_scores.csv`

---

## 2. Part B：深层机制解释（PERCENT 抑制效应为何“异质”）

### 2.1 现象：PERCENT 的抑制不是“对所有争议一刀切”

在 Task 2.2 的分类型结果中，PERCENT 对不同争议类型的影响强度并不相同：  
**fan-favored 往往抑制弱/甚至轻微帮助**，而 **neutral / judge-favored 往往抑制强**。

因此需要回答：这种异质性来自哪里？

### 2.2 机制假设

PERCENT 的核心是“幅度连续计入”（fan_share + judge_percent）。  
如果 fan_share 自身具有更强的厚尾/极端性，那么其“高峰动员 / 低谷崩塌”的结构差异会被 PERCENT 非对称放大，从而产生异质效应。

### 2.3 极端性实验（验证链条）

对应实现：`task2.2/src/extremeness_heterogeneity_experiment.py`

实验做了三件事：

1. **证明 fan_share 的极端性显著强于 judge_percent**（CV/Gini 等指标对比）  
2. **证明不同争议类型的极端性结构不同**（fan_top10_rate / fan_bot10_rate 的组间差异）  
3. **把极端性特征与 \(\Delta_{\text{percent}}\) 的差异联系起来**（相关、分组趋势、交互项）

### 2.4 输出文件

#### 图表（`task2.2/figure/`）

- `Step5_extremeness_evidence.png`（3 面板：分布形态 / 尾部指标 / 分组极端性）
- `Step5_heterogeneity_correlation.png`（3 面板：整体相关 / 分组均值 / 分组斜率）

#### 表格与可粘贴段落（`task2.2/table/`）

- `Step5_extremeness_heterogeneity_table.csv`
- `Step5_conclusion_snippet.txt`

---

## 3. 合并后的最终结论（写作口径）

- **Task 2.2 主线结论**来自整季淘汰链反事实模拟：换规则会改变淘汰序列与最终名次，且在争议集合上可通过统计检验量化差异。  
- **异质性机制结论**由 Step 5 实验补全：fan_share 的极端性分布及其“高峰/低谷”结构差异，通过 PERCENT 的连续幅度计入机制被非对称放大，从而导致 PERCENT 对不同争议类型呈现不同的抑制强度。

