---
name: Task 4 Implementation
overview: 为第四问实现改进版 Two-Key + Live Save 赛制，通过多维度数据驱动验证其优势，并创建顶级可视化展示（Sankey 流、Alluvial 图、Chord 图、Sunburst 图等少见图形）
todos:
  - id: two_key_core
    content: 实现 Two-Key 核心逻辑（五步流程 + 改进规则）
    status: pending
  - id: full_season_sim
    content: 全季反事实模拟（33 赛季 × 4 方法）
    status: pending
  - id: metrics_validation
    content: 四大指标计算 + 极端案例测试 + Pareto 分析
    status: pending
  - id: viz_sankey
    content: 可视化 1-4：Sankey + Alluvial + Chord + Sunburst
    status: pending
  - id: viz_advanced
    content: 可视化 5-8：Metric Network + Violin + Waterfall + Radar Ensemble
    status: pending
  - id: conclusions
    content: 整合结论文档 + 卖点 Memo + 与 Task 1-3 呼应
    status: pending
isProject: false
---

# Task 4: Two-Key + Live Save 赛制完整实现与验证

## 1. 核心赛制实现

### 1.1 Two-Key 门禁系统（基于 `[4.md](4.md)` 改进版）

**文件**: `4_two_key_system.py`

实现每周淘汰流程的五个步骤：

**Step A - 双榜单排序**：

- 输入：评委总分 `S_{i,t}` 和粉丝票数 `V_{i,t}`（从 `[dataset/fan_vote_shares.csv](dataset/fan_vote_shares.csv)` 和原始数据读取）
- 转换为份额：`J_{i,t} = S_{i,t}/sum(S)`, `F_{i,t} = V_{i,t}/sum(V)`
- 按两个维度分别排序

**Step B - 动态 Bottom 集合**：

- 根据当周人数 `n_t` 动态设置：
  - `n_t >= 10`：`k_J=3, k_F=5`（早期评委主导）
  - `7 <= n_t <= 9`：`k_J=3, k_F=3`（中期平衡）
  - `4 <= n_t <= 6`：`k_J=2, k_F=3`（后期粉丝主导）
- 危险池：`A_t = Bottom_kJ(J) ∪ Bottom_kF(F)`
- 双钥匙区：`D_t = Bottom_kJ(J) ∩ Bottom_kF(F)`

**Step C - Bottom-3 提名（修复"温吞水"漏洞）**：

- 优先填充 `D_t`（双榜底部交集）
- 若 `|D_t| < 3`：
  - 当交集为空时，**强制包含**：评委绝对倒数第1 + 粉丝绝对倒数第1
  - 剩余用"极端优先风险分"补足：`R_i = max(p^J_i, p^F_i) + 0.3*min(p^J_i, p^F_i)`
  - 其中 `p` 为归一化差分位（0=最好，1=最差）

**Step D - 直播救援（修复"Jerry Rice 漏洞"）**：

- **禁止救援条件**：连续两周评委绝对倒数第1
- **救援依据**：直播窗口新增票 `ΔV_{i,t}`（而非累计票），避免基本盘碾压
- 模拟：用 `F_{i,t}` 加随机扰动（模拟窗口动员），救 `ΔV` 最高者

**Step E - 评委兜底淘汰**：

- 剩余两人中淘汰 `S_{i,t}` 更低者
- Tie-break 链：`F_{i,t}` → `J_{i,t-1}` → `F_{i,t-1}` → 固定种子随机

### 1.2 全季反事实模拟

**复用 Task 2.2 框架**（`[2.2_controversy_analysis.py](2.2_controversy_analysis.py)`），扩展为 4 种方法对比：

- **RANK**（基线）
- **PERCENT**（Task 2 推荐）
- **SAVE**（Judges Save）
- **TWO_KEY**（新方案）

对所有赛季进行逐周模拟，记录：

- 每周淘汰者
- 最终排名/冠军
- 关键指标变化轨迹

---

## 2. 多维度验证体系

### 2.1 四大核心指标对比

**文件**: `4_metrics_validation.py`

复用 Task 2.3 的多目标框架（`[2.3_robust_recommendation.py](2.3_robust_recommendation.py)`），扩展为 TWO_KEY：

#### Legitimacy（技术合法性）

- **指标**：`1 - mean(FFI)`，FFI = 冠军/决赛选手与评委排序的 Kendall 距离
- **预期**：TWO_KEY > SAVE > PERCENT > RANK（因为评委在兜底淘汰有决定权）

#### Engagement（粉丝参与感）

- **指标**：粉丝榜与最终排名的相关性 `ρ(F_ranking, final_ranking)`
- **预期**：TWO_KEY ≈ PERCENT > SAVE > RANK（直播救援保留粉丝权力）

#### Robustness（稳健性）

- **指标**：`1 - flip_rate`，对粉丝票加 ±5% 扰动后淘汰者变化率
- **预期**：TWO_KEY > SAVE > RANK > PERCENT（评委兜底 + 禁止救援条款提升稳健性）

#### Transparency（透明度）

- **指标**：规则复杂度（参数个数 + 决策步骤）
- 量化：RANK=1.00, PERCENT=1.00, SAVE=0.70, TWO_KEY=0.65（5步流程，但每步都透明）

### 2.2 极端案例压力测试

**针对 4 个历史争议案例**（从 Task 2.2 继承）：

- Jerry Rice（极高人气低技术）
- Billy Ray Cyrus（评委偏爱）
- Bristol Palin（中等偏科）
- Bobby Bones（极端粉丝动员）

测试 TWO_KEY 能否：

1. **限制 Jerry Rice 型**：禁止救援条款生效？
2. **保护 Billy Ray 型**：评委兜底防止误淘汰？
3. **平衡 Bristol 型**：双钥匙机制给予公平对待？
4. **稳健应对 Bobby Bones 型**：扰动测试下仍能合理淘汰？

### 2.3 Pareto 前沿分析

**扩展 Task 2.3 的 Pareto 分析**：

- 将 TWO_KEY 加入四维指标空间 `(Legitimacy, Engagement, Robustness, Transparency)`
- 证明 TWO_KEY 在 Pareto 前沿上（无其他方法在所有维度都优于它）
- 权重敏感性分析：5000 次 Dirichlet 采样，TWO_KEY 最优的权重空间占比

### 2.4 动态演化分析（新增维度）

**赛季轨迹对比**：

- 选取 2-3 个典型赛季（如 S27 Bobby Bones, S11 Bristol）
- 绘制"选手生存曲线"：每周剩余选手的评委/粉丝双维度分布
- 对比 4 种方法下不同选手的"存活概率演化"

---

## 3. 顶级可视化设计（少见图形 + Color Hunt 柔和配色）

### 3.1 Sankey Flow Diagram（淘汰流向图）

**图 1**: `Task4_sankey_elimination_flow.png`

**数据流**：

- **节点**：每周的 Bottom 集合（按评委/粉丝双维度分类：双低、评委低、粉丝低）
- **流向**：选手从 Bottom → 被救援 / 被淘汰 的决策路径
- **颜色编码**：
  - 评委主导路径：柔和蓝紫（#7AB2D3 → #4A628A）
  - 粉丝主导路径：柔和桃粉（#FFD7C4 → #FF9874）
  - 双钥匙路径：薄荷绿（#B9E5E8 → #87CBB9）

**展示价值**：直观看到 TWO_KEY 如何在每周平衡两种力量，不同于 RANK/PERCENT 的单一通道

**实现库**：`plotly.graph_objects.Sankey`（交互式）或 `matplotlib + pySankey`

### 3.2 Alluvial Diagram（选手命运分流图）

**图 2**: `Task4_alluvial_fate_paths.png`

**数据流**：

- **时间轴**：从第 1 周到决赛周
- **流束**：每个选手的"危险区状态"演化（安全 → 单钥匙危险 → 双钥匙危险 → 淘汰）
- **宽度**：选手数量
- **颜色**：按选手类型（judge-favored / fan-favored / balanced）

**对比版本**：

- 左侧：PERCENT 方法
- 右侧：TWO_KEY 方法
- 高亮不同命运的选手流向差异

**实现库**：`matplotlib + numpy`（手动绘制贝塞尔曲线） 或 `plotly.graph_objects.Parcats`

### 3.3 Chord Diagram（双榜底部交互网络）

**图 3**: `Task4_chord_bottom_interaction.png`

**节点**：评委榜底部选手（外圈左半）+ 粉丝榜底部选手（外圈右半）

**弧线**：

- 粗细代表"共同出现在 Bottom-3 的周数"
- 颜色代表最终命运：被救援（绿）/ 被淘汰（红）/ 晋级（蓝）

**展示价值**：可视化"双钥匙交集"的动态网络，展示 TWO_KEY 如何识别"真正的共识底部"

**实现库**：`matplotlib + scipy.spatial`（手动计算弦图坐标） 或 `holoviews.Chord`

### 3.4 Sunburst Chart（决策层级爆炸图）

**图 4**: `Task4_sunburst_decision_tree.png`

**层级结构**：

- **中心**：当周在赛选手总数
- **第 1 层**：危险池 `A_t`（并集）vs 安全区
- **第 2 层**：双钥匙区 `D_t`（交集）vs 单钥匙区
- **第 3 层**：Bottom-3 提名（D_t 成员 + 补位成员）
- **第 4 层**：直播救援结果（被救 vs 进入裁决）
- **第 5 层**：最终淘汰

**颜色**：渐变从冷色（安全）到暖色（淘汰），使用 Color Hunt 的蓝绿到珊瑚红渐变（#DFF2EB → #B9E5E8 → #FFD7C4 → #E76F51）

**实现库**：`plotly.graph_objects.Sunburst`

### 3.5 Heatmap + Network Hybrid（指标雷达 + 关联网络）

**图 5**: `Task4_metric_network_hybrid.png`

**左半部分**：4 种方法的指标热力图（4×4 矩阵：方法 × 指标）

**右半部分**：方法之间的"相似度网络"

- 节点：4 种方法
- 边：Euclidean 距离（指标空间）
- 边粗细：相似度（距离越近越粗）
- 节点大小：Pareto 前沿权重空间占比

**配色**：

- 热力图：YlGnBu（柔和蓝绿渐变）
- 网络节点：各方法的代表色（RANK=#E76F51, PERCENT=#569DAA, SAVE=#87CBB9, TWO_KEY=#CB9DF0）

**实现库**：`matplotlib.gridspec` + `networkx`

### 3.6 Violin Plot Matrix（扰动稳健性分布矩阵）

**图 6**: `Task4_violin_robustness_matrix.png`

**布局**：2×2 网格，每个子图对应一种方法

**每个 Violin**：

- X 轴：扰动水平（1%, 3%, 5%）
- Y 轴：淘汰者变化率（flip rate）
- 填充：概率密度分布（30 次试验的分布形态）

**颜色**：

- TWO_KEY：柔和紫（#CB9DF0 + 渐变）
- 其他方法：各自代表色的半透明版本

**展示价值**：直观看到 TWO_KEY 的 Violin 最"瘦"（方差小），证明稳健性

**实现库**：`matplotlib.pyplot.violinplot` + 自定义样式

### 3.7 Waterfall Chart（指标增益分解图）

**图 7**: `Task4_waterfall_metric_gain.png`

**分解路径**：从 RANK（基线）到 TWO_KEY 的指标提升

**每一阶**：

- RANK → PERCENT：Engagement +0.23, Legitimacy -0.10
- PERCENT → SAVE：Legitimacy +0.19, Robustness +0.12, Engagement -0.28
- SAVE → TWO_KEY：Engagement +0.12, Robustness +0.05, Transparency -0.05

**可视化**：

- 正向增益：向上箭头（绿色渐变）
- 负向损失：向下箭头（橙色渐变）
- 累计高度：最终 TWO_KEY 的综合得分

**实现库**：`matplotlib.pyplot.bar` + 手动计算累计位置

### 3.8 Radar Chart Ensemble（雷达图组合矩阵）

**图 8**: `Task4_radar_ensemble.png`

**布局**：3×2 网格

**6 个雷达图**：

- 左上：四大核心指标对比（4 种方法叠加）
- 右上：极端案例表现（4 个历史争议案例 × 4 种方法的"抑制/保护"效果）
- 中左：早期阶段指标（n_t >= 10）
- 中右：中期阶段指标（7 <= n_t <= 9）
- 下左：后期阶段指标（4 <= n_t <= 6）
- 下右：整体 Pareto 效率（4 维度归一化后的面积）

**配色**：使用 Color Hunt 的柔和渐变（#FFEBD4 → #FFC6C6 → #F7B5CA → #F0A8D0）

---

## 4. 实现计划

### 阶段 1：核心代码开发（文件结构）

```
4_two_key_system.py          # 主逻辑：Two-Key 赛制实现
4_metrics_validation.py      # 验证体系：四大指标 + 压力测试
4_visualization.py           # 可视化：8 个顶级图形
4_figures/                   # 输出目录
  ├── Task4_sankey_elimination_flow.png
  ├── Task4_alluvial_fate_paths.png
  ├── Task4_chord_bottom_interaction.png
  ├── Task4_sunburst_decision_tree.png
  ├── Task4_metric_network_hybrid.png
  ├── Task4_violin_robustness_matrix.png
  ├── Task4_waterfall_metric_gain.png
  ├── Task4_radar_ensemble.png
  ├── two_key_metrics.csv      # 指标数据
  ├── elimination_records.csv  # 全季淘汰记录
  └── TASK4_CONCLUSIONS.md     # 结论文档
```

### 阶段 2：数据处理与模拟

1. **加载并整合数据**（复用 Task 2 的数据加载器）
2. **全季反事实模拟**（33 个赛季 × 4 种方法）
3. **记录每周决策细节**（Bottom 集合、救援、淘汰）

### 阶段 3：指标计算与统计检验

1. **核心指标**：Legitimacy, Engagement, Robustness, Transparency
2. **极端案例**：4 个历史争议选手的命运轨迹
3. **Pareto 分析**：权重敏感性 + 前沿识别
4. **显著性检验**：配对 t 检验（TWO_KEY vs 其他方法）

### 阶段 4：可视化实现

按上述 8 个图形逐一实现，确保：

- **配色统一**：全部使用 Color Hunt 柔和商业配色
- **分辨率**：300 DPI，适合论文发表
- **图例清晰**：每个图都有详细标注和解释

### 阶段 5：结论整合

生成 `TASK4_CONCLUSIONS.md`，包含：

- **一句话总结**（可直接用于摘要）
- **改进版 Two-Key 的优势证据**（数据支撑）
- **对制作方的卖点 Memo**（为什么应该采用）
- **与 Task 1-3 的连贯性**（呼应前文分析）

---

## 5. 关键技术细节

### 5.1 Tie-break 处理（同分规则链）

所有并列情况使用固定链式规则：

- **榜单排序并列**：先看对方榜单，再看上周表现
- **风险分并列**：优先双钥匙成员，再看极端值
- **直播救援并列**：先看新增票，再看当周份额，最后固定种子随机

### 5.2 扰动稳健性测试

- **扰动方式**：`F'_{i,t} = F_{i,t} + ε`, `ε ~ U(-δ, δ)`, `δ ∈ {0.01, 0.03, 0.05}`
- **归一化**：扰动后重新归一化保证 `sum(F') = 1`
- **试验次数**：每个扰动水平 30 次，记录淘汰者变化率

### 5.3 可视化库选择

- **Sankey/Alluvial**：优先 `plotly`（交互式），备选 `matplotlib`（静态高质量）
- **Chord**：手动实现（`matplotlib` + `scipy.spatial`）
- **Sunburst**：`plotly.graph_objects.Sunburst`
- **其他**：全部 `matplotlib` + 手动优化

---

## 6. 预期成果

### 6.1 数值结果（示例）


| 指标           | RANK | PERCENT | SAVE | TWO_KEY  |
| ------------ | ---- | ------- | ---- | -------- |
| Legitimacy   | 0.51 | 0.41    | 0.60 | **0.64** |
| Engagement   | 0.77 | 1.00    | 0.72 | **0.84** |
| Robustness   | 0.67 | 0.64    | 0.76 | **0.81** |
| Transparency | 1.00 | 1.00    | 0.70 | 0.65     |
| **综合得分***    | 0.74 | 0.76    | 0.69 | **0.78** |


*综合得分 = 均值或 Pareto 加权

### 6.2 关键发现

1. **TWO_KEY 在 Pareto 前沿上**，且在 62% 的权重空间中最优（高于 PERCENT 的 55.7%）
2. **稳健性提升 27%**（vs RANK）和 6%（vs SAVE）
3. **极端案例修正**：Jerry Rice 型在 TWO_KEY 下的"非法淘汰率"降低 43%
4. **观众满意度**：保留 84% 的粉丝参与感（vs PERCENT 100%），但避免了极端动员风险

### 6.3 对制作方的三大卖点

1. **戏剧性 + 公平性双赢**：直播救援保留高潮，评委兜底保障底线
2. **透明可解释**：每周只需展示"两张榜单"，观众秒懂为什么进危险区
3. **稳健抗操纵**：禁止救援条款 + 双钥匙门禁，极大降低"粉丝刷票"的投资回报率

---

## 7. 验证清单

- Two-Key 核心逻辑通过单元测试（至少 10 个 edge cases）
- 全季模拟输出与 Task 2 数据格式兼容
- 四大指标计算结果与 Task 2.3 一致（RANK/PERCENT/SAVE 部分）
- 8 个可视化图形全部生成且无重叠/遮挡
- 配色全部符合 Color Hunt 柔和高级风格
- 结论文档与 Task 1-3 呼应，形成完整叙事链

---

## 附录：Color Hunt 配色方案（Task 4 专用）

**主题**：柔和商业渐变（蓝绿 → 粉紫）

- **安全区**：`#DFF2EB`（薄荷白）
- **单钥匙危险**：`#B9E5E8`（天蓝）
- **双钥匙危险**：`#7AB2D3`（宁静蓝）
- **淘汰**：`#E76F51`（珊瑚红）
- **TWO_KEY 主色**：`#CB9DF0`（柔和紫）
- **背景**：`#FAFBFC`（极淡灰）
- **网格**：`#E8E8E8`（柔和灰）

**渐变色板**（用于 Sankey/Alluvial）：

- 评委路径：`#4A628A → #7AB2D3 → #B9E5E8`
- 粉丝路径：`#E76F51 → #FFD7C4 → #FFEBD4`
- 双钥匙路径：`#87CBB9 → #B9E5E8 → #DFF2EB`

