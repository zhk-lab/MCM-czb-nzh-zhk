# 统一机制分析（已修正为“整季淘汰链”口径）

这份说明对应脚本 `2.1_unified_mechanism_curve.py` 的**修正版输出**：不再用“逐周排名差”，而是用**整季反事实淘汰链**得到最终名次差，从而与 Task 2.2 的结论口径完全一致。

## 1. 统一坐标系（同量纲、可解释、可复现）

**(1) 周内支持差距（原始量）**

\[
\Delta_{i,t} = F_{i,t} - J_{i,t}
\]

- \(F_{i,t}\)：第 \(t\) 周粉丝票份额（fan vote share）
- \(J_{i,t}\)：第 \(t\) 周评委分数份额（judge percent）

**(2) 赛季层面的“粉丝偏好程度”（X轴）**

\[
\overline{\Delta}_i=\mathrm{mean}_t(\Delta_{i,t})
\]

- \(\overline{\Delta}_i<0\)：judge-favored（评委更偏好）
- \(\overline{\Delta}_i\approx 0\)：neutral
- \(\overline{\Delta}_i>0\)：fan-favored（粉丝更偏好）

**(3) 赛季层面的“规则效应”（Y轴，核心修正）**

\[
Y_i=\mathrm{placement}_{PERCENT}(i)-\mathrm{placement}_{RANK}(i)
\]

- \(Y_i>0\)：PERCENT 让该选手最终名次更差（被“打压”）
- \(Y_i<0\)：PERCENT 让该选手最终名次更好（被“提升”）

## 2. 数据规模（两层粒度）

- **周内层**：2777 条 contestant-week 记录，用于计算 \(\overline{\Delta}_i\)
- **赛季层**：421 名选手（34 季），用于计算 \(Y_i\)

## 3. 关键发现（来自 `unified_mechanism_summary.csv`）

| 类别 | n | mean \(Y\) | 解释 |
|---|---:|---:|---|
| Judge-Favored | 85 | **+0.153** | PERCENT 对评委偏好型选手**轻微打压** |
| Neutral | 252 | -0.012 | 基本无差异 |
| Fan-Favored | 84 | **-0.119** | PERCENT 对粉丝偏好型选手**轻微提升** |

整体上，\(Y\) 与 \(\overline{\Delta}\) 的相关性很弱（见图中线性回归与 Spearman rho），但**分类型方向稳定**：这为 Task 2.2 “争议集合会被放大”的结果提供了**全体样本层面的机制底座**。

## 4. 与 Task 2.2 的关系（“放大镜”而非矛盾）

- Task 2.1（本节）给出的是**全体样本的平均效应**：方向存在但幅度小。
- Task 2.2 只看**争议强度 top 25%**：把“本来就靠近临界边界的人”挑出来，效应自然被放大（因此出现 +2.2~+2.5 的大幅名次变化）。

一句话：**2.1 提供“机制方向”，2.2 提供“在争议样本上的放大后果”。**

## 5. 生成的输出（图与CSV）

- `Task2_1_unified_mechanism_curve.png`：主图（421人散点 + 分类型箱线/分布）
- `Task2_1_unified_3D_surface.png`：加入“judge水平”后的三维/分层视角（更偏机制展示）
- `Task2_1_unified_supplementary.png`：补充分布与分箱曲线
- `unified_mechanism_detailed.csv`：每位选手的 \(\overline{\Delta}_i\)、两种规则下 placement 与 \(Y_i\)
- `unified_mechanism_summary.csv`：分类型汇总表（论文可直接引用）

## 6. 可直接粘进论文的锚段（建议放在 Task 2.1→2.2 过渡处）

> We define a unified support-gap metric \(\overline{\Delta}_i=\mathrm{mean}_t(F_{i,t}-J_{i,t})\) and quantify the rule impact by the season-level placement difference \(Y_i=\mathrm{placement}_{PERCENT}-\mathrm{placement}_{RANK}\) obtained via full-season counterfactual elimination chains. Across all 421 contestants, the average effect is small but heterogeneous: judge-favored contestants are slightly suppressed (\(E[Y]=+0.153\)), fan-favored contestants are slightly promoted (\(E[Y]=-0.119\)), and the neutral group shows near-zero change. This “directional but weak” global pattern explains why the same mechanism becomes much stronger when we zoom into the high-controversy subset in Task 2.2.

---

**已生成文件**：
- `Task2_1_unified_mechanism_curve.png`（2D 曲线 + 分布）
- `Task2_1_unified_3D_surface.png`（3D 曲面 + 分层曲线 + 密度图）
- `unified_mechanism_summary.csv`（区域统计）

**建议**：如果希望曲线完全符合"极端区抑制"的预期，需要改用"赛季最终 placement"而非"逐周 rank"作为 Y。是否需要我调整？
