## 第一问建模思路：
第一步：我们将粉丝投票的推断建模为一个 贝叶斯逆问题 (Bayesian Inverse Problem)。鉴于解空间是一个受淘汰规则严格限制的高维、非凸 可行域 (Feasible Region)，解析求解极难，因此我们采用 可行域采样 (Feasible Region Sampling) 技术（具体实现为 Hit-and-Run MCMC 算法）。此步骤并非仅仅为了获取离散点来“勾勒边界”，而是为了在高维约束下高效地近似粉丝投票的 后验分布 (Posterior Distribution)，从而量化解的不确定性，为后续推断提供高质量的解空间支撑。
第二步：在此基础上，我们将“寻找最优解”的目标精确定义为寻找 最大后验概率路径 (Maximum A Posteriori, MAP Path)。我们建立 隐马尔可夫模型 (HMM) 作为动态演化框架，以采样得到的粒子集合为状态空间，将“平滑性”量化为转移概率，将“淘汰结果”量化为观测似然。通过 序列推断算法 (如 Viterbi 或 SMC 平滑)，我们在定义域中锁定了一条同时最大化 时间连续性（平滑/不突兀）与 观测一致性（符合历史淘汰）的轨迹。这条轨迹即为统计学意义上可能性最大的粉丝投票演变过程。

其中，贝叶斯逆问题和最大后验概率路径是我们宏观上要实现的目标，然后可行域采样和HMM是具体工程实践的方法论。
1. **heatmap**是用来视化第一大问第二小问的。里面含有图片和数据，可以放在附录或正文中
2. **fan_vote**是用来可视化第一大问的。
3. **fan_vote_estimation_v2.py** 是第一大问模型。



## 第二问建模思路：
问题1:是不是需要一个表来将fan_vote_estimation_v2.py计算得来的每个season每个week每个person的share保存下来，就像C_data里是评委票数，是不是我们需要再建一个表格来记录share。

思路：
1. 我们先获得两个极端：完全按照评委分数的排序结果、完全按照fan share的排序结果。然后获得两个规则排序结果：rank排序结果、percent排序结果。
2. 定义排序之间的距离(度量排序差异)。 
3. 计算两规则之间的距离也就是差异度。
4. 偏向fan的程度也就是FFI。FFI(method)>0是更接近fan,FFI(method)<0是更接近judge,谁的FFI更大就更接近fan。

### 执行思路：
1. 对每个赛季、每一周（在当周 active 选手集合上），构造四种排序：
   * $R_{\text{judge}}$：只用评委分数（或评委 percent）得到的排序
   * $R_{\text{fan}}$：只用估计的 fan share 得到的排序
   * $R_{\text{rank}}$：按 rank 合并规则得到的排序/淘汰
   * $R_{\text{percent}}$：按 percent 合并规则得到的排序/淘汰

2. 选定排序距离 $D(\cdot, \cdot)$ （并规定 tie 处理）：
    * 距离用kendall作为报告里的主要距离函数，footrule作为补充可视化和解释。
    * 采用 统计学/竞赛常规的“平均名次（mid-rank）” 来处理 tie，并在需要淘汰预测时用“并列最小集合”。
3. 计算：
   * **差异度**：$D(R_{\text{rank}}, R_{\text{percent}})$ （以及淘汰者是否一致）
   * **偏向 fan 的程度（FFI）** ： 
     $$\text{FFI}(m) = D(R_m, R_{\text{judge}}) - D(R_m, R_{\text{fan}}), \quad m \in \{ \text{rank}, \text{percent} \}$$
     * $\text{FFI}(m) > 0$：更接近 fan （更“favor fans”）
     * $\text{FFI}(m) < 0$：更接近 judge （更“favor judges”）
     * 比较 $\text{FFI}(\text{rank})$ vs $\text{FFI}(\text{percent})$，谁更大谁更偏向 fan。


### ✅ 执行结果：

**1. 数据准备：Fan Vote Shares 数据表**
- **文件**：`fan_vote_shares.csv`（2779条记录）
- **结构**：`season, week, celebrity_name, fan_vote_share, method`
- **说明**：类似于原始数据中的评委分数表，记录每个选手每周的粉丝投票份额（归一化，每周求和为1）
- **作用**：这是第一问模型的核心输出，为所有后续分析（第二问、第三问等）提供基础数据

**2. 分析实现：投票方法比较**
- **程序**：`task2_voting_method_comparison.py`
- **运行时间**：约62秒
- **参数设置**：快速参数（n_samples=800/500, n_states=60/40，足够用于方法比较）
- **分析范围**：34个赛季，264周，涵盖所有有效淘汰周次

**3. 技术实现细节**
- **排序距离**：Kendall-$\tau_b$ correlation（scipy.stats.kendalltau）
- **Tie 处理**：mid-rank（并列者取平均名次，如并列第2、3名均记为2.5）
- **淘汰判定**：并列最小集合（若多人同分最低，实际淘汰者在集合内即算一致）
- **FFI 定义**：$\text{FFI}(m) = D(R_m, R_{\text{judge}}) - D(R_m, R_{\text{fan}})$

**4. 生成的可视化图表**（保存在 `task2_figures/`，共5张 O 奖级别图表）

| 图表文件 | 类型 | 展示内容 | 关键发现 |
|---------|------|----------|----------|
| `Task2_1_FFI_boxplot.png` | 箱线图 | 两种方法的 FFI 分布对比 | PERCENT 中位数 ~+0.09，RANK 接近 0 |
| `Task2_1_disagreement.png` | 双子图 | (a)Kendall距离分布 (b)淘汰一致率 | 平均距离 0.089，一致率 75% |
| `Task2_1_FFI_scatter.png` | 散点图 | FFI(RANK) vs FFI(PERCENT) 逐周对比 | 大部分点在 y=x 上方 |
| `Task2_1_FFI_heatmap.png` | 热力图 | FFI 跨 Week×Season 分布 | PERCENT 绿色（正FFI）区域更广 |
| `Task2_1_season_summary.png` | 4子图综合 | (a)FFI趋势 (b)距离 (c)一致率 (d)FFI分布 | 全面展示时间趋势和统计分布 |

 📊 核心数据发现

**方法差异度（Method Disagreement）**
- **淘汰一致率**：75.0%（两种方法在 25% 的周次会产生不同的淘汰者）
- **平均 Kendall 距离**：0.0887（0=完全一致，1=完全相反）
- **解释**：两种方法虽然大部分时候结果相同，但在约1/4的情况下会导致不同的淘汰结果

**Fan-Favor 偏向性（Fan-Favor Index）**
- **RANK 方法**：Mean FFI = **-0.0081**（略偏向 judge，几乎中性）
- **PERCENT 方法**：Mean FFI = **+0.0933**（明显偏向 fan）
- **FFI 差异**：+0.1014（PERCENT 比 RANK 显著更 favor fans）

**结论**：
- **PERCENT 方法显著更偏向 fan votes**，其产生的排序结构更接近"纯粉丝投票"的结果
- **RANK 方法相对更平衡**（FFI 接近 0），对 judge 和 fan 的权重较为均衡
- 这解释了为什么 Season 2（Jerry Rice）和 Season 27（Bobby Bones）的"争议"会发生在不同规则下

