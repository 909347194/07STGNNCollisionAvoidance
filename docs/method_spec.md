# 方法规范说明

本文档明确四件事：方法模块的具体接口、分层证书的数学定义、
ST-GNN 的输入特征与输出形式、实验设计与评价指标。代码实现与本文一一对应。

## 1. 方法模块接口

| 层 | 模块 | 主要接口 | 输入 | 输出 | 职责 |
| --- | --- | --- | --- | --- | --- |
| 域模型 | `core.trajectory` | `Trajectory` / `Segment` / `State` | 状态点序列 | 航迹对象（航段、包围盒、时间区间） | 分段线性运动模型与插值 |
| 域模型 | `core.bbox` | `AABB` | 点集 / 最小最大角点 | 相交、距离、交并比、相交体积 | 空间关系计算 |
| 确定性 | `deterministic.indexes` | `build_index(kind, cell, margin)` | `list[Trajectory]` | `list[(tid_a, tid_b)]` | 暴力 / 网格 / STR R-tree 候选召回 |
| 确定性 | `deterministic.grouping` | `group_by_components(n, edges)` | 候选边表 | `list[list[tid]]` | 连通域分组 |
| 确定性 | `deterministic.certificates` | `certify_pair` / `certify_group` | 航迹对 / 航迹组 | `CertResult` / `GroupCertResult` | 分层安全证书判定 |
| GNN | `gnn.graph` | `build_graph(trajs, pairs, config)` | 困难核候选边 | `HeteroGraph`（节点 / 边特征） | 困难核时空图构建 |
| GNN | `gnn.model` | `STGNN.forward / infer` | 图张量 | 边亲和度、节点优先级 | 边条件消息传递推理 |
| GNN | `gnn.guide` | `guide_groups(groups, scores, config, traj_map, candidate_set)` | 确定性分组 + 分数 | `GuideResult`（细化子组、优先级、规则切断边） | 确定性规则细分 + GNN 优先级排序 |
| 验证器 | `verifier.cpa` | `continuous_cpa(a, b, radius, eps)` | 两航迹 | `CPAResult`（是否冲突、最小距离、时刻） | 连续最近接近点精确检测 |
| 验证器 | `verifier.validator` | `validate_groups(groups, traj_map, config)` | 细化航迹组 | `VerifyResult`（冲突集、CPA 次数、认证统计） | 分层认证 + 剩余对 CPA |
| 流水线 | `pipeline.detector` | `ConflictDetector.detect(trajs)` | 航迹列表 | `DetectionOutcome` | 端到端编排与指标汇总 |
| 实验 | `experiments.runner` | `run_density_sweep(...)` | 密度 / 种子 / 方法列表 | `pd.DataFrame` | 密度扫描、重复实验、结果落盘 |
| 数据集 | `dataset.boundary` | `Boundary.from_geojson(path)` | GeoJSON 路径 | 边界对象（局部米制投影 / 包含判定） | 空间范围圈定 |
| 数据集 | `dataset.scenarios` | `generate_scenario(kind, boundary, params, seed)` | 场景规格 + 边界 | `list[Trajectory]` | 两类高冲突场景生成 |
| 数据集 | `dataset.labels` | `extract_decision_points(trajs, config, scenario_id)` | 确定性管线输入 | `list[DecisionPoint]` | 决策点级监督标签提取 |
| 数据集 | `dataset.split` | `split_by_scenario` / `quota_balance_by_level` | 场景 ID / 决策点表 | 训练 / 测试 / OOD 子集 | 场景级划分与配额均衡 |
| 数据集 | `dataset.stats` | `bootstrap_ci` / `assert_zero_miss` | 场景指标 / 冲突集合 | 置信区间 / 断言 | 统计纪律与零漏检硬断言 |

约定：航迹编号 `tid ∈ {0, …, n-1}`；航迹对统一存为 `(min, max)`；
所有时间为闭区间；数值比较容差见 `DetectionConfig.cert`。

## 2. 分层证书的数学定义

### 2.1 符号与运动模型

航迹 $i$ 由状态点序列构成，相邻点之间为匀速直线运动；第 $k$ 段记为

$$
S_{ik} = (p_{ik}^{0}, p_{ik}^{1}, t_{ik}^{0}, t_{ik}^{1}), \quad v_{ik} = \frac{p_{ik}^{1}-p_{ik}^{0}}{t_{ik}^{1}-t_{ik}^{0}}
$$

整条航迹的时间区间 $I_i=[t_i^{0}, t_i^{1}]$，空间包围盒 $B_i=[\underline{x}_i,\overline{x}_i]\subset\mathbb{R}^3$。
安全半径为 $R$，时间 / 空间容差为 $\varepsilon_t,\varepsilon_s$。

### 2.2 证书 C1：时间分离

若两航迹时间区间不重叠，则不可能发生冲突：

$$
I_i \cap I_j = \varnothing
\;\Longleftrightarrow\;
t_i^{1}+\varepsilon_t < t_j^{0} \;\lor\; t_j^{1}+\varepsilon_t < t_i^{0}
\;\Longrightarrow\; \text{SAFE}
$$

### 2.3 证书 C2：空间包围盒分离

两包围盒的最小欧氏距离为各轴间隙的范数：

$$
d_{\mathrm{box}}(B_i,B_j)
=
\left\|
\max\!\left(0,\ \underline{x}_j-\overline{x}_i,\ \underline{x}_i-\overline{x}_j\right)
\right\|_2
$$

若该距离大于安全半径，则无论时间如何重叠都不可能冲突：

$$
d_{\mathrm{box}}(B_i,B_j) > R + \varepsilon_s
\;\Longrightarrow\; \text{SAFE}
$$

### 2.4 证书 C3：相对运动距离下界

对重叠航段对 $(S_{ik}, S_{j\ell})$，相对运动 $r(t)=r_0+v_r t$，其中
$r_0=p_{ik}^{0}-p_{j\ell}^{0}$，$v_r=v_{ik}-v_{j\ell}$。距离平方为关于 $t$ 的二次函数：

$$
d^2(t)=a t^2+b t+c,\quad
a=v_r^\top v_r,\ b=2r_0^\top v_r,\ c=r_0^\top r_0
$$

在重叠区间 $[\tau_0,\tau_1]=I_{ik}\cap I_{j\ell}$ 上取最小值（驻点或端点），
得到该段对的最近接近距离 $d_{k\ell}^{*}$。逐段对取下界：

$$
d_{\min}(i,j)=\min_{k,\ell} d_{k\ell}^{*}, \qquad
d_{\min}(i,j) \ge R - \varepsilon_s
\;\Longrightarrow\; \text{SAFE}
$$

### 2.5 组级证书与零漏检保证

组 $G$ 可整体豁免，当且仅当组内每一对航迹均通过至少一张证书：

$$
\operatorname{Cert}(G)=\bigwedge_{i<j,\ i,j\in G}
\bigvee_{c\in\{\mathrm{C1,C2,C3}\}} \operatorname{Pass}_c(i,j)
$$

未通过组级证书的航迹对全部进入连续 CPA；在分段线性运动模型与上述数值精度
假设下，流水线不存在“未认证且未做 CPA”的候选对，因此漏检率为 0。

## 3. ST-GNN 输入特征与输出形式

### 3.1 节点特征（航迹为节点，共 6 维）

| 维度 | 符号 | 定义 | 归一化 |
| --- | --- | --- | --- |
| 1 | $\bar v_i$ | 平均速度 $\|v\|$ 均值 | /30 m/s |
| 2 | $\Delta t_i$ | 航迹总时长 | /60 s |
| 3 | $K_i$ | 航段数量 | /10 |
| 4 | $\ell_i$ | 包围盒对角线长度 | /100 m |
| 5 | $\bar z_i$ | 平均飞行高度 | /1000 m |
| 6 | $\rho_i$ | 时长占比 $\Delta t_i/(\Delta t_i+1)$ | 无量纲 |

### 3.2 边特征（潜在时空关联为边，共 7 维）

| 维度 | 符号 | 定义 | 归一化 |
| --- | --- | --- | --- |
| 1 | $d_c$ | 包围盒中心距离 | /100 m |
| 2 | $\tau$ | 时间重叠时长占比；不重叠时为负时间间隙 | /60 s |
| 3 | $\Delta v$ | 速度向量差模长 | /30 m/s |
| 4 | $\mathrm{IoU}$ | 包围盒交并比 | 无量纲 |
| 5 | $V_o$ | 相交体积 | /10⁶ m³ |
| 6 | $d_b$ | 包围盒最小间隙 | /100 m |
| 7 | $\Delta z$ | 平均高度差 | /100 m |

### 3.3 输出形式（不含碰撞判断）

- **边亲和度** $s_{ij}\in[0,1]$：sigmoid 输出，表征两航迹适合纳入同一可认证子组的程度；
- **子组优先级** $p_g\in[0,1]$：子组内部边亲和度的聚合（节点嵌入池化头输出同口径），
  表征该子组通过几何安全认证的潜力与处理收益；
- **不输出**碰撞 / 冲突标签，不从零生成全局分组，也不直接执行合并 / 拆分。

职责边界：
- **分组细分由确定性规则完成**：组内按预设几何阈值（航迹中心距离）切分为子组，
  规则不依赖神经网络；
- **GNN 只决定处理策略**：依据亲和度给出子组认证优先级，验证器按优先级降序认证；
- **安全判定完全由确定性证书负责**：GNN 分数不构成任何跳过检测的依据。

## 4. 实验设计与评价指标

### 4.1 实验设计

- **密度扫描**：航迹数量 $N\in\{20,40,60,80,\dots\}$，观察困难核规模、
  证书覆盖率与 CPA 调用比例随密度的变化；
- **重复实验**：每个密度 × 多种子重复，指标取均值（必要时附标准差）；
- **对比方法**：
  1. R-tree 索引 + CPA（`rtree_index`）；
  2. 时空网格索引 + CPA（`grid_index`）；
  3. 纯包围盒方法：分组 + 一级时间 / 包围盒证书（`bbox_only`，关闭运动下界证书）；
  4. 纯确定性分层认证：分组 + 三级分层证书（`det_group`，含相对运动下界）；
  5. ST-GNN 增强完整流水线（`gnn_guided`）；
- **消融设计**：
  * 相对运动证书独立贡献：`det_group` 对照 `bbox_only`（其余配置一致）；
  * GNN 独立贡献：`gnn_guided` 对照 `det_group`（其余配置一致）；
  * 索引类型影响：R-tree 与网格索引在同一流水线内替换；
- **真值基准**：全部航迹对暴力连续 CPA。

### 4.2 评价指标

| 指标 | 定义 |
| --- | --- |
| 端到端耗时 $T_{\mathrm{end}}$ | 单场景流水线总耗时（ms），含索引 / 分组 / 认证 / GNN / 验证分段统计 |
| 候选对数 $P_{\mathrm{cand}}$ | 索引召回的潜在关联航迹对数量 |
| CPA 调用次数 $N_{\mathrm{CPA}}$ | 进入连续 CPA 的航迹对数量 |
| 认证成功对数 $C_{\mathrm{cert}}$ | 通过证书豁免的航迹对数量 |
| 对级证书覆盖率 | $C_{\mathrm{cert}}/P_{\mathrm{cand}}$ |
| 组级认证覆盖率 | 认证通过组数 / 连通域分组总数 |
| 冲突对数 $N_{\mathrm{conf}}$ | 检测出的冲突航迹对数量 |
| 漏检率 | $\mathrm{FN}/\lvert\mathrm{Truth}\rvert$ |
| 精确率 / 召回率 | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$，$\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ |

核心验证目标：在漏检率为 0 的前提下，评估证书与 GNN 引导对组级认证覆盖率、
CPA 调用次数与端到端耗时的实际影响。

### 4.3 统计口径与结果产物

- **置信区间**：效率指标按场景重采样做 percentile bootstrap 95% CI；
  汇总表同时给出均值与 CI 上下界。
- **漏检口径**：`miss_rate` 除均值外报告 worst-case（最大值）与违反场景数；
  零漏检为硬约束，不以均值掩盖个别漏检。
- **配对消融**：H1（`det_group` − `bbox_only`）、H2（`gnn_guided` − `det_group`）
  按同一 `(seed, density, kind)` 场景配对，报告配对均值差与 Wilcoxon 符号秩
  p 值；样本不足或差值全为零时给出说明，不伪造显著性。
- **OOD 标记**：评测场景按训练密度 / 速度区间标记 `id / density_ood / speed_ood`；
  汇总可按 `kind` 拆分。
- **结果落盘**：原始逐场景结果 CSV（含安全半径、运动证书开关、模型路径、
  时间戳等元数据）；方法级汇总表（CSV + Markdown）、按类型拆分表、消融表；
  图件输出到 `figures/`（密度折线、消融条形、困难核规模、OOD 对比、训练曲线）。

## 5. 研究假设与实现边界

### 5.1 待验证假设（非既定结论）

以下两点是本文要通过理论分析与实验检验的**研究假设**，实现本身不预设其成立：

- **H1（运动证书增益）**：相较静态包围盒证书，相对运动下界证书能否利用航向与
  速度信息获得更紧致的安全界限、提高组级认证覆盖率；需由 `det_group` 与
  `bbox_only` 的对照实验检验。
- **H2（GNN 加速收益）**：ST-GNN 的优先级引导能否进一步降低端到端耗时或 CPA
  调用次数；需由 `gnn_guided` 与 `det_group` 的对照实验检验。无训练权重时，
  引导层回退几何兜底分数，此时结果不代表训练后模型的真实收益。

### 5.2 已知简化与后续扩展

- 当前图节点为**航迹级节点**（航段特征聚合到航迹）；研究表述中的“航迹段为节点”
  对应段级时空图，可作为后续扩展：节点细化为航段、边为重叠段对，亲和度再聚合回
  航迹组。分组与安全认证本身在航迹级完成，不受该简化影响。
- 场景生成采用随机航迹点 + 分段匀速直线运动；真实航迹的机动、风速偏差等
  未纳入当前运动模型。
- 训练侧为**子组优先级回归器**（MLP，输入子组聚合特征、回归认证净节省），
  完整 STGNN 图模型（`gnn.model.STGNN`）保留边亲和度接口供后续扩展。
- 输入特征按**训练集统计量做 z-score 标准化**，均值 / 标准差随 checkpoint 落盘，
  推理与评测复用同一组统计量；训练超参 `lr / weight_decay / patience / seed /
  hidden_dim / num_layers / batch_size / epochs` 均可经 `stgnn-train` CLI 调整。

## 6. 离线数据集构建

### 6.1 场景与运动模型

- **运动模型**：分段匀速直线 + 简单圆弧转弯段（多旋翼口径）；固定翼 Dubins
  曲线、复杂加减速为可选扩展，本版本不实现。
- **扰动**：仅保留常值风场（速度叠加）与定位 / 时间戳高斯抖动；改航、临时悬停
  等后期再加入。
- **场景类型**：
  * `delivery_grid`：城市配送网格——水平 / 垂直航线交叉，叠加汇聚到配送枢纽；
  * `vertiport`：垂直起降点附近双向汇聚。
- **边界**：GeoJSON `Polygon` / `MultiPolygon`（CRS84），等距圆柱投影到局部
  米制坐标系；航线端点在包围盒内缩区域内采样，避免抖动推出界外。

### 6.2 标定参数与敏感性分析

只标定对证书影响最大的 4 类参数：速度分布、最小安全间隔（安全半径）、
轨迹时长、空间密度。当前区间为粗标定占位值，待 OpenSky 公开数据或 UTM
试飞数据替换；`ci_95` 字段在拿到实测数据前不构成统计结论。

敏感性分析逐参数在低 / 中 / 高水平重复扫描，报告困难核规模（候选对数、
困难核组数、CPA 次数）的均值、最小值与最大值。

### 6.3 冲突注入与分层标签

- **冲突注入**：按候选对比例随机抽取航迹对并向彼此靠拢，注入比例与注入对
  在场景元数据中单独记录；注入场景与自然场景分开报告。
- **认证层级（辅助标签）**：`L1`（时间 / 包围盒分离通过）、`L2`（相对运动
  下界通过）、`CPA`（无法认证，需进入连续 CPA）。
- **几何构型**：`crossing`（交叉）、`overtaking`（追越）、`convergence`
  （汇聚）、`parallel`（平行，对照构型）。

### 6.4 决策点标签（样本单位为决策点，不是场景）

离线跑纯确定性管线（无 GNN）。困难核先按确定性几何规则细分，**每个细化
子组是一个决策点**——与推理时 GNN 实际排序的对象严格一致；被细分切断的
跨子组候选对（延迟对）单独记录，其兜底代价不计入任何子组。

对子组 $G$，记候选边集 $E_G$；$C_G\subseteq E_G$ 为通过证书被豁免的对。

- **主标签（排序监督信号）认证净节省**：

$$
\mathrm{benefit}(G)=|C_G|\,c_{\mathrm{CPA}}-|E_G|\,c_{\mathrm{cert}}
$$

其中 $c_{\mathrm{CPA}}=1.0$（单次 CPA 代价），$c_{\mathrm{cert}}=0.1$
（单次证书几何计算代价）。语义：相对"不做认证、候选对直接进 CPA"，
对该子组做证书认证的净节省——每豁免一对省一次 CPA，每检查一对付一次
几何计算。全豁免子组为正、全进 CPA 子组为负，跨子组可直接比较。
- **辅助标签**：子组层级取组内最差 pair 层级（CPA > L2 > L1）；
  几何构型取组内众数。
- **细分阈值**：组内候选对中心距离的分位数（默认 25 分位）自适应
  切分，避免固定米制阈值在大尺度边界下把组切成全孤立子组。
- **可选上界**：$k\le 6$ 的小子组可额外计算 oracle 最优处理顺序，
  当前版本保留字段、暂不计算。
- 报告时同时给出决策点数与场景数。

### 6.5 划分与 OOD

- **场景级划分**：训练 / 留出测试按场景 + 随机种子划分，同一$\mathrm{场景}$
  的所有决策点落在同一子集，禁止按航迹对划分。
- **验证集隔离**：训练场景再按固定种子切出一份验证集（默认 20%），仅用于
  早停与模型选择；留出测试集不参与训练与早停，只做最终报告。
- **配额均衡**：训练集按认证层级做有限配额重采样；测试集保持自然分布，
  不做重采样。
- **OOD 切片**：至少密度 OOD（航迹数落在训练密度区间外）与速度分布 OOD
  （场景平均速度落在训练速度区间外）两个维度。数据集 / 评测 CLI 通过
  `--ood-densities`（区间外密度档）与 `--ood-speed-ranges`（分号分隔的区间外
  速度区间）生成对应切片；**所有 OOD 场景强制归入 test**，不参与训练与早停。
- **真实轨迹**：若获取到真实轨迹，仅用于验证证书通过率与行为是否与仿真
  一致，不参与训练；当前未接入真实数据，OOD 以标定区间外仿真代替。

### 6.6 统计纪律

- 所有效率指标按场景做 percentile bootstrap 置信区间（重采样单位为场景）；
- 任何配置必须附带“漏检率 = 0”硬断言：被跳过（豁免）的组必须附有通过的
  安全证书，否则断言失败。
