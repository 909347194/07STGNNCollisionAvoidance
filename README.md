# ST-GNN 引导 + 确定性安全认证的高密度无人机航迹冲突检测加速

针对高密度无人机场景下潜在冲突航迹对庞大、连续 CPA 检测开销高的问题，
构建**确定性组级隔离为主干、ST-GNN 辅助优化、几何验证器保障安全**的
分层冲突检测框架。

> 核心定位：GNN 不判断航迹是否碰撞，而是通过学习引导的局部分组优化，
> 提高确定性安全证书的有效覆盖率，减少昂贵的精确冲突检测开销。

方法接口、证书数学定义、GNN 特征 / 输出与实验指标详见
[`docs/method_spec.md`](docs/method_spec.md)。

## 框架分层

1. **确定性检索与分组**：时空索引（暴力 / 网格 / R-tree）召回候选对，
   连通域分组，时空分离条件剔除可确定无冲突的组；
2. **ST-GNN 困难核优化**：确定性几何规则完成组内细分，GNN 仅输出子组
   认证优先级排序，不直接控制合并 / 拆分；
3. **分层几何验证**：时间分离证书、包围盒分离证书、相对运动距离下界证书
   逐级认证；未豁免航迹对执行连续 CPA，保证零漏检。

## 目录结构

```
├── pyproject.toml              # uv 项目与依赖声明（Python 3.12）
├── docs/method_spec.md         # 方法规范（接口 / 证书 / GNN / 实验指标）
├── scripts/
│   ├── build_dataset.py        # 数据集构建入口（转发 stgnn-dataset）
│   ├── run_experiment.py       # 评测入口（转发 stgnn-run）
│   └── run_all.py              # 一键：数据集 → 训练 → 评测
├── src/stgnn_conflict/
│   ├── config.py               # 阈值、容差、GNN 超参配置
│   ├── core/                   # 域模型：航迹、航段、包围盒
│   ├── deterministic/          # 索引、分组、分层安全证书
│   ├── gnn/                    # 图构建、ST-GNN 模型、分组引导
│   ├── verifier/               # 连续 CPA、分层几何验证器
│   ├── pipeline/               # 端到端流水线编排
│   ├── experiments/            # 场景生成、基线、指标、扫描运行器、汇总、出图
│   ├── dataset/                # 边界、运动、场景、标定、标签、划分、统计
│   ├── training/               # 子组优先级回归训练与加载
│   └── utils/                  # 几何与运动学工具
├── tests/                      # pytest 单元测试
├── models/                     # 训练产物（权重 + 训练历史，运行时生成）
├── data/raw/                   # 边界 GeoJSON 等原始输入
├── data/dataset/               # 离线数据集产物（manifest / 决策点表）
└── data/results/               # 实验产物（CSV / 汇总表 / 图，运行时生成）
```

## 环境准备

### 1. 安装 uv

**Linux / macOS（bash/zsh）**：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

**Windows（PowerShell）**：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

项目要求 Python `>=3.12,<3.13`（已在 `.python-version` 钉住 3.12），
`uv sync` 会自动拉取兼容解释器，无需手动安装 Python。

### 2. 同步依赖

```bash
# 核心依赖（numpy / scipy / pandas），纯确定性管线即可运行
uv sync

# 训练 / 评测完整版（服务器推荐一次装齐）
uv sync --extra torch --extra plots --extra dev
```

| extra | 作用 | 不装的后果 |
| --- | --- | --- |
| `torch` | 子组优先级回归器训练与推理 | 引导层回退几何兜底分数，无法加载训练权重 |
| `plots` | matplotlib 出图 | 表格产物不受影响，仅跳过 PNG 生成 |
| `dev` | pytest / ruff | 无法跑测试与 lint |

## 快速开始

三个命令行入口（等价于 `scripts/` 下的转发脚本）：

| 步骤 | 命令 | 产物 |
| --- | --- | --- |
| 1. 数据集 | `stgnn-dataset` | `data/dataset/{manifest,decision_points}.csv` |
| 2. 训练 | `stgnn-train` | `models/{priority_regressor.pt,train_history.json}` |
| 3. 评测 | `stgnn-run` | `data/results/`（CSV、汇总表、图） |

```bash
# 1) 构建离线数据集（场景级划分 + 密度 / 速度 OOD 标记 + 零漏检断言）
uv run stgnn-dataset \
  --densities 20,40,60,80 --seeds 1,2,3,4,5,6,7,8 \
  --injection-ratio 0.05 \
  --ood-densities 10,100 --ood-speed-ranges "5,10;28,35" \
  --density-train 20,80 --speed-train 10,25 \
  --out data/dataset

# 2) 训练子组优先级回归器（特征按训练集 z-score 标准化，统计量随 checkpoint 落盘）
uv run stgnn-train \
  --dataset-dir data/dataset --models-dir models \
  --epochs 150 --batch-size 64 --lr 1e-3 \
  --weight-decay 1e-4 --patience 15 --seed 42 \
  --device cuda:0

# 3) 评测：五方法对比 + 训练模型推理（自动生成汇总表与图）
uv run stgnn-run \
  --densities 20,40,60,80 --seeds 11,12,13,14,15 \
  --priority-model models/priority_regressor.pt \
  --ood-densities 10,100 --ood-speed-ranges "5,10;28,35" \
  --density-train 20,80 --speed-train 10,25 \
  --out data/results/benchmark.csv
```

> **注意**：`--ood-speed-ranges` 的多个区间用分号分隔，shell 下必须加引号
> （`"5,10;28,35"`），否则分号会被当成命令分隔符。

## 一键全流程（服务器推荐）

`scripts/run_all.py` 串联数据集 → 训练 → 评测，任一步失败即停止：

```bash
uv run python scripts/run_all.py \
  --densities 20,40,60,80 --seeds 1,2,3,4,5,6,7,8 \
  --injection-ratio 0.05 \
  --ood-densities 10,100 --ood-speed-ranges "5,10;28,35" \
  --speed-train 10,25 \
  --epochs 150 --batch-size 64 --weight-decay 1e-4 --patience 15 --seed 42 \
  --eval-seeds 11,12,13,14,15 \
  --device cuda:0
```

常用开关：

| 参数 | 说明 |
| --- | --- |
| `--skip-dataset` | 复用已有数据集，跳过构建 |
| `--skip-train` | 复用已有模型，跳过训练（只重跑评测） |
| `--eval-densities` | 评测密度档（默认 `20,40,60,80`） |
| `--device` | `cuda:0` / `cpu`；留空自动选择 |

纯 CPU 服务器删掉 `--device cuda:0` 即可。

## 实验产物目录

```
data/results/
├── benchmark.csv                        # 逐场景逐方法原始结果（含配置元数据与 OOD 标记）
├── tables/
│   ├── benchmark_summary.csv / .md      # 方法级均值 [95% bootstrap CI]
│   ├── benchmark_summary_by_kind.csv     # 按场景类型拆分汇总
│   └── benchmark_ablation.csv           # H1/H2 配对差值与 Wilcoxon 显著性
└── figures/                              # 密度折线、消融条形、困难核规模、OOD 对比、训练曲线
```

跑完核对：`benchmark_summary.csv` 中 `miss_rate_worst=0`、`recall_worst=1.0`。

## 主要实验方法

| 方法 | 说明 |
| --- | --- |
| `rtree_index` / `grid_index` | 索引召回后直接 CPA，无分组与证书 |
| `bbox_only` | 分组 + 一级时间 / 包围盒证书（关闭运动下界证书） |
| `det_group` | 纯确定性分层认证（含相对运动下界证书） |
| `gnn_guided` | ST-GNN 优先级引导完整流水线 |

消融对照：运动证书贡献 = `det_group` − `bbox_only`（H1）；
GNN 贡献 = `gnn_guided` − `det_group`（H2）。

> 需注意：运动证书是否显著优于包围盒、GNN 是否进一步降低端到端耗时，
> 均为待实验验证的研究假设，而非预设结论。

## 训练与数据口径

### 训练超参（`stgnn-train`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--hidden-dim` | 64 | 隐层宽度 |
| `--num-layers` | 3 | 全连接层数 |
| `--lr` | 1e-3 | AdamW 学习率 |
| `--weight-decay` | 1e-4 | 权重衰减 |
| `--epochs` | 100 | 最大轮数（一键脚本默认 150） |
| `--batch-size` | 64 | 批大小 |
| `--patience` | 15 | 早停轮数（按场景内 Spearman） |
| `--seed` | 42 | 训练 / 初始化随机种子 |
| `--val-ratio` | 0.2 | 从训练场景切出的验证集比例 |

- 监督信号：子组认证净节省 `benefit`（排序回归目标）；
- 验证指标：场景内 Spearman 秩相关 + MAE；
- 特征标准化统计量仅由训练集估计，随 checkpoint 落盘，推理 / 评测复用；
- 训练 / 验证 / 测试均按**场景**划分，禁止按决策点混洗跨场景泄漏。

### 数据集与 OOD

- 场景类型：`delivery_grid`（城市配送网格交叉 + 汇聚）、`vertiport`（起降点汇聚）；
- 运动模型：分段匀速直线 + 风场扰动 + 定位 / 时间戳小抖动；
- 冲突注入比例由 `--injection-ratio` 控制，注入对在元数据中单独标记；
- **OOD 切片**：`--ood-densities`（区间外密度档）、`--ood-speed-ranges`
  （分号分隔的区间外速度区间）；所有 OOD 场景强制归入 `test`，不参与训练与早停；
- 零漏检硬断言：被证书豁免的组必须附有通过的安全证书，否则断言失败。

## 测试与开发

```bash
uv run pytest          # 单元测试
uv run ruff check src tests scripts
```

## 已知简化与边界

- 当前图节点为航迹级（航段特征聚合到航迹）；段级时空图为后续扩展；
- 训练侧为子组优先级 MLP 回归头；完整 STGNN 边亲和度模型保留接口，尚未训练；
- 未接入真实轨迹；标定置信区间为占位，待 OpenSky / UTM 试飞数据实测替换；
- 自然 / 注入场景的评测期分列报告、k≤6 的 oracle 最优顺序为后续项；
- 详细假设、证书数学定义与指标口径见 `docs/method_spec.md`。
