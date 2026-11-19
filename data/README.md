# 数据目录说明

## 目录结构

```
data/
├── raw/                 # 原始输入（纳入版本管理）
│   └── region.geojson   # 场景边界（CRS84，Polygon / MultiPolygon）
├── dataset/             # 离线数据集产物（CSV 被 .gitignore 忽略，由 CLI 生成）
│   ├── manifest.csv         # 场景级元数据：类型、密度、速度、注入比例、划分、OOD 标记
│   └── decision_points.csv  # 决策点级样本：层级 / 几何构型 / 边际收益主标签
├── calibration/         # 标定与敏感性分析产物（由 --sensitivity-out 生成）
└── results/             # 实验扫描结果（CSV）
```

## 构建命令

```powershell
# 生成数据集（场景级 train/test 划分，密度 / 速度 OOD 标记，零漏检硬断言）
uv run stgnn-dataset --densities 20,40,60 --seeds 1,2,3 --injection-ratio 0.05

# 附带困难核规模敏感性分析
uv run stgnn-dataset --sensitivity-out data/calibration/sensitivity.csv
```

## 记录口径

- 样本单位为**决策点**（确定性细分后的子组级）；`manifest.csv` 统计场景数，`decision_points.csv` 统计决策点数，报告时两者并列给出。
- 主标签 `benefit` = 被证书豁免对数 × c_CPA − 检查对数 × c_cert，即认证相对"直接 CPA"的净节省；细分切断的跨子组延迟对单独计数（`deferred_pairs`）。
- 注入场景与自然场景通过 `injected` / `injection_ratio` 字段区分，分开报告。
- 训练集按认证层级有限配额均衡（`--train-level-quota`），测试集保持自然分布。
- 标定区间为占位粗标定，待 OpenSky 公开数据或 UTM 试飞数据替换；真实轨迹
  仅用于验证证书通过率与行为一致性，不参与训练。
