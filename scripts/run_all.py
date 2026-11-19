"""一键启动：数据集构建 → 训练 → 评测。

用法（本地或服务器）：
```
uv run python scripts/run_all.py                     # 默认全流程
uv run python scripts/run_all.py --skip-dataset      # 复用已有数据集
uv run python scripts/run_all.py --epochs 200        # 加大训练轮数
uv run python scripts/run_all.py --device cuda:0     # 指定 GPU
```

产物：
- ``data/dataset/{manifest,decision_points}.csv``
- ``models/{priority_regressor.pt,train_history.json}``
- ``data/results/benchmark.csv`` 与方法级汇总表
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV_BIN = shutil.which("uv") or "uv"


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        sys.exit(f"步骤失败（退出码 {result.returncode}）: {' '.join(str(c) for c in cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集→训练→评测 一键流程")
    parser.add_argument("--densities", default="20,40,60,80", help="数据集航迹数量")
    parser.add_argument("--seeds", default="1,2,3,4,5", help="数据集随机种子")
    parser.add_argument("--injection-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu，默认自动")
    parser.add_argument("--skip-dataset", action="store_true", help="跳过数据集构建")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练（复用已有模型）")
    parser.add_argument("--eval-seeds", default="11,12,13", help="评测场景种子（与训练种子隔离）")
    parser.add_argument("--eval-densities", default="20,40,60,80", help="评测密度")
    parser.add_argument(
        "--speed-train", default="10,25", help="训练速度区间 lo,hi；数据集与评测共用同一 OOD 口径"
    )
    parser.add_argument(
        "--ood-densities", default="", help="区间外密度档（逗号分隔），数据集与评测均生成"
    )
    parser.add_argument(
        "--ood-speed-ranges", default="", help="区间外速度区间（分号分隔 lo,hi），数据集与评测均生成"
    )
    args = parser.parse_args()

    uv = [UV_BIN, "run", "--directory", str(ROOT)]

    # 1. 数据集
    if not args.skip_dataset:
        dataset_cmd = uv + [
            "stgnn-dataset",
            "--densities", args.densities,
            "--seeds", args.seeds,
            "--injection-ratio", str(args.injection_ratio),
            "--density-train", args.densities.split(",")[0] + "," + args.densities.split(",")[-1],
            "--speed-train", args.speed_train,
            "--out", "data/dataset",
        ]
        if args.ood_densities:
            dataset_cmd += ["--ood-densities", args.ood_densities]
        if args.ood_speed_ranges:
            dataset_cmd += ["--ood-speed-ranges", args.ood_speed_ranges]
        _run(dataset_cmd)

    # 2. 训练
    if not args.skip_train:
        train_cmd = uv + [
            "stgnn-train",
            "--dataset-dir", "data/dataset",
            "--models-dir", "models",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--weight-decay", str(args.weight_decay),
            "--patience", str(args.patience),
            "--seed", str(args.seed),
        ]
        if args.device:
            train_cmd += ["--device", args.device]
        _run(train_cmd)

    # 3. 评测（独立种子，加载训练好的优先级模型；汇总表 + 出图）
    density_train = args.densities.split(",")[0] + "," + args.densities.split(",")[-1]
    eval_cmd = uv + [
        "stgnn-run",
        "--densities", args.eval_densities,
        "--seeds", args.eval_seeds,
        "--priority-model", "models/priority_regressor.pt",
        "--density-train", density_train,
        "--speed-train", args.speed_train,
        "--out", "data/results/benchmark.csv",
        "--tables-dir", "data/results/tables",
        "--figures-dir", "data/results/figures",
    ]
    if args.ood_densities:
        eval_cmd += ["--ood-densities", args.ood_densities]
    if args.ood_speed_ranges:
        eval_cmd += ["--ood-speed-ranges", args.ood_speed_ranges]
    _run(eval_cmd)

    print("\n全流程完成：数据集 → 训练 → 评测。")
    print("  数据集: data/dataset/")
    print("  模型:   models/priority_regressor.pt")
    print("  原始结果: data/results/benchmark.csv")
    print("  汇总表: data/results/tables/")
    print("  图:     data/results/figures/")


if __name__ == "__main__":
    main()
