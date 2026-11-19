"""训练命令行入口。

用法示例
--------
``stgnn-train --dataset-dir data/dataset --models-dir models``

流程：读取 manifest / decision_points → 按场景级 split 重建子组样本
→ 训练优先级回归器（早停按场景内 Spearman）→ 落盘模型与训练历史。

划分纪律
--------
- 训练 / 验证 / 测试均按**场景**划分，禁止按决策点混洗跨场景泄漏；
- 验证集从训练场景中再切一份（``--val-ratio``），仅用于早停模型选择；
- 测试集（manifest 中的 test / OOD 场景）不参与训练与早停，仅做最终报告。
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.dataset.boundary import Boundary
from stgnn_conflict.dataset.injection import inject_conflicts
from stgnn_conflict.dataset.samples import build_subgroup_samples
from stgnn_conflict.dataset.scenarios import ScenarioParams, generate_scenario
from stgnn_conflict.training.train import (
    TrainConfig,
    _spearman_by_group,
    train_priority_regressor,
)


def _rebuild_samples(
    manifest_rows: list[dict],
    boundary: Boundary,
    config: DetectionConfig,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """按 manifest 重建子组样本（与构建数据集同种子，保证可复现）。"""
    feats: list[np.ndarray] = []
    labels: list[float] = []
    sids: list[str] = []
    total = len(manifest_rows)
    for idx, row in enumerate(manifest_rows, start=1):
        if verbose:
            print(f"重建场景 {idx}/{total}: {row['scenario_id']}")
        params = ScenarioParams(
            n_uavs=int(row["n_uavs"]),
            duration=float(row["duration"]),
            speed_range=(float(row["speed_lo"]), float(row["speed_hi"])),
        )
        trajectories = generate_scenario(row["kind"], boundary, params, seed=int(row["seed"]))
        # 复放离线数据集时的冲突注入（同种子、同记录比例），保证重建轨迹
        # 与生成决策点标签时的轨迹一致；自然场景 ratio=0 直接跳过。
        ratio = float(row.get("injection_ratio", 0.0) or 0.0)
        if ratio > 0.0:
            rng = np.random.default_rng(int(row["seed"]))
            trajectories, _, _ = inject_conflicts(trajectories, ratio, rng)
        samples = build_subgroup_samples(trajectories, config, scenario_id=row["scenario_id"])
        for s in samples:
            feats.append(s.features)
            labels.append(s.benefit)
            sids.append(s.scenario_id)
    return (
        np.array(feats, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        sids,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ST-GNN 子组优先级训练")
    parser.add_argument("--dataset-dir", default="data/dataset", help="数据集目录")
    parser.add_argument("--models-dir", default="models", help="模型输出目录")
    parser.add_argument("--boundary", default="data/raw/region.geojson", help="边界 GeoJSON 路径")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15, help="早停轮数（按验证 Spearman）")
    parser.add_argument("--seed", type=int, default=42, help="训练 / 初始化随机种子")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="从训练场景切出的验证集比例")
    parser.add_argument("--val-seed", type=int, default=42, help="验证集划分随机种子")
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    args = parser.parse_args(argv)

    import pandas as pd

    manifest = pd.read_csv(f"{args.dataset_dir}/manifest.csv", encoding="utf-8-sig")
    boundary = Boundary.from_geojson(args.boundary)
    config = DetectionConfig()

    rows = manifest.to_dict("records")
    x, y, sids = _rebuild_samples(rows, boundary, config)
    sids_arr = np.array(sids)
    split_map = dict(zip(manifest["scenario_id"], manifest["split"]))

    train_sids = sorted({s for s in sids if split_map[s] == "train"})
    test_sids = sorted({s for s in sids if split_map[s] != "train"})

    # 验证集从训练场景中再切一份；测试集不参与模型选择
    rng = np.random.default_rng(args.val_seed)
    train_shuffled = np.array(train_sids, dtype=object)
    rng.shuffle(train_shuffled)
    n_val = max(1, round(len(train_shuffled) * args.val_ratio))
    val_sids = set(train_shuffled[:n_val].tolist())
    train_sids = [s for s in train_shuffled[n_val:].tolist()]

    train_set, val_set, test_set = set(train_sids), val_sids, set(test_sids)
    train_mask = np.array([s in train_set for s in sids])
    val_mask = np.array([s in val_set for s in sids])
    test_mask = np.array([s in test_set for s in sids])

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]
    sid_train = sids_arr[train_mask].tolist()
    sid_val = sids_arr[val_mask].tolist()
    sid_test = sids_arr[test_mask].tolist()

    print(
        f"训练样本: {len(y_train)}（场景 {len(train_set)}）；"
        f"验证样本: {len(y_val)}（场景 {len(val_set)}）；"
        f"测试样本: {len(y_test)}（场景 {len(test_set)}）"
    )

    tcfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    if args.device:
        tcfg.device = args.device

    in_dim = x_train.shape[1] if len(x_train) else x.shape[1]
    print("\n=== 训练配置 ===")
    print(f"device: {torch.device(tcfg.device)}")
    print(f"模型: MLP {in_dim} → {tcfg.hidden_dim} × {tcfg.num_layers} → 1")
    print(
        f"优化器: AdamW lr={tcfg.lr} weight_decay={tcfg.weight_decay}；"
        f"batch={tcfg.batch_size} epochs={tcfg.epochs} "
        f"patience={tcfg.patience} seed={tcfg.seed}"
    )

    model, result = train_priority_regressor(
        x_train, y_train, sid_train, x_val, y_val, sid_val,
        config=tcfg, out_dir=args.models_dir,
    )
    print(
        f"\n最优场景内验证 Spearman: {result['best_val_spearman']:.4f}；"
        f"最优 epoch: {result['best_epoch']}；训练耗时: {result['total_sec']:.1f}s"
    )

    # 测试集最终报告（不参与早停 / 选择）
    if len(y_test) > 0:
        with torch.no_grad():
            test_pred = model.infer(torch.tensor(x_test, dtype=torch.float32))
        test_spearman = _spearman_by_group(test_pred, y_test, sid_test)
        test_mae = float(np.mean(np.abs(test_pred - y_test)))
        print(f"测试集 Spearman: {test_spearman:.4f}；测试集 MAE: {test_mae:.4f}")

    print(f"模型与训练历史已保存: {args.models_dir}")


if __name__ == "__main__":
    main()
