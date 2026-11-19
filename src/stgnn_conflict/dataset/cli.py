"""数据集构建命令行入口。

用法示例
--------
``stgnn-dataset --densities 20,40,60 --seeds 1,2,3 --injection-ratio 0.05``

输出到 ``data/dataset/``：
- ``manifest.csv``：场景级元数据；
- ``decision_points.csv``：决策点级监督样本。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.dataset.boundary import Boundary
from stgnn_conflict.dataset.builder import ScenarioSpec, assign_splits, build_scenario_records
from stgnn_conflict.dataset.calibration import sensitivity_sweep
from stgnn_conflict.dataset.split import quota_balance_by_level


def _parse_pair(text: str) -> tuple[float, float]:
    lo, hi = text.split(",")
    return float(lo), float(hi)


def _parse_intervals(text: str | None) -> list[tuple[float, float]]:
    """解析分号分隔的 ``lo,hi`` 区间列表，如 ``5,10;28,35``；空串返回 []。"""
    if not text:
        return []
    return [tuple(float(x) for x in pair.split(",")) for pair in text.split(";") if pair]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="构建 ST-GNN 冲突检测离线数据集")
    parser.add_argument("--boundary", default="data/raw/region.geojson", help="边界 GeoJSON 路径")
    parser.add_argument("--out", default="data/dataset", help="数据集输出目录")
    parser.add_argument("--kinds", default="delivery_grid,vertiport", help="场景类型，逗号分隔")
    parser.add_argument("--densities", default="20,40,60", help="航迹数量，逗号分隔")
    parser.add_argument("--seeds", default="1,2,3", help="随机种子，逗号分隔")
    parser.add_argument("--injection-ratio", type=float, default=0.05, help="冲突注入比例（0 表示纯自然场景）")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="场景级训练集比例")
    parser.add_argument("--density-train", default="20,40", help="训练密度区间 lo,hi（区间外标记密度 OOD）")
    parser.add_argument("--speed-train", default="10,25", help="训练速度区间 lo,hi（区间外标记速度 OOD）")
    parser.add_argument(
        "--ood-densities",
        default="",
        help="区间外密度档（逗号分隔），强制归入 test；如 10,100",
    )
    parser.add_argument(
        "--ood-speed-ranges",
        default="",
        help="区间外速度区间（分号分隔 lo,hi），强制归入 test；如 5,10;28,35",
    )
    parser.add_argument(
        "--train-level-quota",
        type=int,
        default=None,
        help="训练集每个认证层级的决策点配额；不设置则不做配额均衡",
    )
    parser.add_argument("--sensitivity-out", default=None, help="困难核规模敏感性分析 CSV 输出路径")
    args = parser.parse_args(argv)

    boundary = Boundary.from_geojson(args.boundary)
    config = DetectionConfig()

    kinds = [k for k in args.kinds.split(",") if k]
    densities = [int(x) for x in args.densities.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    ood_densities = [int(x) for x in args.ood_densities.split(",") if x]
    ood_speed_ranges = _parse_intervals(args.ood_speed_ranges)

    specs = [
        ScenarioSpec(kind=kind, n_uavs=n, seed=seed)
        for kind in kinds
        for n in densities
        for seed in seeds
    ]
    # 密度 OOD：训练密度区间外的航迹数量，仅作留出评测
    specs += [
        ScenarioSpec(kind=kind, n_uavs=n, seed=seed, tag=f"ood-d{n}")
        for kind in kinds
        for n in ood_densities
        for seed in seeds
    ]
    # 速度 OOD：训练速度区间外的速度区间，数量仍用分布内密度档
    for lo, hi in ood_speed_ranges:
        specs += [
            ScenarioSpec(
                kind=kind, n_uavs=n, seed=seed,
                speed_range=(lo, hi), tag=f"ood-v{lo}-{hi}",
            )
            for kind in kinds
            for n in densities
            for seed in seeds
        ]

    manifest, decision_points = build_scenario_records(
        specs, boundary, config, injection_ratio=args.injection_ratio
    )
    manifest, decision_points = assign_splits(
        manifest,
        decision_points,
        train_ratio=args.train_ratio,
        seed=config.seed,
        density_train=_parse_pair(args.density_train),
        speed_train=_parse_pair(args.speed_train),
    )

    # 训练集按层级有限配额均衡；测试集保持自然分布，不做重采样
    decision_points = decision_points.merge(
        manifest[["scenario_id", "injected"]], on="scenario_id", how="left"
    )
    if args.train_level_quota is not None:
        train_mask = decision_points["split"] == "train"
        train_dp = quota_balance_by_level(
            decision_points[train_mask], args.train_level_quota, seed=config.seed
        )
        decision_points = pd.concat([train_dp, decision_points[~train_mask]])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    decision_points.to_csv(out_dir / "decision_points.csv", index=False, encoding="utf-8-sig")

    _print_summary(manifest, decision_points)

    if args.sensitivity_out:
        sweep = sensitivity_sweep(boundary, _base_params(densities[0]), config)
        Path(args.sensitivity_out).parent.mkdir(parents=True, exist_ok=True)
        sweep.to_csv(args.sensitivity_out, index=False, encoding="utf-8-sig")
        print(f"\n敏感性分析已保存: {args.sensitivity_out}")

    print(f"\n数据集已保存: {out_dir}")


def _base_params(n_uavs: int):
    from stgnn_conflict.dataset.scenarios import ScenarioParams

    return ScenarioParams(n_uavs=n_uavs)


def _print_summary(manifest: pd.DataFrame, decision_points: pd.DataFrame) -> None:
    n_scenarios = len(manifest)
    n_points = len(decision_points)
    print("\n=== 数据集概览 ===")
    print(f"场景数: {n_scenarios}；决策点数: {n_points}")
    print("\n场景划分:")
    print(manifest["split"].value_counts().to_string())
    print("\n决策点认证层级分布:")
    print(decision_points["level"].value_counts().to_string())
    print("\n决策点几何构型分布:")
    print(decision_points["geometry"].value_counts().to_string())
    print("\n自然 / 注入场景:")
    print(manifest["injected"].value_counts().to_string())
    print("\n自然 / 注入决策点:")
    print(decision_points["injected"].value_counts().to_string())
    if "ood_density" in manifest:
        print(f"\n密度 OOD 场景数: {int(manifest['ood_density'].sum())}")
    if "ood_speed" in manifest:
        print(f"速度 OOD 场景数: {int(manifest['ood_speed'].sum())}")


if __name__ == "__main__":
    main()
