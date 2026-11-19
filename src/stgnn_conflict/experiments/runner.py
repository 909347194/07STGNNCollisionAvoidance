"""实验运行框架：密度扫描、重复随机种子、基线对比与结果汇总。

实验设计
--------
- 自变量：航迹密度（航迹数量）、检测方法、索引类型；
- 重复：每个密度 × 多种子重复实验后取均值；
- 对比方法：
  * ``rtree_index`` / ``grid_index``：索引-only，召回即 CPA；
  * ``bbox_only``：分组 + 一级时间 / 包围盒证书（关闭运动下界证书）；
  * ``det_group``：纯确定性分层认证（含相对运动下界证书）；
  * ``gnn_guided``：ST-GNN 引导完整流水线；
- 消融：运动证书贡献 = ``det_group`` − ``bbox_only``；GNN 贡献 = ``gnn_guided`` − ``det_group``；
- 评价指标：端到端耗时、候选对数、CPA 调用次数、认证对数 / 对级覆盖率、
  组级认证覆盖率、冲突对数、漏检率、精确率、召回率。
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.dataset.boundary import Boundary
from stgnn_conflict.dataset.scenarios import ScenarioParams, generate_scenario
from stgnn_conflict.experiments.baselines import (
    ground_truth,
    run_bbox_only,
    run_deterministic_group,
    run_gnn_guided,
    run_index_only,
)
from stgnn_conflict.experiments.metrics import evaluate
from stgnn_conflict.experiments.plots import plot_all
from stgnn_conflict.experiments.summary import export_tables

ALL_METHODS = [
    "rtree_index",
    "grid_index",
    "bbox_only",
    "det_group",
    "gnn_guided",
]


def _measure(
    method: str,
    trajectories: list,
    config: DetectionConfig,
    model: object | None = None,
    priority_model: object | None = None,
):
    if method == "rtree_index":
        return run_index_only(trajectories, config, "rtree")
    if method == "grid_index":
        return run_index_only(trajectories, config, "grid")
    if method == "bbox_only":
        return run_bbox_only(trajectories, config, "grid")
    if method == "det_group":
        return run_deterministic_group(trajectories, config, "grid")
    if method == "gnn_guided":
        return run_gnn_guided(trajectories, config, "grid", model, priority_model)
    raise ValueError(f"未知方法: {method}")


def _ood_tag(
    n_uavs: int,
    speed_range: tuple[float, float],
    density_train: tuple[float, float] | None,
    speed_train: tuple[float, float] | None,
) -> str:
    """标记场景是否落在训练区间外（密度 OOD / 速度 OOD / 内分布）。"""
    if density_train is not None and not (density_train[0] <= n_uavs <= density_train[1]):
        return "density_ood"
    if speed_train is not None and not (
        speed_train[0] <= speed_range[0] and speed_range[1] <= speed_train[1]
    ):
        return "speed_ood"
    return "id"


def run_scenario(
    n_uavs: int,
    seed: int,
    config: DetectionConfig,
    methods: list[str],
    model: object | None = None,
    priority_model: object | None = None,
    boundary: Boundary | None = None,
    kind: str = "delivery_grid",
    speed_range: tuple[float, float] = (10.0, 25.0),
    density_train: tuple[float, float] | None = None,
    speed_train: tuple[float, float] | None = None,
    model_path: str | None = None,
    run_timestamp: str = "",
) -> list[dict]:
    """生成单场景，依次运行各方法并汇总指标行。

    场景生成与数据集构建同一套边界感知生成器，保证训练 / 评测分布一致。
    """
    if boundary is not None:
        params = ScenarioParams(n_uavs=n_uavs, speed_range=speed_range)
        trajectories = generate_scenario(kind, boundary, params, seed=seed)
    else:
        from stgnn_conflict.experiments.scenario import generate_scenario as _legacy

        trajectories = _legacy(n_uavs, seed=seed)
    truth = ground_truth(trajectories, config)

    ood_tag = _ood_tag(n_uavs, speed_range, density_train, speed_train)
    meta = {
        "ood_tag": ood_tag,
        "speed_lo": speed_range[0],
        "speed_hi": speed_range[1],
        "safety_radius": config.cert.safety_radius,
        "use_motion_cert": config.cert.use_motion_cert,
        "grid_cell": config.grid_cell,
        "model_path": model_path,
        "run_timestamp": run_timestamp,
    }
    rows: list[dict] = []

    for method in methods:
        outcome = _measure(method, trajectories, config, model, priority_model)
        evaluation = evaluate(outcome.conflicts, truth)
        row: dict[str, object] = {
            "seed": seed,
            "density": n_uavs,
            "kind": kind,
            "method": method,
            **meta,
            **outcome.metrics,
            **outcome.timing,
            **evaluation,
        }
        rows.append(row)
    return rows


def run_density_sweep(
    densities: list[int],
    seeds: list[int],
    config: DetectionConfig,
    methods: list[str],
    out_path: str | None = None,
    model: object | None = None,
    priority_model: object | None = None,
    boundary: Boundary | None = None,
    kinds: list[str] | None = None,
    speed_range: tuple[float, float] = (10.0, 25.0),
    density_train: tuple[float, float] | None = None,
    speed_train: tuple[float, float] | None = None,
    model_path: str | None = None,
    ood_densities: list[int] | None = None,
    ood_speed_ranges: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """跨密度 × 种子运行全部方法，返回结果 DataFrame。

    ``ood_densities`` / ``ood_speed_ranges`` 为训练区间外的留出切片，
    由 :func:`run_scenario` 依据训练区间自动打 ``density_ood / speed_ood`` 标记。
    """
    kinds = kinds or ["delivery_grid"]
    ood_densities = ood_densities or []
    ood_speed_ranges = ood_speed_ranges or []
    slices: list[tuple[int, tuple[float, float]]] = [(n, speed_range) for n in densities]
    slices += [(n, speed_range) for n in ood_densities]
    for lo, hi in ood_speed_ranges:
        slices += [(n, (lo, hi)) for n in densities]

    run_timestamp = datetime.now(UTC).isoformat()
    rows: list[dict] = []
    for seed_index, seed in enumerate(seeds):
        kind = kinds[seed_index % len(kinds)]
        for n_uavs, slice_speed in slices:
            rows.extend(
                run_scenario(
                    n_uavs, seed, config, methods, model, priority_model,
                    boundary, kind, slice_speed, density_train, speed_train,
                    model_path, run_timestamp,
                )
            )

    df = pd.DataFrame(rows)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def _parse_interval(text: str | None) -> tuple[float, float] | None:
    """解析 ``lo,hi`` 区间参数；空串 / None 返回 None。"""
    if not text:
        return None
    lo, hi = (float(x) for x in text.split(","))
    return lo, hi


def _parse_intervals(text: str | None) -> list[tuple[float, float]]:
    """解析分号分隔的 ``lo,hi`` 区间列表；空串返回 []。"""
    if not text:
        return []
    return [tuple(float(x) for x in pair.split(",")) for pair in text.split(";") if pair]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ST-GNN 引导冲突检测密度扫描实验")
    parser.add_argument("--densities", default="20,40,60,80", help="逗号分隔的航迹数量")
    parser.add_argument("--seeds", default="1,2,3", help="逗号分隔的随机种子")
    parser.add_argument("--out", default="data/results/sweep.csv", help="原始结果 CSV 路径")
    parser.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help="逗号分隔的方法列表",
    )
    parser.add_argument(
        "--priority-model",
        default=None,
        help="训练好的子组优先级回归器路径（models/priority_regressor.pt）",
    )
    parser.add_argument(
        "--boundary",
        default="data/raw/region.geojson",
        help="边界 GeoJSON 路径；留空则回退旧版随机点生成器",
    )
    parser.add_argument(
        "--kinds",
        default="delivery_grid,vertiport",
        help="评测场景类型（逗号分隔），按种子轮转",
    )
    parser.add_argument(
        "--speed-range",
        default="10,25",
        help="评测场景速度区间 lo,hi (m/s)",
    )
    parser.add_argument(
        "--ood-densities",
        default="",
        help="区间外密度档（逗号分隔），标记 density OOD；如 10,100",
    )
    parser.add_argument(
        "--ood-speed-ranges",
        default="",
        help="区间外速度区间（分号分隔 lo,hi），标记 speed OOD；如 5,10;28,35",
    )
    parser.add_argument(
        "--density-train",
        default="20,80",
        help="训练密度区间 lo,hi；区间外场景标记 density OOD，留空不标记",
    )
    parser.add_argument(
        "--speed-train",
        default="10,25",
        help="训练速度区间 lo,hi；区间外场景标记 speed OOD，留空不标记",
    )
    parser.add_argument("--tables-dir", default="data/results/tables", help="汇总表输出目录")
    parser.add_argument("--figures-dir", default="data/results/figures", help="图输出目录")
    parser.add_argument("--history", default="models/train_history.json", help="训练历史 JSON（出训练曲线）")
    parser.add_argument("--no-plots", action="store_true", help="跳过出图")
    args = parser.parse_args(argv)

    densities = [int(x) for x in args.densities.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    kinds = [k for k in args.kinds.split(",") if k]
    speed_range = _parse_interval(args.speed_range) or (10.0, 25.0)
    density_train = _parse_interval(args.density_train)
    speed_train = _parse_interval(args.speed_train)
    ood_densities = [int(x) for x in args.ood_densities.split(",") if x]
    ood_speed_ranges = _parse_intervals(args.ood_speed_ranges)

    boundary = Boundary.from_geojson(args.boundary) if args.boundary else None

    priority_model = None
    if args.priority_model:
        from stgnn_conflict.training.train import load_priority_regressor

        priority_model, _ = load_priority_regressor(args.priority_model)
        print(f"已加载子组优先级回归器: {args.priority_model}")

    config = DetectionConfig()
    df = run_density_sweep(
        densities, seeds, config, methods, out_path=args.out,
        priority_model=priority_model, boundary=boundary, kinds=kinds,
        speed_range=speed_range, density_train=density_train,
        speed_train=speed_train, model_path=args.priority_model,
        ood_densities=ood_densities, ood_speed_ranges=ood_speed_ranges,
    )

    summary, ablation = export_tables(df, args.tables_dir)
    print("\n=== 方法级平均指标（均值 [95% CI]）===")
    print(summary.to_string(index=False))

    if not args.no_plots:
        history = args.history if Path(args.history).exists() else None
        produced = plot_all(df, ablation, args.figures_dir, history)
        if produced:
            print(f"\n已生成图: {', '.join(produced)} → {args.figures_dir}")

    print(f"\n原始结果: {args.out}")
    print(f"汇总表: {args.tables_dir}/")


if __name__ == "__main__":
    main()
