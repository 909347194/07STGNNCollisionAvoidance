"""数据集构建编排：场景生成 → 注入 → 决策点标签 → 划分落盘。

输出
----
- ``manifest.csv``：场景级元数据（一行一场景）；
- ``decision_points.csv``：决策点级样本（一行一决策点）。

划分纪律
--------
- 训练 / 留出测试按场景划分，同一 administration 场景不跨子集；
- 注入场景与自然场景在元数据中分别标记，报告时分开统计；
- 每个场景构建后执行零漏检硬断言。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.dataset.boundary import Boundary
from stgnn_conflict.dataset.injection import inject_conflicts
from stgnn_conflict.dataset.labels import DecisionPoint, extract_decision_points
from stgnn_conflict.dataset.scenarios import ScenarioParams, generate_scenario
from stgnn_conflict.dataset.split import ood_density_mask, ood_speed_mask
from stgnn_conflict.dataset.stats import assert_zero_miss
from stgnn_conflict.experiments.baselines import ground_truth, run_deterministic_group


@dataclass
class ScenarioSpec:
    """单组场景生成规格。"""

    kind: str
    n_uavs: int
    seed: int
    duration: float = 60.0
    speed_range: tuple[float, float] = (10.0, 25.0)
    tag: str = ""  # 场景 id 后缀，用于区分同密度 / 同种子的 OOD 切片


def _scenario_mean_speed(trajectories: list) -> float:
    return float(np.mean([t.mean_speed for t in trajectories]))


def build_scenario_records(
    specs: list[ScenarioSpec],
    boundary: Boundary,
    config: DetectionConfig,
    injection_ratio: float = 0.0,
    base_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成全部场景并提取决策点，返回 (场景元数据表, 决策点表)。"""
    manifest_rows: list[dict] = []
    dp_rows: list[dict] = []

    for spec in specs:
        params = ScenarioParams(n_uavs=spec.n_uavs, duration=spec.duration, speed_range=spec.speed_range)
        scenario_id = f"{spec.kind}-s{spec.seed}-n{spec.n_uavs}"
        if spec.tag:
            scenario_id = f"{scenario_id}-{spec.tag}"
        trajectories = generate_scenario(spec.kind, boundary, params, seed=base_seed + spec.seed)

        injected_pairs: list[tuple[int, int]] = []
        actual_ratio = 0.0
        if injection_ratio > 0.0:
            rng = np.random.default_rng(base_seed + spec.seed)
            trajectories, injected_pairs, actual_ratio = inject_conflicts(trajectories, injection_ratio, rng)

        # 零漏检硬断言：确定性管线结果对照暴力真值
        truth = ground_truth(trajectories, config)
        outcome = run_deterministic_group(trajectories, config)
        assert_zero_miss(outcome.conflicts, truth)

        points, deferred = extract_decision_points(
            trajectories, config, scenario_id=scenario_id
        )
        for point in points:
            dp_rows.append(_point_to_row(point))

        manifest_rows.append(
            {
                "scenario_id": points[0].scenario_id if points else scenario_id,
                "kind": spec.kind,
                "seed": spec.seed,
                "n_uavs": spec.n_uavs,
                "duration": spec.duration,
                "speed_lo": spec.speed_range[0],
                "speed_hi": spec.speed_range[1],
                "mean_speed": _scenario_mean_speed(trajectories),
                "injection_ratio": actual_ratio,
                "injected": bool(injected_pairs),
                "n_decision_points": len(points),
                "n_deferred_pairs": len(deferred),
                "n_conflicts": len(truth),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    decision_points = pd.DataFrame(dp_rows)
    return manifest, decision_points


def _point_to_row(point: DecisionPoint) -> dict:
    return {
        "scenario_id": point.scenario_id,
        "group_id": point.group_id,
        "members": ";".join(str(m) for m in point.members),
        "k": point.k,
        "level": point.level,
        "geometry": point.geometry,
        "cpa_saved": point.cpa_saved,
        "cert_pairs_checked": point.cert_pairs_checked,
        "benefit": point.benefit,
        "n_group_candidates": point.n_group_candidates,
        "deferred_pairs": point.deferred_pairs,
    }


def assign_splits(
    manifest: pd.DataFrame,
    decision_points: pd.DataFrame,
    train_ratio: float = 0.7,
    seed: int = 0,
    density_train: tuple[float, float] | None = None,
    speed_train: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """场景级划分，并标记密度 / 速度 OOD 切片。"""
    from stgnn_conflict.dataset.split import split_by_scenario

    scenario_ids = manifest["scenario_id"].tolist()
    train_ids, _test_ids = split_by_scenario(scenario_ids, train_ratio=train_ratio, seed=seed)

    manifest = manifest.copy()
    manifest["split"] = np.where(manifest["scenario_id"].isin(train_ids), "train", "test")

    if density_train is not None:
        manifest["ood_density"] = ood_density_mask(manifest, *density_train)
    if speed_train is not None:
        manifest["ood_speed"] = ood_speed_mask(manifest, *speed_train)

    # OOD 场景只用于留出评测，强制归入 test，不参与训练 / 早停
    ood_mask = manifest.get("ood_density", False) | manifest.get("ood_speed", False)
    manifest.loc[ood_mask, "split"] = "test"

    decision_points = decision_points.merge(
        manifest[["scenario_id", "split", "kind"]], on="scenario_id", how="left"
    )
    return manifest, decision_points
