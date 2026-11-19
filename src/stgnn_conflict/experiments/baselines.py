"""对比基线实现。

基线谱系
--------
- 真值（ground truth）：全部航迹对暴力 CPA，作为漏检率评判基准；
- 索引-only：R-tree / 时空网格召回候选对后直接 CPA，不做分组与证书；
- 纯包围盒方法：分组 + 一级证书（时间 / 包围盒分离），不启用相对运动下界证书；
- 纯确定性分层认证：分组 + 三级分层证书（含相对运动下界），无学习模块；
- ST-GNN 增强方法：完整流水线（确定性主干 + GNN 优先级引导 + 几何验证）。

消融对照
--------
- 相对运动证书贡献：``det_group`` 对照 ``bbox_only``；
- GNN 贡献：``gnn_guided`` 对照 ``det_group``。
"""

from __future__ import annotations

import copy

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.deterministic.indexes import build_index
from stgnn_conflict.pipeline.detector import ConflictDetector, DetectionOutcome
from stgnn_conflict.verifier.cpa import continuous_cpa


def ground_truth(trajectories: list[Trajectory], config: DetectionConfig) -> set[tuple[int, int]]:
    """全暴力 CPA 真值冲突集合。"""
    conflicts: set[tuple[int, int]] = set()
    for i, ta in enumerate(trajectories):
        for tb in trajectories[i + 1 :]:
            result = continuous_cpa(ta, tb, config.cert.safety_radius, config.cert.time_eps)
            if result.conflict:
                conflicts.add((min(ta.traj_id, tb.traj_id), max(ta.traj_id, tb.traj_id)))
    return conflicts


def _cpa_pairs(
    pairs: list[tuple[int, int]], traj_map: dict[int, Trajectory], config: DetectionConfig
) -> tuple[set[tuple[int, int]], int]:
    conflicts: set[tuple[int, int]] = set()
    calls = 0
    for a, b in pairs:
        calls += 1
        result = continuous_cpa(traj_map[a], traj_map[b], config.cert.safety_radius, config.cert.time_eps)
        if result.conflict:
            conflicts.add((min(a, b), max(a, b)))
    return conflicts, calls


def run_index_only(
    trajectories: list[Trajectory], config: DetectionConfig, index_kind: str
) -> DetectionOutcome:
    """索引召回后直接 CPA（无分组、无证书）。"""
    import time

    t0 = time.perf_counter()
    recall_margin = config.cert.safety_radius + config.cert.bbox_margin
    index = build_index(index_kind, config.grid_cell, recall_margin)
    pairs = index.query_pairs(trajectories)
    traj_map = {t.traj_id: t for t in trajectories}
    conflicts, calls = _cpa_pairs(pairs, traj_map, config)
    total_ms = (time.perf_counter() - t0) * 1e3

    outcome = DetectionOutcome(conflicts=conflicts)
    outcome.metrics = {
        "candidate_pairs": len(pairs),
        "cpa_calls": calls,
        "certified_pairs": 0,
        "certified_groups": 0,
        "certified_group_ratio": 0.0,
        "uncertified_pairs": 0,
        "conflicts": len(conflicts),
    }
    outcome.timing = {"total_ms": total_ms}
    return outcome


def run_bbox_only(
    trajectories: list[Trajectory], config: DetectionConfig, index_kind: str = "grid"
) -> DetectionOutcome:
    """纯包围盒方法：分组 + 一级证书（时间 / 包围盒），关闭相对运动下界证书。"""
    bbox_config = copy.deepcopy(config)
    bbox_config.cert.use_motion_cert = False
    return ConflictDetector(bbox_config, index_kind=index_kind, use_gnn=False).detect(
        trajectories
    )


def run_deterministic_group(
    trajectories: list[Trajectory], config: DetectionConfig, index_kind: str = "grid"
) -> DetectionOutcome:
    """纯确定性分层认证基线：分组 + 三级分层证书，无学习模块。"""
    return ConflictDetector(config, index_kind=index_kind, use_gnn=False).detect(trajectories)


def run_gnn_guided(
    trajectories: list[Trajectory],
    config: DetectionConfig,
    index_kind: str = "grid",
    model: object | None = None,
    priority_model: object | None = None,
) -> DetectionOutcome:
    """学习引导完整流水线（可选训练好的子组优先级回归器）。"""
    return ConflictDetector(
        config, index_kind=index_kind, use_gnn=True, model=model, priority_model=priority_model
    ).detect(trajectories)
