"""决策点级监督标签提取。

样本单位是**决策点**，不是场景；报告时需同时给出决策点数与场景数。

决策点粒度
----------
与推理时 GNN 实际排序的对象一致：困难核先按确定性几何阈值细分为子组，
**每个细化子组是一个决策点**。被细分切断的跨子组候选对（延迟对）单独
返回，供场景级统计；其兜底代价不计入任何子组，避免重复计账。

标签定义
--------
- 主标签 ``benefit``：对该子组做证书认证相对"不做认证、候选对直接进
  CPA"的**净节省**：

      benefit = certified × c_CPA − checked × c_cert

  其中 ``certified`` 为被证书豁免（免 CPA）的候选对数，``checked`` 为
  实际检查的候选对数。语义：认证每豁免一对就省一次 CPA，每检查一对
  付出一次几何计算。这是排序监督信号，可直接跨子组比较；
- 辅助标签 ``level``：该子组最终卡在哪一层证书（L1 / L2 / CPA），
  取组内最差 pair 层级；
- ``geometry``：子组主导几何构型（交叉 / 追越 / 汇聚 / 平行）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.dataset.stratification import (
    LEVEL_CPA,
    certificate_level,
    geometry_type,
)
from stgnn_conflict.deterministic.grouping import group_by_components
from stgnn_conflict.deterministic.indexes import build_index
from stgnn_conflict.gnn.guide import (
    SPLIT_DISTANCE_PERCENTILE,
    _subdivide_deterministic,
    adaptive_split_distance,
    group_candidate_pairs,
)

# 代价权重（相对量纲，可在标定后调整）：一次 CPA 调用等价于若干次证书几何计算
CPA_UNIT_COST = 1.0
CERT_UNIT_COST = 0.1

# 保留 re-export：旧引用与文档口径一致
__all__ = ["SPLIT_DISTANCE_PERCENTILE"]

_LEVEL_RANK = {LEVEL_CPA: 2, "L2": 1, "L1": 0}


@dataclass
class DecisionPoint:
    """单个决策点样本。"""

    scenario_id: str
    group_id: str
    members: list[int]
    k: int
    level: str
    geometry: str
    cpa_saved: int  # 被证书豁免、免于 CPA 的候选对数
    cert_pairs_checked: int
    benefit: float
    n_group_candidates: int = 0  # 所属连通域的候选对总数（组级上下文）
    deferred_pairs: int = 0  # 所属连通域被细分切断的延迟对数
    oracle_order: list[int] | None = field(default=None, repr=False)


def _worst_level(levels: list[str]) -> str:
    """子组层级取最差 pair 层级（CPA > L2 > L1）。"""
    return max(levels, key=lambda lv: _LEVEL_RANK[lv])


def _mode(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def extract_decision_points(
    trajectories: list[Trajectory],
    config: DetectionConfig,
    scenario_id: str,
) -> tuple[list[DecisionPoint], list[tuple[int, int]]]:
    """提取细化子组级决策点及其标签。

    Returns
    -------
    (decision_points, deferred_pairs)
        决策点列表；被确定性细分切断的跨子组候选对。
    """
    index = build_index("grid", config.grid_cell, config.cert.safety_radius + config.cert.bbox_margin)
    candidate_pairs = index.query_pairs(trajectories)
    candidate_set = {(min(a, b), max(a, b)) for a, b in candidate_pairs}
    groups = group_by_components(len(trajectories), candidate_pairs)
    traj_map = {t.traj_id: t for t in trajectories}

    points: list[DecisionPoint] = []
    deferred_all: list[tuple[int, int]] = []

    for group_id, group in enumerate(groups):
        group_pairs = group_candidate_pairs(group, candidate_set)
        if not group_pairs:
            continue
        n_candidates = len(group_pairs)

        # 与推理时同一套确定性几何规则细分困难核；
        # 阈值自适应：组内候选对中心距离的指定分位数（无固定米制阈值时
        # 边界尺度差异过大，固定 60 m 会把组切成全孤立子组）
        geo_distance = adaptive_split_distance(group_pairs, traj_map, config)
        subgroups, cut_edges = _subdivide_deterministic(group, traj_map, geo_distance)
        if len(subgroups) == 1:
            subgroups = [group]

        deferred = [p for p in cut_edges if p in candidate_set]
        deferred_all.extend(deferred)

        for sub_id, subgroup in enumerate(subgroups):
            levels: list[str] = []
            geometries: list[str] = []
            certified = 0
            checked = 0
            for i, ta in enumerate(subgroup):
                for tb in subgroup[i + 1 :]:
                    pair = (min(ta, tb), max(ta, tb))
                    if pair not in candidate_set:
                        continue
                    checked += 1
                    level = certificate_level(traj_map[ta], traj_map[tb], config)
                    levels.append(level)
                    geometries.append(geometry_type(traj_map[ta], traj_map[tb]))
                    if level != LEVEL_CPA:
                        certified += 1
            if checked == 0:
                continue

            benefit = certified * CPA_UNIT_COST - checked * CERT_UNIT_COST
            points.append(
                DecisionPoint(
                    scenario_id=scenario_id,
                    group_id=f"{scenario_id}:g{group_id}s{sub_id}",
                    members=list(subgroup),
                    k=len(subgroup),
                    level=_worst_level(levels),
                    geometry=_mode(geometries),
                    cpa_saved=certified,
                    cert_pairs_checked=checked,
                    benefit=float(benefit),
                    n_group_candidates=n_candidates,
                    deferred_pairs=len(deferred),
                    oracle_order=None,  # k ≤ 6 的 oracle 最优顺序为可选扩展项
                )
            )
    return points, deferred_all
