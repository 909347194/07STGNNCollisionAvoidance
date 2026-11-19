"""学习引导的子组优先级排序。

职责边界（与研究定位严格对应）：
- **确定性规则负责分组细分**：大组按预设几何阈值切分为子组，规则不依赖神经网络；
- **GNN 只输出处理策略**：评估各子组通过几何认证的潜力与处理收益，
  给出优先级排序；不直接合并 / 拆分航迹，不参与安全判定；
- 所有安全跳过仍由下游确定性证书完成，GNN 分数不构成任何认证依据。

无 PyTorch / 无训练权重时，可用几何邻近性兜底亲和度，保证流水线可运行。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory


@dataclass
class GuideResult:
    """分组引导输出。"""

    refined_groups: list[list[int]]
    priority: list[float] = field(default_factory=list)
    # 规则细分切断的候选对（确定性规则产生，非网络决策）
    split_pairs: list[tuple[int, int]] = field(default_factory=list)


# 自适应细分：保留组内中心距离最近的候选对比例（其余切断为延迟对）。
# 分位数越高保留边越多，子组越大、延迟对越少；25 分位在两类场景中
# 能稳定切出多子组，供排序学习比较。训练标签与推理必须共用同一规则。
SPLIT_DISTANCE_PERCENTILE = 25.0


def group_candidate_pairs(
    group: list[int], candidate_set: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """组内属于候选集合的候选对（规范化无序键）。"""
    return [
        (min(ta, tb), max(ta, tb))
        for i, ta in enumerate(group)
        for tb in group[i + 1 :]
        if (min(ta, tb), max(ta, tb)) in candidate_set
    ]


def adaptive_split_distance(
    group_pairs: list[tuple[int, int]],
    traj_map: dict[int, Trajectory],
    config: DetectionConfig,
) -> float:
    """组内候选对中心距离的自适应分位阈值；无候选对时回退配置固定阈值。"""
    if group_pairs:
        center_dists = [
            float(np.linalg.norm(traj_map[a].bbox.center - traj_map[b].bbox.center))
            for a, b in group_pairs
        ]
        return float(np.percentile(center_dists, SPLIT_DISTANCE_PERCENTILE))
    return config.gnn.split_geo_distance


def heuristic_edge_scores(
    edge_pairs: list[tuple[int, int]], edge_attr: np.ndarray
) -> dict[tuple[int, int], float]:
    """无模型兜底亲和度：距离越近、包围盒 IoU 与时间重叠越高，分数越高。"""
    scores: dict[tuple[int, int], float] = {}
    for k, (a, b) in enumerate(edge_pairs):
        f = edge_attr[k]
        proximity = 1.0 / (1.0 + f[0])
        iou = f[3]
        time_overlap = max(float(f[1]), 0.0)
        scores[(a, b)] = float(
            np.clip(0.5 * proximity + 0.3 * iou + 0.2 * time_overlap, 0.0, 1.0)
        )
    return scores


def _subdivide_deterministic(
    group: list[int], traj_map: dict[int, Trajectory], geo_distance: float
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """按确定性几何阈值切分组内航迹。

    保留中心距离不超过 ``geo_distance`` 的边，其余边切断；
    残差图重新连通为子组。规则完全由几何距离决定，与网络分数无关。
    """
    keep_edges: list[tuple[int, int]] = []
    cut_edges: list[tuple[int, int]] = []
    for i, ta in enumerate(group):
        for tb in group[i + 1 :]:
            center_dist = float(
                np.linalg.norm(traj_map[ta].bbox.center - traj_map[tb].bbox.center)
            )
            if center_dist <= geo_distance:
                keep_edges.append((ta, tb))
            else:
                cut_edges.append((ta, tb))

    parent = {tid: tid for tid in group}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in keep_edges:
        union(a, b)

    buckets: dict[int, list[int]] = {}
    for tid in group:
        buckets.setdefault(find(tid), []).append(tid)
    subgroups = sorted(buckets.values(), key=lambda g: min(g))
    return subgroups, cut_edges


def _group_priority(group: list[int], scores: dict[tuple[int, int], float]) -> float:
    """子组认证潜力优先级：内部边平均亲和度；无边孤立子组给中性分 0.5。"""
    vals = [
        scores[edge]
        for a in group
        for b in group
        if (edge := (min(a, b), max(a, b))) in scores
    ]
    return float(np.mean(vals)) if vals else 0.5


def guide_groups(
    deterministic_groups: list[list[int]],
    scores: dict[tuple[int, int], float],
    config: DetectionConfig,
    traj_map: dict[int, Trajectory] | None = None,
    candidate_set: set[tuple[int, int]] | None = None,
) -> GuideResult:
    """确定性细分 + 子组优先级排序。

    Parameters
    ----------
    deterministic_groups: 确定性连通域分组（困难核）。
    scores: 边亲和度（GNN 推理或几何兜底），仅用于优先级排序。
    config: 全局配置。
    traj_map: 航迹映射，确定性细分几何阈值判定使用。
    candidate_set: 候选对集合；提供时按候选对计算自适应阈值（与训练标签
        同一规则），否则回退用 scores 中的边。
    """
    refined: list[list[int]] = []
    split_pairs: list[tuple[int, int]] = []

    for group in deterministic_groups:
        if traj_map is not None:
            if candidate_set is not None:
                group_pairs = group_candidate_pairs(group, candidate_set)
            else:
                group_pairs = [p for p in scores if p[0] in group and p[1] in group]
            geo_distance = adaptive_split_distance(group_pairs, traj_map, config)
            subgroups, cut = _subdivide_deterministic(group, traj_map, geo_distance)
            split_pairs.extend(cut)
            refined.extend(subgroups)
        else:
            refined.append(list(group))

    priority = [_group_priority(group, scores) for group in refined]
    return GuideResult(refined_groups=refined, priority=priority, split_pairs=split_pairs)
