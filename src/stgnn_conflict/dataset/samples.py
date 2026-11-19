"""训练样本转换：决策点 → 子组级回归样本。

训练目标与推理时 GNN 的实际输出严格对齐：
- 子组优先级标签 = ``benefit``（认证净节省，排序监督信号）；
- 子组特征 = ``build_graph`` 的节点特征在子组成员上的聚合 + 图边特征聚合；
- 细分规则与阈值与 ``gnn.guide`` 共享，训练 / 推理子组粒度一致。

样本单位是决策点（细化子组），不是场景；训练 / 验证仍按场景划分。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.dataset.labels import CERT_UNIT_COST, CPA_UNIT_COST
from stgnn_conflict.deterministic.certificates import certify_pair
from stgnn_conflict.deterministic.grouping import group_by_components
from stgnn_conflict.deterministic.indexes import build_index
from stgnn_conflict.gnn.graph import _edge_features, _node_features
from stgnn_conflict.gnn.guide import (
    _subdivide_deterministic,
    adaptive_split_distance,
    group_candidate_pairs,
)

# 子组样本特征维度：节点特征聚合（min/mean/max）+ 边特征聚合 + 规模特征
NODE_BLOCK = 6 * 3
EDGE_BLOCK = 7 * 2
SIZE_BLOCK = 3
SUBGROUP_FEATURE_DIM = NODE_BLOCK + EDGE_BLOCK + SIZE_BLOCK


@dataclass
class SubgroupSample:
    """单个子组级训练样本。"""

    scenario_id: str
    group_id: str
    members: list[int]
    k: int
    benefit: float
    features: np.ndarray  # (SUBGROUP_FEATURE_DIM,)


def _subgroup_features(
    members: list[int],
    traj_map: dict[int, Trajectory],
    candidate_set: set[tuple[int, int]],
) -> np.ndarray:
    """子组特征：节点特征 min/mean/max + 组内边特征 mean/std + 规模。

    边特征直接复用 ``graph._edge_features``，与 ``build_graph`` 同口径。
    """
    node_feats = np.array([_node_features(traj_map[tid]) for tid in members])
    pair_pairs = group_candidate_pairs(members, candidate_set)
    edge_arr = np.array(
        [_edge_features(traj_map[a], traj_map[b]) for a, b in pair_pairs]
    )
    if edge_arr.size:
        edge_block = np.concatenate([edge_arr.mean(axis=0), edge_arr.std(axis=0)])
    else:
        edge_block = np.zeros(EDGE_BLOCK)

    k = len(members)
    pair_capacity = max(k * (k - 1) / 2.0, 1.0)
    size_block = np.array(
        [k / 20.0, len(pair_pairs) / 50.0, len(pair_pairs) / pair_capacity]
    )
    return np.concatenate(
        [
            node_feats.min(axis=0),
            node_feats.mean(axis=0),
            node_feats.max(axis=0),
            edge_block,
            size_block,
        ]
    )


def build_subgroup_samples(
    trajectories: list[Trajectory],
    config: DetectionConfig,
    scenario_id: str,
) -> list[SubgroupSample]:
    """对一个场景重建决策点并计算子组特征（与 labels.py 同一套流程）。"""
    index = build_index("grid", config.grid_cell, config.cert.safety_radius + config.cert.bbox_margin)
    candidate_pairs = index.query_pairs(trajectories)
    candidate_set = {(min(a, b), max(a, b)) for a, b in candidate_pairs}
    groups = group_by_components(len(trajectories), candidate_pairs)
    traj_map = {t.traj_id: t for t in trajectories}

    samples: list[SubgroupSample] = []
    for group_id, group in enumerate(groups):
        group_pairs = group_candidate_pairs(group, candidate_set)
        if not group_pairs:
            continue

        geo_distance = adaptive_split_distance(group_pairs, traj_map, config)
        subgroups, _cut = _subdivide_deterministic(group, traj_map, geo_distance)
        if len(subgroups) == 1:
            subgroups = [group]

        for sub_id, subgroup in enumerate(subgroups):
            checked = 0
            certified = 0
            for i, ta in enumerate(subgroup):
                for tb in subgroup[i + 1 :]:
                    pair = (min(ta, tb), max(ta, tb))
                    if pair not in candidate_set:
                        continue
                    checked += 1
                    if certify_pair(traj_map[ta], traj_map[tb], config).safe:
                        certified += 1
            if checked == 0:
                continue
            benefit = certified * CPA_UNIT_COST - checked * CERT_UNIT_COST
            samples.append(
                SubgroupSample(
                    scenario_id=scenario_id,
                    group_id=f"{scenario_id}:g{group_id}s{sub_id}",
                    members=list(subgroup),
                    k=len(subgroup),
                    benefit=float(benefit),
                    features=_subgroup_features(
                        subgroup, traj_map, candidate_set
                    ),
                )
            )
    return samples
