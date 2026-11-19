"""困难核时空图构建。

节点：仍存在潜在关联的航迹；边：索引召回的潜在时空关联。
节点/边特征均为描述时空邻近性与运动差异的几何量，
图结构不包含任何碰撞标签——GNN 只学习分组亲和度与认证优先级。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.utils.geometry import overlap_interval, safe_div

NODE_FEATURE_DIM = 6
EDGE_FEATURE_DIM = 7


@dataclass
class HeteroGraph:
    """困难核异构图（同质节点 + 关联边）。"""

    node_ids: list[int]
    x: np.ndarray  # (N, NODE_FEATURE_DIM)
    edge_index: np.ndarray  # (2, E)，内容为节点下标
    edge_attr: np.ndarray  # (E, EDGE_FEATURE_DIM)
    edge_pairs: list[tuple[int, int]]  # 与边一一对应的 tid 对 (a < b)


def _node_features(traj: Trajectory) -> np.ndarray:
    bbox = traj.bbox
    return np.array(
        [
            traj.mean_speed / 30.0,
            traj.duration / 60.0,
            len(traj.segments) / 10.0,
            bbox.diagonal / 100.0,
            float(np.mean([s.pos[2] for s in traj.states])) / 1000.0,
            traj.duration / (traj.duration + 1.0),
        ]
    )


def _edge_features(a: Trajectory, b: Trajectory) -> np.ndarray:
    center_dist = float(np.linalg.norm(a.bbox.center - b.bbox.center))
    overlap = overlap_interval(a.t0, a.t1, b.t0, b.t1)
    if overlap is None:
        gap = min(abs(a.t0 - b.t1), abs(b.t0 - a.t1))
        time_score = -gap / 60.0
    else:
        time_score = (overlap[1] - overlap[0]) / max(a.duration, b.duration)
    mean_vel_a = np.mean([s.vel for s in a.states], axis=0)
    mean_vel_b = np.mean([s.vel for s in b.states], axis=0)
    vel_diff = float(np.linalg.norm(mean_vel_a - mean_vel_b))
    iou = a.bbox.iou(b.bbox)
    overlap_vol = a.bbox.overlap_volume(b.bbox)
    bbox_gap = a.bbox.min_distance(b.bbox)
    alt_diff = abs(
        float(np.mean([s.pos[2] for s in a.states]))
        - float(np.mean([s.pos[2] for s in b.states]))
    )
    return np.array(
        [
            center_dist / 100.0,
            time_score,
            vel_diff / 30.0,
            iou,
            safe_div(overlap_vol, 1e6),
            bbox_gap / 100.0,
            alt_diff / 100.0,
        ]
    )


def build_graph(
    trajectories: list[Trajectory],
    candidate_pairs: list[tuple[int, int]],
    config: DetectionConfig,
) -> HeteroGraph:
    """由候选航迹对构建困难核图；孤立航迹不进入图。"""
    traj_map = {t.traj_id: t for t in trajectories}
    active_ids = sorted({tid for pair in candidate_pairs for tid in pair})
    id_to_idx = {tid: i for i, tid in enumerate(active_ids)}

    x = np.array([_node_features(traj_map[tid]) for tid in active_ids])

    edge_index: list[list[int]] = [[], []]
    edge_attr: list[np.ndarray] = []
    edge_pairs: list[tuple[int, int]] = []
    for a, b in candidate_pairs:
        ia, ib = id_to_idx[a], id_to_idx[b]
        edge_index[0].append(ia)
        edge_index[1].append(ib)
        edge_attr.append(_edge_features(traj_map[a], traj_map[b]))
        edge_pairs.append((a, b))

    return HeteroGraph(
        node_ids=active_ids,
        x=x,
        edge_index=np.array(edge_index, dtype=int),
        edge_attr=np.array(edge_attr),
        edge_pairs=edge_pairs,
    )
