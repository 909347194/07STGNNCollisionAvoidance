"""引导层测试：确定性细分覆盖完整、优先级口径正确。"""

import numpy as np

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import State, Trajectory
from stgnn_conflict.gnn.graph import build_graph
from stgnn_conflict.gnn.guide import guide_groups, heuristic_edge_scores


def _traj(tid: int, p0: np.ndarray, p1: np.ndarray, duration: float = 10.0) -> Trajectory:
    vel = (p1 - p0) / duration
    return Trajectory(
        traj_id=tid,
        states=[
            State(t=0.0, pos=p0.astype(float), vel=vel.astype(float)),
            State(t=duration, pos=p1.astype(float), vel=vel.astype(float)),
        ],
    )


def test_guide_subdivision_covers_all_and_priority_valid():
    trajs = [
        _traj(0, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])),
        _traj(1, np.array([2.0, 2.0, 0.0]), np.array([12.0, 2.0, 0.0])),
        _traj(2, np.array([300.0, 300.0, 0.0]), np.array([310.0, 300.0, 0.0])),
    ]
    pairs = [(0, 1), (0, 2), (1, 2)]
    graph = build_graph(trajs, pairs, DetectionConfig())
    scores = heuristic_edge_scores(graph.edge_pairs, graph.edge_attr)
    traj_map = {t.traj_id: t for t in trajs}

    guide = guide_groups([[0, 1, 2]], scores, DetectionConfig(), traj_map=traj_map)

    flat = [tid for group in guide.refined_groups for tid in group]
    assert sorted(flat) == [0, 1, 2]
    assert len(guide.priority) == len(guide.refined_groups)
    assert all(0.0 <= p <= 1.0 for p in guide.priority)
    # 远距离航迹对应边被确定性规则切断
    assert {pair for pair in guide.split_pairs} & {(0, 2), (1, 2)}
