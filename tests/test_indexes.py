"""索引召回一致性测试：各索引候选集合应与暴力扫描一致。"""

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.deterministic.indexes import (
    BruteForceIndex,
    GridIndex,
    StaticRTreeIndex,
)
from stgnn_conflict.experiments.scenario import generate_scenario


def _candidate_set(index, trajectories):
    return {tuple(sorted(pair)) for pair in index.query_pairs(trajectories)}


def test_indexes_match_brute_force():
    config = DetectionConfig()
    margin = config.cert.safety_radius + config.cert.bbox_margin
    trajectories = generate_scenario(n_uavs=18, seed=7)

    brute = _candidate_set(BruteForceIndex(margin=margin), trajectories)
    grid = _candidate_set(GridIndex(cell=config.grid_cell, margin=margin), trajectories)
    rtree = _candidate_set(StaticRTreeIndex(margin=margin), trajectories)

    assert grid == brute
    assert rtree == brute
