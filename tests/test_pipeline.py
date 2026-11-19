"""流水线端到端测试：对照暴力真值验证零漏检与指标合理性。"""

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.experiments.baselines import (
    ground_truth,
    run_bbox_only,
    run_deterministic_group,
    run_gnn_guided,
)
from stgnn_conflict.experiments.metrics import evaluate
from stgnn_conflict.experiments.scenario import generate_scenario


def _scenario():
    return generate_scenario(n_uavs=24, seed=3)


def test_deterministic_group_zero_miss():
    trajectories = _scenario()
    config = DetectionConfig()
    truth = ground_truth(trajectories, config)

    outcome = run_deterministic_group(trajectories, config)
    evaluation = evaluate(outcome.conflicts, truth)

    assert evaluation["miss_rate"] == 0.0
    assert outcome.metrics["cpa_calls"] <= outcome.metrics["candidate_pairs"]


def test_gnn_guided_zero_miss():
    trajectories = _scenario()
    config = DetectionConfig()
    truth = ground_truth(trajectories, config)

    outcome = run_gnn_guided(trajectories, config)
    evaluation = evaluate(outcome.conflicts, truth)

    assert evaluation["miss_rate"] == 0.0
    assert outcome.metrics["cpa_calls"] <= outcome.metrics["candidate_pairs"]


def test_bbox_only_zero_miss_and_metrics():
    trajectories = _scenario()
    config = DetectionConfig()
    truth = ground_truth(trajectories, config)

    outcome = run_bbox_only(trajectories, config)
    evaluation = evaluate(outcome.conflicts, truth)

    assert evaluation["miss_rate"] == 0.0
    assert "certified_group_ratio" in outcome.metrics
    assert outcome.metrics["certified_group_ratio"] >= 0.0
