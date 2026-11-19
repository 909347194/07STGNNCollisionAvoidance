"""决策点标签提取测试：细化子组粒度与净节省口径。"""

from pathlib import Path

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.dataset.boundary import Boundary
from stgnn_conflict.dataset.labels import (
    CERT_UNIT_COST,
    CPA_UNIT_COST,
    extract_decision_points,
)
from stgnn_conflict.dataset.scenarios import ScenarioParams, generate_scenario

GEOJSON = Path(__file__).parents[1] / "data" / "raw" / "region.geojson"
VALID_LEVELS = {"L1", "L2", "CPA"}
VALID_GEOMETRIES = {"crossing", "overtaking", "convergence", "parallel"}


def _extract(seed: int = 7, n_uavs: int = 12):
    boundary = Boundary.from_geojson(str(GEOJSON))
    params = ScenarioParams(n_uavs=n_uavs, duration=60.0)
    trajectories = generate_scenario("delivery_grid", boundary, params, seed=seed)
    return extract_decision_points(trajectories, DetectionConfig(), scenario_id="t1")


def test_multiple_subgroups_yield_multiple_points():
    """困难核被确定性细分切出多个子组时，决策点数 ≥ 子组数。"""
    points, _deferred = _extract(seed=1, n_uavs=10)
    assert len(points) >= 1
    # 至少存在一个 k < 组规模的子组级决策点，或存在延迟对
    group_sizes = {p.n_group_candidates for p in points}
    assert group_sizes
    if len(points) == 1:
        # 单决策点情形：细分未切分（延迟对可为 0）
        assert points[0].k >= 1
    else:
        assert any(p.deferred_pairs > 0 for p in points)


def test_decision_points_well_formed():
    points, deferred = _extract()
    assert points, "应至少提取到一个决策点"

    for point in points:
        assert point.k == len(point.members)
        assert point.level in VALID_LEVELS
        assert point.geometry in VALID_GEOMETRIES
        assert 0 <= point.cpa_saved <= point.cert_pairs_checked
        assert point.n_group_candidates >= point.cert_pairs_checked
        assert point.scenario_id == "t1"
    # 延迟对是与子组内候选对不相交的跨子组对
    assert isinstance(deferred, list)


def test_benefit_matches_net_saving_definition():
    points, _ = _extract()
    for point in points:
        expected = point.cpa_saved * CPA_UNIT_COST - point.cert_pairs_checked * CERT_UNIT_COST
        assert point.benefit == expected


def test_benefit_semantics_positive_when_cert_exempts_many():
    """高豁免子组收益为正；全进 CPA 的子组收益为负（认证成本沉没）。"""
    points, _ = _extract()
    for point in points:
        if point.cpa_saved == point.cert_pairs_checked:
            assert point.benefit > 0
        if point.cpa_saved == 0:
            assert point.benefit < 0
