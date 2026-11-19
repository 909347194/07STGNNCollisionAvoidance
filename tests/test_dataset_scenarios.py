"""两类场景生成器测试。"""

from pathlib import Path

from stgnn_conflict.dataset.boundary import Boundary
from stgnn_conflict.dataset.scenarios import ScenarioParams, generate_scenario

GEOJSON = Path(__file__).parents[1] / "data" / "raw" / "region.geojson"


def _boundary() -> Boundary:
    return Boundary.from_geojson(str(GEOJSON))


def test_both_kinds_produce_trajectories_in_boundary():
    boundary = _boundary()
    params = ScenarioParams(n_uavs=10, duration=60.0)
    for kind in ("delivery_grid", "vertiport"):
        trajectories = generate_scenario(kind, boundary, params, seed=1)
        assert len(trajectories) == 10
        for traj in trajectories:
            assert traj.duration > 0.0
            for state in traj.states:
                assert boundary.contains(state.pos[:2])
