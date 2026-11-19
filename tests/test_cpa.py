"""连续 CPA 单元测试。"""

import numpy as np

from stgnn_conflict.core.trajectory import State, Trajectory
from stgnn_conflict.verifier.cpa import continuous_cpa


def _traj(traj_id: int, p0: np.ndarray, p1: np.ndarray, duration: float) -> Trajectory:
    vel = (p1 - p0) / duration
    return Trajectory(
        traj_id=traj_id,
        states=[
            State(t=0.0, pos=p0.astype(float), vel=vel.astype(float)),
            State(t=duration, pos=p1.astype(float), vel=vel.astype(float)),
        ],
    )


def test_cpa_detects_crossing():
    a = _traj(0, np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]), 10.0)
    b = _traj(1, np.array([50.0, -50.0, 0.0]), np.array([50.0, 50.0, 0.0]), 10.0)
    result = continuous_cpa(a, b, radius=30.0)
    assert result.conflict
    assert result.min_dist < 1e-6


def test_cpa_separated_pair_safe():
    a = _traj(0, np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]), 10.0)
    b = _traj(1, np.array([0.0, 200.0, 0.0]), np.array([100.0, 200.0, 0.0]), 10.0)
    result = continuous_cpa(a, b, radius=30.0)
    assert not result.conflict
    assert result.min_dist > 30.0


def test_cpa_multisegment_matches_sampling():
    """多段航迹（段起点非 0）解析 CPA 须与采样真值一致，防止时间基准错位。"""
    a = Trajectory(
        0,
        [
            State(t=0.0, pos=np.array([0.0, 0.0, 0.0]), vel=np.array([0.0, 8.0, 0.0])),
            State(t=5.0, pos=np.array([0.0, 40.0, 0.0]), vel=np.array([0.0, 12.0, 0.0])),
            State(t=10.0, pos=np.array([0.0, 100.0, 0.0]), vel=np.array([0.0, 12.0, 0.0])),
        ],
    )
    b = Trajectory(
        1,
        [
            State(t=0.0, pos=np.array([20.0, 100.0, 0.0]), vel=np.array([0.0, -8.0, 0.0])),
            State(t=5.0, pos=np.array([20.0, 60.0, 0.0]), vel=np.array([0.0, -12.0, 0.0])),
            State(t=10.0, pos=np.array([20.0, 0.0, 0.0]), vel=np.array([0.0, -12.0, 0.0])),
        ],
    )
    result = continuous_cpa(a, b, radius=30.0)

    ts = np.linspace(0.0, 10.0, 400)
    sampled = min(float(np.linalg.norm(a.position_at(t) - b.position_at(t))) for t in ts)
    assert abs(result.min_dist - sampled) < 1e-3
    assert result.conflict
