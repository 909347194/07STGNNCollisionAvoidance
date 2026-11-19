"""航迹域模型单元测试。"""

import numpy as np

from stgnn_conflict.core.trajectory import State, Trajectory


def _straight(traj_id: int, p0: np.ndarray, p1: np.ndarray, duration: float) -> Trajectory:
    vel = (p1 - p0) / duration
    return Trajectory(
        traj_id=traj_id,
        states=[
            State(t=0.0, pos=p0.astype(float), vel=vel.astype(float)),
            State(t=duration, pos=p1.astype(float), vel=vel.astype(float)),
        ],
    )


def test_position_interpolation():
    traj = _straight(0, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), 10.0)
    mid = traj.position_at(5.0)
    assert np.allclose(mid, [5.0, 0.0, 0.0])


def test_bbox_and_time():
    traj = _straight(0, np.array([0.0, -2.0, 5.0]), np.array([4.0, 6.0, 5.0]), 8.0)
    assert traj.t0 == 0.0 and traj.t1 == 8.0
    assert np.allclose(traj.bbox.lo, [0.0, -2.0, 5.0])
    assert np.allclose(traj.bbox.hi, [4.0, 6.0, 5.0])


def test_segment_at_bounds():
    traj = _straight(0, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), 10.0)
    assert traj.segment_at(-1.0).index == 0
    assert traj.segment_at(11.0).index == 0
