"""运动模型与扰动测试。"""

import numpy as np

from stgnn_conflict.dataset.motion import apply_jitter, apply_wind, straight_segment


def test_straight_segment_monotonic_and_speed():
    p0 = np.array([0.0, 0.0, 100.0])
    p1 = np.array([100.0, 0.0, 100.0])
    states = straight_segment(p0, p1, speed=20.0, t_start=0.0)
    assert len(states) == 2
    assert states[1].t > states[0].t
    assert np.isclose(np.linalg.norm(states[0].vel), 20.0)


def test_wind_added_to_velocity():
    p0 = np.array([0.0, 0.0, 100.0])
    p1 = np.array([100.0, 0.0, 100.0])
    states = straight_segment(p0, p1, speed=20.0, t_start=0.0)
    wind = np.array([3.0, -2.0, 0.0])
    drifted = apply_wind(states, wind)
    assert np.allclose(drifted[0].vel, states[0].vel + wind)


def test_jitter_keeps_time_monotonic():
    p0 = np.array([0.0, 0.0, 100.0])
    p1 = np.array([100.0, 0.0, 100.0])
    states = straight_segment(p0, p1, speed=20.0, t_start=0.0)
    rng = np.random.default_rng(0)
    jittered = apply_jitter(states, pos_std=1.0, time_std=0.05, rng=rng)
    assert jittered[1].t > jittered[0].t
