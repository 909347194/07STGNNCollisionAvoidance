"""分层安全证书单元测试。"""

import numpy as np

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import State, Trajectory
from stgnn_conflict.deterministic.certificates import (
    bbox_separation,
    certify_pair,
    motion_lower_bound,
    time_separation,
)


def _traj(traj_id: int, p0: np.ndarray, p1: np.ndarray, t0: float, t1: float) -> Trajectory:
    vel = (p1 - p0) / (t1 - t0)
    return Trajectory(
        traj_id=traj_id,
        states=[
            State(t=t0, pos=p0.astype(float), vel=vel.astype(float)),
            State(t=t1, pos=p1.astype(float), vel=vel.astype(float)),
        ],
    )


def test_time_separation_certificate():
    a = _traj(0, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), 0.0, 5.0)
    b = _traj(1, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), 6.0, 10.0)
    assert time_separation(a, b, eps=1e-3).safe


def test_bbox_separation_certificate():
    a = _traj(0, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), 0.0, 10.0)
    b = _traj(1, np.array([0.0, 200.0, 0.0]), np.array([10.0, 200.0, 0.0]), 0.0, 10.0)
    cert = DetectionConfig()
    assert bbox_separation(a, b, cert.cert.safety_radius, cert.cert.bbox_margin, cert.cert.space_eps).safe


def test_motion_lower_bound_for_parallel_separated():
    a = _traj(0, np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]), 0.0, 10.0)
    b = _traj(1, np.array([0.0, 50.0, 0.0]), np.array([100.0, 50.0, 0.0]), 0.0, 10.0)
    result = motion_lower_bound(a, b, 30.0, 1e-3)
    assert result.safe
    assert result.bound >= 50.0 - 1e-6


def test_certify_pair_uncertain_for_crossing():
    a = _traj(0, np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]), 0.0, 10.0)
    b = _traj(1, np.array([50.0, -50.0, 0.0]), np.array([50.0, 50.0, 0.0]), 0.0, 10.0)
    result = certify_pair(a, b, DetectionConfig())
    assert not result.safe


def test_motion_cert_switch_disabled():
    cfg = DetectionConfig()
    cfg.cert.use_motion_cert = False
    a = _traj(0, np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]), 0.0, 10.0)
    b = _traj(1, np.array([50.0, -50.0, 0.0]), np.array([50.0, 50.0, 0.0]), 0.0, 10.0)
    result = certify_pair(a, b, cfg)
    # 关闭运动证书后，交叉对只能落到 UNCERTAIN，不被误豁免
    assert not result.safe

    separated = _traj(2, np.array([0.0, 200.0, 0.0]), np.array([100.0, 200.0, 0.0]), 0.0, 10.0)
    assert bbox_separation(a, separated, cfg.cert.safety_radius, cfg.cert.bbox_margin, cfg.cert.space_eps).safe
