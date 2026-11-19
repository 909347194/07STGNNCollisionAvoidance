"""分层标签测试：认证层级与几何构型。"""

import numpy as np

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.dataset.motion import straight_segment
from stgnn_conflict.dataset.stratification import (
    GEO_CROSSING,
    GEO_PARALLEL,
    LEVEL_CPA,
    LEVEL_L1,
    LEVEL_L2,
    certificate_level,
    geometry_type,
)


def _traj(tid: int, p0: np.ndarray, p1: np.ndarray, speed: float = 15.0) -> Trajectory:
    states = straight_segment(np.append(p0, 100.0), np.append(p1, 100.0), speed, 0.0)
    return Trajectory(traj_id=tid, states=states)


def test_geometry_crossing_and_parallel():
    crossing_a = _traj(0, np.array([0.0, 0.0]), np.array([100.0, 0.0]))
    crossing_b = _traj(1, np.array([50.0, -50.0]), np.array([50.0, 50.0]))
    assert geometry_type(crossing_a, crossing_b) == GEO_CROSSING

    parallel_a = _traj(2, np.array([0.0, 0.0]), np.array([100.0, 0.0]))
    parallel_b = _traj(3, np.array([0.0, 20.0]), np.array([100.0, 20.0]))
    assert geometry_type(parallel_a, parallel_b) == GEO_PARALLEL


def test_certificate_level_in_valid_set():
    a = _traj(0, np.array([0.0, 0.0]), np.array([100.0, 0.0]))
    b = _traj(1, np.array([50.0, -50.0]), np.array([50.0, 50.0]))
    level = certificate_level(a, b, DetectionConfig())
    assert level in (LEVEL_L1, LEVEL_L2, LEVEL_CPA)
