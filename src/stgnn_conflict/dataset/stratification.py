"""分层标签：认证结果层级与几何构型。"""

from __future__ import annotations

import math

import numpy as np

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.deterministic.certificates import certify_pair

# 认证结果层级
LEVEL_L1 = "L1"          # 一级证书（时间 / 包围盒分离）通过
LEVEL_L2 = "L2"          # 二级证书（相对运动下界）通过
LEVEL_CPA = "CPA"        # 无法认证，需要进入连续 CPA

# 几何构型
GEO_CROSSING = "crossing"
GEO_OVERTAKING = "overtaking"
GEO_CONVERGENCE = "convergence"
GEO_PARALLEL = "parallel"


def certificate_level(a: Trajectory, b: Trajectory, config: DetectionConfig) -> str:
    """判定航迹对最终卡在哪一层证书。"""
    result = certify_pair(a, b, config)
    if result.safe and result.name in ("time_separation", "bbox_separation"):
        return LEVEL_L1
    if result.safe and result.name == "motion_lower_bound":
        return LEVEL_L2
    return LEVEL_CPA


def _mean_velocity(traj: Trajectory) -> np.ndarray:
    return np.mean([s.vel for s in traj.states], axis=0)


def geometry_type(a: Trajectory, b: Trajectory) -> str:
    """启发式几何构型分类。

    - ``crossing``：航向夹角大（接近交叉）；
    - ``overtaking``：同向且存在明显接近速度差；
    - ``convergence``：中等夹角、相互靠近；
    - ``parallel``：航向接近且速度差小（对照构型）。
    """
    va = _mean_velocity(a)
    vb = _mean_velocity(b)
    va_norm = np.linalg.norm(va)
    vb_norm = np.linalg.norm(vb)
    if va_norm < 1e-6 or vb_norm < 1e-6:
        return GEO_CONVERGENCE

    cos_theta = float(np.dot(va, vb) / (va_norm * vb_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))
    speed_diff = float(np.linalg.norm(va - vb))

    if theta > 45.0:
        return GEO_CROSSING
    if theta < 15.0 and speed_diff > va_norm * 0.2:
        return GEO_OVERTAKING
    if theta < 15.0:
        return GEO_PARALLEL
    return GEO_CONVERGENCE
