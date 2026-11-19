"""基础几何与运动学工具函数。

运动模型假设：相邻航迹点之间为匀速直线运动（分段线性轨迹）。
所有时间区间均为闭区间，数值比较使用调用方传入的容差。
"""

from __future__ import annotations

import numpy as np


def overlap_interval(
    a0: float, a1: float, b0: float, b1: float, eps: float = 1e-9
) -> tuple[float, float] | None:
    """返回两个时间区间的重叠闭区间；不重叠返回 None。"""
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi - lo < -eps:
        return None
    return lo, max(lo, hi)


def aabb_min_distance(
    lo1: np.ndarray, hi1: np.ndarray, lo2: np.ndarray, hi2: np.ndarray
) -> float:
    """两个轴对齐包围盒之间的最小欧氏距离（相交时为 0）。"""
    gap = np.maximum(0.0, np.maximum(lo2 - hi1, lo1 - hi2))
    return float(np.linalg.norm(gap))


def closest_approach_linear(
    p0: np.ndarray,
    v0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    t0: float,
    t1: float,
) -> tuple[float, float]:
    """两段匀速直线运动在时间区间 [t0, t1] 内的最近接近距离。

    ``p0 / p1`` 必须是两航迹在区间起点 ``t0`` 处的位置；相对运动在局部
    时间 ``u ∈ [0, t1-t0]`` 内求解，避免段起点非零时绝对时间基准错位。

    Returns
    -------
    (min_dist, t_star)
        最小距离与其发生的绝对时刻。
    """
    r0 = p0 - p1
    vr = v0 - v1
    dt = max(t1 - t0, 0.0)

    a = float(np.dot(vr, vr))
    b = 2.0 * float(np.dot(r0, vr))
    c = float(np.dot(r0, r0))

    if a <= 1e-12:
        u_star = 0.0
        min_sq = c
    else:
        u_crit = -b / (2.0 * a)
        u_star = float(np.clip(u_crit, 0.0, dt))
        if u_crit <= 0.0:
            min_sq = c
        elif u_crit >= dt:
            min_sq = c + b * dt + a * dt * dt
        else:
            min_sq = c - b * b / (4.0 * a)
    min_sq = max(min_sq, 0.0)
    return float(np.sqrt(min_sq)), t0 + u_star


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """避免除零的安全除法。"""
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator
