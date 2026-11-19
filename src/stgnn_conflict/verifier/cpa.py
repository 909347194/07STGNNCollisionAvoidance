"""连续最近接近点（CPA）精确检测。

对两段轨迹的所有重叠航段对，解析求解匀速直线运动的最近接近距离，
全局取最小距离；小于安全半径即判定冲突。这是流水线的最后一道关口：
凡未被确定性证书豁免的候选对，都必须进入 CPA，以保证零漏检。
"""

from __future__ import annotations

from dataclasses import dataclass

from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.utils.geometry import closest_approach_linear, overlap_interval


@dataclass
class CPAResult:
    """CPA 检测结果。"""

    conflict: bool
    min_dist: float
    t_star: float


def continuous_cpa(a: Trajectory, b: Trajectory, radius: float, eps: float = 1e-3) -> CPAResult:
    """计算两航迹在重叠时间内的最小距离并判定冲突。"""
    min_dist = float("inf")
    t_star = float("nan")
    for sa in a.segments:
        for sb in b.segments:
            overlap = overlap_interval(sa.t0, sa.t1, sb.t0, sb.t1)
            if overlap is None:
                continue
            t0, t1 = overlap
            pa0 = sa.position_at(t0)
            pb0 = sb.position_at(t0)
            dist, t_hit = closest_approach_linear(pa0, sa.velocity, pb0, sb.velocity, t0, t1)
            if dist < min_dist:
                min_dist, t_star = dist, t_hit

    if min_dist == float("inf"):
        return CPAResult(conflict=False, min_dist=float("inf"), t_star=float("nan"))

    return CPAResult(conflict=min_dist <= radius + eps, min_dist=min_dist, t_star=t_star)
