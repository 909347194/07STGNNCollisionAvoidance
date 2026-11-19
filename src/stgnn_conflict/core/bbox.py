"""三维轴对齐包围盒（AABB）及其空间关系计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stgnn_conflict.utils.geometry import aabb_min_distance


@dataclass(frozen=True)
class AABB:
    """空间包围盒，lo/hi 形状均为 (3,)。"""

    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def from_points(cls, points: np.ndarray, margin: float = 0.0) -> AABB:
        """由 (N, 3) 点集构造包围盒，可向外膨胀 margin。"""
        lo = points.min(axis=0) - margin
        hi = points.max(axis=0) + margin
        return cls(lo=lo.astype(float), hi=hi.astype(float))

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) * 0.5

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.size))

    def intersects(self, other: AABB, margin: float = 0.0) -> bool:
        """判断两包围盒是否（在 margin 膨胀后）相交。"""
        return bool(
            np.all(self.lo <= other.hi + margin)
            and np.all(other.lo <= self.hi + margin)
        )

    def min_distance(self, other: AABB) -> float:
        """两包围盒最小欧氏距离，相交时为 0。"""
        return aabb_min_distance(self.lo, self.hi, other.lo, other.hi)

    def overlap_volume(self, other: AABB) -> float:
        """相交体积（不相交为 0）。"""
        overlap = np.maximum(0.0, np.minimum(self.hi, other.hi) - np.maximum(self.lo, other.lo))
        return float(np.prod(overlap))

    def iou(self, other: AABB) -> float:
        """交并比；两盒体积均为 0 时返回 0。"""
        vol_self = float(np.prod(self.size))
        vol_other = float(np.prod(other.size))
        inter = self.overlap_volume(other)
        union = vol_self + vol_other - inter
        if union <= 1e-12:
            return 0.0
        return inter / union
