"""航迹域模型：状态点、航段与整条航迹。

运动模型：相邻状态点之间为匀速直线运动；速度由位移除以段时长推导，
状态中记录的 vel 仅用于特征统计（如平均速度）。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from stgnn_conflict.core.bbox import AABB


@dataclass(frozen=True)
class State:
    """单个时刻的飞行状态。"""

    t: float
    pos: np.ndarray  # (3,) 位置 (x, y, z)
    vel: np.ndarray  # (3,) 速度


@dataclass
class Segment:
    """两个航迹点之间的匀速直线航段。"""

    traj_id: int
    index: int
    t0: float
    t1: float
    p0: np.ndarray
    p1: np.ndarray

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def velocity(self) -> np.ndarray:
        if self.duration <= 1e-12:
            return np.zeros(3)
        return (self.p1 - self.p0) / self.duration

    def position_at(self, t: float) -> np.ndarray:
        alpha = (t - self.t0) / self.duration if self.duration > 1e-12 else 0.0
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return self.p0 + alpha * (self.p1 - self.p0)


class Trajectory:
    """整条航迹：有序状态点列表及其派生的航段、包围盒与时间区间。"""

    def __init__(self, traj_id: int, states: list[State]) -> None:
        if len(states) < 2:
            raise ValueError(f"航迹 {traj_id} 至少需要 2 个状态点")
        times = [s.t for s in states]
        if any(b <= a for a, b in itertools.pairwise(times)):
            raise ValueError(f"航迹 {traj_id} 时间戳必须严格递增")

        self.traj_id = traj_id
        self.states: list[State] = list(states)
        self.segments: list[Segment] = self._build_segments()
        self.t0: float = times[0]
        self.t1: float = times[-1]
        self.bbox: AABB = AABB.from_points(np.array([s.pos for s in self.states]))

    def _build_segments(self) -> list[Segment]:
        segments: list[Segment] = []
        for i in range(len(self.states) - 1):
            a, b = self.states[i], self.states[i + 1]
            segments.append(
                Segment(
                    traj_id=self.traj_id,
                    index=i,
                    t0=a.t,
                    t1=b.t,
                    p0=a.pos.copy(),
                    p1=b.pos.copy(),
                )
            )
        return segments

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def mean_speed(self) -> float:
        speeds = [float(np.linalg.norm(s.vel)) for s in self.states]
        return float(np.mean(speeds)) if speeds else 0.0

    def segment_at(self, t: float) -> Segment:
        """返回包含时刻 t 的航段；越界时返回最近端航段。"""
        if t <= self.t0:
            return self.segments[0]
        if t >= self.t1:
            return self.segments[-1]
        for seg in self.segments:
            if seg.t0 - 1e-9 <= t <= seg.t1 + 1e-9:
                return seg
        return self.segments[-1]

    def position_at(self, t: float) -> np.ndarray:
        return self.segment_at(t).position_at(t)

    def time_overlaps(self, other: Trajectory, eps: float = 1e-9) -> bool:
        return not (self.t1 < other.t0 - eps or other.t1 < self.t0 - eps)
