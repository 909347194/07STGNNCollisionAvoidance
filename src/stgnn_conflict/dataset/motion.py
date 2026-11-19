"""运动模型与扰动。

初期实现多旋翼够用模型：
- 匀速直线航段；
- 简单圆弧转弯段；
- 风场扰动（速度叠加）与定位 / 时间戳小抖动。

固定翼 Dubins 曲线、复杂加减速作为可选扩展，本版本不实现。
``State`` 为不可变对象，所有扰动返回新的状态列表。
"""

from __future__ import annotations

import math

import numpy as np

from stgnn_conflict.core.trajectory import State


def _angle_diff(target: float, source: float) -> float:
    """把目标角度转到与源角度夹角最短的有向差值。"""
    return (target - source + math.pi) % (2 * math.pi) - math.pi


def straight_segment(p0: np.ndarray, p1: np.ndarray, speed: float, t_start: float) -> list[State]:
    """匀速直线航段，返回两端状态。"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    duration = float(np.linalg.norm(p1 - p0) / max(speed, 1e-3))
    velocity = (p1 - p0) / max(duration, 1e-12)
    return [
        State(t=t_start, pos=p0.copy(), vel=velocity.copy()),
        State(t=t_start + duration, pos=p1.copy(), vel=velocity.copy()),
    ]


def turn_arc(
    p0: np.ndarray,
    heading0: float,
    p1: np.ndarray,
    heading1: float,
    speed: float,
    t_start: float,
    turn_radius: float = 30.0,
    steps: int = 4,
) -> list[State]:
    """简单转弯圆弧段：按转弯半径生成圆弧中间点。

    Parameters
    ----------
    heading0 / heading1: 入 / 出航段航向角（弧度，局部 xy 平面）。
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    center = p0[:2] + turn_radius * np.array([-math.sin(heading0), math.cos(heading0)])
    a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
    a1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
    delta = _angle_diff(a1, a0)

    xy_points = [
        center + turn_radius * np.array(
            [math.cos(a0 + delta * k / steps), math.sin(a0 + delta * k / steps)]
        )
        for k in range(steps + 1)
    ]
    xy_points[-1] = p1[:2]

    total = float(np.linalg.norm(p1[:2] - p0[:2]) / max(speed, 1e-3))
    states: list[State] = []
    prev_xy = p0[:2]
    for k, xy in enumerate(xy_points):
        t = t_start + total * k / steps
        z = p0[2] if p0.shape[0] > 2 else 0.0
        pos = np.array([xy[0], xy[1], z])
        if k == 0:
            vel = np.zeros(3)
        else:
            seg_t = max(total / steps, 1e-12)
            vel = np.append((xy - prev_xy) / seg_t, 0.0)
        states.append(State(t=t, pos=pos, vel=vel))
        prev_xy = xy
    return states


def apply_wind(states: list[State], wind_vec: np.ndarray) -> list[State]:
    """风场扰动：在各状态速度上叠加常值风向量（路径几何不重算）。"""
    wind_vec = np.asarray(wind_vec, dtype=float)
    return [
        State(t=s.t, pos=s.pos.copy(), vel=s.vel + wind_vec)
        for s in states
    ]


def apply_jitter(
    states: list[State], pos_std: float, time_std: float, rng: np.random.Generator
) -> list[State]:
    """定位高斯抖动与时间戳抖动；保证时间戳严格递增。"""
    jittered = [
        State(
            t=max(0.0, s.t + float(rng.normal(0.0, time_std))),
            pos=s.pos + rng.normal(0.0, pos_std, size=3),
            vel=s.vel.copy(),
        )
        for s in states
    ]
    for i in range(1, len(jittered)):
        if jittered[i].t <= jittered[i - 1].t:
            prev = jittered[i - 1]
            jittered[i] = State(t=prev.t + 1e-3, pos=jittered[i].pos, vel=jittered[i].vel)
    return jittered
