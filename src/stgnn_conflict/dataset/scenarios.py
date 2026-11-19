"""高冲突潜力任务场景生成。

支持两类场景：
- ``delivery_grid``：城市配送网格——水平 / 垂直航线交叉，叠加汇聚到配送枢纽；
- ``vertiport``：垂直起降点附近汇聚——航线在起降点之间双向汇聚。

轨迹在边界局部米制坐标系内生成，统一接入风场与抖动扰动。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stgnn_conflict.core.trajectory import State, Trajectory
from stgnn_conflict.dataset.boundary import Boundary
from stgnn_conflict.dataset.motion import apply_jitter, apply_wind


@dataclass
class ScenarioParams:
    """场景生成参数。"""

    n_uavs: int
    duration: float = 60.0
    speed_range: tuple[float, float] = (10.0, 25.0)
    altitude_range: tuple[float, float] = (80.0, 150.0)
    wind_std: float = 3.0  # m/s，风速标准差
    pos_jitter_std: float = 1.0  # m，定位抖动
    time_jitter_std: float = 0.05  # s，时间戳抖动


def _random_point(boundary: Boundary, rng: np.random.Generator) -> np.ndarray:
    """在边界内拒绝采样一个二维点；失败时回退到包围盒中心。"""
    lo, hi = _inset_bbox(boundary)
    for _ in range(50):
        point = rng.uniform(lo, hi)
        if boundary.contains(point):
            return point
    return (lo + hi) / 2.0


def _build_trajectory(
    traj_id: int,
    waypoints: list[np.ndarray],
    speed: float,
    params: ScenarioParams,
    rng: np.random.Generator,
) -> Trajectory:
    """由航迹点序列装配轨迹，叠加风场与抖动扰动。"""
    pts = np.array(waypoints, dtype=float)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    seg_time = seg_len / max(speed, 1e-3)
    raw_time = np.concatenate([[0.0], np.cumsum(seg_time)])
    scale = params.duration / max(raw_time[-1], 1e-12)
    times = raw_time * scale

    states: list[State] = []
    for k, (t, pos) in enumerate(zip(times, pts)):
        if k < len(pts) - 1:
            direction = pts[k + 1] - pos
        else:
            direction = pos - pts[k - 1]
        direction /= max(np.linalg.norm(direction), 1e-12)
        vel = direction * speed
        states.append(State(t=float(t), pos=pos.copy(), vel=vel))

    wind = rng.normal(0.0, params.wind_std, size=3)
    wind[2] = 0.0
    states = apply_wind(states, wind)
    states = apply_jitter(states, params.pos_jitter_std, params.time_jitter_std, rng)
    return Trajectory(traj_id=traj_id, states=states)


def _inset_bbox(boundary: Boundary, ratio: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """内缩包围盒，避免航线端点贴边界后被抖动推出界外。"""
    lo, hi = boundary.local_bbox()
    pad = (hi - lo) * ratio
    return lo + pad, hi - pad


def _delivery_grid_routes(boundary: Boundary, rng: np.random.Generator, n_uavs: int) -> list[list[np.ndarray]]:
    """网格交叉航线 + 少量汇聚枢纽航线。"""
    lo, hi = _inset_bbox(boundary)
    routes: list[list[np.ndarray]] = []

    n_horizontal = max(2, n_uavs // 2)
    for _ in range(n_horizontal):
        y = rng.uniform(lo[1], hi[1])
        routes.append([np.array([lo[0], y]), np.array([hi[0], y])])

    n_vertical = n_uavs - n_horizontal
    for _ in range(n_vertical):
        x = rng.uniform(lo[0], hi[0])
        routes.append([np.array([x, lo[1]]), np.array([x, hi[1]])])

    # 约四分之一航线附加汇聚到配送枢纽
    hub = _random_point(boundary, rng)
    for route in routes[: max(1, len(routes) // 4)]:
        route.append(hub)
    return routes


def _vertiport_routes(
    boundary: Boundary, rng: np.random.Generator, n_uavs: int
) -> list[list[np.ndarray]]:
    """起降点双向汇聚航线。"""
    n_ports = max(1, n_uavs // 6)
    ports = [_random_point(boundary, rng) for _ in range(n_ports)]
    routes: list[list[np.ndarray]] = []
    for i in range(n_uavs):
        port = ports[i % len(ports)]
        far = _random_point(boundary, rng)
        if i % 2 == 0:
            routes.append([far, port])
        else:
            routes.append([port, far])
    return routes


def generate_scenario(
    kind: str,
    boundary: Boundary,
    params: ScenarioParams,
    seed: int,
) -> list[Trajectory]:
    """按场景类型生成航迹列表。

    Parameters
    ----------
    kind: ``delivery_grid`` 或 ``vertiport``。
    """
    rng = np.random.default_rng(seed)
    if kind == "delivery_grid":
        routes = _delivery_grid_routes(boundary, rng, params.n_uavs)
    elif kind == "vertiport":
        routes = _vertiport_routes(boundary, rng, params.n_uavs)
    else:
        raise ValueError(f"未知场景类型: {kind}")

    trajectories: list[Trajectory] = []
    for traj_id, route in enumerate(routes):
        speed = rng.uniform(*params.speed_range)
        altitude = rng.uniform(*params.altitude_range)
        waypoints = [np.append(point, altitude) for point in route]
        trajectories.append(_build_trajectory(traj_id, waypoints, speed, params, rng))
    return trajectories
