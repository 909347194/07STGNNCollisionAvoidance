"""高密度无人机随机场景生成。

场景模型：每条航迹由若干随机航迹点连成分段直线，段内匀速；
时间轴按航段距离与随机速度分配，并整体缩放到指定时长。
"""

from __future__ import annotations

import numpy as np

from stgnn_conflict.core.trajectory import State, Trajectory


def generate_scenario(
    n_uavs: int,
    bounds: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 1000.0, 1000.0, 300.0),
    duration: float = 60.0,
    n_waypoints: int = 4,
    speed_range: tuple[float, float] = (10.0, 25.0),
    seed: int = 42,
) -> list[Trajectory]:
    """生成随机航迹场景。

    Parameters
    ----------
    n_uavs: 航迹数量（密度由该参数调节）。
    bounds: 场景包围盒 (xmin, ymin, zmin, xmax, ymax, zmax)。
    duration: 场景总时长 s。
    n_waypoints: 每条航迹的航迹点数。
    speed_range: 段内速度采样区间 m/s。
    seed: 随机种子。
    """
    rng = np.random.default_rng(seed)
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    low = np.array([xmin, ymin, zmin])
    high = np.array([xmax, ymax, zmax])

    trajectories: list[Trajectory] = []
    for traj_id in range(n_uavs):
        waypoints = rng.uniform(low=low, high=high, size=(n_waypoints, 3))

        seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        speeds = rng.uniform(speed_range[0], speed_range[1], size=len(seg_lengths))
        speeds = np.maximum(speeds, 1e-3)
        seg_times = seg_lengths / speeds
        raw_times = np.concatenate([[0.0], np.cumsum(seg_times)])
        scale = duration / raw_times[-1]
        times = raw_times * scale

        states: list[State] = []
        for k, (t, pos) in enumerate(zip(times, waypoints)):
            if k < len(waypoints) - 1:
                vel = (waypoints[k + 1] - pos) / max(seg_times[k] * scale, 1e-12)
            else:
                vel = (pos - waypoints[k - 1]) / max(seg_times[-1] * scale, 1e-12)
            states.append(State(t=float(t), pos=pos.astype(float), vel=vel.astype(float)))

        trajectories.append(Trajectory(traj_id=traj_id, states=states))

    return trajectories
