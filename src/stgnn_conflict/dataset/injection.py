"""冲突注入（简化版）。

注入作为可调参数：按候选对数的指定比例随机抽取航迹对，
把其中一方的航迹点向另一方中点靠拢，人为制造接近遭遇。
注入比例与注入对在场景元数据中单独记录，与自然场景分开报告。
"""

from __future__ import annotations

import numpy as np

from stgnn_conflict.core.trajectory import State, Trajectory


def inject_conflicts(
    trajectories: list[Trajectory], ratio: float, rng: np.random.Generator
) -> tuple[list[Trajectory], list[tuple[int, int]], float]:
    """按比例注入人为冲突。

    Returns
    -------
    (trajectories, injected_pairs, injection_ratio)
        调整后的航迹列表、注入航迹对列表、实际注入比例。
    """
    n = len(trajectories)
    n_pairs = n * (n - 1) // 2
    n_inject = round(ratio * n_pairs)
    injected_pairs: list[tuple[int, int]] = []

    if n_inject <= 0:
        return list(trajectories), injected_pairs, 0.0

    adjusted = {t.traj_id: t for t in trajectories}
    for _ in range(n_inject):
        i, j = rng.choice(n, size=2, replace=False)
        ti, tj = trajectories[int(i)], trajectories[int(j)]
        target = tj.position_at((tj.t0 + tj.t1) / 2.0)
        adjusted[ti.traj_id] = _nudge_toward(ti, target)
        injected_pairs.append((min(ti.traj_id, tj.traj_id), max(ti.traj_id, tj.traj_id)))

    trajectories_out = [adjusted[t.traj_id] for t in trajectories]
    injection_ratio = len(injected_pairs) / n_pairs if n_pairs else 0.0
    return trajectories_out, injected_pairs, injection_ratio


def _nudge_toward(traj: Trajectory, target: np.ndarray, pull: float = 0.3) -> Trajectory:
    """把航迹各状态位置向目标点拉拽 ``pull`` 比例，速度保持不变。"""
    states = [
        State(t=s.t, pos=s.pos + (target - s.pos) * pull, vel=s.vel.copy())
        for s in traj.states
    ]
    return Trajectory(traj_id=traj.traj_id, states=states)
