"""分层确定性安全证书。

证书链（由廉价到昂贵）：
1. 时间分离证书：时间区间不重叠 ⇒ 不可能冲突；
2. 空间包围盒分离证书：膨胀后的包围盒最小距离大于安全半径 ⇒ 不可能冲突；
3. 相对运动距离下界证书：逐航段对求解匀速直线运动的最近接近距离下界，
   若所有重叠段对的下界均不小于安全半径 ⇒ 不可能冲突。

任何证书通过即可整体跳过该航迹对的精确 CPA 检测；
全部证书均无法证明安全时，返回 UNCERTAIN，交由下游细分与 CPA。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.utils.geometry import closest_approach_linear, overlap_interval


@dataclass
class CertResult:
    """单证书判定结果。"""

    safe: bool
    name: str
    bound: float = -1.0  # 证明的距离下界；未通过时为 -1


def time_separation(a: Trajectory, b: Trajectory, eps: float) -> CertResult:
    """时间分离证书：两航迹时间区间不重叠。"""
    if overlap_interval(a.t0, a.t1, b.t0, b.t1, eps) is None:
        return CertResult(safe=True, name="time_separation")
    return CertResult(safe=False, name="time_separation")


def bbox_separation(
    a: Trajectory, b: Trajectory, radius: float, margin: float, eps: float
) -> CertResult:
    """空间包围盒分离证书：包围盒最小距离 > 安全半径。"""
    dist = a.bbox.min_distance(b.bbox)
    if dist - margin > radius + eps:
        return CertResult(safe=True, name="bbox_separation", bound=dist)
    return CertResult(safe=False, name="bbox_separation", bound=dist)


def motion_lower_bound(
    a: Trajectory, b: Trajectory, radius: float, eps: float
) -> CertResult:
    """相对运动距离下界证书：逐段对解析求解最近接近距离下界。"""
    min_dist = float("inf")
    for sa in a.segments:
        for sb in b.segments:
            overlap = overlap_interval(sa.t0, sa.t1, sb.t0, sb.t1)
            if overlap is None:
                continue
            t0, t1 = overlap
            pa0 = sa.position_at(t0)
            pb0 = sb.position_at(t0)
            dist, _ = closest_approach_linear(pa0, sa.velocity, pb0, sb.velocity, t0, t1)
            min_dist = min(min_dist, dist)
    if min_dist == float("inf"):
        return CertResult(safe=False, name="motion_lower_bound")
    if min_dist >= radius - eps:
        return CertResult(safe=True, name="motion_lower_bound", bound=min_dist)
    return CertResult(safe=False, name="motion_lower_bound", bound=min_dist)


def certify_pair(a: Trajectory, b: Trajectory, config: DetectionConfig) -> CertResult:
    """按证书链依次判定，返回首个通过的证书；均未通过则 UNCERTAIN。

    二级运动下界证书可通过 ``config.cert.use_motion_cert`` 关闭，
    用于消融实验（仅保留时间 / 包围盒一级证书）。
    """
    cert = config.cert
    result = time_separation(a, b, cert.time_eps)
    if result.safe:
        return result
    result = bbox_separation(a, b, cert.safety_radius, cert.bbox_margin, cert.space_eps)
    if result.safe:
        return result
    if not cert.use_motion_cert:
        return result
    return motion_lower_bound(a, b, cert.safety_radius, cert.space_eps)


@dataclass
class GroupCertResult:
    """组级证书判定结果。"""

    safe: bool
    failed_pairs: list[tuple[int, int]] = field(default_factory=list)
    passed_by: dict[str, int] = field(default_factory=dict)
    pairs_checked: int = 0


def certify_group(
    group: list[int],
    traj_map: dict[int, Trajectory],
    config: DetectionConfig,
    candidate_edges: set[tuple[int, int]] | None = None,
) -> GroupCertResult:
    """尝试为整组航迹颁发安全证书。

    组内所有候选航迹对均通过证书 ⇒ 整组可跳过后续检测；
    任一候选对无法认证 ⇒ 组不安全，记录未能认证的航迹对。
    非候选对已被索引阶段的时空过滤排除，不纳入证书统计。
    """
    failed: list[tuple[int, int]] = []
    passed: dict[str, int] = {}
    pairs_checked = 0

    for i, ta in enumerate(group):
        for tb in group[i + 1 :]:
            pair = (min(ta, tb), max(ta, tb))
            if candidate_edges is not None and pair not in candidate_edges:
                continue
            pairs_checked += 1
            result = certify_pair(traj_map[ta], traj_map[tb], config)
            if result.safe:
                passed[result.name] = passed.get(result.name, 0) + 1
            else:
                failed.append(pair)

    return GroupCertResult(
        safe=not failed,
        failed_pairs=failed,
        passed_by=passed,
        pairs_checked=pairs_checked,
    )
