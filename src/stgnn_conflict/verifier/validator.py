"""分层几何验证器。

输入：（GNN 引导或确定性的）航迹组序列；
处理：
1. 对每组尝试组级安全证书；通过则整组豁免，不进入 CPA；
2. 未通过的组，仅对证书未能豁免的航迹对执行连续 CPA；
3. 所有未豁免候选对最终都到达 CPA，运动模型与数值精度假设见配置，
   在此前提下保证零漏检。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.deterministic.certificates import certify_group, certify_pair
from stgnn_conflict.verifier.cpa import continuous_cpa


@dataclass
class VerifyResult:
    """验证阶段统计与冲突结果。"""

    conflicts: set[tuple[int, int]] = field(default_factory=set)
    cpa_calls: int = 0
    certified_groups: int = 0
    certified_pairs: int = 0
    uncertified_pairs: int = 0
    failed_pairs: list[tuple[int, int]] = field(default_factory=list)


def validate_groups(
    groups: list[list[int]],
    traj_map: dict[int, Trajectory],
    config: DetectionConfig,
    candidate_edges: set[tuple[int, int]] | None = None,
    deferred_pairs: list[tuple[int, int]] | None = None,
) -> VerifyResult:
    """逐组认证、细分并对剩余候选对执行 CPA。

    deferred_pairs: 引导拆分后落在子组之间、未被任何细化组覆盖的候选对，
    必须直接进入 CPA 兜底，保证零漏检。
    """
    result = VerifyResult()

    for group in groups:
        cert = certify_group(group, traj_map, config, candidate_edges)
        # 组内通过任一证书的航迹对即被豁免，无论整组是否通过认证
        exempt_pairs = cert.pairs_checked - len(cert.failed_pairs)
        result.certified_pairs += exempt_pairs

        if cert.safe:
            result.certified_groups += 1
            continue

        result.uncertified_pairs += len(cert.failed_pairs)
        result.failed_pairs.extend(cert.failed_pairs)

        for a, b in cert.failed_pairs:
            result.cpa_calls += 1
            cpa = continuous_cpa(traj_map[a], traj_map[b], config.cert.safety_radius, config.cert.time_eps)
            if cpa.conflict:
                result.conflicts.add((min(a, b), max(a, b)))

    for a, b in deferred_pairs or []:
        # 跨子组候选对同样先经过确定性证书认证；未豁免才进入 CPA
        pair_cert = certify_pair(traj_map[a], traj_map[b], config)
        if pair_cert.safe:
            result.certified_pairs += 1
            continue
        result.cpa_calls += 1
        cpa = continuous_cpa(traj_map[a], traj_map[b], config.cert.safety_radius, config.cert.time_eps)
        if cpa.conflict:
            result.conflicts.add((min(a, b), max(a, b)))

    return result
