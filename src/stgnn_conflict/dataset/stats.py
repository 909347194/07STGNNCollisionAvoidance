"""统计纪律：bootstrap 置信区间与零漏检硬断言。"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """按场景重采样给出均值的 percentile bootstrap 置信区间。

    Parameters
    ----------
    values: 每个场景一个效率指标取值。
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    boot_means = [
        float(np.mean(rng.choice(values, size=values.size, replace=True)))
        for _ in range(n_boot)
    ]
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    }


def assert_zero_miss(
    detected: set[tuple[int, int]],
    truth: set[tuple[int, int]],
    skipped_groups: list[str] | None = None,
    group_certs: dict[str, object] | None = None,
) -> None:
    """零漏检硬断言。

    - 检测结果相对真值不得存在漏检对；
    - 被跳过（豁免）的组必须附带有通过的安全证书，否则断言失败。
    """
    detected, truth = set(detected), set(truth)
    false_negatives = truth - detected
    assert not false_negatives, f"漏检 {len(false_negatives)} 对: {sorted(false_negatives)[:5]}"

    if skipped_groups is not None:
        certs = group_certs or {}
        for group_id in skipped_groups:
            cert = certs.get(group_id)
            assert cert is not None and getattr(cert, "safe", False), (
                f"被跳过的组 {group_id} 未附通过的安全证书"
            )
