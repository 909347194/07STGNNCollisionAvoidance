"""实验统计：bootstrap 置信区间与配对显著性检验。

- 置信区间：按场景重采样的均值 percentile bootstrap（复用数据集统计模块）；
- 配对检验：同一 (seed, density, kind) 场景上基线 vs 处理方法的 Wilcoxon
  符号秩检验；样本不足或差值全为零时给出说明，不伪造 p 值。
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from stgnn_conflict.dataset.stats import bootstrap_ci


def mean_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """均值及其 percentile bootstrap 95% 置信区间。"""
    return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)


def paired_test(
    baseline: list[float] | np.ndarray,
    treatment: list[float] | np.ndarray,
) -> dict[str, object]:
    """配对 Wilcoxon 符号秩检验（基线 vs 处理，按场景一一配对）。

    Returns
    -------
    dict: ``n`` 配对样本数、``mean_diff`` 处理−基线均值差、``p_value``、``note``。
    """
    b = np.asarray(baseline, dtype=float)
    t = np.asarray(treatment, dtype=float)
    n = min(len(b), len(t))
    if n < 2:
        return {"n": n, "mean_diff": float("nan"), "p_value": float("nan"), "note": "样本不足 2 对"}

    b, t = b[:n], t[:n]
    diff = t - b
    if np.allclose(diff, 0.0):
        return {"n": n, "mean_diff": 0.0, "p_value": 1.0, "note": "差值全为零"}

    try:
        res = scipy_stats.wilcoxon(b, t, zero_method="wilcox")
        p_value = float(res.pvalue)
        note = ""
    except ValueError as exc:  # 如差值方差为零等退化情形
        p_value = float("nan")
        note = str(exc)

    return {
        "n": n,
        "mean_diff": float(np.mean(diff)),
        "p_value": p_value,
        "note": note,
    }
