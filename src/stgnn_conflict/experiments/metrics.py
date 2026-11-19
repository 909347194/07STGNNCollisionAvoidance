"""实验评价指标。"""

from __future__ import annotations


def evaluate(detected: set[tuple[int, int]], truth: set[tuple[int, int]]) -> dict[str, float | int]:
    """对照真值集合计算检测质量指标。

    返回
    ----
    true_positive / false_positive / false_negative: 计数；
    miss_rate: 漏检率 = FN / |真值冲突对|；
    precision / recall: 精确率与召回率。
    """
    detected, truth = set(detected), set(truth)
    tp = len(detected & truth)
    fp = len(detected - truth)
    fn = len(truth - detected)
    n_truth = len(truth)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "miss_rate": fn / n_truth if n_truth else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 1.0,
        "recall": tp / n_truth if n_truth > 0 else 1.0,
    }
