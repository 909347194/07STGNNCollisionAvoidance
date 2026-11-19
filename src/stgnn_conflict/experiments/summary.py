"""结果汇总：方法级指标表、bootstrap 置信区间与消融对照。

产物
----
- ``{prefix}_summary.csv``：按方法（可选场景类型）聚合的均值与 95% CI；
- ``{prefix}_ablation.csv``：H1/H2 消融在各指标上的配对差值与显著性；
- ``{prefix}_summary.md``：方法级汇总的 Markdown 表（无 tabulate 依赖，手写表体）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stgnn_conflict.experiments.stats import mean_ci, paired_test

# 进入方法级汇总的效率 / 规模指标
EFFICIENCY_METRICS = [
    "candidate_pairs",
    "cpa_calls",
    "certified_pairs",
    "certified_pair_ratio",
    "certified_group_ratio",
    "hard_groups",
    "n_groups",
    "deferred_pairs",
    "total_ms",
    "conflicts",
]

# 消融定义：(标签, 基线方法, 处理方法)
ABLATIONS = [
    ("H1_motion_cert", "bbox_only", "det_group"),
    ("H2_gnn_guide", "det_group", "gnn_guided"),
]

# 消融对照关注的指标
CONTRAST_METRICS = [
    "cpa_calls",
    "total_ms",
    "certified_group_ratio",
    "certified_pair_ratio",
]

_PAIR_KEYS = ["seed", "density", "kind"]


def _method_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """按分组列聚合方法级指标：均值 + bootstrap CI，漏检取 worst-case。"""
    rows: list[dict] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_tuple))

        for col in EFFICIENCY_METRICS:
            if col not in sub.columns:
                continue
            values = sub[col].dropna().to_numpy(dtype=float)
            ci = mean_ci(values) if values.size else {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            row[f"{col}_mean"] = ci["mean"]
            row[f"{col}_ci_low"] = ci["ci_low"]
            row[f"{col}_ci_high"] = ci["ci_high"]

        if "miss_rate" in sub.columns:
            row["miss_rate_worst"] = float(sub["miss_rate"].max())
            row["miss_violations"] = int((sub["miss_rate"] > 0).sum())
        if "recall" in sub.columns:
            row["recall_worst"] = float(sub["recall"].min())
        rows.append(row)

    return pd.DataFrame(rows)


def _align_methods(
    df: pd.DataFrame, baseline: str, treatment: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按配对键对齐同一批场景上的两种方法结果。"""
    base = df[df["method"] == baseline].set_index(_PAIR_KEYS)
    treat = df[df["method"] == treatment].set_index(_PAIR_KEYS)
    common = base.index.intersection(treat.index)
    return base.loc[common], treat.loc[common]


def ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """H1/H2 消融：配对均值差、Wilcoxon p 值与双侧均值。"""
    rows: list[dict] = []
    for tag, base_name, treat_name in ABLATIONS:
        base, treat = _align_methods(df, base_name, treat_name)
        if base.empty:
            continue
        for metric in CONTRAST_METRICS:
            if metric not in base.columns:
                continue
            test = paired_test(base[metric].to_numpy(), treat[metric].to_numpy())
            rows.append(
                {
                    "ablation": tag,
                    "baseline": base_name,
                    "treatment": treat_name,
                    "metric": metric,
                    "baseline_mean": float(base[metric].mean()),
                    "treatment_mean": float(treat[metric].mean()),
                    **test,
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """手写 Markdown 表体，避免 tabulate 依赖。"""
    if df.empty:
        return "（无数据）"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt) if not np.isnan(v) else "—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def export_tables(
    df: pd.DataFrame, out_dir: str | Path, prefix: str = "benchmark"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """落盘汇总表与消融表，返回 (方法级汇总, 消融表)。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = _method_summary(df, ["method"])
    ablation = ablation_table(df)

    summary.to_csv(out / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(out / f"{prefix}_ablation.csv", index=False, encoding="utf-8-sig")

    # 场景类型拆分汇总（delivery_grid / vertiport 分别报告）
    if "kind" in df.columns:
        summary_by_kind = _method_summary(df, ["method", "kind"])
        summary_by_kind.to_csv(
            out / f"{prefix}_summary_by_kind.csv", index=False, encoding="utf-8-sig"
        )

    md = ["# 方法级汇总（均值 [95% CI]）", "", _markdown_table(summary)]
    (out / f"{prefix}_summary.md").write_text("\n".join(md), encoding="utf-8")
    return summary, ablation
