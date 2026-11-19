"""结果出图：从实验 CSV 绘制论文用图。

matplotlib 为可选依赖（``plots`` extra）；未安装时各绘图函数安全跳过并返回
False，不影响确定性管线与表格产物。输出 PNG 到 ``figures`` 目录。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_METHOD_STYLES = {
    "rtree_index": ("#7f7f7f", "s"),
    "grid_index": ("#bcbd22", "v"),
    "bbox_only": ("#1f77b4", "o"),
    "det_group": ("#2ca02c", "D"),
    "gnn_guided": ("#d62728", "^"),
}

_DENSITY_METRICS = [
    ("cpa_calls", "CPA calls"),
    ("total_ms", "End-to-end runtime (ms)"),
    ("certified_group_ratio", "Group certification ratio"),
    ("certified_pair_ratio", "Pair certification ratio"),
]


def _load_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _style(method: str) -> tuple[str, str]:
    return _METHOD_STYLES.get(method, ("#333333", "o"))


def plot_density_curves(df: pd.DataFrame, out_dir: str | Path) -> bool:
    """指标 vs 航迹密度折线，分方法。"""
    plt = _load_plt()
    if plt is None or df.empty:
        return False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for metric, ylabel in _DENSITY_METRICS:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, sub in df.groupby("method"):
            agg = sub.groupby("density")[metric].mean()
            color, marker = _style(method)
            ax.plot(agg.index, agg.values, marker=marker, color=color, label=method)
        ax.set_xlabel("UAV density N")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"density_{metric}.png", dpi=150)
        plt.close(fig)
    return True


def plot_ablation(ablation: pd.DataFrame, out_dir: str | Path) -> bool:
    """消融配对均值差条形图（H1/H2 × 指标）。"""
    plt = _load_plt()
    if plt is None or ablation.empty:
        return False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = list(ablation["metric"].unique())
    ablations = list(ablation["ablation"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(metrics))
    width = 0.35
    for i, tag in enumerate(ablations):
        means = [
            ablation[(ablation["ablation"] == tag) & (ablation["metric"] == m)]["mean_diff"].mean()
            for m in metrics
        ]
        ax.bar([xi + (i - 0.5) * width for xi in x], means, width, label=tag)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=20, fontsize=8)
    ax.set_ylabel("Paired mean diff (treatment − baseline)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ablation_mean_diff.png", dpi=150)
    plt.close(fig)
    return True


def plot_hardcore(df: pd.DataFrame, out_dir: str | Path) -> bool:
    """困难核规模（连通域数 / 困难组数）vs 密度。"""
    plt = _load_plt()
    if plt is None or df.empty:
        return False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for metric, ylabel in [("n_groups", "Connected components"), ("hard_groups", "Hard-core groups")]:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, sub in df.groupby("method"):
            agg = sub.groupby("density")[metric].mean()
            color, marker = _style(method)
            ax.plot(agg.index, agg.values, marker=marker, color=color, label=method)
        ax.set_xlabel("UAV density N")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"hardcore_{metric}.png", dpi=150)
        plt.close(fig)
    return True


def plot_ood(df: pd.DataFrame, out_dir: str | Path) -> bool:
    """内分布 vs OOD 分组柱状对比（CPA 调用次数与组级覆盖率）。"""
    plt = _load_plt()
    if plt is None or df.empty or "ood_tag" not in df.columns:
        return False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for metric, ylabel in [("cpa_calls", "CPA calls"), ("certified_group_ratio", "Group certification ratio")]:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        methods = list(df["method"].unique())
        tags = list(df["ood_tag"].unique())
        x = range(len(methods))
        width = 0.8 / max(len(tags), 1)
        for i, tag in enumerate(tags):
            means = [
                df[(df["method"] == m) & (df["ood_tag"] == tag)][metric].mean()
                for m in methods
            ]
            ax.bar([xi + (i - 0.5) * width for xi in x], means, width, label=tag)
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=20, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"ood_{metric}.png", dpi=150)
        plt.close(fig)
    return True


def plot_training_curve(history_path: str | Path, out_dir: str | Path) -> bool:
    """训练曲线：损失 / MAE 与场景内 Spearman 随轮次变化。"""
    plt = _load_plt()
    path = Path(history_path)
    if plt is None or not path.exists():
        return False
    history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
    if not history:
        return False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train_loss", color="#1f77b4")
    ax1.plot(epochs, [h["val_mae"] for h in history], label="val_mae", color="#2ca02c")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss / MAE")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_spearman"] for h in history], label="val_spearman", color="#d62728")
    ax2.set_ylabel("Spearman")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", dpi=150)
    plt.close(fig)
    return True


def plot_all(
    df: pd.DataFrame,
    ablation: pd.DataFrame,
    figures_dir: str | Path,
    history_path: str | Path | None = None,
) -> list[str]:
    """生成全部可出的图，返回成功输出的图名列表。"""
    produced: list[str] = []
    if plot_density_curves(df, figures_dir):
        produced.append("density_curves")
    if plot_ablation(ablation, figures_dir):
        produced.append("ablation")
    if plot_hardcore(df, figures_dir):
        produced.append("hardcore")
    if plot_ood(df, figures_dir):
        produced.append("ood")
    if history_path and plot_training_curve(history_path, figures_dir):
        produced.append("training_curve")
    return produced
