"""实验汇总、消融对照与统计辅助的单元测试。"""

import numpy as np
import pandas as pd

from stgnn_conflict.experiments.stats import paired_test
from stgnn_conflict.experiments.summary import ablation_table, export_tables


def _synthetic_df() -> pd.DataFrame:
    rows = []
    methods = ["bbox_only", "det_group", "gnn_guided"]
    for seed in (1, 2, 3):
        for density in (20, 40):
            for method in methods:
                rows.append(
                    {
                        "seed": seed,
                        "density": density,
                        "kind": "delivery_grid",
                        "method": method,
                        "candidate_pairs": 100,
                        "cpa_calls": {"bbox_only": 90, "det_group": 20, "gnn_guided": 15}[method],
                        "certified_pairs": {"bbox_only": 10, "det_group": 80, "gnn_guided": 85}[method],
                        "certified_pair_ratio": 0.1,
                        "certified_group_ratio": {"bbox_only": 0.0, "det_group": 0.5, "gnn_guided": 0.6}[method],
                        "hard_groups": 5,
                        "n_groups": 6,
                        "deferred_pairs": 2,
                        "total_ms": {"bbox_only": 50.0, "det_group": 40.0, "gnn_guided": 38.0}[method],
                        "conflicts": 3,
                        "miss_rate": 0.0,
                        "recall": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_includes_ci_and_worst_case(tmp_path):
    df = _synthetic_df()
    summary, ablation = export_tables(df, tmp_path, prefix="smoke")

    assert (tmp_path / "smoke_summary.csv").exists()
    assert (tmp_path / "smoke_ablation.csv").exists()
    assert (tmp_path / "smoke_summary.md").exists()
    assert (tmp_path / "smoke_summary_by_kind.csv").exists()

    row = summary[summary["method"] == "det_group"].iloc[0]
    assert "cpa_calls_mean" in summary.columns
    assert "cpa_calls_ci_low" in summary.columns
    assert "cpa_calls_ci_high" in summary.columns
    assert row["miss_rate_worst"] == 0.0
    assert row["miss_violations"] == 0
    assert row["recall_worst"] == 1.0

    # 消融表含 H1 / H2 两条标签、四个对照指标
    assert set(ablation["ablation"]) == {"H1_motion_cert", "H2_gnn_guide"}
    assert len(ablation) == 2 * 4


def test_ablation_paired_alignment():
    df = _synthetic_df()
    ablation = ablation_table(df)
    h2 = ablation[(ablation["ablation"] == "H2_gnn_guide") & (ablation["metric"] == "cpa_calls")].iloc[0]
    # gnn_guided(15) 相对 det_group(20) 配对均值差为负（更少 CPA 调用）
    assert h2["mean_diff"] < 0
    assert h2["n"] == 6


def test_paired_test_degenerate_cases():
    assert paired_test([1.0], [2.0])["note"].startswith("样本不足")
    zero = paired_test([1.0, 1.0], [1.0, 1.0])
    assert zero["p_value"] == 1.0
    assert zero["mean_diff"] == 0.0

    res = paired_test([3.0, 2.0, 1.0], [1.0, 1.0, 1.0])
    assert res["mean_diff"] < 0
    assert np.isfinite(res["p_value"])
