"""场景级划分、OOD 切片与统计纪律测试。"""

from types import SimpleNamespace

import pandas as pd
import pytest

from stgnn_conflict.dataset.builder import assign_splits
from stgnn_conflict.dataset.split import (
    ood_density_mask,
    ood_speed_mask,
    quota_balance_by_level,
    split_by_scenario,
)
from stgnn_conflict.dataset.stats import assert_zero_miss, bootstrap_ci


def test_split_scenario_level_disjoint_and_complete():
    ids = [f"s{i}" for i in range(10)]
    train, test = split_by_scenario(ids, train_ratio=0.7, seed=0)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(ids)


def test_ood_masks():
    df = pd.DataFrame({"n_uavs": [10, 30, 70], "mean_speed": [8.0, 17.0, 30.0]})
    assert ood_density_mask(df, 20.0, 50.0).tolist() == [True, False, True]
    assert ood_speed_mask(df, 10.0, 25.0).tolist() == [True, False, True]


def test_assign_splits_forces_ood_to_test():
    manifest = pd.DataFrame(
        {
            "scenario_id": ["id1", "id2", "id3", "ood_d", "ood_v"],
            "kind": ["delivery_grid"] * 5,
            "n_uavs": [30, 30, 30, 100, 30],
            "mean_speed": [17.0, 18.0, 16.0, 17.0, 32.0],
        }
    )
    decision_points = pd.DataFrame({"scenario_id": manifest["scenario_id"]})
    out_manifest, _ = assign_splits(
        manifest, decision_points, train_ratio=0.8, seed=0,
        density_train=(20.0, 60.0), speed_train=(10.0, 25.0),
    )
    splits = dict(zip(out_manifest["scenario_id"], out_manifest["split"]))
    assert splits["ood_d"] == "test"
    assert splits["ood_v"] == "test"
    assert splits["id1"] in {"train", "test"}


def test_bootstrap_ci_brackets_mean():
    values = [10.0, 12.0, 11.0, 13.0, 9.0]
    ci = bootstrap_ci(values, n_boot=100, seed=0)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_quota_balance_caps_per_level():
    df = pd.DataFrame({"level": ["L1"] * 5 + ["CPA"] * 3, "v": list(range(8))})
    out = quota_balance_by_level(df, quota=2, seed=0)
    counts = out.groupby("level").size()
    assert counts["L1"] == 2
    assert counts["CPA"] == 2


def test_zero_miss_passes_on_equal():
    truth = {(0, 1), (2, 3)}
    assert_zero_miss(truth, truth)


def test_zero_miss_raises_on_miss():
    truth = {(0, 1), (2, 3)}
    detected = {(0, 1)}
    with pytest.raises(AssertionError, match="漏检"):
        assert_zero_miss(detected, truth)


def test_zero_miss_requires_certificate_for_skipped_groups():
    truth: set[tuple[int, int]] = set()
    certs = {"g1": SimpleNamespace(safe=True)}
    assert_zero_miss(truth, truth, skipped_groups=["g1"], group_certs=certs)

    with pytest.raises(AssertionError, match="未附通过"):
        assert_zero_miss(truth, truth, skipped_groups=["g2"], group_certs=certs)
