"""数据划分：按场景划分，禁止按航迹对泄漏。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_scenario(
    scenario_ids: list[str], train_ratio: float = 0.7, seed: int = 0
) -> tuple[list[str], list[str]]:
    """场景级划分：同一场景的所有决策点必须落在同一子集。"""
    ids = list(scenario_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    k = round(len(ids) * train_ratio)
    return ids[:k], ids[k:]


def ood_density_mask(df: pd.DataFrame, train_lo: float, train_hi: float) -> pd.Series:
    """密度 OOD：场景航迹数落在训练密度区间之外。"""
    return (df["n_uavs"] < train_lo) | (df["n_uavs"] > train_hi)


def ood_speed_mask(df: pd.DataFrame, train_lo: float, train_hi: float) -> pd.Series:
    """速度分布 OOD：场景平均速度落在训练速度区间之外。"""
    return (df["mean_speed"] < train_lo) | (df["mean_speed"] > train_hi)


def quota_balance_by_level(
    decision_points: pd.DataFrame, quota: int | None, seed: int = 0
) -> pd.DataFrame:
    """训练集按认证层级做有限配额均衡；测试集不调用本函数，保持自然分布。"""
    if quota is None or decision_points.empty:
        return decision_points
    parts: list[pd.DataFrame] = []
    for _, sub in decision_points.groupby("level"):
        parts.append(sub.sample(n=min(quota, len(sub)), random_state=seed))
    return pd.concat(parts).sort_index()
