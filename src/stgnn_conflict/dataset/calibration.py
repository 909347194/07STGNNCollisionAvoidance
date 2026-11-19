"""证书敏感参数标定与敏感性分析。

只标定对安全证书影响最大的 4 类参数：
速度分布、最小安全间隔（安全半径）、轨迹时长、空间密度。

当前数值为粗标定占位区间；正式实验应优先采用公开 OpenSky 数据或
已有 UTM 试飞数据替换，并给出实测置信区间。置信区间字段
``ci_95`` 在拿到实测数据前不得解读为统计结论。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.dataset.boundary import Boundary
from stgnn_conflict.dataset.scenarios import ScenarioParams, generate_scenario
from stgnn_conflict.pipeline.detector import ConflictDetector


@dataclass(frozen=True)
class CalibrationRange:
    """单参数标定区间。"""

    name: str
    low: float
    median: float
    high: float
    unit: str
    ci_95: tuple[float, float]  # 中位数 95% 置信区间（待实测校准）


# 粗标定默认区间（占位，待 OpenSky / UTM 试飞数据替换）
DEFAULT_RANGES: dict[str, CalibrationRange] = {
    "speed": CalibrationRange("speed", 8.0, 15.0, 25.0, "m/s", (12.0, 18.0)),
    "safety_radius": CalibrationRange("safety_radius", 20.0, 30.0, 45.0, "m", (26.0, 34.0)),
    "duration": CalibrationRange("duration", 30.0, 60.0, 120.0, "s", (50.0, 70.0)),
    "density": CalibrationRange("density", 20.0, 40.0, 80.0, "uav/km^2", (30.0, 50.0)),
}


def _hard_core_stats(trajectories: list, config: DetectionConfig) -> dict[str, float]:
    """跑纯确定性管线，提取困难核规模相关统计。"""
    outcome = ConflictDetector(config, use_gnn=False).detect(trajectories)
    m = outcome.metrics
    return {
        "candidate_pairs": float(m["candidate_pairs"]),
        "hard_groups": float(m["hard_groups"]),
        "cpa_calls": float(m["cpa_calls"]),
        "certified_pairs": float(m["certified_pairs"]),
    }


def sensitivity_sweep(
    boundary: Boundary,
    base_params: ScenarioParams,
    config: DetectionConfig,
    repeats: int = 3,
    base_seed: int = 0,
) -> pd.DataFrame:
    """逐参数在低 / 中 / 高水平上扫描，报告困难核规模波动范围。

    Returns
    -------
    DataFrame：每行一个 (参数, 水平, 取值)，附候选对数 / 困难核组数 /
    CPA 次数的均值、最小值与最大值。
    """
    rows: list[dict[str, float | str]] = []

    for param, cal in DEFAULT_RANGES.items():
        for level, value in zip(("low", "median", "high"), (cal.low, cal.median, cal.high)):
            stats: list[dict[str, float]] = []
            for rep in range(repeats):
                params = ScenarioParams(
                    n_uavs=base_params.n_uavs,
                    duration=base_params.duration,
                    speed_range=base_params.speed_range,
                )
                _apply_param(params, config, param, value)
                trajectories = generate_scenario("delivery_grid", boundary, params, base_seed + rep)
                stats.append(_hard_core_stats(trajectories, config))

            agg = _aggregate(stats)
            rows.append(
                {
                    "param": param,
                    "level": level,
                    "value": value,
                    "unit": cal.unit,
                    **agg,
                }
            )

    return pd.DataFrame(rows)


def _apply_param(
    params: ScenarioParams, config: DetectionConfig, param: str, value: float
) -> None:
    """把扫描参数写入场景 / 管线配置。"""
    if param == "speed":
        params.speed_range = (max(2.0, value * 0.7), value * 1.3)
    elif param == "safety_radius":
        config.cert.safety_radius = value
    elif param == "duration":
        params.duration = value
    elif param == "density":
        params.n_uavs = round(value)


def _aggregate(stats: list[dict[str, float]]) -> dict[str, float]:
    keys = stats[0].keys()
    agg: dict[str, float] = {}
    for key in keys:
        values = [s[key] for s in stats]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_min"] = float(np.min(values))
        agg[f"{key}_max"] = float(np.max(values))
    return agg
