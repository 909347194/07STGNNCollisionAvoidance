"""训练与推理闭环测试（torch 可选，缺失时跳过）。"""

import numpy as np
import pytest

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.experiments.baselines import ground_truth, run_gnn_guided
from stgnn_conflict.experiments.metrics import evaluate
from stgnn_conflict.experiments.scenario import generate_scenario

torch = pytest.importorskip("torch")

from stgnn_conflict.dataset.samples import SUBGROUP_FEATURE_DIM
from stgnn_conflict.training.train import (
    PriorityRegressor,
    TrainConfig,
    train_priority_regressor,
)


def _tiny_data(n: int = 8, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 25)).astype(np.float32)
    y = x[:, 0] * 3.0 + rng.normal(scale=0.1, size=n).astype(np.float32)
    sids = ["s1"] * (n // 2) + ["s2"] * (n - n // 2)
    return x, y, sids


def test_train_and_checkpoint_roundtrip(tmp_path):
    x, y, sids = _tiny_data()
    x2, y2, sids2 = _tiny_data(seed=2)
    cfg = TrainConfig(hidden_dim=16, num_layers=1, epochs=5, patience=5, device="cpu")
    model, result = train_priority_regressor(
        x, y, sids, x2, y2, sids2, config=cfg, out_dir=tmp_path, verbose=False
    )
    assert "best_val_spearman" in result
    assert (tmp_path / "priority_regressor.pt").exists()
    assert (tmp_path / "train_history.json").exists()

    preds = model.infer(torch.tensor(x))
    assert preds.shape == (len(x),)

    from stgnn_conflict.training.train import load_priority_regressor

    loaded, ckpt = load_priority_regressor(tmp_path / "priority_regressor.pt")
    preds2 = loaded.infer(torch.tensor(x))
    assert np.allclose(preds, preds2, atol=1e-5)

    # 标准化统计量随 checkpoint 落盘并在推理侧复用
    assert loaded.standardizer is not None
    assert loaded.standardizer.mean.shape == (x.shape[1],)
    assert loaded.standardizer.std.shape == (x.shape[1],)
    assert "feature_mean" in ckpt and "feature_std" in ckpt


def test_priority_model_inference_zero_miss():
    trajectories = generate_scenario(n_uavs=20, seed=5)
    config = DetectionConfig()
    truth = ground_truth(trajectories, config)

    regressor = PriorityRegressor(in_dim=SUBGROUP_FEATURE_DIM, hidden=16, num_layers=1)
    outcome = run_gnn_guided(trajectories, config, priority_model=regressor)
    evaluation = evaluate(outcome.conflicts, truth)

    assert evaluation["miss_rate"] == 0.0
