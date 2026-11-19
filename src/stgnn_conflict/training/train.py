"""ST-GNN 子组优先级回归训练。

监督信号：决策点（细化子组）的认证净节省 ``benefit``；
模型输入：与 ``build_graph`` 同口径的子组聚合特征（MLP 回归头）；
验证指标：场景内 Spearman 秩相关（排序质量）与 MAE。

训练/验证按场景划分（与数据集划分纪律一致），禁止按决策点混洗跨场景泄漏。
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    """训练超参数。"""

    hidden_dim: int = 64
    num_layers: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 64
    patience: int = 15  # 早停轮数（按验证 Spearman）
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


@dataclass
class FeatureStandardizer:
    """按训练集统计量做 z-score 标准化；统计量随 checkpoint 落盘，推理复用。"""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> FeatureStandardizer:
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std = np.where(std < eps, eps, std)
        return cls(mean=mean, std=std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std


class PriorityRegressor(nn.Module):
    """子组优先级回归 MLP：输入子组聚合特征，输出标量优先级。"""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_layers: int,
        standardizer: FeatureStandardizer | None = None,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * num_layers
        self.backbone = nn.Sequential()
        for i, (d_in, d_out) in enumerate(itertools.pairwise(dims)):
            self.backbone.add_module(f"linear_{i}", nn.Linear(d_in, d_out))
            self.backbone.add_module(f"relu_{i}", nn.ReLU())
        self.head = nn.Linear(hidden, 1)
        self.standardizer = standardizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.standardizer is not None:
            x = self.standardizer.transform(x)
        return self.head(self.backbone(x)).squeeze(-1)

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        return self.forward(x).cpu().numpy()


def _spearman_by_group(
    preds: np.ndarray, targets: np.ndarray, scenario_ids: list[str]
) -> float:
    """场景内 Spearman 秩相关均值（同场景样本数 ≥ 2 才计算）。"""
    df_map: dict[str, list[int]] = {}
    for idx, sid in enumerate(scenario_ids):
        df_map.setdefault(sid, []).append(idx)

    corrs: list[float] = []
    for idxs in df_map.values():
        if len(idxs) < 2:
            continue
        p = preds[idxs]
        t = targets[idxs]
        if np.allclose(p, p[0]) or np.allclose(t, t[0]):
            continue
        corr = spearmanr(p, t).statistic
        if np.isfinite(corr):
            corrs.append(float(corr))
    return float(np.mean(corrs)) if corrs else float("nan")


def train_priority_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sid_train: list[str],
    x_val: np.ndarray,
    y_val: np.ndarray,
    sid_val: list[str],
    config: TrainConfig | None = None,
    out_dir: str | Path = "models",
    verbose: bool = True,
) -> tuple[PriorityRegressor, dict]:
    """训练子组优先级回归器，返回 (模型, 训练历史)。

    模型与最优指标落盘 ``out_dir/priority_regressor.pt`` / ``train_history.json``。
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    # 标准化统计量仅由训练集估计，验证 / 测试 / 推理复用同一组统计量
    standardizer = FeatureStandardizer.fit(x_train)
    x_train_s = standardizer.transform(torch.tensor(x_train, dtype=torch.float32)).numpy()
    x_val_s = standardizer.transform(torch.tensor(x_val, dtype=torch.float32)).numpy()
    model = PriorityRegressor(
        x_train.shape[1], cfg.hidden_dim, cfg.num_layers, standardizer
    ).to(device)

    train_ds = TensorDataset(
        torch.tensor(x_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.HuberLoss(delta=5.0)  # benefit 量级差异大，Huber 抗离群

    x_val_t = torch.tensor(x_val_s, dtype=torch.float32).to(device)
    y_val = np.asarray(y_val, dtype=np.float32)

    best_spearman = -np.inf
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    wait = 0
    history: list[dict] = []
    total_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t).cpu().numpy()
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        val_spearman = _spearman_by_group(val_pred, y_val, sid_val)
        epoch_sec = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_mae": val_mae,
                "val_spearman": val_spearman,
                "epoch_sec": epoch_sec,
            }
        )

        # 早停以场景内排序质量为准
        improved = bool(np.isfinite(val_spearman) and val_spearman > best_spearman)
        if improved:
            best_spearman = val_spearman
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose:
            marker = " *" if improved else ""
            print(
                f"epoch {epoch:3d}  loss={np.mean(losses):8.4f}  "
                f"val_mae={val_mae:8.4f}  val_spearman={val_spearman:6.3f}  "
                f"{epoch_sec:5.2f}s{marker}"
            )

        if wait >= cfg.patience:
            if verbose:
                print(
                    f"早停于 epoch {epoch}（最优 epoch {best_epoch}，"
                    f"val_spearman={best_spearman:.3f}）"
                )
            break

    total_sec = time.perf_counter() - total_start

    # 若验证集全程无法计算有效 Spearman（单样本场景等），回退保存末轮权重
    if best_spearman == -np.inf:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = history[-1]["epoch"]
    model.load_state_dict(best_state)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": x_train.shape[1],
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "feature_mean": standardizer.mean.tolist(),
            "feature_std": standardizer.std.tolist(),
        },
        out / "priority_regressor.pt",
    )
    with open(out / "train_history.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_val_spearman": best_spearman,
                "best_epoch": best_epoch,
                "total_sec": total_sec,
                "history": history,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    return model, {
        "best_val_spearman": best_spearman,
        "best_epoch": best_epoch,
        "total_sec": total_sec,
        "history": history,
    }


def load_priority_regressor(path: str | Path) -> tuple[PriorityRegressor, dict]:
    """加载训练好的优先级回归器（含训练集标准化统计量）。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    standardizer = None
    if ckpt.get("feature_mean") is not None:
        standardizer = FeatureStandardizer(
            mean=np.asarray(ckpt["feature_mean"], dtype=np.float32),
            std=np.asarray(ckpt["feature_std"], dtype=np.float32),
        )
    model = PriorityRegressor(
        ckpt["in_dim"], ckpt["hidden_dim"], ckpt["num_layers"], standardizer
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
