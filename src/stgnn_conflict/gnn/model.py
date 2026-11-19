"""ST-GNN 消息传递模型。

定位（与创新点一致）：
- 不预测航迹是否碰撞，不输出碰撞标签；
- 在确定性分组之上学习“分组亲和度”：边分数表征两航迹是否适合被
  纳入同一可认证子组；节点池化分数表征子组通过安全证书的优先级。

实现为边条件消息传递网络，不依赖 torch_geometric，消息聚合用
scatter_add 手动完成。
"""

from __future__ import annotations

import torch
from torch import nn

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.gnn.graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


def _scatter_mean(msg: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """按 index 做邻域均值聚合（无邻居的节点聚合为 0）。"""
    summed = torch.zeros(dim_size, msg.size(1), device=msg.device, dtype=msg.dtype)
    summed.scatter_add_(0, index.unsqueeze(1).expand_as(msg), msg)
    counts = torch.zeros(dim_size, device=msg.device, dtype=msg.dtype)
    counts.scatter_add_(0, index, torch.ones(index.size(0), device=msg.device, dtype=msg.dtype))
    counts = counts.clamp(min=1.0).unsqueeze(1)
    return summed / counts


class STGNN(nn.Module):
    """边条件时空图神经网络。"""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.node_enc = nn.Linear(node_dim, hidden)
        self.edge_enc = nn.Linear(edge_dim, hidden)

        self.edge_mlp = nn.ModuleList(
            [nn.Linear(2 * hidden + hidden, hidden) for _ in range(num_layers)]
        )
        self.node_update = nn.ModuleList(
            [nn.Linear(2 * hidden, hidden) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

        # 边亲和度头：输出 [0,1]，越高越适合同组认证
        self.edge_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # 组优先级头：由节点嵌入池化得到子组可认证优先级
        self.pool_head = nn.Linear(hidden, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        src, dst = edge_index[0], edge_index[1]

        for mlp, update, norm in zip(self.edge_mlp, self.node_update, self.norms):
            msg = torch.relu(mlp(torch.cat([h[src], h[dst], e], dim=-1)))
            agg = _scatter_mean(msg, dst, h.size(0)) + _scatter_mean(msg, src, h.size(0))
            h = norm(h + update(torch.cat([h, agg], dim=-1)))

        edge_feat = torch.cat([h[src], h[dst], e], dim=-1)
        edge_score = torch.sigmoid(self.edge_head(edge_feat)).squeeze(-1)
        group_priority = torch.sigmoid(self.pool_head(h)).squeeze(-1)
        return edge_score, group_priority

    @torch.no_grad()
    def infer(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        return self.forward(x, edge_index, edge_attr)


def build_model(config: DetectionConfig) -> STGNN:
    return STGNN(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        hidden=config.gnn.hidden_dim,
        num_layers=config.gnn.num_layers,
    )
