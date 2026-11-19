"""全局配置：冲突判定阈值、证书容差、GNN 引导参数。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CertConfig:
    """安全证书与冲突判定相关参数。"""

    safety_radius: float = 30.0  # m，3D 冲突判定距离
    time_eps: float = 1e-3  # s，时间区间比较容差
    space_eps: float = 1e-3  # m，空间距离比较容差
    bbox_margin: float = 5.0  # m，索引检索时包围盒膨胀余量
    use_motion_cert: bool = True  # 是否启用二级相对运动下界证书（消融开关）


@dataclass
class GNNConfig:
    """ST-GNN 结构与引导参数。"""

    hidden_dim: int = 32
    num_layers: int = 2
    # 确定性细分规则：组内两航迹中心距离不超过该阈值才保留为同一子组
    split_geo_distance: float = 60.0


@dataclass
class DetectionConfig:
    """流水线总配置。"""

    cert: CertConfig = field(default_factory=CertConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    grid_cell: float = 60.0  # 时空网格索引的空间边长 m
    seed: int = 42
