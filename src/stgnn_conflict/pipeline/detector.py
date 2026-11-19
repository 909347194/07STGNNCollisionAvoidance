"""端到端冲突检测流水线。

分层流程（与研究思路一一对应）：

1. 确定性检索与分组：时空索引召回候选航迹对 ⇒ 连通域分组；
2. 确定性预认证：整组通过证书则豁免，未通过的组进入困难核；
3.（可选）ST-GNN 引导：在困难核图上学习分组亲和度，给出合并 / 拆分建议
   与认证优先级；GNN 不输出碰撞判断；
4. 分层几何验证：对引导后的子组再次认证，未豁免航迹对执行连续 CPA。

接口约定
--------
- 输入：``list[Trajectory]``；
- 输出：``DetectionOutcome``，包含冲突集合、统计指标与分段耗时；
- 漏检保证：任何未被证书豁免的候选对都进入 CPA，详见 ``verifier`` 模块。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from stgnn_conflict.config import DetectionConfig
from stgnn_conflict.core.trajectory import Trajectory
from stgnn_conflict.deterministic.certificates import certify_group
from stgnn_conflict.deterministic.grouping import group_by_components
from stgnn_conflict.deterministic.indexes import build_index
from stgnn_conflict.gnn.graph import build_graph
from stgnn_conflict.gnn.guide import guide_groups, heuristic_edge_scores
from stgnn_conflict.verifier.validator import validate_groups


@dataclass
class DetectionOutcome:
    """流水线输出：冲突集合、指标与分段耗时（毫秒）。"""

    conflicts: set[tuple[int, int]] = field(default_factory=set)
    metrics: dict[str, object] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)


class ConflictDetector:
    """可配置索引类型、是否启用 GNN 引导的冲突检测器。

    ``model`` 为边亲和度 STGNN（可选）；``priority_model`` 为训练好的
    子组优先级回归器（可选）。两者均只影响处理策略，不参与安全判定。
    """

    def __init__(
        self,
        config: DetectionConfig,
        index_kind: str = "grid",
        use_gnn: bool = False,
        model: object | None = None,
        priority_model: object | None = None,
    ) -> None:
        self.config = config
        self.index_kind = index_kind
        self.use_gnn = use_gnn
        self.model = model
        self.priority_model = priority_model

    def _infer_scores(self, graph) -> dict[tuple[int, int], float]:
        """GNN 推理得到边亲和度；模型缺失时回退几何兜底分数。"""
        if self.model is None:
            return heuristic_edge_scores(graph.edge_pairs, graph.edge_attr)

        import torch

        x = torch.tensor(graph.x, dtype=torch.float32)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(graph.edge_attr, dtype=torch.float32)
        edge_score, _ = self.model.infer(x, edge_index, edge_attr)
        return {pair: float(edge_score[k]) for k, pair in enumerate(graph.edge_pairs)}

    def _infer_subgroup_priorities(
        self,
        refined_groups: list[list[int]],
        traj_map: dict[int, Trajectory],
        candidate_set: set[tuple[int, int]],
    ) -> list[float] | None:
        """训练好的子组优先级回归器对细化子组打分；未加载时返回 None。"""
        regressor = self.priority_model
        if regressor is None or not refined_groups:
            return None

        import torch

        from stgnn_conflict.dataset.samples import _subgroup_features

        feats = []
        for group in refined_groups:
            feats.append(
                _subgroup_features(group, traj_map, candidate_set)
            )
        if not feats:
            return None
        x = torch.tensor(np.array(feats), dtype=torch.float32)
        return [float(v) for v in regressor.infer(x)]

    def detect(self, trajectories: list[Trajectory]) -> DetectionOutcome:
        timing: dict[str, float] = {}
        traj_map = {t.traj_id: t for t in trajectories}
        n = len(trajectories)

        # 1. 确定性索引召回
        t0 = time.perf_counter()
        recall_margin = self.config.cert.safety_radius + self.config.cert.bbox_margin
        index = build_index(self.index_kind, self.config.grid_cell, recall_margin)
        candidate_pairs = index.query_pairs(trajectories)
        timing["index_ms"] = (time.perf_counter() - t0) * 1e3

        # 2. 连通域分组
        t0 = time.perf_counter()
        groups = group_by_components(n, candidate_pairs)
        timing["group_ms"] = (time.perf_counter() - t0) * 1e3

        candidate_set = {(min(a, b), max(a, b)) for a, b in candidate_pairs}

        # 3. 确定性预认证：通过的组直接豁免，其余进入困难核
        t0 = time.perf_counter()
        certified_groups = 0
        certified_pairs = 0
        hard_groups: list[list[int]] = []
        for group in groups:
            cert = certify_group(group, traj_map, self.config, candidate_set)
            if cert.safe:
                certified_groups += 1
                certified_pairs += cert.pairs_checked
            else:
                hard_groups.append(group)
        timing["certify_ms"] = (time.perf_counter() - t0) * 1e3

        # 4.（可选）GNN 引导：确定性规则细分 + 子组认证优先级排序
        refined_groups = hard_groups
        split_pairs: list[tuple[int, int]] = []
        priorities: list[float] = []
        if self.use_gnn and hard_groups:
            t0 = time.perf_counter()
            # 仅当加载了边亲和度模型时才构建图并推理；纯优先级回归器场景
            # 跳过无谓的图构建与兜底打分，细分仍由确定性规则完成。
            scores: dict[tuple[int, int], float] = {}
            if self.model is not None:
                hard_ids = {tid for grp in hard_groups for tid in grp}
                hard_edges = [(a, b) for a, b in candidate_pairs if a in hard_ids and b in hard_ids]
                graph = build_graph(trajectories, hard_edges, self.config)
                scores = self._infer_scores(graph)
            guide = guide_groups(
                hard_groups, scores, self.config, traj_map=traj_map,
                candidate_set=candidate_set,
            )
            refined_groups = guide.refined_groups
            split_pairs = guide.split_pairs

            # 优先级来源：优先使用训练好的子组优先级回归器；
            # 缺失时回退图亲和度聚合分数
            model_priorities = self._infer_subgroup_priorities(
                refined_groups, traj_map, candidate_set
            )
            if model_priorities is not None:
                priorities = model_priorities
            else:
                priorities = guide.priority
            timing["gnn_ms"] = (time.perf_counter() - t0) * 1e3
        else:
            timing["gnn_ms"] = 0.0

        # 按认证潜力优先级降序验证（顺序不影响正确性，只影响处理策略）
        if priorities:
            order = sorted(range(len(refined_groups)), key=lambda i: -priorities[i])
            refined_groups = [refined_groups[i] for i in order]

        # 未被任何细化组覆盖的候选对（如被拆分切断的边），进入 CPA 兜底
        refined_internal: set[tuple[int, int]] = set()
        for group in refined_groups:
            for i, ta in enumerate(group):
                for tb in group[i + 1 :]:
                    refined_internal.add((min(ta, tb), max(ta, tb)))
        deferred_pairs = [p for p in candidate_pairs if p not in refined_internal]

        # 5. 分层几何验证 + 连续 CPA
        t0 = time.perf_counter()
        verify = validate_groups(
            refined_groups, traj_map, self.config, candidate_set, deferred_pairs
        )
        timing["verify_ms"] = (time.perf_counter() - t0) * 1e3
        timing["total_ms"] = sum(timing.values())

        metrics: dict[str, object] = {
            "n_traj": n,
            "candidate_pairs": len(candidate_pairs),
            "n_groups": len(groups),
            "certified_groups_det": certified_groups,
            "certified_pairs_det": certified_pairs,
            "hard_groups": len(hard_groups),
            "refined_groups": len(refined_groups),
            "split_pairs": len(split_pairs),
            "deferred_pairs": len(deferred_pairs),
            "cpa_calls": verify.cpa_calls,
            "certified_groups": verify.certified_groups,
            "certified_pairs": verify.certified_pairs,
            "certified_pair_ratio": (
                verify.certified_pairs / len(candidate_pairs) if candidate_pairs else 0.0
            ),
            "certified_group_ratio": (
                verify.certified_groups / len(refined_groups) if refined_groups else 0.0
            ),
            "uncertified_pairs": verify.uncertified_pairs,
            "conflicts": len(verify.conflicts),
        }

        return DetectionOutcome(conflicts=verify.conflicts, metrics=metrics, timing=timing)
