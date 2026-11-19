"""时空候选索引：暴力扫描、均匀网格、静态 STR R-tree。

所有索引只负责“召回潜在关联航迹对”，不做冲突判定；
召回结果再交由分组与证书层处理。时间区间重叠与空间包围盒相交
作为统一的候选过滤条件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stgnn_conflict.core.bbox import AABB
from stgnn_conflict.core.trajectory import Trajectory


class SpatialIndex(Protocol):
    """候选对索引接口。"""

    name: str

    def query_pairs(self, trajectories: list[Trajectory]) -> list[tuple[int, int]]:
        """返回候选航迹对 (tid_a, tid_b)，保证 a < b。"""


def _spatiotemporal_overlap(a: Trajectory, b: Trajectory, margin: float) -> bool:
    return a.time_overlaps(b) and a.bbox.intersects(b.bbox, margin=margin)


class BruteForceIndex:
    """O(n²) 暴力扫描基线，作为索引召回正确性的对照。"""

    name = "brute_force"

    def __init__(self, margin: float = 0.0) -> None:
        self.margin = margin

    def query_pairs(self, trajectories: list[Trajectory]) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        n = len(trajectories)
        for i in range(n):
            for j in range(i + 1, n):
                if _spatiotemporal_overlap(trajectories[i], trajectories[j], self.margin):
                    pairs.append((trajectories[i].traj_id, trajectories[j].traj_id))
        return pairs


class GridIndex:
    """均匀空间网格索引：航迹按包围盒覆盖的格子倒排，再做候选过滤。"""

    name = "grid"

    def __init__(self, cell: float, margin: float = 0.0) -> None:
        if cell <= 0:
            raise ValueError("网格边长必须为正数")
        self.cell = cell
        self.margin = margin

    def _cell_keys(self, bbox: AABB) -> list[tuple[int, int, int]]:
        # 按 margin 膨胀后再覆盖格子，避免邻近候选对因格网边界被漏掉
        lo = np.floor((bbox.lo - self.margin) / self.cell).astype(int)
        hi = np.floor((bbox.hi + self.margin) / self.cell).astype(int)
        keys: list[tuple[int, int, int]] = []
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    keys.append((x, y, z))
        return keys

    def query_pairs(self, trajectories: list[Trajectory]) -> list[tuple[int, int]]:
        bucket: dict[tuple[int, int, int], list[int]] = {}
        for traj in trajectories:
            for key in self._cell_keys(traj.bbox):
                bucket.setdefault(key, []).append(traj.traj_id)

        traj_by_id = {t.traj_id: t for t in trajectories}
        candidate: set[tuple[int, int]] = set()
        for ids in bucket.values():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = sorted((ids[i], ids[j]))
                    candidate.add((a, b))

        return [
            (a, b)
            for a, b in sorted(candidate)
            if _spatiotemporal_overlap(traj_by_id[a], traj_by_id[b], self.margin)
        ]


@dataclass
class _RTreeEntry:
    tid: int
    bbox: AABB


@dataclass
class _RTreeNode:
    bbox: AABB
    children: list[_RTreeNode] | None = None
    entries: list[_RTreeEntry] | None = None


class StaticRTreeIndex:
    """静态批量装载（STR）R-tree，适合一次性场景的批量候选检索。

    建树后不可增删；节点按 fanout 装箱，查询时沿包围盒相交分支剪枝。
    """

    name = "rtree"

    def __init__(self, margin: float = 0.0, fanout: int = 16) -> None:
        self.margin = margin
        self.fanout = fanout
        self._root: _RTreeNode | None = None

    @staticmethod
    def _bbox_of_entries(entries: list[_RTreeEntry]) -> AABB:
        lo = np.min([e.bbox.lo for e in entries], axis=0)
        hi = np.max([e.bbox.hi for e in entries], axis=0)
        return AABB(lo=lo, hi=hi)

    @staticmethod
    def _bbox_of_nodes(nodes: list[_RTreeNode]) -> AABB:
        lo = np.min([n.bbox.lo for n in nodes], axis=0)
        hi = np.max([n.bbox.hi for n in nodes], axis=0)
        return AABB(lo=lo, hi=hi)

    def _pack_leaves(self, entries: list[_RTreeEntry]) -> list[_RTreeNode]:
        entries = sorted(entries, key=lambda e: (e.bbox.center[0], e.bbox.center[1]))
        slices = [entries[i : i + self.fanout] for i in range(0, len(entries), self.fanout)]
        nodes: list[_RTreeNode] = []
        for sl in slices:
            sl = sorted(sl, key=lambda e: e.bbox.center[1])
            nodes.append(_RTreeNode(bbox=self._bbox_of_entries(sl), entries=sl))
        return nodes

    def _pack_internal(self, nodes: list[_RTreeNode]) -> list[_RTreeNode]:
        nodes = sorted(nodes, key=lambda n: (n.bbox.center[0], n.bbox.center[1]))
        slices = [nodes[i : i + self.fanout] for i in range(0, len(nodes), self.fanout)]
        return [
            _RTreeNode(bbox=self._bbox_of_nodes(sl), children=sl) for sl in slices
        ]

    def _build(self, entries: list[_RTreeEntry]) -> _RTreeNode:
        level = self._pack_leaves(entries)
        while len(level) > 1:
            level = self._pack_internal(level)
        return level[0]

    def _collect(self, node: _RTreeNode, bbox: AABB, out: set[int]) -> None:
        if not node.bbox.intersects(bbox, margin=self.margin):
            return
        if node.entries is not None:
            for entry in node.entries:
                if entry.bbox.intersects(bbox, margin=self.margin):
                    out.add(entry.tid)
        else:
            assert node.children is not None
            for child in node.children:
                self._collect(child, bbox, out)

    def query_pairs(self, trajectories: list[Trajectory]) -> list[tuple[int, int]]:
        entries = [_RTreeEntry(t.traj_id, t.bbox) for t in trajectories]
        self._root = self._build(entries)
        traj_by_id = {t.traj_id: t for t in trajectories}

        pairs: set[tuple[int, int]] = set()
        for traj in trajectories:
            neighbors: set[int] = set()
            self._collect(self._root, traj.bbox, neighbors)
            for other in neighbors:
                if other <= traj.traj_id:
                    continue
                if _spatiotemporal_overlap(traj, traj_by_id[other], self.margin):
                    pairs.add((traj.traj_id, other))
        return sorted(pairs)


def build_index(kind: str, cell: float, margin: float) -> SpatialIndex:
    """按名称构造索引，便于实验流水线统一调用。"""
    kind = kind.lower()
    if kind == "brute_force":
        return BruteForceIndex(margin=margin)
    if kind == "grid":
        return GridIndex(cell=cell, margin=margin)
    if kind == "rtree":
        return StaticRTreeIndex(margin=margin)
    raise ValueError(f"未知索引类型: {kind}")
