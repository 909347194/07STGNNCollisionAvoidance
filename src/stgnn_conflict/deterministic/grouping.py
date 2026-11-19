"""基于候选关联图的连通域分组（确定性主干分组）。"""

from __future__ import annotations


class _DSU:
    """并查集。"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_by_components(
    n: int, edges: list[tuple[int, int]], keep_singletons: bool = True
) -> list[list[int]]:
    """将候选关联图按连通域划分为航迹组。

    Parameters
    ----------
    n:
        航迹总数（tid 约定为 0 .. n-1）。
    edges:
        候选航迹对。
    keep_singletons:
        是否保留孤立航迹（单点组）；单点组可直接判定为无需检测。
    """
    dsu = _DSU(n)
    for a, b in edges:
        dsu.union(a, b)

    buckets: dict[int, list[int]] = {}
    for tid in range(n):
        buckets.setdefault(dsu.find(tid), []).append(tid)

    groups = sorted(buckets.values(), key=lambda g: (min(g), len(g)))
    if not keep_singletons:
        groups = [g for g in groups if len(g) > 1]
    return groups
