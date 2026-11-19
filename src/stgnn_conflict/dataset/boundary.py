"""空间边界加载与局部坐标系。

支持 GeoJSON ``Polygon`` / ``MultiPolygon``（CRS84 经纬度）；
场景在局部米制坐标系中生成，必要时可反算回经纬度。
当前版本只取外环，内部空洞不做遮挡判定（够用即可）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS = 6_371_000.0  # m


def _extract_rings(geom: dict) -> list[np.ndarray]:
    """从 GeoJSON 几何中提取外环列表，每个环为 (M, 2) 经纬度数组。"""
    if geom["type"] == "Polygon":
        return [np.array(ring, dtype=float) for ring in geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return [
            np.array(ring, dtype=float)
            for polygon in geom["coordinates"]
            for ring in polygon
        ]
    raise ValueError(f"不支持的几何类型: {geom['type']}")


def _point_in_ring(lon: float, lat: float, ring: np.ndarray) -> bool:
    """射线法判定点是否在多边形环内。"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class Boundary:
    """场景空间边界。"""

    rings: list[np.ndarray]
    lon0: float
    lat0: float

    @classmethod
    def from_geojson(cls, path: str) -> Boundary:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        rings: list[np.ndarray] = []
        for feature in data.get("features", []):
            rings.extend(_extract_rings(feature["geometry"]))
        if not rings:
            raise ValueError("GeoJSON 中未找到有效多边形")
        all_points = np.vstack(rings)
        lon0 = float(all_points[:, 0].mean())
        lat0 = float(all_points[:, 1].mean())
        return cls(rings=rings, lon0=lon0, lat0=lat0)

    def to_local(self, lon: float, lat: float) -> np.ndarray:
        """经纬度 → 局部米制坐标 (x, y)。"""
        x = math.radians(lon - self.lon0) * EARTH_RADIUS * math.cos(math.radians(self.lat0))
        y = math.radians(lat - self.lat0) * EARTH_RADIUS
        return np.array([x, y], dtype=float)

    def to_geographic(self, xy: np.ndarray) -> tuple[float, float]:
        """局部米制坐标 → 经纬度。"""
        lon = self.lon0 + math.degrees(xy[0] / (EARTH_RADIUS * math.cos(math.radians(self.lat0))))
        lat = self.lat0 + math.degrees(xy[1] / EARTH_RADIUS)
        return lon, lat

    def contains(self, xy: np.ndarray) -> bool:
        """判定局部米制坐标点是否落在任一外环内。"""
        lon, lat = self.to_geographic(xy)
        return any(_point_in_ring(lon, lat, ring) for ring in self.rings)

    def local_bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """返回局部坐标系下的 (lo, hi) 包围盒。"""
        points = np.array(
            [self.to_local(lon, lat) for ring in self.rings for lon, lat in ring]
        )
        return points.min(axis=0), points.max(axis=0)
