"""边界加载与坐标投影测试。"""

from pathlib import Path

import numpy as np

from stgnn_conflict.dataset.boundary import Boundary

GEOJSON = Path(__file__).parents[1] / "data" / "raw" / "region.geojson"


def _boundary() -> Boundary:
    return Boundary.from_geojson(str(GEOJSON))


def test_bbox_and_center_inside():
    boundary = _boundary()
    lo, hi = boundary.local_bbox()
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert boundary.contains((lo + hi) / 2.0)


def test_geographic_roundtrip():
    boundary = _boundary()
    lo, hi = boundary.local_bbox()
    point = (lo + hi) / 2.0
    lon, lat = boundary.to_geographic(point)
    back = boundary.to_local(lon, lat)
    assert np.allclose(back, point, atol=1e-6)


def test_outside_point_not_inside():
    boundary = _boundary()
    _lo, hi = boundary.local_bbox()
    outside = hi + np.array([10_000.0, 10_000.0])
    assert not boundary.contains(outside)
