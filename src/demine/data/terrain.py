"""Sinh địa hình, thổ nhưỡng và hiện trạng sử dụng đất của vùng nghiên cứu.

Các lớp này đóng hai vai trò trong bài toán. Thứ nhất, chúng chi phối cơ chế vật
lý: nền đất mềm làm tăng tỉ lệ bom không nổ và làm hố bom bị bồi lấp nhanh hơn.
Thứ hai, chúng quyết định mức độ phơi nhiễm của con người: người dân chỉ gặp tai
nạn ở nơi họ thực sự lui tới.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import GridConfig, TerrainConfig
from .geo import Grid


def _smooth(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Làm trơn trường ngẫu nhiên bằng bộ lọc Gauss tách được."""
    if sigma_cells <= 0:
        return field
    radius = max(1, int(3.0 * sigma_cells))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, field)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, out)
    return out


def _distance_transform(mask: np.ndarray, cell_size: float) -> np.ndarray:
    """Khoảng cách xấp xỉ tới điểm gần nhất của mặt nạ, tính bằng mét.

    Dùng thuật toán quét hai lượt theo khoảng cách chamfer, đủ chính xác cho mục
    đích xây dựng đặc trưng và không phụ thuộc thư viện ngoài.
    """
    big = 1.0e9
    dist = np.where(mask, 0.0, big).astype(float)
    ny, nx = dist.shape
    d1, d2 = 1.0, np.sqrt(2.0)

    for y in range(ny):
        for x in range(nx):
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + d1)
                if x > 0:
                    best = min(best, dist[y - 1, x - 1] + d2)
                if x < nx - 1:
                    best = min(best, dist[y - 1, x + 1] + d2)
            if x > 0:
                best = min(best, dist[y, x - 1] + d1)
            dist[y, x] = best

    for y in range(ny - 1, -1, -1):
        for x in range(nx - 1, -1, -1):
            best = dist[y, x]
            if y < ny - 1:
                best = min(best, dist[y + 1, x] + d1)
                if x > 0:
                    best = min(best, dist[y + 1, x - 1] + d2)
                if x < nx - 1:
                    best = min(best, dist[y + 1, x + 1] + d2)
            if x < nx - 1:
                best = min(best, dist[y, x + 1] + d1)
            dist[y, x] = best

    return dist * cell_size


@dataclass
class Terrain:
    """Toàn bộ các lớp nền của vùng nghiên cứu, cùng kích thước với lưới."""

    elevation_m: np.ndarray
    slope_deg: np.ndarray
    soil_softness: np.ndarray
    is_farmland: np.ndarray
    is_forest: np.ndarray
    dist_village_m: np.ndarray
    dist_road_m: np.ndarray
    dist_river_m: np.ndarray
    village_xy: np.ndarray
    exposure: np.ndarray

    @property
    def shape(self):
        return self.elevation_m.shape


def build_terrain(grid: Grid, cfg: TerrainConfig, rng: np.random.Generator) -> Terrain:
    ny, nx = grid.shape

    # Địa hình: trường ngẫu nhiên làm trơn ở hai bậc kích thước, tạo ra dạng đồi
    # thấp xen thung lũng đặc trưng của vùng trung du.
    base = _smooth(rng.normal(size=(ny, nx)), sigma_cells=14.0)
    detail = _smooth(rng.normal(size=(ny, nx)), sigma_cells=4.0)
    elevation = 120.0 * (base / (np.abs(base).max() + 1e-9)) + 25.0 * detail
    elevation -= elevation.min()

    gy, gx = np.gradient(elevation, grid.cell)
    slope = np.rad2deg(np.arctan(np.hypot(gx, gy)))

    # Sông: đường gãy khúc chạy qua vùng thấp.
    river_mask = np.zeros((ny, nx), dtype=bool)
    for _ in range(cfg.n_rivers):
        y = rng.integers(ny // 5, 4 * ny // 5)
        for x in range(nx):
            y += int(rng.integers(-1, 2))
            y = int(np.clip(y, 1, ny - 2))
            river_mask[y - 1 : y + 2, x] = True

    # Đường bộ: đoạn thẳng nối hai biên.
    road_mask = np.zeros((ny, nx), dtype=bool)
    for _ in range(cfg.n_roads):
        y0, y1 = rng.integers(0, ny, size=2)
        x0, x1 = 0, nx - 1
        n = max(nx, ny) * 2
        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.linspace(y0, y1, n).astype(int)
        ys = np.clip(ys, 0, ny - 1)
        road_mask[ys, xs] = True

    # Làng: cụm dân cư đặt ở nơi độ dốc thấp và gần nguồn nước.
    village_mask = np.zeros((ny, nx), dtype=bool)
    villages = []
    dist_river_tmp = _distance_transform(river_mask, grid.cell)
    favour = np.exp(-dist_river_tmp / 2500.0) * np.exp(-slope / 6.0)
    flat = favour.ravel() / favour.sum()
    picks = rng.choice(favour.size, size=cfg.n_villages, replace=False, p=flat)
    for p in picks:
        vy, vx = divmod(int(p), nx)
        village_mask[
            max(0, vy - 2) : vy + 3, max(0, vx - 2) : vx + 3
        ] = True
        villages.append((vx, vy))

    dist_village = _distance_transform(village_mask, grid.cell)
    dist_road = _distance_transform(road_mask, grid.cell)
    dist_river = _distance_transform(river_mask, grid.cell)

    # Thổ nhưỡng: đất mềm tập trung ở vùng thấp ven sông và nơi ít dốc.
    softness = _smooth(rng.normal(size=(ny, nx)), sigma_cells=9.0)
    softness = (softness - softness.min()) / (np.ptp(softness) + 1e-9)
    softness = 0.45 * softness + 0.35 * np.exp(-dist_river / 1800.0)
    softness += 0.20 * np.exp(-slope / 4.0)
    softness = np.clip(softness, 0.0, 1.0)

    # Đất canh tác: gần làng, ít dốc.
    suitability = np.exp(-dist_village / 1600.0) * np.exp(-slope / 7.0)
    threshold = np.quantile(suitability, 1.0 - cfg.farmland_fraction)
    is_farmland = suitability >= threshold
    is_forest = (~is_farmland) & (slope > np.quantile(slope, 0.55))

    # Mức độ phơi nhiễm của con người: tổ hợp của việc đất có được canh tác hay
    # không, khoảng cách tới khu dân cư và tới đường đi lại.
    exposure = (
        1.00 * is_farmland.astype(float)
        + 0.85 * np.exp(-dist_village / 1200.0)
        + 0.35 * np.exp(-dist_road / 900.0)
    )
    exposure = exposure / (exposure.max() + 1e-9)

    return Terrain(
        elevation_m=elevation,
        slope_deg=slope,
        soil_softness=softness,
        is_farmland=is_farmland,
        is_forest=is_forest,
        dist_village_m=dist_village,
        dist_road_m=dist_road,
        dist_river_m=dist_river,
        village_xy=np.asarray(villages, dtype=float),
        exposure=exposure,
    )
