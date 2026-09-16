"""Hệ quy chiếu của vùng nghiên cứu.

Toàn hệ thống làm việc trên một lưới ô vuông phẳng có gốc quy ước. Mô-đun này
chuyển đổi hai chiều giữa toạ độ mét trong hệ phẳng, chỉ số ô lưới và toạ độ địa
lý dùng để hiển thị bản đồ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..config import GridConfig

# Bán kính Trái Đất trung bình, mét.
EARTH_RADIUS_M = 6_371_000.0


@dataclass
class Grid:
    """Lưới ô vuông của vùng nghiên cứu."""

    cfg: GridConfig

    @property
    def cell(self) -> float:
        return self.cfg.cell_size_m

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.cfg.n_cells_y, self.cfg.n_cells_x)

    @property
    def n_cells(self) -> int:
        return self.cfg.n_cells_x * self.cfg.n_cells_y

    @property
    def width_m(self) -> float:
        return self.cfg.n_cells_x * self.cell

    @property
    def height_m(self) -> float:
        return self.cfg.n_cells_y * self.cell

    def xy_to_index(self, x_m, y_m):
        """Chuyển toạ độ mét sang chỉ số cột và hàng của ô lưới."""
        col = np.floor(np.asarray(x_m, dtype=float) / self.cell).astype(int)
        row = np.floor(np.asarray(y_m, dtype=float) / self.cell).astype(int)
        return col, row

    def inside(self, col, row):
        col = np.asarray(col)
        row = np.asarray(row)
        return (col >= 0) & (col < self.cfg.n_cells_x) & (row >= 0) & (row < self.cfg.n_cells_y)

    def flat_index(self, col, row):
        return np.asarray(row) * self.cfg.n_cells_x + np.asarray(col)

    def index_to_center_xy(self, col, row):
        x = (np.asarray(col, dtype=float) + 0.5) * self.cell
        y = (np.asarray(row, dtype=float) + 0.5) * self.cell
        return x, y

    def xy_to_lonlat(self, x_m, y_m):
        """Chuyển toạ độ mét sang kinh độ và vĩ độ để hiển thị bản đồ."""
        lat0 = np.deg2rad(self.cfg.origin_lat)
        dlat = np.asarray(y_m, dtype=float) / EARTH_RADIUS_M
        dlon = np.asarray(x_m, dtype=float) / (EARTH_RADIUS_M * np.cos(lat0))
        return (
            self.cfg.origin_lon + np.rad2deg(dlon),
            self.cfg.origin_lat + np.rad2deg(dlat),
        )

    def cell_centers_lonlat(self):
        cols, rows = np.meshgrid(
            np.arange(self.cfg.n_cells_x), np.arange(self.cfg.n_cells_y)
        )
        x, y = self.index_to_center_xy(cols.ravel(), rows.ravel())
        return self.xy_to_lonlat(x, y)

    def accumulate(self, x_m, y_m, weights=None):
        """Dồn các điểm về lưới, trả về mảng hai chiều."""
        col, row = self.xy_to_index(x_m, y_m)
        keep = self.inside(col, row)
        col, row = col[keep], row[keep]
        w = None if weights is None else np.asarray(weights, dtype=float)[keep]
        out = np.zeros(self.shape, dtype=float)
        np.add.at(out, (row, col), 1.0 if w is None else w)
        return out
