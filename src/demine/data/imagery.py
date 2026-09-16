"""Dựng ảnh vệ tinh trinh sát lịch sử mô phỏng.

Ảnh trinh sát giải mật của thời kỳ được nghiên cứu là ảnh phim đơn sắc, quét lại
ở độ phân giải vài mét. Ba đặc điểm của loại ảnh này gây khó cho mô hình thị giác
và đều được tái hiện ở đây:

  hạt phim        — nhiễu hạt thô, không phải nhiễu Gauss mịn như ảnh số
  chiếu sáng lệch — độ sáng nền thay đổi chậm trên khắp tấm ảnh
  tương phản thấp — hố bom sau nhiều năm chỉ còn là vệt tròn mờ

Hố bom được dựng theo hình thái thật: lòng hố tối hơn nền do đọng bóng và ẩm,
viền hố sáng hơn do đất bị hất lên. Hình thái viền sáng lòng tối này chính là dấu
hiệu mà người giải đoán ảnh dùng để nhận ra hố bom, nên mô hình cũng phải học
đúng dấu hiệu đó.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..config import ImageryConfig
from ..utils import get_logger
from .geo import Grid
from .sorties import Scene

logger = get_logger(__name__)


@dataclass
class Tile:
    """Một tấm ảnh con cùng nhãn hố bom nằm trong nó."""

    image: np.ndarray
    boxes: np.ndarray
    origin_x_m: float
    origin_y_m: float
    gsd_m: float
    tile_id: int
    split: str

    @property
    def size_px(self) -> int:
        return int(self.image.shape[0])


def _smooth2d(field: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return field
    radius = max(1, int(3.0 * sigma_px))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_px) ** 2)
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, field)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, out)
    return out


class HistoricalImageryRenderer:
    """Kết xuất ảnh trinh sát lịch sử mô phỏng cho một vùng."""

    def __init__(self, cfg: ImageryConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def _background(self, size: int) -> np.ndarray:
        """Nền ảnh: kết cấu ruộng và thảm thực vật, cộng chiếu sáng không đều."""
        rng = self.rng

        texture = _smooth2d(rng.normal(size=(size, size)), sigma_px=2.2)
        texture = texture / (np.abs(texture).max() + 1e-9)

        # Bờ thửa ruộng: các dải song song, đặc trưng của vùng canh tác.
        angle = rng.uniform(0.0, np.pi)
        yy, xx = np.mgrid[0:size, 0:size]
        proj = xx * np.cos(angle) + yy * np.sin(angle)
        period = rng.uniform(34.0, 110.0)
        amplitude = rng.uniform(0.035, 0.105)
        fields = amplitude * np.sin(2.0 * np.pi * proj / period)

        # Mảng ruộng và vạt cây: các vùng sáng tối không đều ở bậc kích thước lớn.
        patches = _smooth2d(rng.normal(size=(size, size)), sigma_px=size / 16.0)
        patches = 0.11 * patches / (np.abs(patches).max() + 1e-9)

        illumination = _smooth2d(rng.normal(size=(size, size)), sigma_px=size / 7.0)
        illumination = illumination / (np.abs(illumination).max() + 1e-9)

        img = (0.52 + 0.17 * texture + fields + patches
               + self.cfg.illumination_sigma * illumination)
        return img

    def _draw_crater(
        self, img: np.ndarray, cx: float, cy: float, diameter_px: float
    ) -> None:
        """Vẽ một hố bom: lòng tối, viền sáng."""
        r = diameter_px / 2.0
        pad = int(np.ceil(r * 1.9)) + 2
        x0, x1 = int(max(0, cx - pad)), int(min(img.shape[1], cx + pad + 1))
        y0, y1 = int(max(0, cy - pad)), int(min(img.shape[0], cy + pad + 1))
        if x1 <= x0 or y1 <= y0:
            return

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - cx, yy - cy)

        # Lòng hố: tối dần vào tâm.
        bowl = -0.30 * np.exp(-(dist ** 2) / (2.0 * (r * 0.55) ** 2))
        # Viền hố: vành sáng quanh mép.
        rim = 0.22 * np.exp(-((dist - r) ** 2) / (2.0 * (r * 0.30) ** 2))

        # Biến dạng nhẹ để hố không tròn hoàn hảo, đúng như ngoài thực địa.
        wobble = 1.0 + 0.12 * np.sin(4.0 * np.arctan2(yy - cy, xx - cx) + self.rng.uniform(0, 6.28))
        img[y0:y1, x0:x1] += (bowl + rim) * wobble

    def render_tile(
        self,
        scene: Scene,
        origin_x_m: float,
        origin_y_m: float,
        tile_id: int,
        split: str,
    ) -> Tile:
        size = self.cfg.tile_size_px
        gsd = self.cfg.ground_sample_distance_m
        extent = size * gsd

        img = self._background(size)

        sel = (
            scene.impact_crater_visible
            & (scene.impact_x >= origin_x_m)
            & (scene.impact_x < origin_x_m + extent)
            & (scene.impact_y >= origin_y_m)
            & (scene.impact_y < origin_y_m + extent)
        )

        boxes = []
        for x, y, d in zip(
            scene.impact_x[sel],
            scene.impact_y[sel],
            scene.impact_crater_diameter_m[sel],
        ):
            px = (x - origin_x_m) / gsd
            # Trục dọc của ảnh hướng xuống, ngược với trục bắc của mặt đất.
            py = size - 1 - (y - origin_y_m) / gsd
            dia_px = d / gsd
            self._draw_crater(img, px, py, dia_px)
            half = dia_px * 0.62
            boxes.append([px - half, py - half, px + half, py + half])

        # Nhiễu hạt phim, thêm sau khi đã vẽ đối tượng.
        img = img + self.rng.normal(0.0, self.cfg.film_grain_sigma, size=img.shape)
        img = np.clip(img, 0.0, 1.0)
        img_u8 = (img * 255.0).astype(np.uint8)

        box_arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if box_arr.size:
            box_arr[:, [0, 2]] = np.clip(box_arr[:, [0, 2]], 0, size - 1)
            box_arr[:, [1, 3]] = np.clip(box_arr[:, [1, 3]], 0, size - 1)
            wide = (box_arr[:, 2] - box_arr[:, 0]) > 3.0
            tall = (box_arr[:, 3] - box_arr[:, 1]) > 3.0
            box_arr = box_arr[wide & tall]

        return Tile(
            image=img_u8,
            boxes=box_arr,
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
            gsd_m=gsd,
            tile_id=tile_id,
            split=split,
        )


def build_tiles(scene: Scene, cfg: ImageryConfig, rng: np.random.Generator) -> List[Tile]:
    """Sinh toàn bộ tập ảnh con và chia tập theo khối không gian.

    Việc chia tập cũng theo khối không gian chứ không ngẫu nhiên: các ảnh con của
    cùng một vùng không được phép nằm ở cả tập huấn luyện lẫn tập kiểm tra.
    """
    grid: Grid = scene.grid
    extent = cfg.tile_size_px * cfg.ground_sample_distance_m
    renderer = HistoricalImageryRenderer(cfg, rng)

    max_x = max(1.0, grid.width_m - extent)
    max_y = max(1.0, grid.height_m - extent)

    # Chia vùng nghiên cứu thành ba dải dọc: huấn luyện, thẩm định, kiểm tra.
    n_train = int(cfg.n_tiles * cfg.train_fraction)
    n_val = int(cfg.n_tiles * cfg.val_fraction)
    n_test = cfg.n_tiles - n_train - n_val

    bounds = {
        "train": (0.0, max_x * cfg.train_fraction),
        "val": (max_x * cfg.train_fraction, max_x * (cfg.train_fraction + cfg.val_fraction)),
        "test": (max_x * (cfg.train_fraction + cfg.val_fraction), max_x),
    }
    counts = {"train": n_train, "val": n_val, "test": n_test}

    tiles: List[Tile] = []
    tid = 0
    for split, n in counts.items():
        lo, hi = bounds[split]
        if hi <= lo:
            lo, hi = 0.0, max_x
        for _ in range(max(0, n)):
            ox = float(rng.uniform(lo, hi))
            oy = float(rng.uniform(0.0, max_y))
            tiles.append(renderer.render_tile(scene, ox, oy, tid, split))
            tid += 1

    n_boxes = sum(t.boxes.shape[0] for t in tiles)
    logger.info(
        "Ảnh lịch sử: %d ảnh con (%d huấn luyện, %d thẩm định, %d kiểm tra) | %d hố bom có nhãn",
        len(tiles),
        counts["train"],
        counts["val"],
        counts["test"],
        n_boxes,
    )
    return tiles
