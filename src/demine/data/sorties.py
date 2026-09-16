"""Mô phỏng hồ sơ không kích và vật nổ còn sót lại.

Đây là tầng nền của toàn bộ bài toán. Mô phỏng tái hiện đúng quan hệ nhân quả có
thật ngoài thực địa, và chính quan hệ đó tạo ra bài toán học máy có ý nghĩa:

  hồ sơ ghi chép  →  điểm rơi thật  →  nổ hoặc không nổ  →  hố bom hoặc vật nổ sót

Ba điểm quan trọng về mặt vật lý được mô hình hoá tường minh:

1. Hồ sơ ghi chép lệch so với điểm rơi thật một khoảng đáng kể, và một phần hồ sơ
   đã thất lạc hoàn toàn. Vì vậy hồ sơ là chỉ báo có nhiễu, không phải chân lý.

2. Bom rơi xuống nền đất mềm có xác suất không nổ cao hơn nhiều, vì đầu nổ chạm
   nổ cần lực cản đủ lớn mới kích hoạt.

3. Hố bom chỉ hình thành ở nơi bom đã nổ. Nghĩa là hố bom nhìn thấy trên ảnh
   là bằng chứng cho biết bom đã rơi ở đó, nhưng đồng thời cho biết chính quả bom
   đó đã nổ rồi. Trên nền mềm, nơi nhiều bom không nổ nhất, lại chính là nơi ít
   hố bom quan sát được nhất vì hố bị bồi lấp nhanh.

Điểm thứ ba là mấu chốt. Nó khiến bài toán không thể giải bằng một nguồn dữ liệu
duy nhất, và làm cho việc hợp nhất ba nguồn trở thành đóng góp thực chất.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import OrdnanceConfig, SortieConfig
from ..utils import get_logger
from .geo import Grid
from .terrain import Terrain

logger = get_logger(__name__)

# Khối lượng quy ước của một quả bom, ki-lô-gam. Dùng để quy đổi ra tải trọng,
# đại lượng mà hồ sơ không kích ghi nhận.
BOMB_MASS_KG = 340.0


@dataclass
class Scene:
    """Toàn bộ trạng thái mô phỏng của vùng nghiên cứu."""

    grid: Grid
    terrain: Terrain

    impact_x: np.ndarray
    impact_y: np.ndarray
    impact_is_dud: np.ndarray
    impact_crater_visible: np.ndarray
    impact_crater_diameter_m: np.ndarray
    impact_mission_id: np.ndarray

    record_x: np.ndarray
    record_y: np.ndarray
    record_n_bombs: np.ndarray
    record_mission_id: np.ndarray

    uxo_count: np.ndarray = field(default=None)
    crater_count: np.ndarray = field(default=None)
    recorded_tonnage: np.ndarray = field(default=None)
    true_tonnage: np.ndarray = field(default=None)

    @property
    def n_uxo(self) -> int:
        return int(self.impact_is_dud.sum())

    @property
    def n_impacts(self) -> int:
        return int(self.impact_x.size)


def _dud_probability(softness: np.ndarray, cfg: OrdnanceConfig) -> np.ndarray:
    """Xác suất một quả bom không nổ, theo độ mềm của nền đất tại điểm rơi."""
    multiplier = 1.0 + (cfg.soft_soil_dud_multiplier - 1.0) * softness
    return np.clip(cfg.base_dud_rate * multiplier, 0.0, 0.85)


def _crater_visibility(softness: np.ndarray, cfg: OrdnanceConfig) -> np.ndarray:
    """Xác suất hố bom còn quan sát được trên ảnh, theo độ mềm của nền đất."""
    return (
        cfg.crater_visible_rate_hard
        + (cfg.crater_visible_rate_soft - cfg.crater_visible_rate_hard) * softness
    )


def simulate_scene(
    grid: Grid,
    terrain: Terrain,
    sortie_cfg: SortieConfig,
    ordnance_cfg: OrdnanceConfig,
    rng: np.random.Generator,
) -> Scene:
    """Sinh toàn bộ phi vụ, điểm rơi, vật nổ còn sót và hồ sơ ghi chép."""

    xs, ys, mids = [], [], []

    # Phân bố điểm ngắm không đồng đều trên lãnh thổ. Không kích trong chiến tranh
    # bám theo mục tiêu quân sự — tuyến vận tải, bến vượt sông, nút giao thông —
    # chứ không rải đều. Hệ quả là ô nhiễm bom mìn ngoài thực địa tập trung thành
    # những hành lang rất đậm xen với những vùng gần như sạch, và chính cấu trúc
    # tập trung đó là thứ làm cho bài toán xếp thứ tự ưu tiên có ý nghĩa.
    target_field = (
        np.exp(-terrain.dist_road_m / 700.0)
        + 0.75 * np.exp(-terrain.dist_river_m / 900.0)
        + 0.35 * np.exp(-terrain.dist_village_m / 1100.0)
        + sortie_cfg.uniform_target_fraction
    )
    target_prob = (target_field / target_field.sum()).ravel()
    ny_grid, nx_grid = grid.shape

    for mission_id in range(sortie_cfg.n_missions):
        # Mỗi phi vụ là một đường bay ngắn với loạt bom rải dọc theo hướng bay.
        pick = int(rng.choice(target_prob.size, p=target_prob))
        ty, tx = divmod(pick, nx_grid)
        cx = (tx + rng.random()) * grid.cell
        cy = (ty + rng.random()) * grid.cell
        heading = rng.uniform(0.0, 2.0 * np.pi)
        n_bombs = int(rng.integers(*sortie_cfg.bombs_per_mission))

        # Khoảng cách giữa các quả trong loạt, mét.
        spacing = rng.uniform(35.0, 75.0)
        along = (np.arange(n_bombs) - (n_bombs - 1) / 2.0) * spacing

        bx = cx + along * np.cos(heading)
        by = cy + along * np.sin(heading)

        bx = bx + rng.normal(0.0, sortie_cfg.impact_dispersion_m, size=n_bombs)
        by = by + rng.normal(0.0, sortie_cfg.impact_dispersion_m, size=n_bombs)

        xs.append(bx)
        ys.append(by)
        mids.append(np.full(n_bombs, mission_id, dtype=int))

    impact_x = np.concatenate(xs)
    impact_y = np.concatenate(ys)
    mission_id = np.concatenate(mids)

    keep = (
        (impact_x >= 0.0)
        & (impact_x < grid.width_m)
        & (impact_y >= 0.0)
        & (impact_y < grid.height_m)
    )
    impact_x, impact_y, mission_id = impact_x[keep], impact_y[keep], mission_id[keep]

    col, row = grid.xy_to_index(impact_x, impact_y)
    softness = terrain.soil_softness[row, col]

    is_dud = rng.random(impact_x.size) < _dud_probability(softness, ordnance_cfg)

    visible_prob = _crater_visibility(softness, ordnance_cfg)
    exploded = ~is_dud
    crater_visible = exploded & (rng.random(impact_x.size) < visible_prob)

    lo, hi = ordnance_cfg.crater_diameter_m
    crater_diameter = np.where(
        crater_visible, rng.uniform(lo, hi, size=impact_x.size), 0.0
    )

    # Hồ sơ ghi chép: một điểm cho mỗi phi vụ, kèm sai số định vị, và một phần
    # phi vụ bị thất lạc hoàn toàn khỏi hồ sơ.
    rec_x, rec_y, rec_n, rec_id = [], [], [], []
    for mid in np.unique(mission_id):
        if rng.random() < sortie_cfg.missing_record_fraction:
            continue
        sel = mission_id == mid
        cx = float(impact_x[sel].mean())
        cy = float(impact_y[sel].mean())
        rec_x.append(cx + rng.normal(0.0, sortie_cfg.record_position_error_m))
        rec_y.append(cy + rng.normal(0.0, sortie_cfg.record_position_error_m))
        rec_n.append(int(sel.sum()))
        rec_id.append(int(mid))

    scene = Scene(
        grid=grid,
        terrain=terrain,
        impact_x=impact_x,
        impact_y=impact_y,
        impact_is_dud=is_dud,
        impact_crater_visible=crater_visible,
        impact_crater_diameter_m=crater_diameter,
        impact_mission_id=mission_id,
        record_x=np.asarray(rec_x, dtype=float),
        record_y=np.asarray(rec_y, dtype=float),
        record_n_bombs=np.asarray(rec_n, dtype=int),
        record_mission_id=np.asarray(rec_id, dtype=int),
    )

    scene.uxo_count = grid.accumulate(impact_x[is_dud], impact_y[is_dud])
    scene.crater_count = grid.accumulate(
        impact_x[crater_visible], impact_y[crater_visible]
    )
    scene.true_tonnage = grid.accumulate(
        impact_x, impact_y, np.full(impact_x.size, BOMB_MASS_KG / 1000.0)
    )
    scene.recorded_tonnage = grid.accumulate(
        scene.record_x,
        scene.record_y,
        scene.record_n_bombs * BOMB_MASS_KG / 1000.0,
    )

    logger.info(
        "Mô phỏng: %d điểm rơi | %d vật nổ còn sót (%.1f%%) | %d hố bom quan sát được | "
        "%d phi vụ có hồ sơ trên tổng %d",
        scene.n_impacts,
        scene.n_uxo,
        100.0 * scene.n_uxo / max(1, scene.n_impacts),
        int(crater_visible.sum()),
        scene.record_x.size,
        sortie_cfg.n_missions,
    )
    return scene
