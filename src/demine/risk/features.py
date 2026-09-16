"""Tầng ba, phần một — xây dựng tập đặc trưng theo từng ô lưới.

Bốn nhóm đặc trưng, đúng như trình bày trong đề xuất:

  nhóm lịch sử   — quy chiếu hồ sơ không kích về lưới qua hạt nhân lan toả
  nhóm quan sát  — mật độ hố bom phát hiện được trên ảnh vệ tinh lịch sử
  nhóm địa hình  — độ cao, độ dốc, độ mềm của nền đất, khoảng cách tới dòng chảy
  nhóm hiện trạng— lớp phủ, khoảng cách tới khu dân cư và tới đường giao thông

Điểm cần lưu ý về phương pháp: hồ sơ không kích ghi một điểm cho mỗi phi vụ, kèm
sai số định vị hàng trăm mét. Nếu dồn thẳng điểm đó vào một ô lưới thì thông tin
bị đặt sai chỗ. Vì vậy mỗi bản ghi được lan toả ra chung quanh bằng một hạt nhân
Gauss có độ rộng đúng bằng sai số định vị đã biết. Đây là cách mô hình hoá tường
minh bất định thay vì giả vờ rằng nó không tồn tại.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..config import RiskConfig
from ..data.geo import Grid
from ..data.sorties import Scene


def gaussian_spread(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Lan toả một trường theo hạt nhân Gauss tách được, bảo toàn tổng."""
    if sigma_cells <= 0:
        return field.copy()
    radius = max(1, int(3.0 * sigma_cells))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, field)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, out)
    return out


def neighbourhood_sum(field: np.ndarray, radius_cells: int) -> np.ndarray:
    """Tổng giá trị trong lân cận vuông bán kính cho trước."""
    if radius_cells <= 0:
        return field.copy()
    k = 2 * radius_cells + 1
    ones = np.ones(k, dtype=float)
    out = np.apply_along_axis(lambda m: np.convolve(m, ones, mode="same"), 0, field)
    out = np.apply_along_axis(lambda m: np.convolve(m, ones, mode="same"), 1, out)
    return out


FEATURE_NAMES: List[str] = [
    "tai_trong_ghi_nhan_lan_toa",
    "tai_trong_lan_can_500m",
    "mat_do_phi_vu_lan_toa",
    "ky_vong_bom_khong_no",
    "ho_bom_mat_do",
    "ho_bom_lan_can_300m",
    "ho_bom_duong_kinh_tb",
    "ho_bom_hieu_chinh_tam_nhin",
    "do_cao",
    "do_doc",
    "do_mem_nen_dat",
    "khoang_cach_song",
    "la_dat_canh_tac",
    "la_rung",
    "khoang_cach_lang",
    "khoang_cach_duong",
    "muc_do_phoi_nhiem",
]


@dataclass
class FeatureTable:
    """Bảng đặc trưng phẳng, mỗi hàng là một ô lưới."""

    X: np.ndarray
    names: List[str]
    col: np.ndarray
    row: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    def column(self, name: str) -> np.ndarray:
        return self.X[:, self.names.index(name)]


def build_features(
    scene: Scene,
    crater_map: np.ndarray,
    crater_diameter_map: np.ndarray,
    cfg: RiskConfig,
    crater_coverage: np.ndarray = None,
) -> FeatureTable:
    """Dựng bảng đặc trưng từ cảnh mô phỏng và kết quả phát hiện hố bom.

    Tham số ``crater_map`` là mật độ hố bom **do mô hình thị giác phát hiện**, chứ
    không phải mật độ thật. Đây là điểm quan trọng: tầng ba tiêu thụ đầu ra của
    tầng hai, kể cả sai sót của nó, đúng như khi vận hành thật.
    """
    grid: Grid = scene.grid
    terrain = scene.terrain

    sigma_cells = cfg.record_kernel_radius_m / grid.cell

    tonnage_spread = gaussian_spread(scene.recorded_tonnage, sigma_cells)
    mission_density = gaussian_spread(
        grid.accumulate(scene.record_x, scene.record_y), sigma_cells
    )
    tonnage_neigh = neighbourhood_sum(tonnage_spread, radius_cells=5)

    crater_spread = gaussian_spread(crater_map, sigma_cells=1.5)
    crater_neigh = neighbourhood_sum(crater_map, radius_cells=3)

    # Hai đặc trưng dưới đây là nơi việc hợp nhất ba nguồn tạo ra giá trị thật, chứ
    # không phải chỉ đặt chúng cạnh nhau. Cả hai đều xuất phát từ cơ chế vật lý đã
    # nêu trong tài liệu, không phải từ việc thử nghiệm mò.
    softness = terrain.soil_softness

    # Thứ nhất — kỳ vọng số bom không nổ. Cái ta muốn biết không phải là bao nhiêu
    # bom đã rơi, mà bao nhiêu quả trong số đó đã không nổ. Tỉ lệ không nổ tăng
    # theo độ mềm của nền đất, vì đầu nổ chạm nổ cần lực cản đủ lớn mới kích hoạt.
    # Nhân hai đại lượng đó với nhau cho ra ngay đại lượng cần ước lượng.
    dud_multiplier = 1.0 + (cfg.dud_softness_gain - 1.0) * softness
    expected_duds = tonnage_spread * dud_multiplier

    # Thứ hai — mật độ hố bom đã hiệu chỉnh thiên lệch quan sát. Hố bom trên nền
    # mềm bị bồi lấp nhanh nên ít quan sát được hơn, đúng ở nơi nhiều bom không nổ
    # nhất. Nếu dùng thẳng số hố bom đếm được thì mô hình sẽ đánh giá thấp có hệ
    # thống chính những vùng nguy hiểm nhất. Chia cho xác suất còn nhìn thấy được
    # sẽ khử đúng thiên lệch đó.
    visibility = np.clip(
        cfg.crater_visibility_hard
        + (cfg.crater_visibility_soft - cfg.crater_visibility_hard) * softness,
        0.15,
        1.0,
    )
    crater_corrected = crater_spread / visibility

    layers = [
        tonnage_spread,
        tonnage_neigh,
        mission_density,
        expected_duds,
        crater_spread,
        crater_neigh,
        crater_diameter_map,
        crater_corrected,
        terrain.elevation_m,
        terrain.slope_deg,
        terrain.soil_softness,
        terrain.dist_river_m,
        terrain.is_farmland.astype(float),
        terrain.is_forest.astype(float),
        terrain.dist_village_m,
        terrain.dist_road_m,
        terrain.exposure,
    ]
    assert len(layers) == len(FEATURE_NAMES)

    X = np.stack([lay.ravel() for lay in layers], axis=1).astype(np.float32)

    # Ở những ô không có ảnh vệ tinh phủ tới, nhóm đặc trưng quan sát được đánh dấu
    # là **khuyết**, chứ không phải bằng không. Cây quyết định tăng cường gradient
    # xử lý được giá trị khuyết một cách tự nhiên: nó học riêng một nhánh cho trường
    # hợp thiếu dữ liệu, thay vì hiểu nhầm rằng nơi đó đã được nhìn và không thấy gì.
    if crater_coverage is not None:
        missing = ~np.asarray(crater_coverage, dtype=bool).ravel()
        crater_cols = [
            FEATURE_NAMES.index(name)
            for name in (
                "ho_bom_mat_do",
                "ho_bom_lan_can_300m",
                "ho_bom_duong_kinh_tb",
                "ho_bom_hieu_chinh_tam_nhin",
            )
        ]
        X[np.ix_(missing, crater_cols)] = np.nan

    cols, rows = np.meshgrid(
        np.arange(grid.cfg.n_cells_x), np.arange(grid.cfg.n_cells_y)
    )
    return FeatureTable(X=X, names=list(FEATURE_NAMES), col=cols.ravel(), row=rows.ravel())
