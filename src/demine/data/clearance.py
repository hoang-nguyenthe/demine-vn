"""Mô phỏng hoạt động rà phá đã thực hiện và hồ sơ tai nạn.

Hai nguồn nhãn này có tính chất thống kê hoàn toàn khác nhau, và chính sự khác
nhau đó là nền tảng của phương pháp kiểm chứng bốn tầng.

Đất đã rà phá **không phải mẫu ngẫu nhiên**. Các cơ quan chuyên môn chọn khoảnh
đất để rà phá dựa trên hồ sơ không kích, mức độ gần khu dân cư và nhu cầu sử dụng
đất. Nếu huấn luyện và đánh giá chỉ trên nguồn này, mô hình sẽ học lại chính phán
đoán của những người đi trước, và mọi chỉ tiêu sẽ bị thổi phồng. Mô-đun này tái
hiện đúng cơ chế chọn mẫu đó để tầng ba có cái mà hiệu chỉnh.

Hồ sơ tai nạn thì ngược lại. Vị trí tai nạn không do ai chọn — đó là nơi người dân
vô tình chạm phải vật nổ trong sinh hoạt. Nó phụ thuộc vào vật nổ có thật ở đó hay
không và mức độ lui tới của con người, chứ không phụ thuộc phán đoán chuyên môn.
Vì vậy đây là tập kiểm chứng độc lập, và là bằng chứng mạnh nhất của đề tài.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AccidentConfig, ClearanceConfig
from ..utils import get_logger
from .sorties import Scene

logger = get_logger(__name__)

# Cạnh một khoảnh rà phá, tính bằng số ô lưới. Năm ô tương ứng 500 mét, là bậc
# kích thước của một khoảnh được giao cho một đội trong thực tế.
PARCEL_CELLS = 5


def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-12 else 1.0)


@dataclass
class ClearanceData:
    """Kết quả rà phá đã thực hiện, theo từng ô lưới."""

    is_cleared: np.ndarray
    items_found: np.ndarray
    parcel_id: np.ndarray
    propensity: np.ndarray

    @property
    def n_cleared_cells(self) -> int:
        return int(self.is_cleared.sum())


@dataclass
class AccidentData:
    """Hồ sơ tai nạn bom mìn đã ghi nhận."""

    col: np.ndarray
    row: np.ndarray
    year: np.ndarray

    @property
    def n(self) -> int:
        return int(self.col.size)


def simulate_clearance(
    scene: Scene,
    tonnage_spread: np.ndarray,
    cfg: ClearanceConfig,
    rng: np.random.Generator,
) -> ClearanceData:
    """Chọn các khoảnh đất đưa vào rà phá theo cơ chế có thiên lệch, rồi rà phá."""
    grid = scene.grid
    terrain = scene.terrain
    ny, nx = grid.shape

    ny_p = int(np.ceil(ny / PARCEL_CELLS))
    nx_p = int(np.ceil(nx / PARCEL_CELLS))

    cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
    parcel_id = (rows // PARCEL_CELLS) * nx_p + (cols // PARCEL_CELLS)

    # Điểm ưu tiên của cơ quan chuyên môn khi chọn khoảnh đất.
    score = (
        cfg.selection_weight_tonnage * _zscore(tonnage_spread)
        + cfg.selection_weight_near_village * _zscore(np.exp(-terrain.dist_village_m / 1200.0))
        + cfg.selection_weight_near_road * _zscore(np.exp(-terrain.dist_road_m / 900.0))
    )
    propensity = 1.0 / (1.0 + np.exp(-score))

    n_parcels = ny_p * nx_p
    parcel_score = np.zeros(n_parcels, dtype=float)
    np.add.at(parcel_score, parcel_id.ravel(), score.ravel())
    parcel_count = np.bincount(parcel_id.ravel(), minlength=n_parcels).astype(float)
    parcel_score = parcel_score / np.maximum(parcel_count, 1.0)

    # Quyết định chọn không hoàn toàn theo điểm: thêm nhiễu để phản ánh các yếu
    # tố ngoài mô hình như ngân sách, đề nghị của địa phương, khả năng tiếp cận.
    parcel_score = parcel_score + rng.normal(0.0, 0.85, size=n_parcels)

    n_select = max(1, int(round(n_parcels * cfg.cleared_area_fraction)))
    chosen = np.argsort(-parcel_score)[:n_select]
    chosen_mask = np.zeros(n_parcels, dtype=bool)
    chosen_mask[chosen] = True

    is_cleared = chosen_mask[parcel_id]

    # Rà phá: thu hồi được phần lớn nhưng không phải toàn bộ vật nổ trong khoảnh.
    true_counts = scene.uxo_count.astype(int)
    found = np.zeros_like(true_counts)
    sel = is_cleared & (true_counts > 0)
    if sel.any():
        found[sel] = rng.binomial(true_counts[sel], cfg.detection_efficiency)

    logger.info(
        "Rà phá đã thực hiện: %d/%d khoảnh (%.1f%% diện tích) | thu hồi %d vật nổ trên tổng %d nằm trong vùng đã rà",
        n_select,
        n_parcels,
        100.0 * is_cleared.mean(),
        int(found.sum()),
        int(true_counts[is_cleared].sum()),
    )
    return ClearanceData(
        is_cleared=is_cleared,
        items_found=found,
        parcel_id=parcel_id,
        propensity=propensity,
    )


def simulate_accidents(
    scene: Scene,
    clearance: ClearanceData,
    cfg: AccidentConfig,
    rng: np.random.Generator,
) -> AccidentData:
    """Sinh hồ sơ tai nạn từ vật nổ còn sót và mức độ phơi nhiễm của con người.

    Vật nổ nằm trong khoảnh đã rà phá và đã được thu hồi thì không còn gây tai nạn
    nữa, nên được loại khỏi nguồn rủi ro.
    """
    grid = scene.grid
    terrain = scene.terrain

    dud = scene.impact_is_dud
    x = scene.impact_x[dud]
    y = scene.impact_y[dud]
    col, row = grid.xy_to_index(x, y)

    # Vật nổ đã bị thu hồi thì loại khỏi nguồn rủi ro.
    removed_fraction = np.zeros(scene.grid.shape, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        removed_fraction = np.where(
            scene.uxo_count > 0, clearance.items_found / np.maximum(scene.uxo_count, 1), 0.0
        )
    still_present = rng.random(x.size) >= removed_fraction[row, col]

    exposure = terrain.exposure[row, col]
    hazard = cfg.annual_incident_rate * exposure * still_present

    acc_col, acc_row, acc_year = [], [], []
    for year in range(1, cfg.n_years + 1):
        hit = rng.random(x.size) < hazard
        if not hit.any():
            continue
        acc_col.append(col[hit])
        acc_row.append(row[hit])
        acc_year.append(np.full(int(hit.sum()), year, dtype=int))
        # Vật nổ đã gây tai nạn thì được xử lý ngay sau đó.
        hazard = np.where(hit, 0.0, hazard)

    if acc_col:
        acc_col = np.concatenate(acc_col)
        acc_row = np.concatenate(acc_row)
        acc_year = np.concatenate(acc_year)
    else:
        acc_col = np.array([], dtype=int)
        acc_row = np.array([], dtype=int)
        acc_year = np.array([], dtype=int)

    logger.info(
        "Hồ sơ tai nạn: %d vụ trong %d năm | %d vụ trước mốc chia thời gian, %d vụ sau",
        acc_col.size,
        cfg.n_years,
        int((acc_year <= cfg.temporal_split_year).sum()),
        int((acc_year > cfg.temporal_split_year).sum()),
    )
    return AccidentData(col=acc_col, row=acc_row, year=acc_year)
