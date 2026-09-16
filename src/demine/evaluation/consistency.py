"""Kiểm tra nhất quán giữa hai nguồn dữ liệu độc lập.

Hồ sơ không kích và ảnh vệ tinh lịch sử không liên quan gì đến nhau về xuất xứ:
một bên là sổ sách tác chiến của phi đội, một bên là dấu vết vật lý còn lại trên
mặt đất và được ghi lại bởi một hệ thống hoàn toàn khác. Nếu hai nguồn này khớp
nhau về mặt không gian, độ tin cậy của cả hai cùng được củng cố mà không cần viện
đến bất kỳ nhãn đối chứng nào.

Phép kiểm tra thứ hai ở bậc độ lớn: từ tổng lượng bom đạn theo hồ sơ và tỉ lệ bom
không nổ đã biết trong tài liệu kỹ thuật, ước lượng khối lượng vật nổ còn sót rồi
so với con số mà mô hình đưa ra. Không đòi hỏi trùng khít, chỉ đòi hỏi cùng bậc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Hệ số tương quan hạng Spearman, cài đặt trực tiếp.

    Dùng tương quan hạng chứ không phải tương quan tuyến tính vì quan hệ giữa tải
    trọng bom và số hố bom quan sát được là quan hệ đồng biến nhưng không tuyến
    tính, và cả hai phân bố đều lệch mạnh về phía giá trị nhỏ.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 3:
        return 0.0

    ra = _rank_average(a)
    rb = _rank_average(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((ra * rb).sum() / denom)


def _rank_average(x: np.ndarray) -> np.ndarray:
    """Hạng của từng phần tử, các giá trị bằng nhau nhận hạng trung bình."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)

    x_sorted = x[order]
    i = 0
    while i < x_sorted.size:
        j = i
        while j + 1 < x_sorted.size and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        if j > i:
            mean_rank = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = mean_rank
        i = j + 1
    return ranks


@dataclass
class ConsistencyReport:
    spearman_crater_vs_tonnage: float
    n_cells_compared: int
    estimated_uxo_from_records: float
    model_expected_uxo: float
    order_of_magnitude_ratio: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "spearman_ho_bom_vs_tai_trong": round(self.spearman_crater_vs_tonnage, 4),
            "so_o_duoc_so_sanh": self.n_cells_compared,
            "uoc_luong_vat_no_tu_ho_so": round(self.estimated_uxo_from_records, 1),
            "ky_vong_vat_no_tu_mo_hinh": round(self.model_expected_uxo, 1),
            "ty_so_bac_do_lon": round(self.order_of_magnitude_ratio, 3),
        }


def check_consistency(
    crater_density: np.ndarray,
    tonnage_spread: np.ndarray,
    n_recorded_bombs: int,
    base_dud_rate: float,
    model_probability: np.ndarray,
) -> ConsistencyReport:
    """Chạy cả hai phép kiểm tra nhất quán và gộp kết quả."""
    a = np.asarray(crater_density, dtype=float).ravel()
    b = np.asarray(tonnage_spread, dtype=float).ravel()

    # Chỉ so sánh trên các ô mà ít nhất một trong hai nguồn có tín hiệu; các ô
    # trống ở cả hai nguồn không mang thông tin và chỉ làm loãng hệ số.
    mask = (a > 1e-9) | (b > 1e-9)
    rho = spearman_correlation(a[mask], b[mask])

    expected_from_records = float(n_recorded_bombs) * base_dud_rate
    expected_from_model = float(np.sum(model_probability))
    ratio = expected_from_model / max(expected_from_records, 1e-9)

    return ConsistencyReport(
        spearman_crater_vs_tonnage=rho,
        n_cells_compared=int(mask.sum()),
        estimated_uxo_from_records=expected_from_records,
        model_expected_uxo=expected_from_model,
        order_of_magnitude_ratio=ratio,
    )
