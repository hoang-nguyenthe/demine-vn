"""Chỉ tiêu đánh giá hiệu quả xếp thứ tự ưu tiên rà phá.

Đây là nhóm chỉ tiêu quan trọng nhất của đề tài, vì nó là đại lượng duy nhất
chuyển thẳng được thành ý nghĩa thực tiễn. Độ chính xác phân loại không nói lên
điều gì cho người lập kế hoạch rà phá; câu hỏi của họ là: nếu đi theo thứ tự này
thì sau khi làm xong một phần diện tích, đã xử lý được bao nhiêu phần nguy cơ.

Hai chỉ tiêu được cài đặt:

*Đường cong hiệu quả rà phá.* Sắp các khoảnh đất theo thứ tự mô hình đề xuất, rồi
vẽ tỉ lệ vật nổ đã thu hồi theo tỉ lệ diện tích đã rà. Đường chéo là phương án
quét trải đều. Diện tích nằm giữa đường cong và đường chéo là phần giá trị mà mô
hình tạo ra.

*Tỉ lệ bao phủ tai nạn.* Tỉ lệ các vụ tai nạn đã thực sự xảy ra rơi vào phần diện
tích được mô hình xếp nguy cơ cao nhất. Vì vị trí tai nạn không do ai chọn, đây là
phép kiểm chứng độc lập với mọi phán đoán chuyên môn đã có trước.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ClearanceCurve:
    """Đường cong hiệu quả rà phá."""

    area_fraction: np.ndarray
    recovered_fraction: np.ndarray
    recovered_at_20pct: float
    recovered_at_10pct: float
    recovered_at_50pct: float
    gain_over_uniform: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "thu_hoi_tai_10pct_dien_tich": round(self.recovered_at_10pct, 4),
            "thu_hoi_tai_20pct_dien_tich": round(self.recovered_at_20pct, 4),
            "thu_hoi_tai_50pct_dien_tich": round(self.recovered_at_50pct, 4),
            "loi_the_so_voi_quet_deu": round(self.gain_over_uniform, 4),
        }


def clearance_efficiency_curve(
    scores: np.ndarray, items: np.ndarray, cell_weight: np.ndarray = None
) -> ClearanceCurve:
    """Dựng đường cong hiệu quả rà phá.

    Tham số ``items`` là số vật nổ thực sự có trên mỗi ô, ``scores`` là điểm nguy
    cơ do mô hình gán. Nếu các ô có diện tích khác nhau thì truyền ``cell_weight``;
    ở đây mọi ô đều bằng nhau nên tham số này thường bỏ trống.
    """
    scores = np.asarray(scores, dtype=float)
    items = np.asarray(items, dtype=float)
    weight = (
        np.ones_like(items) if cell_weight is None else np.asarray(cell_weight, float)
    )

    order = np.argsort(-scores, kind="mergesort")
    items_sorted = items[order]
    weight_sorted = weight[order]

    cum_area = np.cumsum(weight_sorted) / max(weight_sorted.sum(), 1e-12)
    total_items = items_sorted.sum()
    cum_items = np.cumsum(items_sorted) / (total_items if total_items > 0 else 1.0)

    cum_area = np.concatenate([[0.0], cum_area])
    cum_items = np.concatenate([[0.0], cum_items])

    def at(fraction: float) -> float:
        return float(np.interp(fraction, cum_area, cum_items))

    # Diện tích dưới đường cong, trừ đi 0,5 của đường chéo, rồi chuẩn hoá về [0, 1].
    auc = float(np.trapezoid(cum_items, cum_area))
    gain = (auc - 0.5) / 0.5

    return ClearanceCurve(
        area_fraction=cum_area,
        recovered_fraction=cum_items,
        recovered_at_10pct=at(0.10),
        recovered_at_20pct=at(0.20),
        recovered_at_50pct=at(0.50),
        gain_over_uniform=gain,
    )


@dataclass
class AccidentCoverage:
    """Kết quả kiểm chứng bằng hồ sơ tai nạn."""

    n_accidents: int
    capture_at_10pct: float
    capture_at_20pct: float
    capture_at_30pct: float
    lift_at_20pct: float
    split_label: str = "toàn bộ"
    detail: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {
            "tap_kiem_chung": self.split_label,
            "so_vu_tai_nan": self.n_accidents,
            "bao_phu_tai_10pct": round(self.capture_at_10pct, 4),
            "bao_phu_tai_20pct": round(self.capture_at_20pct, 4),
            "bao_phu_tai_30pct": round(self.capture_at_30pct, 4),
            "he_so_vuot_ngau_nhien_tai_20pct": round(self.lift_at_20pct, 3),
        }


def accident_coverage(
    scores: np.ndarray,
    accident_cell_index: np.ndarray,
    split_label: str = "toàn bộ",
) -> AccidentCoverage:
    """Đo tỉ lệ vụ tai nạn rơi vào phần diện tích nguy cơ cao nhất.

    ``accident_cell_index`` là chỉ số phẳng của ô lưới nơi xảy ra từng vụ tai nạn,
    theo đúng thứ tự hàng của ``scores``.
    """
    scores = np.asarray(scores, dtype=float)
    idx = np.asarray(accident_cell_index, dtype=int)
    n = int(idx.size)

    if n == 0:
        return AccidentCoverage(0, 0.0, 0.0, 0.0, 0.0, split_label)

    order = np.argsort(-scores, kind="mergesort")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    accident_rank = rank[idx] / float(order.size)

    def capture(fraction: float) -> float:
        return float(np.mean(accident_rank <= fraction))

    c20 = capture(0.20)
    return AccidentCoverage(
        n_accidents=n,
        capture_at_10pct=capture(0.10),
        capture_at_20pct=c20,
        capture_at_30pct=capture(0.30),
        lift_at_20pct=c20 / 0.20,
        split_label=split_label,
    )


def normalised_rank(x: np.ndarray) -> np.ndarray:
    """Chuyển một dãy điểm thành thứ hạng chuẩn hoá trong khoảng [0, 1].

    Làm việc trên thứ hạng thay vì trên giá trị gốc là bắt buộc khi cần trộn hai
    đại lượng có thang đo và hình dạng phân bố khác hẳn nhau. Nhân hai đại lượng
    như vậy với nhau hầu như không làm thay đổi thứ tự, vì đại lượng có dải động
    lớn hơn sẽ áp đảo hoàn toàn.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(-x, kind="mergesort")
    rank = np.empty(x.size, dtype=float)
    rank[order] = np.arange(x.size, dtype=float)
    return 1.0 - rank / max(1.0, x.size - 1.0)


def priority_index(
    ranking_score: np.ndarray,
    exposure: np.ndarray,
    exposure_weight: float = 0.35,
) -> np.ndarray:
    """Chỉ số ưu tiên tổng hợp giữa nguy cơ còn vật nổ và mức phơi nhiễm.

    Xác suất còn vật nổ không phải là tất cả. Một ô nằm giữa rừng, xác suất cao,
    nhưng nhiều năm không ai đặt chân tới, thì nguy cơ gây thương vong thấp hơn một
    ô xác suất trung bình nằm ngay cạnh trường học. Chỉ số này kết hợp hai vế đó để
    phục vụ phân bổ nguồn lực, trong khi xác suất gốc vẫn được giữ nguyên và báo
    cáo riêng.

    Phép trộn được thực hiện **trên thứ hạng**, không phải trên giá trị. Đây là chi
    tiết quyết định: phân bố xác suất của bài toán này cực kỳ lệch, phần lớn ô có
    giá trị rất nhỏ và một số ít ô gần bằng một. Nếu nhân thẳng xác suất với mức
    phơi nhiễm thì thứ tự gần như không đổi, và mức phơi nhiễm trở thành một tham
    số không có tác dụng gì.
    """
    s = np.asarray(ranking_score, dtype=float)
    e = np.asarray(exposure, dtype=float)
    e = (e - e.min()) / (np.ptp(e) + 1e-12)

    w = float(np.clip(exposure_weight, 0.0, 1.0))
    if w <= 0.0:
        return normalised_rank(s)
    return (1.0 - w) * normalised_rank(s) + w * normalised_rank(s * e)
