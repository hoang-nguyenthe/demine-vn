"""Phân tích đóng góp thành phần.

Bốn cấu hình được so sánh, đúng như cam kết trong đề xuất. Mục đích không phải để
chứng minh hệ thống đầy đủ là tốt nhất — điều đó gần như chắc chắn — mà để trả lời
câu hỏi mà hội đồng sẽ đặt ra: **từng nguồn dữ liệu đóng góp được bao nhiêu, và có
nguồn nào thừa không**.

  A — chỉ hồ sơ không kích
  B — chỉ hố bom phát hiện trên ảnh vệ tinh
  C — hợp nhất hai nguồn trên, không có đặc trưng địa hình và hiện trạng
  D — hệ thống đầy đủ

So sánh A với B đặc biệt đáng chú ý. Hồ sơ không kích có sai số định vị lớn nhưng
phủ khắp; hố bom thì chính xác về vị trí nhưng thiếu hụt có hệ thống đúng ở nơi nền
đất mềm — tức là đúng nơi nhiều vật nổ còn sót nhất. Hai nguồn sai theo hai kiểu
khác nhau, nên hợp nhất lại có giá trị thật chứ không phải cộng thêm cho đẹp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..config import RiskConfig
from ..risk.features import FEATURE_NAMES
from ..risk.model import RiskModel
from ..utils import get_logger
from .prioritisation import (
    accident_coverage,
    clearance_efficiency_curve,
    priority_index,
)

logger = get_logger(__name__)

FEATURE_GROUPS: Dict[str, List[str]] = {
    "ho_so_khong_kich": [
        "tai_trong_ghi_nhan_lan_toa",
        "tai_trong_lan_can_500m",
        "mat_do_phi_vu_lan_toa",
        "ky_vong_bom_khong_no",
    ],
    "ho_bom_anh_ve_tinh": [
        "ho_bom_mat_do",
        "ho_bom_lan_can_300m",
        "ho_bom_duong_kinh_tb",
        "ho_bom_hieu_chinh_tam_nhin",
    ],
    "dia_hinh_hien_trang": [
        "do_cao",
        "do_doc",
        "do_mem_nen_dat",
        "khoang_cach_song",
        "la_dat_canh_tac",
        "la_rung",
        "khoang_cach_lang",
        "khoang_cach_duong",
        "muc_do_phoi_nhiem",
    ],
}

CONFIGURATIONS: Dict[str, List[str]] = {
    "A — chỉ hồ sơ không kích": ["ho_so_khong_kich"],
    "B — chỉ hố bom từ ảnh vệ tinh": ["ho_bom_anh_ve_tinh"],
    "C — hợp nhất hai nguồn, không địa hình": [
        "ho_so_khong_kich",
        "ho_bom_anh_ve_tinh",
    ],
    "D — hệ thống đầy đủ": [
        "ho_so_khong_kich",
        "ho_bom_anh_ve_tinh",
        "dia_hinh_hien_trang",
    ],
}


@dataclass
class AblationRow:
    name: str
    n_features: int
    recovered_at_20pct: float
    gain_over_uniform: float
    accident_capture_at_20pct: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "cau_hinh": self.name,
            "so_dac_trung": self.n_features,
            "thu_hoi_tai_20pct": round(self.recovered_at_20pct, 4),
            "loi_the_so_voi_quet_deu": round(self.gain_over_uniform, 4),
            "bao_phu_tai_nan_tai_20pct": round(self.accident_capture_at_20pct, 4),
        }


def _column_indices(groups: List[str]) -> List[int]:
    wanted = []
    for g in groups:
        wanted.extend(FEATURE_GROUPS[g])
    return [FEATURE_NAMES.index(name) for name in wanted]


def run_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    was_selected_train: np.ndarray,
    X_all: np.ndarray,
    X_test: np.ndarray,
    items_test: np.ndarray,
    accident_index_test: np.ndarray,
    cfg: RiskConfig,
    exposure_test: np.ndarray = None,
) -> List[AblationRow]:
    """Chạy cả bốn cấu hình trên cùng một phép chia tập.

    Cả bốn cấu hình đều đi qua đúng cùng một quy tắc xếp hạng của hệ thống hoàn
    chỉnh, kể cả bước trộn mức phơi nhiễm. Nhờ vậy con số của cấu hình đầy đủ khớp
    với con số công bố ở phần kết quả chính, và người đọc so sánh được trực tiếp.
    """
    rows: List[AblationRow] = []

    for name, groups in CONFIGURATIONS.items():
        cols = _column_indices(groups)
        names = [FEATURE_NAMES[i] for i in cols]

        model = RiskModel(cfg=cfg)
        model.fit(
            X_train[:, cols],
            y_train,
            was_selected_train,
            X_all[:, cols],
            feature_names=names,
        )
        prob = model.predict_ranking_score(X_test[:, cols])
        if exposure_test is not None:
            prob = priority_index(
                prob, exposure_test, exposure_weight=cfg.priority_exposure_weight
            )

        curve = clearance_efficiency_curve(prob, items_test)
        coverage = accident_coverage(prob, accident_index_test, name)

        rows.append(
            AblationRow(
                name=name,
                n_features=len(cols),
                recovered_at_20pct=curve.recovered_at_20pct,
                gain_over_uniform=curve.gain_over_uniform,
                accident_capture_at_20pct=coverage.capture_at_20pct,
            )
        )
        logger.info(
            "  %-38s thu hồi tại 20%% diện tích: %.3f | bao phủ tai nạn: %.3f",
            name,
            curve.recovered_at_20pct,
            coverage.capture_at_20pct,
        )

    return rows
