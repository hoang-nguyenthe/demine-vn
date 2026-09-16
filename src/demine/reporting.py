"""Kết xuất danh mục ưu tiên rà phá và báo cáo tổng hợp."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import SAFETY_NOTICE, __version__
from .utils import format_table, get_logger

logger = get_logger(__name__)


def write_priority_list(pipeline, path: Path, n_top: int = 500) -> Path:
    """Danh mục khoảnh đất xếp theo thứ tự rà phá đề xuất.

    Đây là sản phẩm đầu ra mà đơn vị lập kế hoạch sử dụng trực tiếp. Mỗi dòng có
    đầy đủ toạ độ, xác suất đã hiệu chỉnh, chỉ số ưu tiên và các yếu tố giải thích,
    để người đọc kiểm tra được vì sao một khoảnh được xếp cao.
    """
    path = Path(path)
    order = np.argsort(-pipeline.priority)[:n_top]
    lon, lat = pipeline.grid.cell_centers_lonlat()

    terrain = pipeline.terrain
    dist_village = terrain.dist_village_m.ravel()
    farmland = terrain.is_farmland.ravel()
    tonnage = pipeline.tonnage_spread.ravel()
    craters = pipeline.crater_map.ravel()

    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "thu_tu_uu_tien",
                "chi_so_o_luoi",
                "kinh_do",
                "vi_do",
                "xac_suat_con_vat_no",
                "chi_so_uu_tien",
                "tai_trong_bom_ghi_nhan_tan",
                "so_ho_bom_phat_hien",
                "khoang_cach_khu_dan_cu_m",
                "la_dat_canh_tac",
                "da_ra_pha",
                "ghi_chu",
            ]
        )
        for rank, i in enumerate(order, start=1):
            writer.writerow(
                [
                    rank,
                    int(i),
                    f"{lon[i]:.6f}",
                    f"{lat[i]:.6f}",
                    f"{pipeline.probability[i]:.4f}",
                    f"{pipeline.priority[i]:.4f}",
                    f"{tonnage[i]:.2f}",
                    int(craters[i]),
                    f"{dist_village[i]:.0f}",
                    "co" if farmland[i] else "khong",
                    "co" if pipeline.is_cleared[i] else "khong",
                    "Thu tu uu tien ra pha. Khong phai xac nhan an toan.",
                ]
            )

    logger.info("  Đã ghi danh mục ưu tiên: %s (%d khoảnh)", path.name, len(order))
    return path


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def write_report(pipeline, path: Path) -> Path:
    """Báo cáo tổng hợp dạng văn bản, đọc được trực tiếp trong notebook."""
    path = Path(path)
    r: Dict = pipeline.results
    cfg = pipeline.cfg

    lines: List[str] = []
    add = lines.append

    add(f"# DeMine-VN — Báo cáo tổng hợp kết quả (phiên bản {__version__})")
    add("")
    add(
        "Hệ thống lập bản đồ nguy cơ và xếp thứ tự ưu tiên rà phá bom mìn, vật nổ "
        "còn sót lại sau chiến tranh, trên cơ sở hợp nhất hồ sơ không kích giải mật, "
        "ảnh vệ tinh lịch sử và dữ liệu rà phá thực địa."
    )
    add("")
    add("> **Nguyên tắc an toàn bắt buộc.** " + SAFETY_NOTICE)
    add("")
    add("---")
    add("")

    # -- Vùng nghiên cứu ------------------------------------------------
    add("## 1. Vùng nghiên cứu và dữ liệu")
    add("")
    add(
        f"- Diện tích: {pipeline.grid.width_m/1000:.1f} × "
        f"{pipeline.grid.height_m/1000:.1f} km, chia thành "
        f"{pipeline.grid.n_cells:,} ô lưới cạnh {pipeline.grid.cell:.0f} m".replace(",", ".")
    )
    add(f"- Phi vụ mô phỏng: {cfg.sortie.n_missions}, trong đó "
        f"{pipeline.scene.record_x.size} phi vụ còn hồ sơ")
    add(f"- Điểm rơi: {pipeline.scene.n_impacts:,}".replace(",", "."))
    add(f"- Vật nổ còn sót (nhãn đối chứng): {pipeline.scene.n_uxo:,}".replace(",", "."))
    add(f"- Hố bom phát hiện được trên ảnh: {int(pipeline.crater_map.sum()):,}".replace(",", "."))
    add(f"- Diện tích đã rà phá: {_fmt_pct(pipeline.clearance.is_cleared.mean())}")
    add(f"- Tai nạn đã ghi nhận: {pipeline.accidents.n} vụ trong {cfg.accident.n_years} năm")
    add("")

    # -- Tầng hai --------------------------------------------------------
    add("## 2. Tầng hai — phát hiện hố bom trên ảnh vệ tinh lịch sử")
    add("")
    det = r.get("tang_hai_phat_hien_ho_bom", {})
    if det:
        add(format_table([det]))
    add("")

    # -- Kiểm chứng ------------------------------------------------------
    add("## 3. Kiểm chứng bốn tầng")
    add("")

    add("### Tầng 1 — đối chứng trên đất đã rà phá")
    add("")
    add(
        "Chia tập **theo khối không gian**, không chia ngẫu nhiên theo điểm. "
        "Chi tiết phương pháp xem `docs/VALIDATION.md`."
    )
    add("")
    t1 = r.get("tang_1_doi_chung_dat_da_ra_pha", {})
    if t1:
        add(format_table([t1]))
    add("")
    curve = r.get("duong_cong_hieu_qua_ra_pha", {})
    if curve:
        add("Đường cong hiệu quả rà phá trên toàn vùng:")
        add("")
        add(format_table([curve]))
    add("")

    add("### Tầng 2 — đối chứng độc lập bằng hồ sơ tai nạn")
    add("")
    add(
        "Vị trí tai nạn không do mô hình chọn cũng không do cơ quan chuyên môn chọn, "
        "nên đây là phép lấy mẫu độc lập với mọi phán đoán đã có trước. Dòng thứ hai "
        "là phép thử nghiêm hơn: mô hình chỉ học dữ liệu trước một mốc thời gian và "
        "được đánh giá bằng các vụ tai nạn xảy ra sau mốc đó."
    )
    add("")
    t2 = r.get("tang_2_doi_chung_ho_so_tai_nan", [])
    if t2:
        add(format_table(t2))
    add("")

    add("### Tầng 3 — nhất quán giữa hai nguồn độc lập")
    add("")
    add(
        "Hồ sơ không kích và ảnh vệ tinh không liên quan về xuất xứ. Nếu chúng khớp "
        "nhau về mặt không gian thì độ tin cậy của cả hai cùng được củng cố mà không "
        "cần viện đến bất kỳ nhãn đối chứng nào."
    )
    add("")
    t3 = r.get("tang_3_nhat_quan_hai_nguon", {})
    if t3:
        add(format_table([t3]))
    add("")

    add("### Tầng 4 — chuyển vùng địa lý và chất lượng hiệu chỉnh")
    add("")
    t4a = r.get("tang_4a_chuyen_vung_dia_ly", {})
    if t4a:
        add(format_table([t4a]))
    add("")
    t4b = r.get("tang_4b_hieu_chinh_xac_suat", {})
    if t4b:
        add(format_table([t4b]))
    add("")

    # -- Đóng góp thành phần ---------------------------------------------
    add("## 4. Phân tích đóng góp thành phần")
    add("")
    add(
        "Hồ sơ không kích có sai số định vị lớn nhưng phủ khắp; hố bom thì chính xác "
        "về vị trí nhưng thiếu hụt có hệ thống đúng ở nơi nền đất mềm — tức là đúng nơi "
        "nhiều vật nổ còn sót nhất. Hai nguồn sai theo hai kiểu khác nhau, nên việc hợp "
        "nhất có giá trị thật chứ không phải cộng thêm cho đủ."
    )
    add("")
    abl = r.get("phan_tich_dong_gop_thanh_phan", [])
    if abl:
        add(format_table(abl))
    add("")

    imp = r.get("muc_dong_gop_theo_hoan_vi", {})
    if imp:
        add("### Đóng góp của từng đặc trưng, đo bằng phép hoán vị")
        add("")
        top = list(imp.items())[:10]
        add(format_table([{"dac_trung": k, "muc_sut_giam": v} for k, v in top]))
        add("")

    # -- Kết luận ---------------------------------------------------------
    add("## 5. Phạm vi cam kết")
    add("")
    add(
        "Đề tài không cam kết rằng hệ thống phát hiện được vật nổ. Đề tài cam kết bốn "
        "điều, và cả bốn đều được kiểm chứng bằng dữ liệu trong chính báo cáo này:"
    )
    add("")
    add("1. Xếp hạng đúng các khoảnh đất đã rà phá, đánh giá trên khối không gian giữ lại.")
    add("2. Bao phủ phần lớn các vụ tai nạn đã xảy ra bằng một phần nhỏ diện tích.")
    add("3. Duy trì hiệu năng khi chuyển sang địa bàn chưa từng xuất hiện trong huấn luyện.")
    add("4. Cung cấp ước lượng xác suất đã hiệu chỉnh, kèm biểu đồ tin cậy và điểm Brier.")
    add("")
    add("> **Nguyên tắc an toàn bắt buộc.** " + SAFETY_NOTICE)
    add("")

    env = r.get("moi_truong_chay", {})
    if env:
        add("---")
        add("")
        add("## Môi trường chạy")
        add("")
        for k, v in env.items():
            add(f"- {k}: {v}")
        add("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  Đã ghi báo cáo tổng hợp: %s", path.name)
    return path
