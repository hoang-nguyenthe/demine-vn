"""Hình minh hoạ phục vụ báo cáo và trình diễn trước hội đồng."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)

# Bảng màu pastel thống nhất với bản đề xuất.
INK = "#1C3557"
BLUE = "#4C7FB0"
MINT = "#5FA383"
BLUSH = "#C9705C"
CREAM = "#DFC58A"


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9.5,
            "axes.edgecolor": "#9DBCDA",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": "#E4ECF4",
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
        }
    )
    return plt


def fig_clearance_curve(pipeline, path: Path) -> None:
    """Đường cong hiệu quả rà phá — hình quan trọng nhất của đề tài."""
    plt = _setup()
    curve = pipeline.curve

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(
        curve.area_fraction * 100,
        curve.recovered_fraction * 100,
        color=INK,
        linewidth=2.2,
        label="Rà phá theo thứ tự mô hình đề xuất",
    )
    ax.plot([0, 100], [0, 100], color=BLUSH, linestyle="--", linewidth=1.5,
            label="Quét trải đều")

    ax.axvline(20, color=CREAM, linewidth=1.2, linestyle=":")
    y20 = curve.recovered_at_20pct * 100
    ax.plot([20], [y20], marker="o", color=MINT, markersize=7, zorder=5)
    ax.annotate(
        f"tại 20% diện tích\nthu hồi {y20:.0f}% vật nổ",
        xy=(20, y20),
        # Đặt chú thích trong vùng trống dưới đường chéo, đủ cao để không chạm
        # khung chú giải ở góc dưới bên phải.
        xytext=(33, 25),
        color=INK,
        arrowprops=dict(arrowstyle="->", color=MINT, linewidth=1.2),
    )

    ax.set_xlabel("Tỉ lệ diện tích đã rà phá (%)")
    ax.set_ylabel("Tỉ lệ vật nổ đã thu hồi (%)")
    ax.set_title("Hiệu quả xếp thứ tự ưu tiên rà phá", color=INK, fontsize=11.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_risk_map(pipeline, path: Path) -> None:
    """Bản đồ nguy cơ theo ô lưới, kèm vị trí tai nạn đã ghi nhận."""
    plt = _setup()
    prob = pipeline.probability.reshape(pipeline.grid.shape)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(prob, origin="lower", cmap="YlOrRd", vmin=0.0,
                   vmax=float(np.quantile(prob, 0.995)))
    if pipeline.accidents.n:
        ax.scatter(
            pipeline.accidents.col,
            pipeline.accidents.row,
            s=16,
            facecolor="none",
            edgecolor="#1A1A1A",
            linewidth=0.9,
            label="Tai nạn đã ghi nhận",
        )
        ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Xác suất còn tồn tại vật nổ", color=INK)
    ax.set_title(
        "Bản đồ nguy cơ và vị trí tai nạn thực tế", color=INK, fontsize=11.5
    )
    ax.set_xlabel("Ô lưới theo hướng đông")
    ax.set_ylabel("Ô lưới theo hướng bắc")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sources(pipeline, path: Path) -> None:
    """Ba lớp dữ liệu đặt cạnh nhau: hồ sơ, hố bom, vật nổ thật."""
    plt = _setup()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))

    layers = [
        (pipeline.tonnage_spread, "Hồ sơ không kích\n(đã lan toả theo sai số)", "Blues"),
        (pipeline.crater_map, "Hố bom phát hiện\ntrên ảnh vệ tinh", "Greens"),
        (pipeline.scene.uxo_count, "Vật nổ còn sót\n(nhãn đối chứng mô phỏng)", "Reds"),
    ]
    for ax, (layer, title, cmap) in zip(axes, layers):
        vmax = float(np.quantile(layer, 0.995)) or 1.0
        ax.imshow(layer, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, color=INK, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle(
        "Ba nguồn dữ liệu độc lập trên cùng một vùng nghiên cứu",
        color=INK,
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_calibration(pipeline, path: Path) -> None:
    """Biểu đồ tin cậy."""
    from ..evaluation.calibration import reliability_curve

    plt = _setup()
    pred, true, counts = reliability_curve(
        pipeline.probability[pipeline.is_cleared],
        pipeline.y_observed[pipeline.is_cleared],
        pipeline.cfg.risk.n_calibration_bins,
    )
    valid = counts > 0

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color=BLUSH, linewidth=1.4,
            label="Hiệu chỉnh hoàn hảo")
    ax.plot(pred[valid], true[valid], marker="o", color=INK, linewidth=1.8,
            markersize=5, label="Mô hình")
    ax.set_xlabel("Xác suất mô hình dự báo")
    ax.set_ylabel("Tần suất thực tế")
    ax.set_title("Biểu đồ tin cậy", color=INK, fontsize=11.5)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_ablation(pipeline, path: Path) -> None:
    """So sánh bốn cấu hình của phân tích đóng góp thành phần."""
    plt = _setup()
    rows = pipeline.results.get("phan_tich_dong_gop_thanh_phan", [])
    if not rows:
        return

    names = [r["cau_hinh"].split("—")[0].strip() for r in rows]
    recovered = [r["thu_hoi_tai_20pct"] * 100 for r in rows]
    coverage = [r["bao_phu_tai_nan_tai_20pct"] * 100 for r in rows]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - width / 2, recovered, width, color=BLUE, label="Thu hồi vật nổ tại 20% diện tích")
    ax.bar(x + width / 2, coverage, width, color=MINT, label="Bao phủ tai nạn tại 20% diện tích")
    ax.axhline(20, color=BLUSH, linestyle="--", linewidth=1.2)
    ax.text(len(names) - 0.55, 21.5, "mức quét trải đều", color=BLUSH, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Phần trăm")
    ax.set_title("Đóng góp của từng nguồn dữ liệu", color=INK, fontsize=11.5)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_importance(pipeline, path: Path) -> None:
    """Mức đóng góp của từng đặc trưng theo phép hoán vị."""
    plt = _setup()
    imp = pipeline.results.get("muc_dong_gop_theo_hoan_vi", {})
    if not imp:
        return

    items = list(imp.items())[:10][::-1]
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.barh(names, values, color=BLUE)
    ax.set_xlabel("Mức sụt giảm hiệu quả ưu tiên khi xáo trộn đặc trưng")
    ax.set_title("Đóng góp của từng đặc trưng", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sample_tiles(pipeline, path: Path) -> None:
    """Vài ảnh vệ tinh lịch sử kèm nhãn hố bom."""
    plt = _setup()
    tiles = sorted(pipeline.tiles, key=lambda t: -t.boxes.shape[0])[:4]
    if not tiles:
        return

    fig, axes = plt.subplots(1, len(tiles), figsize=(3.1 * len(tiles), 3.4))
    if len(tiles) == 1:
        axes = [axes]
    for ax, tile in zip(axes, tiles):
        ax.imshow(tile.image, cmap="gray")
        for b in tile.boxes:
            ax.add_patch(
                plt.Rectangle(
                    (b[0], b[1]),
                    b[2] - b[0],
                    b[3] - b[1],
                    fill=False,
                    edgecolor="#FFD166",
                    linewidth=1.0,
                )
            )
        ax.set_title(f"{tile.boxes.shape[0]} hố bom", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle("Ảnh vệ tinh lịch sử mô phỏng và nhãn hố bom", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_all_figures(pipeline, out_dir: Path) -> List[Path]:
    """Sinh toàn bộ hình minh hoạ, bỏ qua hình nào lỗi mà không chặn quy trình."""
    jobs = [
        ("01_duong_cong_hieu_qua_ra_pha.png", fig_clearance_curve),
        ("02_ban_do_nguy_co.png", fig_risk_map),
        ("03_ba_nguon_du_lieu.png", fig_sources),
        ("04_bieu_do_tin_cay.png", fig_calibration),
        ("05_dong_gop_thanh_phan.png", fig_ablation),
        ("06_dong_gop_dac_trung.png", fig_importance),
        ("07_anh_ve_tinh_mau.png", fig_sample_tiles),
    ]
    made: List[Path] = []
    for name, fn in jobs:
        path = Path(out_dir) / name
        try:
            fn(pipeline, path)
            if path.exists():
                made.append(path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Không dựng được hình %s: %s", name, exc)
    logger.info("  Đã dựng %d hình minh hoạ", len(made))
    return made
