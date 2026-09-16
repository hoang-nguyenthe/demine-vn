#!/usr/bin/env python
"""Quét tham số tầng ba và đo khoảng cách tới trần thông tin.

Tệp lệnh này trả lời một câu hỏi mà đội thi phải trả lời được trước hội đồng:
**mô hình còn cải thiện được nữa không, hay đã chạm trần của dữ liệu**.

Cách làm: dựng lại vùng nghiên cứu bằng đúng hạt giống ngẫu nhiên của lần chạy
chính, nạp lại trọng số phát hiện hố bom đã huấn luyện, rồi chỉ quét các lựa chọn
của tầng ba. Nhờ vậy không phải huấn luyện lại mô hình thị giác, và mọi cấu hình
được so sánh trên cùng một bộ dữ liệu và cùng một phép chia tập theo khối.

Ba trần được đo để đối chiếu:

* **Trần tuyệt đối** — xếp hạng bằng chính số vật nổ thật. Đây là mức không thể
  vượt qua, dùng để biết thang đo.
* **Trần khi biết mọi điểm rơi** — xếp hạng bằng mật độ điểm rơi thật. Mức này cho
  biết bài toán sẽ giải được tới đâu nếu tư liệu lịch sử còn nguyên vẹn.
* **Trần khi chỉ có hồ sơ thực tế** — xếp hạng bằng tải trọng ghi nhận đã lan toả.
  Đây mới là trần thật của điều kiện dữ liệu hiện có.

Chạy::

    python scripts/tune_risk.py --weights outputs/runs/yolo_crater/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demine.config import RunConfig  # noqa: E402
from demine.data.clearance import simulate_accidents, simulate_clearance  # noqa: E402
from demine.data.dataset import detections_to_grid, read_tile_metadata, write_dataset  # noqa: E402
from demine.data.geo import Grid  # noqa: E402
from demine.data.imagery import build_tiles  # noqa: E402
from demine.data.sorties import simulate_scene  # noqa: E402
from demine.data.terrain import build_terrain  # noqa: E402
from demine.evaluation.prioritisation import (  # noqa: E402
    accident_coverage,
    clearance_efficiency_curve,
    priority_index,
)
from demine.evaluation.spatial_cv import make_spatial_blocks, spatial_holdout  # noqa: E402
from demine.risk.features import FEATURE_NAMES, build_features, gaussian_spread  # noqa: E402
from demine.risk.model import RiskModel  # noqa: E402
from demine.utils import get_logger, seed_everything, setup_logging  # noqa: E402

logger = get_logger("tune")


def build_world(cfg: RunConfig, weights: Path, data_dir: Path):
    """Dựng lại vùng nghiên cứu và lớp hố bom phát hiện được."""
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    grid = Grid(cfg.grid)
    terrain = build_terrain(grid, cfg.terrain, rng)
    scene = simulate_scene(grid, terrain, cfg.sortie, cfg.ordnance, rng)

    tiles = build_tiles(scene, cfg.imagery, rng)
    root = Path(data_dir) / "imagery"
    if not (root / "data.yaml").exists():
        write_dataset(tiles, root)
    meta = read_tile_metadata(root)

    from demine.detect import build_detector

    detector = build_detector(cfg)
    detector.load(Path(weights))
    images = sorted(root.glob("images/*/*.png"))
    preds = detector.predict_sharded(images, conf=cfg.detect.confidence_threshold)
    crater_map, crater_diam, coverage = detections_to_grid(preds, meta, grid)

    features = build_features(
        scene, crater_map, crater_diam, cfg.risk, crater_coverage=coverage
    )
    tonnage = gaussian_spread(
        scene.recorded_tonnage, cfg.risk.record_kernel_radius_m / grid.cell
    )
    clearance = simulate_clearance(scene, tonnage, cfg.clearance, rng)
    accidents = simulate_accidents(scene, clearance, cfg.accident, rng)

    return dict(
        grid=grid,
        terrain=terrain,
        scene=scene,
        features=features,
        tonnage=tonnage,
        clearance=clearance,
        accidents=accidents,
    )


def measure_ceilings(w) -> dict:
    """Đo ba mức trần để biết mô hình còn dư địa bao nhiêu."""
    grid, scene, terrain = w["grid"], w["scene"], w["terrain"]
    items = scene.uxo_count.ravel()
    flat = grid.flat_index(w["accidents"].col, w["accidents"].row)

    def score(x, label):
        curve = clearance_efficiency_curve(x, items)
        cov = accident_coverage(x, flat)
        return {
            "muc": label,
            "thu_hoi_20pct": round(curve.recovered_at_20pct, 4),
            "bao_phu_tai_nan_20pct": round(cov.capture_at_20pct, 4),
        }

    return {
        "tran_tuyet_doi": score(items, "biết chính xác vật nổ ở đâu"),
        "tran_biet_moi_diem_roi": score(
            scene.true_tonnage.ravel(), "biết mọi điểm rơi"
        ),
        "tran_ho_so_thuc_te": score(
            w["tonnage"].ravel(), "chỉ có hồ sơ với sai số thật"
        ),
        "tran_phoi_nhiem": score(terrain.exposure.ravel(), "chỉ có mức phơi nhiễm"),
    }


def evaluate(w, cfg: RunConfig, exposure_weight: float, label: str) -> dict:
    """Huấn luyện tầng ba với một cấu hình và đo trên khối không gian giữ lại."""
    grid, scene = w["grid"], w["scene"]
    X = w["features"].X
    items = scene.uxo_count.ravel()
    is_cleared = w["clearance"].is_cleared.ravel()
    y_obs = (w["clearance"].items_found.ravel() > 0).astype(int)
    exposure = w["terrain"].exposure.ravel()

    blocks = make_spatial_blocks(
        w["features"].col, w["features"].row, cfg.risk.n_spatial_blocks, grid.shape
    )
    fit_mask, cal_mask = spatial_holdout(
        blocks[is_cleared], test_fraction=0.25, seed=cfg.risk.seed
    )

    model = RiskModel(cfg=cfg.risk)
    model.fit(
        X[is_cleared],
        y_obs[is_cleared],
        was_selected=is_cleared,
        X_all=X,
        feature_names=list(FEATURE_NAMES),
        calibration_mask=cal_mask,
    )
    prob = model.predict_probability(X)
    prio = priority_index(prob, exposure, exposure_weight=exposure_weight)

    curve = clearance_efficiency_curve(prio, items)
    flat = grid.flat_index(w["accidents"].col, w["accidents"].row)
    cov = accident_coverage(prio, flat)

    late = w["accidents"].year > cfg.accident.temporal_split_year
    cov_late = accident_coverage(
        prio,
        grid.flat_index(w["accidents"].col[late], w["accidents"].row[late])
        if late.any()
        else np.array([], dtype=int),
    )

    return {
        "cau_hinh": label,
        "trong_so_phoi_nhiem": exposure_weight,
        "thu_hoi_10pct": round(curve.recovered_at_10pct, 4),
        "thu_hoi_20pct": round(curve.recovered_at_20pct, 4),
        "thu_hoi_50pct": round(curve.recovered_at_50pct, 4),
        "bao_phu_tai_nan_20pct": round(cov.capture_at_20pct, 4),
        "bao_phu_tai_nan_chia_thoi_gian": round(cov_late.capture_at_20pct, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Quét tham số tầng ba của DeMine-VN.")
    ap.add_argument("--weights", required=True, help="Trọng số phát hiện hố bom")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs/tinh_chinh_tang_ba.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    setup_logging()
    cfg = RunConfig()
    if args.quick:
        cfg.apply_quick_mode()

    logger.info("Dựng lại vùng nghiên cứu và nạp trọng số đã huấn luyện…")
    w = build_world(cfg, Path(args.weights), Path(args.data_dir))

    logger.info("Đo các mức trần…")
    ceilings = measure_ceilings(w)
    for key, val in ceilings.items():
        logger.info(
            "  %-26s thu hồi @20%%: %.3f | bao phủ tai nạn: %.3f",
            val["muc"],
            val["thu_hoi_20pct"],
            val["bao_phu_tai_nan_20pct"],
        )

    logger.info("Quét trọng số mức độ phơi nhiễm trong chỉ số ưu tiên…")
    rows = []
    for ew in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]:
        row = evaluate(w, cfg, ew, f"trọng số phơi nhiễm = {ew:.2f}")
        rows.append(row)
        logger.info(
            "  ew=%.2f | thu hồi @20%%: %.3f | tai nạn @20%%: %.3f | tai nạn chia thời gian: %.3f",
            ew,
            row["thu_hoi_20pct"],
            row["bao_phu_tai_nan_20pct"],
            row["bao_phu_tai_nan_chia_thoi_gian"],
        )

    report = {"cac_muc_tran": ceilings, "quet_trong_so_phoi_nhiem": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Đã ghi kết quả quét: %s", out)

    best_recover = max(rows, key=lambda r: r["thu_hoi_20pct"])
    best_accident = max(rows, key=lambda r: r["bao_phu_tai_nan_20pct"])
    ceiling = ceilings["tran_ho_so_thuc_te"]["thu_hoi_20pct"]
    print()
    print("=" * 66)
    print(f"Trần của điều kiện dữ liệu hiện có   : {ceiling:.3f}")
    print(
        f"Tốt nhất về thu hồi vật nổ           : {best_recover['thu_hoi_20pct']:.3f} "
        f"(trọng số phơi nhiễm {best_recover['trong_so_phoi_nhiem']:.2f})"
    )
    print(
        f"Tốt nhất về bao phủ tai nạn          : {best_accident['bao_phu_tai_nan_20pct']:.3f} "
        f"(trọng số phơi nhiễm {best_accident['trong_so_phoi_nhiem']:.2f})"
    )
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
