#!/usr/bin/env python
"""Kiểm thử nhanh toàn bộ hệ thống, chạy được trên CPU trong khoảng một phút.

Không cần pytest. Chạy::

    python tests/test_smoke.py

Mục đích là bắt lỗi hồi quy ở những chỗ dễ sai nhất về mặt phương pháp, chứ không
chỉ kiểm tra mã có chạy hay không. Cụ thể, có hai kiểm thử đáng chú ý:

* Kiểm thử số 5 xác nhận rằng việc chia tập **theo khối không gian** cho kết quả
  thấp hơn rõ rệt so với chia ngẫu nhiên theo điểm. Nếu hai cách cho kết quả như
  nhau thì việc chia khối đã bị hỏng, và mọi chỉ tiêu của đề tài sẽ bị thổi phồng.

* Kiểm thử số 7 xác nhận rằng hợp nhất ba nguồn tốt hơn từng nguồn riêng lẻ. Nếu
  không thì việc hợp nhất không mang lại giá trị và luận điểm trung tâm của đề tài
  không đứng vững.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demine.config import RunConfig  # noqa: E402
from demine.data.clearance import simulate_accidents, simulate_clearance  # noqa: E402
from demine.data.dataset import write_dataset  # noqa: E402
from demine.data.geo import Grid  # noqa: E402
from demine.data.imagery import build_tiles  # noqa: E402
from demine.data.sorties import simulate_scene  # noqa: E402
from demine.data.terrain import build_terrain  # noqa: E402
from demine.evaluation.calibration import (  # noqa: E402
    expected_calibration_error,
    fit_isotonic,
)
from demine.evaluation.consistency import spearman_correlation  # noqa: E402
from demine.evaluation.detection_metrics import evaluate_detections, iou_matrix  # noqa: E402
from demine.evaluation.prioritisation import (  # noqa: E402
    accident_coverage,
    clearance_efficiency_curve,
)
from demine.evaluation.spatial_cv import make_spatial_blocks, spatial_holdout  # noqa: E402
from demine.risk.features import FEATURE_NAMES, build_features, gaussian_spread  # noqa: E402
from demine.risk.model import RiskModel  # noqa: E402
from demine.utils import setup_logging  # noqa: E402

PASSED: list = []
FAILED: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ✔ {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  ✘ {name}{(' — ' + detail) if detail else ''}")


def build_world(seed: int = 5):
    cfg = RunConfig().apply_quick_mode()
    grid = Grid(cfg.grid)
    rng = np.random.default_rng(seed)
    terrain = build_terrain(grid, cfg.terrain, rng)
    scene = simulate_scene(grid, terrain, cfg.sortie, cfg.ordnance, rng)
    return cfg, grid, terrain, scene, rng


def main() -> int:
    setup_logging(logging.WARNING)
    print("=" * 62)
    print("KIỂM THỬ NHANH — DeMine-VN")
    print("=" * 62)

    cfg, grid, terrain, scene, rng = build_world()

    # 1 ---------------------------------------------------------------
    print("\n1. Hệ quy chiếu và lưới")
    col, row = grid.xy_to_index(np.array([0.0, 550.0]), np.array([0.0, 250.0]))
    check("chuyển toạ độ sang chỉ số ô", (col[1] == 5) and (row[1] == 2))
    lon, lat = grid.xy_to_lonlat(np.array([0.0]), np.array([0.0]))
    check(
        "gốc lưới khớp toạ độ quy ước",
        abs(float(lon[0]) - grid.cfg.origin_lon) < 1e-9
        and abs(float(lat[0]) - grid.cfg.origin_lat) < 1e-9,
    )

    # 2 ---------------------------------------------------------------
    print("\n2. Mô phỏng vật lý")
    check("có điểm rơi", scene.n_impacts > 100, f"{scene.n_impacts} điểm rơi")
    check("có vật nổ còn sót", scene.n_uxo > 20, f"{scene.n_uxo} vật nổ")
    dud_rate = scene.n_uxo / max(1, scene.n_impacts)
    check(
        "tỉ lệ không nổ nằm trong dải hợp lý",
        0.08 <= dud_rate <= 0.35,
        f"{dud_rate:.3f}",
    )

    soft = terrain.soil_substrate if hasattr(terrain, "soil_substrate") else terrain.soil_softness
    c, r = grid.xy_to_index(scene.impact_x, scene.impact_y)
    soft_at_impact = soft[r, c]
    dud_soft = soft_at_impact[scene.impact_is_dud].mean()
    dud_hard = soft_at_impact[~scene.impact_is_dud].mean()
    check(
        "bom không nổ tập trung ở nền đất mềm hơn",
        dud_soft > dud_hard,
        f"{dud_soft:.3f} so với {dud_hard:.3f}",
    )

    vis_soft = soft_at_impact[scene.impact_crater_visible].mean()
    check(
        "hố bom quan sát được tập trung ở nền cứng hơn",
        vis_soft < soft_at_impact.mean(),
        f"{vis_soft:.3f} so với {soft_at_impact.mean():.3f}",
    )

    # 3 ---------------------------------------------------------------
    print("\n3. Ảnh vệ tinh và bộ dữ liệu")
    cfg.imagery.n_tiles = 6
    tiles = build_tiles(scene, cfg.imagery, rng)
    n_boxes = sum(t.boxes.shape[0] for t in tiles)
    check("kết xuất được ảnh con", len(tiles) == 6)
    check("ảnh con có nhãn hố bom", n_boxes > 0, f"{n_boxes} hố bom")

    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = write_dataset(tiles, Path(tmp) / "imagery")
        check("ghi được bộ dữ liệu theo quy ước YOLO", yaml_path.exists())
        n_png = len(list((Path(tmp) / "imagery").glob("images/*/*.png")))
        check("ảnh được ghi đầy đủ", n_png == len(tiles), f"{n_png} ảnh")

    # 4 ---------------------------------------------------------------
    print("\n4. Chỉ tiêu phát hiện")
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    m = iou_matrix(a, b)
    check("IoU trùng khớp hoàn toàn bằng 1", abs(m[0, 0] - 1.0) < 1e-9)
    check("IoU rời nhau bằng 0", abs(m[0, 1]) < 1e-9)

    class _Det:
        def __init__(self, image_id, boxes, scores):
            self.image_id, self.boxes, self.scores = image_id, boxes, scores

    preds = [_Det("t", np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.9]))]
    gt = {"t": np.array([[0.0, 0.0, 10.0, 10.0]])}
    met = evaluate_detections(preds, gt, 0.25)
    check("dự báo hoàn hảo cho mAP bằng 1", met.ap50 > 0.99, f"mAP {met.ap50:.3f}")

    # 5 ---------------------------------------------------------------
    print("\n5. Chia tập theo khối không gian")
    crater_map = scene.crater_count.astype(float)
    feats = build_features(scene, crater_map, np.zeros(grid.shape), cfg.risk)
    X = feats.X
    y = (scene.uxo_count.ravel() > 0).astype(int)
    items = scene.uxo_count.ravel().astype(float)

    blocks = make_spatial_blocks(
        feats.col, feats.row, cfg.risk.n_spatial_blocks, grid.shape
    )
    check(
        "sinh được nhiều khối không gian",
        len(np.unique(blocks)) >= 4,
        f"{len(np.unique(blocks))} khối",
    )

    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_score(train_mask, test_mask):
        m = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, min_samples_leaf=25, random_state=0
        )
        m.fit(X[train_mask], y[train_mask])
        p = m.predict_proba(X[test_mask])[:, 1]
        return clearance_efficiency_curve(p, items[test_mask]).recovered_at_20pct

    tr_block, te_block = spatial_holdout(blocks, 0.3, seed=1)
    rng2 = np.random.default_rng(0)
    rand_mask = rng2.random(X.shape[0]) < 0.7
    score_block = fit_score(tr_block, te_block)
    score_random = fit_score(rand_mask, ~rand_mask)
    check(
        "chia ngẫu nhiên theo điểm cho kết quả cao hơn chia theo khối",
        score_random > score_block,
        f"ngẫu nhiên {score_random:.3f} so với khối {score_block:.3f}",
    )
    check(
        "chia theo khối vẫn tốt hơn quét trải đều",
        score_block > 0.25,
        f"{score_block:.3f} so với 0.20",
    )

    # 6 ---------------------------------------------------------------
    print("\n6. Kiểm chứng và hiệu chỉnh")
    ts = gaussian_spread(
        scene.recorded_tonnage, cfg.risk.record_kernel_radius_m / grid.cell
    )
    mask = (ts > 1e-9) | (crater_map > 1e-9)
    rho = spearman_correlation(ts[mask], crater_map[mask])
    check(
        "hai nguồn độc lập tương quan dương",
        rho > 0.3,
        f"Spearman {rho:.3f}",
    )

    clearance = simulate_clearance(scene, ts, cfg.clearance, rng)
    accidents = simulate_accidents(scene, clearance, cfg.accident, rng)
    check("có khoảnh đất đã rà phá", clearance.n_cleared_cells > 50)
    check("có hồ sơ tai nạn", accidents.n > 5, f"{accidents.n} vụ")

    acc_flat = grid.flat_index(accidents.col, accidents.row)
    m = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.08, min_samples_leaf=25, random_state=0
    )
    m.fit(X[tr_block], y[tr_block])
    prob_all = m.predict_proba(X)[:, 1]
    cov = accident_coverage(prob_all, acc_flat)
    check(
        "bao phủ tai nạn cao hơn mức ngẫu nhiên",
        cov.capture_at_20pct > 0.25,
        f"{cov.capture_at_20pct:.3f} so với 0.20",
    )

    cal = fit_isotonic(np.array([0.1, 0.2, 0.7, 0.9]), np.array([0.0, 0.0, 1.0, 1.0]))
    out = cal.predict(np.array([0.15, 0.8]))
    check("hồi quy đẳng hướng đơn điệu", out[1] >= out[0])
    ece = expected_calibration_error(
        np.array([0.1] * 90 + [0.9] * 10), np.array([0] * 90 + [1] * 10), 10
    )
    check("sai số hiệu chỉnh nhỏ với dự báo tốt", ece < 0.15, f"{ece:.3f}")

    # 7 ---------------------------------------------------------------
    print("\n7. Giá trị của việc hợp nhất ba nguồn")
    groups = {
        "chỉ hồ sơ": [
            "tai_trong_ghi_nhan_lan_toa",
            "tai_trong_lan_can_500m",
            "mat_do_phi_vu_lan_toa",
            "ky_vong_bom_khong_no",
        ],
        "chỉ hố bom": [
            "ho_bom_mat_do",
            "ho_bom_lan_can_300m",
            "ho_bom_duong_kinh_tb",
            "ho_bom_hieu_chinh_tam_nhin",
        ],
    }
    scores = {}
    for label, names in groups.items():
        cols = [FEATURE_NAMES.index(n) for n in names]
        mm = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, min_samples_leaf=25, random_state=0
        )
        mm.fit(X[tr_block][:, cols], y[tr_block])
        p = mm.predict_proba(X[te_block][:, cols])[:, 1]
        scores[label] = clearance_efficiency_curve(
            p, items[te_block]
        ).recovered_at_20pct

    scores["đầy đủ"] = score_block
    check(
        "hợp nhất tốt hơn từng nguồn riêng lẻ",
        scores["đầy đủ"] >= max(scores["chỉ hồ sơ"], scores["chỉ hố bom"]) - 1e-9,
        " | ".join(f"{k} {v:.3f}" for k, v in scores.items()),
    )

    # 8 ---------------------------------------------------------------
    print("\n8. Mô hình nguy cơ đầy đủ với hiệu chỉnh thiên lệch")
    is_cleared = clearance.is_cleared.ravel()
    y_obs = (clearance.items_found.ravel() > 0).astype(int)
    model = RiskModel(cfg=cfg.risk)
    model.fit(
        X[is_cleared],
        y_obs[is_cleared],
        was_selected=is_cleared,
        X_all=X,
        feature_names=list(FEATURE_NAMES),
    )
    p = model.predict_probability(X)
    check("xác suất nằm trong khoảng hợp lệ", bool((p >= 0).all() and (p <= 1).all()))
    check(
        "mô hình phân biệt được ô có và không có vật nổ",
        p[y == 1].mean() > p[y == 0].mean(),
        f"{p[y == 1].mean():.4f} so với {p[y == 0].mean():.4f}",
    )
    imp = model.feature_importance()
    check("giải thích được đóng góp đặc trưng", len(imp) > 0, f"{len(imp)} đặc trưng")

    # 9 ---------------------------------------------------------------
    print("\n9. Chỉ số ưu tiên phải thực sự đổi thứ hạng")
    from demine.evaluation.prioritisation import normalised_rank, priority_index

    rank_score = model.predict_ranking_score(X)
    n_tied = int((p == np.quantile(p, 0.5)).sum())
    n_tied_rank = int((rank_score == np.quantile(rank_score, 0.5)).sum())
    check(
        "điểm xếp hạng phá được thế hoà của xác suất",
        n_tied_rank < max(1, n_tied // 10),
        f"{n_tied} ô hoà theo xác suất, còn {n_tied_rank} theo điểm xếp hạng",
    )

    prio0 = priority_index(rank_score, expo_all := terrain.exposure.ravel(), 0.0)
    prio1 = priority_index(rank_score, expo_all, 0.35)
    top0 = set(np.argsort(-prio0)[: int(0.2 * prio0.size)])
    top1 = set(np.argsort(-prio1)[: int(0.2 * prio1.size)])
    changed = 1.0 - len(top0 & top1) / max(1, len(top0))
    check(
        "trọng số phơi nhiễm thực sự làm đổi danh mục ưu tiên",
        changed > 0.02,
        f"{100*changed:.1f}% số ô trong nhóm 20% đầu bị thay",
    )

    cov0 = accident_coverage(prio0, acc_flat).capture_at_20pct
    cov1 = accident_coverage(prio1, acc_flat).capture_at_20pct
    check(
        "trộn mức phơi nhiễm cải thiện bao phủ tai nạn",
        cov1 >= cov0 - 1e-9,
        f"{cov0:.3f} lên {cov1:.3f}",
    )

    # ------------------------------------------------------------------
    print()
    print("=" * 62)
    if FAILED:
        print(f"CHƯA ĐẠT — {len(FAILED)} kiểm thử thất bại:")
        for f in FAILED:
            print(f"   - {f}")
        print("=" * 62)
        return 1

    print(f"TOÀN BỘ {len(PASSED)} KIỂM THỬ ĐÃ ĐẠT — mã nguồn sẵn sàng chạy trên Kaggle.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
