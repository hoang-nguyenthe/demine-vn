"""Điều phối toàn bộ quy trình bốn tầng, từ dữ liệu đến bản đồ ưu tiên.

Trình tự chạy:

  1. Dựng vùng nghiên cứu: địa hình, thổ nhưỡng, hiện trạng sử dụng đất.
  2. Mô phỏng phi vụ không kích, điểm rơi, vật nổ còn sót và hồ sơ ghi chép.
  3. Kết xuất ảnh vệ tinh lịch sử và ghi bộ dữ liệu theo quy ước YOLO.
  4. Huấn luyện và đánh giá mô hình phát hiện hố bom  (tầng hai).
  5. Quy hố bom phát hiện được về lưới, dựng bảng đặc trưng bốn nhóm.
  6. Mô phỏng hoạt động rà phá đã thực hiện và hồ sơ tai nạn.
  7. Huấn luyện mô hình nguy cơ trên phần đất đã rà phá  (tầng ba).
  8. Chạy đủ bốn tầng kiểm chứng.
  9. Sinh bản đồ, hình minh hoạ, danh mục ưu tiên và báo cáo tổng hợp  (tầng bốn).

Mọi bước đều ghi nhật ký và mọi kết quả trung gian đều được lưu ra đĩa, để có thể
chạy lại từng phần mà không phải chạy lại toàn bộ.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import RunConfig
from .data.clearance import simulate_accidents, simulate_clearance
from .data.dataset import (
    detections_to_grid,
    read_ground_truth,
    read_tile_metadata,
    write_dataset,
)
from .data.geo import Grid
from .data.imagery import build_tiles
from .data.sorties import BOMB_MASS_KG, simulate_scene
from .data.terrain import build_terrain
from .evaluation.ablation import run_ablation
from .evaluation.calibration import calibration_report
from .evaluation.consistency import check_consistency
from .evaluation.detection_metrics import evaluate_detections
from .evaluation.prioritisation import (
    accident_coverage,
    clearance_efficiency_curve,
    priority_index,
)
from .evaluation.spatial_cv import (
    block_kfold,
    make_spatial_blocks,
    spatial_holdout,
    transfer_split,
)
from .risk.features import FEATURE_NAMES, build_features, gaussian_spread
from .risk.model import RiskModel, permutation_importance
from .utils import describe_environment, ensure_dir, get_logger, seed_everything

logger = get_logger(__name__)


class Pipeline:
    """Quy trình đầu cuối của DeMine-VN."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.out = ensure_dir(cfg.output_dir)
        self.data_dir = ensure_dir(cfg.data_dir)
        self.results: Dict[str, object] = {}
        self.rng = np.random.default_rng(cfg.seed)
        seed_everything(cfg.seed)

    # ------------------------------------------------------------------
    def step_1_build_area(self):
        logger.info("Bước 1 — dựng vùng nghiên cứu")
        self.grid = Grid(self.cfg.grid)
        self.terrain = build_terrain(self.grid, self.cfg.terrain, self.rng)
        logger.info(
            "  Vùng nghiên cứu %.1f × %.1f km | %d ô lưới cạnh %.0f m",
            self.grid.width_m / 1000.0,
            self.grid.height_m / 1000.0,
            self.grid.n_cells,
            self.grid.cell,
        )
        return self

    def step_2_simulate(self):
        logger.info("Bước 2 — mô phỏng phi vụ và vật nổ còn sót")
        self.scene = simulate_scene(
            self.grid, self.terrain, self.cfg.sortie, self.cfg.ordnance, self.rng
        )
        return self

    def step_3_imagery(self):
        logger.info("Bước 3 — kết xuất ảnh vệ tinh lịch sử")
        self.tiles = build_tiles(self.scene, self.cfg.imagery, self.rng)
        self.data_yaml = write_dataset(self.tiles, self.data_dir / "imagery")
        self.tile_meta = read_tile_metadata(self.data_dir / "imagery")
        return self

    def step_4_detect(self, detector=None, train: bool = True):
        logger.info("Bước 4 — phát hiện hố bom trên ảnh (tầng hai)")
        from .detect import build_detector

        detector = detector or build_detector(self.cfg)
        self.detector = detector

        if train:
            info = detector.train(self.data_yaml, self.cfg)
            logger.info("  Đã huấn luyện: %s", info.get("weights", "(không rõ)"))
        else:
            # Không huấn luyện thì phải nạp trọng số đã có, nếu không tầng hai sẽ
            # không có mô hình nào để suy luận.
            runs = Path(self.cfg.output_dir).resolve() / "runs"
            candidates = [
                runs / "yolo_crater" / "weights" / "best.pt",
                runs / "frcnn_crater" / "weights" / "best.pt",
            ]
            weights = next((c for c in candidates if c.exists()), None)
            if weights is None:
                found = sorted(
                    runs.rglob("weights/best.pt"),
                    key=lambda q: q.stat().st_mtime,
                    reverse=True,
                )
                weights = found[0] if found else None
            if weights is None:
                raise FileNotFoundError(
                    "Không tìm thấy trọng số đã huấn luyện trong "
                    f"{Path(self.cfg.output_dir) / 'runs'}. Hãy bỏ tuỳ chọn "
                    "--no-train-detector, hoặc trỏ --output-dir tới thư mục đã có "
                    "trọng số."
                )
            logger.info("  Nạp trọng số đã có: %s", weights)
            detector.load(weights)

        root = self.data_dir / "imagery"
        test_images = sorted((root / "images" / "test").glob("*.png"))
        preds = detector.predict_sharded(
            test_images, conf=self.cfg.detect.confidence_threshold
        )
        gt = read_ground_truth(root, "test")
        metrics = evaluate_detections(
            preds, gt, score_threshold=self.cfg.detect.confidence_threshold
        )
        self.results["tang_hai_phat_hien_ho_bom"] = metrics.as_dict()
        logger.info(
            "  mAP@0.5 = %.3f | Recall = %.3f | %d hố bom thật, %d dự báo",
            metrics.ap50,
            metrics.recall,
            metrics.n_true,
            metrics.n_pred,
        )

        # Suy luận trên toàn bộ ảnh để dựng lớp hố bom phủ khắp vùng nghiên cứu.
        all_images = sorted(root.glob("images/*/*.png"))
        all_preds = detector.predict_sharded(
            all_images, conf=self.cfg.detect.confidence_threshold
        )
        self.crater_map, self.crater_diameter_map, self.crater_coverage = (
            detections_to_grid(all_preds, self.tile_meta, self.grid)
        )
        n_craters = int(self.crater_map.sum())
        logger.info("  Đã quy %d hố bom phát hiện được về lưới", n_craters)

        # Nếu tầng hai gần như không phát hiện được gì thì nhóm đặc trưng quan sát
        # sẽ toàn số không, và mọi kết quả liên quan tới nguồn ảnh vệ tinh sẽ bằng
        # không theo. Đó là hệ quả của việc huấn luyện chưa đủ, không phải kết luận
        # khoa học, nên phải được nêu rõ thay vì để người đọc tự suy diễn.
        expected = max(1, int(0.05 * self.scene.impact_crater_visible.sum()))
        self.detector_underfitted = n_craters < expected
        if self.detector_underfitted:
            logger.warning(
                "Tầng hai chỉ phát hiện %d hố bom, quá thấp so với mức kỳ vọng. "
                "Nhóm đặc trưng ảnh vệ tinh sẽ gần như không mang thông tin. "
                "Hãy tăng số chu kỳ huấn luyện hoặc chạy trên GPU.",
                n_craters,
            )
        coverage_fraction = float(self.crater_coverage.mean())
        logger.info(
            "  Ảnh vệ tinh phủ %.1f%% diện tích vùng nghiên cứu; phần còn lại được "
            "đánh dấu là khuyết dữ liệu chứ không phải bằng không.",
            100.0 * coverage_fraction,
        )
        self.results["canh_bao_tang_hai"] = {
            "so_ho_bom_phat_hien_toan_vung": n_craters,
            "ty_le_dien_tich_co_anh_ve_tinh": round(coverage_fraction, 4),
            "huan_luyen_chua_du": bool(self.detector_underfitted),
        }
        return self

    def step_5_features(self):
        logger.info("Bước 5 — dựng bảng đặc trưng bốn nhóm (tầng ba, phần một)")
        self.features = build_features(
            self.scene,
            self.crater_map,
            self.crater_diameter_map,
            self.cfg.risk,
            crater_coverage=self.crater_coverage,
        )
        self.tonnage_spread = gaussian_spread(
            self.scene.recorded_tonnage,
            self.cfg.risk.record_kernel_radius_m / self.grid.cell,
        )
        logger.info(
            "  %d ô × %d đặc trưng", self.features.n_rows, len(FEATURE_NAMES)
        )
        return self

    def step_6_clearance(self):
        logger.info("Bước 6 — mô phỏng rà phá đã thực hiện và hồ sơ tai nạn")
        self.clearance = simulate_clearance(
            self.scene, self.tonnage_spread, self.cfg.clearance, self.rng
        )
        self.accidents = simulate_accidents(
            self.scene, self.clearance, self.cfg.accident, self.rng
        )
        return self

    # ------------------------------------------------------------------
    def _prepare_matrices(self):
        """Chuẩn bị các mảng phẳng dùng chung cho tầng ba và phần kiểm chứng."""
        self.y_true = (self.scene.uxo_count.ravel() > 0).astype(int)
        self.items = self.scene.uxo_count.ravel().astype(float)
        self.is_cleared = self.clearance.is_cleared.ravel()
        self.items_found = self.clearance.items_found.ravel().astype(float)
        self.y_observed = (self.items_found > 0).astype(int)
        self.exposure = self.terrain.exposure.ravel()

        self.blocks = make_spatial_blocks(
            self.features.col,
            self.features.row,
            self.cfg.risk.n_spatial_blocks,
            self.grid.shape,
        )
        self.accident_flat = (
            self.grid.flat_index(self.accidents.col, self.accidents.row)
            if self.accidents.n
            else np.array([], dtype=int)
        )

    def step_7_risk_model(self):
        logger.info("Bước 7 — huấn luyện mô hình nguy cơ (tầng ba, phần hai)")
        self._prepare_matrices()

        X = self.features.X
        labelled = self.is_cleared

        # Tập hiệu chỉnh xác suất được tách riêng theo khối, không dùng để học.
        fit_mask, cal_mask = spatial_holdout(
            self.blocks[labelled], test_fraction=0.25, seed=self.cfg.risk.seed
        )

        self.model = RiskModel(cfg=self.cfg.risk)
        self.model.fit(
            X[labelled],
            self.y_observed[labelled],
            was_selected=labelled,
            X_all=X,
            feature_names=list(FEATURE_NAMES),
            calibration_mask=cal_mask,
        )

        self.probability = self.model.predict_probability(X)
        # Điểm xếp hạng khác xác suất: xem giải thích ở RiskModel.predict_ranking_score.
        self.ranking_score = self.model.predict_ranking_score(X)
        self.priority = priority_index(
            self.ranking_score,
            self.exposure,
            exposure_weight=self.cfg.risk.priority_exposure_weight,
        )
        logger.info(
            "  Xác suất trung bình %.4f | kỳ vọng tổng số vật nổ còn lại %.0f",
            float(self.probability.mean()),
            float(self.probability.sum()),
        )
        return self

    # ------------------------------------------------------------------
    def step_8_validate(self):
        logger.info("Bước 8 — kiểm chứng bốn tầng")
        X = self.features.X

        # -- Tầng một: kiểm chứng chéo theo khối trên đất đã rà phá -------
        logger.info("  Tầng 1 — đối chứng trên đất đã rà phá, chia theo khối không gian")
        labelled_idx = np.flatnonzero(self.is_cleared)
        fold_scores, fold_gains = [], []

        for train_mask, test_mask in block_kfold(
            self.blocks[labelled_idx], self.cfg.risk.n_cv_folds, seed=self.cfg.risk.seed
        ):
            tr = labelled_idx[train_mask]
            te = labelled_idx[test_mask]
            if te.size < 20 or self.y_observed[tr].sum() < 5:
                continue
            m = RiskModel(cfg=self.cfg.risk)
            m.fit(
                X[tr],
                self.y_observed[tr],
                was_selected=self.is_cleared,
                X_all=X,
                feature_names=list(FEATURE_NAMES),
            )
            p = m.predict_ranking_score(X[te])
            curve = clearance_efficiency_curve(p, self.items_found[te])
            fold_scores.append(curve.recovered_at_20pct)
            fold_gains.append(curve.gain_over_uniform)

        self.results["tang_1_doi_chung_dat_da_ra_pha"] = {
            "so_lan_chia": len(fold_scores),
            "thu_hoi_tai_20pct_trung_binh": round(float(np.mean(fold_scores)), 4)
            if fold_scores
            else 0.0,
            "do_lech_chuan": round(float(np.std(fold_scores)), 4) if fold_scores else 0.0,
            "loi_the_so_voi_quet_deu": round(float(np.mean(fold_gains)), 4)
            if fold_gains
            else 0.0,
        }
        logger.info(
            "    thu hồi tại 20%% diện tích: %.3f ± %.3f qua %d lần chia",
            float(np.mean(fold_scores)) if fold_scores else 0.0,
            float(np.std(fold_scores)) if fold_scores else 0.0,
            len(fold_scores),
        )

        # Đường cong hiệu quả trên toàn vùng, dùng nhãn thật để báo cáo.
        self.curve = clearance_efficiency_curve(self.priority, self.items)
        self.results["duong_cong_hieu_qua_ra_pha"] = self.curve.as_dict()

        # -- Tầng hai: kiểm chứng độc lập bằng hồ sơ tai nạn --------------
        logger.info("  Tầng 2 — đối chứng độc lập bằng hồ sơ tai nạn")
        cov_all = accident_coverage(self.priority, self.accident_flat, "toàn bộ")

        late = self.accidents.year > self.cfg.accident.temporal_split_year
        cov_late = accident_coverage(
            self.priority,
            self.grid.flat_index(self.accidents.col[late], self.accidents.row[late])
            if late.any()
            else np.array([], dtype=int),
            f"sau năm thứ {self.cfg.accident.temporal_split_year}",
        )

        self.results["tang_2_doi_chung_ho_so_tai_nan"] = [
            cov_all.as_dict(),
            cov_late.as_dict(),
        ]
        logger.info(
            "    bao phủ tại 20%% diện tích: %.3f (toàn bộ) | %.3f (chia theo thời gian)",
            cov_all.capture_at_20pct,
            cov_late.capture_at_20pct,
        )

        # -- Tầng ba: nhất quán giữa hai nguồn độc lập --------------------
        logger.info("  Tầng 3 — nhất quán giữa hồ sơ không kích và ảnh vệ tinh")
        n_recorded_bombs = int(self.scene.record_n_bombs.sum())
        # Chỉ so sánh trên phần có ảnh vệ tinh. So trên vùng không có ảnh là so mật
        # độ hố bom bằng không với tải trọng bom khác không, và sẽ cho hệ số gần bằng
        # không vì lý do hoàn toàn nhân tạo.
        cov = self.crater_coverage
        consistency = check_consistency(
            self.crater_map[cov],
            self.tonnage_spread[cov],
            n_recorded_bombs,
            self.cfg.ordnance.base_dud_rate,
            self.probability,
        )
        self.results["tang_3_nhat_quan_hai_nguon"] = consistency.as_dict()
        logger.info(
            "    Spearman = %.3f trên %d ô | tỉ số bậc độ lớn = %.2f",
            consistency.spearman_crater_vs_tonnage,
            consistency.n_cells_compared,
            consistency.order_of_magnitude_ratio,
        )

        # -- Tầng bốn: chuyển vùng và hiệu chỉnh --------------------------
        logger.info("  Tầng 4 — chuyển vùng địa lý và chất lượng hiệu chỉnh")
        west, east = transfer_split(self.features.col, self.grid.shape)

        base = self.results["tang_1_doi_chung_dat_da_ra_pha"][
            "thu_hoi_tai_20pct_trung_binh"
        ]
        transfer_value = 0.0
        train_sel = west & self.is_cleared
        test_sel = east & self.is_cleared
        if train_sel.sum() > 50 and test_sel.sum() > 20 and self.y_observed[train_sel].sum() >= 5:
            m = RiskModel(cfg=self.cfg.risk)
            m.fit(
                X[train_sel],
                self.y_observed[train_sel],
                was_selected=self.is_cleared,
                X_all=X,
                feature_names=list(FEATURE_NAMES),
            )
            p = m.predict_ranking_score(X[test_sel])
            transfer_value = clearance_efficiency_curve(
                p, self.items_found[test_sel]
            ).recovered_at_20pct

        drop = (base - transfer_value) / base if base > 1e-9 else 0.0
        self.results["tang_4a_chuyen_vung_dia_ly"] = {
            "thu_hoi_tai_20pct_cung_vung": round(base, 4),
            "thu_hoi_tai_20pct_vung_moi": round(transfer_value, 4),
            "muc_suy_giam_tuong_doi": round(drop, 4),
        }
        logger.info(
            "    huấn luyện nửa tây, áp dụng nửa đông: %.3f → %.3f (suy giảm %.1f%%)",
            base,
            transfer_value,
            100.0 * drop,
        )

        cal = calibration_report(
            self.probability[self.is_cleared],
            self.y_observed[self.is_cleared],
            self.cfg.risk.n_calibration_bins,
        )
        self.results["tang_4b_hieu_chinh_xac_suat"] = cal
        logger.info(
            "    sai số hiệu chỉnh kỳ vọng = %.4f | điểm Brier = %.4f",
            cal["sai_so_hieu_chinh_ky_vong"],
            cal["diem_brier"],
        )
        return self

    def step_9_ablation(self):
        logger.info("Bước 9 — phân tích đóng góp thành phần")
        X = self.features.X
        train_mask, test_mask = spatial_holdout(
            self.blocks, test_fraction=0.3, seed=self.cfg.risk.seed + 1
        )
        tr = np.flatnonzero(train_mask & self.is_cleared)
        te = np.flatnonzero(test_mask)

        acc_test = (
            np.array(
                [i for i in self.accident_flat if test_mask[i]], dtype=int
            )
            if self.accident_flat.size
            else np.array([], dtype=int)
        )
        # Chỉ số tai nạn phải được quy về vị trí trong tập kiểm tra.
        remap = -np.ones(test_mask.size, dtype=int)
        remap[te] = np.arange(te.size)
        acc_test_local = remap[acc_test]
        acc_test_local = acc_test_local[acc_test_local >= 0]

        rows = run_ablation(
            X[tr],
            self.y_observed[tr],
            self.is_cleared,
            X,
            X[te],
            self.items[te],
            acc_test_local,
            self.cfg.risk,
            exposure_test=self.exposure[te],
        )
        self.results["phan_tich_dong_gop_thanh_phan"] = [r.as_dict() for r in rows]

        self.results["muc_dong_gop_dac_trung"] = self.model.feature_importance()
        self.results["muc_dong_gop_theo_hoan_vi"] = permutation_importance(
            self.model, X[te], self.items[te], n_repeats=2, seed=self.cfg.risk.seed
        )
        return self

    def step_10_outputs(self):
        logger.info("Bước 10 — kết xuất bản đồ, hình minh hoạ và báo cáo")
        from .reporting import write_priority_list, write_report
        from .viz.figures import make_all_figures
        from .viz.maps import make_priority_map

        fig_dir = ensure_dir(self.out / "figures")
        make_all_figures(self, fig_dir)
        make_priority_map(self, self.out / "ban_do_uu_tien.html")
        write_priority_list(self, self.out / "danh_muc_uu_tien_ra_pha.csv")

        self.results["moi_truong_chay"] = describe_environment()
        (self.out / "ket_qua.json").write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_report(self, self.out / "bao_cao_tong_hop.md")
        logger.info("  Đã ghi toàn bộ kết quả vào %s", self.out)
        return self

    # ------------------------------------------------------------------
    def run(self, detector=None, train_detector: bool = True):
        (
            self.step_1_build_area()
            .step_2_simulate()
            .step_3_imagery()
            .step_4_detect(detector=detector, train=train_detector)
            .step_5_features()
            .step_6_clearance()
            .step_7_risk_model()
            .step_8_validate()
            .step_9_ablation()
            .step_10_outputs()
        )
        return self.results
