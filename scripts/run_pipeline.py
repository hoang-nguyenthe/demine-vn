#!/usr/bin/env python
"""Chạy toàn bộ quy trình DeMine-VN từ dòng lệnh.

Ví dụ::

    python scripts/run_pipeline.py                     # chạy đầy đủ
    python scripts/run_pipeline.py --quick             # chế độ rút gọn
    python scripts/run_pipeline.py --no-train-detector # dùng lại trọng số đã có
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demine.config import RunConfig  # noqa: E402
from demine.pipeline import Pipeline  # noqa: E402
from demine.utils import gpu_utilisation, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Chạy quy trình DeMine-VN.")
    ap.add_argument("--quick", action="store_true", help="Chế độ rút gọn, chạy nhanh")
    ap.add_argument("--epochs", type=int, default=None, help="Số chu kỳ huấn luyện")
    ap.add_argument("--tiles", type=int, default=None, help="Số ảnh con")
    ap.add_argument("--missions", type=int, default=None, help="Số phi vụ mô phỏng")
    ap.add_argument("--seed", type=int, default=None, help="Hạt giống ngẫu nhiên")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument(
        "--backend",
        choices=["auto", "ultralytics", "torchvision"],
        default="auto",
        help="Phương án phát hiện",
    )
    ap.add_argument(
        "--no-train-detector",
        action="store_true",
        help="Bỏ qua huấn luyện, dùng trọng số đã có",
    )
    args = ap.parse_args()

    setup_logging()

    cfg = RunConfig()
    if args.quick:
        cfg.apply_quick_mode()
    if args.epochs is not None:
        cfg.detect.epochs = args.epochs
    if args.tiles is not None:
        cfg.imagery.n_tiles = args.tiles
    if args.missions is not None:
        cfg.sortie.n_missions = args.missions
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.detect.seed = args.seed
        cfg.risk.seed = args.seed
    cfg.detect.backend = args.backend
    cfg.output_dir = args.output_dir
    cfg.data_dir = args.data_dir

    gpu = gpu_utilisation()
    if gpu:
        print("Tình trạng GPU:")
        print(gpu)

    Pipeline(cfg).run(train_detector=not args.no_train_detector)

    out = Path(cfg.output_dir)
    print()
    print("=" * 62)
    print("HOÀN TẤT. Các tệp kết quả:")
    print(f"  {out / 'bao_cao_tong_hop.md'}")
    print(f"  {out / 'ket_qua.json'}")
    print(f"  {out / 'ban_do_uu_tien.html'}")
    print(f"  {out / 'danh_muc_uu_tien_ra_pha.csv'}")
    print(f"  {out / 'figures'}/")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
