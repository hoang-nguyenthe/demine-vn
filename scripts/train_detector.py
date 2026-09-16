#!/usr/bin/env python
"""Huấn luyện riêng tầng phát hiện hố bom, trong một tiến trình độc lập.

Lý do tách riêng: khi huấn luyện phân tán trên hai GPU, Ultralytics khởi tạo nhóm
tiến trình ở phía sau. Việc này không ổn định nếu gọi thẳng trong nhân của
notebook — nhân có thể treo hoặc giữ lại tiến trình con sau khi chạy xong. Gọi qua
một tiến trình độc lập thì mọi tài nguyên được giải phóng sạch khi kết thúc.

Ví dụ::

    python scripts/train_detector.py --data data/imagery/data.yaml --epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demine.config import RunConfig  # noqa: E402
from demine.detect import build_detector  # noqa: E402
from demine.utils import gpu_utilisation, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Huấn luyện tầng phát hiện hố bom.")
    ap.add_argument("--data", required=True, help="Đường dẫn tới data.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument(
        "--backend", choices=["auto", "ultralytics", "torchvision"], default="auto"
    )
    ap.add_argument(
        "--devices",
        default="0,1",
        help="Danh sách GPU, phân tách bằng dấu phẩy. Để trống nghĩa là chạy trên CPU.",
    )
    args = ap.parse_args()

    setup_logging()

    cfg = RunConfig()
    cfg.detect.epochs = args.epochs
    cfg.detect.batch_size = args.batch
    cfg.detect.image_size = args.imgsz
    cfg.detect.backend = args.backend
    cfg.output_dir = args.output_dir
    cfg.detect.devices = (
        [int(d) for d in args.devices.split(",") if d.strip() != ""]
        if args.devices
        else []
    )
    cfg.imagery.tile_size_px = args.imgsz

    gpu = gpu_utilisation()
    if gpu:
        print("Tình trạng GPU trước khi huấn luyện:")
        print(gpu)

    detector = build_detector(cfg)
    info = detector.train(Path(args.data), cfg)

    print()
    print("Đã huấn luyện xong.")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
