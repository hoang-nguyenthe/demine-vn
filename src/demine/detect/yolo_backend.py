"""Bộ phát hiện hố bom dựa trên Ultralytics YOLO, khai thác đồng thời hai GPU T4.

Huấn luyện phân tán được kích hoạt bằng cách truyền danh sách nhiều thiết bị cho
tham số ``device``. Ultralytics tự khởi tạo tiến trình phân tán ở phía sau; để quá
trình này ổn định trên Kaggle, việc huấn luyện nên được gọi từ một tiến trình con
độc lập thay vì gọi trực tiếp trong nhân của notebook. Tệp
``scripts/train_detector.py`` đảm nhiệm vai trò đó.

Cấu hình tăng cường dữ liệu được điều chỉnh riêng cho ảnh trinh sát lịch sử: ảnh
đơn sắc nên tắt toàn bộ phép biến đổi màu; hố bom không có hướng ưu tiên nên xoay
toàn dải và lật theo cả hai trục đều hợp lệ; hố bom là đối tượng nhỏ nên hạn chế
biên độ co giãn để không làm mất đối tượng.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils import get_logger
from .interface import BaseDetector, DetectionOutput

LOG = get_logger("detect.yolo")


def resolve_model_spec(preferred: str = "yolo11n.pt") -> str:
    """Chọn điểm khởi tạo mô hình phù hợp với điều kiện kết nối mạng.

    Trọng số đã huấn luyện trước cần tải về từ máy chủ của Ultralytics. Khi
    notebook chạy ở chế độ ngắt mạng, hệ thống chuyển sang khởi tạo mô hình từ tệp
    mô tả kiến trúc — quá trình huấn luyện vẫn diễn ra bình thường, chỉ cần thêm
    một số chu kỳ để hội tụ.
    """
    local = Path(preferred)
    if local.exists():
        return str(local)

    try:
        import urllib.request

        urllib.request.urlopen("https://github.com", timeout=6)
        return preferred
    except Exception:
        fallback = preferred.replace(".pt", ".yaml")
        LOG.warning(
            "Không có kết nối mạng: khởi tạo mô hình từ kiến trúc %s thay vì "
            "trọng số huấn luyện trước.",
            fallback,
        )
        return fallback


class YOLODetector(BaseDetector):
    name = "ultralytics"

    def __init__(self, devices: List[int], cfg) -> None:
        self.devices = list(devices)
        self.cfg = cfg
        self.model = None
        self.weights_path: Optional[Path] = None

    # -- Huấn luyện --------------------------------------------------------
    def train(self, data_yaml: Path, cfg=None) -> Dict:
        from ultralytics import YOLO

        cfg = cfg or self.cfg
        spec = resolve_model_spec(cfg.detect.model_spec)
        self.model = YOLO(spec)

        from ..utils import probe_devices

        kind = probe_devices().kind
        if kind == "mps":
            device_arg = "mps"
        elif len(self.devices) > 1:
            device_arg = self.devices
        else:
            device_arg = self.devices[0] if self.devices else "cpu"
        LOG.info("Bắt đầu huấn luyện YOLO trên thiết bị: %s", device_arg)

        # Ultralytics hiểu tham số project theo đường dẫn tương đối với thư mục
        # cấu hình riêng của nó, nên nếu truyền đường dẫn tương đối thì kết quả sẽ
        # rơi vào một chỗ khác hẳn với chỗ ta mong đợi. Luôn truyền đường dẫn tuyệt
        # đối để trọng số nằm đúng nơi các bước sau đi tìm.
        runs_dir = Path(cfg.output_dir).resolve() / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        self.model.train(
            data=str(data_yaml),
            epochs=cfg.detect.epochs,
            imgsz=cfg.detect.image_size,
            batch=cfg.detect.batch_size,
            workers=cfg.detect.workers,
            device=device_arg,
            project=str(runs_dir),
            name="yolo_crater",
            exist_ok=True,
            seed=cfg.detect.seed,
            pretrained=spec.endswith(".pt"),
            verbose=True,
            plots=False,
            # Ảnh trinh sát là ảnh đơn sắc: tắt mọi phép tăng cường theo màu.
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.30,
            # Hố bom không có hướng ưu tiên nên xoay và lật tự do.
            degrees=180.0,
            fliplr=0.5,
            flipud=0.5,
            # Hố bom là đối tượng nhỏ: giữ biên độ co giãn ở mức vừa phải.
            mosaic=0.5,
            mixup=0.0,
            translate=0.06,
            scale=0.25,
        )

        best = runs_dir / "yolo_crater" / "weights" / "best.pt"
        if not best.exists():
            # Phòng trường hợp phiên bản Ultralytics đặt kết quả ở nơi khác: tìm
            # tệp trọng số mới nhất thay vì báo là không có.
            found = sorted(
                runs_dir.rglob("weights/best.pt"),
                key=lambda q: q.stat().st_mtime,
                reverse=True,
            )
            if not found:
                found = sorted(
                    Path.cwd().rglob("yolo_crater*/weights/best.pt"),
                    key=lambda q: q.stat().st_mtime,
                    reverse=True,
                )
            best = found[0] if found else best
        self.weights_path = best if best.exists() else None
        return {
            "backend": self.name,
            "weights": str(best),
            "devices": self.devices,
            "epochs": cfg.detect.epochs,
        }

    # -- Nạp trọng số ------------------------------------------------------
    def load(self, weights: Path) -> "YOLODetector":
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.weights_path = Path(weights)
        LOG.info("Đã nạp trọng số: %s", weights)
        return self

    # -- Suy luận ----------------------------------------------------------
    def predict(
        self, image_paths: Sequence[Path], conf: float = 0.25
    ) -> List[DetectionOutput]:
        if self.model is None:
            raise RuntimeError("Chưa nạp mô hình. Gọi train() hoặc load() trước.")

        from ..utils import probe_devices

        kind = probe_devices().kind
        device = "mps" if kind == "mps" else (self.devices[0] if self.devices else "cpu")
        outs: List[DetectionOutput] = []
        paths = [Path(p) for p in image_paths]

        for i in range(0, len(paths), 32):
            chunk = paths[i : i + 32]
            results = self.model.predict(
                [str(p) for p in chunk],
                conf=conf,
                imgsz=self.cfg.detect.image_size,
                device=device,
                verbose=False,
            )
            for p, r in zip(chunk, results):
                outs.append(_to_output(p, r))
        return outs

    # -- Suy luận phân tán trên hai GPU ------------------------------------
    def predict_sharded(
        self, image_paths: Sequence[Path], conf: float = 0.25
    ) -> List[DetectionOutput]:
        """Chia đôi khối lượng suy luận cho hai GPU nhằm rút ngắn thời gian."""
        if len(self.devices) < 2:
            return self.predict(image_paths, conf)

        import threading

        paths = [Path(p) for p in image_paths]
        mid = len(paths) // 2
        shards = [paths[:mid], paths[mid:]]
        results: Dict[int, List[DetectionOutput]] = {}

        def worker(idx: int, subset: Sequence[Path], dev: int) -> None:
            from ultralytics import YOLO

            local = YOLO(str(self.weights_path)) if self.weights_path else self.model
            outs: List[DetectionOutput] = []
            for i in range(0, len(subset), 32):
                chunk = subset[i : i + 32]
                rs = local.predict(
                    [str(p) for p in chunk],
                    conf=conf,
                    imgsz=self.cfg.detect.image_size,
                    device=dev,
                    verbose=False,
                )
                for p, r in zip(chunk, rs):
                    outs.append(_to_output(p, r))
            results[idx] = outs

        threads = [
            threading.Thread(target=worker, args=(i, shards[i], self.devices[i]))
            for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return results.get(0, []) + results.get(1, [])


def _to_output(path: Path, result) -> DetectionOutput:
    if result.boxes is None or len(result.boxes) == 0:
        return DetectionOutput(
            path.stem, np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        )
    return DetectionOutput(
        path.stem,
        result.boxes.xyxy.cpu().numpy().astype(np.float32),
        result.boxes.conf.cpu().numpy().astype(np.float32),
    )
