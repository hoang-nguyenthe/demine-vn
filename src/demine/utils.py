"""Tiện ích dùng chung: nhật ký, gieo hạt ngẫu nhiên, thư mục."""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-26s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def seed_everything(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên để kết quả tái lập được."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def describe_environment() -> dict:
    """Ghi nhận môi trường chạy để đưa vào báo cáo."""
    info = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if info["gpu_count"]:
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(info["gpu_count"])
            ]
    except Exception:
        info["torch"] = None
        info["cuda_available"] = False
        info["gpu_count"] = 0
    return info


def format_int(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def package_available(name: str) -> bool:
    """Kiểm tra một gói Python có nạp được hay không."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@dataclass
class DeviceInfo:
    n_gpu: int
    names: list
    kind: str = "cpu"


def probe_devices() -> DeviceInfo:
    """Thăm dò thiết bị tăng tốc hiện có.

    Hỗ trợ ba trường hợp: nhiều card NVIDIA (môi trường Kaggle), bộ tăng tốc Metal
    trên máy Apple dùng chip dòng M, và chỉ có CPU. Việc nhận biết Metal là cần thiết
    để đội thi phát triển và thử nghiệm được ngay trên máy cá nhân, thay vì phải chờ
    tới lúc lên Kaggle mới chạy được.
    """
    try:
        import torch

        if torch.cuda.is_available():
            n = int(torch.cuda.device_count())
            return DeviceInfo(
                n, [torch.cuda.get_device_name(i) for i in range(n)], "cuda"
            )
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return DeviceInfo(1, ["Apple Metal (MPS)"], "mps")
    except Exception:
        pass
    return DeviceInfo(0, [], "cpu")


def torch_device(index: int = 0):
    """Trả về thiết bị torch phù hợp với môi trường hiện tại."""
    import torch

    info = probe_devices()
    if info.kind == "cuda":
        return torch.device(f"cuda:{index}")
    if info.kind == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def resolve_devices(requested, allow_multi: bool = True):
    """Lọc danh sách GPU yêu cầu theo số thiết bị thực có."""
    info = probe_devices()
    if info.n_gpu == 0:
        return []
    if info.kind == "mps":
        # Metal chỉ phơi bày một thiết bị duy nhất, không có khái niệm chỉ số card.
        return [0]
    usable = [d for d in requested if d < info.n_gpu]
    if not usable:
        usable = [0]
    if not allow_multi:
        usable = usable[:1]
    return usable


def gpu_utilisation() -> Optional[str]:
    """Đọc mức sử dụng GPU qua nvidia-smi, trả về None nếu không có."""
    import subprocess

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def format_table(rows, headers=None) -> str:
    """Định dạng danh sách bản ghi thành bảng văn bản căn cột."""
    if not rows:
        return "(không có dữ liệu)"
    headers = headers or list(rows[0].keys())
    widths = {
        h: max(len(str(h)), max(len(str(r.get(h, ""))) for r in rows)) for h in headers
    }
    line = " | ".join(str(h).ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    body = [
        " | ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers) for r in rows
    ]
    return "\n".join([line, sep] + body)
