"""Ghi bộ dữ liệu ảnh ra đĩa theo quy ước YOLO và đọc lại.

Quy ước thư mục là hợp đồng dữ liệu giữa tầng một và tầng hai. Khi thay dữ liệu
mô phỏng bằng ảnh vệ tinh giải mật thật, chỉ cần tạo ra đúng cấu trúc thư mục này
là toàn bộ phần còn lại của hệ thống chạy không cần sửa. Xem ``docs/ADAPT_NEW_DATA.md``.

    <root>/images/{train,val,test}/<tile_id>.png
    <root>/labels/{train,val,test}/<tile_id>.txt
    <root>/tiles.jsonl
    <root>/data.yaml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..utils import ensure_dir, get_logger
from .imagery import Tile

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")


def save_image(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(array).save(path)


def load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"))


def _to_yolo_line(box: np.ndarray, size: int) -> str:
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2.0 / size
    yc = (y1 + y2) / 2.0 / size
    w = (x2 - x1) / size
    h = (y2 - y1) / size
    return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def write_dataset(tiles: List[Tile], root: Path) -> Path:
    """Ghi toàn bộ ảnh con, nhãn, siêu dữ liệu và tệp mô tả bộ dữ liệu."""
    root = ensure_dir(root)
    for split in SPLITS:
        ensure_dir(root / "images" / split)
        ensure_dir(root / "labels" / split)

    meta_path = root / "tiles.jsonl"
    counts: Dict[str, int] = {s: 0 for s in SPLITS}

    with meta_path.open("w", encoding="utf-8") as fh:
        for tile in tiles:
            name = f"tile_{tile.tile_id:05d}"
            img_path = root / "images" / tile.split / f"{name}.png"
            lbl_path = root / "labels" / tile.split / f"{name}.txt"

            save_image(img_path, tile.image)
            lines = [_to_yolo_line(b, tile.size_px) for b in tile.boxes]
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            counts[tile.split] += 1
            fh.write(
                json.dumps(
                    {
                        "tile_id": tile.tile_id,
                        "name": name,
                        "split": tile.split,
                        "origin_x_m": tile.origin_x_m,
                        "origin_y_m": tile.origin_y_m,
                        "gsd_m": tile.gsd_m,
                        "size_px": tile.size_px,
                        "n_craters": int(tile.boxes.shape[0]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: ho_bom",
                "",
            ]
        ),
        encoding="utf-8",
    )

    logger.info(
        "Đã ghi bộ dữ liệu ảnh: %d huấn luyện | %d thẩm định | %d kiểm tra → %s",
        counts["train"],
        counts["val"],
        counts["test"],
        root,
    )
    return yaml_path


def read_tile_metadata(root: Path) -> List[dict]:
    path = Path(root) / "tiles.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_ground_truth(root: Path, split: str) -> Dict[str, np.ndarray]:
    """Đọc nhãn thật của một tập, trả về khung bao theo đơn vị điểm ảnh."""
    root = Path(root)
    lbl_dir = root / "labels" / split
    img_dir = root / "images" / split
    out: Dict[str, np.ndarray] = {}

    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = img_dir / f"{lbl.stem}.png"
        if not img.exists():
            continue
        size = load_image(img).shape[0]
        boxes = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = map(float, parts)
            boxes.append(
                [
                    (xc - w / 2) * size,
                    (yc - h / 2) * size,
                    (xc + w / 2) * size,
                    (yc + h / 2) * size,
                ]
            )
        out[lbl.stem] = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return out


def coverage_mask(tile_meta: List[dict], grid) -> np.ndarray:
    """Đánh dấu những ô lưới thực sự nằm trong phạm vi có ảnh vệ tinh.

    Đây là một phân biệt bắt buộc, không phải chi tiết phụ. Ảnh vệ tinh lịch sử
    không bao giờ phủ kín toàn bộ vùng nghiên cứu — kho ảnh giải mật là những dải
    chụp rời rạc. Nếu gán số không cho cả những ô không có ảnh, mô hình sẽ đọc
    thành ``nơi này đã được nhìn và không thấy hố bom nào``, trong khi sự thật là
    ``nơi này chưa từng được nhìn``. Hai điều đó khác nhau hoàn toàn, và nhầm lẫn
    giữa chúng khiến mô hình đánh giá thấp toàn bộ phần lãnh thổ chưa có ảnh.
    """
    covered = np.zeros(grid.shape, dtype=bool)
    for m in tile_meta:
        extent = float(m["size_px"]) * float(m["gsd_m"])
        x0, y0 = float(m["origin_x_m"]), float(m["origin_y_m"])
        c0, r0 = grid.xy_to_index(np.array([x0]), np.array([y0]))
        c1, r1 = grid.xy_to_index(np.array([x0 + extent]), np.array([y0 + extent]))
        c0 = int(np.clip(c0[0], 0, grid.cfg.n_cells_x - 1))
        c1 = int(np.clip(c1[0], 0, grid.cfg.n_cells_x - 1))
        r0 = int(np.clip(r0[0], 0, grid.cfg.n_cells_y - 1))
        r1 = int(np.clip(r1[0], 0, grid.cfg.n_cells_y - 1))
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
    return covered


def detections_to_grid(
    detections, tile_meta: List[dict], grid
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Quy các hố bom phát hiện được về lưới ô của vùng nghiên cứu.

    Trả về ba lớp: số lượng hố bom trên mỗi ô, đường kính trung bình của hố bom
    trên ô đó, và mặt nạ cho biết ô nào thực sự có ảnh vệ tinh phủ tới.
    """
    by_name = {m["name"]: m for m in tile_meta}
    count = np.zeros(grid.shape, dtype=float)
    diameter_sum = np.zeros(grid.shape, dtype=float)

    for det in detections:
        meta = by_name.get(det.image_id)
        if meta is None or det.boxes.shape[0] == 0:
            continue
        gsd = float(meta["gsd_m"])
        size = int(meta["size_px"])
        ox, oy = float(meta["origin_x_m"]), float(meta["origin_y_m"])

        cx_px = (det.boxes[:, 0] + det.boxes[:, 2]) / 2.0
        cy_px = (det.boxes[:, 1] + det.boxes[:, 3]) / 2.0
        width_px = det.boxes[:, 2] - det.boxes[:, 0]

        x_m = ox + cx_px * gsd
        y_m = oy + (size - 1 - cy_px) * gsd
        diameter_m = width_px * gsd / 0.62 / 2.0 * 2.0

        col, row = grid.xy_to_index(x_m, y_m)
        keep = grid.inside(col, row)
        np.add.at(count, (row[keep], col[keep]), 1.0)
        np.add.at(diameter_sum, (row[keep], col[keep]), diameter_m[keep])

    mean_diameter = np.divide(
        diameter_sum, count, out=np.zeros_like(diameter_sum), where=count > 0
    )
    return count, mean_diameter, coverage_mask(tile_meta, grid)
