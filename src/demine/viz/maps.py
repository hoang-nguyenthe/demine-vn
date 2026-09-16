"""Bản đồ ưu tiên rà phá, kết xuất thành một tệp web tự chứa.

Bản đồ được dựng bằng Folium khi có sẵn, và lùi về một trang web tự vẽ bằng thẻ
canvas khi không có — để bước này không bao giờ làm gãy quy trình vì thiếu gói.

Nguyên tắc an toàn được in cố định ngay trên giao diện, không thể tắt: hệ thống
xếp thứ tự ưu tiên, không xác nhận an toàn. Bảng màu cũng được chọn theo nguyên
tắc đó — không có màu xanh lá cho vùng nguy cơ thấp, vì xanh lá đọc thành an toàn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import SAFETY_NOTICE
from ..utils import get_logger

logger = get_logger(__name__)

# Thang màu từ vàng nhạt tới đỏ sẫm. Cố ý không dùng xanh lá ở đầu thang.
COLORS = ["#FBF3E3", "#FBE2B4", "#F6BE7E", "#E8895C", "#C9452F", "#8E1B12"]


def _top_cells(pipeline, n_top: int = 400):
    priority = pipeline.priority
    order = np.argsort(-priority)[:n_top]
    lon, lat = pipeline.grid.cell_centers_lonlat()
    return order, lon, lat


def _color_for(value: float, breaks: np.ndarray) -> str:
    idx = int(np.searchsorted(breaks, value, side="right"))
    return COLORS[min(idx, len(COLORS) - 1)]


def make_priority_map(pipeline, path: Path, n_top: int = 400) -> Path:
    path = Path(path)
    order, lon, lat = _top_cells(pipeline, n_top)
    priority = pipeline.priority
    probability = pipeline.probability
    breaks = np.quantile(priority[order], [0.2, 0.4, 0.6, 0.8, 0.95])

    try:
        import folium

        centre = [float(np.mean(lat)), float(np.mean(lon))]
        fmap = folium.Map(location=centre, zoom_start=12, tiles="CartoDB positron")

        half = pipeline.grid.cell / 111_320.0 / 2.0
        for i in order:
            color = _color_for(priority[i], breaks)
            folium.Rectangle(
                bounds=[
                    [lat[i] - half, lon[i] - half],
                    [lat[i] + half, lon[i] + half],
                ],
                color=color,
                weight=0.4,
                fill=True,
                fill_color=color,
                fill_opacity=0.72,
                popup=folium.Popup(
                    f"<b>Ô lưới {int(i)}</b><br>"
                    f"Xác suất còn vật nổ: {probability[i]:.3f}<br>"
                    f"Chỉ số ưu tiên: {priority[i]:.3f}<br>"
                    f"<i>Chưa được rà phá — không phải xác nhận an toàn.</i>",
                    max_width=280,
                ),
            ).add_to(fmap)

        for c, r in zip(pipeline.accidents.col, pipeline.accidents.row):
            idx = int(r) * pipeline.grid.cfg.n_cells_x + int(c)
            folium.CircleMarker(
                location=[lat[idx], lon[idx]],
                radius=3.2,
                color="#1A1A1A",
                weight=1.2,
                fill=False,
                popup="Vị trí tai nạn đã ghi nhận",
            ).add_to(fmap)

        banner = f"""
        <div style="position: fixed; bottom: 18px; left: 18px; z-index: 9999;
                    background: #FAEAE6; border-left: 5px solid #C9705C;
                    padding: 10px 14px; max-width: 430px; border-radius: 4px;
                    font-family: Georgia, serif; font-size: 12.5px; color: #1C3557;">
          <b>Nguyên tắc an toàn bắt buộc.</b><br>{SAFETY_NOTICE}
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(banner))
        fmap.save(str(path))
        logger.info("  Đã dựng bản đồ ưu tiên bằng Folium: %s", path.name)
        return path

    except Exception as exc:
        logger.warning("Không dùng được Folium (%s), chuyển sang bản đồ tự vẽ.", exc)

    cells = [
        {
            "x": int(i % pipeline.grid.cfg.n_cells_x),
            "y": int(i // pipeline.grid.cfg.n_cells_x),
            "p": round(float(probability[i]), 4),
            "u": round(float(priority[i]), 4),
            "c": _color_for(priority[i], breaks),
        }
        for i in order
    ]
    payload = json.dumps(
        {
            "cells": cells,
            "nx": pipeline.grid.cfg.n_cells_x,
            "ny": pipeline.grid.cfg.n_cells_y,
            "accidents": [
                {"x": int(c), "y": int(r)}
                for c, r in zip(pipeline.accidents.col, pipeline.accidents.row)
            ],
        },
        ensure_ascii=False,
    )

    html = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<title>DeMine-VN — Bản đồ ưu tiên rà phá</title>
<style>
 body {{ margin:0; font-family: Georgia, serif; background:#FBF9F5; color:#1C3557; }}
 header {{ padding:16px 22px; border-bottom:1px solid #9DBCDA; }}
 h1 {{ margin:0; font-size:19px; }}
 p.sub {{ margin:6px 0 0; font-size:13px; color:#3F576F; }}
 #wrap {{ padding:18px 22px; }}
 canvas {{ border:1px solid #9DBCDA; background:#fff; max-width:100%; }}
 .notice {{ margin-top:16px; background:#FAEAE6; border-left:5px solid #C9705C;
            padding:12px 16px; font-size:13px; border-radius:4px; max-width:760px; }}
 .legend {{ margin-top:12px; font-size:12.5px; color:#3F576F; }}
 .sw {{ display:inline-block; width:22px; height:11px; margin:0 4px 0 12px;
        vertical-align:middle; border:1px solid #ccc; }}
</style></head><body>
<header>
  <h1>DeMine-VN — Bản đồ ưu tiên rà phá</h1>
  <p class="sub">Ô càng sẫm màu, thứ tự ưu tiên rà phá càng cao. Vòng tròn đen là vị trí tai nạn đã ghi nhận.</p>
</header>
<div id="wrap">
  <canvas id="cv" width="1100" height="760"></canvas>
  <div class="legend">Mức ưu tiên:
    <span class="sw" style="background:{COLORS[0]}"></span>thấp
    <span class="sw" style="background:{COLORS[2]}"></span>trung bình
    <span class="sw" style="background:{COLORS[5]}"></span>cao nhất
  </div>
  <div class="notice"><b>Nguyên tắc an toàn bắt buộc.</b> {SAFETY_NOTICE}</div>
</div>
<script>
const D = {payload};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const sx = cv.width / D.nx, sy = cv.height / D.ny;
for (const c of D.cells) {{
  ctx.fillStyle = c.c;
  ctx.fillRect(c.x*sx, cv.height - (c.y+1)*sy, Math.max(sx,1.5), Math.max(sy,1.5));
}}
ctx.strokeStyle = '#1A1A1A'; ctx.lineWidth = 1.1;
for (const a of D.accidents) {{
  ctx.beginPath();
  ctx.arc(a.x*sx + sx/2, cv.height - (a.y+0.5)*sy, 3.2, 0, 6.2832);
  ctx.stroke();
}}
</script></body></html>
"""
    path.write_text(html, encoding="utf-8")
    logger.info("  Đã dựng bản đồ ưu tiên tự vẽ: %s", path.name)
    return path
