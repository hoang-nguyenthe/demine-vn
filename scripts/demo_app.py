#!/usr/bin/env python
"""DeMine-VN — Bảng điều khiển hỗ trợ ưu tiên rà phá bom mìn.

Chạy: streamlit run scripts/demo_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="DeMine-VN — Bản đồ ưu tiên rà phá",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# APPLE.COM PALETTE
# ============================================================================
COL_INK       = "#1D1D1F"
COL_MUTED     = "#6E6E73"
COL_SUB       = "#86868B"
COL_BG        = "#FBFBFD"
COL_CARD      = "#FFFFFF"
COL_HAIRLINE  = "#E5E5EA"
COL_BORDER    = "#D2D2D7"

# Bảng màu ngữ nghĩa
COL_HIGH      = "#B62A2A"   # nguy cơ cao
COL_MED       = "#B7791F"   # nguy cơ trung
COL_LOW       = "#2C7A7B"   # đã rà phá / an toàn tương đối
COL_CLEARED   = "#5B7F5A"
COL_ACCENT    = "#0071E3"   # Apple blue

CSS = f"""
<style>
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                     "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
    }}
    .stApp {{ background: {COL_BG}; color: {COL_INK}; }}
    .main .block-container {{ padding-top: 2rem; max-width: 1240px; }}

    section[data-testid="stSidebar"] {{
        background: #F5F5F7;
        border-right: 1px solid {COL_HAIRLINE};
    }}
    section[data-testid="stSidebar"] * {{ color: {COL_INK}; }}
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
        color: {COL_MUTED} !important; font-size: 12px; font-weight: 400;
        text-transform: none; letter-spacing: 0;
    }}
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
        color: {COL_INK} !important; font-size: 26px; font-weight: 600;
        letter-spacing: -0.02em;
    }}
    section[data-testid="stSidebar"] hr {{ border-color: {COL_HAIRLINE}; }}

    div[data-baseweb="tab-list"] {{
        gap: 0; background: transparent; padding: 0;
        border-bottom: 1px solid {COL_HAIRLINE}; border-radius: 0;
        box-shadow: none;
    }}
    button[data-baseweb="tab"] {{
        border-radius: 0 !important; padding: 12px 20px !important;
        font-weight: 400 !important; color: {COL_MUTED} !important;
        background: transparent !important; font-size: 14px !important;
        border-bottom: 2px solid transparent !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: transparent !important; color: {COL_INK} !important;
        border-bottom: 2px solid {COL_INK} !important; font-weight: 500 !important;
    }}
    div[data-baseweb="tab-highlight"] {{ display: none; }}

    [data-testid="stMetric"] {{
        background: {COL_CARD}; padding: 18px 20px; border-radius: 12px;
        border: 1px solid {COL_HAIRLINE}; box-shadow: none;
    }}
    [data-testid="stMetric"] label {{ color: {COL_MUTED} !important; font-weight: 400; font-size: 13px; }}
    [data-testid="stMetricValue"] {{
        color: {COL_INK} !important; font-weight: 600;
        letter-spacing: -0.02em; font-size: 28px;
    }}

    h1 {{ color: {COL_INK} !important; font-weight: 600; letter-spacing: -0.03em; }}
    h2, h3 {{ color: {COL_INK} !important; font-weight: 500; letter-spacing: -0.02em; }}
    h4, h5, h6 {{ color: {COL_INK} !important; font-weight: 500; }}

    .hero-num {{
        font-size: 56px; font-weight: 600; letter-spacing: -0.03em;
        color: {COL_INK}; line-height: 1;
    }}
    .hero-label {{
        font-size: 13px; color: {COL_MUTED}; margin-top: 8px; font-weight: 400;
    }}

    .kpi-strip {{ display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
    .kpi {{
        flex: 1; min-width: 180px; background: {COL_CARD};
        padding: 18px 20px; border-radius: 12px; border: 1px solid {COL_HAIRLINE};
    }}
    .kpi .kpi-label {{ color: {COL_MUTED}; font-size: 13px; font-weight: 400; }}
    .kpi .kpi-value {{
        color: {COL_INK}; font-size: 32px; font-weight: 600;
        margin-top: 6px; letter-spacing: -0.02em; line-height: 1.1;
    }}
    .kpi .kpi-note {{ color: {COL_MUTED}; font-size: 12px; margin-top: 6px; line-height: 1.4; }}

    .warning-card {{
        background: #FEF2F2; border: 1px solid #F6C2C2;
        border-radius: 12px; padding: 18px 22px; margin: 20px 0;
    }}
    .warning-card .warning-title {{
        color: #7A1F1F; font-weight: 600; font-size: 14px; margin-bottom: 6px;
        letter-spacing: -0.01em;
    }}
    .warning-card .warning-body {{
        color: #3F1010; font-size: 13px; line-height: 1.6;
    }}

    .callout {{
        background: {COL_CARD}; border: 1px solid {COL_HAIRLINE};
        border-radius: 12px; padding: 20px 22px; margin: 14px 0;
    }}
    .callout .callout-title {{
        color: {COL_INK}; font-size: 15px; font-weight: 500;
        letter-spacing: -0.01em; margin-bottom: 10px;
    }}
    .callout .callout-body {{
        color: {COL_MUTED}; font-size: 13.5px; line-height: 1.65;
    }}
    .callout .callout-body b {{ color: {COL_INK}; font-weight: 500; }}

    .streamlit-expanderHeader {{
        background: {COL_CARD} !important; border: 1px solid {COL_HAIRLINE} !important;
        border-radius: 10px !important; font-weight: 400 !important;
    }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
OUTPUTS = ROOT / "outputs"


@st.cache_data
def load_results():
    return json.loads((OUTPUTS / "ket_qua.json").read_text())


@st.cache_data
def load_priority():
    p = OUTPUTS / "danh_muc_uu_tien_ra_pha.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data
def load_figure(name: str) -> np.ndarray | None:
    p = OUTPUTS / "figures" / name
    if not p.exists():
        return None
    return np.array(Image.open(p))


try:
    KQ = load_results()
except Exception:
    st.error("Không tìm thấy `outputs/ket_qua.json`. Chạy pipeline trước.")
    st.stop()

PRIOR = load_priority()

# ============================================================================
# HEADER
# ============================================================================
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(
        f"<div style='font-size:34px;font-weight:600;letter-spacing:-0.03em;"
        f"color:{COL_INK};margin-bottom:4px;'>DeMine-VN</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:{COL_MUTED};font-size:15px;font-weight:400;line-height:1.5;'>"
        f"Hệ thống hỗ trợ xếp thứ tự ưu tiên rà phá bom mìn, vật nổ còn sót lại sau chiến "
        f"tranh — hợp nhất hồ sơ không kích, ảnh vệ tinh và dữ liệu rà phá thực địa.</div>",
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        f"<div style='text-align:right;padding-top:14px;color:{COL_MUTED};"
        f"font-size:12px;font-weight:400;line-height:1.5;'>"
        f"Bản trình diễn nghiên cứu<br>Vùng thí điểm Quảng Trị – Thừa Thiên Huế</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"<div style='height:1px;background:{COL_HAIRLINE};margin:20px 0 8px 0;'></div>",
    unsafe_allow_html=True,
)

# Nguyên tắc an toàn — luôn hiển thị
st.markdown(
    f"""<div class='warning-card'>
    <div class='warning-title'>Nguyên tắc an toàn</div>
    <div class='warning-body'>
    Hệ thống chỉ xếp thứ tự ưu tiên rà phá. Hệ thống <b>không bao giờ</b>
    tuyên bố một khu đất là an toàn. Mọi khu đất, kể cả khi được mô hình chấm
    mức nguy cơ thấp nhất, vẫn phải được rà phá đầy đủ theo quy trình kỹ thuật
    hiện hành trước khi đưa vào sử dụng.
    </div>
    </div>""",
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR
# ============================================================================
det = KQ["tang_hai_phat_hien_ho_bom"]
curve = KQ["duong_cong_hieu_qua_ra_pha"]
tier1 = KQ["tang_1_doi_chung_dat_da_ra_pha"]
tier4b = KQ.get("tang_4b_hieu_chinh_xac_suat", {})

with st.sidebar:
    st.markdown(
        f"<div style='font-size:11px;font-weight:500;color:{COL_MUTED};"
        f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px;'>"
        f"Chỉ tiêu vận hành</div>",
        unsafe_allow_html=True,
    )
    st.metric("Phát hiện hố bom (F1)", f"{det['f1']:.3f}")
    st.metric("Thu hồi vật nổ tại 20% diện tích", f"{curve['thu_hoi_tai_20pct_dien_tich']:.1%}")
    st.metric("Lợi thế so với quét đều", f"+{curve['loi_the_so_voi_quet_deu']:.1%}")
    st.metric("Điểm Brier hiệu chỉnh", f"{tier4b.get('diem_brier', 0):.4f}")

    st.markdown(
        f"<div style='height:1px;background:{COL_HAIRLINE};margin:20px 0;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div style="background:{COL_CARD};border:1px solid {COL_HAIRLINE};
                     padding:14px 16px;border-radius:12px;">
        <div style="color:{COL_INK};font-weight:500;font-size:13px;margin-bottom:8px;
                    letter-spacing:-0.01em;">Phạm vi sử dụng</div>
        <div style="color:{COL_MUTED};font-size:12px;line-height:1.6;">
        Sản phẩm dành cho lực lượng chức năng và cơ quan quản lý chương trình
        khắc phục hậu quả bom mìn. Kết quả xếp hạng ưu tiên mang tính hỗ trợ
        nghiệp vụ; quyết định triển khai thuộc thẩm quyền cơ quan chức năng.
        </div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='height:1px;background:{COL_HAIRLINE};margin:20px 0;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='color:{COL_MUTED};font-size:11px;line-height:1.6;'>"
        f"Kiến trúc: YOLO11 phát hiện hố bom · Gradient Boosting xếp hạng<br>"
        f"Dữ liệu vùng thí điểm: 220×180 ô 100 m<br>"
        f"Vùng thí điểm: Quảng Trị – Thừa Thiên Huế</div>",
        unsafe_allow_html=True,
    )

# ============================================================================
# TABS
# ============================================================================
tab_over, tab_map, tab_det, tab_curve, tab_val, tab_data, tab_kpi = st.tabs([
    "Tổng quan",
    "Bản đồ ưu tiên",
    "Phát hiện hố bom",
    "Đường cong hiệu quả",
    "Kiểm chứng bốn tầng",
    "Nguồn dữ liệu",
    "Chỉ tiêu tổng hợp",
])

# ============================================================================
# TAB: TỔNG QUAN
# ============================================================================
with tab_over:
    st.markdown(
        f"<div style='font-size:26px;font-weight:500;letter-spacing:-0.02em;"
        f"color:{COL_INK};margin-top:8px;'>Vấn đề còn nguyên vẹn sau nửa thế kỷ</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:{COL_MUTED};margin-top:8px;font-size:15px;line-height:1.6;max-width:820px;'>"
        f"Chiến tranh tại Việt Nam kết thúc năm 1975. Năm mươi năm sau, hậu quả vật lý "
        f"của nó vẫn nằm nguyên trong lòng đất. Bốn con số dưới đây, do Trung tâm Hành "
        f"động bom mìn Quốc gia Việt Nam công bố, mô tả quy mô của vấn đề."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label, note in [
        (c1, "6,1", "triệu héc-ta ô nhiễm",
         "chiếm 18,71% tổng diện tích cả nước"),
        (c2, "63/63", "tỉnh, thành phố",
         "toàn quốc có ô nhiễm bom mìn"),
        (c3, "800.000", "tấn bom đạn",
         "ước tính còn sót lại trong đất"),
        (c4, "40.000+", "người thiệt mạng",
         "sau 1975; hơn 60.000 người bị thương"),
    ]:
        with col:
            st.markdown(
                f"<div class='hero-num'>{num}</div>"
                f"<div style='color:{COL_INK};font-size:14px;font-weight:500;margin-top:6px;'>{label}</div>"
                f"<div class='hero-label'>{note}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown(
            f"<div style='font-size:22px;font-weight:500;letter-spacing:-0.02em;"
            f"color:{COL_INK};margin-bottom:12px;'>Vai trò của DeMine-VN</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:{COL_MUTED};font-size:14.5px;line-height:1.7;'>"
            f"Quy trình rà phá gồm hai bước. Bước một là khảo sát phi kỹ thuật để "
            f"khoanh vùng. Bước hai là rà phá kỹ thuật bằng máy dò trên từng mét vuông. "
            f"Bước thứ hai không thể rút ngắn bằng công nghệ thông tin — vẫn phải có "
            f"người cầm máy đi qua từng mét đất.<br><br>"
            f"Điểm nghẽn nằm ở bước một. Khi thông tin khoanh vùng còn thô, lực lượng "
            f"rà phá phải quét trải đều, dẫn đến phần lớn công sức được dồn vào những "
            f"khoảnh đất vốn không chứa vật nổ.<br><br>"
            f"<b>DeMine-VN thu hẹp phạm vi bước một</b> bằng cách hợp nhất ba nguồn "
            f"chứng cứ: (1) hồ sơ phi vụ không kích giải mật, (2) ảnh vệ tinh lịch sử "
            f"cho phép phát hiện hố bom cũ, (3) dữ liệu rà phá thực địa đã hoàn thành. "
            f"Đầu ra là bản đồ nguy cơ trên lưới 100 mét và danh mục xếp thứ tự ưu tiên."
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            f"""<div class='callout'>
            <div class='callout-title'>Lợi thế so với quét đều</div>
            <div class='callout-body'>
            Với cùng nguồn lực rà phá <b>20% diện tích</b> vùng nghiên cứu, hệ thống
            đầy đủ thu hồi <b>{curve['thu_hoi_tai_20pct_dien_tich']:.1%}</b> tổng
            khối lượng vật nổ tồn dư — cao hơn phương án quét đều
            <b>{curve['loi_the_so_voi_quet_deu']:.1%}</b> theo giá trị tuyệt đối.<br><br>
            Ở mức <b>50% diện tích</b>, tỉ lệ thu hồi đạt
            <b>{curve['thu_hoi_tai_50pct_dien_tich']:.1%}</b>. Ưu điểm tăng theo diện
            tích được ưu tiên trước, phản ánh tính chất "hình chóp Pareto" của phân bố
            ô nhiễm thực tế.
            </div></div>""",
            unsafe_allow_html=True,
        )

# ============================================================================
# TAB: BẢN ĐỒ ƯU TIÊN
# ============================================================================
with tab_map:
    from streamlit.components.v1 import html as st_html
    import folium
    from folium.plugins import HeatMap

    st.subheader("Bản đồ ưu tiên rà phá — vùng thí điểm Quảng Trị")
    st.markdown(
        f"<div style='color:{COL_MUTED};margin-bottom:14px;font-size:14px;line-height:1.55;'>"
        f"Mỗi điểm là tâm ô lưới 100 mét. Nhiệt độ màu thể hiện chỉ số ưu tiên rà phá — "
        f"kết hợp xác suất còn vật nổ và các yếu tố hỗ trợ (khoảng cách khu dân cư, "
        f"phù hợp canh tác, đã hoặc chưa được rà phá). Điểm càng sáng, thứ tự ưu tiên "
        f"càng cao.</div>",
        unsafe_allow_html=True,
    )

    if PRIOR.empty:
        st.info("Chưa có bản đồ ưu tiên. Chạy pipeline trước.")
    else:
        # Filter — top N
        top_n = st.slider("Hiển thị bao nhiêu ô ưu tiên hàng đầu",
                           min_value=100, max_value=min(2000, len(PRIOR)),
                           value=500, step=100)
        subset = PRIOR.head(top_n)

        m = folium.Map(
            location=[subset["vi_do"].mean(), subset["kinh_do"].mean()],
            zoom_start=12,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        # Heatmap layer
        heat_data = [[r["vi_do"], r["kinh_do"], r["chi_so_uu_tien"]]
                       for _, r in subset.iterrows()]
        HeatMap(heat_data, radius=14, blur=18, max_zoom=13,
                 gradient={0.3: "#5B7F5A", 0.55: "#B7791F", 0.8: "#B62A2A"}).add_to(m)

        # Top 30 cells as marker with popup
        for _, r in subset.head(30).iterrows():
            popup = folium.Popup(
                f"<div style='font-family:-apple-system,sans-serif;min-width:220px;'>"
                f"<div style='font-weight:600;font-size:13px;color:#1D1D1F;'>"
                f"Ô ưu tiên #{int(r['thu_tu_uu_tien'])}</div>"
                f"<hr style='margin:6px 0;border:none;border-top:1px solid #E5E5EA;'>"
                f"<table style='font-size:12px;color:#3C3C43;'>"
                f"<tr><td style='color:#6E6E73;padding-right:8px;'>Chỉ số ưu tiên</td>"
                f"<td><b>{r['chi_so_uu_tien']:.4f}</b></td></tr>"
                f"<tr><td style='color:#6E6E73;padding-right:8px;'>Xác suất còn vật nổ</td>"
                f"<td>{r['xac_suat_con_vat_no']:.4f}</td></tr>"
                f"<tr><td style='color:#6E6E73;padding-right:8px;'>Tải trọng bom (tấn)</td>"
                f"<td>{r['tai_trong_bom_ghi_nhan_tan']:.2f}</td></tr>"
                f"<tr><td style='color:#6E6E73;padding-right:8px;'>Số hố bom quan sát</td>"
                f"<td>{int(r['so_ho_bom_phat_hien'])}</td></tr>"
                f"<tr><td style='color:#6E6E73;padding-right:8px;'>Khoảng cách khu dân cư</td>"
                f"<td>{int(r['khoang_cach_khu_dan_cu_m'])} m</td></tr>"
                f"<tr><td style='color:#6E6E73;padding-right:8px;'>Đất canh tác</td>"
                f"<td>{r['la_dat_canh_tac']}</td></tr>"
                f"</table>"
                f"<div style='margin-top:8px;font-size:10.5px;color:#8E8E93;line-height:1.5;'>"
                f"Thứ tự ưu tiên rà phá. Không phải xác nhận an toàn.</div>"
                f"</div>",
                max_width=320,
            )
            folium.CircleMarker(
                location=[r["vi_do"], r["kinh_do"]],
                radius=5, color="#B62A2A", weight=1.5, fill=True,
                fillColor="#B62A2A", fillOpacity=0.85,
                popup=popup,
            ).add_to(m)

        legend_html = f'''
        <div style="position:fixed; top:100px; right:20px; z-index:9999;
                     background:rgba(255,255,255,0.95); padding:14px 18px;
                     border-radius:12px; font-family:-apple-system,sans-serif; font-size:12px;
                     border:1px solid rgba(0,0,0,0.06);
                     box-shadow:0 4px 16px rgba(0,0,0,0.08); color:#1D1D1F;">
            <div style="font-weight:500; font-size:11px; text-transform:uppercase;
                        letter-spacing:0.06em; color:#6E6E73; margin-bottom:8px;">Chỉ số ưu tiên</div>
            <div style="margin:4px 0;"><span style="display:inline-block;width:10px;height:10px;background:{COL_HIGH};border-radius:50%;margin-right:8px;"></span>Cao (0,8–1,0)</div>
            <div style="margin:4px 0;"><span style="display:inline-block;width:10px;height:10px;background:{COL_MED};border-radius:50%;margin-right:8px;"></span>Trung (0,55–0,8)</div>
            <div style="margin:4px 0;"><span style="display:inline-block;width:10px;height:10px;background:{COL_LOW};border-radius:50%;margin-right:8px;"></span>Thấp (dưới 0,55)</div>
            <div style="margin-top:8px;font-size:11px;color:#86868B;">30 ô đầu có popup chi tiết</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        st_html(m.get_root().render(), height=680)

        st.markdown(
            f'''<div class="kpi-strip">
            <div class="kpi">
                <div class="kpi-label">Ô đang hiển thị</div>
                <div class="kpi-value">{top_n:,}</div>
                <div class="kpi-note">trên tổng {len(PRIOR):,} ô lưới</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Xác suất trung bình</div>
                <div class="kpi-value">{subset['xac_suat_con_vat_no'].mean():.3f}</div>
                <div class="kpi-note">còn vật nổ trong nhóm hiển thị</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Tải trọng bom trung bình</div>
                <div class="kpi-value">{subset['tai_trong_bom_ghi_nhan_tan'].mean():.2f}</div>
                <div class="kpi-note">tấn, theo hồ sơ giải mật</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Có hố bom quan sát</div>
                <div class="kpi-value">{(subset['so_ho_bom_phat_hien'] > 0).sum():,}</div>
                <div class="kpi-note">ô có ít nhất một hố bom trên ảnh vệ tinh</div>
            </div>
            </div>''',
            unsafe_allow_html=True,
        )

# ============================================================================
# TAB: PHÁT HIỆN HỐ BOM
# ============================================================================
with tab_det:
    st.subheader("Phát hiện hố bom trên ảnh vệ tinh lịch sử")
    st.markdown(
        f"<div style='color:{COL_MUTED};margin-bottom:14px;font-size:14px;line-height:1.55;'>"
        f"Mô hình YOLO11 được huấn luyện để phát hiện hố bom trên ảnh vệ tinh độ phân "
        f"giải trung bình. Kết quả bên dưới là chỉ tiêu trên tập kiểm định của vùng "
        f"thí điểm."
        f"</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cols[0].metric("Precision", f"{det['precision']:.3f}")
    cols[1].metric("Recall", f"{det['recall']:.3f}")
    cols[2].metric("F1", f"{det['f1']:.3f}")
    cols[3].metric("mAP@0.5", f"{det['mAP@0.5']:.3f}")
    cols[4].metric("mAP@0.5:0.95", f"{det['mAP@0.5:0.95']:.3f}")

    st.markdown(
        f"<div style='color:{COL_MUTED};margin-top:16px;font-size:13.5px;line-height:1.55;'>"
        f"Trên tập kiểm định: <b>{det['so_ho_bom_that']:,}</b> hố bom thật, mô hình dự "
        f"báo <b>{det['so_ho_bom_du_bao']:,}</b> hố. Sai lệch giữa hai con số ~"
        f"{abs(det['so_ho_bom_du_bao'] - det['so_ho_bom_that'])} hố phản ánh biên "
        f"dương/âm giả trong ngưỡng chấp nhận được của bài toán quy mô rộng."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    fig07 = load_figure("07_anh_ve_tinh_mau.png")
    if fig07 is not None:
        st.markdown(
            f"<div style='font-size:16px;font-weight:500;color:{COL_INK};"
            f"letter-spacing:-0.01em;margin:8px 0 6px 0;'>Ví dụ ảnh vệ tinh và hố bom được phát hiện</div>",
            unsafe_allow_html=True,
        )
        st.image(fig07, use_container_width=True)

    fig02 = load_figure("02_ban_do_nguy_co.png")
    if fig02 is not None:
        st.markdown(
            f"<div style='font-size:16px;font-weight:500;color:{COL_INK};"
            f"letter-spacing:-0.01em;margin:16px 0 6px 0;'>Bản đồ nguy cơ trên lưới 100 mét</div>",
            unsafe_allow_html=True,
        )
        st.image(fig02, use_container_width=True)

    # Cảnh báo tăng hai
    warn = KQ.get("canh_bao_tang_hai", {})
    if warn:
        st.markdown(
            f"""<div class='callout'>
            <div class='callout-title'>Giới hạn của phát hiện ảnh vệ tinh</div>
            <div class='callout-body'>
            Chỉ <b>{warn.get('ty_le_dien_tich_co_anh_ve_tinh', 0):.1%}</b> diện tích
            vùng thí điểm có ảnh vệ tinh đủ độ phân giải để phát hiện hố bom. Phần
            còn lại phải dựa vào hồ sơ không kích và mô hình địa hình. Đây là lý do
            vì sao hệ thống hợp nhất ba nguồn thay vì chỉ dùng ảnh vệ tinh.
            </div></div>""",
            unsafe_allow_html=True,
        )

# ============================================================================
# TAB: ĐƯỜNG CONG HIỆU QUẢ
# ============================================================================
with tab_curve:
    st.subheader("Đường cong hiệu quả rà phá theo diện tích ưu tiên")
    st.markdown(
        f"<div style='color:{COL_MUTED};margin-bottom:14px;font-size:14px;line-height:1.55;'>"
        f"Trục ngang: tỉ lệ diện tích được rà phá theo thứ tự ưu tiên. Trục dọc: tỉ lệ "
        f"khối lượng vật nổ đã được thu hồi. Đường thẳng chéo là kịch bản quét đều. "
        f"Khoảng cách giữa đường cong của mô hình và đường chéo là lợi thế của việc "
        f"ưu tiên."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Đường cong hiệu quả — dựng từ Priority CSV
    import plotly.graph_objects as go

    if not PRIOR.empty:
        s = PRIOR.sort_values("chi_so_uu_tien", ascending=False)
        weight = s["xac_suat_con_vat_no"].values
        cum = np.cumsum(weight) / weight.sum()
        area = np.arange(1, len(cum) + 1) / len(cum)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=area, y=cum, mode="lines",
            line=dict(color=COL_INK, width=2.5),
            name="DeMine-VN — hệ thống đầy đủ",
            hovertemplate="Diện tích: %{x:.1%}<br>Vật nổ thu hồi: %{y:.1%}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=COL_MUTED, width=1.5, dash="dot"),
            name="Kịch bản quét đều",
        ))
        # Điểm mốc 10/20/50%
        for pct, y in [
            (0.10, curve["thu_hoi_tai_10pct_dien_tich"]),
            (0.20, curve["thu_hoi_tai_20pct_dien_tich"]),
            (0.50, curve["thu_hoi_tai_50pct_dien_tich"]),
        ]:
            fig.add_trace(go.Scatter(
                x=[pct], y=[y], mode="markers+text",
                marker=dict(size=10, color=COL_ACCENT, line=dict(color="white", width=2)),
                text=[f"{y:.0%}"], textposition="top center",
                textfont=dict(color=COL_INK, size=12),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.update_layout(
            height=460,
            plot_bgcolor=COL_CARD, paper_bgcolor=COL_CARD,
            xaxis=dict(title="Tỉ lệ diện tích được rà phá theo thứ tự ưu tiên",
                        tickformat=".0%", gridcolor=COL_HAIRLINE, linecolor=COL_HAIRLINE,
                        range=[0, 1]),
            yaxis=dict(title="Tỉ lệ khối lượng vật nổ đã thu hồi",
                        tickformat=".0%", gridcolor=COL_HAIRLINE, linecolor=COL_HAIRLINE,
                        range=[0, 1.02]),
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center",
                         bgcolor="rgba(0,0,0,0)", font=dict(color=COL_INK, size=12)),
            font=dict(color=COL_INK, family="-apple-system"),
            margin=dict(l=60, r=30, t=60, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'''<div class="kpi-strip">
        <div class="kpi">
            <div class="kpi-label">Rà phá 10% diện tích</div>
            <div class="kpi-value">{curve['thu_hoi_tai_10pct_dien_tich']:.1%}</div>
            <div class="kpi-note">vật nổ đã thu hồi</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Rà phá 20% diện tích</div>
            <div class="kpi-value">{curve['thu_hoi_tai_20pct_dien_tich']:.1%}</div>
            <div class="kpi-note">vật nổ đã thu hồi</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Rà phá 50% diện tích</div>
            <div class="kpi-value">{curve['thu_hoi_tai_50pct_dien_tich']:.1%}</div>
            <div class="kpi-note">vật nổ đã thu hồi</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Lợi thế trung bình</div>
            <div class="kpi-value" style="color:{COL_ACCENT}">+{curve['loi_the_so_voi_quet_deu']:.1%}</div>
            <div class="kpi-note">so với phương án quét đều</div>
        </div>
        </div>''',
        unsafe_allow_html=True,
    )

# ============================================================================
# TAB: KIỂM CHỨNG BỐN TẦNG
# ============================================================================
with tab_val:
    st.subheader("Kiểm chứng độc lập ở bốn tầng")
    st.markdown(
        f"<div style='color:{COL_MUTED};margin-bottom:14px;font-size:14px;line-height:1.55;'>"
        f"Bài toán rà phá không có nhãn hoàn hảo. Không thể chờ nhiều năm để biết dự "
        f"báo có đúng hay không. Vì vậy hệ thống được kiểm chứng bằng bốn tầng chứng "
        f"cứ độc lập — đối chứng nội bộ, đối chứng lịch sử tai nạn, đối chứng chéo hai "
        f"nguồn, và kiểm tra tính chuyển vùng địa lý."
        f"</div>",
        unsafe_allow_html=True,
    )

    tabs_val = st.tabs([
        "1 — Đất đã rà phá",
        "2 — Hồ sơ tai nạn",
        "3 — Nhất quán hai nguồn",
        "4 — Chuyển vùng & hiệu chỉnh",
    ])

    with tabs_val[0]:
        t1 = KQ["tang_1_doi_chung_dat_da_ra_pha"]
        st.markdown(
            f"<div style='color:{COL_MUTED};font-size:14px;line-height:1.55;'>"
            f"Chia dữ liệu ngẫu nhiên nhiều lần, huấn luyện trên phần đầu và kiểm tra "
            f"trên phần sau. Đo tỉ lệ thu hồi vật nổ tại 20% diện tích ưu tiên đầu."
            f"</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Số lần chia", str(t1["so_lan_chia"]))
        c2.metric("Thu hồi tại 20% (trung bình)", f"{t1['thu_hoi_tai_20pct_trung_binh']:.1%}")
        c3.metric("Độ lệch chuẩn", f"{t1['do_lech_chuan']:.4f}")

    with tabs_val[1]:
        t2 = KQ.get("tang_2_doi_chung_ho_so_tai_nan", [])
        if t2:
            df2 = pd.DataFrame(t2)
            df2.columns = ["Tập kiểm chứng", "Số vụ tai nạn",
                             "Bao phủ 10%", "Bao phủ 20%", "Bao phủ 30%",
                             "Hệ số vượt ngẫu nhiên (20%)"]
            st.markdown(
                f"<div style='color:{COL_MUTED};font-size:14px;line-height:1.55;margin-bottom:8px;'>"
                f"Kiểm chứng ngoài bằng hồ sơ tai nạn có thật đã ghi nhận trong quá khứ. "
                f"Bảng bên dưới cho biết thứ tự ưu tiên của mô hình có bao phủ được các "
                f"điểm tai nạn thực tế hay không."
                f"</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(df2.style.format({
                "Bao phủ 10%": "{:.1%}",
                "Bao phủ 20%": "{:.1%}",
                "Bao phủ 30%": "{:.1%}",
                "Hệ số vượt ngẫu nhiên (20%)": "{:.2f}×",
            }), use_container_width=True, hide_index=True)

    with tabs_val[2]:
        t3 = KQ.get("tang_3_nhat_quan_hai_nguon", {})
        st.markdown(
            f"<div style='color:{COL_MUTED};font-size:14px;line-height:1.55;'>"
            f"Số hố bom phát hiện qua ảnh vệ tinh và tải trọng bom trong hồ sơ không "
            f"kích phải tương quan với nhau — hai nguồn khác nhau, cùng một hiện tượng "
            f"vật lý. Thiếu tương quan có nghĩa là ít nhất một trong hai nguồn có sai lệch hệ thống."
            f"</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Spearman hố bom vs tải trọng",
                    f"{t3.get('spearman_ho_bom_vs_tai_trong', 0):.3f}")
        c2.metric("Ước lượng từ hồ sơ",
                    f"{t3.get('uoc_luong_vat_no_tu_ho_so', 0):,.0f}")
        c3.metric("Kỳ vọng từ mô hình",
                    f"{t3.get('ky_vong_vat_no_tu_mo_hinh', 0):,.0f}")

    with tabs_val[3]:
        t4a = KQ.get("tang_4a_chuyen_vung_dia_ly", {})
        t4b = KQ.get("tang_4b_hieu_chinh_xac_suat", {})
        st.markdown(
            f"<div style='color:{COL_MUTED};font-size:14px;line-height:1.55;'>"
            f"Huấn luyện trên một vùng địa lý và kiểm chứng trên vùng khác chưa từng "
            f"thấy — kiểm tra xem mô hình có phụ thuộc quá nhiều vào đặc điểm địa "
            f"phương hay không."
            f"</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Cùng vùng — thu hồi 20%",
                    f"{t4a.get('thu_hoi_tai_20pct_cung_vung', 0):.1%}")
        c2.metric("Vùng mới — thu hồi 20%",
                    f"{t4a.get('thu_hoi_tai_20pct_vung_moi', 0):.1%}",
                    delta=f"{t4a.get('muc_suy_giam_tuong_doi', 0):.1%} so với cùng vùng")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{COL_MUTED};font-size:14px;line-height:1.55;margin:6px 0 8px 0;'>"
            f"Hiệu chỉnh xác suất — đo bằng điểm Brier và sai số hiệu chỉnh kỳ vọng. "
            f"Điểm Brier càng nhỏ, xác suất dự báo càng đáng tin."
            f"</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Điểm Brier", f"{t4b.get('diem_brier', 0):.4f}")
        c2.metric("Sai số hiệu chỉnh", f"{t4b.get('sai_so_hieu_chinh_ky_vong', 0):.4f}")
        c3.metric("Xác suất trung bình", f"{t4b.get('xac_suat_du_bao_trung_binh', 0):.4f}")
        c4.metric("Tỉ lệ thực tế", f"{t4b.get('ty_le_duong_thuc_te', 0):.4f}")

    fig04 = load_figure("04_bieu_do_tin_cay.png")
    if fig04 is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:16px;font-weight:500;color:{COL_INK};"
            f"letter-spacing:-0.01em;margin:8px 0 6px 0;'>Biểu đồ hiệu chỉnh</div>",
            unsafe_allow_html=True,
        )
        st.image(fig04, use_container_width=True)

# ============================================================================
# TAB: NGUỒN DỮ LIỆU
# ============================================================================
with tab_data:
    st.subheader("Ba nguồn dữ liệu và cách hợp nhất")
    st.markdown(
        f"<div style='color:{COL_MUTED};margin-bottom:14px;font-size:14px;line-height:1.55;max-width:820px;'>"
        f"Hệ thống hợp nhất ba nguồn chứng cứ độc lập. Không có nguồn nào là hoàn hảo. "
        f"Sức mạnh của mô hình nằm ở sự bổ sung: điểm yếu của nguồn này được nguồn khác che phủ."
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class='callout'>
            <div class='callout-title'>1. Hồ sơ không kích giải mật</div>
            <div class='callout-body'>
            Nguồn tham chiếu: bộ dữ liệu THOR (Theater History of Operations Reports)
            do Bộ Quốc phòng Hoa Kỳ giải mật, ghi chép hơn 4 triệu phi vụ chiến tranh
            Việt Nam.<br><br>
            <b>Điểm mạnh.</b> Bao phủ toàn quốc, có toạ độ mục tiêu và tải trọng.<br>
            <b>Điểm yếu.</b> Sai số định vị lịch sử ~180 m; ~8% phi vụ mất hồ sơ.
            </div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class='callout'>
            <div class='callout-title'>2. Ảnh vệ tinh lịch sử</div>
            <div class='callout-body'>
            Ảnh vệ tinh độ phân giải trung bình cho phép phát hiện hố bom cũ vẫn còn
            lộ trên mặt đất — mỗi hố tương ứng ít nhất một quả bom đã nổ.<br><br>
            <b>Điểm mạnh.</b> Bằng chứng vật lý trực tiếp, độc lập với hồ sơ.<br>
            <b>Điểm yếu.</b> Trên nền đất mềm, hố bị bồi lấp; hố cũng có thể bị che khuất
            bởi thực vật.
            </div></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class='callout'>
            <div class='callout-title'>3. Dữ liệu rà phá thực địa</div>
            <div class='callout-body'>
            Bản ghi các khoảnh đã được đội rà phá quét sạch trong quá khứ — thông tin
            có tính xác thực cao nhất vì đến từ máy dò thực tế.<br><br>
            <b>Điểm mạnh.</b> Nhãn ground-truth cho hiệu chỉnh mô hình.<br>
            <b>Điểm yếu.</b> Chỉ bao phủ một phần diện tích; không phân bố đồng đều.
            </div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    fig03 = load_figure("03_ba_nguon_du_lieu.png")
    if fig03 is not None:
        st.markdown(
            f"<div style='font-size:16px;font-weight:500;color:{COL_INK};"
            f"letter-spacing:-0.01em;margin:8px 0 6px 0;'>Ba lớp dữ liệu chồng lên nhau trên cùng lưới 100 mét</div>",
            unsafe_allow_html=True,
        )
        st.image(fig03, use_container_width=True)

    # Ablation
    st.markdown(
        f"<div style='font-size:20px;font-weight:500;color:{COL_INK};"
        f"letter-spacing:-0.02em;margin:24px 0 12px 0;'>"
        f"Đóng góp riêng của từng nguồn — nghiên cứu ablation</div>",
        unsafe_allow_html=True,
    )
    abl = KQ.get("phan_tich_dong_gop_thanh_phan", [])
    if abl:
        df = pd.DataFrame(abl)
        df.columns = ["Cấu hình", "Số đặc trưng",
                        "Thu hồi 20%", "Lợi thế so với quét đều",
                        "Bao phủ tai nạn (20%)"]
        st.dataframe(df.style.format({
            "Thu hồi 20%": "{:.1%}",
            "Lợi thế so với quét đều": "{:.1%}",
            "Bao phủ tai nạn (20%)": "{:.1%}",
        }), use_container_width=True, hide_index=True)

    st.markdown(
        f"""<div class='callout' style='margin-top:16px;'>
        <div class='callout-title'>Bộ dữ liệu tham chiếu công khai</div>
        <div class='callout-body'>
        Mã nguồn được thiết kế để làm việc với dữ liệu thật khi được cung cấp.
        Các bộ dữ liệu mở tham chiếu:<br><br>
        <b>THOR — Theater History of Operations Reports.</b> Hồ sơ giải mật của Bộ
        Quốc phòng Hoa Kỳ về các phi vụ ném bom giai đoạn 1965–1975. Có thể tải
        về từ data.mil hoặc catalog.data.gov.<br><br>
        <b>Landmine and Cluster Munition Monitor — Vietnam Country Profile.</b>
        Số liệu tai nạn cập nhật hằng năm, do The Monitor công bố.<br><br>
        <b>VNMAC.</b> Báo cáo thường niên của Trung tâm Hành động bom mìn Quốc gia
        Việt Nam về tình trạng ô nhiễm cấp tỉnh và tiến độ khắc phục.
        </div></div>""",
        unsafe_allow_html=True,
    )

# ============================================================================
# TAB: CHỈ TIÊU TỔNG HỢP
# ============================================================================
with tab_kpi:
    st.subheader("Chỉ tiêu tổng hợp bản chạy hiện tại")

    inner = st.tabs(["Đặc trưng đóng góp lớn nhất",
                       "Đóng góp theo hoán vị",
                       "Môi trường chạy"])

    with inner[0]:
        feats = KQ.get("muc_dong_gop_dac_trung", {})
        if feats:
            df = pd.DataFrame([{"Đặc trưng": k, "Trọng số": v}
                                 for k, v in list(feats.items())[:15]])
            st.dataframe(df.style.format({"Trọng số": "{:.4f}"}),
                          use_container_width=True, hide_index=True)

    with inner[1]:
        perm = KQ.get("muc_dong_gop_theo_hoan_vi", {})
        if perm:
            df = pd.DataFrame([{"Đặc trưng": k, "Mức đóng góp": v}
                                 for k, v in list(perm.items())[:15]])
            st.dataframe(df.style.format({"Mức đóng góp": "{:.4f}"}),
                          use_container_width=True, hide_index=True)

    with inner[2]:
        env = KQ.get("moi_truong_chay", {})
        if env:
            df = pd.DataFrame([{"Thông tin": k, "Giá trị": str(v)}
                                 for k, v in env.items()])
            st.dataframe(df, use_container_width=True, hide_index=True)

    fig05 = load_figure("05_dong_gop_thanh_phan.png")
    fig06 = load_figure("06_dong_gop_dac_trung.png")
    if fig05 is not None or fig06 is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        if fig05 is not None:
            with cc1: st.image(fig05, use_container_width=True)
        if fig06 is not None:
            with cc2: st.image(fig06, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown(
    f"<div style='height:1px;background:{COL_HAIRLINE};margin:32px 0 12px 0;'></div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='color:{COL_MUTED};font-size:12px;text-align:center;padding:16px 0;line-height:1.6;'>"
    f"DeMine-VN · Hệ thống hỗ trợ xếp thứ tự ưu tiên rà phá bom mìn<br>"
    f"Kết quả mô hình mang tính hỗ trợ nghiệp vụ. Không thay thế quy trình rà phá kỹ thuật."
    f"</div>",
    unsafe_allow_html=True,
)
