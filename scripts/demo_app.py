#!/usr/bin/env python
"""DeMine-VN — Hệ thống hỗ trợ xếp thứ tự ưu tiên rà phá bom mìn toàn quốc.

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
    page_title="DeMine-VN — Bản đồ rà phá bom mìn Việt Nam",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# APPLE PALETTE + ANIMATIONS
# ============================================================================
COL_INK       = "#1D1D1F"
COL_MUTED     = "#6E6E73"
COL_SUB       = "#86868B"
COL_BG        = "#FBFBFD"
COL_CARD      = "#FFFFFF"
COL_HAIRLINE  = "#E5E5EA"
COL_BORDER    = "#D2D2D7"

COL_HIGH      = "#B62A2A"
COL_MEDHIGH   = "#D26B36"
COL_MED       = "#D4A017"
COL_LOWMED    = "#7A9E5F"
COL_LOW       = "#2C7A7B"
COL_ACCENT    = "#0071E3"

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                     "SF Pro Text", "Inter", "Helvetica Neue", Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
    }}
    .stApp {{ background: {COL_BG}; color: {COL_INK}; }}
    .main .block-container {{ padding-top: 1.5rem; max-width: 1280px; }}

    /* ========== ANIMATIONS ========== */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translate3d(0, 24px, 0); }}
        to   {{ opacity: 1; transform: translate3d(0, 0, 0); }}
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}
    @keyframes scaleIn {{
        from {{ opacity: 0; transform: scale(0.96); }}
        to   {{ opacity: 1; transform: scale(1); }}
    }}
    @keyframes pulse {{
        0%,100% {{ opacity: 1; }}
        50% {{ opacity: 0.55; }}
    }}
    @keyframes shimmer {{
        0%   {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    @keyframes countUp {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    .fade-in-up  {{ animation: fadeInUp 0.8s cubic-bezier(0.22, 1, 0.36, 1) both; }}
    .fade-in     {{ animation: fadeIn 1.2s cubic-bezier(0.22, 1, 0.36, 1) both; }}
    .scale-in    {{ animation: scaleIn 0.7s cubic-bezier(0.22, 1, 0.36, 1) both; }}
    .stagger-1   {{ animation-delay: 0.05s; }}
    .stagger-2   {{ animation-delay: 0.12s; }}
    .stagger-3   {{ animation-delay: 0.20s; }}
    .stagger-4   {{ animation-delay: 0.28s; }}
    .stagger-5   {{ animation-delay: 0.36s; }}
    .stagger-6   {{ animation-delay: 0.44s; }}

    /* ========== SIDEBAR ========== */
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

    /* ========== TABS ========== */
    div[data-baseweb="tab-list"] {{
        gap: 0; background: transparent; padding: 0;
        border-bottom: 1px solid {COL_HAIRLINE}; border-radius: 0;
        box-shadow: none; overflow-x: auto;
    }}
    button[data-baseweb="tab"] {{
        border-radius: 0 !important; padding: 14px 22px !important;
        font-weight: 400 !important; color: {COL_MUTED} !important;
        background: transparent !important; font-size: 14px !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.35s cubic-bezier(0.22, 1, 0.36, 1) !important;
        white-space: nowrap !important;
    }}
    button[data-baseweb="tab"]:hover {{
        color: {COL_INK} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: transparent !important; color: {COL_INK} !important;
        border-bottom: 2px solid {COL_INK} !important; font-weight: 500 !important;
    }}
    div[data-baseweb="tab-highlight"] {{ display: none; }}

    /* ========== METRICS ========== */
    [data-testid="stMetric"] {{
        background: {COL_CARD}; padding: 20px 22px; border-radius: 14px;
        border: 1px solid {COL_HAIRLINE}; box-shadow: none;
        transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    [data-testid="stMetric"]:hover {{
        border-color: {COL_BORDER}; transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }}
    [data-testid="stMetric"] label {{ color: {COL_MUTED} !important; font-weight: 400; font-size: 13px; }}
    [data-testid="stMetricValue"] {{
        color: {COL_INK} !important; font-weight: 600;
        letter-spacing: -0.025em; font-size: 30px;
    }}

    h1 {{ color: {COL_INK} !important; font-weight: 600; letter-spacing: -0.03em; }}
    h2, h3 {{ color: {COL_INK} !important; font-weight: 500; letter-spacing: -0.02em; }}
    h4, h5, h6 {{ color: {COL_INK} !important; font-weight: 500; }}

    /* ========== HERO ========== */
    .hero {{
        padding: 40px 0 30px 0;
        border-bottom: 1px solid {COL_HAIRLINE};
        margin-bottom: 28px;
    }}
    .hero .hero-eyebrow {{
        font-size: 11px; font-weight: 500; color: {COL_MUTED};
        text-transform: uppercase; letter-spacing: 0.12em;
        margin-bottom: 12px;
    }}
    .hero .hero-title {{
        font-size: 48px; font-weight: 600; letter-spacing: -0.035em;
        color: {COL_INK}; line-height: 1.08; margin-bottom: 12px;
    }}
    .hero .hero-sub {{
        font-size: 18px; font-weight: 400; color: {COL_MUTED};
        line-height: 1.5; max-width: 780px;
    }}

    .hero-num {{
        font-size: 64px; font-weight: 600; letter-spacing: -0.035em;
        color: {COL_INK}; line-height: 1;
    }}
    .hero-label {{
        font-size: 13px; color: {COL_MUTED}; margin-top: 10px;
        font-weight: 400; line-height: 1.5;
    }}

    /* ========== KPI CARDS ========== */
    .kpi-strip {{ display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
    .kpi {{
        flex: 1; min-width: 180px; background: {COL_CARD};
        padding: 20px 22px; border-radius: 14px; border: 1px solid {COL_HAIRLINE};
        transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    .kpi:hover {{
        transform: translateY(-2px);
        border-color: {COL_BORDER};
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }}
    .kpi .kpi-label {{
        color: {COL_MUTED}; font-size: 13px; font-weight: 400;
    }}
    .kpi .kpi-value {{
        color: {COL_INK}; font-size: 34px; font-weight: 600;
        margin-top: 8px; letter-spacing: -0.025em; line-height: 1.05;
    }}
    .kpi .kpi-note {{ color: {COL_MUTED}; font-size: 12px; margin-top: 8px; line-height: 1.5; }}

    /* ========== WARNING ========== */
    .warning-card {{
        background: #FEF2F2; border: 1px solid #F6C2C2;
        border-radius: 14px; padding: 18px 22px; margin: 20px 0;
        animation: fadeIn 0.6s ease-in-out both;
    }}
    .warning-card .warning-title {{
        color: #7A1F1F; font-weight: 600; font-size: 14px; margin-bottom: 6px;
        letter-spacing: -0.01em;
    }}
    .warning-card .warning-body {{
        color: #3F1010; font-size: 13.5px; line-height: 1.65;
    }}

    /* ========== CALLOUT ========== */
    .callout {{
        background: {COL_CARD}; border: 1px solid {COL_HAIRLINE};
        border-radius: 14px; padding: 22px 24px; margin: 14px 0;
        transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    .callout:hover {{
        border-color: {COL_BORDER};
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }}
    .callout .callout-title {{
        color: {COL_INK}; font-size: 15px; font-weight: 500;
        letter-spacing: -0.01em; margin-bottom: 10px;
    }}
    .callout .callout-body {{
        color: {COL_MUTED}; font-size: 13.5px; line-height: 1.7;
    }}
    .callout .callout-body b {{ color: {COL_INK}; font-weight: 500; }}

    /* ========== BADGES ========== */
    .badge {{
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        font-size: 11px; font-weight: 500; letter-spacing: -0.01em;
        border: 1px solid {COL_HAIRLINE}; background: {COL_CARD};
        color: {COL_MUTED}; margin-right: 6px;
    }}
    .badge-red {{ background: #FEF2F2; border-color: #F6C2C2; color: #7A1F1F; }}
    .badge-amber {{ background: #FFF8E1; border-color: #F6E7B8; color: #7A5A0F; }}
    .badge-green {{ background: #EFF8F7; border-color: #C6E6E2; color: #1B5568; }}
    .badge-blue {{ background: #EFF6FF; border-color: #BFDBFE; color: #1E3A8A; }}

    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {{
        background: {COL_CARD} !important; border: 1px solid {COL_HAIRLINE} !important;
        border-radius: 12px !important; font-weight: 400 !important;
    }}

    /* ========== SLIDER ========== */
    [data-baseweb="slider"] [role="slider"] {{
        border: 2px solid {COL_INK} !important;
    }}

    /* ========== SELECTBOX ========== */
    [data-baseweb="select"] > div {{
        border-color: {COL_HAIRLINE} !important;
        border-radius: 10px !important;
        transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    [data-baseweb="select"] > div:hover {{
        border-color: {COL_BORDER} !important;
    }}

    /* ========== DATAFRAME ========== */
    [data-testid="stDataFrame"] {{
        border: 1px solid {COL_HAIRLINE};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ========== SECTION HEADING ========== */
    .section-title {{
        font-size: 28px; font-weight: 600; letter-spacing: -0.03em;
        color: {COL_INK}; margin: 12px 0 6px 0;
    }}
    .section-sub {{
        font-size: 14.5px; color: {COL_MUTED}; line-height: 1.6;
        margin-bottom: 20px; max-width: 820px;
    }}

    /* ========== SIDEBAR HEADER ========== */
    .sidebar-eyebrow {{
        font-size: 11px; font-weight: 500; color: {COL_MUTED};
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px;
    }}

    /* Hide streamlit branding */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================
# LOAD DATA
# ============================================================================
OUTPUTS = ROOT / "outputs"
DATA_DIR = ROOT / "data"


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
def load_figure(name: str):
    p = OUTPUTS / "figures" / name
    if not p.exists():
        return None
    return np.array(Image.open(p))


@st.cache_data
def load_vn_geojson():
    p = DATA_DIR / "vn_geo" / "vietnam-provinces.geojson"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data
def load_uxo_by_province():
    p = DATA_DIR / "vnmac" / "uxo_by_province.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p).drop_duplicates(subset=["tinh"]).reset_index(drop=True)
    return df


try:
    KQ = load_results()
except Exception:
    st.error("Không tìm thấy `outputs/ket_qua.json`. Chạy pipeline trước.")
    st.stop()

PRIOR = load_priority()
GEO_VN = load_vn_geojson()
UXO_PROV = load_uxo_by_province()


# ============================================================================
# HERO
# ============================================================================
st.markdown(
    f"""
    <div class="hero fade-in">
        <div class="hero-eyebrow">Bản trình diễn nghiên cứu · Phạm vi toàn quốc</div>
        <div class="hero-title">DeMine-VN</div>
        <div class="hero-sub">
            Hệ thống hỗ trợ lập bản đồ nguy cơ và xếp thứ tự ưu tiên rà phá bom
            mìn, vật nổ còn sót lại sau chiến tranh — hợp nhất hồ sơ không kích,
            ảnh vệ tinh và dữ liệu rà phá thực địa trên phạm vi toàn quốc.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# WARNING
# ============================================================================
st.markdown(
    f"""<div class='warning-card fade-in stagger-1'>
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
det   = KQ["tang_hai_phat_hien_ho_bom"]
curve = KQ["duong_cong_hieu_qua_ra_pha"]
tier1 = KQ["tang_1_doi_chung_dat_da_ra_pha"]
tier4b = KQ.get("tang_4b_hieu_chinh_xac_suat", {})

with st.sidebar:
    st.markdown("<div class='sidebar-eyebrow'>Chỉ tiêu vận hành</div>",
                 unsafe_allow_html=True)
    st.metric("Phát hiện hố bom (F1)", f"{det['f1']:.3f}")
    st.metric("Thu hồi tại 20% diện tích", f"{curve['thu_hoi_tai_20pct_dien_tich']:.1%}")
    st.metric("Lợi thế so với quét đều", f"+{curve['loi_the_so_voi_quet_deu']:.1%}")
    st.metric("Điểm Brier hiệu chỉnh", f"{tier4b.get('diem_brier', 0):.4f}")

    st.markdown(
        f"<div style='height:1px;background:{COL_HAIRLINE};margin:20px 0;'></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sidebar-eyebrow'>Quy mô dữ liệu</div>",
                 unsafe_allow_html=True)
    if not UXO_PROV.empty:
        st.metric("Tỉnh, thành phố khảo sát", f"{len(UXO_PROV)}/63")
        st.metric("Ô lưới ưu tiên",
                    f"{len(PRIOR):,}" if not PRIOR.empty else "—")

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
        f"<b style='color:{COL_INK};'>Kiến trúc.</b> YOLO11 phát hiện hố bom · "
        f"Gradient Boosting xếp hạng · Bốn tầng kiểm chứng độc lập.<br><br>"
        f"<b style='color:{COL_INK};'>Nguồn dữ liệu.</b> Hồ sơ không kích giải "
        f"mật, ảnh vệ tinh lịch sử, dữ liệu rà phá thực địa, số liệu công bố "
        f"của VNMAC.</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# TABS
# ============================================================================
tab_over, tab_national, tab_prov, tab_map, tab_det, tab_curve, tab_val, tab_data, tab_kpi = st.tabs([
    "Tổng quan",
    "Bản đồ toàn quốc",
    "Chi tiết theo tỉnh",
    "Vùng thí điểm chi tiết",
    "Phát hiện hố bom",
    "Đường cong hiệu quả",
    "Kiểm chứng độc lập",
    "Nguồn dữ liệu",
    "Chỉ tiêu tổng hợp",
])


# ============================================================================
# TAB: TỔNG QUAN
# ============================================================================
with tab_over:
    st.markdown(
        "<div class='section-title fade-in-up'>Vấn đề còn nguyên vẹn sau nửa thế kỷ</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub fade-in-up stagger-1'>"
        "Chiến tranh tại Việt Nam kết thúc năm 1975. Năm mươi năm sau, hậu quả "
        "vật lý của nó vẫn nằm nguyên trong lòng đất trên phạm vi toàn quốc. "
        "Bốn con số dưới đây do Trung tâm Hành động bom mìn Quốc gia Việt Nam công bố."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    hero_items = [
        (c1, "6,1", "triệu héc-ta ô nhiễm", "chiếm 18,71% tổng diện tích cả nước"),
        (c2, "63/63", "tỉnh, thành phố", "toàn quốc có ô nhiễm bom mìn"),
        (c3, "800K", "tấn bom đạn", "ước tính còn sót lại trong đất"),
        (c4, "100K+", "thương vong", "sau 1975 (40K tử vong, 60K bị thương)"),
    ]
    for i, (col, num, label, note) in enumerate(hero_items):
        with col:
            st.markdown(
                f"<div class='fade-in-up stagger-{i+2}'>"
                f"<div class='hero-num'>{num}</div>"
                f"<div style='color:{COL_INK};font-size:14px;font-weight:500;margin-top:8px;'>{label}</div>"
                f"<div class='hero-label'>{note}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown(
            "<div class='section-title fade-in-up stagger-5' style='font-size:24px;'>"
            "Vai trò của DeMine-VN"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:{COL_MUTED};font-size:14.5px;line-height:1.75;' class='fade-in-up stagger-6'>"
            f"Quy trình rà phá gồm hai bước. Bước một là khảo sát phi kỹ thuật để "
            f"khoanh vùng. Bước hai là rà phá kỹ thuật bằng máy dò trên từng mét vuông. "
            f"Bước thứ hai không thể rút ngắn bằng công nghệ thông tin — vẫn phải có "
            f"người cầm máy đi qua từng mét đất.<br><br>"
            f"Điểm nghẽn nằm ở bước một. Khi thông tin khoanh vùng còn thô, lực lượng "
            f"rà phá phải quét trải đều, dẫn đến phần lớn công sức được dồn vào những "
            f"khoảnh đất vốn không chứa vật nổ.<br><br>"
            f"<b style='color:{COL_INK};'>DeMine-VN thu hẹp phạm vi bước một</b> bằng "
            f"cách hợp nhất ba nguồn chứng cứ: hồ sơ phi vụ không kích, ảnh vệ tinh "
            f"lịch sử, và dữ liệu rà phá thực địa đã hoàn thành. Đầu ra là bản đồ nguy "
            f"cơ toàn quốc trên lưới 100 mét và danh mục xếp thứ tự ưu tiên."
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            f"""<div class='callout fade-in-up stagger-6'>
            <div class='callout-title'>Lợi thế so với quét đều</div>
            <div class='callout-body'>
            Với cùng nguồn lực rà phá <b>20% diện tích</b>, hệ thống thu hồi
            <b>{curve['thu_hoi_tai_20pct_dien_tich']:.1%}</b> tổng khối lượng vật nổ tồn dư
            — cao hơn phương án quét đều <b>{curve['loi_the_so_voi_quet_deu']:.1%}</b>
            theo giá trị tuyệt đối.<br><br>
            Ở mức <b>50% diện tích</b>, tỉ lệ thu hồi đạt
            <b>{curve['thu_hoi_tai_50pct_dien_tich']:.1%}</b>. Ưu điểm tăng theo diện
            tích được ưu tiên trước, phản ánh tính chất "hình chóp Pareto" của phân bố
            ô nhiễm thực tế.
            </div></div>""",
            unsafe_allow_html=True,
        )


# ============================================================================
# TAB: BẢN ĐỒ TOÀN QUỐC
# ============================================================================
with tab_national:
    st.markdown(
        "<div class='section-title'>Bản đồ mức ô nhiễm bom mìn toàn quốc</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>"
        "63 tỉnh, thành phố được tô màu theo tỉ lệ diện tích ô nhiễm. "
        "Số liệu tổng hợp từ báo cáo công khai của VNMAC và các đơn vị rà phá "
        "đối tác. Chọn vào một tỉnh để xem chỉ số cụ thể."
        "</div>",
        unsafe_allow_html=True,
    )

    if GEO_VN is None or UXO_PROV.empty:
        st.info("Chưa có dữ liệu bản đồ toàn quốc.")
    else:
        import folium
        from folium.features import GeoJsonTooltip
        import branca.colormap as cm
        from streamlit.components.v1 import html as st_html

        metric_choice = st.selectbox(
            "Chỉ số hiển thị",
            options=[
                ("ty_le_o_nhiem_pct", "Tỉ lệ diện tích ô nhiễm (%)"),
                ("dien_tich_o_nhiem_ha", "Diện tích ô nhiễm (ha)"),
                ("so_vu_tai_nan_2010_2022", "Số vụ tai nạn 2010–2022"),
                ("so_nan_nhan_2010_2022", "Số nạn nhân 2010–2022"),
                ("ky_tai_thu_hoi_tan_2015_2023", "Vật nổ thu hồi 2015–2023 (tấn)"),
            ],
            format_func=lambda x: x[1],
        )
        metric_col, metric_label = metric_choice

        df = UXO_PROV.groupby("tinh", as_index=False).agg({
            "dien_tich_tinh_ha": "first",
            "dien_tich_o_nhiem_ha": "max",
            "ty_le_o_nhiem_pct": "max",
            "so_vu_tai_nan_2010_2022": "max",
            "so_nan_nhan_2010_2022": "max",
            "ky_tai_thu_hoi_tan_2015_2023": "max",
            "ma_vung": "first",
        })

        # Prepare color scale
        vals = df[metric_col].astype(float)
        vmin, vmax = float(vals.min()), float(vals.max())
        colormap = cm.LinearColormap(
            colors=["#EFF8F7", "#7A9E5F", "#D4A017", "#D26B36", "#B62A2A"],
            vmin=vmin, vmax=vmax, caption=metric_label,
        )

        # Enrich geojson features with data attributes
        geo_enriched = json.loads(json.dumps(GEO_VN))  # deep copy
        val_by_tinh = dict(zip(df["tinh"], df[metric_col]))
        row_by_tinh = df.set_index("tinh").to_dict("index")
        for feat in geo_enriched["features"]:
            name = feat["properties"].get("ten_tinh")
            v = val_by_tinh.get(name)
            feat["properties"]["metric_val"] = v
            r = row_by_tinh.get(name, {})
            feat["properties"]["ty_le_pct"] = f"{r.get('ty_le_o_nhiem_pct', 0):.1f}%" if r else "—"
            feat["properties"]["dien_tich_o_nhiem"] = f"{int(r.get('dien_tich_o_nhiem_ha', 0)):,} ha" if r else "—"
            feat["properties"]["so_nan_nhan"] = f"{int(r.get('so_nan_nhan_2010_2022', 0)):,}" if r else "—"
            feat["properties"]["ky_tai_thu_hoi"] = f"{int(r.get('ky_tai_thu_hoi_tan_2015_2023', 0)):,} tấn" if r else "—"

        m = folium.Map(
            location=[15.9, 107.6], zoom_start=6,
            tiles="OpenStreetMap", control_scale=True,
            min_zoom=5, max_zoom=10,
        )

        def _style(feature):
            v = feature["properties"].get("metric_val")
            if v is None:
                return {"fillColor": "#DDDDDD", "color": "#FFFFFF",
                         "weight": 0.6, "fillOpacity": 0.4}
            return {"fillColor": colormap(v), "color": "#FFFFFF",
                     "weight": 0.8, "fillOpacity": 0.82}

        def _highlight(feature):
            return {"weight": 2.5, "color": "#1D1D1F", "fillOpacity": 0.92}

        gj = folium.GeoJson(
            geo_enriched,
            style_function=_style,
            highlight_function=_highlight,
            tooltip=GeoJsonTooltip(
                fields=["ten_tinh", "ty_le_pct", "dien_tich_o_nhiem",
                        "so_nan_nhan", "ky_tai_thu_hoi"],
                aliases=["Tỉnh", "Tỉ lệ ô nhiễm", "Diện tích ô nhiễm",
                          "Nạn nhân 2010–2022", "Vật nổ thu hồi"],
                sticky=True,
                labels=True,
                localize=True,
                style=(
                    "background:rgba(255,255,255,0.96);"
                    "backdrop-filter:blur(12px);"
                    "border:1px solid #E5E5EA;border-radius:10px;"
                    "padding:10px 12px;"
                    "font-family:-apple-system,BlinkMacSystemFont,sans-serif;"
                    "font-size:12.5px;color:#1D1D1F;"
                    "box-shadow:0 4px 16px rgba(0,0,0,0.08);"
                ),
            ),
        )
        gj.add_to(m)
        colormap.add_to(m)

        st_html(m.get_root().render(), height=680)

        # KPI strip below map
        top5 = df.nlargest(5, metric_col)
        st.markdown(
            f"<div style='font-size:11px;font-weight:500;color:{COL_MUTED};"
            f"text-transform:uppercase;letter-spacing:0.06em;margin:20px 0 10px 0;'>"
            f"Năm địa phương dẫn đầu — {metric_label.lower()}</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(5)
        for i, (col, (_, r)) in enumerate(zip(cols, top5.iterrows())):
            val = r[metric_col]
            if metric_col == "ty_le_o_nhiem_pct":
                display = f"{val:.1f}%"
            elif metric_col in ["dien_tich_o_nhiem_ha", "ky_tai_thu_hoi_tan_2015_2023"]:
                display = f"{int(val):,}"
            else:
                display = f"{int(val):,}"
            with col:
                st.markdown(
                    f"<div class='kpi fade-in-up stagger-{i+1}'>"
                    f"<div class='kpi-label'>{r['tinh']}</div>"
                    f"<div class='kpi-value' style='font-size:26px;'>{display}</div>"
                    f"<div class='kpi-note'>Vùng: {r['ma_vung']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ============================================================================
# TAB: CHI TIẾT THEO TỈNH
# ============================================================================
with tab_prov:
    st.markdown(
        "<div class='section-title'>Thông tin ô nhiễm theo tỉnh</div>",
        unsafe_allow_html=True,
    )

    if UXO_PROV.empty:
        st.info("Chưa có dữ liệu tỉnh.")
    else:
        df = UXO_PROV.groupby("tinh", as_index=False).agg({
            "dien_tich_tinh_ha": "first",
            "dien_tich_o_nhiem_ha": "max",
            "ty_le_o_nhiem_pct": "max",
            "so_vu_tai_nan_2010_2022": "max",
            "so_nan_nhan_2010_2022": "max",
            "ky_tai_thu_hoi_tan_2015_2023": "max",
            "ma_vung": "first",
            "ghi_chu": "first",
        }).sort_values("ty_le_o_nhiem_pct", ascending=False)

        province_sel = st.selectbox(
            "Chọn tỉnh, thành phố",
            options=df["tinh"].tolist(),
            index=0,
        )
        row = df[df["tinh"] == province_sel].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tỉ lệ diện tích ô nhiễm", f"{row['ty_le_o_nhiem_pct']:.1f}%")
        c2.metric("Diện tích ô nhiễm", f"{int(row['dien_tich_o_nhiem_ha']):,} ha")
        c3.metric("Nạn nhân 2010–2022", f"{int(row['so_nan_nhan_2010_2022']):,}")
        c4.metric("Vật nổ thu hồi 2015–2023", f"{int(row['ky_tai_thu_hoi_tan_2015_2023']):,} tấn")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class='callout fade-in-up'>
            <div class='callout-title'>{row['tinh']} — ghi chú chuyên môn</div>
            <div class='callout-body'>
            {row['ghi_chu']}. Vùng địa lý: <b>{row['ma_vung']}</b>. Diện tích tự nhiên
            tỉnh: <b>{int(row['dien_tich_tinh_ha']):,}</b> ha. Trong đó
            <b>{int(row['dien_tich_o_nhiem_ha']):,}</b> ha bị ô nhiễm hoặc nghi ngờ ô
            nhiễm, chiếm <b>{row['ty_le_o_nhiem_pct']:.1f}%</b> tổng diện tích.
            </div></div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size:20px;font-weight:500;color:#1D1D1F;"
            "letter-spacing:-0.02em;margin:24px 0 12px 0;'>"
            "Bảng đầy đủ 63 tỉnh, thành phố</div>",
            unsafe_allow_html=True,
        )
        df_display = df.copy()
        df_display.columns = [
            "Tỉnh", "Vùng", "Diện tích tự nhiên (ha)",
            "Diện tích ô nhiễm (ha)", "Tỉ lệ ô nhiễm (%)",
            "Số vụ tai nạn 2010–2022", "Số nạn nhân 2010–2022",
            "Vật nổ thu hồi 2015–2023 (tấn)", "Ghi chú",
        ]
        st.dataframe(
            df_display.style.format({
                "Diện tích tự nhiên (ha)": "{:,.0f}",
                "Diện tích ô nhiễm (ha)": "{:,.0f}",
                "Tỉ lệ ô nhiễm (%)": "{:.1f}",
                "Số vụ tai nạn 2010–2022": "{:,}",
                "Số nạn nhân 2010–2022": "{:,}",
                "Vật nổ thu hồi 2015–2023 (tấn)": "{:,}",
            }).background_gradient(
                subset=["Tỉ lệ ô nhiễm (%)"],
                cmap="OrRd",
                vmin=5, vmax=80,
            ),
            use_container_width=True, hide_index=True, height=420,
        )


# ============================================================================
# TAB: VÙNG THÍ ĐIỂM
# ============================================================================
with tab_map:
    from streamlit.components.v1 import html as st_html
    import folium
    from folium.plugins import HeatMap

    st.markdown(
        "<div class='section-title'>Vùng thí điểm Quảng Trị — bản đồ ưu tiên rà phá</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>"
        "Vùng thí điểm quy mô 220×180 ô lưới 100 mét trên vùng biên giới Quảng "
        "Trị – Thừa Thiên Huế, nơi ô nhiễm bom mìn nặng nhất cả nước. Kết quả "
        "sẽ được nhân rộng sang các tỉnh khác theo lộ trình."
        "</div>",
        unsafe_allow_html=True,
    )

    if PRIOR.empty:
        st.info("Chưa có bản đồ ưu tiên.")
    else:
        top_n = st.slider(
            "Hiển thị bao nhiêu ô ưu tiên hàng đầu",
            min_value=100, max_value=min(2000, len(PRIOR)),
            value=500, step=100,
        )
        subset = PRIOR.head(top_n)

        m = folium.Map(
            location=[subset["vi_do"].mean(), subset["kinh_do"].mean()],
            zoom_start=12, tiles="OpenStreetMap", control_scale=True,
        )
        heat_data = [[r["vi_do"], r["kinh_do"], r["chi_so_uu_tien"]]
                       for _, r in subset.iterrows()]
        HeatMap(heat_data, radius=14, blur=18, max_zoom=13,
                 gradient={0.3: "#5B7F5A", 0.55: "#B7791F", 0.8: "#B62A2A"}).add_to(m)

        for _, r in subset.head(30).iterrows():
            popup = folium.Popup(
                f"<div style='font-family:-apple-system,sans-serif;min-width:240px;color:#1D1D1F;'>"
                f"<div style='font-weight:600;font-size:14px;letter-spacing:-0.01em;'>"
                f"Ô ưu tiên #{int(r['thu_tu_uu_tien'])}</div>"
                f"<div style='height:1px;background:#E5E5EA;margin:8px 0;'></div>"
                f"<table style='font-size:12.5px;'>"
                f"<tr><td style='color:#6E6E73;padding:3px 12px 3px 0;'>Chỉ số ưu tiên</td>"
                f"<td><b>{r['chi_so_uu_tien']:.4f}</b></td></tr>"
                f"<tr><td style='color:#6E6E73;padding:3px 12px 3px 0;'>Xác suất còn vật nổ</td>"
                f"<td>{r['xac_suat_con_vat_no']:.4f}</td></tr>"
                f"<tr><td style='color:#6E6E73;padding:3px 12px 3px 0;'>Tải trọng bom</td>"
                f"<td>{r['tai_trong_bom_ghi_nhan_tan']:.2f} tấn</td></tr>"
                f"<tr><td style='color:#6E6E73;padding:3px 12px 3px 0;'>Hố bom quan sát</td>"
                f"<td>{int(r['so_ho_bom_phat_hien'])}</td></tr>"
                f"<tr><td style='color:#6E6E73;padding:3px 12px 3px 0;'>Khu dân cư gần nhất</td>"
                f"<td>{int(r['khoang_cach_khu_dan_cu_m'])} m</td></tr>"
                f"<tr><td style='color:#6E6E73;padding:3px 12px 3px 0;'>Đất canh tác</td>"
                f"<td>{r['la_dat_canh_tac']}</td></tr>"
                f"</table>"
                f"<div style='margin-top:10px;padding-top:8px;border-top:1px solid #E5E5EA;"
                f"font-size:11px;color:#8E8E93;line-height:1.5;'>"
                f"Thứ tự ưu tiên rà phá. Không phải xác nhận an toàn.</div></div>",
                max_width=340,
            )
            folium.CircleMarker(
                location=[r["vi_do"], r["kinh_do"]],
                radius=5, color="#B62A2A", weight=1.5, fill=True,
                fillColor="#B62A2A", fillOpacity=0.85, popup=popup,
            ).add_to(m)

        legend_html = f'''
        <div style="position:fixed; top:100px; right:20px; z-index:9999;
                     background:rgba(255,255,255,0.94);
                     backdrop-filter: saturate(180%) blur(20px);
                     -webkit-backdrop-filter: saturate(180%) blur(20px);
                     padding:14px 18px; border-radius:12px;
                     font-family:-apple-system,sans-serif; font-size:12px;
                     border:1px solid rgba(0,0,0,0.06);
                     box-shadow:0 4px 16px rgba(0,0,0,0.08); color:#1D1D1F;">
            <div style="font-weight:500; font-size:11px; text-transform:uppercase;
                        letter-spacing:0.06em; color:#6E6E73; margin-bottom:10px;">
                Chỉ số ưu tiên</div>
            <div style="margin:5px 0;"><span style="display:inline-block;width:10px;height:10px;
                background:{COL_HIGH};border-radius:50%;margin-right:8px;"></span>Cao (0,8–1,0)</div>
            <div style="margin:5px 0;"><span style="display:inline-block;width:10px;height:10px;
                background:{COL_MED};border-radius:50%;margin-right:8px;"></span>Trung (0,55–0,8)</div>
            <div style="margin:5px 0;"><span style="display:inline-block;width:10px;height:10px;
                background:{COL_LOW};border-radius:50%;margin-right:8px;"></span>Thấp</div>
            <div style="margin-top:10px;font-size:11px;color:#86868B;">
                30 ô đầu có popup chi tiết</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        st_html(m.get_root().render(), height=680)

        st.markdown(
            f'''<div class="kpi-strip">
            <div class="kpi fade-in-up stagger-1">
                <div class="kpi-label">Ô đang hiển thị</div>
                <div class="kpi-value">{top_n:,}</div>
                <div class="kpi-note">trên tổng {len(PRIOR):,} ô lưới</div>
            </div>
            <div class="kpi fade-in-up stagger-2">
                <div class="kpi-label">Xác suất trung bình</div>
                <div class="kpi-value">{subset['xac_suat_con_vat_no'].mean():.3f}</div>
                <div class="kpi-note">còn vật nổ trong nhóm hiển thị</div>
            </div>
            <div class="kpi fade-in-up stagger-3">
                <div class="kpi-label">Tải trọng bom TB</div>
                <div class="kpi-value">{subset['tai_trong_bom_ghi_nhan_tan'].mean():.2f}</div>
                <div class="kpi-note">tấn, theo hồ sơ giải mật</div>
            </div>
            <div class="kpi fade-in-up stagger-4">
                <div class="kpi-label">Có hố bom quan sát</div>
                <div class="kpi-value">{(subset['so_ho_bom_phat_hien'] > 0).sum():,}</div>
                <div class="kpi-note">ô có ít nhất một hố bom trên ảnh</div>
            </div>
            </div>''',
            unsafe_allow_html=True,
        )


# ============================================================================
# TAB: PHÁT HIỆN HỐ BOM
# ============================================================================
with tab_det:
    st.markdown(
        "<div class='section-title'>Phát hiện hố bom trên ảnh vệ tinh</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>"
        "Mô hình YOLO11 được huấn luyện để phát hiện hố bom trên ảnh vệ tinh "
        "độ phân giải trung bình. Kết quả bên dưới là chỉ tiêu trên tập kiểm định."
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    metrics = [
        ("Precision", f"{det['precision']:.3f}"),
        ("Recall", f"{det['recall']:.3f}"),
        ("F1", f"{det['f1']:.3f}"),
        ("mAP@0.5", f"{det['mAP@0.5']:.3f}"),
        ("mAP@0.5:0.95", f"{det['mAP@0.5:0.95']:.3f}"),
    ]
    for col, (lbl, val) in zip(cols, metrics):
        col.metric(lbl, val)

    st.markdown(
        f"<div style='color:{COL_MUTED};margin-top:18px;font-size:13.5px;line-height:1.6;'>"
        f"Trên tập kiểm định: <b>{det['so_ho_bom_that']:,}</b> hố bom thật, mô hình dự "
        f"báo <b>{det['so_ho_bom_du_bao']:,}</b> hố. Sai lệch giữa hai con số "
        f"({abs(det['so_ho_bom_du_bao'] - det['so_ho_bom_that'])} hố) phản ánh biên "
        f"dương/âm giả trong ngưỡng chấp nhận được của bài toán quy mô rộng."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    fig07 = load_figure("07_anh_ve_tinh_mau.png")
    if fig07 is not None:
        st.markdown(
            f"<div style='font-size:17px;font-weight:500;color:{COL_INK};"
            f"letter-spacing:-0.01em;margin:12px 0 8px 0;'>"
            f"Ví dụ ảnh vệ tinh và hố bom được phát hiện</div>",
            unsafe_allow_html=True,
        )
        st.image(fig07, use_container_width=True)

    fig02 = load_figure("02_ban_do_nguy_co.png")
    if fig02 is not None:
        st.markdown(
            f"<div style='font-size:17px;font-weight:500;color:{COL_INK};"
            f"letter-spacing:-0.01em;margin:20px 0 8px 0;'>"
            f"Bản đồ nguy cơ trên lưới 100 mét</div>",
            unsafe_allow_html=True,
        )
        st.image(fig02, use_container_width=True)

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
    st.markdown(
        "<div class='section-title'>Đường cong hiệu quả rà phá</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>"
        "Trục ngang: tỉ lệ diện tích được rà phá theo thứ tự ưu tiên. Trục dọc: "
        "tỉ lệ khối lượng vật nổ đã thu hồi. Đường thẳng chéo là kịch bản quét đều. "
        "Khoảng cách giữa đường cong của mô hình và đường chéo là lợi thế của việc ưu tiên."
        "</div>",
        unsafe_allow_html=True,
    )

    import plotly.graph_objects as go

    if not PRIOR.empty:
        s = PRIOR.sort_values("chi_so_uu_tien", ascending=False)
        weight = s["xac_suat_con_vat_no"].values
        cum = np.cumsum(weight) / weight.sum()
        area = np.arange(1, len(cum) + 1) / len(cum)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=COL_MUTED, width=1.5, dash="dot"),
            name="Kịch bản quét đều",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=area, y=cum, mode="lines",
            line=dict(color=COL_INK, width=3),
            fill="tonexty", fillcolor="rgba(0,113,227,0.08)",
            name="DeMine-VN — hệ thống đầy đủ",
            hovertemplate="Diện tích: %{x:.1%}<br>Vật nổ thu hồi: %{y:.1%}<extra></extra>",
        ))
        for pct, y in [
            (0.10, curve["thu_hoi_tai_10pct_dien_tich"]),
            (0.20, curve["thu_hoi_tai_20pct_dien_tich"]),
            (0.50, curve["thu_hoi_tai_50pct_dien_tich"]),
        ]:
            fig.add_trace(go.Scatter(
                x=[pct], y=[y], mode="markers+text",
                marker=dict(size=12, color=COL_ACCENT,
                            line=dict(color="white", width=2)),
                text=[f"{y:.0%}"], textposition="top center",
                textfont=dict(color=COL_INK, size=12,
                              family="-apple-system"),
                showlegend=False, hoverinfo="skip",
            ))

        fig.update_layout(
            height=480,
            plot_bgcolor=COL_CARD, paper_bgcolor=COL_CARD,
            xaxis=dict(title="Tỉ lệ diện tích được rà phá theo thứ tự ưu tiên",
                        tickformat=".0%", gridcolor=COL_HAIRLINE,
                        linecolor=COL_HAIRLINE, range=[0, 1]),
            yaxis=dict(title="Tỉ lệ khối lượng vật nổ đã thu hồi",
                        tickformat=".0%", gridcolor=COL_HAIRLINE,
                        linecolor=COL_HAIRLINE, range=[0, 1.02]),
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center",
                         bgcolor="rgba(0,0,0,0)",
                         font=dict(color=COL_INK, size=12)),
            font=dict(color=COL_INK, family="-apple-system"),
            margin=dict(l=60, r=30, t=60, b=50),
            transition=dict(duration=500, easing="cubic-in-out"),
        )
        st.plotly_chart(fig, use_container_width=True,
                         config={"displaylogo": False})

    st.markdown(
        f'''<div class="kpi-strip">
        <div class="kpi fade-in-up stagger-1">
            <div class="kpi-label">Rà phá 10% diện tích</div>
            <div class="kpi-value">{curve['thu_hoi_tai_10pct_dien_tich']:.1%}</div>
            <div class="kpi-note">vật nổ đã thu hồi</div>
        </div>
        <div class="kpi fade-in-up stagger-2">
            <div class="kpi-label">Rà phá 20% diện tích</div>
            <div class="kpi-value">{curve['thu_hoi_tai_20pct_dien_tich']:.1%}</div>
            <div class="kpi-note">vật nổ đã thu hồi</div>
        </div>
        <div class="kpi fade-in-up stagger-3">
            <div class="kpi-label">Rà phá 50% diện tích</div>
            <div class="kpi-value">{curve['thu_hoi_tai_50pct_dien_tich']:.1%}</div>
            <div class="kpi-note">vật nổ đã thu hồi</div>
        </div>
        <div class="kpi fade-in-up stagger-4">
            <div class="kpi-label">Lợi thế trung bình</div>
            <div class="kpi-value" style="color:{COL_ACCENT}">+{curve['loi_the_so_voi_quet_deu']:.1%}</div>
            <div class="kpi-note">so với phương án quét đều</div>
        </div>
        </div>''',
        unsafe_allow_html=True,
    )


# ============================================================================
# TAB: KIỂM CHỨNG
# ============================================================================
with tab_val:
    st.markdown(
        "<div class='section-title'>Kiểm chứng độc lập ở bốn tầng</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>"
        "Bài toán rà phá không có nhãn hoàn hảo. Không thể chờ nhiều năm để biết dự "
        "báo có đúng hay không. Vì vậy hệ thống được kiểm chứng bằng bốn tầng chứng "
        "cứ độc lập — đối chứng nội bộ, đối chứng lịch sử tai nạn, đối chứng chéo hai "
        "nguồn, và kiểm tra tính chuyển vùng địa lý."
        "</div>",
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
            f"<div style='color:{COL_MUTED};font-size:14px;line-height:1.6;'>"
            f"Chia dữ liệu ngẫu nhiên nhiều lần, huấn luyện trên phần đầu và kiểm tra "
            f"trên phần sau. Đo tỉ lệ thu hồi vật nổ tại 20% diện tích ưu tiên đầu."
            f"</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Số lần chia", str(t1["so_lan_chia"]))
        c2.metric("Thu hồi tại 20% (trung bình)",
                    f"{t1['thu_hoi_tai_20pct_trung_binh']:.1%}")
        c3.metric("Độ lệch chuẩn", f"{t1['do_lech_chuan']:.4f}")

    with tabs_val[1]:
        t2 = KQ.get("tang_2_doi_chung_ho_so_tai_nan", [])
        if t2:
            df2 = pd.DataFrame(t2)
            df2.columns = ["Tập kiểm chứng", "Số vụ tai nạn",
                             "Bao phủ 10%", "Bao phủ 20%", "Bao phủ 30%",
                             "Hệ số vượt ngẫu nhiên (20%)"]
            st.markdown(
                f"<div style='color:{COL_MUTED};font-size:14px;line-height:1.6;margin-bottom:8px;'>"
                f"Kiểm chứng ngoài bằng hồ sơ tai nạn có thật đã ghi nhận trong quá khứ."
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
            f"<div style='color:{COL_MUTED};font-size:14px;line-height:1.6;'>"
            f"Số hố bom phát hiện qua ảnh vệ tinh và tải trọng bom trong hồ sơ không "
            f"kích phải tương quan với nhau — hai nguồn khác nhau, cùng một hiện tượng vật lý."
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
            f"<div style='color:{COL_MUTED};font-size:14px;line-height:1.6;'>"
            f"Huấn luyện trên một vùng địa lý và kiểm chứng trên vùng khác chưa từng thấy."
            f"</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Cùng vùng — thu hồi 20%",
                    f"{t4a.get('thu_hoi_tai_20pct_cung_vung', 0):.1%}")
        c2.metric("Vùng mới — thu hồi 20%",
                    f"{t4a.get('thu_hoi_tai_20pct_vung_moi', 0):.1%}",
                    delta=f"{t4a.get('muc_suy_giam_tuong_doi', 0):.1%}")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Điểm Brier", f"{t4b.get('diem_brier', 0):.4f}")
        c2.metric("Sai số hiệu chỉnh", f"{t4b.get('sai_so_hieu_chinh_ky_vong', 0):.4f}")
        c3.metric("Xác suất trung bình", f"{t4b.get('xac_suat_du_bao_trung_binh', 0):.4f}")
        c4.metric("Tỉ lệ thực tế", f"{t4b.get('ty_le_duong_thuc_te', 0):.4f}")

    fig04 = load_figure("04_bieu_do_tin_cay.png")
    if fig04 is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:17px;font-weight:500;color:{COL_INK};"
            f"letter-spacing:-0.01em;margin:8px 0 6px 0;'>Biểu đồ hiệu chỉnh</div>",
            unsafe_allow_html=True,
        )
        st.image(fig04, use_container_width=True)


# ============================================================================
# TAB: NGUỒN DỮ LIỆU
# ============================================================================
with tab_data:
    st.markdown(
        "<div class='section-title'>Nguồn dữ liệu — công khai và tham chiếu</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>"
        "Hệ thống hợp nhất ba nguồn chứng cứ độc lập. Không có nguồn nào là hoàn hảo. "
        "Sức mạnh của mô hình nằm ở sự bổ sung."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for col, cls, title, body in [
        (c1, "stagger-1", "Hồ sơ không kích giải mật",
         "Bộ dữ liệu THOR (Theater History of Operations Reports) do Bộ Quốc phòng "
         "Hoa Kỳ giải mật, ghi chép hơn 4 triệu phi vụ chiến tranh Việt Nam. "
         "<b>Điểm mạnh.</b> Bao phủ toàn quốc, có toạ độ mục tiêu và tải trọng. "
         "<b>Điểm yếu.</b> Sai số định vị lịch sử ~180 m; ~8% phi vụ mất hồ sơ."),
        (c2, "stagger-2", "Ảnh vệ tinh lịch sử",
         "Ảnh vệ tinh độ phân giải trung bình cho phép phát hiện hố bom cũ vẫn còn "
         "lộ trên mặt đất. <b>Điểm mạnh.</b> Bằng chứng vật lý trực tiếp, độc lập "
         "với hồ sơ. <b>Điểm yếu.</b> Trên nền đất mềm, hố bị bồi lấp; hố cũng có "
         "thể bị che khuất bởi thực vật."),
        (c3, "stagger-3", "Dữ liệu rà phá thực địa",
         "Bản ghi các khoảnh đã được đội rà phá quét sạch trong quá khứ — thông tin "
         "có tính xác thực cao nhất vì đến từ máy dò thực tế. <b>Điểm mạnh.</b> "
         "Nhãn ground-truth cho hiệu chỉnh mô hình. <b>Điểm yếu.</b> Chỉ bao phủ "
         "một phần diện tích."),
    ]:
        with col:
            st.markdown(
                f"<div class='callout fade-in-up {cls}'>"
                f"<div class='callout-title'>{title}</div>"
                f"<div class='callout-body'>{body}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:20px;font-weight:500;color:#1D1D1F;"
        "letter-spacing:-0.02em;margin:16px 0 12px 0;'>"
        "Đóng góp riêng của từng nguồn — nghiên cứu ablation</div>",
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
        f"""<div class='callout' style='margin-top:20px;'>
        <div class='callout-title'>Bộ dữ liệu công khai tham chiếu</div>
        <div class='callout-body'>
        <b>THOR — Theater History of Operations Reports.</b> Hồ sơ giải mật của Bộ
        Quốc phòng Hoa Kỳ về các phi vụ ném bom giai đoạn 1965–1975. Có thể tải
        về từ <code>data.mil</code> hoặc <code>catalog.data.gov</code>.<br><br>
        <b>Landmine and Cluster Munition Monitor — Vietnam Country Profile.</b>
        Số liệu tai nạn cập nhật hằng năm, do The Monitor công bố tại
        <code>the-monitor.org</code>.<br><br>
        <b>VNMAC.</b> Báo cáo thường niên của Trung tâm Hành động bom mìn Quốc gia
        Việt Nam về tình trạng ô nhiễm cấp tỉnh và tiến độ khắc phục.<br><br>
        <b>OpenStreetMap và GADM.</b> Biên giới hành chính 63 tỉnh, thành phố dùng
        cho bản đồ giám sát.
        </div></div>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# TAB: CHỈ TIÊU TỔNG HỢP
# ============================================================================
with tab_kpi:
    st.markdown(
        "<div class='section-title'>Chỉ tiêu tổng hợp bản chạy hiện tại</div>",
        unsafe_allow_html=True,
    )

    inner = st.tabs([
        "Đặc trưng đóng góp lớn nhất",
        "Đóng góp theo hoán vị",
        "Môi trường chạy",
    ])

    with inner[0]:
        feats = KQ.get("muc_dong_gop_dac_trung", {})
        if feats:
            df = pd.DataFrame([{"Đặc trưng": k, "Trọng số": v}
                                 for k, v in list(feats.items())[:15]])
            st.dataframe(df.style.format({"Trọng số": "{:.4f}"})
                              .background_gradient(subset=["Trọng số"], cmap="Blues"),
                          use_container_width=True, hide_index=True)

    with inner[1]:
        perm = KQ.get("muc_dong_gop_theo_hoan_vi", {})
        if perm:
            df = pd.DataFrame([{"Đặc trưng": k, "Mức đóng góp": v}
                                 for k, v in list(perm.items())[:15]])
            st.dataframe(df.style.format({"Mức đóng góp": "{:.4f}"})
                              .background_gradient(subset=["Mức đóng góp"], cmap="Purples"),
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
    f"<div style='height:1px;background:{COL_HAIRLINE};margin:40px 0 12px 0;'></div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='color:{COL_MUTED};font-size:12px;text-align:center;padding:16px 0;line-height:1.7;'>"
    f"DeMine-VN · Hệ thống hỗ trợ xếp thứ tự ưu tiên rà phá bom mìn<br>"
    f"Kết quả mô hình mang tính hỗ trợ nghiệp vụ. Không thay thế quy trình rà phá kỹ thuật."
    f"</div>",
    unsafe_allow_html=True,
)
