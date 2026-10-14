"""
Package Inspection & Shipment Readiness - Streamlit Frontend
Light logistics / warehouse-control-room interface with per-page background images.
"""

from pathlib import Path
import os
import sys

# Prevent Anaconda numexpr / NumPy 2.x binary conflict
if "numexpr" not in sys.modules:
    sys.modules["numexpr"] = None

import io
import csv
import zipfile
import tempfile
import base64
import cv2
from PIL import Image
import streamlit as st

# Ensure local directory is in sys.path
_APP_DIR = Path(__file__).parent.resolve()
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from readiness_engine import (
    assess_package,
    DAMAGE_CONF_THRESHOLD,
    PACKAGE_NOT_FOUND_THRESHOLD,
    PACKAGE_LOW_CONFIDENCE_THRESHOLD,
)

# ==============================================================================
# EDITABLE CONSTANTS & CONFIGURATION
# ==============================================================================
DEFAULT_MODEL_PATH = "best_yolov8s.pt"
OVERLAY_OPACITY = 0.90

# About Page Content Constants (short & factual, no marketing/hardware terms)
ABOUT_CARD_1_TITLE = "What it does"
ABOUT_CARD_1_TEXT = (
    "Upload a package photo. The system detects the package and visible damage, "
    "checks for a barcode, and gives one verdict: Ready for Shipment, "
    "Repackage Required or Manual Inspection."
)

ABOUT_CARD_2_TITLE = "How the verdict is decided"
ABOUT_CARD_2_RULES = [
    (f"No package found (confidence below {PACKAGE_NOT_FOUND_THRESHOLD:.2f})", "Manual Inspection"),
    (f"Damage confidence {DAMAGE_CONF_THRESHOLD:.2f} or higher", "Repackage Required"),
    (f"Package confidence below {PACKAGE_LOW_CONFIDENCE_THRESHOLD:.2f}", "Manual Inspection (image unclear)"),
    ("Otherwise", "Ready for Shipment"),
]
ABOUT_CARD_2_NOTE = "The barcode result is shown for information and does not change the verdict."

ABOUT_CARD_3_TITLE = "Model"
ABOUT_CARD_3_TEXT = "YOLOv8s detector (classes: package, damage)."

# Decision Rules for Sidebar (read-only)
SIDEBAR_DECISION_RULES = f"""• No package found (< {PACKAGE_NOT_FOUND_THRESHOLD:.2f}) -> Manual Inspection
• Damage confidence >= {DAMAGE_CONF_THRESHOLD:.2f} -> Repackage Required
• Package confidence < {PACKAGE_LOW_CONFIDENCE_THRESHOLD:.2f} -> Manual Inspection (unclear)
• Otherwise -> Ready for Shipment"""

# Unified Verdict Theme Colors
VERDICT_THEMES = {
    "Ready for Shipment": {
        "text": "#1E9E5A",
        "bg": "#E7F6EE",
        "border": "#1E9E5A",
        "icon": "✓",
        "tag": "PASS • DISPATCH APPROVED",
    },
    "Repackage Required": {
        "text": "#D64545",
        "bg": "#FDECEC",
        "border": "#D64545",
        "icon": "✕",
        "tag": "REJECT • REPACKAGE REQUIRED",
    },
    "Manual Inspection": {
        "text": "#C98512",
        "bg": "#FFF4DD",
        "border": "#C98512",
        "icon": "⚠",
        "tag": "HOLD • MANUAL INSPECTION",
    },
}


# ==============================================================================
# BACKGROUND IMAGE PROCESSING & CACHING
# ==============================================================================
@st.cache_data
def get_page_background_css(page_name: str, opacity: float) -> str:
    """
    Loads page background (bg1.png, bg2.png, bg3.png), downscales to max 1600px,
    converts to JPEG quality 80, base64 encodes it, and wraps it in a fixed cover
    CSS with a layered white gradient overlay. Falls back to light gradient if missing.
    """
    page_files = {
        "Inspect": "bg1.png",
        "Batch": "bg2.png",
        "About": "bg3.png",
    }
    filename = page_files.get(page_name, "bg1.png")
    img_path = Path(__file__).parent / filename

    fallback_css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #F7F4EE 0%, #EAF0F6 100%) fixed !important;
        background-size: cover !important;
    }
    </style>
    """

    if not img_path.exists():
        return fallback_css

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            max_w = 1600
            if img.width > max_w:
                new_h = int(img.height * (max_w / img.width))
                img = img.resize((max_w, new_h), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        overlay = (
            f"linear-gradient(rgba(255, 255, 255, {opacity:.2f}), "
            f"rgba(255, 255, 255, {opacity:.2f}))"
        )
        return f"""
        <style>
        .stApp {{
            background: {overlay}, url("data:image/jpeg;base64,{b64_str}") no-repeat center center fixed !important;
            background-size: cover !important;
        }}
        </style>
        """
    except Exception:
        return fallback_css


# ==============================================================================
# GLOBAL LIGHT THEME CSS
# ==============================================================================
def css(active_page: str):
    """Injects core light-theme CSS and the active page background."""
    # 1. Page Background
    bg_css = get_page_background_css(active_page, OVERLAY_OPACITY)
    st.markdown(bg_css, unsafe_allow_html=True)

    # 2. General Light-Theme Component Styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Hide Streamlit default menu/footer */
        #MainMenu {visibility: hidden; display: none;}
        header {visibility: hidden; display: none;}
        footer {visibility: hidden; display: none;}

        /* Base Typography & Canvas */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #12203A;
        }

        .stApp {
            color: #12203A;
        }

        /* Light Cards */
        .qc-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 22px 24px;
            box-shadow: 0 4px 20px -2px rgba(18, 32, 58, 0.08);
            margin-bottom: 20px;
        }

        /* Sidebar Styling (Light Theme) */
        section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-right: 1px solid #E2E8F0 !important;
            color: #12203A !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #E2E8F0 !important;
        }

        /* Chips in Header */
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: #FFF4DD;
            color: #C98512;
            border: 1px solid rgba(201, 133, 18, 0.35);
            margin-right: 8px;
            margin-top: 6px;
        }

        /* Empty State Dashed Area */
        .empty-state-box {
            border: 2px dashed #CBD5E1;
            border-radius: 16px;
            padding: 38px 24px;
            text-align: center;
            background: rgba(255, 255, 255, 0.85);
            color: #5A6E85;
            margin-top: 10px;
        }

        /* Navigation Radio styled as horizontal tabs */
        div[data-testid="stRadio"] > div {
            display: flex !important;
            flex-direction: row !important;
            gap: 8px !important;
            background: rgba(255, 255, 255, 0.92) !important;
            padding: 6px 8px !important;
            border-radius: 9999px !important;
            border: 1px solid #E2E8F0 !important;
            width: fit-content !important;
            box-shadow: 0 2px 10px rgba(18, 32, 58, 0.05) !important;
            margin-bottom: 24px !important;
        }
        div[data-testid="stRadio"] label {
            padding: 8px 22px !important;
            border-radius: 9999px !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            color: #12203A !important;
            cursor: pointer !important;
            margin: 0 !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stRadio"] label > div:first-child {
            display: none !important;
        }
        div[data-testid="stRadio"] label:has(input:checked) {
            background-color: #E8A33D !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 8px rgba(232, 163, 61, 0.35) !important;
        }
        div[data-testid="stRadio"] label:has(input:checked) span,
        div[data-testid="stRadio"] label:has(input:checked) p {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            background-color: #E8A33D !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 2px 6px rgba(232, 163, 61, 0.25) !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover {
            background-color: #D69330 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(232, 163, 61, 0.35) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MODEL CACHING
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_cached_model(path: str):
    """Loads YOLO model once and utilizes engine's internal cache."""
    from readiness_engine import _get_model
    return _get_model(path)


# ==============================================================================
# COMMON UI HELPERS
# ==============================================================================
def render_header():
    """Renders the top logistics control-room hero header."""
    st.markdown(
        """
        <div style="margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid #E2E8F0;">
            <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 4px;">
                <span style="font-size: 2.2rem;">📦</span>
                <div>
                    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #12203A; letter-spacing: -0.02em;">
                        Package Inspection & Shipment Readiness
                    </h1>
                    <div style="font-size: 0.95rem; color: #5A6E85; margin-top: 2px;">
                        Automated package quality check and dispatch verification
                    </div>
                </div>
            </div>
            <div style="margin-top: 8px;">
                <span class="chip">🎯 YOLOv8s</span>
                <span class="chip">🛡️ Damage detection</span>
                <span class="chip">🏷️ Barcode check</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Renders the sidebar navigation, operational workflow, rules, and model config."""
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="font-size: 1.3rem;">⚙️</span>
            <span style="font-size: 1.15rem; font-weight: 700; color: #12203A;">Settings & Rules</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_input = st.sidebar.text_input(
        "Model Path",
        value=DEFAULT_MODEL_PATH,
        help="Path to YOLO weights file (e.g. best_yolov8s.pt)",
    )

    if not os.path.exists(model_input):
        st.sidebar.warning(f"⚠️ Model file '{model_input}' was not found.")
    else:
        st.sidebar.caption("✅ Model weights verified")

    st.sidebar.markdown("---")

    # Workflow Steps
    st.sidebar.markdown(
        """
        <div style="font-size: 0.82rem; font-weight: 700; color: #C98512; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 12px;">
            How It Works
        </div>
        <div style="margin-bottom: 12px; font-size: 0.88rem; line-height: 1.45; color: #12203A;">
            <strong>1. Upload Image:</strong> Provide package photo (JPG, JPEG, PNG).
        </div>
        <div style="margin-bottom: 12px; font-size: 0.88rem; line-height: 1.45; color: #12203A;">
            <strong>2. Run Detection:</strong> YOLOv8s locates package and checks for surface damages.
        </div>
        <div style="margin-bottom: 16px; font-size: 0.88rem; line-height: 1.45; color: #12203A;">
            <strong>3. Verify Barcode:</strong> Multi-tier scan inspects label and assigns dispatch verdict.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # Decision Rules Box
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.82rem; font-weight: 700; color: #C98512; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px;">
            Decision Rules (Read-Only)
        </div>
        <div style="
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 12px 14px;
            font-family: monospace;
            font-size: 0.78rem;
            color: #12203A;
            line-height: 1.55;
            white-space: pre-wrap;
        ">{SIDEBAR_DECISION_RULES.strip()}</div>
        """,
        unsafe_allow_html=True,
    )

    return model_input


def render_verdict(verdict: str, reason: str):
    """Renders the large colour-coded verdict card using unified light palette."""
    theme = VERDICT_THEMES.get(
        verdict,
        {
            "text": "#12203A",
            "bg": "#F8FAFC",
            "border": "#CBD5E1",
            "icon": "•",
            "tag": "VERDICT",
        },
    )

    html = f"""
    <div style="
        background: {theme['bg']};
        border: 2px solid {theme['border']};
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 16px rgba(18, 32, 58, 0.06);
        margin-bottom: 18px;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 0.74rem; letter-spacing: 0.08em; font-weight: 700; color: {theme['text']}; text-transform: uppercase;">
                {theme['tag']}
            </span>
            <span style="font-size: 1.35rem; font-weight: 800; color: {theme['text']};">{theme['icon']}</span>
        </div>
        <div style="font-size: 1.85rem; font-weight: 800; color: {theme['text']}; letter-spacing: -0.01em; margin-bottom: 8px;">
            {verdict}
        </div>
        <div style="font-size: 0.92rem; line-height: 1.5; color: #12203A; border-top: 1px solid rgba(18, 32, 58, 0.12); padding-top: 12px; margin-top: 10px;">
            <strong>Reason:</strong> {reason}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_gauges(package_conf: float, damage_conf: float):
    """Renders telemetry gauges in light-card format."""
    pkg_pct = int(round(min(max(package_conf, 0.0), 1.0) * 100))
    dmg_pct = int(round(min(max(damage_conf, 0.0), 1.0) * 100))

    pkg_color = (
        "#1E9E5A"
        if package_conf >= PACKAGE_LOW_CONFIDENCE_THRESHOLD
        else ("#C98512" if package_conf >= PACKAGE_NOT_FOUND_THRESHOLD else "#D64545")
    )
    dmg_color = "#D64545" if damage_conf >= DAMAGE_CONF_THRESHOLD else "#94A3B8"

    html = f"""
    <div class="qc-card" style="margin-bottom: 18px;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #C98512; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 14px;">
            Confidence Telemetry
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                    <span style="font-size: 0.86rem; color: #5A6E85; font-weight: 500;">Package Confidence</span>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #12203A;">{package_conf:.2f} <span style="font-size: 0.76rem; color: #8C9BAE;">({pkg_pct}%)</span></span>
                </div>
                <div style="background: #E2E8F0; border-radius: 9999px; height: 10px; overflow: hidden;">
                    <div style="width: {pkg_pct}%; background: {pkg_color}; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div style="font-size: 0.72rem; color: #8C9BAE; margin-top: 5px;">Cutoffs: ≥{PACKAGE_LOW_CONFIDENCE_THRESHOLD:.2f} clear | &lt;{PACKAGE_NOT_FOUND_THRESHOLD:.2f} not found</div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                    <span style="font-size: 0.86rem; color: #5A6E85; font-weight: 500;">Damage Confidence</span>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #12203A;">{damage_conf:.2f} <span style="font-size: 0.76rem; color: #8C9BAE;">({dmg_pct}%)</span></span>
                </div>
                <div style="background: #E2E8F0; border-radius: 9999px; height: 10px; overflow: hidden;">
                    <div style="width: {dmg_pct}%; background: {dmg_color}; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div style="font-size: 0.72rem; color: #8C9BAE; margin-top: 5px;">Threshold: ≥{DAMAGE_CONF_THRESHOLD:.2f} triggers repackage</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_barcode(barcode_status: str):
    """Renders the 3-state barcode status card (never simple Yes/No)."""
    if barcode_status == "readable":
        theme = {
            "border": "#1E9E5A",
            "bg": "#E7F6EE",
            "title_color": "#1E9E5A",
            "badge_bg": "#1E9E5A",
            "icon": "✓",
            "title": "Barcode readable",
            "desc": "Valid barcode decoded and verified intact.",
        }
    elif barcode_status == "detected_unreadable":
        theme = {
            "border": "#C98512",
            "bg": "#FFF4DD",
            "title_color": "#C98512",
            "badge_bg": "#C98512",
            "icon": "⚠",
            "title": "Barcode detected but unreadable - verify manually",
            "desc": "Label region identified, but could not be decoded (creased, torn, or blurry).",
        }
    else:  # "not_found"
        theme = {
            "border": "#CBD5E1",
            "bg": "#F1F5F9",
            "title_color": "#64748B",
            "badge_bg": "#64748B",
            "icon": "—",
            "title": "No barcode detected",
            "desc": "No barcode or shipping label found on the visible surface.",
        }

    html = f"""
    <div style="
        background: {theme['bg']};
        border: 1px solid {theme['border']};
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(18, 32, 58, 0.04);
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 24px;
                    height: 24px;
                    border-radius: 50%;
                    background: {theme['badge_bg']};
                    color: #FFFFFF;
                    font-size: 0.8rem;
                    font-weight: 800;
                ">{theme['icon']}</span>
                <span style="font-size: 1.02rem; font-weight: 700; color: {theme['title_color']};">
                    {theme['title']}
                </span>
            </div>
            <span style="font-size: 0.72rem; color: #5A6E85; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                Label Check
            </span>
        </div>
        <div style="font-size: 0.84rem; color: #5A6E85; margin-left: 34px;">
            {theme['desc']}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ==============================================================================
# TAB 1: INSPECT PAGE
# ==============================================================================
def page_inspect(model_path: str):
    """Inspect page: upload single package image and display annotated detection & telemetry."""
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown(
            """
            <div style="font-size: 1.05rem; font-weight: 700; color: #12203A; margin-bottom: 8px;">
                Upload Package Photo
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload package image",
            type=["jpg", "jpeg", "png"],
            help="Select a JPG, JPEG, or PNG package photo",
            label_visibility="collapsed",
            key="single_image_uploader",
        )

        if uploaded_file is not None:
            # Check if this file was already processed and stored
            current_stored = st.session_state.get("inspect_result")
            is_new_upload = (
                current_stored is None
                or current_stored.get("filename") != uploaded_file.name
                or current_stored.get("file_size") != uploaded_file.size
            )

            if is_new_upload:
                ext = os.path.splitext(uploaded_file.name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    with st.spinner("Inspecting package..."):
                        load_cached_model(model_path)
                        result = assess_package(tmp_path, model_path=model_path)

                        annotated_bgr = result["annotated_result"].plot()
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                        st.session_state["inspect_result"] = {
                            "filename": uploaded_file.name,
                            "file_size": uploaded_file.size,
                            "annotated_rgb": annotated_rgb,
                            "verdict": result["verdict"],
                            "reason": result["reason"],
                            "package_confidence": result["package_confidence"],
                            "damage_confidence": result["damage_confidence"],
                            "barcode_status": result["barcode_status"],
                        }
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    # Check whether we have a result in session state to display
    stored = st.session_state.get("inspect_result")

    if stored is not None:
        with col_left:
            st.markdown(
                """
                <div style="margin-top: 18px; font-size: 1.05rem; font-weight: 700; color: #12203A; margin-bottom: 8px;">
                    Annotated Detection Result
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(
                stored["annotated_rgb"],
                caption=f"File: {stored['filename']} — Bounding Boxes Drawn",
                use_container_width=True,
            )

        with col_right:
            st.markdown(
                """
                <div style="font-size: 1.05rem; font-weight: 700; color: #12203A; margin-bottom: 8px;">
                    Inspection Assessment
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_verdict(stored["verdict"], stored["reason"])
            render_gauges(stored["package_confidence"], stored["damage_confidence"])
            render_barcode(stored["barcode_status"])
    else:
        with col_left:
            st.markdown(
                """
                <div class="empty-state-box">
                    <div style="font-size: 2.2rem; margin-bottom: 8px;">📦</div>
                    <div style="font-size: 1.02rem; font-weight: 600; color: #12203A; margin-bottom: 4px;">
                        Awaiting Package Photo
                    </div>
                    <div style="font-size: 0.86rem; color: #5A6E85;">
                        Drop a package photo above or browse (JPG, JPEG, PNG).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_right:
            st.markdown(
                """
                <div style="font-size: 1.05rem; font-weight: 700; color: #12203A; margin-bottom: 8px;">
                    Inspection Assessment
                </div>
                <div class="empty-state-box">
                    <div style="font-size: 2.2rem; margin-bottom: 8px;">📊</div>
                    <div style="font-size: 1.02rem; font-weight: 600; color: #12203A; margin-bottom: 4px;">
                        Telemetry Inactive
                    </div>
                    <div style="font-size: 0.86rem; color: #5A6E85;">
                        Verdict, confidence gauges, and label check will appear once an image is uploaded.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 2: BATCH PAGE
# ==============================================================================
def page_batch(model_path: str):
    """Batch page: upload multiple images, run assessment, show summary counts, table, grid, and downloads."""
    st.markdown(
        """
        <div style="font-size: 1.15rem; font-weight: 700; color: #12203A; margin-bottom: 4px;">
            Batch Package Processing
        </div>
        <div style="font-size: 0.9rem; color: #5A6E85; margin-bottom: 16px;">
            Upload multiple package images to execute batch quality assessment and download reports.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload multiple package images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple files for batch inspection",
        key="batch_files_uploader",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) selected.**")
        start_btn = st.button("🚀 Process Batch Images", type="primary")

        if start_btn:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            processed_data = []

            load_cached_model(model_path)

            for idx, uf in enumerate(uploaded_files):
                status_text.markdown(f"⏳ Inspecting image **{idx + 1}/{len(uploaded_files)}**: `{uf.name}`...")
                ext = os.path.splitext(uf.name)[1] or ".jpg"

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uf.getvalue())
                    tmp_path = tmp_file.name

                try:
                    res = assess_package(tmp_path, model_path=model_path)
                    annotated_bgr = res["annotated_result"].plot()
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                processed_data.append({
                    "filename": uf.name,
                    "annotated_rgb": annotated_rgb,
                    "verdict": res["verdict"],
                    "package_confidence": res["package_confidence"],
                    "damage_confidence": res["damage_confidence"],
                    "barcode_status": res["barcode_status"],
                    "reason": res["reason"],
                })

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.markdown("✅ **Batch processing completed!**")
            st.session_state["batch_results"] = processed_data

    # Display batch results from session state if available
    batch_results = st.session_state.get("batch_results")

    if not batch_results:
        if not uploaded_files:
            st.markdown(
                """
                <div class="empty-state-box">
                    <div style="font-size: 2.2rem; margin-bottom: 8px;">📁</div>
                    <div style="font-size: 1.02rem; font-weight: 600; color: #12203A; margin-bottom: 4px;">
                        No Batch Images Loaded
                    </div>
                    <div style="font-size: 0.86rem; color: #5A6E85;">
                        Select or drag multiple package photos above to begin batch inspection.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # 1. Summary Count Tiles
    cnt_ready = sum(1 for r in batch_results if r["verdict"] == "Ready for Shipment")
    cnt_repack = sum(1 for r in batch_results if r["verdict"] == "Repackage Required")
    cnt_manual = sum(1 for r in batch_results if r["verdict"] == "Manual Inspection")

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin: 24px 0 20px 0;">
            <div style="
                background: #E7F6EE;
                border: 1px solid #1E9E5A;
                border-radius: 14px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(30, 158, 90, 0.08);
            ">
                <div style="font-size: 0.78rem; font-weight: 700; color: #1E9E5A; text-transform: uppercase;">Ready for Shipment</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #1E9E5A; margin-top: 4px;">{cnt_ready}</div>
            </div>
            <div style="
                background: #FDECEC;
                border: 1px solid #D64545;
                border-radius: 14px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(214, 69, 69, 0.08);
            ">
                <div style="font-size: 0.78rem; font-weight: 700; color: #D64545; text-transform: uppercase;">Repackage Required</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #D64545; margin-top: 4px;">{cnt_repack}</div>
            </div>
            <div style="
                background: #FFF4DD;
                border: 1px solid #C98512;
                border-radius: 14px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(201, 133, 18, 0.08);
            ">
                <div style="font-size: 0.78rem; font-weight: 700; color: #C98512; text-transform: uppercase;">Manual Inspection</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #C98512; margin-top: 4px;">{cnt_manual}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Results Table
    st.markdown(
        """
        <div style="font-size: 1.05rem; font-weight: 700; color: #12203A; margin-bottom: 8px;">
            Results Table
        </div>
        """,
        unsafe_allow_html=True,
    )

    table_rows = [
        {
            "Filename": r["filename"],
            "Verdict": r["verdict"],
            "Package Conf": f"{r['package_confidence']:.2f}",
            "Damage Conf": f"{r['damage_confidence']:.2f}",
            "Barcode Status": r["barcode_status"],
        }
        for r in batch_results
    ]
    st.dataframe(table_rows, use_container_width=True)

    # 3. Action Downloads (CSV + Annotated Images ZIP)
    col_dl1, col_dl2 = st.columns(2, gap="medium")

    # CSV Download
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["Filename", "Verdict", "Package Confidence", "Damage Confidence", "Barcode Status", "Reason"])
    for r in batch_results:
        writer.writerow([
            r["filename"],
            r["verdict"],
            r["package_confidence"],
            r["damage_confidence"],
            r["barcode_status"],
            r["reason"],
        ])
    csv_data = csv_buf.getvalue()

    with col_dl1:
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name="batch_inspection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ZIP Download of Annotated Images
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in batch_results:
            bgr = cv2.cvtColor(r["annotated_rgb"], cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                base_name = os.path.splitext(r["filename"])[0]
                zf.writestr(f"{base_name}_annotated.jpg", enc.tobytes())

    with col_dl2:
        st.download_button(
            label="📦 Download Annotated Images (ZIP)",
            data=zip_buf.getvalue(),
            file_name="annotated_packages.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.markdown("---")

    # 4. Filter & Responsive 3-Column Grid of Annotated Results
    st.markdown(
        """
        <div style="font-size: 1.1rem; font-weight: 700; color: #12203A; margin-bottom: 4px;">
            Annotated Results
        </div>
        """,
        unsafe_allow_html=True,
    )

    verdict_options = ["Ready for Shipment", "Repackage Required", "Manual Inspection"]
    selected_verdicts = st.multiselect(
        "Filter by verdict",
        options=verdict_options,
        default=verdict_options,
        key="verdict_filter_multiselect",
    )

    filtered = [r for r in batch_results if r["verdict"] in selected_verdicts]

    if not filtered:
        st.info("No images match the selected verdict filter.")
        return

    # Render in 3-column grid
    cols = st.columns(3, gap="medium")
    for idx, item in enumerate(filtered):
        col = cols[idx % 3]
        v_theme = VERDICT_THEMES.get(item["verdict"], {"text": "#12203A", "bg": "#F8FAFC", "border": "#CBD5E1"})

        # Format barcode text
        bc_status = item["barcode_status"]
        if bc_status == "readable":
            bc_display = "readable"
            bc_color = "#1E9E5A"
        elif bc_status == "detected_unreadable":
            bc_display = "detected but unreadable"
            bc_color = "#C98512"
        else:
            bc_display = "not detected"
            bc_color = "#64748B"

        with col:
            st.markdown(
                f"""
                <div class="qc-card" style="padding: 14px; margin-bottom: 16px;">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #12203A; margin-bottom: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{item['filename']}">
                        📄 {item['filename']}
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.image(item["annotated_rgb"], use_container_width=True)

            st.markdown(
                f"""
                    <div style="margin-top: 8px; margin-bottom: 8px;">
                        <span style="
                            background: {v_theme['bg']};
                            color: {v_theme['text']};
                            border: 1px solid {v_theme['border']};
                            padding: 3px 10px;
                            border-radius: 9999px;
                            font-size: 0.78rem;
                            font-weight: 700;
                        ">{item['verdict']}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #5A6E85; line-height: 1.45;">
                        <div><strong>Package Conf:</strong> {item['package_confidence']:.2f}</div>
                        <div><strong>Damage Conf:</strong> {item['damage_confidence']:.2f}</div>
                        <div><strong>Barcode:</strong> <span style="color: {bc_color}; font-weight: 600;">{bc_display}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 3: ABOUT PAGE
# ==============================================================================
def page_about():
    """About page: short and factual documentation across three cards (no marketing terms)."""
    st.markdown(
        """
        <div style="font-size: 1.2rem; font-weight: 800; color: #12203A; margin-bottom: 6px;">
            About This System
        </div>
        <div style="font-size: 0.92rem; color: #5A6E85; margin-bottom: 20px;">
            System overview, decision rules, and model specifications.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Card 1: What it does
    st.markdown(
        f"""
        <div class="qc-card">
            <div style="font-size: 0.85rem; font-weight: 700; color: #C98512; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px;">
                {ABOUT_CARD_1_TITLE}
            </div>
            <div style="font-size: 0.95rem; color: #12203A; line-height: 1.6;">
                {ABOUT_CARD_1_TEXT}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Card 2: How the verdict is decided
    rules_html = "".join(
        f"<li style='margin-bottom: 6px;'><strong>{cond}</strong> &rarr; <span style='font-weight: 600;'>{res}</span></li>"
        for cond, res in ABOUT_CARD_2_RULES
    )
    st.markdown(
        f"""
        <div class="qc-card">
            <div style="font-size: 0.85rem; font-weight: 700; color: #C98512; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px;">
                {ABOUT_CARD_2_TITLE}
            </div>
            <ul style="font-size: 0.93rem; color: #12203A; line-height: 1.6; padding-left: 20px; margin-bottom: 12px;">
                {rules_html}
            </ul>
            <div style="font-size: 0.88rem; color: #5A6E85; font-style: italic; border-top: 1px solid #E2E8F0; padding-top: 8px;">
                {ABOUT_CARD_2_NOTE}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Card 3: Model
    st.markdown(
        f"""
        <div class="qc-card">
            <div style="font-size: 0.85rem; font-weight: 700; color: #C98512; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px;">
                {ABOUT_CARD_3_TITLE}
            </div>
            <div style="font-size: 0.95rem; color: #12203A; line-height: 1.6;">
                {ABOUT_CARD_3_TEXT}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN APPLICATION ROUTINE
# ==============================================================================
def main():
    st.set_page_config(
        page_title="Package Inspection & Shipment Readiness",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Top Hero Header
    render_header()

    # Sidebar Controls & Rules
    model_path = render_sidebar()

    # Custom Horizontal Navigation (styled as tabs)
    selected_page = st.radio(
        "Navigation",
        options=["Inspect", "Batch", "About"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation_tab",
    )

    # Inject Light Theme CSS & Page-Specific Background (bg1 / bg2 / bg3)
    css(selected_page)

    # Route to selected page
    if selected_page == "Inspect":
        page_inspect(model_path)
    elif selected_page == "Batch":
        page_batch(model_path)
    elif selected_page == "About":
        page_about()


if __name__ == "__main__":
    main()
