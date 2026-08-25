"""Dark, production-grade SaaS visual theme -- CSS injection, applied on request.

Native theming (``.streamlit/config.toml``) is the default way to brand a Streamlit
app, but this module's specific look (segmented pill tabs, bordered metric cards,
the ``saas-card`` finding blocks used in ``tabs_briefing.py``) needs CSS beyond what
``config.toml`` alone can express -- injected here, explicitly, rather than as the
app's default styling approach.
"""

from __future__ import annotations

import streamlit as st

_PRODUCTION_THEME_CSS = """
<style>
/* 1. Global Canvas & Background */
.stApp {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* 2. Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}

/* 3. Segmented Navigation Tabs */
[data-testid="stTabs"] {
    gap: 8px;
    margin-bottom: 24px;
}
[data-testid="stTabs"] button[role="tab"] {
    background-color: #21262d !important;
    border-radius: 8px !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
    padding: 8px 18px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease-in-out;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #f0f6fc !important;
    background-color: #30363d !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background-color: #238636 !important;
    border-color: #2ea043 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(46, 160, 67, 0.25) !important;
}

/* 4. Metric Cards */
[data-testid="stMetric"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #58a6ff !important;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    color: #f0f6fc !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
}

/* 5. Custom Card Containers */
.saas-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.saas-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f0f6fc;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.saas-card-body {
    font-size: 0.95rem;
    color: #c9d1d9;
    line-height: 1.5;
}

/* 6. Status Pills & Multi-select Chips (Analysis Period) */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: #21262d !important;
    border: 1px solid #388bfd33 !important;
    border-radius: 6px !important;
    color: #58a6ff !important;
}

/* 7. File Uploader Box */
[data-testid="stFileUploader"] section {
    background-color: #0d1117 !important;
    border: 1.5px dashed #30363d !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #58a6ff !important;
}

/* 8. Modern Buttons */
.stButton > button {
    background-color: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 6px 16px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease-in-out !important;
}
.stButton > button:hover {
    background-color: #30363d !important;
    color: #f0f6fc !important;
    border-color: #8b949e !important;
}
</style>
"""


def apply_production_theme() -> None:
    st.markdown(_PRODUCTION_THEME_CSS, unsafe_allow_html=True)
