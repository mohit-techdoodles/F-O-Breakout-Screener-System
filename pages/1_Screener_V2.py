"""
Screener Architecture V2.0: MTF Breakout & Money Zone Screener
Multi-page Streamlit Interface
Target Execution: Indian Equities (NSE) & F&O
"""

import datetime
import importlib
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import fo_stocks
import fibonacci_screener_v2
import mtf_breakout
import screener_v2_engine
import swing_optimization_filters

importlib.reload(fo_stocks)
importlib.reload(fibonacci_screener_v2)
importlib.reload(mtf_breakout)
importlib.reload(screener_v2_engine)
importlib.reload(swing_optimization_filters)

from fo_stocks import TOP_30_LIQUID, NIFTY_50, BANK_NIFTY, get_fo_universe
from screener_v2_engine import (
    fetch_universe_data_v2,
    fetch_daily_1y_data_v2,
    fetch_intraday_data_v2,
    apply_screener_v2_filters,
    calculate_mtf_v2,
    ensure_decimal_columns
)
from mtf_breakout import PINE_SCRIPT_V5_CODE
from pwa_launcher import inject_pwa_headers, render_app_launcher_sidebar

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Screener Architecture V2.0 | MTF & Money Zone",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject PWA Progressive Web App headers & Manifest
inject_pwa_headers()

# ----------------- Custom Styling -----------------
st.markdown("""
<style>
    .reportview-container {
        background-color: #0E1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1A1F2C 0%, #11141E 100%);
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-lbl {
        font-size: 0.80rem;
        color: #9E9E9E;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .metric-val {
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 4px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    }
    .badge-bull {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00E676;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid rgba(0, 230, 118, 0.35);
    }
    .badge-bear {
        background-color: rgba(255, 82, 82, 0.15);
        color: #FF5252;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid rgba(255, 82, 82, 0.35);
    }
    .badge-squeeze {
        background-color: rgba(171, 71, 188, 0.15);
        color: #CE93D8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid rgba(171, 71, 188, 0.35);
    }
    .badge-cpr {
        background-color: rgba(255, 179, 0, 0.15);
        color: #FFB300;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid rgba(255, 179, 0, 0.35);
    }
    .badge-swing {
        background-color: rgba(255, 179, 0, 0.15);
        color: #FFB300;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid rgba(255, 179, 0, 0.35);
    }
    .badge-coil {
        background-color: rgba(0, 229, 255, 0.15);
        color: #00E5FF;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid rgba(0, 229, 255, 0.35);
    }
    /* ========================================================================= */
    /* ENHANCED TABLE SCROLLBARS: BOTH HORIZONTAL & VERTICAL VISIBLE ON HOVER   */
    /* ========================================================================= */
    .stDataFrame,
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"],
    .stDataFrameGlideDataEditor {
        border-radius: 8px;
        overflow: visible !important;
    }

    /* WebKit / Blink (Chrome, Edge, Safari) */
    ::-webkit-scrollbar {
        -webkit-appearance: none !important;
        width: 11px !important;
        height: 11px !important;
        display: block !important;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A !important;
        border-radius: 5px !important;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(59, 130, 246, 0.5) !important;
        border-radius: 5px !important;
        border: 1px solid rgba(15, 23, 42, 0.8) !important;
        min-width: 25px !important;
        min-height: 25px !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #60A5FA !important;
        box-shadow: 0 0 10px rgba(96, 165, 250, 0.9) !important;
    }
    ::-webkit-scrollbar-corner {
        background: #0F172A !important;
    }

    /* High-contrast vibrant scrollbars inside all Streamlit tables */
    [data-testid="stDataFrame"] ::-webkit-scrollbar,
    .stDataFrame ::-webkit-scrollbar,
    .dvn-scroller::-webkit-scrollbar {
        -webkit-appearance: none !important;
        width: 11px !important;
        height: 11px !important;
        display: block !important;
    }
    [data-testid="stDataFrame"] ::-webkit-scrollbar-track,
    .stDataFrame ::-webkit-scrollbar-track,
    .dvn-scroller::-webkit-scrollbar-track {
        background: #0B1120 !important;
        border-radius: 5px !important;
    }
    [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb,
    .stDataFrame ::-webkit-scrollbar-thumb,
    .dvn-scroller::-webkit-scrollbar-thumb {
        background: #3B82F6 !important;
        border-radius: 5px !important;
        border: 1.5px solid #0F172A !important;
    }
    [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb:hover,
    .stDataFrame ::-webkit-scrollbar-thumb:hover,
    .dvn-scroller::-webkit-scrollbar-thumb:hover {
        background: #93C5FD !important;
        box-shadow: 0 0 12px rgba(147, 197, 253, 0.95) !important;
    }

    /* Firefox standard */
    * {
        scrollbar-color: #3B82F6 #0F172A !important;
        scrollbar-width: thin !important;
    }
    .nav-banner {
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Navigation & Header -----------------
with st.container():
    col_nav1, col_nav2 = st.columns([3, 1])
    with col_nav1:
        st.markdown("""
        ### 🎯 Screener Architecture V2.0: MTF Breakout & Money Zone Screener
        <span style="color: #94A3B8; font-size: 0.95rem;">
        Indian Equities (NSE) & F&O &bull; 1D / 1H Initial Balance / 15m VWAP &bull; Forward-Looking Money Zone Matrix
        </span>
        """, unsafe_allow_html=True)
    with col_nav2:
        try:
            st.page_link("app.py", label="🏛️ Switch to Classic Screener V1", icon="🔙", use_container_width=True)
        except Exception:
            st.markdown("[🏛️ **Switch to Classic Screener V1**](/)")

st.markdown("---")

# ----------------- Sidebar Controls -----------------
st.sidebar.title("🎛️ Screener V2.0 Settings")

# Navigation Link in Sidebar
try:
    st.sidebar.page_link("app.py", label="🏛️ Go to Classic Screener V1", icon="🔙")
except Exception:
    st.sidebar.markdown("[🏛️ **Go to Classic Screener V1**](/)")

# App Launcher & Mobile Install Helper
render_app_launcher_sidebar()
st.sidebar.markdown("---")

# Universe Selection
st.sidebar.subheader("📋 Stock Universe")
universe_choice = st.sidebar.selectbox(
    "Select Universe Preset",
    ["Top 30 Liquid F&O", "Nifty 50", "Bank Nifty", "Full F&O Universe (180+)", "Custom Selection"],
    index=0
)

all_fo = get_fo_universe()
if universe_choice == "Top 30 Liquid F&O":
    selected_symbols = TOP_30_LIQUID
elif universe_choice == "Nifty 50":
    selected_symbols = NIFTY_50
elif universe_choice == "Bank Nifty":
    selected_symbols = BANK_NIFTY
elif universe_choice == "Full F&O Universe (180+)":
    selected_symbols = all_fo
else:
    selected_symbols = st.sidebar.multiselect(
        "Choose Stocks",
        options=all_fo,
        default=TOP_30_LIQUID[:10]
    )

timeframe = st.sidebar.selectbox("Intraday Timeframe", ["5m", "15m"], index=0, help="5m allows precise 1H break-in tag validation.")

st.sidebar.markdown("---")
st.sidebar.subheader("1. Core MTF Strategy Validations")

# 1.0 Global Flag: bypass_52w_proximity
bypass_52w = st.sidebar.toggle(
    "🔓 bypass_52w_proximity",
    value=True,
    help="Section 1: When enabled, ignores the 52-Week High / Low proximity restriction, allowing valid breakouts across the entire universe."
)

# 1.1 Initial Hour (1H) Logic Execution
h1_filter = st.sidebar.selectbox(
    "1.1 Initial Hour (1H) Logic Execution",
    [
        "All",
        # Option A
        "Option A: 1H Close > Prev 1H (Breakout)",
        "Option A: 1H Close < Prev 1H (Breakdown)",
        "Option A: 1H Close == Prev 1H (Break-in/Equal)",
        # Option B
        "Option B: Live Price > 1H Close (Breakout)",
        "Option B: Live Price < 1H Close (Breakdown)",
        "Option B: Break-in (Failed 1H Close Breakout/down)",
        # Option C
        "Option C: Range Breakout > 1H High",
        "Option C: Range Breakdown < 1H Low",
        "Option C: Range Break-in (±0.2% Tag Pullback)"
    ],
    index=0,
    help="Option A: 10:15 close vs prev 10:15 close. Option B: Live LTP vs 10:15 close. Option C: Live LTP vs 10:15 High/Low range with ±0.2% tag & re-entry."
)

mtf_confluence = st.sidebar.selectbox(
    "MTF Confluence Filter",
    ["All", "Bullish Only", "Bearish Only", "Any Confluence (Bullish or Bearish)"],
    index=0,
    help="Bullish: 1H Close > PDH & LTP > VWAP. Bearish: 1H Close < PDL & LTP < VWAP."
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Advanced Data Validations (Volume & F&O)")

require_rvol = st.sidebar.checkbox(
    "Require RVOL ≥ 1.5",
    value=False,
    help="2.1: Relative Volume = Current_Volume / SMA(Volume_30_days). Valid breakouts require RVOL >= 1.5."
)

require_delivery = st.sidebar.checkbox(
    "Require Delivery % > 50%",
    value=False,
    help="2.1: Flags stocks with high institutional delivery accumulation."
)

oi_filter = st.sidebar.selectbox(
    "Derivatives OI Buildup",
    ["All", "Long Buildup", "Short Covering", "Short Buildup", "Long Unwinding"],
    index=0,
    help="2.2: Long Buildup (LTP↑, OI↑), Short Covering (LTP↑, OI↓), Short Buildup (LTP↓, OI↑), Long Unwinding (LTP↓, OI↓)."
)

pcr_filter = st.sidebar.selectbox(
    "Put-Call Ratio (PCR) Filter",
    ["All", "Squeeze (PCR < 0.6)", "Bullish Support (PCR > 1.2)"],
    index=0,
    help="2.2: PCR < 0.6 indicates oversold potential short-covering squeeze; PCR > 1.2 indicates strong bullish support."
)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Weekly Pivot & Money Zone Matrix")

money_zone_filter = st.sidebar.selectbox(
    "Forward Watchlist Setup (Monday Action)",
    [
        "All Setups",
        "Any Forward Watchlist Setup",
        "Setup 1: Volatility Squeeze (Inside Value)",
        "Setup 2: Narrow CPR (< 0.5%)",
        "Setup 3: Trend Continuation (Higher CPR & POC)",
        "Setup 4: Camarilla Alignment (H4 ≈ VAH ±0.2%)"
    ],
    index=0,
    help="Section 3: Forward-looking Friday-for-Monday setup matrix."
)

st.sidebar.markdown("---")
st.sidebar.subheader("4. Advanced Fibonacci & Confluence")

fib_confluence_filter = st.sidebar.selectbox(
    "Fibonacci Confluence Filter",
    [
        "All",
        "Any Advanced Fibonacci Confluence Setup",
        "⭐ Golden Pocket (0.618) Confluence (Option C Break-in)",
        "🟡 At Golden Pocket (0.618 ±0.5%)",
        "⚡ Pullback in Golden Zone (0.50 - 0.618)",
        "1. MTF Fibonacci Cluster Active",
        "2. Fibonacci Extension Breakout (1.272 / 1.618)",
        "3. Volume Profile + Golden Pocket Confluence",
        "4. Fibonacci Rejection (Wick) Scanner (>= 60%)",
        "5. Golden Pivot Confluence (Fib + S1/S2/R1/R2 + Value Area)",
        "6. Camarilla Reversal Trap (L3/H3 Trap)",
        "7. Institutional Premium/Discount Oscillator (Extreme)",
        "8. Macro Anchor (Monthly Value + Weekly Fib)",
        "9. Weekly CPR Squeeze & Fib Expansion",
        "10. Inside Week + Golden Pocket Compression"
    ],
    index=0,
    help="Select from Classic Golden Pocket setups (Option C Break-in, At 0.618, Golden Zone) or Advanced Points 1 - 10."
)

st.sidebar.markdown("---")
st.sidebar.subheader("5. Swing Trade Optimizations (High-Conviction)")

benchmark_choice = st.sidebar.selectbox(
    "Benchmark Index (Relative Strength)",
    [
        "Nifty Total Market (Broad 750 / 500)",
        "Nifty 50 (^NSEI)",
        "Bank Nifty (^NSEBANK)"
    ],
    index=0,
    help="Section 2: Benchmark for 14-day Relative Strength Alpha (Ret_Stock - Ret_Index). Default is Nifty Total Market (Broad 750 / 500 Index)."
)

bench_map = {
    "Nifty Total Market (Broad 750 / 500)": "NIFTY_TOTAL_MARKET",
    "Nifty 50 (^NSEI)": "NIFTY50",
    "Bank Nifty (^NSEBANK)": "BANKNIFTY"
}
bench_key = bench_map[benchmark_choice]

require_coiled_spring = st.sidebar.checkbox(
    "Require Coiled Spring (NR7 or BB Squeeze)",
    value=False,
    help="Section 3: Filters for extreme volatility compression (NR7 Range Contraction or BB Width < 120-Day Average) directly at key structural levels."
)

rs_option = st.sidebar.selectbox(
    "Relative Strength (RS vs Benchmark)",
    ["All", "RS Outperforming Index (Alpha > 0%)", "RS Underperforming Index (Alpha < 0%)"],
    index=0,
    help="Section 2: Require Outperforming for long swing setups, Underperforming for short swing setups."
)

oi_2d_filter = st.sidebar.selectbox(
    "2-Day OI Buildup (Derivatives Confluence)",
    ["All", "Long Buildup (2D >= 5%)", "Short Buildup (2D >= 5%)"],
    index=0,
    help="Section 1: 2-day rolling window OI change with minimum 5% threshold."
)

require_high_conviction = st.sidebar.checkbox(
    "⭐ High-Conviction Swing Candidates Only",
    value=False,
    help="Master Confluence: Requires Coiled Spring + (Fibonacci or Money Zone level) + Directional RS Outperformance + Derivatives OI Buildup."
)

range_pos_choice = st.sidebar.selectbox(
    "Intraday Range Position (Close >= 0.70 / 0.75)",
    [
        "All",
        "Strong Close (>= 0.70 Range)",
        "Elite Close (>= 0.75 Range)",
        "Fut Strong Close (>= 0.70 Range)",
        "Fut Elite Close (>= 0.75 Range)",
        "Weak Close (<= 0.30 Range)"
    ],
    index=0,
    help="Filters stocks closing at the upper extremes of the day's high-low range (Close - Low) / (High - Low) >= 0.70 or 0.75 for both Spot and Futures with NSE tick precision (.70, .75)."
)

# Two-tier optimized caching:
# 1. Daily 1-year data cached for 2 hours (historical bars don't change intraday)
# 2. Intraday 5-day data cached for 120 seconds
# 3. Benchmark daily data cached for 2 hours
@st.cache_data(ttl=7200, show_spinner=False)
def get_daily_1y_data_cached(syms):
    return fetch_daily_1y_data_v2(syms)

@st.cache_data(ttl=120, show_spinner=False)
def get_intraday_data_cached(syms, tf):
    return fetch_intraday_data_v2(syms, timeframe=tf)

@st.cache_data(ttl=7200, show_spinner=False)
def get_benchmark_data_cached(b_key):
    return swing_optimization_filters.fetch_benchmark_daily_data(b_key)

@st.cache_data(ttl=120, show_spinner=False)
def get_v2_data(syms, tf, b_key):
    d_1y = get_daily_1y_data_cached(syms)
    d_intra = get_intraday_data_cached(syms, tf)
    d_bench = get_benchmark_data_cached(b_key)
    return fetch_universe_data_v2(
        syms,
        timeframe=tf,
        df_1y=d_1y,
        df_intraday=d_intra,
        df_benchmark=d_bench,
        benchmark_name=b_key
    )

# Refresh Buttons
c_ref1, c_ref2 = st.sidebar.columns([3, 1])
with c_ref1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        get_intraday_data_cached.clear()
        get_v2_data.clear()
with c_ref2:
    if st.button("⚡ Reset", help="Hard reset all caches including 1Y Daily data"):
        st.cache_data.clear()

# ----------------- Data Ingestion -----------------
with st.spinner(f"Scanning market data for {len(selected_symbols)} F&O symbols (Architecture V2.0)..."):
    raw_df, base_meta = get_v2_data(selected_symbols, timeframe, bench_key)
    if "intraday_cache" in base_meta:
        st.session_state["intraday_cache_v2"] = base_meta["intraday_cache"]
    if "daily_cache" in base_meta:
        st.session_state["daily_cache_v2"] = base_meta["daily_cache"]

if raw_df.empty:
    st.warning("⚠️ No data was returned. Please verify network connectivity or market symbols.")
    st.stop()

# Vectorized Filter Execution
all_df, bull_df, bear_df, stats = apply_screener_v2_filters(
    raw_df,
    bypass_52w_proximity=bypass_52w,
    h1_option_filter=h1_filter,
    require_rvol=require_rvol,
    require_high_delivery=require_delivery,
    oi_buildup_filter=oi_filter,
    pcr_squeeze_filter=pcr_filter,
    money_zone_setup_filter=money_zone_filter,
    mtf_confluence_filter=mtf_confluence,
    fib_confluence_filter=fib_confluence_filter,
    rs_filter=rs_option,
    coiled_spring_filter="Coiled Spring (NR7 or BB Squeeze)" if require_coiled_spring else "All",
    oi_2d_filter=oi_2d_filter,
    swing_high_conviction_filter="High-Conviction Swing Only" if require_high_conviction else "All",
    range_pos_filter=range_pos_choice
)
all_df = ensure_decimal_columns(all_df)

# ----------------- Top Metric Cards Row -----------------
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Scanned</div>
        <div class="metric-val" style="color: #64B5F6;">{stats.get('total_scanned', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Bullish Breakouts</div>
        <div class="metric-val" style="color: #00E676;">{stats.get('bullish_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Bearish Breakdowns</div>
        <div class="metric-val" style="color: #FF5252;">{stats.get('bearish_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">MTF Confluences</div>
        <div class="metric-val" style="color: #AB47BC;">{stats.get('mtf_confluence_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">RVOL ≥ 1.5x (Volume)</div>
        <div class="metric-val" style="color: #FF9800;">{stats.get('rvol_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Monday Money Zone</div>
        <div class="metric-val" style="color: #00E5FF;">{stats.get('forward_mz_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Fibonacci & Confluence KPI Row -----------------
f1, f2, f3, f4, f5, f6 = st.columns(6)
with f1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">MTF Fib Clusters</div>
        <div class="metric-val" style="color: #FFD54F;">{stats.get('fib_clusters_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with f2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">VP + Golden Pocket</div>
        <div class="metric-val" style="color: #FFB300;">{stats.get('vol_fib_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with f3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">15m Wick Rejection</div>
        <div class="metric-val" style="color: #26A69A;">{stats.get('fib_rejection_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with f4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Golden Pivots</div>
        <div class="metric-val" style="color: #66BB6A;">{stats.get('golden_pivot_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with f5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Camarilla Traps</div>
        <div class="metric-val" style="color: #FF7043;">{stats.get('camarilla_trap_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with f6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Weekly CPR & Fib Exp</div>
        <div class="metric-val" style="color: #AB47BC;">{stats.get('weekly_cpr_fib_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Swing Optimization KPI Row -----------------
sw_col1, sw_col2, sw_col3, sw_col4 = st.columns(4)
with sw_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">⭐ High-Conviction Swings</div>
        <div class="metric-val" style="color: #FFD54F;">{stats.get('swing_high_conviction_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with sw_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">⚡ Coiled Springs (NR7/BB)</div>
        <div class="metric-val" style="color: #00E5FF;">{stats.get('coiled_spring_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with sw_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">🚀 RS Outperforming</div>
        <div class="metric-val" style="color: #00E676;">{stats.get('rs_outperforming_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with sw_col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">📈 2D Long Buildup (OI ≥ 5%)</div>
        <div class="metric-val" style="color: #AB47BC;">{stats.get('long_buildup_2d_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Main View Tabs -----------------
tab1, tab2, tab3, tab_fib, tab_swing, tab4, tab5 = st.tabs([
    "🎯 Core MTF Breakouts",
    "📊 Volume & F&O Validations",
    "🔮 Weekly Money Zone Matrix",
    "📐 Advanced Fibonacci & Confluence",
    "🚀 High-Conviction Swing (Spec)",
    "📈 Interactive Chart",
    "📜 Architecture Spec & Pine Script"
])

# =====================================================================
# TAB 1: CORE MTF BREAKOUTS
# =====================================================================
with tab1:
    st.markdown("#### 1. Core Multi-Timeframe Strategy Matrix (1D / 1H Initial Balance / 15m VWAP)")
    st.caption("Captures 1H Initial Balance at 10:15 AM IST, 15m Anchored VWAP from 09:15 AM, and evaluates Options A, B, and C.")

    mtf_cols = [
        "Symbol", "LTP", "Chg%", "PDH", "PDL", "IB_Close", "IB_High", "IB_Low",
        "VWAP", "Dist_VWAP%", "MTF_VWAP_Status", "Opt_A_Status", "Opt_B_Status", "Opt_C_Status", "MTF_Signal"
    ]
    
    view_df = all_df[mtf_cols].copy() if not all_df.empty else pd.DataFrame(columns=mtf_cols)

    # Format dataframe
    st.dataframe(
        view_df.style.format({
            "LTP": "₹{:.2f}",
            "Chg%": "{:+.2f}%",
            "PDH": "₹{:.2f}",
            "PDL": "₹{:.2f}",
            "IB_Close": "₹{:.2f}",
            "IB_High": "₹{:.2f}",
            "IB_Low": "₹{:.2f}",
            "VWAP": "₹{:.2f}",
            "Dist_VWAP%": "{:+.2f}%"
        }),
        use_container_width=True,
        height=480
    )

    csv_data = view_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export MTF Data (CSV)", csv_data, "mtf_screener_v2.csv", "text/csv")


# =====================================================================
# TAB 2: VOLUME & F&O DERIVATIVES VALIDATIONS
# =====================================================================
with tab2:
    st.markdown("#### 2. Advanced Data Validations (Volume & F&O Derivatives)")
    st.caption("Filters out false breakouts using Relative Volume (RVOL ≥ 1.5), Delivery % > 50%, Derivatives OI Buildup (1D & 2D Rolling), and Put-Call Ratio (PCR).")

    # Interactive View Focus Selector (Eliminates off-screen clipping)
    c_tab2_view, c_tab2_slider = st.columns([3, 2])
    with c_tab2_view:
        tab2_view_mode = st.radio(
            "📋 Table View Mode:",
            [
                "🎯 Derivatives & OI Focus (Direct View)",
                "🏷️ Spot vs Future Pricing Matrix (.70/.75 Decimals)",
                "📜 Complete Matrix (Volume + Delivery + OI + PCR)",
                "📈 Past Historical OI Trend (T-0 to T-5 Days)"
            ],
            index=0,
            horizontal=True,
            help="Switch between focused Derivatives view, Spot vs Future dual pricing (.70/.75 precision), Complete Matrix, or Multi-Day Past OI Tracker."
        )

    if tab2_view_mode == "🎯 Derivatives & OI Focus (Direct View)":
        st.caption("⚡ **Direct View:** Clean layout showing only Derivatives, OI Buildup, and PCR columns without needing horizontal scrolling.")
        deriv_cols = [
            "Symbol", "LTP", "Chg%", "OI_Change%", "OI_2D_Change%", "OI_Buildup", "OI_Buildup_2D", "PCR", "PCR_Condition"
        ]
        available_deriv = [c for c in deriv_cols if c in all_df.columns]
        deriv_df = all_df[available_deriv].copy() if not all_df.empty else pd.DataFrame(columns=available_deriv)

        st.dataframe(
            deriv_df.style.format({
                "LTP": "₹{:.2f}",
                "Chg%": "{:+.2f}%",
                "OI_Change%": "{:+.2f}%",
                "OI_2D_Change%": "{:+.2f}%",
                "PCR": "{:.2f}"
            }),
            use_container_width=True,
            height=450
        )

    elif tab2_view_mode == "🏷️ Spot vs Future Pricing Matrix (.70/.75 Decimals)":
        st.caption("⚡ **Spot vs Future Dual Pricing & Exact .70 / .75 Decimal Scanner:** Strictly identifies stocks whose Spot price or Future price ends in exact .70 or .75 decimal paise (e.g., ₹11049.70 or ₹11049.75), separated across Intraday (High/Low) and Market Closing with 1H Opening Breakout/Breakdown direction.")

        # Dedicated Filter Selector: Defaults to strictly showing ONLY exact .70 / .75 decimal qualifying stocks
        c_p_flt, c_p_desc = st.columns([3, 2])
        with c_p_flt:
            p_filter_mode = st.selectbox(
                "🎯 Decimal & Range Filter Mode (Strict View):",
                [
                    "🎯 Exact .70 / .75 Decimals (Spot or Future - Strict)",
                    "🔥 Closing .70 / .75 Decimals Only (Spot Close or Fut Close)",
                    "📍 Intraday .70 / .75 Decimals Only (High / Low)",
                    "🚀 Bullish 1H Breakouts (> 10:15 AM High)",
                    "🔻 Bearish 1H Breakdowns (< 10:15 AM Low)",
                    "⭐ Range Ratio >= 0.75 (Intraday or 1H Close)",
                    "🟢 Range Ratio >= 0.70 (Intraday or 1H Close)",
                    "🌐 Show All Scanned Stocks (Unfiltered)"
                ],
                index=0,
                help="By default, strictly displays ONLY stocks whose Spot or Future price ends in exact .70 or .75 decimal paise."
            )

        # Safe helper to extract boolean Series avoiding KeyError scalar indexing
        def _get_bool_s(d: pd.DataFrame, col_name: str) -> pd.Series:
            if col_name in d.columns:
                s = d[col_name]
                return s.fillna(False).astype(bool) if hasattr(s, "fillna") else pd.Series(False, index=d.index)
            if "LTP" in d.columns:
                p_col = d["Spot_Close"] if "Spot_Close" in d.columns else d["LTP"]
                return p_col.apply(lambda p: int(round(float(p) * 100)) % 100 in (70, 75))
            return pd.Series(False, index=d.index)

        # Apply strict filtering
        if p_filter_mode == "🎯 Exact .70 / .75 Decimals (Spot or Future - Strict)":
            p_mask = _get_bool_s(all_df, "Has_Exact_Decimal_70_75")
        elif p_filter_mode == "🔥 Closing .70 / .75 Decimals Only (Spot Close or Fut Close)":
            p_mask = _get_bool_s(all_df, "Has_Close_Decimal_70_75")
        elif p_filter_mode == "📍 Intraday .70 / .75 Decimals Only (High / Low)":
            p_mask = _get_bool_s(all_df, "Has_Intra_Decimal_70_75")
        elif p_filter_mode == "🚀 Bullish 1H Breakouts (> 10:15 AM High)":
            p_mask = all_df["One_Hour_Break_Status"].astype(str).str.contains("Bullish") if "One_Hour_Break_Status" in all_df.columns else pd.Series(False, index=all_df.index)
        elif p_filter_mode == "🔻 Bearish 1H Breakdowns (< 10:15 AM Low)":
            p_mask = all_df["One_Hour_Break_Status"].astype(str).str.contains("Bearish") if "One_Hour_Break_Status" in all_df.columns else pd.Series(False, index=all_df.index)
        elif p_filter_mode == "⭐ Range Ratio >= 0.75 (Intraday or 1H Close)":
            c1 = all_df["Intra_Range_Pos"] >= 0.75 if "Intra_Range_Pos" in all_df.columns else pd.Series(False, index=all_df.index)
            c2 = all_df["Close_vs_1H_Range_Pos"] >= 0.75 if "Close_vs_1H_Range_Pos" in all_df.columns else pd.Series(False, index=all_df.index)
            p_mask = c1 | c2
        elif p_filter_mode == "🟢 Range Ratio >= 0.70 (Intraday or 1H Close)":
            c1 = all_df["Intra_Range_Pos"] >= 0.70 if "Intra_Range_Pos" in all_df.columns else pd.Series(False, index=all_df.index)
            c2 = all_df["Close_vs_1H_Range_Pos"] >= 0.70 if "Close_vs_1H_Range_Pos" in all_df.columns else pd.Series(False, index=all_df.index)
            p_mask = c1 | c2
        else:
            p_mask = pd.Series(True, index=all_df.index)

        pricing_filtered = all_df[p_mask].copy() if not all_df.empty else pd.DataFrame(columns=all_df.columns)

        # Summary Metrics
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Exact .70 / .75 Matches", int(_get_bool_s(all_df, "Has_Exact_Decimal_70_75").sum()), f"of {len(all_df)} scanned")
        pm2.metric("Closing .70 / .75", int(_get_bool_s(all_df, "Has_Close_Decimal_70_75").sum()), "Spot/Fut Close")
        pm3.metric("1H Bullish Breakouts", int(all_df["One_Hour_Break_Status"].astype(str).str.contains("Bullish").sum() if "One_Hour_Break_Status" in all_df.columns else 0))
        pm4.metric("1H Bearish Breakdowns", int(all_df["One_Hour_Break_Status"].astype(str).str.contains("Bearish").sum() if "One_Hour_Break_Status" in all_df.columns else 0))

        pricing_cols = [
            "Symbol", "Spot_Close", "Spot_Decimal", "Fut_Close", "Fut_Decimal", "Decimal_Match_Tag",
            "One_Hour_Break_Status", "Chg%",
            "Spot_High", "Spot_Low", "Fut_High", "Fut_Low",
            "Intra_Range_Pos", "Intra_Range_Tag",
            "Close_vs_1H_Range_Pos",
            "Fut_Basis", "Fut_Basis%", "Days_To_Expiry"
        ]
        available_pricing = [c for c in pricing_cols if c in pricing_filtered.columns]
        pricing_df = pricing_filtered[available_pricing].copy() if not pricing_filtered.empty else pd.DataFrame(columns=available_pricing)

        st.dataframe(
            pricing_df.style.format({
                "Spot_Close": "₹{:.2f}",
                "Fut_Close": "₹{:.2f}",
                "Chg%": "{:+.2f}%",
                "Spot_High": "₹{:.2f}",
                "Spot_Low": "₹{:.2f}",
                "Fut_High": "₹{:.2f}",
                "Fut_Low": "₹{:.2f}",
                "Intra_Range_Pos": "{:.2f}",
                "Close_vs_1H_Range_Pos": "{:.2f}",
                "Fut_Basis": "₹{:+.2f}",
                "Fut_Basis%": "{:+.2f}%"
            }),
            use_container_width=True,
            height=450,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Spot_Close": st.column_config.TextColumn("Spot Close ₹", width="small"),
                "Spot_Decimal": st.column_config.TextColumn("Spot Dec", help="Paise decimal of Spot Close (.70, .75, etc.)", width="small"),
                "Fut_Close": st.column_config.TextColumn("Fut Close ₹", width="small"),
                "Fut_Decimal": st.column_config.TextColumn("Fut Dec", help="Paise decimal of Future Close (.70, .75, etc.)", width="small"),
                "Decimal_Match_Tag": st.column_config.TextColumn("🎯 Decimal Match", help="Exact .70 or .75 detection across Spot Close, Fut Close, or Intraday High/Low", width="medium"),
                "One_Hour_Break_Status": st.column_config.TextColumn("1H Breakout Status", help="Direction after 10:15 AM: Bullish breakout above 1H High, Bearish breakdown below 1H Low, or Inside range.", width="medium"),
                "Chg%": st.column_config.TextColumn("Day Chg %", width="small"),
                "Intra_Range_Pos": st.column_config.NumberColumn("Intra Range Pos", help="Live intraday position ((LTP - Day Low) / (Day High - Day Low)).", format="%.2f"),
                "Close_vs_1H_Range_Pos": st.column_config.NumberColumn("Close vs 1H Pos", help="Position relative to 1H range ((Close - 1H Low) / (1H High - 1H Low)).", format="%.2f"),
                "Fut_Basis": st.column_config.TextColumn("Basis (₹)", width="small"),
                "Fut_Basis%": st.column_config.TextColumn("Basis %", width="small"),
                "Days_To_Expiry": st.column_config.NumberColumn("Days Expiry", width="small")
            }
        )

    elif tab2_view_mode == "📈 Past Historical OI Trend (T-0 to T-5 Days)":
        with c_tab2_slider:
            past_oi_days = st.slider(
                "Scroll Past Lookback Period (Days):",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Slide to dynamically inspect past OI change % over 1, 2, 3, 4, or 5 trading sessions."
            )

        st.markdown(f"##### 📊 Past Historical OI Change % (Tracking past {past_oi_days} days & individual sessions)")
        
        # Pick the appropriate cumulative OI column based on slider
        slider_col_map = {1: "OI_1D%", 2: "OI_2D%", 3: "OI_3D%", 4: "OI_3D%", 5: "OI_5D%"}
        active_slider_col = slider_col_map.get(past_oi_days, "OI_2D%")

        hist_cols = [
            "Symbol", "LTP", "Chg%", active_slider_col, "OI_Change%", "OI_T1%", "OI_T2%", "OI_T3%", "OI_T4%", "OI_Buildup_2D"
        ]
        available_hist = [c for c in hist_cols if c in all_df.columns]
        hist_df = all_df[available_hist].copy() if not all_df.empty else pd.DataFrame(columns=available_hist)

        # Rename active slider col for user clarity
        rename_map = {
            active_slider_col: f"Selected Past {past_oi_days}D OI%",
            "OI_Change%": "Today (T0) OI%",
            "OI_T1%": "T-1 Day OI%",
            "OI_T2%": "T-2 Day OI%",
            "OI_T3%": "T-3 Day OI%",
            "OI_T4%": "T-4 Day OI%",
            "OI_Buildup_2D": "2D Buildup Category"
        }
        hist_display_df = hist_df.rename(columns=rename_map)

        st.dataframe(
            hist_display_df.style.format({
                "LTP": "₹{:.2f}",
                "Chg%": "{:+.2f}%",
                f"Selected Past {past_oi_days}D OI%": "{:+.2f}%",
                "Today (T0) OI%": "{:+.2f}%",
                "T-1 Day OI%": "{:+.2f}%",
                "T-2 Day OI%": "{:+.2f}%",
                "T-3 Day OI%": "{:+.2f}%",
                "T-4 Day OI%": "{:+.2f}%"
            }),
            use_container_width=True,
            height=450
        )

    else:
        # Complete Matrix
        fo_cols = [
            "Symbol", "LTP", "Chg%", "RVOL", "Volume_SMA30", 
            "Delivery%", "OI_Change%", "OI_2D_Change%", "OI_Buildup", "OI_Buildup_2D", "PCR", "PCR_Condition"
        ]
        available_fo_cols = [c for c in fo_cols if c in all_df.columns]
        view_fo_df = all_df[available_fo_cols].copy() if not all_df.empty else pd.DataFrame(columns=available_fo_cols)

        fmt_dict = {
            "LTP": "₹{:.2f}",
            "Chg%": "{:+.2f}%",
            "RVOL": "{:.2f}x",
            "Volume_SMA30": "{:,}",
            "Delivery%": "{:.1f}%",
            "OI_Change%": "{:+.2f}%",
            "OI_2D_Change%": "{:+.2f}%",
            "PCR": "{:.2f}"
        }

        st.dataframe(
            view_fo_df.style.format({k: v for k, v in fmt_dict.items() if k in view_fo_df.columns}),
            use_container_width=True,
            height=480,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "LTP": st.column_config.TextColumn("LTP", width="small"),
                "Chg%": st.column_config.TextColumn("Chg%", width="small"),
                "RVOL": st.column_config.TextColumn("RVOL", width="small"),
                "Volume_SMA30": st.column_config.TextColumn("Vol 30D Avg", width="small"),
                "Delivery%": st.column_config.TextColumn("Delivery%", width="small"),
                "OI_Change%": st.column_config.TextColumn("OI 1D%", width="small"),
                "OI_2D_Change%": st.column_config.TextColumn("OI 2D%", width="small"),
                "OI_Buildup": st.column_config.TextColumn("1D Buildup", width="small"),
                "OI_Buildup_2D": st.column_config.TextColumn("2D Buildup", width="medium"),
                "PCR": st.column_config.TextColumn("PCR", width="small"),
                "PCR_Condition": st.column_config.TextColumn("PCR Status", width="medium"),
            }
        )

        st.markdown("""
        <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3B82F6; padding: 8px 14px; border-radius: 6px; margin-top: 10px; font-size: 0.88rem; color: #93C5FD;">
            💡 <b>Scroll & Navigation Tip:</b> Drag horizontally inside the table above using your mouse trackpad or the highlighted blue scrollbar at the bottom to view columns past <b>OI Change %</b> (including 2D OI Buildup, PCR, and PCR Condition). Alternatively, switch to <b>🎯 Derivatives & OI Focus</b> above for a zero-scroll direct view.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🏷️ Spot vs Future Pricing Matrix (Exact .70 / .75 Decimals, Intraday High/Low & Closing)", expanded=False):
        st.markdown("##### 🏷️ Spot vs Future Dual Pricing (Strict .70 / .75 Decimal Tick Scanner)")
        st.caption("Strict filter view: Displays tick-aligned decimals (.70, .75) for Spot and Future, with separate Intraday Range Location and Closing vs 1H Opening Range Position.")

        c_exp_f, _ = st.columns([3, 2])
        with c_exp_f:
            exp_strict = st.checkbox("🎯 Strict Filter: Show ONLY Exact .70 / .75 Decimals (Spot & Future)", value=True, key="chk_exp_strict_pricing")

        if exp_strict:
            exp_mask = _get_bool_s(all_df, "Has_Exact_Decimal_70_75")
            exp_base_df = all_df[exp_mask].copy() if not all_df.empty else pd.DataFrame(columns=all_df.columns)
        else:
            exp_base_df = all_df.copy()

        exp_pricing_cols = [
            "Symbol", "Spot_Close", "Spot_Decimal", "Fut_Close", "Fut_Decimal", "Decimal_Match_Tag",
            "One_Hour_Break_Status", "Chg%",
            "Spot_High", "Spot_Low", "Fut_High", "Fut_Low",
            "Intra_Range_Pos", "Close_vs_1H_Range_Pos",
            "Fut_Basis", "Fut_Basis%", "Days_To_Expiry"
        ]
        available_exp = [c for c in exp_pricing_cols if c in exp_base_df.columns]
        exp_p_df = exp_base_df[available_exp].copy() if not exp_base_df.empty else pd.DataFrame(columns=available_exp)

        st.dataframe(
            exp_p_df.style.format({
                "Spot_Close": "₹{:.2f}",
                "Fut_Close": "₹{:.2f}",
                "Chg%": "{:+.2f}%",
                "Spot_High": "₹{:.2f}",
                "Spot_Low": "₹{:.2f}",
                "Fut_High": "₹{:.2f}",
                "Fut_Low": "₹{:.2f}",
                "Intra_Range_Pos": "{:.2f}",
                "Close_vs_1H_Range_Pos": "{:.2f}",
                "Fut_Basis": "₹{:+.2f}",
                "Fut_Basis%": "{:+.2f}%"
            }),
            use_container_width=True,
            height=380,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Intra_Range_Pos": st.column_config.NumberColumn("Intra Range Pos", format="%.2f"),
                "One_Hour_Break_Status": st.column_config.TextColumn("1H Breakout Status", width="medium"),
                "Close_vs_1H_Range_Pos": st.column_config.NumberColumn("Close vs 1H Pos", format="%.2f")
            }
        )


# =====================================================================
# TAB 3: WEEKLY PIVOT & MONEY ZONE MATRIX
# =====================================================================
with tab3:
    st.markdown("#### 3. Weekly Pivot & Money Zone Matrix (Forward Watchlist for Upcoming Monday)")
    st.info("💡 **Execution Matrix:** Forward-looking calculations (run on Friday afternoons / weekends) to generate the high-conviction watchlist for the upcoming week.")

    mz_cols = [
        "Symbol", "LTP", "Next_W_Pivot", "Next_W_High_CPR", "Next_W_Low_CPR", "Next_W_CPR_Width%",
        "Next_W_Cam_H4", "Next_W_VAH", "Next_W_VAL", "Watchlist_Setups"
    ]
    view_mz = all_df[mz_cols].copy() if not all_df.empty else pd.DataFrame(columns=mz_cols)

    st.dataframe(
        view_mz.style.format({
            "LTP": "₹{:.2f}",
            "Next_W_Pivot": "₹{:.2f}",
            "Next_W_High_CPR": "₹{:.2f}",
            "Next_W_Low_CPR": "₹{:.2f}",
            "Next_W_CPR_Width%": "{:.3f}%",
            "Next_W_Cam_H4": "₹{:.2f}",
            "Next_W_VAH": "₹{:.2f}",
            "Next_W_VAL": "₹{:.2f}"
        }),
        use_container_width=True,
        height=450
    )

    # Strategy breakdown cards
    st.markdown("---")
    st.subheader("📋 Monday Strategy Execution Rules")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""
        **🌀 Volatility Squeeze**
        - **Condition:** Next VAH < Curr VAH & Next VAL > Curr VAL (Inside Value)
        - **Execution:** 1H close above Weekly R1 / VAH
        """)
    with s2:
        st.markdown("""
        **🎯 Narrow CPR**
        - **Condition:** Next CPR Width < 0.5% of LTP
        - **Execution:** Breakout above High_CPR with RVOL > 1.5
        """)
    with s3:
        st.markdown("""
        **📈 Trend Continuation**
        - **Condition:** Next CPR > Curr CPR & Next POC > Curr POC
        - **Execution:** Pullback to POC/CPR, bounce validated by 15m VWAP crossover
        """)
    with s4:
        st.markdown("""
        **⚡ Camarilla Alignment**
        - **Condition:** Weekly Cam H4 within 0.2% of Weekly VAH
        - **Execution:** LTP crosses H4 + 0.15% buffer
        """)


# =====================================================================
# TAB FIB: ADVANCED FIBONACCI & HIGH BETA CONFLUENCE (POINTS 1 - 10)
# =====================================================================
with tab_fib:
    st.markdown("#### 📐 Advanced Fibonacci Retracement & High Beta F&O Confluence Engine")
    st.caption("Module: `Fibonacci_Screener_V2` | Implements Multi-Timeframe (MTF) clustering, Blue Sky extensions, Volume Profile confluence, and institutional price action.")

    f_sub1, f_sub2, f_sub3 = st.tabs([
        "Part I: Advanced Fibonacci (Points 1-4)",
        "Part II: High Beta F&O Confluences (Points 5-10)",
        "Full Fibonacci Screener Monitor & CSV Export"
    ])

    with f_sub1:
        st.markdown("##### Part I: Golden Pocket (0.618), MTF Clusters, Extensions, VP Confluence & Wick Rejections")
        p1_cols = [
            "Symbol", "LTP", "Chg%", "Swing_Type_1H", "Fib_0618_1H",
            "Ext_Target_1", "Ext_Target_2", "Dist_To_Ext_1%", "Ext_Setup_Tag", "Is_Ext_Breakout",
            "Is_Golden_Pocket_Confluence", "Is_Golden_Pocket", "Is_Golden_Zone_Pullback",
            "Is_Cluster_Active", "Cluster_Level_Price", "Cluster_Labels",
            "Vol_Fib_Confluence_Flag", "Confluence_Node_Type", "Confluence_Node_Price",
            "Is_Fib_Rejection", "Rejection_Level", "Rejection_Wick%", "Rejection_Tag"
        ]
        available_p1_cols = [c for c in p1_cols if c in all_df.columns]
        p1_df = all_df[available_p1_cols].copy()

        # Quick Highlights
        cl_active_df = p1_df[p1_df.get("Is_Cluster_Active", False) == True]
        ext_active_df = p1_df[p1_df.get("Is_Ext_Breakout", False) == True]
        vp_gp_active_df = p1_df[p1_df.get("Vol_Fib_Confluence_Flag", False) == True]
        wick_active_df = p1_df[p1_df.get("Is_Fib_Rejection", False) == True]

        col_h1, col_h2, col_h3, col_h4, col_h5, col_h6 = st.columns(6)
        col_h1.metric("⭐ GP Confluence", stats.get("fib_gp_confluence_count", 0))
        col_h2.metric("🟡 At Golden Pocket", stats.get("fib_at_gp_count", 0))
        col_h3.metric("MTF Clusters", len(cl_active_df))
        col_h4.metric("Extension Breakouts", len(ext_active_df))
        col_h5.metric("VP + GP Confluence", len(vp_gp_active_df))
        col_h6.metric("15m Wick Rejection", len(wick_active_df))

        fmt_p1 = {
            "LTP": "₹{:.2f}",
            "Chg%": "{:+.2f}%",
            "Fib_0618_1H": "₹{:.2f}",
            "Ext_Target_1": "₹{:.2f}",
            "Ext_Target_2": "₹{:.2f}",
            "Dist_To_Ext_1%": "{:+.2f}%",
            "Cluster_Level_Price": "₹{:.2f}",
            "Confluence_Node_Price": "₹{:.2f}",
            "Rejection_Level": "₹{:.2f}",
            "Rejection_Wick%": "{:.1f}%"
        }
        h_p1 = min(460, max(260, 55 + len(p1_df) * 36))
        st.dataframe(
            p1_df.style.format({k: v for k, v in fmt_p1.items() if k in p1_df.columns}),
            use_container_width=True,
            height=h_p1
        )

    with f_sub2:
        st.markdown("##### Part II: Golden Pivots, Camarilla Traps, Prem/Disc Oscillator & Macro Swings")
        p2_cols = [
            "Symbol", "LTP", "Chg%", "Fib_0618_1H",
            "Weekly_Fib_Target_1", "Weekly_Fib_Target_2", "Weekly_CPR_Squeeze_Active", "Weekly_Fib_Exp_Bull", "Weekly_Fib_Exp_Bear",
            "Is_Golden_Pivot_Bullish", "Is_Golden_Pivot_Bearish", "Golden_Pivot_Zone",
            "Is_Camarilla_Trap_Bull", "Is_Camarilla_Trap_Bear", "Camarilla_Trap_Level", "Camarilla_Trap_Tag",
            "Prem_Disc_Status", "Dist_From_POC_ATR",
            "Is_Macro_Anchor_Bull", "Is_Macro_Anchor_Bear", "Macro_Anchor_Zone", "Macro_Anchor_Tag",
            "Is_Inside_Week_GP_Bull", "Is_Inside_Week_GP_Bear", "Inside_Week_GP_Status"
        ]
        available_p2_cols = [c for c in p2_cols if c in all_df.columns]
        p2_df = all_df[available_p2_cols].copy()

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        col_k1.metric("Golden Pivots", int((all_df.get("Is_Golden_Pivot_Bullish", False) | all_df.get("Is_Golden_Pivot_Bearish", False)).sum()))
        col_k2.metric("Camarilla Traps", int((all_df.get("Is_Camarilla_Trap_Bull", False) | all_df.get("Is_Camarilla_Trap_Bear", False)).sum()))
        col_k3.metric("Macro Anchor Swings", int((all_df.get("Is_Macro_Anchor_Bull", False) | all_df.get("Is_Macro_Anchor_Bear", False)).sum()))
        col_k4.metric("Inside Week @ 0.618", int((all_df.get("Is_Inside_Week_GP_Bull", False) | all_df.get("Is_Inside_Week_GP_Bear", False)).sum()))

        fmt_p2 = {
            "LTP": "₹{:.2f}",
            "Chg%": "{:+.2f}%",
            "Fib_0618_1H": "₹{:.2f}",
            "Weekly_Fib_Target_1": "₹{:.2f}",
            "Weekly_Fib_Target_2": "₹{:.2f}",
            "Golden_Pivot_Zone": "₹{:.2f}",
            "Camarilla_Trap_Level": "₹{:.2f}",
            "Macro_Anchor_Zone": "₹{:.2f}"
        }
        h_p2 = min(460, max(260, 55 + len(p2_df) * 36))
        st.dataframe(
            p2_df.style.format({k: v for k, v in fmt_p2.items() if k in p2_df.columns}),
            use_container_width=True,
            height=h_p2
        )

    with f_sub3:
        st.markdown("##### Full Universe Advanced Fibonacci Monitor")
        sym_search = st.text_input("🔍 Search Symbol in Fibonacci Screener", "").strip().upper()
        display_fib_df = all_df
        if sym_search:
            display_fib_df = display_fib_df[display_fib_df["Symbol"].str.contains(sym_search)]

        h_fib_mon = min(520, max(280, 55 + len(display_fib_df) * 36))
        st.dataframe(display_fib_df, use_container_width=True, height=h_fib_mon)

        csv_fib = display_fib_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Fibonacci Screener V2.0 Data (CSV)",
            data=csv_fib,
            file_name=f"Fibonacci_Screener_V2_{base_meta.get('live_date', 'today')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Documentation & Formula Guide
    with st.expander("📖 Developer Documentation: Advanced Fibonacci & Target Specifications Guide"):
        st.markdown(r"""
        ### Part I: Advanced Fibonacci & Target Enhancements
        1. **Fibonacci Cluster (MTF Confluence) Filter:**
           - Detects institutional support/resistance where 1-Hour Macro and 15-Minute Micro Fibonacci levels align within `±0.5%`.
           - `ABS(1H_Fib_Value - 15m_Fib_Value) <= CONFLUENCE_TOLERANCE_PCT`.
        2. **Fibonacci Extension Breakouts & Breakdowns (1.272 & 1.618):**
           - Projects 1st (1.272) and 2nd (1.618) profit-booking targets for both bullish breakouts beyond 1.0 (Swing High / Blue Sky) and bearish breakdowns below 0.0 (Swing Low) with volume confirmation.
           - Upside Breakout Targets: `Ext 1 = Swing_Low + (Range × 1.272)`, `Ext 2 = Swing_Low + (Range × 1.618)`.
           - Downside Breakdown Targets: `Ext 1 = Swing_High - (Range × 1.272)`, `Ext 2 = Swing_High - (Range × 1.618)`.
        3. **Volume Profile + Golden Pocket Confluence:**
           - Flags 0.618 Golden Pocket alignment with Volume Profile Point of Control (POC) or Top 3 High Volume Nodes (HVNs) within `±0.5%`.
        4. **Fibonacci Rejection (Wick) Scanner:**
           - 15-minute candlestick anatomy checking for institutional buying/selling pressure at 0.500 or 0.618 with lower/upper wick $\ge 60\%$ of total range.

        ### Part II: High Beta F&O Confluence Filters
        5. **The 'Golden Pivot' Confluence:** 1H Fib 0.618/0.500 within `±0.5%` of Daily Pivot S1/S2 AND within `±0.5%` of VAL/POC.
        6. **High Beta Camarilla Reversal Trap:** Camarilla L3 overlaps with 1H Fib 0.618; 15m candle breaches L3 then closes back inside Money Zone.
        7. **Institutional Premium/Discount Oscillator:** Measures distance from POC $> 2 \times ATR_{14}$ combined with S3/R3 and Fib -0.618 / 1.618 extensions.
        8. **The Macro Anchor (Monthly Value + Weekly Fib):** 3-15 day swing hold when Weekly Fib aligns with MPOC and Monthly Pivots with Daily candlestick confirmation.
        9. **Weekly CPR Squeeze & Fib Expansion:** Volatility compression (Weekly CPR $< 0.5\%$) projecting targets +1.272 / +1.618 or -0.272 / -0.618.
        10. **Inside Week + Golden Pocket Compression:** Inside Week compression with Close within $\pm 1\%$ of Fib 0.618 and Weekly POC in the bottom 30% (Accumulation) or top 30% (Distribution).
        """)


# =====================================================================
# TAB SWING: HIGH-CONVICTION SWING TRADE OPTIMIZATIONS (DEVELOPER SPEC)
# =====================================================================
with tab_swing:
    st.markdown("#### 🚀 High-Conviction Swing Trade Optimizations")
    st.caption("Module: `Swing_Optimization_Filters` | Multidimensional Confluence: Derivatives 2D OI Buildup (≥5%), 14-Day Relative Strength vs Nifty Total Market, and Volatility Compression (Coiled Spring: NR7 + BB Squeeze).")

    # Swing KPI Row
    sw1, sw2, sw3, sw4 = st.columns(4)
    with sw1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">⭐ High-Conviction Swings</div>
            <div class="metric-val" style="color: #FFD54F;">{stats.get('swing_high_conviction_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with sw2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">⚡ Coiled Springs (NR7/BB)</div>
            <div class="metric-val" style="color: #00E5FF;">{stats.get('coiled_spring_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with sw3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">🚀 RS Outperforming (Alpha > 0)</div>
            <div class="metric-val" style="color: #00E676;">{stats.get('rs_outperforming_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with sw4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">📈 2D Long Buildup (OI ≥ 5%)</div>
            <div class="metric-val" style="color: #AB47BC;">{stats.get('long_buildup_2d_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Sub-views inside Swing Tab
    sw_sub1, sw_sub2, sw_sub3 = st.tabs([
        "⭐ High-Conviction Swing Candidates",
        "⚡ Volatility Compression: Coiled Spring (NR7 + BB Squeeze)",
        "📊 Full Universe Swing Optimization Monitor"
    ])

    with sw_sub1:
        st.markdown("##### ⭐ High-Conviction Swing Trade Candidates (Multi-Dimensional Alignment)")
        st.caption("Requires Coiled Spring compression directly at a Fibonacci or Money Zone structural level, validated by Relative Strength vs Nifty Total Market and Derivatives Open Interest buildup.")
        
        high_conv_df = all_df[all_df["Is_Swing_Setup"] == True].copy() if not all_df.empty else pd.DataFrame()
        if not high_conv_df.empty:
            sc_cols = [
                "Symbol", "LTP", "Chg%", "Swing_Setup_Tag", "RS_Alpha%", "Ret_Stock_14D%", "Ret_Index_14D%",
                "RS_Status", "OI_2D_Change%", "OI_Buildup_2D", "Coiled_Spring_Tag", "Swing_Confluence_Reasons"
            ]
            available_sc_cols = [c for c in sc_cols if c in high_conv_df.columns]
            h_sc = min(460, max(260, 55 + len(high_conv_df) * 36))
            st.dataframe(high_conv_df[available_sc_cols], use_container_width=True, height=h_sc)
        else:
            st.info("ℹ️ No stocks currently meet the strict High-Conviction Swing multi-dimensional criteria simultaneously. Check the 'Full Universe Swing Monitor' tab below to view all symbol rankings.")

    with sw_sub2:
        st.markdown("##### ⚡ The 'Coiled Spring' Monitor: NR7 Range Contraction & 120-Day BB Squeeze")
        st.caption("Isolates symbols with extreme volatility compression ready for expansion before breakout.")
        coiled_df = all_df[all_df["Is_Coiled_Spring"] == True].copy() if not all_df.empty else pd.DataFrame()
        if not coiled_df.empty:
            coil_cols = [
                "Symbol", "LTP", "Chg%", "Coiled_Spring_Tag", "Is_NR7", "Is_BB_Squeeze",
                "BB_Width%", "BB_Width_120_Avg%", "RS_Alpha%", "OI_2D_Change%"
            ]
            available_coil_cols = [c for c in coil_cols if c in coiled_df.columns]
            h_coil = min(460, max(260, 55 + len(coiled_df) * 36))
            st.dataframe(coiled_df[available_coil_cols], use_container_width=True, height=h_coil)
        else:
            st.info("ℹ️ No symbols currently in NR7 or BB Squeeze compression state.")

    with sw_sub3:
        st.markdown("##### 📋 Full Universe Swing Optimization Monitor & CSV Export")
        sw_monitor_cols = [
            "Symbol", "LTP", "Chg%", "RS_Alpha%", "Ret_Stock_14D%", "Ret_Index_14D%", "RS_Status",
            "OI_2D_Change%", "OI_Buildup_2D", "Is_Long_Buildup_2D", "Is_Short_Buildup_2D",
            "Is_NR7", "Is_BB_Squeeze", "Coiled_Spring_Tag", "Swing_Setup_Tag", "Swing_Confluence_Reasons"
        ]
        available_mon_cols = [c for c in sw_monitor_cols if c in all_df.columns]
        mon_df = all_df[available_mon_cols].copy() if not all_df.empty else pd.DataFrame(columns=sw_monitor_cols)
        h_mon = min(520, max(280, 55 + len(mon_df) * 36))
        st.dataframe(mon_df, use_container_width=True, height=h_mon)

        csv_sw = mon_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Swing Optimization Data (CSV)",
            data=csv_sw,
            file_name=f"Swing_Optimization_Filters_{base_meta.get('live_date', 'today')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Strategy Execution Guide
    st.markdown("---")
    st.subheader("📚 Developer Specification Architecture: Swing Optimization Rules")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        **1. Derivatives Confluence**
        - **Metric:** 2-day rolling window (`OI_LOOKBACK_DAYS = 2`).
        - **Long Buildup:** Price Delta > 0 AND OI Delta ≥ 5.0%.
        - **Short Buildup:** Price Delta < 0 AND OI Delta ≥ 5.0%.
        - **Institutional Purpose:** Confirms genuine smart money accumulation at Fibonacci/Value Area levels.
        """)
    with r2:
        st.markdown(f"""
        **2. Sector & Index Relative Strength (RS)**
        - **Metric:** 14-day rolling return vs **{benchmark_choice}**.
        - **Formula:** `rs_alpha_pct = Ret_Stock - Ret_Index`.
        - **Bullish Rule:** Requires `Ret_Stock > Ret_Index` (Alpha > 0%).
        - **Bearish Rule:** Requires `Ret_Stock < Ret_Index` (Alpha < 0%).
        - **Purpose:** Eliminates lagging stocks, focusing only on leaders.
        """)
    with r3:
        st.markdown(f"""
        **3. The 'Coiled Spring'**
        - **NR7:** Today's True Range < MIN(Previous 6 days' True Range).
        - **BB Squeeze:** Current Bandwidth < 120-Day Average Bandwidth.
        - **Master Confluence:** Flagged when `(NR7 OR BB Squeeze)` aligns with a **Fibonacci** or **Money Zone** level.
        - **Purpose:** Pinpoints maximum coiled energy before explosive trend moves.
        """)


# =====================================================================
# TAB 4: INTERACTIVE TECHNICAL CHART
# =====================================================================
with tab4:
    st.markdown("#### 5. Multi-Timeframe Interactive Technical Chart & Fibonacci Overlay")
    chart_sym = st.selectbox("Select Symbol for Deep Dive", options=selected_symbols, index=0)

    try:
        # Retrieve from in-memory intraday cache first for zero-latency instant rendering (< 5ms)
        df_chart_raw = st.session_state.get("intraday_cache_v2", {}).get(chart_sym)
        if df_chart_raw is None or df_chart_raw.empty:
            t_sym = f"{chart_sym}.NS"
            df_chart_raw = yf.download(t_sym, period="5d", interval=timeframe, progress=False)
            if isinstance(df_chart_raw.columns, pd.MultiIndex):
                df_chart_raw.columns = df_chart_raw.columns.get_level_values(-1)

        if not df_chart_raw.empty:
            chart_mtf = calculate_mtf_v2(df_chart_raw)
            c_today = chart_mtf["curr_15m_df"]

            if not c_today.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])

                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=c_today.index,
                    open=c_today["Open"],
                    high=c_today["High"],
                    low=c_today["Low"],
                    close=c_today["Close"],
                    name="15m Price",
                    increasing_line_color="#00E676",
                    decreasing_line_color="#FF5252"
                ), row=1, col=1)

                # Daily Anchored VWAP
                fig.add_trace(go.Scatter(
                    x=c_today.index,
                    y=c_today["VWAP"],
                    name="Daily VWAP (09:15 Anchor)",
                    line=dict(color="#29B6F6", width=2)
                ), row=1, col=1)

                # PDH / PDL
                if not np.isnan(chart_mtf["pdh"]):
                    fig.add_hline(y=chart_mtf["pdh"], line=dict(color="#66BB6A", dash="dash"), annotation_text="PDH", row=1, col=1)
                if not np.isnan(chart_mtf["pdl"]):
                    fig.add_hline(y=chart_mtf["pdl"], line=dict(color="#EF5350", dash="dash"), annotation_text="PDL", row=1, col=1)

                # 1H Initial Balance Lines
                if not np.isnan(chart_mtf["ib_high"]):
                    fig.add_hline(y=chart_mtf["ib_high"], line=dict(color="#AB47BC", dash="dot"), annotation_text="1H IB High", row=1, col=1)
                if not np.isnan(chart_mtf["ib_low"]):
                    fig.add_hline(y=chart_mtf["ib_low"], line=dict(color="#AB47BC", dash="dot"), annotation_text="1H IB Low", row=1, col=1)
                if not np.isnan(chart_mtf["ib_close"]):
                    fig.add_hline(y=chart_mtf["ib_close"], line=dict(color="#FFD54F", dash="dashdot"), annotation_text="10:15 1H Close", row=1, col=1)

                # Fibonacci & Target Extensions Overlay
                stock_lookup = all_df[all_df["Symbol"] == chart_sym]
                if not stock_lookup.empty:
                    s_row = stock_lookup.iloc[0]
                    f_618 = s_row.get("Fib_0618_1H", np.nan)
                    e_t1 = s_row.get("Ext_Target_1", np.nan)
                    e_t2 = s_row.get("Ext_Target_2", np.nan)

                    if not np.isnan(f_618) and f_618 > 0:
                        fig.add_hline(y=f_618, line=dict(color="#FFD54F", width=1.5, dash="dash"), annotation_text=f"Fib 0.618 (₹{f_618:.2f})", row=1, col=1)
                    if not np.isnan(e_t1) and e_t1 > 0:
                        fig.add_hline(y=e_t1, line=dict(color="#00E5FF", width=1.5, dash="dot"), annotation_text=f"Ext T1 1.272 (₹{e_t1:.2f})", row=1, col=1)
                    if not np.isnan(e_t2) and e_t2 > 0:
                        fig.add_hline(y=e_t2, line=dict(color="#00E676", width=1.5, dash="dot"), annotation_text=f"Ext T2 1.618 (₹{e_t2:.2f})", row=1, col=1)

                # Volume Bar
                fig.add_trace(go.Bar(
                    x=c_today.index,
                    y=c_today["Volume"],
                    name="Volume",
                    marker_color="#455A64"
                ), row=2, col=1)

                fig.update_layout(
                    title=f"{chart_sym} - Screener Architecture V2.0 Confluence Overlay",
                    template="plotly_dark",
                    height=580,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)

                if not stock_lookup.empty:
                    s_row = stock_lookup.iloc[0]
                    st.markdown(f"""
                    <div style="background: rgba(255, 213, 79, 0.08); border: 1px solid #FFD54F; border-radius: 6px; padding: 10px 14px; margin-top: 10px;">
                        <span style="font-weight: 700; color: #FFD54F;">📐 Fibonacci Retracement & Confluence ({s_row.get('Swing_Type_1H', 'Neutral')}):</span> &nbsp;
                        <b>1H Swing:</b> ₹{s_row.get('Swing_Low_1H')} - ₹{s_row.get('Swing_High_1H')} &nbsp;|&nbsp;
                        <b>Golden Pocket (0.618):</b> ₹{s_row.get('Fib_0618_1H')} &nbsp;|&nbsp;
                        <span style="color: #00E5FF; font-weight: 700;">Ext Target 1 (1.272):</span> ₹{s_row.get('Ext_Target_1')} &nbsp;|&nbsp;
                        <span style="color: #00E676; font-weight: 700;">Ext Target 2 (1.618):</span> ₹{s_row.get('Ext_Target_2')} &nbsp;|&nbsp;
                        <b>Cluster:</b> {s_row.get('Cluster_Labels', 'None')} &nbsp;|&nbsp;
                        <b>VP Confluence:</b> {s_row.get('Confluence_Node_Type', 'None')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Waiting for intraday session candles to populate.")
        else:
            st.warning("Could not download intraday chart candles.")
    except Exception as e:
        st.error(f"Error rendering chart: {e}")


# =====================================================================
# TAB 5: ARCHITECTURE SPEC & PINE SCRIPT V5
# =====================================================================
with tab5:
    st.markdown("#### 📜 Screener Architecture V2.0 Specifications & Pine Script v5")
    st.markdown(r"""
    This system implements the exact specifications from **Screener Architecture V2.0 (November 2026)** and **Advanced Fibonacci & Target Enhancements**:
    - **1. Core MTF Strategy:**
      - Daily (1D): PDH / PDL shifted by 1 period.
      - Hourly (1H): Initial Balance locked at 10:15 AM IST (Captures 1H High, Low, and Close).
      - 15-Minute (15m): Anchored VWAP reset daily at 09:15 AM IST with Typical Price $(H + L + C) / 3$.
      - Global flag `bypass_52w_proximity` to trade breakouts across the universe.
    - **1.1 Initial Hour (1H) Logic Execution:**
      - Option A: 1H Close vs Prev 1H Close.
      - Option B: Live vs 1H Close ($LTP > Today\_1015\_Close$).
      - Option C: IB Range Breakout ($LTP > 1H\ High$ / $< 1H\ Low$) and Break-ins when price tags $\pm 0.2\%$ outside range and re-enters.
    - **2. Advanced Validations:**
      - Relative Volume: $RVOL = Current\_Volume / SMA(Volume, 30\_days) \ge 1.5$.
      - Delivery %: Exchange feed / institutional accumulation $> 50\%$.
      - Derivatives & OI: Long Buildup, Short Covering, Short Buildup, Long Unwinding.
      - Put-Call Ratio (PCR): Squeeze $< 0.6$ or Support $> 1.2$.
    - **3. Weekly Pivot & Money Zone Matrix:**
      - Standard CPR: $Pivot = (H+L+C)/3, BC = (H+L)/2, TC = (Pivot-BC)+Pivot$.
      - Camarilla H4/L4: $Range = H-L, H4 = C + (Range \times 1.1 / 2)$.
      - Forward-looking Friday-for-Monday setups: Volatility Squeeze, Narrow CPR ($< 0.5\%$), Trend Continuation, Camarilla Alignment.
    - **4. Advanced Fibonacci & High Beta F&O Confluences (Module: Fibonacci_Screener_V2):**
      - MTF Clusters (1H & 15m), Extension Breakouts (1.272 & 1.618), Volume Profile + Golden Pocket Confluence, 15m Wick Rejection Scanner ($\ge 60\%$).
      - Golden Pivots (Fib + Pivots + Value Area), Camarilla Reversal Traps (L3/H3), Prem/Disc Oscillator, Macro Anchor (Monthly + Weekly), Weekly CPR Squeeze & Fib Expansion, Inside Week + 0.618 Compression.
    """)

    st.subheader("TradingView Pine Script (v5) Code")
    st.code(PINE_SCRIPT_V5_CODE, language="pinescript")

