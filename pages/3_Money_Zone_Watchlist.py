"""
Multi-Page Screener Module: Money Zone & Derivatives Watchlist
File: pages/3_Money_Zone_Watchlist.py

Displays a dedicated screen showcasing the list of stocks qualified across all
non-Fibonacci categories (Money Zone, Core MTF Breakouts, Volume & Derivatives):
  Part I: Core Directional & MTF Breakouts
    1. Core MTF Bullish Breakouts (LTP > Yesterday's VAH + MTF Confluence)
    2. Core MTF Bearish Breakdowns (LTP < Yesterday's VAL + MTF Confluence)
    3. 1H Initial Balance (IB) Range Breakouts (> 10:15 AM High)
    4. 1H Initial Balance (IB) Range Breakdowns (< 10:15 AM Low)
  Part II: Weekly Pivot & Money Zone Matrix Setups (Forward Watchlist)
    5. Setup 1: Volatility Squeeze (Inside Value Area: Next VAH < VAH & Next VAL > VAL)
    6. Setup 2: Narrow Central Pivot Range (Next Week CPR < 0.5% Width)
    7. Setup 3: Trend Continuation (Higher Upcoming CPR & Higher POC)
    8. Setup 4: Camarilla Alignment (Upcoming Weekly H4 ≈ VAH ±0.2%)
  Part III: Volume & Order Flow Validations
    9. Relative Volume Surge (RVOL >= 1.5x of 30-Day SMA)
    10. Institutional Delivery Accumulation (> 50% Delivery)
  Part IV: Derivatives & Open Interest (OI) Quadrants
    11. Derivatives 1D Long Buildup (Price Up + OI Up)
    12. Derivatives 1D Short Covering (Price Up + OI Down)
    13. Derivatives 1D Short Buildup (Price Down + OI Up)
    14. Put-Call Ratio (PCR) Extremes (Bullish Support > 1.2 or Squeeze < 0.6)
  Part V: Intraday Close Strength & 0.70 / 0.75 Range Position
    15. Strong Intraday Close (Range Position >= 0.70 or >= 0.75 with NSE Tick Precision)

This page runs completely independently and leaves existing screener pages (app.py and pages/1_Screener_V2.py) untouched.
"""

import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import streamlit as st

from fo_stocks import TOP_30_LIQUID, NIFTY_50, BANK_NIFTY, get_fo_universe
from screener_v2_engine import (
    fetch_universe_data_v2,
    fetch_daily_1y_data_v2,
    fetch_intraday_data_v2,
    ensure_decimal_columns
)
import swing_optimization_filters
from pwa_launcher import inject_pwa_headers, render_app_launcher_sidebar

# ----------------- Streamlit Page Configuration -----------------
st.set_page_config(
    page_title="Money Zone Watchlist | F&O Screener",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 10px;
    }
    .metric-lbl {
        font-size: 0.72rem;
        color: #9E9E9E;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-val {
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 3px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    }
    .category-header-card {
        background: linear-gradient(90deg, #161B26 0%, #1C2433 100%);
        border-left: 5px solid #8B5CF6;
        border-top: 1px solid #2A3447;
        border-right: 1px solid #2A3447;
        border-bottom: 1px solid #2A3447;
        border-radius: 8px;
        padding: 12px 18px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .category-badge-active {
        background-color: rgba(139, 92, 246, 0.18);
        color: #A78BFA;
        border: 1px solid #8B5CF6;
        padding: 3px 10px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .category-badge-empty {
        background-color: rgba(148, 163, 184, 0.12);
        color: #94A3B8;
        border: 1px solid #475569;
        padding: 3px 10px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.85rem;
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
</style>
""", unsafe_allow_html=True)


# =====================================================================
# SIDEBAR CONTROLS & DATA INGESTION
# =====================================================================
st.sidebar.title("🔮 Money Zone Watchlist")
st.sidebar.caption("Page 4: Money Zone, Core Breakouts, Volume & Derivatives Categories")

# Navigation links to other pages
with st.sidebar.expander("🧭 Quick Page Navigation", expanded=False):
    st.markdown("""
    * 🏛️ **[Classic Screener V1](/)**
    * ⚙️ **[Screener V2.0](/1_Screener_V2)**
    * 📐 **[Fibonacci Watchlist](/2_Fibonacci_Watchlist)**
    * 🔮 **[Money Zone Watchlist (Current)](/3_Money_Zone_Watchlist)**
    """)

# Stock Universe Presets
st.sidebar.subheader("📋 Stock Universe")
universe_choice = st.sidebar.selectbox(
    "Select Universe Preset",
    ["Top 30 Liquid F&O", "Nifty 50", "Bank Nifty", "Full F&O Universe (180+)", "Custom Selection"],
    index=0,
    help="Select the pool of F&O contracts to scan across all Money Zone categories."
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

# Intraday Timeframe
timeframe = st.sidebar.selectbox(
    "Intraday Timeframe",
    ["5m", "15m", "1h"],
    index=0,
    help="Underlying candle timeframe used for intraday swing and VWAP computations."
)

# Benchmark Index for Relative Strength
benchmark_choice = st.sidebar.selectbox(
    "Benchmark Index (RS Alpha)",
    [
        "Nifty Total Market (Broad 750 / 500)",
        "Nifty 50 (^NSEI)",
        "Bank Nifty (^NSEBANK)"
    ],
    index=0,
    help="Benchmark index for market performance comparison."
)
bench_map = {
    "Nifty Total Market (Broad 750 / 500)": "NIFTY_TOTAL_MARKET",
    "Nifty 50 (^NSEI)": "NIFTY50",
    "Bank Nifty (^NSEBANK)": "BANKNIFTY"
}
bench_key = bench_map[benchmark_choice]

# Cached Data Ingestion (Shared with Screener V2 for zero-latency instant loading)
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
    if st.button("⚡ Reset", help="Clear all market caches"):
        st.cache_data.clear()

# ----------------- Data Loading -----------------
with st.spinner(f"Loading market data for {len(selected_symbols)} F&O symbols across Money Zone categories..."):
    raw_df, base_meta = get_v2_data(selected_symbols, timeframe, bench_key)

if raw_df.empty:
    st.warning("⚠️ No data was returned. Please verify internet connectivity or symbol list.")
    st.stop()


# =====================================================================
# FILTERING ENGINE: BUILD THE MONEY ZONE & NON-FIBONACCI CATEGORIES
# =====================================================================
categories_data: Dict[int, Dict[str, Any]] = {}

# 1. Core MTF Bullish Breakouts (LTP > Yesterday's VAH + MTF Confluence)
mask_1 = (raw_df.get("Bullish_Breakout", False) == True) | ((raw_df["LTP"] > raw_df["PD_VAH"]) & raw_df.get("MTF_Bullish", False))
df_cat1 = raw_df[mask_1].copy().sort_values(by="Dist_52W_High%", ascending=False).reset_index(drop=True)
categories_data[1] = {
    "num": 1,
    "icon": "🚀",
    "name": "Core MTF Bullish Breakouts (LTP > VAH + MTF)",
    "group": "Part I: Core MTF Breakouts",
    "desc": "Price breaking out above yesterday's 70% Value Area High (VAH) validated by multi-timeframe (1D, 1H Initial Balance, 15m VWAP) trend confluence.",
    "df": df_cat1,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "PD_VAH", "Dist_VAH%", "MTF_Signal", "Opt_A_Status", "Opt_B_Status", "Opt_C_Status",
        "Spot_Close", "Fut_Close", "Spot_Range_Pos"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "PD_VAH": "₹{:.2f}", "Dist_VAH%": "{:+.2f}%",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}", "Spot_Range_Pos": "{:.2f}"
    }
}

# 2. Core MTF Bearish Breakdowns (LTP < Yesterday's VAL + MTF Confluence)
mask_2 = (raw_df.get("Bearish_Breakdown", False) == True) | ((raw_df["LTP"] < raw_df["PD_VAL"]) & raw_df.get("MTF_Bearish", False))
df_cat2 = raw_df[mask_2].copy().sort_values(by="Dist_52W_Low%", ascending=True).reset_index(drop=True)
categories_data[2] = {
    "num": 2,
    "icon": "🔻",
    "name": "Core MTF Bearish Breakdowns (LTP < VAL + MTF)",
    "group": "Part I: Core MTF Breakouts",
    "desc": "Price breaking down below yesterday's 70% Value Area Low (VAL) validated by multi-timeframe bearish alignment.",
    "df": df_cat2,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "PD_VAL", "Dist_VAL%", "MTF_Signal", "Opt_A_Status", "Opt_B_Status", "Opt_C_Status",
        "Spot_Close", "Fut_Close", "Spot_Range_Pos"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "PD_VAL": "₹{:.2f}", "Dist_VAL%": "{:+.2f}%",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}", "Spot_Range_Pos": "{:.2f}"
    }
}

# 3. 1H Initial Balance (IB) Range Breakouts (> 10:15 AM High)
mask_3 = raw_df["LTP"] > raw_df["IB_High"]
df_cat3 = raw_df[mask_3].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[3] = {
    "num": 3,
    "icon": "🎯",
    "name": "1H Initial Balance (IB) Range Breakouts (> 10:15 AM High)",
    "group": "Part I: Core MTF Breakouts",
    "desc": "Option C Breakout: Price trading strictly above the 1-Hour Initial Balance High established at 10:15 AM IST.",
    "df": df_cat3,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "IB_High", "IB_Low", "IB_Close", "Opt_C_Status",
        "Spot_Close", "Fut_Close", "Spot_Range_Pos"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "IB_High": "₹{:.2f}", "IB_Low": "₹{:.2f}", "IB_Close": "₹{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}", "Spot_Range_Pos": "{:.2f}"
    }
}

# 4. 1H Initial Balance (IB) Range Breakdowns (< 10:15 AM Low)
mask_4 = raw_df["LTP"] < raw_df["IB_Low"]
df_cat4 = raw_df[mask_4].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[4] = {
    "num": 4,
    "icon": "📉",
    "name": "1H Initial Balance (IB) Range Breakdowns (< 10:15 AM Low)",
    "group": "Part I: Core MTF Breakouts",
    "desc": "Option C Breakdown: Price trading strictly below the 1-Hour Initial Balance Low established at 10:15 AM IST.",
    "df": df_cat4,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "IB_High", "IB_Low", "IB_Close", "Opt_C_Status",
        "Spot_Close", "Fut_Close", "Spot_Range_Pos"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "IB_High": "₹{:.2f}", "IB_Low": "₹{:.2f}", "IB_Close": "₹{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}", "Spot_Range_Pos": "{:.2f}"
    }
}

# 5. Volatility Squeeze (Inside Value Area)
mask_5 = raw_df.get("Setup_Vol_Squeeze", False) == True
df_cat5 = raw_df[mask_5].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[5] = {
    "num": 5,
    "icon": "🔮",
    "name": "Setup 1: Volatility Squeeze (Inside Value Area)",
    "group": "Part II: Weekly Pivot & Money Zone Matrix",
    "desc": "Forward-looking Friday-for-Monday setup: Next Week's VAH < Current VAH AND Next Week's VAL > Current VAL (Inside Value Area compression). Trigger on 1H close above Weekly R1/VAH.",
    "df": df_cat5,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Next_W_VAH", "Next_W_VAL", "Next_W_Pivot", "Watchlist_Setups",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Next_W_VAH": "₹{:.2f}", "Next_W_VAL": "₹{:.2f}", "Next_W_Pivot": "₹{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 6. Narrow Central Pivot Range (CPR < 0.5% Width)
mask_6 = raw_df.get("Setup_Narrow_CPR", False) == True
df_cat6 = raw_df[mask_6].copy().sort_values(by="Next_W_CPR_Width%").reset_index(drop=True)
categories_data[6] = {
    "num": 6,
    "icon": "⚡",
    "name": "Setup 2: Narrow CPR (< 0.5% Width)",
    "group": "Part II: Weekly Pivot & Money Zone Matrix",
    "desc": "Extreme upcoming pivot compression: Upcoming Weekly CPR width is < 0.5% of LTP. Anticipates massive directional expansion upon breakout.",
    "df": df_cat6,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Next_W_CPR_Width%", "Next_W_High_CPR", "Next_W_Low_CPR", "Next_W_Pivot",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Next_W_CPR_Width%": "{:.3f}%", "Next_W_High_CPR": "₹{:.2f}",
        "Next_W_Low_CPR": "₹{:.2f}", "Next_W_Pivot": "₹{:.2f}", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 7. Trend Continuation (Higher Upcoming CPR & POC)
mask_7 = raw_df.get("Setup_Trend_Cont", False) == True
df_cat7 = raw_df[mask_7].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[7] = {
    "num": 7,
    "icon": "📈",
    "name": "Setup 3: Trend Continuation (Higher CPR & POC)",
    "group": "Part II: Weekly Pivot & Money Zone Matrix",
    "desc": "Bullish continuation structure: Next Week's CPR is higher than Current CPR AND Next Week's POC is higher than Current POC. Execution on pullback bounce validated by 15m VWAP.",
    "df": df_cat7,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Next_W_Pivot", "Next_W_High_CPR", "Next_W_Low_CPR", "Next_W_VAH",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Next_W_Pivot": "₹{:.2f}", "Next_W_High_CPR": "₹{:.2f}",
        "Next_W_Low_CPR": "₹{:.2f}", "Next_W_VAH": "₹{:.2f}", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 8. Camarilla Level Alignment (H4 ≈ VAH ±0.2%)
mask_8 = raw_df.get("Setup_Cam_Align", False) == True
df_cat8 = raw_df[mask_8].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[8] = {
    "num": 8,
    "icon": "⚖️",
    "name": "Setup 4: Camarilla Alignment (H4 ≈ VAH ±0.2%)",
    "group": "Part II: Weekly Pivot & Money Zone Matrix",
    "desc": "Confluence alignment: Upcoming Weekly Camarilla H4 level aligns within ±0.2% of Weekly Value Area High (VAH). Execution trigger when LTP crosses H4 + 0.15% buffer.",
    "df": df_cat8,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Next_W_Cam_H4", "Next_W_VAH", "Next_W_VAL", "Next_W_Pivot",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Next_W_Cam_H4": "₹{:.2f}", "Next_W_VAH": "₹{:.2f}",
        "Next_W_VAL": "₹{:.2f}", "Next_W_Pivot": "₹{:.2f}", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 9. Relative Volume Surge (RVOL >= 1.5x)
mask_9 = raw_df.get("Is_RVOL_Valid", False) == True
df_cat9 = raw_df[mask_9].copy().sort_values(by="RVOL", ascending=False).reset_index(drop=True)
categories_data[9] = {
    "num": 9,
    "icon": "📊",
    "name": "Relative Volume Surge (RVOL >= 1.5x of 30D SMA)",
    "group": "Part III: Volume & Order Flow Validations",
    "desc": "High volume expansion: Current volume is >= 1.5x the 30-day historical moving average volume, filtering out false and low-liquidity breakouts.",
    "df": df_cat9,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "RVOL", "Volume_SMA30", "Day_Volume", "RVOL_Tag",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "RVOL": "{:.2f}x", "Volume_SMA30": "{:,}",
        "Day_Volume": "{:,}", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 10. Institutional Delivery Accumulation (> 50%)
mask_10 = raw_df.get("Is_High_Delivery", False) == True
df_cat10 = raw_df[mask_10].copy().sort_values(by="Delivery%", ascending=False).reset_index(drop=True)
categories_data[10] = {
    "num": 10,
    "icon": "📦",
    "name": "Institutional Delivery Accumulation (> 50% Delivery)",
    "group": "Part III: Volume & Order Flow Validations",
    "desc": "Institutional accumulation: Over 50% of the session's volume is marked for delivery transfer on the NSE exchange feed, indicating smart money holding commitment.",
    "df": df_cat10,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Delivery%", "Delivery_Tag", "Day_Volume", "Volume_SMA30",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Delivery%": "{:.1f}%", "Day_Volume": "{:,}",
        "Volume_SMA30": "{:,}", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 11. Derivatives 1D Long Buildup (Price Up + OI Up)
mask_11 = raw_df.get("OI_Buildup", "") == "Long Buildup"
df_cat11 = raw_df[mask_11].copy().sort_values(by="OI_Change%", ascending=False).reset_index(drop=True)
categories_data[11] = {
    "num": 11,
    "icon": "🟢",
    "name": "Derivatives Long Buildup (Price ↑ + OI ↑)",
    "group": "Part IV: Derivatives & Open Interest Quadrants",
    "desc": "Bullish institutional commitment: Asset price is advancing accompanied by positive expansion in Futures Open Interest.",
    "df": df_cat11,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "OI_Change%", "OI_Buildup", "PCR", "PCR_Condition",
        "Spot_Close", "Fut_Close", "Spot_Range_Pos"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "OI_Change%": "{:+.2f}%", "PCR": "{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}", "Spot_Range_Pos": "{:.2f}"
    }
}

# 12. Derivatives 1D Short Covering (Price Up + OI Down)
mask_12 = raw_df.get("OI_Buildup", "") == "Short Covering"
df_cat12 = raw_df[mask_12].copy().sort_values(by="Chg%", ascending=False).reset_index(drop=True)
categories_data[12] = {
    "num": 12,
    "icon": "⚡",
    "name": "Derivatives Short Covering (Price ↑ + OI ↓)",
    "group": "Part IV: Derivatives & Open Interest Quadrants",
    "desc": "Short squeeze momentum: Asset price is advancing while Futures Open Interest declines, showing aggressive unwinding and panic covering by short sellers.",
    "df": df_cat12,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "OI_Change%", "OI_Buildup", "PCR", "PCR_Condition",
        "Spot_Close", "Fut_Close", "Spot_Range_Pos"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "OI_Change%": "{:+.2f}%", "PCR": "{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}", "Spot_Range_Pos": "{:.2f}"
    }
}

# 13. Derivatives 1D Short Buildup (Price Down + OI Up)
mask_13 = raw_df.get("OI_Buildup", "") == "Short Buildup"
df_cat13 = raw_df[mask_13].copy().sort_values(by="OI_Change%", ascending=False).reset_index(drop=True)
categories_data[13] = {
    "num": 13,
    "icon": "🔴",
    "name": "Derivatives Short Buildup (Price ↓ + OI ↑)",
    "group": "Part IV: Derivatives & Open Interest Quadrants",
    "desc": "Bearish institutional pressure: Asset price is declining accompanied by expanding Futures Open Interest, indicating aggressive fresh short positions.",
    "df": df_cat13,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "OI_Change%", "OI_Buildup", "PCR", "PCR_Condition",
        "Spot_Close", "Fut_Close", "Spot_Range_Pos"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "OI_Change%": "{:+.2f}%", "PCR": "{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}", "Spot_Range_Pos": "{:.2f}"
    }
}

# 14. Put-Call Ratio (PCR) Extremes (Bullish Support > 1.2 or Squeeze < 0.6)
mask_14 = (raw_df["PCR"] > 1.2) | (raw_df["PCR"] < 0.6)
df_cat14 = raw_df[mask_14].copy().sort_values(by="PCR", ascending=False).reset_index(drop=True)
categories_data[14] = {
    "num": 14,
    "icon": "⚖️",
    "name": "Put-Call Ratio (PCR) Extremes (Support > 1.2 or Squeeze < 0.6)",
    "group": "Part IV: Derivatives & Open Interest Quadrants",
    "desc": "Options market sentiment bounds: PCR > 1.2 indicates strong put-writing support floor; PCR < 0.6 indicates oversold call-side squeeze setup.",
    "df": df_cat14,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "PCR", "PCR_Condition", "OI_Change%", "OI_Buildup",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "PCR": "{:.2f}", "OI_Change%": "{:+.2f}%",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 15. Exact .70 / .75 Decimal Tick Precision (Spot & Future)
raw_df = ensure_decimal_columns(raw_df)
mask_15 = raw_df["Has_Exact_Decimal_70_75"] if "Has_Exact_Decimal_70_75" in raw_df.columns else pd.Series(False, index=raw_df.index)
df_cat15 = raw_df[mask_15].copy().reset_index(drop=True)
categories_data[15] = {
    "num": 15,
    "icon": "🪙",
    "name": "Exact .70 / .75 Decimal Tick Precision (Spot & Future)",
    "group": "Part V: Strict Decimal Tick Scanner (.70 / .75)",
    "desc": "Strict tick scanner: Displays ONLY stocks whose Spot price or Future price ends in exact .70 or .75 decimal paise (e.g., ₹11049.70 or ₹11049.75), separated across Intraday (High/Low) and Market Closing with 1H opening breakout/breakdown direction.",
    "df": df_cat15,
    "display_cols": [
        "Symbol", "Spot_Close", "Spot_Decimal", "Fut_Close", "Fut_Decimal",
        "Decimal_Match_Tag", "One_Hour_Break_Status", "Chg%",
        "Spot_High", "Spot_Low", "Fut_High", "Fut_Low", "Fut_Basis"
    ],
    "fmt": {
        "Spot_Close": "₹{:.2f}",
        "Fut_Close": "₹{:.2f}",
        "Chg%": "{:+.2f}%",
        "Spot_High": "₹{:.2f}",
        "Spot_Low": "₹{:.2f}",
        "Fut_High": "₹{:.2f}",
        "Fut_Low": "₹{:.2f}",
        "Fut_Basis": "₹{:+.2f}"
    }
}


# =====================================================================
# MAIN PAGE HEADER & CONTROLS
# =====================================================================
st.title("🔮 Money Zone & Derivatives Watchlist Monitor")
st.markdown(
    f"Dedicated list view tracking all **{len(raw_df)}** scanned F&O stocks classified across **15 core Money Zone, "
    f"Breakout, Volume, Derivatives & 0.70/0.75 Range Position categories**."
)

# Top Bar: Symbol Search & Category Jump Filter
c_search, c_jump, c_view_toggle = st.columns([2, 2, 1])
with c_search:
    search_sym = st.text_input("🔍 Search Stock Symbol (e.g., RELIANCE, TCS)", "").strip().upper()

with c_jump:
    jump_options = ["Show All Categories"] + [f"{k}. {v['name']}" for k, v in categories_data.items()]
    selected_jump = st.selectbox("🎯 Category View Focus", jump_options, index=0)

with c_view_toggle:
    st.write("")
    st.write("")
    expand_all = st.checkbox("Expand All Cards", value=True)

# If user is searching for a specific stock, show an instant qualification badge summary!
if search_sym:
    st.markdown(f"#### 🔎 Qualification Report for **{search_sym}**:")
    matched_cats = []
    for k, v in categories_data.items():
        if not v["df"].empty and search_sym in v["df"]["Symbol"].values:
            matched_cats.append((k, v["name"], v["group"]))
    
    if matched_cats:
        st.success(f"✅ **{search_sym}** qualifies in **{len(matched_cats)} of 15** Money Zone categories:")
        cols_qual = st.columns(min(len(matched_cats), 4))
        for idx, (c_num, c_name, c_grp) in enumerate(matched_cats):
            with cols_qual[idx % len(cols_qual)]:
                st.markdown(f"""
                <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid #8B5CF6; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">{c_grp}</div>
                    <div style="font-weight: 700; color: #A78BFA; font-size: 0.92rem;">#{c_num}. {c_name}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info(f"ℹ️ **{search_sym}** was scanned but did not trigger any of the 15 Money Zone setup criteria today.")


# =====================================================================
# TOP KPI RIBBON (ALL CATEGORY COUNTS AT A GLANCE)
# =====================================================================
st.markdown("### 📊 Live Category Summary Ribbon")
kpi_row1 = st.columns(8)
kpi_row2 = st.columns(7)

for i in range(1, 9):
    c_info = categories_data[i]
    count = len(c_info["df"])
    with kpi_row1[i - 1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">#{c_info['num']} {c_info['icon']}</div>
            <div class="metric-val" style="color: {'#A78BFA' if count > 0 else '#94A3B8'};">{count}</div>
            <div class="metric-lbl" style="font-size: 0.65rem;">{c_info['name'][:18]}...</div>
        </div>
        """, unsafe_allow_html=True)

for i in range(9, 16):
    c_info = categories_data[i]
    count = len(c_info["df"])
    with kpi_row2[i - 9]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">#{c_info['num']} {c_info['icon']}</div>
            <div class="metric-val" style="color: {'#A78BFA' if count > 0 else '#94A3B8'};">{count}</div>
            <div class="metric-lbl" style="font-size: 0.65rem;">{c_info['name'][:18]}...</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")


# =====================================================================
# DISPLAY THE CATEGORY WATCHLISTS
# =====================================================================
categories_to_display = []
if selected_jump == "Show All Categories":
    categories_to_display = list(categories_data.keys())
else:
    chosen_num = int(selected_jump.split(".")[0])
    categories_to_display = [chosen_num]

for cat_num in categories_to_display:
    c_info = categories_data[cat_num]
    df_cat = c_info["df"]
    
    # Filter by search symbol if provided
    if search_sym:
        df_cat = df_cat[df_cat["Symbol"].str.contains(search_sym)] if not df_cat.empty else df_cat

    stock_count = len(df_cat)
    badge_html = (
        f'<span class="category-badge-active">✅ {stock_count} Stocks Qualified</span>'
        if stock_count > 0 else
        f'<span class="category-badge-empty">⚪ 0 Stocks</span>'
    )

    # Category Section Container
    st.markdown(f"""
    <div class="category-header-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">{c_info['group']}</span>
                <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-top: 2px;">
                    {c_info['icon']} Category {c_info['num']}: {c_info['name']}
                </div>
            </div>
            <div>
                {badge_html}
            </div>
        </div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 6px;">
            {c_info['desc']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(
        f"📋 View Stock List for Category {c_info['num']} ({stock_count} Stocks)",
        expanded=(expand_all and stock_count > 0)
    ):
        if df_cat.empty:
            st.info(f"⚪ No F&O stocks currently qualify under **Category {c_info['num']}: {c_info['name']}** in the active session.")
        else:
            # Pick available display columns
            avail_cols = [col for col in c_info["display_cols"] if col in df_cat.columns]
            view_sub_df = df_cat[avail_cols].copy()

            # Format dictionary
            fmt = {k: v for k, v in c_info["fmt"].items() if k in view_sub_df.columns}

            h_cat = min(420, max(260, 55 + len(view_sub_df) * 36))
            st.dataframe(
                view_sub_df.style.format(fmt),
                use_container_width=True,
                height=h_cat
            )

            # Export button per category
            cat_csv = view_sub_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Export Category {c_info['num']} CSV",
                data=cat_csv,
                file_name=f"money_zone_cat_{c_info['num']}_{datetime.date.today()}.csv",
                mime="text/csv",
                key=f"dl_mz_cat_{c_info['num']}"
            )

st.markdown("---")

# =====================================================================
# COMBINED EXPORT ALL MONEY ZONE CATEGORIES
# =====================================================================
st.markdown("### 📥 Combined Money Zone Master Export")
st.caption("Download a single unified spreadsheet containing all qualified stocks across all Money Zone categories.")

master_records = []
for k, v in categories_data.items():
    if not v["df"].empty:
        for _, row in v["df"].iterrows():
            master_records.append({
                "Category_Number": k,
                "Category_Name": v["name"],
                "Category_Group": v["group"],
                "Symbol": row.get("Symbol", ""),
                "LTP": row.get("LTP", np.nan),
                "Chg%": row.get("Chg%", np.nan),
                "Spot_Close": row.get("Spot_Close", np.nan),
                "Fut_Close": row.get("Fut_Close", np.nan),
                "Spot_Range_Pos": row.get("Spot_Range_Pos", np.nan),
                "Days_To_Expiry": row.get("Days_To_Expiry", np.nan)
            })

if master_records:
    master_df = pd.DataFrame(master_records)
    c_m1, c_m2 = st.columns([2, 4])
    with c_m1:
        st.download_button(
            label="📥 Download Combined Money Zone CSV",
            data=master_df.to_csv(index=False).encode("utf-8"),
            file_name=f"all_money_zone_watchlist_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c_m2:
        st.write(f"Total qualified instances across all Money Zone categories: **{len(master_df)} records** across **{master_df['Symbol'].nunique()} unique stocks**.")
else:
    st.info("No stocks qualified across any category to export.")
