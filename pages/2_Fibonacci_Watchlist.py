"""
Multi-Page Screener Module: Category Watchlist (13 Setups)
File: pages/2_Category_Watchlist.py

Displays a dedicated screen showcasing the list of stocks qualified across all
thirteen institutional setup categories:
  Part I: Advanced Fibonacci Setups (Points 1 - 4)
    1. MTF Fibonacci Clusters (1H & 15m)
    2. Fibonacci Extension Breakouts & Downside Breakdowns (1.272 & 1.618 Targets)
    3. Volume Profile + Golden Pocket Confluence (POC & HVNs)
    4. Fibonacci 15m Rejection Wick Scanner (>= 60% Wick)
  Part II: High Beta F&O Confluences (Points 5 - 10)
    5. The 'Golden Pivot' Confluence (Fib + S1/S2/R1/R2 + Value Area)
    6. High Beta Camarilla Reversal Trap (L3/H3 Trap)
    7. Institutional Premium/Discount Oscillator (Extreme ATR Distance)
    8. The Macro Anchor (Monthly MPOC + Weekly Fib)
    9. Weekly CPR Squeeze & Fib Expansion (+1.272/+1.618 or -0.272/-0.618)
    10. Inside Week + Golden Pocket Compression (Close ≈ 0.618 & POC in 30%)
  Part III: High-Conviction Swing Trade Optimizations (Points 11 - 13)
    11. 2-Day Rolling Derivatives OI Buildup (>= 5% Threshold)
    12. 14-Day Relative Strength Leaders (RS Alpha Outperforming vs Index)
    13. Coiled Spring Volatility Compression (NR7 Range Contraction + 120D BB Squeeze)

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
    fetch_intraday_data_v2
)
import swing_optimization_filters
from pwa_launcher import inject_pwa_headers, render_app_launcher_sidebar

# ----------------- Streamlit Page Configuration -----------------
st.set_page_config(
    page_title="Fibonacci Watchlist (13 Setups) | F&O Screener",
    page_icon="📐",
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
        border-left: 5px solid #3B82F6;
        border-top: 1px solid #2A3447;
        border-right: 1px solid #2A3447;
        border-bottom: 1px solid #2A3447;
        border-radius: 8px;
        padding: 12px 18px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .category-badge-active {
        background-color: rgba(16, 185, 129, 0.18);
        color: #10B981;
        border: 1px solid #10B981;
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
st.sidebar.title("📐 Fibonacci Watchlist")
st.sidebar.caption("Page 3: Institutional Fibonacci, Confluence & Swing Setup Watchlists")

# Navigation links to other pages
with st.sidebar.expander("🧭 Quick Page Navigation", expanded=False):
    st.markdown("""
    * 🏛️ **[Classic Screener V1](/)**
    * ⚙️ **[Screener V2.0](/1_Screener_V2)**
    * 📐 **[Fibonacci Watchlist (Current)](/2_Fibonacci_Watchlist)**
    * 🔮 **[Money Zone Watchlist](/3_Money_Zone_Watchlist)**
    """)

# Stock Universe Presets
st.sidebar.subheader("📋 Stock Universe")
universe_choice = st.sidebar.selectbox(
    "Select Universe Preset",
    ["Top 30 Liquid F&O", "Nifty 50", "Bank Nifty", "Full F&O Universe (180+)", "Custom Selection"],
    index=0,
    help="Select the pool of F&O contracts to scan across all 13 categories."
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
    help="Benchmark index for 14-day Relative Strength Alpha (Ret_Stock - Ret_Index)."
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
    if st.button("🔄 Refresh Data", width="stretch"):
        get_intraday_data_cached.clear()
        get_v2_data.clear()
with c_ref2:
    if st.button("⚡ Reset", help="Clear all market caches"):
        st.cache_data.clear()

# ----------------- Data Loading -----------------
with st.spinner(f"Loading market data for {len(selected_symbols)} F&O symbols across all 13 categories..."):
    raw_df, base_meta = get_v2_data(selected_symbols, timeframe, bench_key)

if raw_df.empty:
    st.warning("⚠️ No data was returned. Please verify internet connectivity or symbol list.")
    st.stop()


# =====================================================================
# FILTERING ENGINE: BUILD THE 13 CATEGORY DATAFRAMES
# =====================================================================
categories_data: Dict[int, Dict[str, Any]] = {}

# 1. MTF Fibonacci Cluster (1H & 15m)
mask_1 = raw_df.get("Is_Cluster_Active", False) == True
df_cat1 = raw_df[mask_1].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[1] = {
    "num": 1,
    "icon": "🎯",
    "name": "MTF Fibonacci Cluster (1H & 15m Confluence)",
    "group": "Part I: Advanced Fibonacci",
    "desc": "Institutional support/resistance zones where Macro (1H) and Micro (15m) Fibonacci retracements align within ±0.5%.",
    "df": df_cat1,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Cluster_Level_Price", "Cluster_Labels", "Swing_Type_1H",
        "Spot_Close", "Fut_Close", "Spot_Range_Pos"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Cluster_Level_Price": "₹{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}", "Spot_Range_Pos": "{:.2f}"
    }
}

# 2. Fibonacci Extension Breakouts & Downside Breakdowns (1.272 & 1.618)
mask_2 = raw_df.get("Is_Ext_Breakout", False) == True
df_cat2 = raw_df[mask_2].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[2] = {
    "num": 2,
    "icon": "🚀",
    "name": "Fibonacci Extension Breakouts & Breakdowns (1.272 & 1.618 Targets)",
    "group": "Part I: Advanced Fibonacci",
    "desc": "Projects 1st (1.272) and 2nd (1.618) profit-booking targets for both upside Blue Sky breakouts (> 1.0) and downside breakdown trends (< 0.0) with volume confirmation.",
    "df": df_cat2,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Swing_Type_1H", "Fib_0618_1H", "Ext_Target_1", "Ext_Target_2", "Dist_To_Ext_1%", "Ext_Setup_Tag",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Fib_0618_1H": "₹{:.2f}", "Ext_Target_1": "₹{:.2f}", "Ext_Target_2": "₹{:.2f}",
        "Dist_To_Ext_1%": "{:+.2f}%", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 3. Volume Profile + Golden Pocket Confluence
mask_3 = raw_df.get("Vol_Fib_Confluence_Flag", False) == True
df_cat3 = raw_df[mask_3].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[3] = {
    "num": 3,
    "icon": "📊",
    "name": "Volume Profile + Golden Pocket Confluence",
    "group": "Part I: Advanced Fibonacci",
    "desc": "Flags 0.618 Golden Pocket alignment with Volume Profile Point of Control (POC) or Top 3 High Volume Nodes (HVNs) within ±0.5%.",
    "df": df_cat3,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Fib_0618_1H", "Confluence_Node_Type", "Confluence_Node_Price", "PD_POC",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Fib_0618_1H": "₹{:.2f}", "Confluence_Node_Price": "₹{:.2f}",
        "PD_POC": "₹{:.2f}", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 4. Fibonacci 15m Rejection Wick Scanner (>= 60%)
mask_4 = raw_df.get("Is_Fib_Rejection", False) == True
df_cat4 = raw_df[mask_4].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[4] = {
    "num": 4,
    "icon": "🕯️",
    "name": "Fibonacci Rejection (15m Wick Scanner >= 60%)",
    "group": "Part I: Advanced Fibonacci",
    "desc": "15-minute candlestick anatomy detecting institutional absorption: candle touches 0.500 or 0.618 with lower/upper wick making up >= 60% of candle range.",
    "df": df_cat4,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Rejection_Tag", "Rejection_Level", "Rejection_Wick%",
        "Spot_Close", "Fut_Close", "Spot_Range_Pos"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Rejection_Level": "₹{:.2f}", "Rejection_Wick%": "{:.1f}%",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}", "Spot_Range_Pos": "{:.2f}"
    }
}

# 5. Golden Pivot Confluence (Fib + S1/S2/R1/R2 + Value Area)
mask_5 = (raw_df.get("Is_Golden_Pivot_Bullish", False) == True) | (raw_df.get("Is_Golden_Pivot_Bearish", False) == True)
df_cat5 = raw_df[mask_5].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[5] = {
    "num": 5,
    "icon": "⭐",
    "name": "The 'Golden Pivot' Confluence (Fib + Pivots + Value Area)",
    "group": "Part II: High Beta Confluences",
    "desc": "High-probability reversal setup where 1H Fib 0.618/0.500 aligns within ±0.5% of Daily Support/Resistance Pivots AND Previous Day Value Area (VAL/VAH/POC).",
    "df": df_cat5,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Golden_Pivot_Zone", "Is_Golden_Pivot_Bullish", "Is_Golden_Pivot_Bearish",
        "PD_VAL", "PD_VAH", "Spot_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Golden_Pivot_Zone": "₹{:.2f}",
        "PD_VAL": "₹{:.2f}", "PD_VAH": "₹{:.2f}", "Spot_Close": "₹{:.2f}"
    }
}

# 6. High Beta Camarilla Reversal Trap (L3/H3 Trap)
mask_6 = (raw_df.get("Is_Camarilla_Trap_Bull", False) == True) | (raw_df.get("Is_Camarilla_Trap_Bear", False) == True)
df_cat6 = raw_df[mask_6].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[6] = {
    "num": 6,
    "icon": "⚡",
    "name": "High Beta Camarilla Reversal Trap (L3/H3 Trap)",
    "group": "Part II: High Beta Confluences",
    "desc": "Captures institutional stop hunts: 15m candle breaches Camarilla L3/H3 overlapping with 1H Fib 0.618 / 1.272, then immediately closes back inside the Money Zone.",
    "df": df_cat6,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Camarilla_Trap_Tag", "Camarilla_Trap_Level", "Is_Camarilla_Trap_Bull", "Is_Camarilla_Trap_Bear",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Camarilla_Trap_Level": "₹{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 7. Institutional Premium/Discount Oscillator (Extreme ATR overextension)
mask_7 = (raw_df.get("Is_Prem_Disc_Oversold", False) == True) | (raw_df.get("Is_Prem_Disc_Overbought", False) == True)
df_cat7 = raw_df[mask_7].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[7] = {
    "num": 7,
    "icon": "🌊",
    "name": "Institutional Premium/Discount Oscillator (Extreme)",
    "group": "Part II: High Beta Confluences",
    "desc": "Detects 'rubber band' overextensions where price distance from Daily POC exceeds 2x ATR14 combined with S3/R3 and Fib -0.618 / 1.618 extensions.",
    "df": df_cat7,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Prem_Disc_Status", "Dist_From_POC_ATR", "PD_POC",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Dist_From_POC_ATR": "{:.2f}x", "PD_POC": "₹{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 8. The Macro Anchor (Monthly MPOC + Weekly Fib)
mask_8 = (raw_df.get("Is_Macro_Anchor_Bull", False) == True) | (raw_df.get("Is_Macro_Anchor_Bear", False) == True)
df_cat8 = raw_df[mask_8].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[8] = {
    "num": 8,
    "icon": "⚓",
    "name": "The Macro Anchor (Monthly MPOC + Weekly Fib)",
    "group": "Part II: High Beta Confluences",
    "desc": "Designed for 3-15 day swing holds: deep structural pullbacks into heavy institutional liquidity established at Previous Month POC (MPOC) and Weekly Fib 0.618.",
    "df": df_cat8,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Macro_Anchor_Tag", "Macro_Anchor_Zone", "Is_Macro_Anchor_Bull", "Is_Macro_Anchor_Bear",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Macro_Anchor_Zone": "₹{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 9. Weekly CPR Squeeze & Fib Expansion
mask_9 = raw_df.get("Weekly_CPR_Squeeze_Active", False) == True
df_cat9 = raw_df[mask_9].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[9] = {
    "num": 9,
    "icon": "📐",
    "name": "Weekly CPR Squeeze & Fib Expansion",
    "group": "Part II: High Beta Confluences",
    "desc": "Extreme weekly volatility compression (Weekly CPR < 0.5% width) projecting targets +1.272 / +1.618 on upside breakout, or -0.272 / -0.618 on downside breakdown.",
    "df": df_cat9,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Fib_0618_1H", "Weekly_Fib_Target_1", "Weekly_Fib_Target_2", "Weekly_CPR_Width%",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Fib_0618_1H": "₹{:.2f}", "Weekly_CPR_Width%": "{:.3f}%",
        "Weekly_Fib_Target_1": "₹{:.2f}", "Weekly_Fib_Target_2": "₹{:.2f}",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 10. Inside Week + Golden Pocket Compression
mask_10 = (raw_df.get("Is_Inside_Week_GP_Bull", False) == True) | (raw_df.get("Is_Inside_Week_GP_Bear", False) == True)
df_cat10 = raw_df[mask_10].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[10] = {
    "num": 10,
    "icon": "📦",
    "name": "Inside Week + Golden Pocket Compression",
    "group": "Part II: High Beta Confluences",
    "desc": "Geometric pause setup: Inside Week candle with Close within ±1% of Macro 0.618 Fib, and Weekly POC in the bottom 30% (Accumulation) or top 30% (Distribution).",
    "df": df_cat10,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Inside_Week_GP_Status", "Is_Inside_Week_GP_Bull", "Is_Inside_Week_GP_Bear",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 11. 2-Day Rolling Derivatives OI Buildup (>= 5%)
mask_11 = (raw_df.get("Is_Long_Buildup_2D", False) == True) | (raw_df.get("Is_Short_Buildup_2D", False) == True)
df_cat11 = raw_df[mask_11].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[11] = {
    "num": 11,
    "icon": "📈",
    "name": "2-Day Rolling Derivatives OI Buildup (>= 5%)",
    "group": "Part III: Swing Trade Optimizations",
    "desc": "Filters for genuine institutional commitments where Futures Open Interest expanded by >= 5.0% over a 2-day rolling window with price confirmation.",
    "df": df_cat11,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "OI_2D_Change%", "OI_Buildup_2D", "OI_Change%", "PCR",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "OI_2D_Change%": "{:+.2f}%", "OI_Change%": "{:+.2f}%",
        "PCR": "{:.2f}", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 12. 14-Day Relative Strength Leaders (RS Alpha Outperforming vs Index)
mask_12 = raw_df.get("Is_RS_Outperforming", False) == True
df_cat12 = raw_df[mask_12].copy().sort_values(by="RS_Alpha%", ascending=False).reset_index(drop=True) if "RS_Alpha%" in raw_df.columns else raw_df[mask_12].copy().reset_index(drop=True)
categories_data[12] = {
    "num": 12,
    "icon": "🦁",
    "name": "14-Day Relative Strength Leaders (RS Alpha > 0% vs Index)",
    "group": "Part III: Swing Trade Optimizations",
    "desc": "Isolates market leaders demonstrating true momentum outperformance: 14-Day Stock Return exceeds Benchmark Return (Alpha > 0%).",
    "df": df_cat12,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "RS_Alpha%", "Ret_Stock_14D%", "Ret_Index_14D%", "RS_Status",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "RS_Alpha%": "{:+.2f}%", "Ret_Stock_14D%": "{:+.2f}%",
        "Ret_Index_14D%": "{:+.2f}%", "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}

# 13. Coiled Spring Volatility Compression (NR7 + 120D BB Squeeze)
mask_13 = raw_df.get("Is_Coiled_Spring", False) == True
df_cat13 = raw_df[mask_13].copy().sort_values(by="Symbol").reset_index(drop=True)
categories_data[13] = {
    "num": 13,
    "icon": "🌀",
    "name": "Coiled Spring Volatility Squeeze (NR7 + 120D BB Squeeze)",
    "group": "Part III: Swing Trade Optimizations",
    "desc": "Extreme volatility contraction directly at key levels: True Range is narrowest of past 7 days (NR7) OR Bollinger Band Width is below its 120-day historical average.",
    "df": df_cat13,
    "display_cols": [
        "Symbol", "LTP", "Chg%", "Coiled_Spring_Tag", "Is_NR7", "Is_BB_Squeeze", "BB_Width%", "BB_Width_120_Avg%",
        "Spot_Close", "Fut_Close"
    ],
    "fmt": {
        "LTP": "₹{:.2f}", "Chg%": "{:+.2f}%", "BB_Width%": "{:.2f}%", "BB_Width_120_Avg%": "{:.2f}%",
        "Spot_Close": "₹{:.2f}", "Fut_Close": "₹{:.2f}"
    }
}


# =====================================================================
# MAIN PAGE HEADER & CONTROLS
# =====================================================================
st.title("📐 Fibonacci & Swing Watchlist Monitor")
st.markdown(
    f"Dedicated list view tracking all **{len(raw_df)}** scanned F&O stocks classified across **all 13 Fibonacci, Confluence & Swing setup categories** "
    f"(Advanced Fibonacci, High Beta Confluences, and Swing Trade Optimizations)."
)

# Top Bar: Symbol Search & Category Jump Filter
c_search, c_jump, c_view_toggle = st.columns([2, 2, 1])
with c_search:
    search_sym = st.text_input("🔍 Search Stock Symbol (e.g., RELIANCE, TCS)", "").strip().upper()

with c_jump:
    jump_options = ["Show All 13 Categories"] + [f"{k}. {v['name']}" for k, v in categories_data.items()]
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
        st.success(f"✅ **{search_sym}** qualifies in **{len(matched_cats)} of 13** setup categories:")
        cols_qual = st.columns(min(len(matched_cats), 4))
        for idx, (c_num, c_name, c_grp) in enumerate(matched_cats):
            with cols_qual[idx % len(cols_qual)]:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">{c_grp}</div>
                    <div style="font-weight: 700; color: #10B981; font-size: 0.92rem;">#{c_num}. {c_name}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info(f"ℹ️ **{search_sym}** was scanned but did not trigger any of the 13 strict setup criteria today.")


# =====================================================================
# TOP KPI RIBBON (ALL 13 CATEGORY COUNTS AT A GLANCE)
# =====================================================================
st.markdown("### 📊 Live Category Summary Ribbon")
kpi_row1 = st.columns(7)
kpi_row2 = st.columns(6)

for i in range(1, 8):
    c_info = categories_data[i]
    count = len(c_info["df"])
    with kpi_row1[i - 1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">#{c_info['num']} {c_info['icon']}</div>
            <div class="metric-val" style="color: {'#10B981' if count > 0 else '#94A3B8'};">{count}</div>
            <div class="metric-lbl" style="font-size: 0.65rem;">{c_info['name'][:18]}...</div>
        </div>
        """, unsafe_allow_html=True)

for i in range(8, 14):
    c_info = categories_data[i]
    count = len(c_info["df"])
    with kpi_row2[i - 8]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">#{c_info['num']} {c_info['icon']}</div>
            <div class="metric-val" style="color: {'#10B981' if count > 0 else '#94A3B8'};">{count}</div>
            <div class="metric-lbl" style="font-size: 0.65rem;">{c_info['name'][:18]}...</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")


# =====================================================================
# DISPLAY THE 13 CATEGORY WATCHLISTS
# =====================================================================
categories_to_display = []
if selected_jump == "Show All 13 Categories":
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
                width="stretch",
                height=h_cat
            )

            # Export button per category
            cat_csv = view_sub_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Export Category {c_info['num']} CSV",
                data=cat_csv,
                file_name=f"category_{c_info['num']}_{datetime.date.today()}.csv",
                mime="text/csv",
                key=f"dl_cat_{c_info['num']}"
            )

st.markdown("---")

# =====================================================================
# COMBINED EXPORT ALL 13 CATEGORIES
# =====================================================================
st.markdown("### 📥 Combined 13-Category Master Export")
st.caption("Download a single unified spreadsheet containing all qualified stocks mapped with their respective category tags.")

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
            label="📥 Download Combined All-Categories CSV",
            data=master_df.to_csv(index=False).encode("utf-8"),
            file_name=f"all_13_categories_watchlist_{datetime.date.today()}.csv",
            mime="text/csv",
            width="stretch"
        )
    with c_m2:
        st.write(f"Total qualified instances across all 13 categories: **{len(master_df)} records** across **{master_df['Symbol'].nunique()} unique stocks**.")
else:
    st.info("No stocks qualified across any category to export.")
