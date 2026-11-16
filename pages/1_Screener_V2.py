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
import screener_v2_engine
import fibonacci_screener_v2
import mtf_breakout

importlib.reload(fo_stocks)
importlib.reload(screener_v2_engine)
importlib.reload(fibonacci_screener_v2)
importlib.reload(mtf_breakout)

from fo_stocks import TOP_30_LIQUID, NIFTY_50, BANK_NIFTY, get_fo_universe
from screener_v2_engine import (
    fetch_universe_data_v2,
    apply_screener_v2_filters,
    calculate_mtf_v2
)
from mtf_breakout import PINE_SCRIPT_V5_CODE

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Screener Architecture V2.0 | MTF & Money Zone",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Cache data loading
@st.cache_data(ttl=90, show_spinner=False)
def get_v2_data(syms, tf):
    return fetch_universe_data_v2(syms, timeframe=tf)

# Refresh Button
if st.sidebar.button("🔄 Refresh Market Data", use_container_width=True):
    st.cache_data.clear()

# ----------------- Data Ingestion -----------------
with st.spinner(f"Scanning market data for {len(selected_symbols)} F&O symbols (Architecture V2.0)..."):
    raw_df, base_meta = get_v2_data(selected_symbols, timeframe)

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
    fib_confluence_filter=fib_confluence_filter
)

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

# ----------------- Main View Tabs -----------------
tab1, tab2, tab3, tab_fib, tab4, tab5 = st.tabs([
    "🎯 Core MTF Breakouts",
    "📊 Volume & F&O Validations",
    "🔮 Weekly Money Zone Matrix",
    "📐 Advanced Fibonacci & Confluence",
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
    st.caption("Filters out false breakouts using Relative Volume (RVOL ≥ 1.5), Delivery % > 50%, OI Buildup Quadrants, and Put-Call Ratio (PCR).")

    fo_cols = [
        "Symbol", "LTP", "Chg%", "RVOL", "Volume_SMA30", "RVOL_Tag", 
        "Delivery%", "Delivery_Tag", "OI_Change%", "OI_Buildup", "PCR", "PCR_Condition"
    ]
    view_fo_df = all_df[fo_cols].copy() if not all_df.empty else pd.DataFrame(columns=fo_cols)

    st.dataframe(
        view_fo_df.style.format({
            "LTP": "₹{:.2f}",
            "Chg%": "{:+.2f}%",
            "RVOL": "{:.2f}x",
            "Volume_SMA30": "{:,}",
            "Delivery%": "{:.1f}%",
            "OI_Change%": "{:+.2f}%",
            "PCR": "{:.2f}"
        }),
        use_container_width=True,
        height=480
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
            "Is_Golden_Pocket_Confluence", "Is_Golden_Pocket", "Is_Golden_Zone_Pullback",
            "Is_Cluster_Active", "Cluster_Level_Price", "Cluster_Labels",
            "Is_Ext_Breakout", "Ext_Target_1", "Ext_Target_2", "Dist_To_Ext_1%", "Ext_Setup_Tag",
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

        st.dataframe(p1_df, use_container_width=True)

    with f_sub2:
        st.markdown("##### Part II: Golden Pivots, Camarilla Traps, Prem/Disc Oscillator & Macro Swings")
        p2_cols = [
            "Symbol", "LTP", "Chg%",
            "Is_Golden_Pivot_Bullish", "Is_Golden_Pivot_Bearish", "Golden_Pivot_Zone",
            "Is_Camarilla_Trap_Bull", "Is_Camarilla_Trap_Bear", "Camarilla_Trap_Level", "Camarilla_Trap_Tag",
            "Prem_Disc_Status", "Dist_From_POC_ATR",
            "Is_Macro_Anchor_Bull", "Is_Macro_Anchor_Bear", "Macro_Anchor_Zone", "Macro_Anchor_Tag",
            "Weekly_CPR_Squeeze_Active", "Weekly_Fib_Exp_Bull", "Weekly_Fib_Exp_Bear", "Weekly_Fib_Target_1", "Weekly_Fib_Target_2",
            "Is_Inside_Week_GP_Bull", "Is_Inside_Week_GP_Bear", "Inside_Week_GP_Status"
        ]
        available_p2_cols = [c for c in p2_cols if c in all_df.columns]
        p2_df = all_df[available_p2_cols].copy()

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        col_k1.metric("Golden Pivots", int((all_df.get("Is_Golden_Pivot_Bullish", False) | all_df.get("Is_Golden_Pivot_Bearish", False)).sum()))
        col_k2.metric("Camarilla Traps", int((all_df.get("Is_Camarilla_Trap_Bull", False) | all_df.get("Is_Camarilla_Trap_Bear", False)).sum()))
        col_k3.metric("Macro Anchor Swings", int((all_df.get("Is_Macro_Anchor_Bull", False) | all_df.get("Is_Macro_Anchor_Bear", False)).sum()))
        col_k4.metric("Inside Week @ 0.618", int((all_df.get("Is_Inside_Week_GP_Bull", False) | all_df.get("Is_Inside_Week_GP_Bear", False)).sum()))

        st.dataframe(p2_df, use_container_width=True)

    with f_sub3:
        st.markdown("##### Full Universe Advanced Fibonacci Monitor")
        sym_search = st.text_input("🔍 Search Symbol in Fibonacci Screener", "").strip().upper()
        display_fib_df = all_df
        if sym_search:
            display_fib_df = display_fib_df[display_fib_df["Symbol"].str.contains(sym_search)]

        st.dataframe(display_fib_df, use_container_width=True)

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
        2. **Fibonacci Extension Breakouts (1.272 & 1.618):**
           - Targets for Blue Sky setups breaking out beyond 1.0 (Swing High) with Volume confirmation.
           - `Ext 1 = Swing_Low + (Range × 1.272)`, `Ext 2 = Swing_Low + (Range × 1.618)`.
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
# TAB 4: INTERACTIVE TECHNICAL CHART
# =====================================================================
with tab4:
    st.markdown("#### 5. Multi-Timeframe Interactive Technical Chart & Fibonacci Overlay")
    chart_sym = st.selectbox("Select Symbol for Deep Dive", options=selected_symbols, index=0)

    try:
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

