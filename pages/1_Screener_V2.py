"""
Screener Architecture V2.0: MTF Breakout & Money Zone Screener
Multi-page Streamlit Interface
Target Execution: Indian Equities (NSE) & F&O
"""

import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
    mtf_confluence_filter=mtf_confluence
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

# ----------------- Main View Tabs -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Core MTF Breakouts",
    "📊 Volume & F&O Validations",
    "🔮 Weekly Money Zone Matrix",
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
# TAB 4: INTERACTIVE TECHNICAL CHART
# =====================================================================
with tab4:
    st.markdown("#### 4. Multi-Timeframe Interactive Technical Chart")
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
    st.markdown("""
    This system implements the exact specifications from **Screener Architecture V2.0 (November 2026)**:
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
      - Forward-looking Friday-for-Monday setups:
        1. Volatility Squeeze (Next Week Inside Value)
        2. Narrow CPR ($< 0.5\%$)
        3. Trend Continuation (Higher CPR & POC)
        4. Camarilla Alignment ($H4 \approx VAH \pm 0.2\%$)
    """)

    st.subheader("TradingView Pine Script (v5) Code")
    st.code(PINE_SCRIPT_V5_CODE, language="pinescript")

