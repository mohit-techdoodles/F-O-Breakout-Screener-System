"""
NSE F&O Breakout Screener (VAH/VAL, CPR Money Zone, Camarilla Pivots & Bollinger Squeeze)
Streamlit Application
"""

import datetime
import importlib
import inspect
import time
import pandas as pd
import streamlit as st

import fo_stocks
import screener_engine
import chart_utils
import bollinger_screener
import volume_profile
import quantitative_filters
import mtf_breakout

importlib.reload(fo_stocks)
importlib.reload(screener_engine)
importlib.reload(chart_utils)
importlib.reload(bollinger_screener)
importlib.reload(volume_profile)
importlib.reload(quantitative_filters)
importlib.reload(mtf_breakout)

from fo_stocks import VERIFIED_FO_STOCKS, NIFTY_50, BANK_NIFTY, TOP_30_LIQUID, get_fo_universe
from screener_engine import fetch_universe_data, apply_screener_filters, fetch_and_screen, get_single_stock_chart_data
from chart_utils import create_breakout_chart
from mtf_breakout import PINE_SCRIPT_V5_CODE


def get_width_kw(fn):
    """Seamlessly handles Streamlit's transition from use_container_width=True to width='stretch'"""
    try:
        return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


# Page configuration
st.set_page_config(
    page_title="F&O Breakout & Bollinger Squeeze Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #262b36 100%);
        border: 1px solid #363c4e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-lbl {
        color: #9aa0a6;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Setup
st.sidebar.title("⚡ Screener Controls")

# Switch to Screener V2.0 Navigation Link
st.sidebar.info("✨ **Looking for Screener V2.0?**\n\nTry the new MTF & Money Zone Screener V2.0:")
try:
    st.sidebar.page_link("pages/1_Screener_V2.py", label="🚀 Launch Screener Architecture V2.0", icon="🎯")
except Exception:
    st.sidebar.markdown("[🚀 **Launch Screener Architecture V2.0**](/Screener_V2)")
st.sidebar.markdown("---")

# Universe Selection
universe_option = st.sidebar.selectbox(
    "Select Stock Universe",
    ["All F&O (~180 Stocks)", "Nifty 50", "Bank Nifty", "Top 30 Liquid F&O", "Custom Watchlist"],
    index=0
)

all_universe_symbols = get_fo_universe()

if universe_option == "All F&O (~180 Stocks)":
    selected_symbols = all_universe_symbols
elif universe_option == "Nifty 50":
    selected_symbols = NIFTY_50
elif universe_option == "Bank Nifty":
    selected_symbols = BANK_NIFTY
elif universe_option == "Top 30 Liquid F&O":
    selected_symbols = TOP_30_LIQUID
else:
    selected_symbols = st.sidebar.multiselect(
        "Choose Stocks",
        options=all_universe_symbols,
        default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"]
    )

# Timeframe Selection
timeframe = st.sidebar.selectbox("Intraday Candle Timeframe", ["5m", "15m"], index=0)

# Proximity Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Proximity Triggers")
high_proximity = st.sidebar.slider(
    "52-Week High Proximity (%)", 
    min_value=90, max_value=99, value=95, step=1,
    help="Default 95% means LTP must be within 5% of the 52-Week High"
) / 100.0

low_proximity = st.sidebar.slider(
    "52-Week Low Proximity (%)", 
    min_value=101, max_value=110, value=105, step=1,
    help="Default 105% means LTP must be within 5% of the 52-Week Low"
) / 100.0

ignore_52w = st.sidebar.checkbox(
    "🔓 Irrespective of 52-Week High/Low (Show All Breakouts)",
    value=False,
    help="When enabled, bypasses the 52-Week High/Low proximity requirement so all valid breakouts across the universe are shown."
)

# Pivot & Money Zone Criteria
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Pivot & Money Zone Criteria")

enable_cpr = st.sidebar.checkbox(
    "Require CPR Filter (Money Zone)",
    value=True,
    help="Bullish Setup: LTP must be above Previous Day CPR Top.\nBearish Setup: LTP must be below Previous Day CPR Bottom."
)

only_narrow_cpr = st.sidebar.checkbox(
    "🎯 Only Narrow Daily CPR (Trending Setup)",
    value=False,
    help="Filter stocks with narrow CPR width, which signals an explosive breakout or high-probability trending session."
)

narrow_cpr_max = st.sidebar.slider(
    "Max Narrow CPR Width (%)",
    min_value=0.10,
    max_value=1.00,
    value=0.35,
    step=0.05,
    help="CPR Width % = |TC - BC| / Pivot * 100.\n🎯 Ultra Narrow: <= 0.15%\n⚡ Narrow: <= 0.35%\nAverage: 0.35% - 0.70%\nWide: > 0.70%"
)

camarilla_mode = st.sidebar.selectbox(
    "Camarilla Pivot Criteria",
    ["None", "H4 / L4 Breakout (Strong)", "H3 / L3 Range Breakout"],
    index=0,
    help="H4/L4: LTP > H4 for Bullish or < L4 for Bearish.\nH3/L3: LTP > H3 for Bullish or < L3 for Bearish."
)

# Bollinger Bands Squeeze & Breakout Filter
st.sidebar.markdown("---")
st.sidebar.subheader("🌀 Bollinger Squeeze & Breakout")

bollinger_filter = st.sidebar.selectbox(
    "Bollinger Bands Filter",
    [
        "None", 
        "Squeeze Breakout (Master Confluence)", 
        "Currently Squeezing (Coil)", 
        "Price Outside Bands (Upper/Lower)"
    ],
    index=0,
    help="Squeeze Breakout: Squeeze in recent sessions + Close pushing beyond Upper/Lower Band."
)

# Quantitative Filters (RVOL & BBW Contraction - Technical Feature Spec)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quantitative Filters")

enable_rvol = st.sidebar.checkbox(
    "Enable RVOL Filter",
    value=False,
    help="Filters for stocks experiencing institutional volume anomalies against their 20-day average."
)
rvol_threshold = 1.0
if enable_rvol:
    rvol_threshold = st.sidebar.slider(
        "Minimum RVOL Multiplier",
        min_value=1.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Filters for volume X times the 20-day average volume."
    )

enable_bbw_squeeze = st.sidebar.checkbox(
    "Require Volatility Contraction",
    value=False,
    help="Checks if Bollinger Bandwidth was tighter than its 20-day average on the prior session (T-1) before the breakout."
)

enable_weekly = st.sidebar.checkbox(
    "Require Weekly Context Alignment",
    value=False,
    help="Section 8: Bullish requires LTP > Weekly VAH & Bullish/Inside Bias. Bearish requires LTP < Weekly VAL & Bearish/Inside Bias."
)

enable_nr7 = st.sidebar.checkbox(
    "Require NR7 Volatility Compression",
    value=False,
    help="Section 5: Filters for stocks where today or yesterday recorded the narrowest daily range of the last 7 sessions."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 MTF & Initial Hour (1H) Filters")
enable_mtf = st.sidebar.checkbox(
    "Require MTF Breakout (1H & VWAP)",
    value=False,
    help="Developer Guide MTF Algorithm: Bullish requires 1H Close (10:15) > PDH & LTP > VWAP. Bearish requires 1H Close < PDL & LTP < VWAP."
)
mtf_mode = "Any / Both"
if enable_mtf:
    mtf_mode = st.sidebar.selectbox(
        "MTF Filter Direction",
        ["Any / Both", "Bullish (1H > PDH & Above VWAP)", "Bearish (1H < PDL & Below VWAP)"],
        index=0
    )

h1_filter = st.sidebar.selectbox(
    "Initial Hour (1H) Breakout / Break-in",
    [
        "All / Off",
        # Option A: 1H Close vs Prev 1H Close
        "Option A: Breakout > Prev 1H (Bullish)",
        "Option A: Breakdown < Prev 1H (Bearish)",
        "Option A: Break-in (Inside Prev 1H)",
        # Option B: Live Price vs 1H Close
        "Option B: Price > 1H Close (Bullish)",
        "Option B: Price < 1H Close (Bearish)",
        "Option B: Break-in (Failed 1H Close Breakout)",
        # Option C: 1H Range (Initial Balance High/Low)
        "Option C: Range Breakout > 1H High (Bullish)",
        "Option C: Range Breakdown < 1H Low (Bearish)",
        "Option C: Range Break-in (Failed Breakout / Re-entered Range)"
    ],
    index=0,
    help="Select Option A (1H Close vs Prev 1H Close), Option B (Live Price vs 1H Close), or Option C (Initial Balance 1H Range Breakout or Break-in)."
)

# Auto-refresh
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Live Refresh")
auto_refresh = st.sidebar.selectbox("Auto Refresh Interval", ["Off", "30 Seconds", "60 Seconds", "120 Seconds"], index=0)

refresh_clicked = st.sidebar.button("🔄 Scan Now", **get_width_kw(st.button))

# Helper function to cache raw market data & indicator calculations
# Heavy network fetch + indicator computation is cached by symbols & timeframe.
# Vectorized filters are applied dynamically in memory in < 1 millisecond.
@st.cache_data(ttl=90, show_spinner=False)
def get_cached_raw_data(syms, tf):
    return fetch_universe_data(syms, timeframe=tf)


# Header & Market Status (Explicitly in Indian Standard Time - IST)
try:
    from zoneinfo import ZoneInfo
    ist_tz = ZoneInfo("Asia/Kolkata")
    now_ist = datetime.datetime.now(ist_tz)
except Exception:
    import pytz
    ist_tz = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.datetime.now(ist_tz)

market_open = (
    now_ist.weekday() < 5 and 
    datetime.time(9, 15) <= now_ist.time() <= datetime.time(15, 30)
)

col_title, col_time = st.columns([3, 1])
with col_title:
    st.title("📈 F&O Breakout & Volatility Screener")
    st.caption("Strategy: Volume Profile (VAH/VAL) + CPR Money Zone + Camarilla + Bollinger Bands Squeeze + MTF Breakout")

with col_time:
    status_color = "#00E676" if market_open else "#FF9100"
    status_text = "🟢 MARKET LIVE" if market_open else "🟠 MARKET CLOSED"
    st.markdown(f"""
        <div style="text-align: right; padding-top: 10px;">
            <span style="font-weight: 700; color: {status_color}; font-size: 15px;">{status_text}</span><br>
            <span style="color: #9aa0a6; font-size: 13px;">IST Time: {now_ist.strftime('%d-%b-%Y %H:%M:%S')}</span>
        </div>
    """, unsafe_allow_html=True)

# Execute Screening with progress feedback
if refresh_clicked:
    st.cache_data.clear()

with st.spinner(f"Fetching market data for {len(selected_symbols)} F&O stocks..."):
    master_df, base_meta = get_cached_raw_data(selected_symbols, timeframe)

if master_df.empty:
    st.warning("⚠️ No data was returned. Please verify your connection or refresh.")
    st.stop()

# Instant in-memory filter execution (< 1ms)
all_df, bull_df, bear_df, meta = apply_screener_filters(
    master_df,
    base_meta,
    high_proximity_pct=high_proximity,
    low_proximity_pct=low_proximity,
    ignore_52w=ignore_52w,
    h1_close_filter=h1_filter,
    enable_cpr_filter=enable_cpr,
    only_narrow_cpr=only_narrow_cpr,
    narrow_cpr_max_pct=narrow_cpr_max,
    camarilla_filter=camarilla_mode,
    bollinger_filter=bollinger_filter,
    enable_rvol=enable_rvol,
    rvol_threshold=rvol_threshold,
    enable_bbw_squeeze=enable_bbw_squeeze,
    enable_weekly_filter=enable_weekly,
    enable_nr7_filter=enable_nr7,
    enable_mtf_filter=enable_mtf,
    mtf_mode=mtf_mode
)

# Extract squeezing & MTF stocks
squeeze_df = all_df[all_df["BB_Is_Squeezing"]].sort_values(by="BB_Bandwidth%", ascending=True).copy()
mtf_bull_df = all_df[all_df["MTF_Bullish"]].sort_values(by="Dist_VWAP%", ascending=False).copy()
mtf_bear_df = all_df[all_df["MTF_Bearish"]].sort_values(by="Dist_VWAP%", ascending=True).copy()
mtf_total_count = len(mtf_bull_df) + len(mtf_bear_df)

# Top Metric Cards Row
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Scanned</div>
        <div class="metric-val" style="color: #64B5F6;">{meta.get('total_scanned', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Bullish Breakouts</div>
        <div class="metric-val" style="color: #00E676;">{meta.get('bullish_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Bearish Breakdowns</div>
        <div class="metric-val" style="color: #FF5252;">{meta.get('bearish_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">BB Squeezing (Coil)</div>
        <div class="metric-val" style="color: #AB47BC;">{len(squeeze_df)}</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Tomorrow Narrow CPR</div>
        <div class="metric-val" style="color: #FFB300;">{meta.get('next_narrow_cpr_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m6:
    ref_d = meta.get('ref_date', 'N/A')
    live_d = meta.get('live_date', 'N/A')
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Active Session</div>
        <div class="metric-val" style="font-size: 13px; color: #FFF; padding-top: 6px;">
            Ref: {ref_d}<br>
            <span style="color: #00E676; font-size: 12px;">Live: {live_d}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Secondary Metric Cards (Quantitative & MTF Overview)
q1, q2, q3, q4, q5 = st.columns(5)
with q1:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 3px solid #00E676;">
        <div class="metric-lbl">🎯 MTF Bullish (1H > PDH)</div>
        <div class="metric-val" style="color: #00E676; font-size: 20px;">{meta.get('mtf_bull_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with q2:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 3px solid #FF5252;">
        <div class="metric-lbl">🎯 MTF Bearish (1H < PDL)</div>
        <div class="metric-val" style="color: #FF5252; font-size: 20px;">{meta.get('mtf_bear_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with q3:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 3px solid #FFA726;">
        <div class="metric-lbl">🔥 High RVOL (≥ 1.5x)</div>
        <div class="metric-val" style="color: #FFA726; font-size: 20px;">{meta.get('rvol_pass_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with q4:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 3px solid #29B6F6;">
        <div class="metric-lbl">⚡ Prior BBW Contracted</div>
        <div class="metric-val" style="color: #29B6F6; font-size: 20px;">{meta.get('bbw_contract_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with q5:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 3px solid #AB47BC;">
        <div class="metric-lbl">📦 NR7 Coiled (Today)</div>
        <div class="metric-val" style="color: #AB47BC; font-size: 20px;">{meta.get('nr7_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab_mtf, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    f"🚀 Bullish Breakouts ({len(bull_df)})", 
    f"🔻 Bearish Breakdowns ({len(bear_df)})", 
    f"🎯 MTF Breakouts ({mtf_total_count})",
    f"🌅 Tomorrow's CPR Watchlist ({meta.get('next_narrow_cpr_count', 0)})",
    f"🌀 Squeeze Radar ({len(squeeze_df)})",
    "📈 Interactive Chart & Visualizer", 
    f"📋 All F&O Monitor ({len(all_df)})",
    "ℹ️ Strategy & Documentation"
])

# ----------------- TAB 1: BULLISH BREAKOUTS -----------------
with tab1:
    cpr_rule_txt = "✅ **CPR Money Zone:** `LTP > PD_CPR_Top`" if enable_cpr else "⚪ CPR Money Zone filter is optional/off"
    narrow_rule_txt = f"✅ **Narrow CPR Filter:** Active (≤ {narrow_cpr_max:.2f}%)" if only_narrow_cpr else "⚪ Narrow CPR filter is optional/off"
    rvol_rule_txt = f"✅ **RVOL Filter:** Active (≥ {rvol_threshold:.1f}x)" if enable_rvol else "⚪ RVOL filter is optional/off"
    bbw_rule_txt = "✅ **BBW Contraction:** Required on T-1" if enable_bbw_squeeze else "⚪ BBW Contraction is optional/off"
    weekly_rule_txt = "✅ **Weekly Structure Alignment:** `LTP > W_VAH` & Bullish/Inside Bias" if enable_weekly else "⚪ Weekly structure filter is optional/off"
    nr7_rule_txt = "✅ **NR7 Volatility Compression:** Required on T-1 or T0" if enable_nr7 else "⚪ NR7 compression filter is optional/off"
    cam_rule_txt = f"✅ **Camarilla:** Active ({camarilla_mode})" if camarilla_mode != "None" else "⚪ Camarilla filter is optional/off"
    bb_rule_txt = f"✅ **Bollinger Filter:** Active ({bollinger_filter})" if bollinger_filter != "None" else "⚪ Bollinger filter is optional/off"
    mtf_rule_txt = f"✅ **MTF Breakout Filter:** Active ({mtf_mode})" if enable_mtf else "⚪ MTF Breakout filter is optional/off"
    
    st.markdown(f"""
    > **Bullish Setup Confluence Rules:**  
    > 1. **Value Area Breakout:** `LTP > PD_VAH` (Breaking out above yesterday's 70% value area)  
    > 2. **52-Week High Proximity:** `LTP >= 52W_High * {int(high_proximity*100)}%` (Trading within {100-int(high_proximity*100)}% of 52-Week High)  
    > 3. {cpr_rule_txt}  
    > 4. {narrow_rule_txt}  
    > 5. {rvol_rule_txt}  
    > 6. {bbw_rule_txt}  
    > 7. {weekly_rule_txt}  
    > 8. {nr7_rule_txt}  
    > 9. {cam_rule_txt}  
    > 10. {bb_rule_txt}  
    > 11. {mtf_rule_txt}  
    """)
    
    if bull_df.empty:
        st.info("ℹ️ No F&O stocks currently meet all active Bullish Breakout criteria. Watch for emerging breakouts above yesterday's VAH, CPR, and Bollinger Bands.")
    else:
        st.success(f"🎯 Found **{len(bull_df)}** Bullish Breakout candidate(s)!")
        
        display_bull = bull_df[[
            "Symbol", "LTP", "Chg%", "MTF_Signal", "First_1H_Close", "RVOL", "BBW_Contract_Status", "NR7_Status", "W_CPR_Bias", "W_VAH", "Dist_W_VAH%", 
            "PD_VAH", "Dist_VAH%", "52W_High", "Dist_52W_High%", 
            "CPR_Status", "CPR_Type", "PD_CPR_Width%", "PD_CPR_Top", "Cam_Status", "BB_Status", "BB_Bandwidth%", "PD_POC"
        ]].copy()
        
        display_bull = display_bull.rename(columns={
            "LTP": "LTP (₹)",
            "Chg%": "Day Chg %",
            "MTF_Signal": "MTF Signal",
            "First_1H_Close": "1H Close (10:15) ₹",
            "RVOL": "RVOL (20D)",
            "BBW_Contract_Status": "BBW Contraction",
            "NR7_Status": "NR7 Status",
            "W_CPR_Bias": "Weekly Bias",
            "W_VAH": "Weekly VAH (₹)",
            "Dist_W_VAH%": "% vs W_VAH",
            "PD_VAH": "Yesterday VAH (₹)",
            "Dist_VAH%": "% Above VAH",
            "52W_High": "52W High (₹)",
            "Dist_52W_High%": "% From 52W High",
            "CPR_Status": "CPR Money Zone",
            "CPR_Type": "CPR Width Type",
            "PD_CPR_Width%": "CPR Width %",
            "PD_CPR_Top": "CPR Top (₹)",
            "Cam_Status": "Camarilla Status",
            "BB_Status": "Bollinger Bands Status",
            "BB_Bandwidth%": "BB Bandwidth %",
            "PD_POC": "Yesterday POC (₹)"
        })
        
        styler = display_bull.style.format({
            "LTP (₹)": "₹{:.2f}",
            "Day Chg %": "{:+.2f}%",
            "1H Close (10:15) ₹": "₹{:.2f}",
            "RVOL (20D)": "{:.2f}x",
            "Weekly VAH (₹)": "₹{:.2f}",
            "% vs W_VAH": "{:+.2f}%",
            "Yesterday VAH (₹)": "₹{:.2f}",
            "% Above VAH": "+{:.2f}%",
            "52W High (₹)": "₹{:.2f}",
            "% From 52W High": "{:+.2f}%",
            "CPR Width %": "{:.3f}%",
            "CPR Top (₹)": "₹{:.2f}",
            "BB Bandwidth %": "{:.2f}%",
            "Yesterday POC (₹)": "₹{:.2f}"
        })
        
        style_fn = getattr(styler, "map", getattr(styler, "applymap", None))
        if style_fn:
            styler = style_fn(
                lambda v: "color: #00E676; font-weight: bold;", 
                subset=["% Above VAH", "% From 52W High"]
            )
        
        st.dataframe(
            styler,
            **get_width_kw(st.dataframe),
            hide_index=True
        )


# ----------------- TAB 2: BEARISH BREAKDOWNS -----------------
with tab2:
    cpr_rule_txt = "✅ **CPR Money Zone:** `LTP < PD_CPR_Bottom`" if enable_cpr else "⚪ CPR Money Zone filter is optional/off"
    narrow_rule_txt = f"✅ **Narrow CPR Filter:** Active (≤ {narrow_cpr_max:.2f}%)" if only_narrow_cpr else "⚪ Narrow CPR filter is optional/off"
    rvol_rule_txt = f"✅ **RVOL Filter:** Active (≥ {rvol_threshold:.1f}x)" if enable_rvol else "⚪ RVOL filter is optional/off"
    bbw_rule_txt = "✅ **BBW Contraction:** Required on T-1" if enable_bbw_squeeze else "⚪ BBW Contraction is optional/off"
    weekly_rule_txt = "✅ **Weekly Structure Alignment:** `LTP < W_VAL` & Bearish/Inside Bias" if enable_weekly else "⚪ Weekly structure filter is optional/off"
    nr7_rule_txt = "✅ **NR7 Volatility Compression:** Required on T-1 or T0" if enable_nr7 else "⚪ NR7 compression filter is optional/off"
    cam_rule_txt = f"✅ **Camarilla:** Active ({camarilla_mode})" if camarilla_mode != "None" else "⚪ Camarilla filter is optional/off"
    bb_rule_txt = f"✅ **Bollinger Filter:** Active ({bollinger_filter})" if bollinger_filter != "None" else "⚪ Bollinger filter is optional/off"
    mtf_rule_txt = f"✅ **MTF Breakout Filter:** Active ({mtf_mode})" if enable_mtf else "⚪ MTF Breakout filter is optional/off"

    st.markdown(f"""
    > **Bearish Setup Confluence Rules:**  
    > 1. **Value Area Breakdown:** `LTP < PD_VAL` (Breaking down below yesterday's 70% value area)  
    > 2. **52-Week Low Proximity:** `LTP <= 52W_Low * {int(low_proximity*100)}%` (Trading within {int(low_proximity*100)-100}% of 52-Week Low)  
    > 3. {cpr_rule_txt}  
    > 4. {narrow_rule_txt}  
    > 5. {rvol_rule_txt}  
    > 6. {bbw_rule_txt}  
    > 7. {weekly_rule_txt}  
    > 8. {nr7_rule_txt}  
    > 9. {cam_rule_txt}  
    > 10. {bb_rule_txt}  
    > 11. {mtf_rule_txt}  
    """)
    
    if bear_df.empty:
        st.info("ℹ️ No F&O stocks currently meet all active Bearish Breakdown criteria.")
    else:
        st.error(f"🔻 Found **{len(bear_df)}** Bearish Breakdown candidate(s)!")
        
        display_bear = bear_df[[
            "Symbol", "LTP", "Chg%", "MTF_Signal", "First_1H_Close", "RVOL", "BBW_Contract_Status", "NR7_Status", "W_CPR_Bias", "W_VAL", "Dist_W_VAL%",
            "PD_VAL", "Dist_VAL%", "52W_Low", "Dist_52W_Low%", 
            "CPR_Status", "CPR_Type", "PD_CPR_Width%", "PD_CPR_Bottom", "Cam_Status", "BB_Status", "BB_Bandwidth%", "PD_POC"
        ]].copy()
        
        display_bear = display_bear.rename(columns={
            "LTP": "LTP (₹)",
            "Chg%": "Day Chg %",
            "MTF_Signal": "MTF Signal",
            "First_1H_Close": "1H Close (10:15) ₹",
            "RVOL": "RVOL (20D)",
            "BBW_Contract_Status": "BBW Contraction",
            "NR7_Status": "NR7 Status",
            "W_CPR_Bias": "Weekly Bias",
            "W_VAL": "Weekly VAL (₹)",
            "Dist_W_VAL%": "% vs W_VAL",
            "PD_VAL": "Yesterday VAL (₹)",
            "Dist_VAL%": "% Below VAL",
            "52W_Low": "52W Low (₹)",
            "Dist_52W_Low%": "% From 52W Low",
            "CPR_Status": "CPR Money Zone",
            "CPR_Type": "CPR Width Type",
            "PD_CPR_Width%": "CPR Width %",
            "PD_CPR_Bottom": "CPR Bottom (₹)",
            "Cam_Status": "Camarilla Status",
            "BB_Status": "Bollinger Bands Status",
            "BB_Bandwidth%": "BB Bandwidth %",
            "PD_POC": "Yesterday POC (₹)"
        })
        
        styler = display_bear.style.format({
            "LTP (₹)": "₹{:.2f}",
            "Day Chg %": "{:+.2f}%",
            "1H Close (10:15) ₹": "₹{:.2f}",
            "RVOL (20D)": "{:.2f}x",
            "Weekly VAL (₹)": "₹{:.2f}",
            "% vs W_VAL": "{:+.2f}%",
            "Yesterday VAL (₹)": "₹{:.2f}",
            "% Below VAL": "{:.2f}%",
            "52W Low (₹)": "₹{:.2f}",
            "% From 52W Low": "{:+.2f}%",
            "CPR Width %": "{:.3f}%",
            "CPR Bottom (₹)": "₹{:.2f}",
            "BB Bandwidth %": "{:.2f}%",
            "Yesterday POC (₹)": "₹{:.2f}"
        })
        
        style_fn = getattr(styler, "map", getattr(styler, "applymap", None))
        if style_fn:
            styler = style_fn(
                lambda v: "color: #FF5252; font-weight: bold;", 
                subset=["% Below VAL", "% From 52W Low"]
            )
        
        st.dataframe(
            styler,
            **get_width_kw(st.dataframe),
            hide_index=True
        )


# ----------------- TAB MTF: MULTI-TIMEFRAME BREAKOUT STRATEGY -----------------
with tab_mtf:
    st.subheader("🎯 Multi-Timeframe (MTF) Breakout Strategy")
    st.caption("First-Hour (10:15 AM IST) Closes vs Daily PDH/PDL + 15-Minute Daily Anchored VWAP")

    st.markdown(r"""
    > **Multi-Timeframe Architecture:**
    > 1. **Daily Timeframe:** Captures Previous Day High (PDH) and Previous Day Low (PDL) shifted by 1 to prevent lookahead bias.
    > 2. **1-Hour Timeframe:** Captures the closing price of the first 60 minutes (4th 15-minute candle at **10:15 AM IST**) $\rightarrow \text{First\_1H\_Close}$.
    > 3. **15-Minute Timeframe:** Calculates institutional Daily Anchored VWAP ($\frac{\text{High} + \text{Low} + \text{Close}}{3}$).
    > 
    > * 🟢 **Bullish Breakout:** $\text{First\_1H\_Close} > \text{PDH}$ **AND** $\text{Current 15m Close} > \text{VWAP}$
    > * 🔴 **Bearish Breakdown:** $\text{First\_1H\_Close} < \text{PDL}$ **AND** $\text{Current 15m Close} < \text{VWAP}$
    """)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"#### 🚀 Bullish MTF Breakouts ({len(mtf_bull_df)})")
        if mtf_bull_df.empty:
            st.info("ℹ️ No stocks currently have a First-Hour close above PDH while holding above intraday VWAP.")
        else:
            display_m_bull = mtf_bull_df[[
                "Symbol", "LTP", "Chg%", "First_1H_Close", "PDH", "VWAP", "Dist_VWAP%", "MTF_1H_Status", "MTF_VWAP_Status"
            ]].copy().rename(columns={
                "LTP": "LTP (₹)",
                "Chg%": "Day Chg %",
                "First_1H_Close": "1H Close (10:15) ₹",
                "PDH": "Prev Day High ₹",
                "VWAP": "15m VWAP ₹",
                "Dist_VWAP%": "% vs VWAP",
                "MTF_1H_Status": "1H Status",
                "MTF_VWAP_Status": "VWAP Status"
            })
            st.dataframe(
                display_m_bull.style.format({
                    "LTP (₹)": "₹{:.2f}",
                    "Day Chg %": "{:+.2f}%",
                    "1H Close (10:15) ₹": "₹{:.2f}",
                    "Prev Day High ₹": "₹{:.2f}",
                    "15m VWAP ₹": "₹{:.2f}",
                    "% vs VWAP": "{:+.2f}%"
                }),
                **get_width_kw(st.dataframe),
                hide_index=True
            )

    with col_m2:
        st.markdown(f"#### 🔻 Bearish MTF Breakdowns ({len(mtf_bear_df)})")
        if mtf_bear_df.empty:
            st.info("ℹ️ No stocks currently have a First-Hour close below PDL while holding below intraday VWAP.")
        else:
            display_m_bear = mtf_bear_df[[
                "Symbol", "LTP", "Chg%", "First_1H_Close", "PDL", "VWAP", "Dist_VWAP%", "MTF_1H_Status", "MTF_VWAP_Status"
            ]].copy().rename(columns={
                "LTP": "LTP (₹)",
                "Chg%": "Day Chg %",
                "First_1H_Close": "1H Close (10:15) ₹",
                "PDL": "Prev Day Low ₹",
                "VWAP": "15m VWAP ₹",
                "Dist_VWAP%": "% vs VWAP",
                "MTF_1H_Status": "1H Status",
                "MTF_VWAP_Status": "VWAP Status"
            })
            st.dataframe(
                display_m_bear.style.format({
                    "LTP (₹)": "₹{:.2f}",
                    "Day Chg %": "{:+.2f}%",
                    "1H Close (10:15) ₹": "₹{:.2f}",
                    "Prev Day Low ₹": "₹{:.2f}",
                    "15m VWAP ₹": "₹{:.2f}",
                    "% vs VWAP": "{:+.2f}%"
                }),
                **get_width_kw(st.dataframe),
                hide_index=True
            )

    # All Stocks MTF Status Monitor Table
    st.markdown("#### 📋 All Stocks — MTF & Initial Hour (1H) Status Monitor (Options A, B & C)")
    display_mtf_all = all_df[[
        "Symbol", "LTP", "Chg%", "MTF_Signal", "Opt_A_Status", "Prev_1H_Close", "Curr_1H_Close", 
        "Opt_B_Status", "Opt_C_Status", "Curr_1H_High", "Curr_1H_Low", "VWAP", "Dist_VWAP%"
    ]].copy().rename(columns={
        "LTP": "LTP (₹)",
        "Chg%": "Day Chg %",
        "MTF_Signal": "MTF Signal",
        "Opt_A_Status": "Option A (1H vs Prev 1H)",
        "Prev_1H_Close": "Prev 1H Close ₹",
        "Curr_1H_Close": "Today 1H Close ₹",
        "Opt_B_Status": "Option B (Live vs 1H Close)",
        "Opt_C_Status": "Option C (1H Range IB)",
        "Curr_1H_High": "1H High ₹",
        "Curr_1H_Low": "1H Low ₹",
        "VWAP": "VWAP ₹",
        "Dist_VWAP%": "% vs VWAP"
    })
    st.dataframe(
        display_mtf_all.style.format({
            "LTP (₹)": "₹{:.2f}",
            "Day Chg %": "{:+.2f}%",
            "Prev 1H Close ₹": "₹{:.2f}",
            "Today 1H Close ₹": "₹{:.2f}",
            "1H High ₹": "₹{:.2f}",
            "1H Low ₹": "₹{:.2f}",
            "VWAP ₹": "₹{:.2f}",
            "% vs VWAP": "{:+.2f}%"
        }),
        **get_width_kw(st.dataframe),
        hide_index=True
    )

    with st.expander("📜 TradingView Pine Script (v5) Code & Integration Guide"):
        st.markdown("""
        Copy and paste this script directly into **TradingView Pine Editor (v5)** to display the MTF Breakout indicators and VWAP overlays on your 15-minute charts:
        """)
        st.code(PINE_SCRIPT_V5_CODE, language="pinescript")


# ----------------- TAB 3: TOMORROW'S CPR WATCHLIST (POST-MARKET PRE-PLANNER) -----------------
with tab3:
    st.subheader(f"🌅 Tomorrow's CPR & Camarilla Watchlist (Planning for {meta.get('live_date', 'Next Session')})")
    st.markdown(f"""
    > **Post-Market Determinism:**  
    > Because today's market session has completed, today's High, Low, and Close define **tomorrow's institutional CPR and Camarilla levels with 100% certainty**.  
    > *Use this pre-market radar every evening (after 3:30 PM IST) to identify which stocks will have **Ultra Narrow CPR** tomorrow. These stocks have the highest probability of explosive trending breakouts when tomorrow's opening bell rings.*
    """)

    # Controls inside Tab 3
    cpr_col1, cpr_col2, cpr_col3 = st.columns([2, 2, 2])
    with cpr_col1:
        only_narrow_tomorrow = st.checkbox("🎯 Show Only Tomorrow's Narrow CPR (≤ 0.35%)", value=False)
    with cpr_col2:
        cpr_type_filter = st.selectbox(
            "Filter by Tomorrow's CPR Classification",
            ["All", "🎯 Ultra Narrow (≤ 0.15%)", "⚡ Narrow (≤ 0.35%)", "Average", "Wide"],
            index=0
        )
    with cpr_col3:
        sort_choice = st.selectbox("Sort Order", ["Tightest CPR First (Ascending)", "Widest CPR First (Descending)"], index=0)

    # Filter dataframe
    t_df = all_df.copy()
    if only_narrow_tomorrow:
        t_df = t_df[t_df["Next_Is_Narrow"]]
    
    if cpr_type_filter == "🎯 Ultra Narrow (≤ 0.15%)":
        t_df = t_df[t_df["Next_CPR_Type"] == "🎯 Ultra Narrow"]
    elif cpr_type_filter == "⚡ Narrow (≤ 0.35%)":
        t_df = t_df[t_df["Next_CPR_Type"] == "⚡ Narrow"]
    elif cpr_type_filter == "Average":
        t_df = t_df[t_df["Next_CPR_Type"] == "Average"]
    elif cpr_type_filter == "Wide":
        t_df = t_df[t_df["Next_CPR_Type"] == "Wide"]

    is_asc = (sort_choice == "Tightest CPR First (Ascending)")
    t_df = t_df.sort_values(by="Next_CPR_Width%", ascending=is_asc).reset_index(drop=True)

    # Quick metric summary
    n_ultra = int((all_df["Next_CPR_Type"] == "🎯 Ultra Narrow").sum())
    n_narrow = int((all_df["Next_CPR_Type"] == "⚡ Narrow").sum())
    n_avg = int((all_df["Next_CPR_Type"] == "Average").sum())
    n_wide = int((all_df["Next_CPR_Type"] == "Wide").sum())

    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("🎯 Ultra Narrow (≤ 0.15%)", f"{n_ultra} Stocks", "Highest Trend Probability")
    sm2.metric("⚡ Narrow (≤ 0.35%)", f"{n_narrow} Stocks", "Trending Potential")
    sm3.metric("Average (0.35% - 0.70%)", f"{n_avg} Stocks", "Normal Volatility")
    sm4.metric("Wide (> 0.70%)", f"{n_wide} Stocks", "Range-bound / Sideways")

    st.markdown("<br>", unsafe_allow_html=True)

    if t_df.empty:
        st.info("No stocks match the selected Tomorrow CPR filter.")
    else:
        disp_tomorrow = t_df[[
            "Symbol", "LTP", "Chg%", "Next_CPR_Type", "Next_CPR_Width%", "Next_CPR_Top", 
            "Next_CPR_Pivot", "Next_CPR_Bottom", "Next_Cam_H4", "Next_Cam_L4", "RVOL", "BBW_Contract_Status"
        ]].copy()

        disp_tomorrow = disp_tomorrow.rename(columns={
            "LTP": "Today Close / LTP (₹)",
            "Chg%": "Today Chg %",
            "Next_CPR_Type": "Tomorrow CPR Type",
            "Next_CPR_Width%": "Tomorrow CPR Width %",
            "Next_CPR_Top": "Tomorrow CPR Top (₹)",
            "Next_CPR_Pivot": "Tomorrow Pivot (₹)",
            "Next_CPR_Bottom": "Tomorrow CPR Bot (₹)",
            "Next_Cam_H4": "Tomorrow Cam H4 (₹)",
            "Next_Cam_L4": "Tomorrow Cam L4 (₹)",
            "RVOL": "Today RVOL",
            "BBW_Contract_Status": "BBW Contraction"
        })

        st.dataframe(
            disp_tomorrow.style.format({
                "Today Close / LTP (₹)": "₹{:.2f}",
                "Today Chg %": "{:+.2f}%",
                "Tomorrow CPR Width %": "{:.3f}%",
                "Tomorrow CPR Top (₹)": "₹{:.2f}",
                "Tomorrow Pivot (₹)": "₹{:.2f}",
                "Tomorrow CPR Bot (₹)": "₹{:.2f}",
                "Tomorrow Cam H4 (₹)": "₹{:.2f}",
                "Tomorrow Cam L4 (₹)": "₹{:.2f}",
                "Today RVOL": "{:.2f}x"
            }),
            **get_width_kw(st.dataframe),
            hide_index=True
        )

        csv_tomorrow = disp_tomorrow.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Tomorrow's CPR Levels as CSV",
            data=csv_tomorrow,
            file_name=f"tomorrow_cpr_levels_{meta.get('live_date', 'tomorrow')}.csv",
            mime="text/csv"
        )


# ----------------- TAB 4: SQUEEZE RADAR (PRE-BREAKOUT WATCHLIST) -----------------
with tab4:
    st.subheader("🌀 Bollinger Squeeze Radar (Volatility Compression Watchlist)")
    st.markdown("""
    > **Squeeze Definition (Developer Spec):**  
    > `Current Bandwidth <= Historical Minimum Bandwidth (120 Days) * 1.10`  
    > *These stocks are experiencing severe volatility compression. A powerful directional expansion is imminent.*
    """)
    
    if squeeze_df.empty:
        st.info("ℹ️ No F&O stocks are currently in an extreme squeeze (bandwidth within 10% of 120-day historical low).")
    else:
        st.markdown(f"**Found {len(squeeze_df)} stock(s) currently coiling in a tight Bollinger Squeeze:**")
        
        disp_squeeze = squeeze_df[[
            "Symbol", "LTP", "Chg%", "BB_Bandwidth%", "BB_Min_Bandwidth%", "BB_Upper", "BB_Middle", "BB_Lower", "RVOL", "BBW_Contract_Status", "CPR_Status", "Signal"
        ]].copy()
        
        disp_squeeze = disp_squeeze.rename(columns={
            "LTP": "LTP (₹)",
            "Chg%": "Day Chg %",
            "BB_Bandwidth%": "Bandwidth %",
            "BB_Min_Bandwidth%": "120D Min Bandwidth %",
            "BB_Upper": "Upper Band (₹)",
            "BB_Middle": "SMA20 (₹)",
            "BB_Lower": "Lower Band (₹)",
            "RVOL": "RVOL",
            "BBW_Contract_Status": "BBW Contraction",
            "CPR_Status": "CPR Status"
        })
        
        st.dataframe(
            disp_squeeze.style.format({
                "LTP (₹)": "₹{:.2f}",
                "Day Chg %": "{:+.2f}%",
                "Bandwidth %": "{:.2f}%",
                "120D Min Bandwidth %": "{:.2f}%",
                "Upper Band (₹)": "₹{:.2f}",
                "SMA20 (₹)": "₹{:.2f}",
                "Lower Band (₹)": "₹{:.2f}",
                "RVOL": "{:.2f}x"
            }),
            **get_width_kw(st.dataframe),
            hide_index=True
        )


# ----------------- TAB 5: INTERACTIVE CHART -----------------
with tab5:
    st.subheader("📊 Stock Visualizer & Profile Analysis")
    
    default_chart_sym = all_df["Symbol"].iloc[0]
    if not bull_df.empty:
        default_chart_sym = bull_df["Symbol"].iloc[0]
    elif not bear_df.empty:
        default_chart_sym = bear_df["Symbol"].iloc[0]
    elif not squeeze_df.empty:
        default_chart_sym = squeeze_df["Symbol"].iloc[0]

    chart_col, _ = st.columns([2, 1])
    with chart_col:
        chart_sym = st.selectbox(
            "Select Stock to Inspect", 
            options=all_df["Symbol"].tolist(),
            index=all_df["Symbol"].tolist().index(default_chart_sym) if default_chart_sym in all_df["Symbol"].tolist() else 0
        )

    # Fetch single stock candle data
    chart_data = get_single_stock_chart_data(chart_sym, timeframe=timeframe)
    if chart_data and not chart_data["prev_df"].empty and not chart_data["curr_df"].empty:
        stock_row = all_df[all_df["Symbol"] == chart_sym].iloc[0]
        
        fig = create_breakout_chart(
            symbol=chart_sym,
            prev_df=chart_data["prev_df"],
            curr_df=chart_data["curr_df"],
            vp=chart_data["vp"],
            h52=stock_row["52W_High"],
            l52=stock_row["52W_Low"],
            ltp=stock_row["LTP"],
            cpr=chart_data.get("cpr"),
            cam=chart_data.get("cam"),
            bb=chart_data.get("bb"),
            mtf=chart_data.get("mtf")
        )
        st.plotly_chart(fig, **get_width_kw(st.plotly_chart))
        
        # Summary details below chart
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("LTP", f"₹{stock_row['LTP']:.2f}", f"{stock_row['Chg%']:+.2f}%")
        c2.metric("PD_VAH (Green)", f"₹{stock_row['PD_VAH']:.2f}")
        c3.metric("PD_VAL (Red)", f"₹{stock_row['PD_VAL']:.2f}")
        c4.metric(f"CPR ({stock_row['CPR_Type']})", f"₹{stock_row['PD_CPR_Top']:.2f}", f"Width: {stock_row['PD_CPR_Width%']:.3f}%")
        c5.metric("Cam H4 / L4", f"₹{stock_row['PD_Cam_H4']:.2f}", f"L4: ₹{stock_row['PD_Cam_L4']:.2f}")
        c6.metric("BB Bandwidth", f"{stock_row['BB_Bandwidth%']:.2f}%", f"Min: {stock_row['BB_Min_Bandwidth%']:.2f}%")

        # MTF Indicator Banner
        m_stat = stock_row.get("MTF_Signal", "None")
        h1_c = stock_row.get("First_1H_Close", 0.0)
        vwap_val = stock_row.get("VWAP", 0.0)
        pdh_val = stock_row.get("PDH", 0.0)
        pdl_val = stock_row.get("PDL", 0.0)
        st.markdown(f"""
        <div style="background: rgba(156, 39, 176, 0.08); border: 1px solid #ab47bc; border-radius: 6px; padding: 10px 14px; margin-top: 10px;">
            <span style="font-weight: 700; color: #E040FB;">🎯 MTF Breakout Algorithm Levels:</span> &nbsp;
            <b>Signal:</b> {m_stat} &nbsp;|&nbsp;
            <b>1H Close (10:15):</b> ₹{h1_c} &nbsp;|&nbsp;
            <b>15m VWAP:</b> ₹{vwap_val} ({stock_row.get('MTF_VWAP_Status', 'N/A')}) &nbsp;|&nbsp;
            <b>PDH:</b> ₹{pdh_val} &nbsp;|&nbsp;
            <b>PDL:</b> ₹{pdl_val} &nbsp;|&nbsp;
            <b>1H Status:</b> {stock_row.get('MTF_1H_Status', 'N/A')}
        </div>
        """, unsafe_allow_html=True)

        # Tomorrow's Projections Banner
        if "next_cpr" in chart_data and chart_data["next_cpr"]:
            nc = chart_data["next_cpr"]
            n_cam = chart_data.get("next_cam", {})
            q = chart_data.get("quant", {})
            st.markdown(f"""
            <div style="background: rgba(33, 150, 243, 0.08); border: 1px solid #1e88e5; border-radius: 6px; padding: 10px 14px; margin-top: 10px;">
                <span style="font-weight: 700; color: #29B6F6;">🌅 Projected Tomorrow's Levels (Post-Market):</span> &nbsp;
                <b>CPR:</b> {nc.get('type')} ({nc.get('width_pct', 0):.3f}%) &nbsp;|&nbsp;
                <b>Top:</b> ₹{nc.get('top')} &nbsp;|&nbsp;
                <b>Pivot:</b> ₹{nc.get('pivot')} &nbsp;|&nbsp;
                <b>Bot:</b> ₹{nc.get('bottom')} &nbsp;|&nbsp;
                <b>Cam H4:</b> ₹{n_cam.get('h4')} &nbsp;|&nbsp;
                <b>Cam L4:</b> ₹{n_cam.get('l4')}<br>
                <span style="font-weight: 700; color: #AB47BC;">📅 Weekly Context (Sections 6-8):</span> &nbsp;
                <b>Bias:</b> {q.get('w_cpr_bias', 'N/A')} &nbsp;|&nbsp;
                <b>W_VAH:</b> ₹{q.get('w_vah', 0)} &nbsp;|&nbsp;
                <b>W_VAL:</b> ₹{q.get('w_val', 0)} &nbsp;|&nbsp;
                <b>NR7 Compression:</b> {"⚡ Yes (Coil)" if q.get("is_nr7") else ("Prior NR7" if q.get("prior_is_nr7") else "No")} &nbsp;|&nbsp;
                <b>RVOL:</b> {q.get('rvol', 1.0)}x
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Candle data not available for this stock.")


# ----------------- TAB 6: ALL F&O MONITOR -----------------
with tab6:
    st.subheader(f"All Scanned F&O Stocks ({len(all_df)})")
    
    search_query = st.text_input("🔍 Filter by Stock Symbol", "").strip().upper()
    filter_df = all_df
    if search_query:
        filter_df = filter_df[filter_df["Symbol"].str.contains(search_query)]

    display_all = filter_df[[
        "Symbol", "LTP", "Chg%", "Signal", "MTF_Signal", "MTF_1H_Status", "First_1H_Close", "VWAP",
        "RVOL", "BBW_Contract_Status", "NR7_Status", 
        "W_CPR_Bias", "W_VAH", "W_VAL", "Next_CPR_Type", "Next_CPR_Width%",
        "PD_VAH", "Dist_VAH%", "PD_VAL", "Dist_VAL%", 
        "CPR_Status", "CPR_Type", "PD_CPR_Width%", "PD_CPR_Top", "PD_CPR_Bottom", 
        "Cam_Status", "BB_Status", "BB_Bandwidth%", "52W_High", "Dist_52W_High%"
    ]].copy()

    st.dataframe(
        display_all.style.format({
            "LTP": "₹{:.2f}",
            "Chg%": "{:+.2f}%",
            "First_1H_Close": "₹{:.2f}",
            "VWAP": "₹{:.2f}",
            "RVOL": "{:.2f}x",
            "W_VAH": "₹{:.2f}",
            "W_VAL": "₹{:.2f}",
            "Next_CPR_Width%": "{:.3f}%",
            "PD_VAH": "₹{:.2f}",
            "Dist_VAH%": "{:+.2f}%",
            "PD_VAL": "₹{:.2f}",
            "Dist_VAL%": "{:+.2f}%",
            "PD_CPR_Width%": "{:.3f}%",
            "PD_CPR_Top": "₹{:.2f}",
            "PD_CPR_Bottom": "₹{:.2f}",
            "BB_Bandwidth%": "{:.2f}%",
            "52W_High": "₹{:.2f}",
            "Dist_52W_High%": "{:+.2f}%"
        }),
        **get_width_kw(st.dataframe),
        hide_index=True
    )

    # Download CSV
    csv_bytes = filter_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv_bytes,
        file_name=f"fo_breakout_screener_{datetime.date.today()}.csv",
        mime="text/csv"
    )


# ----------------- TAB 7: DOCUMENTATION -----------------
with tab7:
    st.markdown(r"""
    ### 📖 Institutional Breakout Strategy & Confluence Framework
    
    This screener tracks high-probability breakout momentum by combining **Volume Profile**, **Central Pivot Range (CPR / Money Zone)**, **Camarilla Pivots**, **Bollinger Bands Squeeze & Volatility Expansion**, **Quantitative RVOL & BBW Contraction**, **Weekly CPR Bias & Weekly Value Area**, and **Trailing 52-Week Extremes**.
    
    ---

    #### 1. Volume Profile (Value Area 70%)
    - **PD_VAH (Previous Day Value Area High)**: The upper price level enclosing **70%** of yesterday's trading volume.
    - **PD_VAL (Previous Day Value Area Low)**: The lower price level enclosing **70%** of yesterday's trading volume.
    - **PD_POC (Point of Control)**: The single price tick that transacted the highest volume in yesterday's session.
    
    ---

    #### 2. Central Pivot Range (CPR / Money Zone) & Tomorrow's Pre-Market Projection
    CPR defines the institutional fair-value zone:
    - **Pivot (P)** = $\frac{High + Low + Close}{3}$
    - **Bottom Central (BC)** = $\frac{High + Low}{2}$
    - **Top Central (TC)** = $(P - BC) + P = 2P - BC$
    - **CPR Top** = $\max(TC, BC)$, **CPR Bottom** = $\min(TC, BC)$
    
    * **Bullish Money Zone Rule:** $\text{LTP} > \text{PD_CPR_Top}$ (Institutional buyers are accepting prices above fair value)
    * **Bearish Money Zone Rule:** $\text{LTP} < \text{PD_CPR_Bottom}$ (Institutional sellers are dominating below fair value)

    ##### 🎯 The Power of Narrow Daily CPR:
    - **CPR Width** = $|\text{TC} - \text{BC}|$
    - **CPR Width %** = $\frac{|\text{TC} - \text{BC}|}{\text{Pivot}} \times 100$
    
    | Classification | CPR Width % | Market Character & Trading Significance |
    | :--- | :--- | :--- |
    | **🎯 Ultra Narrow** | $\le 0.15\%$ | Extreme consolidation yesterday. **Highest probability of an explosive directional trend / trending day.** |
    | **⚡ Narrow** | $0.15\% - 0.35\%$ | High potential for sustained trend and clean breakout moves. |
    | **Average** | $0.35\% - 0.70\%$ | Moderate trending capability with standard intraday volatility. |
    | **Wide** | $> 0.70\%$ | Broad value area. **High likelihood of sideways / choppy / range-bound day.** Avoid breakout trades; prefer range reversals. |

    ##### 🌅 Can we know Tomorrow's CPR after market closes?
    **Yes, 100% definitively!** Because CPR relies exclusively on the High, Low, and Close of the session, once the market closes at 3:30 PM IST (and official settlement prices finalize by 3:45 PM IST), today's numbers are permanent. Tomorrow's CPR, Pivot, Top Central, Bottom Central, and Camarilla H4/L4 are fully locked and known before tomorrow's opening bell.

    ---

    #### 3. Quantitative Filters (RVOL & Volatility Contraction)
    *(Implemented from Developer Technical Feature Specification)*

    ##### A. Relative Volume (RVOL):
    - Measures current stock volume against its historical 20-period Simple Moving Average:
      $$\text{RVOL} = \frac{\text{Current Volume}}{\text{SMA}_{20}(\text{Volume})}$$
    - A breakout accompanied by high RVOL ($\ge 1.5\times$) confirms institutional participation and filters out low-probability retail traps.

    ##### B. Bollinger Bandwidth (BBW) Contraction:
    - Normalizes bandwidth as percentage:
      $$\text{BBW} = \frac{\text{Upper Band} - \text{Lower Band}}{\text{SMA}_{20}(\text{Close})} \times 100$$
    - Evaluates 20-period moving average of bandwidth: $\text{BBW\_SMA}_{20} = \text{SMA}_{20}(\text{BBW})$.
    - Contraction condition: $\text{Is\_BBW\_Contracting} = (\text{BBW} < \text{BBW\_SMA}_{20})$.
    - **Sequential Pipeline Rule:** Uses $\text{shift}(1)$ to verify that the bands were contracted on the day **prior** ($T-1$) to today's breakout day.

    ---

    #### 4. Open Interest (OI) Buildup & NR7 Volatility Compression (Section 5)
    - **NR7 (Narrow Range 7)**: True when today's daily range ($\text{High} - \text{Low}$) is the tightest of the last 7 trading days. NR7 represents extreme coiling energy; explosive directional breakouts typically occur within 1–2 sessions of an NR7 bar.
    - **Open Interest Buildup Matrix**:
      - **Long Buildup**: Price $\uparrow$ + OI $\uparrow$ (Aggressive fresh long accumulation)
      - **Short Buildup**: Price $\downarrow$ + OI $\uparrow$ (Aggressive institutional short creation)
      - **Short Covering**: Price $\uparrow$ + OI $\downarrow$ (Short sellers covering panic positions)
      - **Long Unwinding**: Price $\downarrow$ + OI $\downarrow$ (Profit booking / stop-loss hit on longs)

    ---

    #### 5. Weekly CPR (WCPR) & Value Relationship Logic (Section 6)
    Defines the macro weekly directional bias:
    - **Weekly Pivot**: $\frac{\text{Prev\_W\_High} + \text{Prev\_W\_Low} + \text{Prev\_W\_Close}}{3}$
    - **Weekly BC**: $\frac{\text{Prev\_W\_High} + \text{Prev\_W\_Low}}{2}$
    - **Weekly TC**: $2 \times \text{W\_Pivot} - \text{W\_BC}$
    - **Weekly CPR Value Relationships (Bias)**:
      - 🟢 **Bullish (Higher Value)**: Current Week $\text{Bottom CPR} > \text{Previous Week Top CPR}$
      - 🔴 **Bearish (Lower Value)**: Current Week $\text{Top CPR} < \text{Previous Week Bottom CPR}$
      - ⚡ **Breakout (Inside Value)**: Current Week CPR is completely contained inside Previous Week CPR (High-probability trending week).

    ---

    #### 6. Weekly Value Area (WVA) Structural Approximations (Section 7)
    Approximates the weekly 70% institutional value area:
    - $\text{W\_POC} = \frac{\text{Prev\_W\_High} + \text{Prev\_W\_Low} + \text{Prev\_W\_Close}}{3}$
    - $\text{W\_Range} = \text{Prev\_W\_High} - \text{Prev\_W\_Low}$
    - $\text{W\_VAH} = \text{W\_POC} + (\text{W\_Range} \times 0.35)$
    - $\text{W\_VAL} = \text{W\_POC} - (\text{W\_Range} \times 0.35)$

    ---

    #### 7. Integrated Multi-Timeframe Confluence Filter (Section 8)
    To ensure daily breakouts are supported by macro institutional order flow:
    - **Bullish Confluence**: Daily Breakout ($Close > \text{PD\_VAH}$) supported by:
      1. $\text{Close} > \text{W\_VAH}$ (Trading above Weekly Value Area High)
      2. Weekly CPR Bias $\in [\text{"Bullish (Higher Value)"}, \text{"Breakout (Inside Value)"}]$
    - **Bearish Confluence**: Daily Breakdown ($Close < \text{PD\_VAL}$) supported by:
      1. $\text{Close} < \text{W\_VAL}$ (Trading below Weekly Value Area Low)
      2. Weekly CPR Bias $\in [\text{"Bearish (Lower Value)"}, \text{"Breakout (Inside Value)"}]$

    ---

    #### 8. Bollinger Bands Squeeze & Volatility Expansion (Developer Spec)
    - **Middle Band (SMA20)**: 20-period Simple Moving Average of Close.
    - **Upper Band**: $\text{SMA20} + (\text{StdDev} \times 2.0)$
    - **Lower Band**: $\text{SMA20} - (\text{StdDev} \times 2.0)$
    - **Bandwidth**: $\frac{\text{Upper Band} - \text{Lower Band}}{\text{SMA20}}$
    - **Historical Minimum Bandwidth**: Lowest Bandwidth recorded over 120-period lookback window (~6 months).
    
    * **Condition A (The Squeeze):** $\text{Current Bandwidth} \le (\text{Min Bandwidth} \times 1.10)$ (Extreme volatility compression)
    * **Condition B (The Breakout):** $\text{Close} > \text{Upper Band}$ (Bullish) or $\text{Close} < \text{Lower Band}$ (Bearish)
    * **Master Confluence:** Squeeze on previous period ($T-1$) and Breakout on current period ($T_0$).

    ---

    #### 9. Camarilla Pivots (Breakout Triggers)
    - $\text{Range} = High - Low$
    - **H4 (Bullish Breakout)** = $Close + \text{Range} \times 0.55$
    - **L4 (Bearish Breakdown)** = $Close - \text{Range} \times 0.55$
    
    ---

    #### 10. Trailing 52-Week Extremes
    - **52W High**: Trailing 52-week highest traded price.
    - **52W Low**: Trailing 52-week lowest traded price.
    
    ---

    #### 11. Multi-Timeframe (MTF) Breakout Algorithm (Developer Guide Spec)
    Synthesizes three distinct timeframes on a single execution chart:
    - **Daily Timeframe**: Tracks Previous Day High (PDH) and Previous Day Low (PDL) shifted by 1 to eliminate lookahead bias.
    - **1-Hour Timeframe (First 60 mins)**: Captures the closing price of the 4th 15-minute candle at **10:15 AM IST** ($\text{First\_1H\_Close}$).
    - **15-Minute Timeframe**: Computes institutional Daily Anchored VWAP utilizing the Typical Price formula:
      $$\text{Typical Price} = \frac{\text{High} + \text{Low} + \text{Close}}{3}, \quad \text{VWAP} = \frac{\sum (\text{Typical Price} \times \text{Volume})}{\sum \text{Volume}}$$
    - **Signal Confluence Rules**:
      - 🟢 **Bullish Breakout**: $\text{First\_1H\_Close} > \text{PDH}$ **AND** $\text{Current 15m Close} > \text{VWAP}$
      - 🔴 **Bearish Breakdown**: $\text{First\_1H\_Close} < \text{PDL}$ **AND** $\text{Current 15m Close} < \text{VWAP}$

    ---

    #### 12. Initial Hour (1H) Breakout & Break-in Modes (Options A, B & C)
    - **Option A (1H Close vs Prev 1H Close)**:
      - 🟢 **Breakout**: Current Day 1H Close ($10:15$) $>$ Previous Day 1H Close ($10:15$)
      - 🔴 **Breakdown**: Current Day 1H Close $<$ Previous Day 1H Close
      - ⚪ **Break-in**: Current Day 1H Close is contained inside Previous Day's 1H range
    - **Option B (Live Market Price vs 1H Close)**:
      - 🟢 **Breakout**: Live Price $>$ Current Day 1H Close
      - 🔴 **Breakdown**: Live Price $<$ Current Day 1H Close
      - 🔄 **Break-in**: Price broke out earlier today, but failed and re-entered below the 1H close level
    - **Option C (Initial Balance 1H High / Low Range)**:
      - 🟢 **Range Breakout**: Live Price $>$ Today 1H High
      - 🔴 **Range Breakdown**: Live Price $<$ Today 1H Low
      - 🔄 **Range Break-in**: Price broke outside the 1H Range earlier but failed and re-entered the 1H range (Classic IB Failed Breakout Reversal)
    """)

    with st.expander("📜 View TradingView Pine Script (v5) Implementation"):
        st.code(PINE_SCRIPT_V5_CODE, language="pinescript")

# Auto-refresh handling
if auto_refresh != "Off":
    delay = 30 if "30" in auto_refresh else (60 if "60" in auto_refresh else 120)
    time.sleep(delay)
    st.rerun()
