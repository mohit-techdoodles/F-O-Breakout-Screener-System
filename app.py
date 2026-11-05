"""
NSE F&O Breakout Screener (VAH/VAL, CPR Money Zone & Camarilla Pivots & 52-Week Extremes)
Streamlit Application
"""

import datetime
import inspect
import time
import pandas as pd
import streamlit as st

from fo_stocks import VERIFIED_FO_STOCKS, NIFTY_50, BANK_NIFTY, TOP_30_LIQUID, get_fo_universe
from screener_engine import fetch_and_screen, get_single_stock_chart_data
from chart_utils import create_breakout_chart


def get_width_kw(fn):
    """Seamlessly handles Streamlit's transition from use_container_width=True to width='stretch'"""
    try:
        return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


# Page configuration
st.set_page_config(
    page_title="F&O Breakout Screener | VAH/VAL, CPR & Camarilla",
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
        font-size: 28px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-lbl {
        color: #9aa0a6;
        font-size: 13px;
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

# Pivot & Money Zone Criteria
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Pivot & Money Zone Criteria")

enable_cpr = st.sidebar.checkbox(
    "Require CPR Filter (Money Zone)",
    value=True,
    help="Bullish Setup: LTP must be above Previous Day CPR Top.\nBearish Setup: LTP must be below Previous Day CPR Bottom."
)

camarilla_mode = st.sidebar.selectbox(
    "Camarilla Pivot Criteria",
    ["None", "H4 / L4 Breakout (Strong)", "H3 / L3 Range Breakout"],
    index=0,
    help="H4/L4: LTP > H4 for Bullish or < L4 for Bearish.\nH3/L3: LTP > H3 for Bullish or < L3 for Bearish."
)

# Auto-refresh
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Live Refresh")
auto_refresh = st.sidebar.selectbox("Auto Refresh Interval", ["Off", "30 Seconds", "60 Seconds", "120 Seconds"], index=0)

refresh_clicked = st.sidebar.button("🔄 Scan Now", **get_width_kw(st.button))

# Helper function to cache screening data (cached for 60 seconds)
@st.cache_data(ttl=60, show_spinner=False)
def get_cached_screener_data(syms, tf, h_prox, l_prox, cpr_flag, cam_mode):
    return fetch_and_screen(
        syms, 
        timeframe=tf, 
        high_proximity_pct=h_prox, 
        low_proximity_pct=l_prox,
        enable_cpr_filter=cpr_flag,
        camarilla_filter=cam_mode
    )


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
    st.title("📈 F&O Breakout Screener")
    st.caption("Strategy: Volume Profile (VAH/VAL) + CPR Money Zone + Camarilla Pivots & 52-Week Extremes")

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
with st.spinner(f"Scanning {len(selected_symbols)} F&O stocks with Volume Profile, CPR & Camarilla..."):
    if refresh_clicked:
        st.cache_data.clear()
    all_df, bull_df, bear_df, meta = get_cached_screener_data(
        selected_symbols, timeframe, high_proximity, low_proximity, enable_cpr, camarilla_mode
    )

if all_df.empty:
    st.warning("⚠️ No data was returned. Please verify your connection or refresh.")
    st.stop()

# Top Metric Cards Row
m1, m2, m3, m4 = st.columns(4)
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
        <div class="metric-lbl">Bullish Breakouts (Long)</div>
        <div class="metric-val" style="color: #00E676;">{meta.get('bullish_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Bearish Breakdowns (Short)</div>
        <div class="metric-val" style="color: #FF5252;">{meta.get('bearish_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    ref_d = meta.get('ref_date', 'N/A')
    live_d = meta.get('live_date', 'N/A')
    cpr_tag = "CPR: Active" if enable_cpr else "CPR: Off"
    cam_tag = f"Cam: {camarilla_mode.split()[0]}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Filters & Session</div>
        <div class="metric-val" style="font-size: 14px; color: #FFF; padding-top: 6px;">
            {cpr_tag} | {cam_tag}<br>
            <span style="color: #9aa0a6; font-size: 12px;">Ref: {ref_d} | Live: {live_d}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"🚀 Bullish Breakouts ({len(bull_df)})", 
    f"🔻 Bearish Breakdowns ({len(bear_df)})", 
    "📈 Interactive Chart & Profile", 
    f"📋 All F&O Monitor ({len(all_df)})",
    "ℹ️ Strategy & Documentation"
])

# ----------------- TAB 1: BULLISH BREAKOUTS -----------------
with tab1:
    cpr_rule_txt = "✅ **CPR Money Zone:** `LTP > PD_CPR_Top` (Above yesterday's Central Pivot Range)" if enable_cpr else "⚪ CPR Money Zone filter is optional/off"
    cam_rule_txt = f"✅ **Camarilla:** Filter active ({camarilla_mode})" if camarilla_mode != "None" else "⚪ Camarilla filter is optional/off"
    
    st.markdown(f"""
    > **Bullish Setup Confluence Rules:**  
    > 1. **Value Area Breakout:** `LTP > PD_VAH` (Breaking out above yesterday's 70% value area)  
    > 2. **52-Week High Proximity:** `LTP >= 52W_High * {int(high_proximity*100)}%` (Trading within {100-int(high_proximity*100)}% of 52-Week High)  
    > 3. {cpr_rule_txt}  
    > 4. {cam_rule_txt}  
    """)
    
    if bull_df.empty:
        st.info("ℹ️ No F&O stocks currently meet all active Bullish Breakout criteria. Watch for emerging breakouts above yesterday's VAH and CPR.")
    else:
        st.success(f"🎯 Found **{len(bull_df)}** Bullish Breakout candidate(s)!")
        
        display_bull = bull_df[[
            "Symbol", "LTP", "Chg%", "PD_VAH", "Dist_VAH%", "52W_High", "Dist_52W_High%", 
            "CPR_Status", "PD_CPR_Top", "Cam_Status", "PD_Cam_H4", "PD_POC"
        ]].copy()
        
        display_bull = display_bull.rename(columns={
            "LTP": "LTP (₹)",
            "Chg%": "Day Chg %",
            "PD_VAH": "Yesterday VAH (₹)",
            "Dist_VAH%": "% Above VAH",
            "52W_High": "52W High (₹)",
            "Dist_52W_High%": "% From 52W High",
            "CPR_Status": "CPR Money Zone",
            "PD_CPR_Top": "CPR Top (₹)",
            "Cam_Status": "Camarilla Status",
            "PD_Cam_H4": "Cam H4 (₹)",
            "PD_POC": "Yesterday POC (₹)"
        })
        
        styler = display_bull.style.format({
            "LTP (₹)": "₹{:.2f}",
            "Day Chg %": "{:+.2f}%",
            "Yesterday VAH (₹)": "₹{:.2f}",
            "% Above VAH": "+{:.2f}%",
            "52W High (₹)": "₹{:.2f}",
            "% From 52W High": "{:+.2f}%",
            "CPR Top (₹)": "₹{:.2f}",
            "Cam H4 (₹)": "₹{:.2f}",
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
    cpr_rule_txt = "✅ **CPR Money Zone:** `LTP < PD_CPR_Bottom` (Below yesterday's Central Pivot Range)" if enable_cpr else "⚪ CPR Money Zone filter is optional/off"
    cam_rule_txt = f"✅ **Camarilla:** Filter active ({camarilla_mode})" if camarilla_mode != "None" else "⚪ Camarilla filter is optional/off"

    st.markdown(f"""
    > **Bearish Setup Confluence Rules:**  
    > 1. **Value Area Breakdown:** `LTP < PD_VAL` (Breaking down below yesterday's 70% value area)  
    > 2. **52-Week Low Proximity:** `LTP <= 52W_Low * {int(low_proximity*100)}%` (Trading within {int(low_proximity*100)-100}% of 52-Week Low)  
    > 3. {cpr_rule_txt}  
    > 4. {cam_rule_txt}  
    """)
    
    if bear_df.empty:
        st.info("ℹ️ No F&O stocks currently meet all active Bearish Breakdown criteria.")
    else:
        st.error(f"🔻 Found **{len(bear_df)}** Bearish Breakdown candidate(s)!")
        
        display_bear = bear_df[[
            "Symbol", "LTP", "Chg%", "PD_VAL", "Dist_VAL%", "52W_Low", "Dist_52W_Low%", 
            "CPR_Status", "PD_CPR_Bottom", "Cam_Status", "PD_Cam_L4", "PD_POC"
        ]].copy()
        
        display_bear = display_bear.rename(columns={
            "LTP": "LTP (₹)",
            "Chg%": "Day Chg %",
            "PD_VAL": "Yesterday VAL (₹)",
            "Dist_VAL%": "% Below VAL",
            "52W_Low": "52W Low (₹)",
            "Dist_52W_Low%": "% From 52W Low",
            "CPR_Status": "CPR Money Zone",
            "PD_CPR_Bottom": "CPR Bottom (₹)",
            "Cam_Status": "Camarilla Status",
            "PD_Cam_L4": "Cam L4 (₹)",
            "PD_POC": "Yesterday POC (₹)"
        })
        
        styler = display_bear.style.format({
            "LTP (₹)": "₹{:.2f}",
            "Day Chg %": "{:+.2f}%",
            "Yesterday VAL (₹)": "₹{:.2f}",
            "% Below VAL": "{:.2f}%",
            "52W Low (₹)": "₹{:.2f}",
            "% From 52W Low": "{:+.2f}%",
            "CPR Bottom (₹)": "₹{:.2f}",
            "Cam L4 (₹)": "₹{:.2f}",
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


# ----------------- TAB 3: INTERACTIVE CHART -----------------
with tab3:
    st.subheader("📊 Stock Visualizer & Profile Analysis")
    
    # Default selection
    default_chart_sym = all_df["Symbol"].iloc[0]
    if not bull_df.empty:
        default_chart_sym = bull_df["Symbol"].iloc[0]
    elif not bear_df.empty:
        default_chart_sym = bear_df["Symbol"].iloc[0]

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
            cam=chart_data.get("cam")
        )
        st.plotly_chart(fig, **get_width_kw(st.plotly_chart))
        
        # Summary details below chart
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("LTP", f"₹{stock_row['LTP']:.2f}", f"{stock_row['Chg%']:+.2f}%")
        c2.metric("PD_VAH (Green)", f"₹{stock_row['PD_VAH']:.2f}")
        c3.metric("PD_VAL (Red)", f"₹{stock_row['PD_VAL']:.2f}")
        c4.metric("CPR Top / Bot", f"₹{stock_row['PD_CPR_Top']:.2f}", f"Bot: ₹{stock_row['PD_CPR_Bottom']:.2f}")
        c5.metric("Cam H4 / L4", f"₹{stock_row['PD_Cam_H4']:.2f}", f"L4: ₹{stock_row['PD_Cam_L4']:.2f}")
        c6.metric("52W High (Trigger)", f"₹{stock_row['52W_High']:.2f}", f"₹{stock_row['52W_High']*high_proximity:.2f}")
    else:
        st.info("Candle data not available for this stock.")


# ----------------- TAB 4: ALL F&O MONITOR -----------------
with tab4:
    st.subheader(f"All Scanned F&O Stocks ({len(all_df)})")
    
    search_query = st.text_input("🔍 Filter by Stock Symbol", "").strip().upper()
    filter_df = all_df
    if search_query:
        filter_df = filter_df[filter_df["Symbol"].str.contains(search_query)]

    display_all = filter_df[[
        "Symbol", "LTP", "Chg%", "Signal", "PD_VAH", "Dist_VAH%", "PD_VAL", "Dist_VAL%", 
        "CPR_Status", "PD_CPR_Top", "PD_CPR_Bottom", "Cam_Status", "PD_Cam_H4", "PD_Cam_L4", "52W_High", "Dist_52W_High%"
    ]].copy()

    st.dataframe(
        display_all.style.format({
            "LTP": "₹{:.2f}",
            "Chg%": "{:+.2f}%",
            "PD_VAH": "₹{:.2f}",
            "Dist_VAH%": "{:+.2f}%",
            "PD_VAL": "₹{:.2f}",
            "Dist_VAL%": "{:+.2f}%",
            "PD_CPR_Top": "₹{:.2f}",
            "PD_CPR_Bottom": "₹{:.2f}",
            "PD_Cam_H4": "₹{:.2f}",
            "PD_Cam_L4": "₹{:.2f}",
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


# ----------------- TAB 5: DOCUMENTATION -----------------
with tab5:
    st.markdown(r"""
    ### 📖 Institutional Breakout Strategy & Confluence Framework
    
    This screener tracks high-probability breakout momentum by combining **Volume Profile**, **Central Pivot Range (CPR / Money Zone)**, **Camarilla Pivots**, and **Trailing 52-Week Extremes**.
    
    ---

    #### 1. Volume Profile (Value Area 70%)
    - **PD_VAH (Previous Day Value Area High)**: The upper price level enclosing **70%** of yesterday's trading volume.
    - **PD_VAL (Previous Day Value Area Low)**: The lower price level enclosing **70%** of yesterday's trading volume.
    - **PD_POC (Point of Control)**: The single price tick that transacted the highest volume in yesterday's session.
    
    ---

    #### 2. Central Pivot Range (CPR / Money Zone)
    CPR defines the institutional fair-value zone:
    - **Pivot (P)** = $\\frac{High + Low + Close}{3}$
    - **Bottom Central (BC)** = $\\frac{High + Low}{2}$
    - **Top Central (TC)** = $(P - BC) + P = 2P - BC$
    - **CPR Top** = $\\max(TC, BC)$, **CPR Bottom** = $\\min(TC, BC)$
    
    * **Bullish Money Zone Rule:** $\\text{LTP} > \\text{PD\_CPR\_Top}$ (Institutional buyers are accepting prices above fair value)
    * **Bearish Money Zone Rule:** $\\text{LTP} < \\text{PD\_CPR\_Bottom}$ (Institutional sellers are dominating below fair value)

    ---

    #### 3. Camarilla Pivots (Breakout Triggers)
    Camarilla calculates mathematical boundaries based on the previous session's volatility range:
    - $\\text{Range} = High - Low$
    - **H4 (Bullish Breakout)** = $Close + \\text{Range} \\times 0.55$
    - **H3 (Range Top)** = $Close + \\text{Range} \\times 0.275$
    - **L3 (Range Bottom)** = $Close - \\text{Range} \\times 0.275$
    - **L4 (Bearish Breakdown)** = $Close - \\text{Range} \\times 0.55$
    
    * A breakout above **H4** confirms strong expansion momentum.
    * A breakdown below **L4** confirms aggressive liquidation.

    ---

    #### 4. Trailing 52-Week Extremes
    - **52W High**: Trailing 52-week highest traded price.
    - **52W Low**: Trailing 52-week lowest traded price.
    - Ensures that breakouts are not false rallies into heavy overhead supply, but occur near structural highs/lows.
    """)

# Auto-refresh handling
if auto_refresh != "Off":
    delay = 30 if "30" in auto_refresh else (60 if "60" in auto_refresh else 120)
    time.sleep(delay)
    st.rerun()
