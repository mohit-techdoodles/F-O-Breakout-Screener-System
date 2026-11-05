"""
NSE F&O Breakout Screener (VAH/VAL & 52-Week Extremes)
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
    page_title="F&O Breakout Screener | VAH/VAL & 52W Extremes",
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
    .bullish-badge {
        background-color: rgba(38, 166, 154, 0.2);
        color: #00E676;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #00E676;
    }
    .bearish-badge {
        background-color: rgba(239, 83, 80, 0.2);
        color: #FF5252;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #FF5252;
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
    ["All F&O (~185 Stocks)", "Nifty 50", "Bank Nifty", "Top 30 Liquid F&O", "Custom Watchlist"],
    index=0
)

all_universe_symbols = get_fo_universe()

if universe_option == "All F&O (~185 Stocks)":
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

# Auto-refresh
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Live Refresh")
auto_refresh = st.sidebar.selectbox("Auto Refresh Interval", ["Off", "30 Seconds", "60 Seconds", "120 Seconds"], index=0)

refresh_clicked = st.sidebar.button("🔄 Scan Now", **get_width_kw(st.button))

# Helper function to cache screening data (cached for 60 seconds)
@st.cache_data(ttl=60, show_spinner=False)
def get_cached_screener_data(syms, tf, h_prox, l_prox):
    return fetch_and_screen(syms, timeframe=tf, high_proximity_pct=h_prox, low_proximity_pct=l_prox)


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
    st.caption("Strategy: Volume Profile Value Area Breakouts (PD_VAH / PD_VAL) near 52-Week Extremes")

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
with st.spinner(f"Scanning {len(selected_symbols)} F&O stocks with {timeframe} Volume Profile & 52-Week Extremes..."):
    if refresh_clicked:
        st.cache_data.clear()
    all_df, bull_df, bear_df, meta = get_cached_screener_data(
        selected_symbols, timeframe, high_proximity, low_proximity
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
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Session Info</div>
        <div class="metric-val" style="font-size: 15px; color: #FFF; padding-top: 8px;">
            Ref (VP): {ref_d}<br>Live: {live_d}
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
    st.markdown("""
    > **Bullish Setup Rules:**  
    > 1. **Value Area Breakout**: `LTP > PD_VAH` (Breaking out above yesterday's 70% value area)  
    > 2. **52-Week High Proximity**: `LTP >= 52W_High * 0.95` (Trading within 5% of its 52-Week High)
    """)
    
    if bull_df.empty:
        st.info("ℹ️ No F&O stocks currently meet both Bullish Breakout criteria. Watch for emerging breakouts above yesterday's VAH.")
    else:
        st.success(f"🎯 Found **{len(bull_df)}** Bullish Breakout candidate(s)!")
        
        display_bull = bull_df[[
            "Symbol", "LTP", "Chg%", "PD_VAH", "Dist_VAH%", "52W_High", "Dist_52W_High%", "PD_POC"
        ]].copy()
        
        display_bull = display_bull.rename(columns={
            "LTP": "LTP (₹)",
            "Chg%": "Day Chg %",
            "PD_VAH": "Yesterday VAH (₹)",
            "Dist_VAH%": "% Above VAH",
            "52W_High": "52W High (₹)",
            "Dist_52W_High%": "% From 52W High",
            "PD_POC": "Yesterday POC (₹)"
        })
        
        # Highlight styling with compatibility for pandas 2 and 3
        styler = display_bull.style.format({
            "LTP (₹)": "₹{:.2f}",
            "Day Chg %": "{:+.2f}%",
            "Yesterday VAH (₹)": "₹{:.2f}",
            "% Above VAH": "+{:.2f}%",
            "52W High (₹)": "₹{:.2f}",
            "% From 52W High": "{:+.2f}%",
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
    st.markdown("""
    > **Bearish Setup Rules:**  
    > 1. **Value Area Breakdown**: `LTP < PD_VAL` (Breaking down below yesterday's 70% value area)  
    > 2. **52-Week Low Proximity**: `LTP <= 52W_Low * 1.05` (Trading within 5% of its 52-Week Low)
    """)
    
    if bear_df.empty:
        st.info("ℹ️ No F&O stocks currently meet both Bearish Breakdown criteria.")
    else:
        st.error(f"🔻 Found **{len(bear_df)}** Bearish Breakdown candidate(s)!")
        
        display_bear = bear_df[[
            "Symbol", "LTP", "Chg%", "PD_VAL", "Dist_VAL%", "52W_Low", "Dist_52W_Low%", "PD_POC"
        ]].copy()
        
        display_bear = display_bear.rename(columns={
            "LTP": "LTP (₹)",
            "Chg%": "Day Chg %",
            "PD_VAL": "Yesterday VAL (₹)",
            "Dist_VAL%": "% Below VAL",
            "52W_Low": "52W Low (₹)",
            "Dist_52W_Low%": "% From 52W Low",
            "PD_POC": "Yesterday POC (₹)"
        })
        
        # Highlight styling with compatibility for pandas 2 and 3
        styler = display_bear.style.format({
            "LTP (₹)": "₹{:.2f}",
            "Day Chg %": "{:+.2f}%",
            "Yesterday VAL (₹)": "₹{:.2f}",
            "% Below VAL": "{:.2f}%",
            "52W Low (₹)": "₹{:.2f}",
            "% From 52W Low": "{:+.2f}%",
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
    
    # Default selection: first bullish breakout, or first bearish, or first in universe
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
            ltp=stock_row["LTP"]
        )
        st.plotly_chart(fig, **get_width_kw(st.plotly_chart))
        
        # Summary details below chart
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("LTP", f"₹{stock_row['LTP']:.2f}", f"{stock_row['Chg%']:+.2f}%")
        c2.metric("PD_VAH (Green)", f"₹{stock_row['PD_VAH']:.2f}")
        c3.metric("PD_POC (Yellow)", f"₹{stock_row['PD_POC']:.2f}")
        c4.metric("PD_VAL (Red)", f"₹{stock_row['PD_VAL']:.2f}")
        c5.metric("52W High (Trigger 95%)", f"₹{stock_row['52W_High']:.2f}", f"₹{stock_row['52W_High']*high_proximity:.2f}")
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
        "Symbol", "LTP", "Chg%", "Signal", "PD_VAH", "Dist_VAH%", "PD_VAL", "Dist_VAL%", "PD_POC", "52W_High", "Dist_52W_High%", "52W_Low"
    ]].copy()

    st.dataframe(
        display_all.style.format({
            "LTP": "₹{:.2f}",
            "Chg%": "{:+.2f}%",
            "PD_VAH": "₹{:.2f}",
            "Dist_VAH%": "{:+.2f}%",
            "PD_VAL": "₹{:.2f}",
            "Dist_VAL%": "{:+.2f}%",
            "PD_POC": "₹{:.2f}",
            "52W_High": "₹{:.2f}",
            "Dist_52W_High%": "{:+.2f}%",
            "52W_Low": "₹{:.2f}"
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
    st.markdown("""
    ### 📖 Screener Strategy & Methodology
    
    This screener tracks high-probability breakout momentum in **NSE F&O Permitted Stocks** by uniting two institutional concepts:
    
    #### 1. Volume Profile (Previous Day Value Area)
    - **PD_VAH (Previous Day Value Area High)**: The upper bound price level enclosing **70%** of the previous trading day's traded volume.
    - **PD_VAL (Previous Day Value Area Low)**: The lower bound price level enclosing **70%** of the previous trading day's traded volume.
    - **PD_POC (Point of Control)**: The single price tick / bin that transacted the highest volume in yesterday's session.
    
    #### 2. Trailing 52-Week Extremes
    - **52W High**: Trailing 52-week highest traded price.
    - **52W Low**: Trailing 52-week lowest traded price.
    
    ---
    
    #### 🚀 Setup A: Bullish Breakout (Long)
    * **Condition 1**: `LTP > PD_VAH` (Price is accepted above yesterday's value area)
    * **Condition 2**: `LTP >= 52W_High * 0.95` (Price is within 5% of all-time/52-week highs, ensuring strength and momentum)
    
    #### 🔻 Setup B: Bearish Breakdown (Short)
    * **Condition 1**: `LTP < PD_VAL` (Price is accepted below yesterday's value area)
    * **Condition 2**: `LTP <= 52W_Low * 1.05` (Price is within 5% of 52-week lows, indicating severe breakdown pressure)
    """)

# Auto-refresh handling
if auto_refresh != "Off":
    delay = 30 if "30" in auto_refresh else (60 if "60" in auto_refresh else 120)
    time.sleep(delay)
    st.rerun()
