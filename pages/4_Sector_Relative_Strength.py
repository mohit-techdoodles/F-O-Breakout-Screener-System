"""
Page: 4_Sector_Relative_Strength.py
Sector Relative Strength vs. Nifty 750 (Mansfield RS & Sector Rotation Pipeline)

Implements Developer Setup, Mathematical Specifications, Data Pipeline Architecture,
and Production Safeguards from the Technical Implementation Guide:
- Benchmark: Nifty 750 (Nifty Total Market Index)
- Simple RS Ratio (St / Bt) & 50-Day Moving Average Crossover Signal
- Mansfield Relative Strength (Zero-Line Normalized: (RS / SMA_n(RS) - 1) * 100)
- Multi-Horizon Excess Returns (1M / 21d, 3M / 63d, 6M / 126d, 12M / 252d)
- Weighted Composite RS Score for Cross-Sectoral Ranking
- Output Contract: sector_rs_ranking.csv
"""

import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import rs_calculator
from rs_calculator import (
    RSConfig,
    SECTOR_TICKER_MAP,
    BENCHMARK_TICKER_DEFAULT,
    fetch_aligned_sector_closes,
    load_aligned_closes,
    resample_weekly,
    compute_relative_strength,
    save_ranking_csv
)
from pwa_launcher import inject_pwa_headers, render_app_launcher_sidebar


# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Sector Relative Strength vs Nifty 750 | Mansfield RS",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject PWA Progressive Web App headers & Launcher
inject_pwa_headers()
render_app_launcher_sidebar()

# Custom CSS styling for metric cards and high contrast tables
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-lbl {
        font-size: 0.80rem;
        color: #787B86;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #D1D4DC;
    }
    .lead-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .lead-pos {
        background: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid rgba(0, 230, 118, 0.4);
    }
    .lead-neg {
        background: rgba(255, 82, 82, 0.15);
        color: #FF5252;
        border: 1px solid rgba(255, 82, 82, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ----------------- Sidebar Configuration -----------------
st.sidebar.markdown("### 🏛️ Sector RS Engine Settings")
st.sidebar.caption("Benchmark: **Nifty 750 (Nifty Total Market Index)**")

bench_choice = st.sidebar.selectbox(
    "🎯 Benchmark Index:",
    ["Nifty Total Market (Broad 750 / 500 Proxy)", "Nifty 50 (^NSEI)"],
    index=0,
    help="Section 1: Nifty Total Market Index combines Nifty 500 and Microcap 250 for broad 750-stock baseline."
)
bench_ticker = BENCHMARK_TICKER_DEFAULT if "Total Market" in bench_choice else "^NSEI"

tf_choice = st.sidebar.radio(
    "⏱️ Timeframe Analysis:",
    ["Daily (200-Session Mansfield RS)", "Weekly (52-Week Mansfield RS)"],
    index=0,
    help="Section 7: Weekly resampling option aggregates Friday closes for classic Stan Weinstein 52-week Mansfield RS."
)
is_weekly = "Weekly" in tf_choice

mansfield_win_default = 52 if is_weekly else 200
signal_ma_default = 20 if is_weekly else 50

with st.sidebar.expander("⚙️ Advanced Lookback & Weights", expanded=False):
    mansfield_win = st.slider("Mansfield Window (n):", 20, 250, mansfield_win_default, 5)
    signal_ma_win = st.slider("Signal MA Window on RS Ratio:", 10, 100, signal_ma_default, 5)
    
    st.markdown("**Composite RS Weights:**")
    st.caption("Default: 50% 3M (63d) + 30% 6M (126d) + 20% 12M (252d)")
    w_3m = st.slider("Weight 3-Month (63d):", 0.0, 1.0, 0.50, 0.05)
    w_6m = st.slider("Weight 6-Month (126d):", 0.0, 1.0, 0.30, 0.05)
    w_12m = st.slider("Weight 12-Month (252d):", 0.0, 1.0, 0.20, 0.05)

cfg = RSConfig(
    benchmark_col="NIFTY_750",
    mansfield_window=mansfield_win,
    signal_ma_window=signal_ma_win,
    return_windows=(5, 13, 26, 52) if is_weekly else (21, 63, 126, 252),
    composite_weights={13: w_3m, 26: w_6m, 52: w_12m} if is_weekly else {63: w_3m, 126: w_6m, 252: w_12m}
)

if st.sidebar.button("🔄 Refresh Market Data", width="stretch"):
    st.cache_data.clear()


# ----------------- Data Caching & Ingestion -----------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_aligned_data_cached(b_ticker: str, weekly_mode: bool) -> pd.DataFrame:
    raw_df = fetch_aligned_sector_closes(period="2y", benchmark_col="NIFTY_750", benchmark_ticker=b_ticker)
    if weekly_mode:
        return resample_weekly(raw_df)
    return raw_df


with st.spinner("Synchronizing EOD data for Nifty 750 and 13 NSE Sectoral Indices..."):
    try:
        aligned_closes = get_aligned_data_cached(bench_ticker, is_weekly)
        results = compute_relative_strength(aligned_closes, cfg=cfg)
        ranking_df = results["ranking_table"]
        rs_ratio_df = results["rs_ratio"]
        signal_ma_df = results["signal_ma"]
        mansfield_rs_df = results["mansfield_rs"]
        excess_returns_dict = results["excess_returns"]
        latest_date = results["latest_date"]
    except Exception as e:
        st.error(f"❌ Error loading Sector RS pipeline: {e}")
        st.stop()


# ----------------- Header Banner -----------------
st.markdown("## 🏛️ Sector Relative Strength vs. Nifty 750")
st.markdown(
    f"""
    <div style="font-size: 14px; color: #9E9E9E; margin-top: -8px; margin-bottom: 16px;">
        Institutional Sector Rotation Pipeline & Mansfield Relative Strength Engine | 
        <b>Benchmark:</b> {bench_choice} | <b>Timeframe:</b> {tf_choice} | <b>Latest Session:</b> {latest_date}
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- Top Metric KPI Row -----------------
top_sector = ranking_df.iloc[0]["sector"] if not ranking_df.empty else "N/A"
top_score = ranking_df.iloc[0]["composite_rs_score"] if not ranking_df.empty else 0.0

runner_up = ranking_df.iloc[1]["sector"] if len(ranking_df) > 1 else "N/A"
runner_score = ranking_df.iloc[1]["composite_rs_score"] if len(ranking_df) > 1 else 0.0

outperforming_count = int((ranking_df["mansfield_rs"] > 0).sum())
above_ma_count = int(ranking_df["above_signal_ma"].sum())
lagging_count = int((ranking_df["mansfield_rs"] < 0).sum())

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">🥇 #1 Sector Leader</div>
        <div class="metric-val" style="color: #00E676;">{top_sector}</div>
        <div style="font-size: 0.75rem; color: #81C784;">Composite: +{top_score:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">🥈 #2 Sector Momentum</div>
        <div class="metric-val" style="color: #64B5F6;">{runner_up}</div>
        <div style="font-size: 0.75rem; color: #90CAF9;">Composite: +{runner_score:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">🚀 Mansfield RS > 0</div>
        <div class="metric-val" style="color: #FFD54F;">{outperforming_count} <span style="font-size: 1rem; color: #787B86;">/ {len(ranking_df)}</span></div>
        <div style="font-size: 0.75rem; color: #FFE082;">Outperforming broad market</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">⚡ Above 50-Day Signal MA</div>
        <div class="metric-val" style="color: #00E5FF;">{above_ma_count} <span style="font-size: 1rem; color: #787B86;">/ {len(ranking_df)}</span></div>
        <div style="font-size: 0.75rem; color: #80D8FF;">Positive RS slope trend</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">🔻 Lagging Sectors</div>
        <div class="metric-val" style="color: #FF5252;">{lagging_count} <span style="font-size: 1rem; color: #787B86;">/ {len(ranking_df)}</span></div>
        <div style="font-size: 0.75rem; color: #FF8A80;">Mansfield RS &lt; 0.0</div>
    </div>
    """, unsafe_allow_html=True)

with m6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">📊 History Aligned</div>
        <div class="metric-val" style="color: #AB47BC;">{len(aligned_closes)}</div>
        <div style="font-size: 0.75rem; color: #CE93D8;">Synchronized sessions</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# MAIN VIEW: RANKING TABLE & VISUAL ANALYTICS
# =====================================================================
tab_tbl, tab_mrs, tab_ratio, tab_excess, tab_stocks, tab_upload, tab_spec = st.tabs([
    "📋 Sector RS Ranking Table",
    "📈 Mansfield RS (Zero-Line)",
    "🎯 Simple RS Ratio & 50 SMA",
    "📊 Multi-Horizon Excess Returns",
    "🔍 F&O Stock-to-Sector Matrix",
    "📁 CSV Ingestion (Bhavcopy)",
    "📖 Architecture & Rules"
])

# ----------------- TAB 1: SECTOR RS RANKING TABLE -----------------
with tab_tbl:
    st.markdown("#### 📋 Primary Screener Artifact: `sector_rs_ranking.csv`")
    st.caption("Normalized cross-sector comparison ranked by weighted Composite RS Score. Mansfield RS > 0 indicates institutional outperformance relative to Nifty 750.")

    # Format dataframe for display
    display_df = ranking_df.copy()
    display_df["Trend_Signal"] = display_df["above_signal_ma"].apply(
        lambda x: "🟢 Above 50 SMA (Bullish)" if x else "🔴 Below 50 SMA (Lagging)"
    )

    format_dict = {
        "rs_ratio": "{:.4f}",
        "mansfield_rs": "{:+.2f}",
        "composite_rs_score": "{:+.2f}"
    }
    for col in display_df.columns:
        if col.startswith("excess_"):
            format_dict[col] = "{:+.2f}%"

    h_tbl = min(550, max(300, 50 + len(display_df) * 36))
    
    styler_obj = display_df.style.format(format_dict)
    try:
        styler_obj = styler_obj.background_gradient(
            subset=["mansfield_rs", "composite_rs_score"],
            cmap="RdYlGn",
            vmin=-10,
            vmax=10
        )
    except Exception:
        def _color_score(val):
            try:
                v = float(val)
                if v > 0:
                    return "color: #00E676; font-weight: 600;"
                elif v < 0:
                    return "color: #FF5252; font-weight: 600;"
            except Exception:
                pass
            return ""
        if hasattr(styler_obj, "map"):
            styler_obj = styler_obj.map(_color_score, subset=["mansfield_rs", "composite_rs_score"])
        elif hasattr(styler_obj, "applymap"):
            styler_obj = styler_obj.applymap(_color_score, subset=["mansfield_rs", "composite_rs_score"])

    st.dataframe(
        styler_obj,
        width="stretch",
        height=h_tbl,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "sector": st.column_config.TextColumn("Sectoral Index", width="medium"),
            "rs_ratio": st.column_config.NumberColumn("RS Ratio (St / Bt)", help="Raw relative valuation ratio against Nifty 750.", format="%.4f"),
            "Trend_Signal": st.column_config.TextColumn("Signal MA (50)", width="medium"),
            "above_signal_ma": st.column_config.CheckboxColumn("Above MA", width="small"),
            "mansfield_rs": st.column_config.NumberColumn("Mansfield RS", help="Zero-line normalized indicator: ((RS / SMA_n(RS)) - 1) * 100.", format="%.2f"),
            "composite_rs_score": st.column_config.NumberColumn("Composite RS Score", help="Weighted multi-horizon excess return score (50% 3M + 30% 6M + 20% 12M).", format="%.2f"),
        }
    )

    # One-Click CSV Export matching Section 6 Output Contract
    csv_bytes = ranking_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Official sector_rs_ranking.csv",
        data=csv_bytes,
        file_name=f"sector_rs_ranking_{latest_date}.csv",
        mime="text/csv",
        width="stretch"
    )


# ----------------- TAB 2: MANSFIELD RS CHART (ZERO-LINE) -----------------
with tab_mrs:
    st.markdown("#### 📈 Mansfield Relative Strength (Zero-Line Centered Comparison)")
    st.caption("Standardizes sectors regardless of absolute point scales. Green bars indicate institutional leadership (> 0); Red bars indicate structural lag (< 0).")

    # Bar chart of latest Mansfield RS
    sorted_mrs = ranking_df.sort_values(by="mansfield_rs", ascending=True)
    colors = ["#00E676" if x >= 0 else "#FF5252" for x in sorted_mrs["mansfield_rs"]]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=sorted_mrs["sector"],
        x=sorted_mrs["mansfield_rs"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1E222D", width=1)),
        text=[f"{v:+.2f}" for v in sorted_mrs["mansfield_rs"]],
        textposition="outside"
    ))
    fig_bar.add_vline(x=0.0, line_width=2, line_dash="solid", line_color="#FFFFFF")
    fig_bar.update_layout(
        title=f"Cross-Sector Mansfield Relative Strength vs Zero Line ({latest_date})",
        template="plotly_dark",
        xaxis_title="Mansfield RS Score (Zero-Centered)",
        yaxis_title="Sector",
        height=480,
        margin=dict(l=20, r=40, t=50, b=30)
    )
    st.plotly_chart(fig_bar, width="stretch")

    # Multi-sector historical timeline chart
    st.markdown("##### ⏱️ Historical Mansfield RS Trajectory (Rolling Rotation)")
    selected_sectors = st.multiselect(
        "Select Sectors to Overlay:",
        options=list(mansfield_rs_df.columns),
        default=list(ranking_df["sector"].iloc[:5])
    )
    if selected_sectors:
        fig_ts = go.Figure()
        for s in selected_sectors:
            fig_ts.add_trace(go.Scatter(
                x=mansfield_rs_df.index[-180:],
                y=mansfield_rs_df[s].iloc[-180:],
                mode="lines",
                name=s,
                line=dict(width=2)
            ))
        fig_ts.add_hline(y=0.0, line_width=2, line_dash="dash", line_color="#E0E0E0")
        fig_ts.update_layout(
            title="Mansfield RS 6-Month Trajectory (Zero-Line Normalized)",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Mansfield RS",
            height=400,
            margin=dict(l=20, r=20, t=40, b=30)
        )
        st.plotly_chart(fig_ts, width="stretch")


# ----------------- TAB 3: SIMPLE RS RATIO & 50 SMA -----------------
with tab_ratio:
    st.markdown("#### 🎯 Simple RS Ratio ($S_t / B_t$) vs. 50-Period Signal MA")
    st.caption("Section 3.1: Upward crossover of Simple RS Ratio above its 50-day moving average indicates emerging sector leadership.")

    sec_pick = st.selectbox("Select Sectoral Index to Inspect:", list(rs_ratio_df.columns), index=0)

    rs_ser = rs_ratio_df[sec_pick].dropna()
    ma_ser = signal_ma_df[sec_pick].dropna()
    common_idx = rs_ser.index.intersection(ma_ser.index)[-180:]

    fig_ratio = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)

    # Row 1: RS Ratio & Signal MA
    fig_ratio.add_trace(go.Scatter(
        x=common_idx,
        y=rs_ser.loc[common_idx],
        name=f"{sec_pick} RS Ratio",
        line=dict(color="#00E5FF", width=2.5)
    ), row=1, col=1)

    fig_ratio.add_trace(go.Scatter(
        x=common_idx,
        y=ma_ser.loc[common_idx],
        name=f"50 SMA (Signal MA)",
        line=dict(color="#FFB300", width=1.8, dash="dash")
    ), row=1, col=1)

    # Row 2: Spread (RS - SMA)
    spread = rs_ser.loc[common_idx] - ma_ser.loc[common_idx]
    spread_colors = ["#00E676" if x >= 0 else "#FF5252" for x in spread]
    fig_ratio.add_trace(go.Bar(
        x=common_idx,
        y=spread,
        name="Spread (RS - 50 SMA)",
        marker=dict(color=spread_colors)
    ), row=2, col=1)

    fig_ratio.update_layout(
        title=f"{sec_pick} Simple RS Ratio vs 50-Day Moving Average ({bench_choice})",
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=50, b=30)
    )
    st.plotly_chart(fig_ratio, width="stretch")


# ----------------- TAB 4: MULTI-HORIZON EXCESS RETURNS -----------------
with tab_excess:
    st.markdown("#### 📊 Multi-Horizon Percentage Excess Return Matrix")
    st.caption("Section 3.3: Percentage return over k sessions minus Nifty 750 return: [(St / St-k - 1) - (Bt / Bt-k - 1)] * 100.")

    cols_excess = [c for c in ranking_df.columns if c.startswith("excess_")]
    fig_heat = go.Figure(data=go.Heatmap(
        z=ranking_df[cols_excess].values,
        x=[c.replace("excess_", "").replace("_pct", "").upper() for c in cols_excess],
        y=ranking_df["sector"],
        colorscale="RdYlGn",
        zmid=0.0,
        text=ranking_df[cols_excess].map(lambda x: f"{x:+.1f}%").values,
        texttemplate="%{text}",
        colorbar=dict(title="Excess Return %")
    ))
    fig_heat.update_layout(
        title="Excess Return Heatmap vs. Nifty 750 (1M, 3M, 6M, 12M Horizons)",
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=50, b=30)
    )
    st.plotly_chart(fig_heat, width="stretch")


# ----------------- TAB 5: F&O STOCK-TO-SECTOR MATRIX -----------------
with tab_stocks:
    st.markdown("#### 🔍 Stock-to-Sector Relative Strength Alignment")
    st.caption("Pair breakout setups with sector leadership tailwinds. High-conviction trades occur when both the stock and its sector outperform Nifty 750.")

    stock_sector_map = {
        "INFY": "NIFTY_IT", "TCS": "NIFTY_IT", "HCLTECH": "NIFTY_IT", "WIPRO": "NIFTY_IT", "TECHM": "NIFTY_IT", "COFORGE": "NIFTY_IT",
        "HDFCBANK": "NIFTY_BANK", "ICICIBANK": "NIFTY_BANK", "KOTAKBANK": "NIFTY_BANK", "AXISBANK": "NIFTY_BANK", "INDUSINDBK": "NIFTY_BANK",
        "SBIN": "NIFTY_PSU_BANK", "BANKBARODA": "NIFTY_PSU_BANK", "PNB": "NIFTY_PSU_BANK", "CANBK": "NIFTY_PSU_BANK",
        "BAJFINANCE": "NIFTY_FIN_SERVICE", "BAJAJFINSV": "NIFTY_FIN_SERVICE", "CHOLAFIN": "NIFTY_FIN_SERVICE", "MUTHOOTFIN": "NIFTY_FIN_SERVICE",
        "MARUTI": "NIFTY_AUTO", "M&M": "NIFTY_AUTO", "BAJAJ-AUTO": "NIFTY_AUTO", "EICHERMOT": "NIFTY_AUTO", "TATAMOTORS": "NIFTY_AUTO", "TVSMOTOR": "NIFTY_AUTO",
        "TATASTEEL": "NIFTY_METAL", "JSWSTEEL": "NIFTY_METAL", "HINDALCO": "NIFTY_METAL", "JINDALSTEL": "NIFTY_METAL", "VEDL": "NIFTY_METAL",
        "SUNPHARMA": "NIFTY_PHARMA", "CIPLA": "NIFTY_PHARMA", "DRREDDY": "NIFTY_PHARMA", "DIVISLAB": "NIFTY_PHARMA", "LUPIN": "NIFTY_PHARMA",
        "ITC": "NIFTY_FMCG", "HINDUNILVR": "NIFTY_FMCG", "NESTLEIND": "NIFTY_FMCG", "BRITANNIA": "NIFTY_FMCG", "TATACONSUM": "NIFTY_FMCG", "DABUR": "NIFTY_FMCG",
        "RELIANCE": "NIFTY_ENERGY", "ONGC": "NIFTY_ENERGY", "NTPC": "NIFTY_ENERGY", "POWERGRID": "NIFTY_ENERGY", "COALINDIA": "NIFTY_ENERGY", "BPCL": "NIFTY_ENERGY",
        "DLF": "NIFTY_REALTY", "GODREJPROP": "NIFTY_REALTY", "OBEROIRLTY": "NIFTY_REALTY",
        "LT": "NIFTY_INFRA", "ULTRACEMCO": "NIFTY_INFRA", "AMBUJACEM": "NIFTY_INFRA",
        "TITAN": "NIFTY_CONSUMPTION", "TRENT": "NIFTY_CONSUMPTION", "ASIANPAINT": "NIFTY_CONSUMPTION"
    }

    sym_inspect = st.selectbox("Select F&O Symbol:", sorted(list(stock_sector_map.keys())), index=0)
    matched_sector = stock_sector_map.get(sym_inspect, "NIFTY_50")

    sector_rank_row = ranking_df[ranking_df["sector"] == matched_sector]
    if not sector_rank_row.empty:
        s_row = sector_rank_row.iloc[0]
        s_rank = int(s_row["rank"])
        s_mrs = float(s_row["mansfield_rs"])
        s_comp = float(s_row["composite_rs_score"])
        s_ma = bool(s_row["above_signal_ma"])

        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        col_st1.metric(f"{sym_inspect} Sector", matched_sector)
        col_st2.metric("Sector RS Rank", f"#{s_rank} of {len(ranking_df)}")
        col_st3.metric("Sector Mansfield RS", f"{s_mrs:+.2f}")
        col_st4.metric("50-Day Signal MA", "🟢 Above (Bullish)" if s_ma else "🔴 Below (Lagging)")

        if s_rank <= 3 and s_mrs > 0:
            st.success(f"🚀 **Strong Sector Tailwinds:** {sym_inspect}'s parent sector `{matched_sector}` is currently in the **Top 3 Leadership** with Mansfield RS of {s_mrs:+.2f}. Breakouts on this stock have institutional confirmation!")
        elif s_mrs > 0:
            st.info(f"🟢 **Positive Sector RS:** `{matched_sector}` is outperforming Nifty 750 (Mansfield RS: {s_mrs:+.2f}).")
        else:
            st.warning(f"⚠️ **Sector Headwind Alert:** `{matched_sector}` is lagging Nifty 750 (Mansfield RS: {s_mrs:+.2f}, Rank #{s_rank}). Long breakouts may face sector drag.")


# ----------------- TAB 6: CSV INGESTION (BHAVCOPY) -----------------
with tab_upload:
    st.markdown("#### 📁 Ingest Official Historical CSV Files")
    st.caption("Section 2.1: Upload wide-format EOD closing files from `niftyindices.com` or NSE index Bhavcopy archives.")

    uploaded_file = st.file_uploader(
        "Upload Wide-Format Historical Closes CSV (Columns: date, NIFTY_750, NIFTY_IT, NIFTY_BANK, ...)",
        type=["csv"]
    )
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file, parse_dates=["date"]).set_index("date").sort_index().dropna(how="any")
            st.write("Uploaded Data Preview (Last 5 Rows):", custom_df.tail())
            custom_res = compute_relative_strength(custom_df, cfg=cfg)
            st.success("✅ Custom CSV Relative Strength Computed Successfully!")
            st.dataframe(custom_res["ranking_table"], width="stretch")
        except Exception as e:
            st.error(f"❌ Error parsing custom CSV: {e}")


# ----------------- TAB 7: ARCHITECTURE & RULES -----------------
with tab_spec:
    st.markdown(r"""
    ### 🏛️ Sector Relative Strength Technical Architecture Guide

    #### 1. Scope & Benchmark
    - **Benchmark Index:** **Nifty 750 (Nifty Total Market Index)**. Combines Nifty 500 and Microcap 250, capturing 750 stocks for an unbiased broad-market baseline.
    - **Target Assets:** Major NSE Sectoral Indices (`NIFTY_IT`, `NIFTY_BANK`, `NIFTY_AUTO`, `NIFTY_METAL`, `NIFTY_PHARMA`, `NIFTY_FMCG`, `NIFTY_ENERGY`, `NIFTY_REALTY`, etc.).

    #### 2. Mathematical Specifications
    - **Simple RS Ratio:**
      $$\text{RS Ratio}_t = \frac{S_t}{B_t}$$
      An upward cross of RS Ratio above its 50-day SMA signals emerging leadership.
    - **Mansfield Relative Strength (Zero-Line Centered):**
      $$\text{Mansfield RS}_t = \left(\frac{\text{RS Ratio}_t}{\text{SMA}_n(\text{RS Ratio})} - 1\right) \times 100$$
      Where $n = 200$ sessions on daily data or $n = 52$ weeks on weekly resampled Friday closes.
    - **Multi-Horizon Excess Return:**
      $$\text{Excess Return}_k = \left[\left(\frac{S_t}{S_{t-k}} - 1\right) - \left(\frac{B_t}{B_{t-k}} - 1\right)\right] \times 100$$
    - **Composite RS Score:**
      $$\text{Composite} = 0.50 \times \text{Excess}_{63} + 0.30 \times \text{Excess}_{126} + 0.20 \times \text{Excess}_{252}$$

    #### 3. Output Contract (`sector_rs_ranking.csv`)
    The primary screener artifact exports the typed schema:
    `rank`, `sector`, `rs_ratio`, `above_signal_ma`, `mansfield_rs`, `excess_*d_pct`, `composite_rs_score`.
    """)
