"""
Interactive Plotly Chart with Candlesticks, Volume Profile, CPR, Camarilla & Bollinger Bands
"""

from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_breakout_chart(
    symbol: str,
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    vp: dict,
    h52: float,
    l52: float,
    ltp: float,
    cpr: Optional[dict] = None,
    cam: Optional[dict] = None,
    bb: Optional[dict] = None
) -> go.Figure:
    """
    Renders an institutional-grade Candlestick chart with:
    - Candlesticks for previous day and current day
    - PD_VAH (Green Dashed Line)
    - PD_POC (Yellow Line)
    - PD_VAL (Red Dashed Line)
    - CPR Zone (Cyan / Blue Lines)
    - Camarilla H4 / L4 (Orange / Purple Breakout Levels)
    - Bollinger Bands (Upper, SMA20, Lower)
    - 52W High and 95% threshold line
    - Volume bars at the bottom
    """
    # Combine previous and current day candles
    combined = pd.concat([prev_df, curr_df]).sort_index()
    combined.index = pd.to_datetime(combined.index)

    # 2 rows: Main Price + Volume
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25]
    )

    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=combined.index,
            open=combined["Open"],
            high=combined["High"],
            low=combined["Low"],
            close=combined["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # Volume Bar trace
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(combined["Close"], combined["Open"])]
    fig.add_trace(
        go.Bar(
            x=combined.index,
            y=combined["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.6
        ),
        row=2, col=1
    )

    # Volume Profile Lines
    pd_vah = vp.get("PD_VAH")
    pd_val = vp.get("PD_VAL")
    pd_poc = vp.get("PD_POC")

    if pd_vah and not np.isnan(pd_vah):
        fig.add_hline(
            y=pd_vah, 
            line_dash="dash", 
            line_color="#00E676", 
            line_width=1.5,
            annotation_text=f"PD_VAH (₹{pd_vah})", 
            annotation_position="top right",
            annotation_font=dict(color="#00E676", size=11),
            row=1, col=1
        )

    if pd_poc and not np.isnan(pd_poc):
        fig.add_hline(
            y=pd_poc, 
            line_dash="dot", 
            line_color="#FFD600", 
            line_width=1.5,
            annotation_text=f"PD_POC (₹{pd_poc})", 
            annotation_position="right",
            annotation_font=dict(color="#FFD600", size=11),
            row=1, col=1
        )

    if pd_val and not np.isnan(pd_val):
        fig.add_hline(
            y=pd_val, 
            line_dash="dash", 
            line_color="#FF5252", 
            line_width=1.5,
            annotation_text=f"PD_VAL (₹{pd_val})", 
            annotation_position="bottom right",
            annotation_font=dict(color="#FF5252", size=11),
            row=1, col=1
        )

    # CPR Lines (Money Zone)
    if cpr:
        cpr_top = cpr.get("top")
        cpr_bot = cpr.get("bottom")
        if cpr_top and not np.isnan(cpr_top):
            fig.add_hline(
                y=cpr_top,
                line_dash="dot",
                line_color="#29B6F6",
                line_width=1.2,
                annotation_text=f"CPR Top (₹{cpr_top})",
                annotation_position="top left",
                annotation_font=dict(color="#29B6F6", size=10),
                row=1, col=1
            )
        if cpr_bot and not np.isnan(cpr_bot):
            fig.add_hline(
                y=cpr_bot,
                line_dash="dot",
                line_color="#29B6F6",
                line_width=1.2,
                annotation_text=f"CPR Bottom (₹{cpr_bot})",
                annotation_position="bottom left",
                annotation_font=dict(color="#29B6F6", size=10),
                row=1, col=1
            )

    # Camarilla H4 and L4 Breakout Lines
    if cam:
        h4 = cam.get("h4")
        l4 = cam.get("l4")
        if h4 and not np.isnan(h4):
            fig.add_hline(
                y=h4,
                line_dash="longdash",
                line_color="#FF9800",
                line_width=1.2,
                annotation_text=f"Cam H4 Breakout (₹{h4})",
                annotation_position="top right",
                annotation_font=dict(color="#FF9800", size=10),
                row=1, col=1
            )
        if l4 and not np.isnan(l4):
            fig.add_hline(
                y=l4,
                line_dash="longdash",
                line_color="#E040FB",
                line_width=1.2,
                annotation_text=f"Cam L4 Breakdown (₹{l4})",
                annotation_position="bottom right",
                annotation_font=dict(color="#E040FB", size=10),
                row=1, col=1
            )

    # Bollinger Bands
    if bb:
        bb_upper = bb.get("upper")
        bb_lower = bb.get("lower")
        bb_mid = bb.get("sma20")
        if bb_upper and not np.isnan(bb_upper):
            fig.add_hline(
                y=bb_upper,
                line_dash="dash",
                line_color="#4FC3F7",
                line_width=1.0,
                annotation_text=f"BB Upper (₹{bb_upper})",
                annotation_position="top left",
                annotation_font=dict(color="#4FC3F7", size=9),
                row=1, col=1
            )
        if bb_mid and not np.isnan(bb_mid):
            fig.add_hline(
                y=bb_mid,
                line_dash="dot",
                line_color="#FFA726",
                line_width=1.0,
                annotation_text=f"BB SMA20 (₹{bb_mid})",
                annotation_position="left",
                annotation_font=dict(color="#FFA726", size=9),
                row=1, col=1
            )
        if bb_lower and not np.isnan(bb_lower):
            fig.add_hline(
                y=bb_lower,
                line_dash="dash",
                line_color="#4FC3F7",
                line_width=1.0,
                annotation_text=f"BB Lower (₹{bb_lower})",
                annotation_position="bottom left",
                annotation_font=dict(color="#4FC3F7", size=9),
                row=1, col=1
            )

    # 52W High threshold (95% level)
    if h52 and not np.isnan(h52):
        h52_trigger = h52 * 0.95
        fig.add_hline(
            y=h52_trigger,
            line_dash="dashdot",
            line_color="#AB47BC",
            line_width=1.2,
            annotation_text=f"52W High 95% (₹{h52_trigger:.2f})",
            annotation_position="top left",
            annotation_font=dict(color="#AB47BC", size=10),
            row=1, col=1
        )

    # Layout styling (Dark terminal theme)
    fig.update_layout(
        title=f"<b>{symbol}</b> — Intraday Price, Volume Profile, CPR, Camarilla & Bollinger Bands (LTP: ₹{ltp:.2f})",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=660,
        margin=dict(l=40, r=60, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig
