"""
Interactive Plotly Chart with Candlesticks, Volume Profile, VAH, VAL, POC & 52W Extremes
"""

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
    ltp: float
) -> go.Figure:
    """
    Renders an institutional-grade Candlestick chart with Volume Profile levels:
    - Candlesticks for previous day and current day
    - PD_VAH (Green Dashed Line)
    - PD_POC (Yellow Line)
    - PD_VAL (Red Dashed Line)
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

    # Add Reference Lines across the chart
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

    # 52W High threshold (95% level)
    if h52 and not np.isnan(h52):
        h52_trigger = h52 * 0.95
        fig.add_hline(
            y=h52_trigger,
            line_dash="dashdot",
            line_color="#00B0FF",
            line_width=1.2,
            annotation_text=f"52W High 95% Trigger (₹{h52_trigger:.2f})",
            annotation_position="top left",
            annotation_font=dict(color="#00B0FF", size=10),
            row=1, col=1
        )

    # Layout styling (Dark terminal theme)
    fig.update_layout(
        title=f"<b>{symbol}</b> — Intraday Price & Volume Profile Levels (LTP: ₹{ltp:.2f})",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=620,
        margin=dict(l=40, r=60, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig

