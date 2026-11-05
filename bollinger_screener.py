"""
Bollinger Bands Squeeze & Breakout Detection Engine
Implements Developer Specification: Bollinger Bands Screener (Equity F&O)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_multiplier: float = 2.0,
    squeeze_lookback: int = 120,
    tolerance: float = 1.10
) -> Dict[str, any]:
    """
    Calculates Bollinger Bands, Bandwidth, and Squeeze conditions as defined in developer spec:
    
    1. SMA20: 20-period Simple Moving Average
    2. Upper Band: SMA20 + (StdDev * 2)
    3. Lower Band: SMA20 - (StdDev * 2)
    4. Bandwidth: (Upper - Lower) / SMA20
    5. Min Bandwidth: 120-period minimum Bandwidth
    6. Squeeze: Bandwidth <= (Min Bandwidth * 1.10)
    7. Breakouts: Close > Upper Band (Bullish) or Close < Lower Band (Bearish)
    8. Master Confluence: Squeeze on T-1 (or within recent 3 bars) & Breakout on T0
    """
    if df.empty or len(df) < period:
        return {
            "sma20": np.nan,
            "upper": np.nan,
            "lower": np.nan,
            "bandwidth": np.nan,
            "min_bandwidth": np.nan,
            "is_squeezing": False,
            "was_squeezing": False,
            "bullish_breakout": False,
            "bearish_breakout": False,
            "bullish_squeeze_signal": False,
            "bearish_squeeze_signal": False,
            "status": "Insufficient Data"
        }

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < period:
        return {
            "sma20": np.nan,
            "upper": np.nan,
            "lower": np.nan,
            "bandwidth": np.nan,
            "min_bandwidth": np.nan,
            "is_squeezing": False,
            "was_squeezing": False,
            "bullish_breakout": False,
            "bearish_breakout": False,
            "bullish_squeeze_signal": False,
            "bearish_squeeze_signal": False,
            "status": "Insufficient Data"
        }
    sma20 = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    upper = sma20 + (std_dev * std_multiplier)
    lower = sma20 - (std_dev * std_multiplier)

    # Bandwidth normalized by SMA20
    bandwidth = (upper - lower) / sma20

    # Min bandwidth over historical lookback (default: 120 periods)
    lookback = min(squeeze_lookback, len(bandwidth))
    min_bandwidth = bandwidth.rolling(window=lookback, min_periods=max(20, lookback // 2)).min()

    # Squeeze condition: Bandwidth <= (Min Bandwidth * 1.10)
    is_squeeze = bandwidth <= (min_bandwidth * tolerance)

    # Current values (T0)
    curr_c = float(close.iloc[-1])
    curr_sma = float(sma20.iloc[-1])
    curr_upper = float(upper.iloc[-1])
    curr_lower = float(lower.iloc[-1])
    curr_bw = float(bandwidth.iloc[-1])
    curr_min_bw = float(min_bandwidth.iloc[-1]) if not np.isnan(min_bandwidth.iloc[-1]) else curr_bw
    curr_is_sq = bool(is_squeeze.iloc[-1])

    # Squeeze on prior period(s) - T-1 or within last 3 sessions
    was_sq_t1 = bool(is_squeeze.iloc[-2]) if len(is_squeeze) >= 2 else False
    was_sq_recent = bool(is_squeeze.iloc[-4:-1].any()) if len(is_squeeze) >= 4 else was_sq_t1

    # Breakout conditions on current bar
    curr_bull_bo = curr_c > curr_upper
    curr_bear_bo = curr_c < curr_lower

    # Master Confluence Signals
    bull_sq_signal = was_sq_recent and curr_bull_bo
    bear_sq_signal = was_sq_recent and curr_bear_bo

    # Categorize status string
    if bull_sq_signal:
        status = "💥 Bullish Squeeze Breakout"
    elif bear_sq_signal:
        status = "🔻 Bearish Squeeze Breakdown"
    elif curr_is_sq:
        status = "🌀 In Tight Squeeze"
    elif curr_bull_bo:
        status = "⚡ Above Upper Band"
    elif curr_bear_bo:
        status = "⚡ Below Lower Band"
    else:
        status = "Normal"

    return {
        "sma20": round(curr_sma, 2),
        "upper": round(curr_upper, 2),
        "lower": round(curr_lower, 2),
        "bandwidth": round(curr_bw * 100.0, 2),  # in percentage
        "min_bandwidth": round(curr_min_bw * 100.0, 2),  # in percentage
        "is_squeezing": curr_is_sq,
        "was_squeezing": was_sq_recent,
        "bullish_breakout": curr_bull_bo,
        "bearish_breakout": curr_bear_bo,
        "bullish_squeeze_signal": bull_sq_signal,
        "bearish_squeeze_signal": bear_sq_signal,
        "status": status,
        "series": {
            "sma20": sma20,
            "upper": upper,
            "lower": lower
        }
    }
