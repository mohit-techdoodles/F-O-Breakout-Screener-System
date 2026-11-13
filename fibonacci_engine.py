"""
Fibonacci Retracement Engine
Section 4 of Screener Architecture (November 2026)
Calculates dynamic Fibonacci retracements based on Fractal Pivot Points:
- 1-Hour Swings (Macro Context): 20 to 30 hourly candles (approx. 3 to 5 trading days)
- 15-Minute Swings (Micro Context): 10 to 15 15m candles (approx. 2.5 hours)
- The Golden Pocket (0.618): High-probability setup when 1-Hour Break-in (Option C) occurs at Fib 0.618
- Automated Targets: Inverse extensions (-0.272 and -0.618)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


def calculate_fractal_pivots(df_resampled: pd.DataFrame, window_bars: int = 25) -> Tuple[float, float, str, int, int]:
    """
    Identifies Fractal Pivot Points (standard 5-bar fractal) over a backward-looking window.
    A bar is a High Fractal if its High is greater than the 2 bars before and 2 bars after.
    A bar is a Low Fractal if its Low is lower than the 2 bars before and 2 bars after.
    Fallback: Extrema in rolling window if strict 5-bar fractal hasn't completed on recent bars.
    """
    if df_resampled.empty or len(df_resampled) < 5:
        return np.nan, np.nan, "Neutral", 0, 0

    w_df = df_resampled.iloc[-window_bars:].copy()
    n = len(w_df)

    fractal_highs = []
    fractal_lows = []

    highs = w_df["High"].values
    lows = w_df["Low"].values

    # 5-bar fractal detection: requires 2 bars left and 2 bars right
    for i in range(2, n - 2):
        if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
            highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
            fractal_highs.append((i, highs[i]))

        if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
            lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
            fractal_lows.append((i, lows[i]))

    # Determine Swing High
    if fractal_highs:
        # Highest fractal in window
        best_high = max(fractal_highs, key=lambda x: x[1])
        swing_high = float(best_high[1])
        high_pos = best_high[0]
    else:
        # Fallback to absolute max in window
        high_pos = int(np.argmax(highs))
        swing_high = float(highs[high_pos])

    # Determine Swing Low
    if fractal_lows:
        # Lowest fractal in window
        best_low = min(fractal_lows, key=lambda x: x[1])
        swing_low = float(best_low[1])
        low_pos = best_low[0]
    else:
        # Fallback to absolute min in window
        low_pos = int(np.argmin(lows))
        swing_low = float(lows[low_pos])

    # Determine Swing Direction
    if low_pos < high_pos:
        swing_type = "Bullish"  # Low occurred before High (Up-Swing)
    elif high_pos < low_pos:
        swing_type = "Bearish"  # High occurred before Low (Down-Swing)
    else:
        swing_type = "Neutral"

    return swing_high, swing_low, swing_type, high_pos, low_pos


def calculate_fibonacci_levels(
    sub_intra: pd.DataFrame,
    ltp: float,
    is_1h_break_in: bool = False,
    timeframe_mode: str = "1-Hour (Macro)"
) -> Dict[str, Any]:
    """
    Computes Fibonacci Retracement and Automated Inverse Profit Targets:
    - 1-Hour Swings (Macro): 25 hourly candles
    - 15-Minute Swings (Micro): 12 15m candles
    - The Golden Pocket (0.618)
    - Inverse Extensions: Target 1 (-0.272) and Target 2 (-0.618)
    """
    default_res = {
        "swing_high": np.nan,
        "swing_low": np.nan,
        "swing_range": 0.0,
        "swing_type": "Neutral",
        "fib_0236": np.nan,
        "fib_0382": np.nan,
        "fib_0500": np.nan,
        "fib_0618": np.nan,
        "fib_0786": np.nan,
        "target_1": np.nan,  # -0.272 extension
        "target_2": np.nan,  # -0.618 extension
        "retrace_pct": 0.0,
        "dist_0618_pct": 999.0,
        "is_golden_pocket": False,
        "is_golden_pocket_confluence": False,
        "fib_status": "No Data",
        "fib_timeframe": timeframe_mode
    }

    if sub_intra is None or sub_intra.empty or len(sub_intra) < 5:
        return default_res

    df = sub_intra.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return default_res

    # Clean columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Resample based on Context Timeframe
    is_macro = "1-Hour" in timeframe_mode or "1h" in timeframe_mode.lower()
    rule = "1h" if is_macro else "15min"
    window_bars = 25 if is_macro else 12

    try:
        df_resampled = df.resample(rule, closed="left", label="left").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna(subset=["Close"])
    except Exception:
        return default_res

    if df_resampled.empty or len(df_resampled) < 3:
        return default_res

    # Extract fractal swing
    swing_high, swing_low, swing_type, _, _ = calculate_fractal_pivots(df_resampled, window_bars=window_bars)

    if np.isnan(swing_high) or np.isnan(swing_low) or swing_high <= swing_low:
        return default_res

    swing_range = swing_high - swing_low

    # Retracements & Inverse Targets Calculation
    if swing_type == "Bullish":
        # Upward swing: Retracement is measured downward from swing high
        fib_0236 = swing_high - (swing_range * 0.236)
        fib_0382 = swing_high - (swing_range * 0.382)
        fib_0500 = swing_high - (swing_range * 0.500)
        fib_0618 = swing_high - (swing_range * 0.618)  # Golden Pocket
        fib_0786 = swing_high - (swing_range * 0.786)

        # Automated Targets: Inverse extensions above Swing High
        target_1 = swing_high + (swing_range * 0.272)
        target_2 = swing_high + (swing_range * 0.618)

        retrace_pct = ((swing_high - ltp) / swing_range * 100.0) if swing_range > 0 else 0.0

    else:
        # Downward swing: Retracement is measured upward from swing low
        fib_0236 = swing_low + (swing_range * 0.236)
        fib_0382 = swing_low + (swing_range * 0.382)
        fib_0500 = swing_low + (swing_range * 0.500)
        fib_0618 = swing_low + (swing_range * 0.618)  # Golden Pocket
        fib_0786 = swing_low + (swing_range * 0.786)

        # Automated Targets: Inverse extensions below Swing Low
        target_1 = swing_low - (swing_range * 0.272)
        target_2 = swing_low - (swing_range * 0.618)

        retrace_pct = ((ltp - swing_low) / swing_range * 100.0) if swing_range > 0 else 0.0

    # Distance to Golden Pocket (0.618)
    dist_0618_pct = abs(ltp - fib_0618) / fib_0618 * 100.0 if fib_0618 > 0 else 999.0
    is_golden_pocket = bool(dist_0618_pct <= 0.50)  # within 0.5% tolerance

    # Golden Pocket Confluence (4.2): 1-Hour Break-in (Option C) occurring at Fib 0.618
    is_golden_pocket_confluence = bool(is_1h_break_in and is_golden_pocket)

    # Status Tag
    if is_golden_pocket_confluence:
        fib_status = "⭐ High Probability (Golden Pocket 0.618 Confluence)"
    elif is_golden_pocket:
        fib_status = "🟡 At Golden Pocket (0.618)"
    elif ltp > swing_high:
        fib_status = f"🚀 Above Swing High (T1: ₹{round(target_1, 2)})"
    elif ltp < swing_low:
        fib_status = f"🔻 Below Swing Low (T1: ₹{round(target_1, 2)})"
    elif retrace_pct <= 38.2:
        fib_status = "⚡ Shallow Pullback (< 38.2%)"
    elif retrace_pct <= 61.8:
        fib_status = "⚖️ Normal Retracement (38.2% - 61.8%)"
    else:
        fib_status = "⚠️ Deep Retracement (> 61.8%)"

    return {
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "swing_range": round(swing_range, 2),
        "swing_type": swing_type,
        "fib_0236": round(fib_0236, 2),
        "fib_0382": round(fib_0382, 2),
        "fib_0500": round(fib_0500, 2),
        "fib_0618": round(fib_0618, 2),
        "fib_0786": round(fib_0786, 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
        "retrace_pct": round(retrace_pct, 1),
        "dist_0618_pct": round(dist_0618_pct, 2),
        "is_golden_pocket": is_golden_pocket,
        "is_golden_pocket_confluence": is_golden_pocket_confluence,
        "fib_status": fib_status,
        "fib_timeframe": timeframe_mode
    }


def compute_dual_timeframe_fibonacci(
    sub_intra: pd.DataFrame,
    ltp: float,
    is_1h_break_in: bool = False
) -> Dict[str, Any]:
    """
    Computes both 1-Hour Macro and 15-Minute Micro Fibonacci contexts for a stock
    so they can be dynamically displayed and filtered without additional data fetches.
    """
    fib_macro = calculate_fibonacci_levels(
        sub_intra, ltp=ltp, is_1h_break_in=is_1h_break_in, timeframe_mode="1-Hour (Macro)"
    )
    fib_micro = calculate_fibonacci_levels(
        sub_intra, ltp=ltp, is_1h_break_in=is_1h_break_in, timeframe_mode="15-Minute (Micro)"
    )

    return {
        "macro": fib_macro,
        "micro": fib_micro
    }

