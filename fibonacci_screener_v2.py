"""
Module: Fibonacci_Screener_V2
Advanced Fibonacci & Target Enhancements
Implements:
Part I: Advanced Fibonacci & Target Enhancements
  1. Fibonacci Cluster (MTF Confluence) Filter (1H Macro & 15m Micro)
  2. Fibonacci Extension Breakouts (1.272 & 1.618 Blue Sky targets)
  3. Volume Profile + Golden Pocket Confluence (POC & Top 3 HVNs)
  4. Fibonacci Rejection (Wick) Scanner (15m Candlestick anatomy >= 60% lower wick)

Part II: High Beta F&O Confluence Filters
  5. Intraday: The "Golden Pivot" Confluence Filter (Fib + S1/S2/R1/R2 + VAL/POC/VAH)
  6. Intraday: High Beta Camarilla Reversal Trap (L3/H3 + Fib 0.618/1.272 + Reclaim)
  7. Intraday: Institutional Premium/Discount Oscillator (Distance from POC > 2*ATR + S3/R3 + Fib Ext)
  8. Swing: The Macro Anchor (Monthly Value MPOC + Monthly Pivots + Weekly Fib + Daily Candlestick)
  9. Swing: Weekly CPR Squeeze & Fib Expansion (Weekly CPR < 0.5% + Breakout + 1.272 & 1.618 Targets)
  10. Swing: Inside Week + Golden Pocket Compression (Inside Week + Fib 0.618 ±1% + Weekly POC in 30%)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


# =====================================================================
# 1. CORE SWING & RETRACEMENT PRIMITIVES
# =====================================================================

def calculate_fractal_pivots(df_resampled: pd.DataFrame, window_bars: int = 25) -> Tuple[float, float, str, int, int]:
    """
    Standard 5-bar Fractal Pivot Point detector over backward-looking window.
    Fallback: Rolling window extrema.
    """
    if df_resampled.empty or len(df_resampled) < 5:
        return np.nan, np.nan, "Neutral", 0, 0

    w_df = df_resampled.iloc[-window_bars:].copy()
    n = len(w_df)
    highs = w_df["High"].values
    lows = w_df["Low"].values

    fractal_highs = []
    fractal_lows = []

    for i in range(2, n - 2):
        if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
            highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
            fractal_highs.append((i, highs[i]))

        if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
            lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
            fractal_lows.append((i, lows[i]))

    if fractal_highs:
        best_high = max(fractal_highs, key=lambda x: x[1])
        swing_high, high_pos = float(best_high[1]), best_high[0]
    else:
        high_pos = int(np.argmax(highs))
        swing_high = float(highs[high_pos])

    if fractal_lows:
        best_low = min(fractal_lows, key=lambda x: x[1])
        swing_low, low_pos = float(best_low[1]), best_low[0]
    else:
        low_pos = int(np.argmin(lows))
        swing_low = float(lows[low_pos])

    swing_type = "Bullish" if low_pos < high_pos else ("Bearish" if high_pos < low_pos else "Neutral")
    return swing_high, swing_low, swing_type, high_pos, low_pos


def get_fibonacci_array(swing_high: float, swing_low: float, swing_type: str) -> Dict[str, float]:
    """
    Returns dictionary of retracement levels: 0.236, 0.382, 0.500, 0.618, 0.786.
    """
    if np.isnan(swing_high) or np.isnan(swing_low) or swing_high <= swing_low:
        return {}
    r = swing_high - swing_low
    if swing_type == "Bullish":
        return {
            "0.236": swing_high - (r * 0.236),
            "0.382": swing_high - (r * 0.382),
            "0.500": swing_high - (r * 0.500),
            "0.618": swing_high - (r * 0.618),
            "0.786": swing_high - (r * 0.786),
        }
    else:
        return {
            "0.236": swing_low + (r * 0.236),
            "0.382": swing_low + (r * 0.382),
            "0.500": swing_low + (r * 0.500),
            "0.618": swing_low + (r * 0.618),
            "0.786": swing_low + (r * 0.786),
        }


# =====================================================================
# PART I: ADVANCED FIBONACCI & TARGET ENHANCEMENTS (POINTS 1 - 4)
# =====================================================================

def calculate_fibonacci_clusters(
    fibs_1h: Dict[str, float],
    fibs_15m: Dict[str, float],
    tolerance_pct: float = 0.005
) -> Tuple[bool, float, str, List[Dict[str, Any]]]:
    """
    Point 1: Fibonacci Cluster (MTF Confluence) Filter
    Detects institutional support/resistance zones where macro (1-Hour)
    and micro (15-Minute) Fibonacci retracement levels align within tolerance (0.15% to 0.5%).
    Trigger: ABS(1H_Fib_Value - 15m_Fib_Value) / 1H_Fib_Value <= tolerance_pct.
    """
    if not fibs_1h or not fibs_15m:
        return False, np.nan, "", []

    clusters = []
    labels = []

    for k1, v1 in fibs_1h.items():
        if np.isnan(v1) or v1 <= 0:
            continue
        for k2, v2 in fibs_15m.items():
            if np.isnan(v2) or v2 <= 0:
                continue
            diff_pct = abs(v1 - v2) / v1
            if diff_pct <= tolerance_pct:
                avg_price = round((v1 + v2) / 2.0, 2)
                lbl = f"1H_{k1}_15m_{k2}"
                clusters.append({
                    "label": lbl,
                    "price": avg_price,
                    "diff_pct": round(diff_pct * 100.0, 3)
                })
                labels.append(lbl)

    is_active = len(clusters) > 0
    # Primary cluster level: average of all cluster prices or primary cluster
    primary_price = float(np.mean([c["price"] for c in clusters])) if clusters else np.nan
    cluster_labels_str = ", ".join(labels) if labels else "None"

    return is_active, round(primary_price, 2) if not np.isnan(primary_price) else np.nan, cluster_labels_str, clusters


def calculate_extension_breakouts(
    swing_high: float,
    swing_low: float,
    ltp: float,
    curr_vol: float,
    avg_vol: float
) -> Tuple[bool, float, float, float, str]:
    """
    Point 2: Fibonacci Extension Breakouts (1.272 & 1.618)
    Calculates forward-looking price targets for assets breaking out beyond 1.0 (Swing High)
    or down beyond 0.0 (Swing Low) tracking pure momentum (Blue Sky setups).
    Bullish Trigger: Close > Swing_High AND Current_Volume > Avg_Volume
    Bearish Trigger: Close < Swing_Low AND Current_Volume > Avg_Volume
    """
    if np.isnan(swing_high) or np.isnan(swing_low) or swing_high <= swing_low:
        return False, np.nan, np.nan, 0.0, "None"

    r = swing_high - swing_low
    # Bullish forward extensions
    ext_1_bull = swing_low + (r * 1.272)
    ext_2_bull = swing_low + (r * 1.618)

    # Bearish forward extensions
    ext_1_bear = swing_high - (r * 1.272)
    ext_2_bear = swing_high - (r * 1.618)

    vol_confirmed = curr_vol > avg_vol if avg_vol > 0 else True

    if ltp > swing_high and vol_confirmed:
        dist_pct = ((ext_1_bull - ltp) / ltp * 100.0) if ltp > 0 else 0.0
        return True, round(ext_1_bull, 2), round(ext_2_bull, 2), round(dist_pct, 2), "Bullish Blue Sky"
    elif ltp < swing_low and vol_confirmed:
        dist_pct = ((ltp - ext_1_bear) / ltp * 100.0) if ltp > 0 else 0.0
        return True, round(ext_1_bear, 2), round(ext_2_bear, 2), round(dist_pct, 2), "Bearish Extension"
    else:
        # Default projected levels
        dist_pct = ((ext_1_bull - ltp) / ltp * 100.0) if ltp > 0 else 0.0
        return False, round(ext_1_bull, 2), round(ext_2_bull, 2), round(dist_pct, 2), "None"


def calculate_volume_profile_and_hvns(df_candles: pd.DataFrame, num_bins: int = 70) -> Tuple[float, List[float]]:
    """
    Extracts Point of Control (POC) and Top 3 High Volume Nodes (HVNs).
    """
    if df_candles.empty or len(df_candles) < 2:
        return np.nan, []

    cols = {c.lower(): c for c in df_candles.columns}
    high_col = cols.get("high", "High")
    low_col = cols.get("low", "Low")
    vol_col = cols.get("volume", "Volume")

    day_high = float(df_candles[high_col].max())
    day_low = float(df_candles[low_col].min())
    total_volume = float(df_candles[vol_col].sum())

    if day_high <= day_low or total_volume <= 0:
        return np.nan, []

    bin_edges = np.linspace(day_low, day_high, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_volumes = np.zeros(num_bins, dtype=np.float64)

    highs = df_candles[high_col].values
    lows = df_candles[low_col].values
    vols = df_candles[vol_col].values

    for h, l, v in zip(highs, lows, vols):
        if v <= 0:
            continue
        if h <= l:
            idx = np.searchsorted(bin_edges, l, side="right") - 1
            idx = np.clip(idx, 0, num_bins - 1)
            bin_volumes[idx] += v
        else:
            overlap = np.maximum(0.0, np.minimum(h, bin_edges[1:]) - np.maximum(l, bin_edges[:-1]))
            bin_volumes += v * (overlap / (h - l))

    poc_idx = int(np.argmax(bin_volumes))
    poc_price = float(bin_centers[poc_idx])

    # Find local peaks for High Volume Nodes (HVNs)
    peaks = []
    for i in range(1, num_bins - 1):
        if bin_volumes[i] > bin_volumes[i-1] and bin_volumes[i] > bin_volumes[i+1]:
            if i != poc_idx:
                peaks.append((i, bin_volumes[i], bin_centers[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)
    top_hvns = [float(p[2]) for p in peaks[:3]]

    return poc_price, top_hvns


def calculate_vol_fib_confluence(
    fib_0618: float,
    poc_price: float,
    top_hvns: List[float],
    tolerance_pct: float = 0.005,
    require_poc_only: bool = False
) -> Tuple[bool, str, float]:
    """
    Point 3: Volume Profile + Golden Pocket Confluence
    Merges Volume Profile POC and top 3 HVNs with 0.618 Golden Pocket.
    Trigger: ABS(Fib_0.618 - POC) / POC <= tolerance OR ABS(Fib_0.618 - HVN) / HVN <= tolerance.
    """
    if np.isnan(fib_0618) or fib_0618 <= 0:
        return False, "None", np.nan

    # Check POC
    if not np.isnan(poc_price) and poc_price > 0:
        if abs(fib_0618 - poc_price) / poc_price <= tolerance_pct:
            return True, "POC", round(poc_price, 2)

    # Check HVNs
    if not require_poc_only and top_hvns:
        for hvn in top_hvns:
            if not np.isnan(hvn) and hvn > 0:
                if abs(fib_0618 - hvn) / hvn <= tolerance_pct:
                    return True, "HVN", round(hvn, 2)

    return False, "None", np.nan


def calculate_fib_rejection_wicks(
    df_15m: pd.DataFrame,
    fib_0500: float,
    fib_0618: float,
    wick_ratio_threshold: float = 0.60
) -> Tuple[bool, float, float, str]:
    """
    Point 4: Fibonacci Rejection (Wick) Scanner
    Analyzes 15-minute candlestick anatomy at 0.500 or 0.618 retracement levels.
    Trigger Conditions:
    1. Low intersects or drops below Zone_0.500 or Zone_0.618 (±0.2% tolerance)
    2. Close is strictly ABOVE the Fib level
    3. Lower_Wick >= Candle_Range * WICK_RATIO_THRESHOLD (0.60)
    Also supports Bearish Upper Wick rejection at Fib resistance.
    """
    if df_15m.empty or len(df_15m) < 1:
        return False, np.nan, 0.0, "None"

    # Analyze the most recent completed or live 15-minute candle
    candle = df_15m.iloc[-1]
    o = float(candle["Open"])
    h = float(candle["High"])
    l = float(candle["Low"])
    c = float(candle["Close"])

    candle_range = h - l
    if candle_range <= 0:
        return False, np.nan, 0.0, "None"

    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    target_levels = [("0.618", fib_0618), ("0.500", fib_0500)]

    # 1. Bullish Lower Wick Rejection
    for lbl, lvl in target_levels:
        if np.isnan(lvl) or lvl <= 0:
            continue
        zone_max = lvl * 1.002
        # Low reached zone, Close strictly above Fib level
        if l <= zone_max and c > lvl:
            wick_ratio = lower_wick / candle_range
            if wick_ratio >= wick_ratio_threshold:
                return True, round(lvl, 2), round(wick_ratio * 100.0, 1), f"Bullish Wick Rejection ({lbl})"

    # 2. Bearish Upper Wick Rejection
    for lbl, lvl in target_levels:
        if np.isnan(lvl) or lvl <= 0:
            continue
        zone_min = lvl * 0.998
        # High reached zone, Close strictly below Fib level
        if h >= zone_min and c < lvl:
            wick_ratio = upper_wick / candle_range
            if wick_ratio >= wick_ratio_threshold:
                return True, round(lvl, 2), round(wick_ratio * 100.0, 1), f"Bearish Wick Rejection ({lbl})"

    return False, np.nan, 0.0, "None"


# =====================================================================
# PART II: HIGH BETA F&O CONFLUENCE FILTERS (POINTS 5 - 10)
# =====================================================================

def calculate_golden_pivot_confluence(
    fib_0500: float,
    fib_0618: float,
    fib_0786: float,
    prev_h: float,
    prev_l: float,
    prev_c: float,
    pd_val: float,
    pd_vah: float,
    pd_poc: float,
    tolerance_pct: float = 0.005
) -> Tuple[bool, bool, float]:
    """
    Point 5: Intraday: The "Golden Pivot" Confluence Filter
    Bullish Setup: 1H Fib 0.618 or 0.500 within ±0.5% of Daily Support Pivot (S1 or S2)
                   AND within ±0.5% of Previous Day's VAL or POC.
    Bearish Setup: 1H Fib 0.618 or 0.786 within ±0.5% of Daily Resistance Pivot (R1 or R2)
                   AND within ±0.5% of Previous Day's VAH.
    """
    if any(np.isnan([prev_h, prev_l, prev_c])) or prev_h <= prev_l:
        return False, False, np.nan

    # Standard Floor Pivots
    p = (prev_h + prev_l + prev_c) / 3.0
    s1 = 2.0 * p - prev_h
    s2 = p - (prev_h - prev_l)
    r1 = 2.0 * p - prev_l
    r2 = p + (prev_h - prev_l)

    is_bull = False
    is_bear = False
    zone_prices = []

    # Bullish check
    for fib in [fib_0618, fib_0500]:
        if np.isnan(fib) or fib <= 0:
            continue
        # Support pivot match
        s_match = None
        for s in [s1, s2]:
            if abs(fib - s) / fib <= tolerance_pct:
                s_match = s
                break
        # Value Area match (VAL or POC)
        va_match = None
        for va in [pd_val, pd_poc]:
            if not np.isnan(va) and va > 0 and abs(fib - va) / fib <= tolerance_pct:
                va_match = va
                break

        if s_match is not None and va_match is not None:
            is_bull = True
            zone_prices = [fib, s_match, va_match]
            break

    # Bearish check
    if not is_bull:
        for fib in [fib_0618, fib_0786]:
            if np.isnan(fib) or fib <= 0:
                continue
            r_match = None
            for r in [r1, r2]:
                if abs(fib - r) / fib <= tolerance_pct:
                    r_match = r
                    break
            va_match = None
            if not np.isnan(pd_vah) and pd_vah > 0 and abs(fib - pd_vah) / fib <= tolerance_pct:
                va_match = pd_vah

            if r_match is not None and va_match is not None:
                is_bear = True
                zone_prices = [fib, r_match, va_match]
                break

    zone = float(np.median(zone_prices)) if zone_prices else np.nan
    return is_bull, is_bear, round(zone, 2) if not np.isnan(zone) else np.nan


def calculate_camarilla_reversal_trap(
    prev_h: float,
    prev_l: float,
    prev_c: float,
    fib_0618: float,
    ext_1272: float,
    df_15m: pd.DataFrame,
    tolerance_pct: float = 0.005
) -> Tuple[bool, bool, float, str]:
    """
    Point 6: Intraday: High Beta Camarilla Reversal Trap
    Bullish (L3 Trap): Daily Camarilla L3 overlaps with 1H Fib 0.618.
                       Trigger when a 15m candle Low < L3 but Close > L3.
    Bearish (H3 Trap): Daily Camarilla H3 overlaps with 1H Fib Extension 1.272.
                       Trigger when a 15m candle High > H3 but Close < H3.
    """
    if any(np.isnan([prev_h, prev_l, prev_c])) or prev_h <= prev_l or df_15m.empty:
        return False, False, np.nan, "None"

    rng = prev_h - prev_l
    h3 = prev_c + rng * 1.1 / 4.0
    l3 = prev_c - rng * 1.1 / 4.0

    last_candle = df_15m.iloc[-1]
    c_low = float(last_candle["Low"])
    c_high = float(last_candle["High"])
    c_close = float(last_candle["Close"])

    # Bullish L3 Trap
    if not np.isnan(fib_0618) and fib_0618 > 0:
        if abs(l3 - fib_0618) / l3 <= tolerance_pct:
            if c_low < l3 and c_close > l3:
                return True, False, round(l3, 2), "Bullish L3 Trap (Reclaimed Money Zone)"

    # Bearish H3 Trap
    if not np.isnan(ext_1272) and ext_1272 > 0:
        if abs(h3 - ext_1272) / h3 <= tolerance_pct:
            if c_high > h3 and c_close < h3:
                return False, True, round(h3, 2), "Bearish H3 Trap (Rejected Extension)"

    return False, False, np.nan, "None"


def calculate_prem_disc_oscillator(
    ltp: float,
    daily_poc: float,
    df_daily: pd.DataFrame,
    prev_h: float,
    prev_l: float,
    prev_c: float,
    fib_down_ext: float,
    fib_up_ext: float
) -> Tuple[bool, bool, str, float]:
    """
    Point 7: Intraday: Institutional Premium/Discount Oscillator
    Distance_from_POC = ABS(CMP - Daily_POC).
    Oversold (Discount): Distance > 2 * ATR_14 AND CMP is near Pivot S3 AND CMP near Fib -0.618.
    Overbought (Premium): Distance > 2 * ATR_14 AND CMP is near Pivot R3 AND CMP near Fib 1.618.
    """
    if np.isnan(daily_poc) or daily_poc <= 0 or df_daily.empty or len(df_daily) < 15:
        return False, False, "Neutral", 0.0

    # Calculate 14-period ATR
    highs = df_daily["High"].values
    lows = df_daily["Low"].values
    closes = df_daily["Close"].values
    n = len(df_daily)

    tr_list = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)

    atr_14 = float(np.mean(tr_list[-14:])) if len(tr_list) >= 14 else (prev_h - prev_l)
    if atr_14 <= 0:
        return False, False, "Neutral", 0.0

    dist_from_poc = abs(ltp - daily_poc)
    dist_atr = round(dist_from_poc / atr_14, 2)

    # Classic Floor S3 & R3
    p = (prev_h + prev_l + prev_c) / 3.0
    s3 = prev_l - 2.0 * (prev_h - p)
    r3 = prev_h + 2.0 * (p - prev_l)

    # Threshold checks
    is_oversold = False
    is_overbought = False

    if dist_from_poc > (2.0 * atr_14):
        # Oversold test: CMP near S3 and near Fib downside extension
        near_s3 = (abs(ltp - s3) / s3 <= 0.01) or (ltp <= s3 * 1.005)
        near_fib_down = (not np.isnan(fib_down_ext)) and ((abs(ltp - fib_down_ext) / fib_down_ext <= 0.01) or (ltp <= fib_down_ext * 1.005))
        if near_s3 and near_fib_down:
            is_oversold = True

        # Overbought test: CMP near R3 and near Fib upside extension
        near_r3 = (abs(ltp - r3) / r3 <= 0.01) or (ltp >= r3 * 0.995)
        near_fib_up = (not np.isnan(fib_up_ext)) and ((abs(ltp - fib_up_ext) / fib_up_ext <= 0.01) or (ltp >= fib_up_ext * 0.995))
        if near_r3 and near_fib_up:
            is_overbought = True

    status = "Oversold (Discount)" if is_oversold else ("Overbought (Premium)" if is_overbought else "Neutral")
    return is_oversold, is_overbought, status, dist_atr


def calculate_macro_anchor(
    df_daily: pd.DataFrame,
    ltp: float,
    macro_confluence_pct: float = 0.015
) -> Tuple[bool, bool, float, str]:
    """
    Point 8: Swing: The Macro Anchor (Monthly Value + Weekly Fib)
    Designed for 3-15 day holds.
    Bullish: Daily Close approaches Weekly Fib 0.618 / 0.786 AND aligns with MPOC
             AND Monthly Pivot S1 / S2 + Daily candlestick confirmation (Hammer / Bullish Engulfing).
    Bearish: Aligns with MPOC + Monthly Pivot R1 / R2 + Shooting Star / Bearish Engulfing.
    """
    if df_daily.empty or len(df_daily) < 40:
        return False, False, np.nan, "None"

    # Resample Monthly to get Previous Month metrics
    df_m = df_daily.resample("ME").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Close"])

    if len(df_m) < 2:
        return False, False, np.nan, "None"

    m_prev = df_m.iloc[-2]
    mh, ml, mc = float(m_prev["High"]), float(m_prev["Low"]), float(m_prev["Close"])
    mp = (mh + ml + mc) / 3.0
    ms1 = 2.0 * mp - mh
    ms2 = mp - (mh - ml)
    mr1 = 2.0 * mp - ml
    mr2 = mp + (mh - ml)
    mpoc = (mh + ml + mc) / 3.0  # Proxy for MPOC from monthly VWAP / typical price

    # Weekly Swing Fib
    df_w = df_daily.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Close"])

    if len(df_w) < 4:
        return False, False, np.nan, "None"

    w_lookback = df_w.iloc[-15:]
    w_high = float(w_lookback["High"].max())
    w_low = float(w_lookback["Low"].min())
    w_rng = w_high - w_low

    if w_rng <= 0:
        return False, False, np.nan, "None"

    w_fib_0618 = w_high - (w_rng * 0.618)
    w_fib_0786 = w_high - (w_rng * 0.786)

    # Check Daily Candlestick confirmation (latest bar vs previous bar)
    d_last = df_daily.iloc[-1]
    d_prev = df_daily.iloc[-2]

    d_o, d_h, d_l, d_c = float(d_last["Open"]), float(d_last["High"]), float(d_last["Low"]), float(d_last["Close"])
    p_o, p_c = float(d_prev["Open"]), float(d_prev["Close"])

    body = abs(d_c - d_o)
    lower_shadow = min(d_o, d_c) - d_l
    upper_shadow = d_h - max(d_o, d_c)

    is_hammer = (lower_shadow >= 1.8 * body) and (upper_shadow <= 0.6 * body) and (d_c >= d_o)
    is_bull_engulf = (p_c < p_o) and (d_c > d_o) and (d_c >= p_o) and (d_o <= p_c)
    bull_candlestick = is_hammer or is_bull_engulf

    is_shooting_star = (upper_shadow >= 1.8 * body) and (lower_shadow <= 0.6 * body) and (d_c <= d_o)
    is_bear_engulf = (p_c > p_o) and (d_c < d_o) and (d_c <= p_o) and (d_o >= p_c)
    bear_candlestick = is_shooting_star or is_bear_engulf

    # Bullish Macro Anchor
    is_bull = False
    is_bear = False
    anchor_zone = np.nan

    for w_fib in [w_fib_0618, w_fib_0786]:
        if abs(ltp - w_fib) / w_fib <= macro_confluence_pct:
            if abs(ltp - mpoc) / mpoc <= macro_confluence_pct:
                for ms in [ms1, ms2]:
                    if abs(ltp - ms) / ms <= macro_confluence_pct:
                        if bull_candlestick:
                            is_bull = True
                            anchor_zone = float(np.median([w_fib, mpoc, ms]))
                            break
        if is_bull:
            break

    # Bearish Macro Anchor
    if not is_bull:
        for w_fib in [w_fib_0618, w_fib_0786]:
            if abs(ltp - w_fib) / w_fib <= macro_confluence_pct:
                if abs(ltp - mpoc) / mpoc <= macro_confluence_pct:
                    for mr in [mr1, mr2]:
                        if abs(ltp - mr) / mr <= macro_confluence_pct:
                            if bear_candlestick:
                                is_bear = True
                                anchor_zone = float(np.median([w_fib, mpoc, mr]))
                                break
            if is_bear:
                break

    desc = "Bullish Macro Anchor" if is_bull else ("Bearish Macro Anchor" if is_bear else "None")
    return is_bull, is_bear, round(anchor_zone, 2) if not np.isnan(anchor_zone) else np.nan, desc


def calculate_weekly_cpr_fib_expansion(
    df_daily: pd.DataFrame,
    ltp: float
) -> Tuple[bool, bool, bool, float, float]:
    """
    Point 9: Swing: Weekly CPR Squeeze & Fib Expansion
    Compression Check: (Weekly_TC - Weekly_BC) / Weekly_BC < 0.005 (0.5% width).
    Bullish Expansion: Daily close > Previous Week's VAH AND > Weekly CPR.
                       Project +1.272 and +1.618 extensions.
    Bearish Expansion: Daily close < Previous Week's VAL AND < Weekly CPR.
                       Project -0.272 and -0.618 extensions.
    """
    if df_daily.empty or len(df_daily) < 15:
        return False, False, False, np.nan, np.nan

    df_w = df_daily.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Close"])

    if len(df_w) < 2:
        return False, False, False, np.nan, np.nan

    prev_w = df_w.iloc[-2]
    pw_h, pw_l, pw_c = float(prev_w["High"]), float(prev_w["Low"]), float(prev_w["Close"])
    pw_rng = pw_h - pw_l

    if pw_rng <= 0:
        return False, False, False, np.nan, np.nan

    w_pivot = (pw_h + pw_l + pw_c) / 3.0
    w_bc = (pw_h + pw_l) / 2.0
    w_tc = 2.0 * w_pivot - w_bc

    cpr_top = max(w_tc, w_bc)
    cpr_bot = min(w_tc, w_bc)
    cpr_width_pct = abs(w_tc - w_bc) / cpr_bot if cpr_bot > 0 else 1.0

    is_squeeze = cpr_width_pct < 0.005  # < 0.5% width

    # Weekly Value Area proxy from standard Camarilla / range
    pw_vah = pw_c + pw_rng * 1.1 / 4.0  # H3
    pw_val = pw_c - pw_rng * 1.1 / 4.0  # L3

    # Automated targets
    t1_bull = pw_l + (pw_rng * 1.272)
    t2_bull = pw_l + (pw_rng * 1.618)
    t1_bear = pw_h - (pw_rng * 1.272)
    t2_bear = pw_h - (pw_rng * 1.618)

    is_exp_bull = is_squeeze and (ltp > pw_vah) and (ltp > cpr_top)
    is_exp_bear = is_squeeze and (ltp < pw_val) and (ltp < cpr_bot)

    if is_exp_bull:
        return True, True, False, round(t1_bull, 2), round(t2_bull, 2)
    elif is_exp_bear:
        return True, False, True, round(t1_bear, 2), round(t2_bear, 2)
    else:
        return is_squeeze, False, False, round(t1_bull, 2), round(t2_bull, 2)


def calculate_inside_week_gp_compression(
    df_daily: pd.DataFrame,
    macro_fib_0618: float
) -> Tuple[bool, bool, str]:
    """
    Point 10: Swing: Inside Week + Golden Pocket Compression
    Condition 1: Inside Week (Current Week High < Prev Week High AND Current Week Low > Prev Week Low)
    Condition 2: Inside Week Close within ±1% of Macro 0.618 Fib level
    Condition 3: Weekly POC in lower 30% of range (Bullish) or upper 30% (Bearish).
    """
    if df_daily.empty or len(df_daily) < 15 or np.isnan(macro_fib_0618) or macro_fib_0618 <= 0:
        return False, False, "None"

    df_w = df_daily.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Close"])

    if len(df_w) < 2:
        return False, False, "None"

    curr_w = df_w.iloc[-1]
    prev_w = df_w.iloc[-2]

    cw_h, cw_l, cw_c = float(curr_w["High"]), float(curr_w["Low"]), float(curr_w["Close"])
    pw_h, pw_l = float(prev_w["High"]), float(prev_w["Low"])

    # Condition 1: Inside Week
    is_inside_week = (cw_h < pw_h) and (cw_l > pw_l)
    if not is_inside_week:
        return False, False, "None"

    # Condition 2: Close within ±1% of Macro 0.618
    diff_pct = abs(cw_c - macro_fib_0618) / macro_fib_0618
    if diff_pct > 0.01:
        return False, False, "None"

    # Condition 3: Weekly POC accumulation / distribution
    cw_rng = cw_h - cw_l
    if cw_rng <= 0:
        return False, False, "None"

    # Weekly POC estimated from typical price / intraday volume
    weekly_poc = (cw_h + cw_l + cw_c) / 3.0

    is_bull_accum = weekly_poc <= (cw_l + 0.30 * cw_rng)
    is_bear_dist = weekly_poc >= (cw_h - 0.30 * cw_rng)

    if is_bull_accum:
        return True, False, "Bullish Inside Week (Accumulation @ Fib 0.618)"
    elif is_bear_dist:
        return False, True, "Bearish Inside Week (Distribution @ Fib 0.618)"
    else:
        return False, False, "Inside Week @ Fib 0.618 (Neutral Volume)"


# =====================================================================
# MASTER WRAPPER: CALCULATE ALL FIBONACCI V2 ENHANCEMENTS
# =====================================================================

def calculate_fibonacci_v2_enhancements(
    sub_intra: pd.DataFrame,
    sub_1y: pd.DataFrame,
    prev_candles: pd.DataFrame,
    curr_candles: pd.DataFrame,
    ltp: float,
    vp: Dict[str, float],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Master pipeline evaluating all 10 Fibonacci & High Beta F&O Confluence enhancements.
    Returns dictionary with all metrics and boolean flags.
    """
    cfg = config or {}
    cluster_tol = cfg.get("cluster_tolerance_pct", 0.005)
    vol_tol = cfg.get("vol_confluence_tolerance", 0.005)
    req_poc_only = cfg.get("require_poc_only", False)
    wick_thresh = cfg.get("wick_ratio_threshold", 0.60)
    macro_tol = cfg.get("macro_confluence_pct", 0.015)

    default_res = {
        # Point 1: MTF Clusters
        "is_cluster_active": False,
        "cluster_level_price": np.nan,
        "cluster_labels": "None",
        # Point 2: Extension Breakouts
        "is_ext_breakout": False,
        "ext_target_1": np.nan,
        "ext_target_2": np.nan,
        "dist_to_ext_1_pct": 0.0,
        "ext_setup_tag": "None",
        # Point 3: Volume Profile + GP
        "vol_fib_confluence_flag": False,
        "confluence_node_type": "None",
        "confluence_node_price": np.nan,
        # Point 4: Wick Rejection
        "is_fib_rejection": False,
        "rejection_level": np.nan,
        "rejection_wick_pct": 0.0,
        "rejection_tag": "None",
        # Point 5: Golden Pivot
        "is_golden_pivot_bullish": False,
        "is_golden_pivot_bearish": False,
        "golden_pivot_zone": np.nan,
        # Point 6: Camarilla Trap
        "is_camarilla_trap_bull": False,
        "is_camarilla_trap_bear": False,
        "camarilla_trap_level": np.nan,
        "camarilla_trap_tag": "None",
        # Point 7: Prem/Disc Oscillator
        "is_prem_disc_oversold": False,
        "is_prem_disc_overbought": False,
        "prem_disc_status": "Neutral",
        "dist_from_poc_atr": 0.0,
        # Point 8: Macro Anchor
        "is_macro_anchor_bull": False,
        "is_macro_anchor_bear": False,
        "macro_anchor_zone": np.nan,
        "macro_anchor_tag": "None",
        # Point 9: Weekly CPR Squeeze & Exp
        "weekly_cpr_squeeze_active": False,
        "weekly_fib_exp_bull": False,
        "weekly_fib_exp_bear": False,
        "weekly_fib_target_1": np.nan,
        "weekly_fib_target_2": np.nan,
        # Point 10: Inside Week + GP
        "is_inside_week_gp_bull": False,
        "is_inside_week_gp_bear": False,
        "inside_week_gp_status": "None",
        # Core Baseline Swings
        "swing_high_1h": np.nan,
        "swing_low_1h": np.nan,
        "swing_type_1h": "Neutral",
        "fib_0618_1h": np.nan,
        "fib_0500_1h": np.nan,
    }

    if sub_intra is None or sub_intra.empty or len(sub_intra) < 5:
        return default_res

    # Clean intraday dataframe
    df_in = sub_intra.copy()
    if isinstance(df_in.columns, pd.MultiIndex):
        df_in.columns = df_in.columns.get_level_values(0)

    # Resample 1H and 15m
    try:
        df_1h = df_in.resample("1h", closed="left", label="left").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna(subset=["Close"])

        df_15m = df_in.resample("15min", closed="left", label="left").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna(subset=["Close"])
    except Exception:
        return default_res

    if df_1h.empty or df_15m.empty:
        return default_res

    # 1H Macro Swings & 15m Micro Swings
    sh_1h, sl_1h, st_1h, _, _ = calculate_fractal_pivots(df_1h, window_bars=25)
    sh_15m, sl_15m, st_15m, _, _ = calculate_fractal_pivots(df_15m, window_bars=12)

    fibs_1h = get_fibonacci_array(sh_1h, sl_1h, st_1h)
    fibs_15m = get_fibonacci_array(sh_15m, sl_15m, st_15m)

    fib_0618 = fibs_1h.get("0.618", np.nan)
    fib_0500 = fibs_1h.get("0.500", np.nan)
    fib_0786 = fibs_1h.get("0.786", np.nan)

    default_res["swing_high_1h"] = round(sh_1h, 2) if not np.isnan(sh_1h) else np.nan
    default_res["swing_low_1h"] = round(sl_1h, 2) if not np.isnan(sl_1h) else np.nan
    default_res["swing_type_1h"] = st_1h
    default_res["fib_0618_1h"] = round(fib_0618, 2) if not np.isnan(fib_0618) else np.nan
    default_res["fib_0500_1h"] = round(fib_0500, 2) if not np.isnan(fib_0500) else np.nan

    # Point 1: MTF Clusters
    is_cl, cl_price, cl_lbls, _ = calculate_fibonacci_clusters(fibs_1h, fibs_15m, tolerance_pct=cluster_tol)
    default_res["is_cluster_active"] = is_cl
    default_res["cluster_level_price"] = cl_price
    default_res["cluster_labels"] = cl_lbls

    # Point 2: Extension Breakouts
    curr_v = float(curr_candles["Volume"].iloc[-1]) if not curr_candles.empty else 0.0
    avg_v = float(df_1h["Volume"].iloc[-20:].mean()) if len(df_1h) >= 5 else 0.0
    is_ext, ext_1, ext_2, dist_ext, ext_tag = calculate_extension_breakouts(sh_1h, sl_1h, ltp, curr_v, avg_v)
    default_res["is_ext_breakout"] = is_ext
    default_res["ext_target_1"] = ext_1
    default_res["ext_target_2"] = ext_2
    default_res["dist_to_ext_1_pct"] = dist_ext
    default_res["ext_setup_tag"] = ext_tag

    # Point 3: Volume Profile + Golden Pocket Confluence
    poc_val, hvns = calculate_volume_profile_and_hvns(prev_candles)
    is_vol_fib, node_type, node_price = calculate_vol_fib_confluence(
        fib_0618, poc_val, hvns, tolerance_pct=vol_tol, require_poc_only=req_poc_only
    )
    default_res["vol_fib_confluence_flag"] = is_vol_fib
    default_res["confluence_node_type"] = node_type
    default_res["confluence_node_price"] = node_price

    # Point 4: Fibonacci Rejection (Wick) Scanner
    is_wick, rej_lvl, rej_pct, rej_tag = calculate_fib_rejection_wicks(
        df_15m, fib_0500, fib_0618, wick_ratio_threshold=wick_thresh
    )
    default_res["is_fib_rejection"] = is_wick
    default_res["rejection_level"] = rej_lvl
    default_res["rejection_wick_pct"] = rej_pct
    default_res["rejection_tag"] = rej_tag

    # Previous day OHLC
    prev_h = float(prev_candles["High"].max()) if not prev_candles.empty else np.nan
    prev_l = float(prev_candles["Low"].min()) if not prev_candles.empty else np.nan
    prev_c = float(prev_candles["Close"].iloc[-1]) if not prev_candles.empty else np.nan

    pd_val = vp.get("PD_VAL", np.nan)
    pd_vah = vp.get("PD_VAH", np.nan)
    pd_poc = vp.get("PD_POC", poc_val)

    # Point 5: Golden Pivot Confluence
    is_gp_bull, is_gp_bear, gp_zone = calculate_golden_pivot_confluence(
        fib_0500, fib_0618, fib_0786, prev_h, prev_l, prev_c, pd_val, pd_vah, pd_poc, tolerance_pct=cluster_tol
    )
    default_res["is_golden_pivot_bullish"] = is_gp_bull
    default_res["is_golden_pivot_bearish"] = is_gp_bear
    default_res["golden_pivot_zone"] = gp_zone

    # Point 6: Camarilla Reversal Trap
    is_cam_bull, is_cam_bear, cam_trap_lvl, cam_tag = calculate_camarilla_reversal_trap(
        prev_h, prev_l, prev_c, fib_0618, ext_1, df_15m, tolerance_pct=cluster_tol
    )
    default_res["is_camarilla_trap_bull"] = is_cam_bull
    default_res["is_camarilla_trap_bear"] = is_cam_bear
    default_res["camarilla_trap_level"] = cam_trap_lvl
    default_res["camarilla_trap_tag"] = cam_tag

    # Clean Daily DataFrame for Points 7, 8, 9, 10
    df_d = sub_1y.copy() if sub_1y is not None else pd.DataFrame()
    if isinstance(df_d.columns, pd.MultiIndex):
        df_d.columns = df_d.columns.get_level_values(0)

    # Point 7: Institutional Premium/Discount Oscillator
    fib_down_ext = sl_1h - (sh_1h - sl_1h) * 0.618 if (sh_1h > sl_1h) else np.nan
    fib_up_ext = ext_2
    is_ovs, is_ovb, prem_status, dist_atr = calculate_prem_disc_oscillator(
        ltp, pd_poc, df_d, prev_h, prev_l, prev_c, fib_down_ext, fib_up_ext
    )
    default_res["is_prem_disc_oversold"] = is_ovs
    default_res["is_prem_disc_overbought"] = is_ovb
    default_res["prem_disc_status"] = prem_status
    default_res["dist_from_poc_atr"] = dist_atr

    # Point 8: Macro Anchor
    is_ma_bull, is_ma_bear, ma_zone, ma_tag = calculate_macro_anchor(
        df_d, ltp, macro_confluence_pct=macro_tol
    )
    default_res["is_macro_anchor_bull"] = is_ma_bull
    default_res["is_macro_anchor_bear"] = is_ma_bear
    default_res["macro_anchor_zone"] = ma_zone
    default_res["macro_anchor_tag"] = ma_tag

    # Point 9: Weekly CPR Squeeze & Fib Expansion
    is_w_cpr, is_w_bull, is_w_bear, wt1, wt2 = calculate_weekly_cpr_fib_expansion(df_d, ltp)
    default_res["weekly_cpr_squeeze_active"] = is_w_cpr
    default_res["weekly_fib_exp_bull"] = is_w_bull
    default_res["weekly_fib_exp_bear"] = is_w_bear
    default_res["weekly_fib_target_1"] = wt1
    default_res["weekly_fib_target_2"] = wt2

    # Point 10: Inside Week + Golden Pocket Compression
    is_iw_bull, is_iw_bear, iw_status = calculate_inside_week_gp_compression(df_d, fib_0618)
    default_res["is_inside_week_gp_bull"] = is_iw_bull
    default_res["is_inside_week_gp_bear"] = is_iw_bear
    default_res["inside_week_gp_status"] = iw_status

    return default_res

