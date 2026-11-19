"""
1-Hour Head & Shoulders and Inverse Head & Shoulders Pattern Scanner
=====================================================================
Python reference implementation for automated institutional reversal detection on hourly OHLCV data.
Supports:
1. Bearish Head & Shoulders (Top Reversal)
2. Bullish Inverse Head & Shoulders (Bottom Reversal)
3. Dynamic Sloped Necklines (Linear Projection across bar indices)
4. Volume-Confirmed Breakdowns / Breakouts vs 20-period Volume SMA
5. Measured-Move Price Targets & Right-Shoulder Anchored Stop-Loss
"""

from typing import Dict, List, Optional, Any, Tuple
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from scipy.signal import argrelextrema
except ImportError:
    # Pure NumPy fallback matching scipy.signal.argrelextrema behavior
    def argrelextrema(arr: np.ndarray, comparator, order: int = 3):
        n = len(arr)
        extrema = []
        for i in range(order, n - order):
            val = arr[i]
            if comparator(val, arr[i - order : i]).all() and comparator(val, arr[i + 1 : i + order + 1]).all():
                extrema.append(i)
        return (np.array(extrema, dtype=int),)

from fo_stocks import get_yf_symbol, sanitize_symbol


def normalize_1h_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes yfinance 1h dataframe, flattens MultiIndex headers, and appends Vol_MA20."""
    if df.empty:
        return df

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    # Standardize column casing
    col_map = {c: c.capitalize() for c in out.columns}
    out.rename(columns=col_map, inplace=True)

    out = out.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if out.empty:
        return out

    if "Timestamp" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out["Timestamp"] = out.index
        else:
            out.reset_index(inplace=True)
            out.rename(columns={out.columns[0]: "Timestamp"}, inplace=True)

    if "Volume" in out.columns:
        out["Vol_MA20"] = out["Volume"].rolling(20, min_periods=5).mean()
    else:
        out["Volume"] = 0.0
        out["Vol_MA20"] = 0.0

    return out


def detect_bearish_head_and_shoulders(
    df: pd.DataFrame,
    order: int = 3,
    shoulder_tol: float = 0.04,
    neckline_tol: float = 0.04,
    min_head_prominence: float = 0.008,
    max_bars_since_rs: int = 35
) -> Optional[Dict[str, Any]]:
    """
    Detects the most recent valid Bearish Head & Shoulders (Top Reversal) pattern on 1H data.
    
    Rules:
    1. Head Dominance: Head high strictly exceeds Left Shoulder and Right Shoulder highs,
       and is the single highest price across the entire span [ls_idx : rs_idx].
    2. Head Prominence: Head must rise above higher shoulder by at least min_head_prominence (0.8%).
    3. Shoulder Symmetry: Relative price difference between shoulders <= shoulder_tol (4%).
    4. Neckline Troughs: T1 is lowest low between LS and Head; T2 is lowest low between Head and RS.
    5. Sloped Neckline: Dynamic linear projection y = T1 + slope * (x - T1_idx).
    6. Invalidation Guard: Any candle close after Right Shoulder exceeding RS high invalidates setup.
    """
    if len(df) < max(40, order * 6):
        return None

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    volumes = df["Volume"].to_numpy(dtype=float)
    vol_ma = df["Vol_MA20"].to_numpy(dtype=float) if "Vol_MA20" in df.columns else np.zeros_like(volumes)

    peak_indices = argrelextrema(highs, np.greater_equal, order=order)[0]
    if len(peak_indices) < 3:
        return None

    last_idx = len(df) - 1

    # Scan from the most recent triplet backwards
    for idx in range(len(peak_indices) - 3, -1, -1):
        ls_idx = peak_indices[idx]
        h_idx = peak_indices[idx + 1]
        rs_idx = peak_indices[idx + 2]

        if (last_idx - rs_idx) > max_bars_since_rs:
            break

        if (h_idx - ls_idx) < 2 or (rs_idx - h_idx) < 2:
            continue

        ls_price = highs[ls_idx]
        h_price = highs[h_idx]
        rs_price = highs[rs_idx]

        # 1. Head must be highest
        if h_price <= ls_price or h_price <= rs_price:
            continue
        if np.max(highs[ls_idx : rs_idx + 1]) > h_price:
            continue

        # 2. Head prominence
        max_shoulder = max(ls_price, rs_price)
        if (h_price - max_shoulder) / h_price < min_head_prominence:
            continue

        # 3. Shoulder symmetry
        if abs(ls_price - rs_price) / max_shoulder > shoulder_tol:
            continue

        # 4. Identify neckline troughs (deepest lows between LS-Head and Head-RS)
        t1_rel = np.argmin(lows[ls_idx + 1 : h_idx])
        t1_idx = ls_idx + 1 + int(t1_rel)

        t2_rel = np.argmin(lows[h_idx + 1 : rs_idx])
        t2_idx = h_idx + 1 + int(t2_rel)

        t1_price = lows[t1_idx]
        t2_price = lows[t2_idx]

        # 5. Neckline alignment check (slope limit)
        if abs(t2_price - t1_price) / max(t1_price, t2_price) > neckline_tol:
            continue

        if t2_idx <= t1_idx:
            continue

        slope = (t2_price - t1_price) / float(t2_idx - t1_idx)
        neckline_at = lambda x: t1_price + slope * (x - t1_idx)

        # 6. Invalidation guard: price closed above Right Shoulder high
        if np.any(closes[rs_idx + 1 :] > rs_price):
            continue

        current_close = closes[last_idx]
        current_neckline = neckline_at(last_idx)
        pattern_height = h_price - neckline_at(h_idx)
        if pattern_height <= 0:
            continue

        # Check for breakdown after Right Shoulder
        breakdown_idx = None
        for i in range(rs_idx + 1, last_idx + 1):
            if closes[i] < neckline_at(i):
                breakdown_idx = i
                break

        if breakdown_idx is not None:
            bd_neckline = neckline_at(breakdown_idx)
            target_price = bd_neckline - pattern_height
            vol_val = vol_ma[breakdown_idx] if breakdown_idx < len(vol_ma) else 0.0
            vol_confirmed = bool(not np.isnan(vol_val) and vol_val > 0 and volumes[breakdown_idx] > vol_val)
            status = "BREAKDOWN (Vol Confirmed)" if vol_confirmed else "BREAKDOWN (Low Vol)"
            trigger_ts = df["Timestamp"].iloc[breakdown_idx] if "Timestamp" in df.columns else str(breakdown_idx)
        else:
            target_price = current_neckline - pattern_height
            status = "FORMING (Watch Neckline)"
            trigger_ts = df["Timestamp"].iloc[rs_idx] if "Timestamp" in df.columns else str(rs_idx)

        stop_loss = rs_price
        risk = stop_loss - current_close
        reward = current_close - target_price
        rr_ratio = round(float(reward / risk), 2) if risk > 0 else 0.0
        drop_pct = round(float((current_close - target_price) / current_close * 100.0), 2) if current_close > 0 else 0.0

        ls_ts = df["Timestamp"].iloc[ls_idx] if "Timestamp" in df.columns else str(ls_idx)
        h_ts = df["Timestamp"].iloc[h_idx] if "Timestamp" in df.columns else str(h_idx)
        rs_ts = df["Timestamp"].iloc[rs_idx] if "Timestamp" in df.columns else str(rs_idx)

        return {
            "Pattern": "Head & Shoulders (Bearish)",
            "Direction": "Bearish",
            "Status": status,
            "Current_Price": round(float(current_close), 2),
            "Neckline_Now": round(float(current_neckline), 2),
            "Target": round(float(target_price), 2),
            "Stop_Loss_RS": round(float(stop_loss), 2),
            "Move_Pct": drop_pct,
            "R_R": rr_ratio,
            "Pattern_Height": round(float(pattern_height), 2),
            "LS_Price": round(float(ls_price), 2),
            "Head_Price": round(float(h_price), 2),
            "RS_Price": round(float(rs_price), 2),
            "Trough_1_Price": round(float(t1_price), 2),
            "Trough_2_Price": round(float(t2_price), 2),
            "LS_Time": ls_ts,
            "Head_Time": h_ts,
            "RS_Time": rs_ts,
            "Signal_Time": trigger_ts,
            "Bars_Since_RS": int(last_idx - rs_idx)
        }

    return None


def detect_bullish_inverse_head_and_shoulders(
    df: pd.DataFrame,
    order: int = 3,
    shoulder_tol: float = 0.04,
    neckline_tol: float = 0.04,
    min_head_prominence: float = 0.008,
    max_bars_since_rs: int = 35
) -> Optional[Dict[str, Any]]:
    """
    Detects the most recent valid Bullish Inverse Head & Shoulders (Bottom Reversal) pattern on 1H data.
    
    Rules (Mirrored from Section 6.1):
    1. Head Dominance: Head low strictly lower than Left Shoulder and Right Shoulder lows,
       and is the single lowest price across the entire span [ls_idx : rs_idx].
    2. Head Prominence: Head must drop below lower shoulder by at least min_head_prominence (0.8%).
    3. Shoulder Symmetry: Relative price difference between shoulders <= shoulder_tol (4%).
    4. Neckline Peaks: P1 is highest high between LS and Head; P2 is highest high between Head and RS.
    5. Sloped Neckline: Dynamic linear projection y = P1 + slope * (x - P1_idx).
    6. Invalidation Guard: Any candle close after Right Shoulder dropping below RS low invalidates setup.
    """
    if len(df) < max(40, order * 6):
        return None

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    volumes = df["Volume"].to_numpy(dtype=float)
    vol_ma = df["Vol_MA20"].to_numpy(dtype=float) if "Vol_MA20" in df.columns else np.zeros_like(volumes)

    trough_indices = argrelextrema(lows, np.less_equal, order=order)[0]
    if len(trough_indices) < 3:
        return None

    last_idx = len(df) - 1

    # Scan from the most recent triplet backwards
    for idx in range(len(trough_indices) - 3, -1, -1):
        ls_idx = trough_indices[idx]
        h_idx = trough_indices[idx + 1]
        rs_idx = trough_indices[idx + 2]

        if (last_idx - rs_idx) > max_bars_since_rs:
            break

        if (h_idx - ls_idx) < 2 or (rs_idx - h_idx) < 2:
            continue

        ls_price = lows[ls_idx]
        h_price = lows[h_idx]
        rs_price = lows[rs_idx]

        # 1. Head must be lowest
        if h_price >= ls_price or h_price >= rs_price:
            continue
        if np.min(lows[ls_idx : rs_idx + 1]) < h_price:
            continue

        # 2. Head prominence
        min_shoulder = min(ls_price, rs_price)
        if (min_shoulder - h_price) / h_price < min_head_prominence:
            continue

        # 3. Shoulder symmetry
        if abs(ls_price - rs_price) / min_shoulder > shoulder_tol:
            continue

        # 4. Identify neckline peaks (highest highs between LS-Head and Head-RS)
        p1_rel = np.argmax(highs[ls_idx + 1 : h_idx])
        p1_idx = ls_idx + 1 + int(p1_rel)

        p2_rel = np.argmax(highs[h_idx + 1 : rs_idx])
        p2_idx = h_idx + 1 + int(p2_rel)

        p1_price = highs[p1_idx]
        p2_price = highs[p2_idx]

        # 5. Neckline alignment check (slope limit)
        if abs(p2_price - p1_price) / min(p1_price, p2_price) > neckline_tol:
            continue

        if p2_idx <= p1_idx:
            continue

        slope = (p2_price - p1_price) / float(p2_idx - p1_idx)
        neckline_at = lambda x: p1_price + slope * (x - p1_idx)

        # 6. Invalidation guard: price closed below Right Shoulder low
        if np.any(closes[rs_idx + 1 :] < rs_price):
            continue

        current_close = closes[last_idx]
        current_neckline = neckline_at(last_idx)
        pattern_height = neckline_at(h_idx) - h_price
        if pattern_height <= 0:
            continue

        # Check for breakout above neckline after Right Shoulder
        breakout_idx = None
        for i in range(rs_idx + 1, last_idx + 1):
            if closes[i] > neckline_at(i):
                breakout_idx = i
                break

        if breakout_idx is not None:
            bd_neckline = neckline_at(breakout_idx)
            target_price = bd_neckline + pattern_height
            vol_val = vol_ma[breakout_idx] if breakout_idx < len(vol_ma) else 0.0
            vol_confirmed = bool(not np.isnan(vol_val) and vol_val > 0 and volumes[breakout_idx] > vol_val)
            status = "BREAKOUT (Vol Confirmed)" if vol_confirmed else "BREAKOUT (Low Vol)"
            trigger_ts = df["Timestamp"].iloc[breakout_idx] if "Timestamp" in df.columns else str(breakout_idx)
        else:
            target_price = current_neckline + pattern_height
            status = "FORMING (Watch Neckline)"
            trigger_ts = df["Timestamp"].iloc[rs_idx] if "Timestamp" in df.columns else str(rs_idx)

        stop_loss = rs_price
        risk = current_close - stop_loss
        reward = target_price - current_close
        rr_ratio = round(float(reward / risk), 2) if risk > 0 else 0.0
        gain_pct = round(float((target_price - current_close) / current_close * 100.0), 2) if current_close > 0 else 0.0

        ls_ts = df["Timestamp"].iloc[ls_idx] if "Timestamp" in df.columns else str(ls_idx)
        h_ts = df["Timestamp"].iloc[h_idx] if "Timestamp" in df.columns else str(h_idx)
        rs_ts = df["Timestamp"].iloc[rs_idx] if "Timestamp" in df.columns else str(rs_idx)

        return {
            "Pattern": "Inverse H&S (Bullish)",
            "Direction": "Bullish",
            "Status": status,
            "Current_Price": round(float(current_close), 2),
            "Neckline_Now": round(float(current_neckline), 2),
            "Target": round(float(target_price), 2),
            "Stop_Loss_RS": round(float(stop_loss), 2),
            "Move_Pct": gain_pct,
            "R_R": rr_ratio,
            "Pattern_Height": round(float(pattern_height), 2),
            "LS_Price": round(float(ls_price), 2),
            "Head_Price": round(float(h_price), 2),
            "RS_Price": round(float(rs_price), 2),
            "Peak_1_Price": round(float(p1_price), 2),
            "Peak_2_Price": round(float(p2_price), 2),
            "LS_Time": ls_ts,
            "Head_Time": h_ts,
            "RS_Time": rs_ts,
            "Signal_Time": trigger_ts,
            "Bars_Since_RS": int(last_idx - rs_idx)
        }

    return None


def fetch_hourly_data_chunked(symbols: List[str], period: str = "60d") -> Dict[str, pd.DataFrame]:
    """
    Downloads 60-day 1-Hour OHLCV data for symbols with parallel multi-threading and normalizes columns.
    """
    clean_syms = [s.strip().upper() for s in symbols if sanitize_symbol(s)]
    if not clean_syms:
        return {}

    yf_tickers = [get_yf_symbol(s) for s in clean_syms]
    sym_by_yf = {get_yf_symbol(s): s for s in clean_syms}

    try:
        data = yf.download(
            yf_tickers,
            period=period,
            interval="1h",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True
        )
    except Exception:
        return {}

    out_dict: Dict[str, pd.DataFrame] = {}
    if data.empty:
        return out_dict

    # Single-ticker response
    if len(yf_tickers) == 1:
        s = clean_syms[0]
        norm = normalize_1h_dataframe(data)
        if not norm.empty:
            out_dict[s] = norm
        return out_dict

    # Multi-ticker response
    for yf_t, sym in sym_by_yf.items():
        if yf_t in data.columns.levels[0]:
            sub = data[yf_t].dropna(subset=["Open", "High", "Low", "Close"]).copy()
            if len(sub) >= 40:
                norm = normalize_1h_dataframe(sub)
                if not norm.empty:
                    out_dict[sym] = norm

    return out_dict


def scan_classical_patterns_1h(
    symbols: List[str],
    hourly_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    order: int = 3,
    shoulder_tol: float = 0.04,
    neckline_tol: float = 0.04,
    min_head_prominence: float = 0.008,
    max_bars_since_rs: int = 35
) -> pd.DataFrame:
    """
    Scans a universe of F&O symbols for active 1H Bearish Head & Shoulders and Bullish Inverse Head & Shoulders setups.
    Returns a unified summary DataFrame.
    """
    if hourly_dfs is None:
        hourly_dfs = fetch_hourly_data_chunked(symbols, period="60d")

    records: List[Dict[str, Any]] = []

    for sym in symbols:
        df_1h = hourly_dfs.get(sym)
        if df_1h is None or df_1h.empty or len(df_1h) < 40:
            continue

        # 1. Bearish Head & Shoulders
        res_bear = detect_bearish_head_and_shoulders(
            df_1h,
            order=order,
            shoulder_tol=shoulder_tol,
            neckline_tol=neckline_tol,
            min_head_prominence=min_head_prominence,
            max_bars_since_rs=max_bars_since_rs
        )
        if res_bear:
            res_bear["Symbol"] = sym
            records.append(res_bear)

        # 2. Bullish Inverse Head & Shoulders
        res_bull = detect_bullish_inverse_head_and_shoulders(
            df_1h,
            order=order,
            shoulder_tol=shoulder_tol,
            neckline_tol=neckline_tol,
            min_head_prominence=min_head_prominence,
            max_bars_since_rs=max_bars_since_rs
        )
        if res_bull:
            res_bull["Symbol"] = sym
            records.append(res_bull)

    if not records:
        return pd.DataFrame()

    out_df = pd.DataFrame(records)
    # Order columns
    pref_cols = [
        "Symbol", "Pattern", "Direction", "Status", "Current_Price", "Neckline_Now",
        "Target", "Stop_Loss_RS", "Move_Pct", "R_R",
        "LS_Price", "Head_Price", "RS_Price", "Pattern_Height",
        "Bars_Since_RS", "Signal_Time", "LS_Time", "Head_Time", "RS_Time"
    ]
    avail_cols = [c for c in pref_cols if c in out_df.columns]
    return out_df[avail_cols]
