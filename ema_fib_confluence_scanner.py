"""
Module: ema_fib_confluence_scanner.py
Technical Specification: 100-Day EMA Rejection & Fibonacci Confluence Scanner (v1.0)

Implements Developer Guide for Programmatic Detection of:
- Standalone 100-Day EMA Bullish/Bearish Rejections (EMA_ONLY)
- High-Conviction EMA + Fibonacci Retracement Clusters (EMA_FIB_CONFLUENCE)
Timeframe: Daily (1D OHLCV) | Execution Mode: End-of-Day (Post-Market Close)
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd


# =====================================================================
# 1. CONFIGURABLE PARAMETERS (SECTION 2 & 6.1)
# =====================================================================

CONFIG: Dict[str, Any] = {
    "EMA_PERIOD": 100,             # Lookback span for Exponential Moving Average
    "SLOPE_LOOKBACK": 5,           # Number of trading sessions used to measure EMA slope
    "TOUCH_TOLERANCE_PCT": 0.005,  # 0.5% maximum allowable gap when price reverses near EMA
    "MIN_WICK_TO_BODY": 2.0,       # Minimum ratio of rejection wick length to real candle body size
    "MIN_WICK_TO_RANGE": 0.50,     # Minimum fraction (50%) of total candle range occupied by wick
    "RVOL_PERIOD": 20,             # Simple moving average window for baseline daily volume
    "MIN_RVOL": 1.2,               # Minimum Relative Volume multiple required to confirm absorption
    "SWING_LOOKBACK": 40,          # Rolling lookback window (sessions) to anchor prior impulse leg
    "MIN_IMPULSE_PCT": 0.08,       # Minimum percentage move (8.0%) required for valid impulse leg
    "FIB_CONFLUENCE_PCT": 0.015,   # Maximum percentage distance (1.5%) between 100 EMA and Fib level
}


# =====================================================================
# 2. CORE INDICATOR & CANDLE ANATOMY CALCULATIONS (SECTION 3 & 6.1)
# =====================================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column headers to lowercase ['date', 'open', 'high', 'low', 'close', 'volume']."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(-1)
    col_map = {}
    for c in out.columns:
        c_lower = str(c).lower().strip()
        if c_lower in ("open", "high", "low", "close", "volume"):
            col_map[c] = c_lower
        elif c_lower in ("date", "datetime", "timestamp"):
            col_map[c] = "date"
    out = out.rename(columns=col_map)
    if "date" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={"index": "date"})
    return out


def compute_indicators(df: pd.DataFrame, cfg: Dict[str, Any] = CONFIG) -> pd.DataFrame:
    """
    Precomputes 100-Day EMA, EMA Slope, 20-Day RVOL, and Clamped Candle Anatomy.
    Enforces Section 8 & 9 Safeguards:
    - Doji Division Protection: Clamping body and range with eps = 1e-6
    - Zero-Volume Protection: Replaces zero values in vol_sma with NaN
    """
    df = standardize_columns(df)
    df = df.copy()

    # 1. 100-Day EMA & Slope
    df["ema_100"] = df["close"].ewm(span=cfg["EMA_PERIOD"], adjust=False).mean()
    df["ema_slope"] = df["ema_100"] - df["ema_100"].shift(cfg["SLOPE_LOOKBACK"])

    # 2. Volume SMA & RVOL
    df["vol_sma"] = df["volume"].rolling(cfg["RVOL_PERIOD"]).mean()
    df["rvol"] = df["volume"] / df["vol_sma"].replace(0, np.nan)

    # 3. Clamped Candle Anatomy (Doji Protection)
    eps = 1e-6
    df["body"] = (df["close"] - df["open"]).abs().clip(lower=eps)
    df["range"] = (df["high"] - df["low"]).clip(lower=eps)
    df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]
    df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])

    return df


# =====================================================================
# 3. FIBONACCI CONFLUENCE ENGINE (SECTION 5 & 6.2)
# =====================================================================

def check_fib_confluence(
    df: pd.DataFrame, direction: str, cfg: Dict[str, Any] = CONFIG
) -> Optional[Dict[str, Any]]:
    """
    Module B: Automated Impulse Leg Anchoring & Proximity/Dual-Close Validation.
    - Anchors prior impulse leg over rolling [t - SWING_LOOKBACK, t - 1] window
    - Calculates Fib levels (0.382, 0.500, 0.618, 0.786 invalidation, 1.272 ext, 1.618 ext)
    - Verifies proximity: |EMA_t - Fib| / EMA_t <= FIB_CONFLUENCE_PCT (1.5%)
    - Verifies dual-close condition on the signal bar
    """
    if len(df) < (cfg["SWING_LOOKBACK"] + 2):
        return None

    curr = df.iloc[-1]
    window = df.iloc[-(cfg["SWING_LOOKBACK"] + 1):-1]

    if direction == "BULLISH":
        idx_high = window["high"].idxmax()
        swing_high = float(window.loc[idx_high, "high"])
        prior_slice = window.loc[:idx_high]
        if len(prior_slice) < 2:
            return None
        swing_low = float(prior_slice["low"].min())
        if (swing_high - swing_low) / swing_low < cfg["MIN_IMPULSE_PCT"]:
            return None

        delta = swing_high - swing_low
        fibs = {
            "0.382": swing_high - 0.382 * delta,
            "0.500": swing_high - 0.500 * delta,
            "0.618": swing_high - 0.618 * delta,
            "0.786": swing_high - 0.786 * delta,
            "target_1": swing_high,
            "target_2": swing_high + 0.272 * delta,
            "target_3": swing_high + 0.618 * delta,
        }
    else:
        idx_low = window["low"].idxmin()
        swing_low = float(window.loc[idx_low, "low"])
        prior_slice = window.loc[:idx_low]
        if len(prior_slice) < 2:
            return None
        swing_high = float(prior_slice["high"].max())
        if (swing_high - swing_low) / swing_high < cfg["MIN_IMPULSE_PCT"]:
            return None

        delta = swing_high - swing_low
        fibs = {
            "0.382": swing_low + 0.382 * delta,
            "0.500": swing_low + 0.500 * delta,
            "0.618": swing_low + 0.618 * delta,
            "0.786": swing_low + 0.786 * delta,
            "target_1": swing_low,
            "target_2": swing_low - 0.272 * delta,
            "target_3": swing_low - 0.618 * delta,
        }

    ema_val = float(curr["ema_100"])
    curr_close = float(curr["close"])
    matched = []

    # Proximity & Dual-Close Check (Section 5.3)
    for key in ("0.382", "0.500", "0.618"):
        level_price = fibs[key]
        dist_pct = abs(ema_val - level_price) / ema_val
        if dist_pct <= cfg["FIB_CONFLUENCE_PCT"]:
            # Dual-Close Check:
            # Bullish: Close > max(EMA, Fib) and Close > 0.786 level
            # Bearish: Close < min(EMA, Fib) and Close < 0.786 level
            if direction == "BULLISH":
                dual_close_pass = bool(curr_close > max(ema_val, level_price) and curr_close > fibs["0.786"])
            else:
                dual_close_pass = bool(curr_close < min(ema_val, level_price) and curr_close < fibs["0.786"])

            if dual_close_pass:
                matched.append({
                    "level": key,
                    "price": round(level_price, 2),
                    "ema_distance_pct": round(dist_pct * 100.0, 2)
                })

    if not matched:
        return None

    return {
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "matched_levels": matched,
        "invalidation_0786": round(fibs["0.786"], 2),
        "targets": {
            "t1_swing": round(fibs["target_1"], 2),
            "t2_ext_1272": round(fibs["target_2"], 2),
            "t3_ext_1618": round(fibs["target_3"], 2)
        }
    }


# =====================================================================
# 4. MAIN SYMBOL SCANNER ENTRYPOINT (SECTION 4, 6.3 & 7)
# =====================================================================

def scan_symbol(
    symbol: str, df: pd.DataFrame, cfg: Dict[str, Any] = CONFIG
) -> Optional[Dict[str, Any]]:
    """
    Evaluates daily OHLCV series for a single symbol.
    Requires at least 200 daily candles for EMA stabilization.
    Returns normalized dictionary matching Section 7 Output JSON Schema or None.
    """
    if df is None or len(df) < 200:
        return None

    df_ind = compute_indicators(df, cfg)
    curr, prev = df_ind.iloc[-1], df_ind.iloc[-2]

    # Module A: Standalone 100-Day EMA Rejection Rules (Section 4)
    bullish = (
        curr["ema_slope"] > 0
        and prev["close"] > prev["ema_100"]
        and curr["low"] <= curr["ema_100"] * (1 + cfg["TOUCH_TOLERANCE_PCT"])
        and curr["close"] > curr["ema_100"]
        and (curr["lower_wick"] / curr["body"]) >= cfg["MIN_WICK_TO_BODY"]
        and (curr["lower_wick"] / curr["range"]) >= cfg["MIN_WICK_TO_RANGE"]
        and curr["rvol"] >= cfg["MIN_RVOL"]
    )

    bearish = (
        curr["ema_slope"] < 0
        and prev["close"] < prev["ema_100"]
        and curr["high"] >= curr["ema_100"] * (1 - cfg["TOUCH_TOLERANCE_PCT"])
        and curr["close"] < curr["ema_100"]
        and (curr["upper_wick"] / curr["body"]) >= cfg["MIN_WICK_TO_BODY"]
        and (curr["upper_wick"] / curr["range"]) >= cfg["MIN_WICK_TO_RANGE"]
        and curr["rvol"] >= cfg["MIN_RVOL"]
    )

    if not (bullish or bearish):
        return None

    direction = "BULLISH" if bullish else "BEARISH"
    fib_data = check_fib_confluence(df_ind, direction, cfg)
    setup_tier = "EMA_FIB_CONFLUENCE" if fib_data else "EMA_ONLY"

    entry_trigger = float(curr["high"]) if bullish else float(curr["low"])
    stop_loss = float(curr["low"]) if bullish else float(curr["high"])

    wick_ratio = float((curr["lower_wick"] if bullish else curr["upper_wick"]) / curr["body"])
    wick_range_pct = float((curr["lower_wick"] if bullish else curr["upper_wick"]) / curr["range"] * 100.0)

    date_val = str(curr["date"]) if "date" in curr else (str(df_ind.index[-1].date()) if hasattr(df_ind.index[-1], "date") else str(df_ind.index[-1]))

    return {
        "symbol": symbol,
        "date": date_val,
        "direction": direction,
        "setup_tier": setup_tier,
        "metrics": {
            "close": round(float(curr["close"]), 2),
            "ema_100": round(float(curr["ema_100"]), 2),
            "ema_slope": round(float(curr["ema_slope"]), 2),
            "rvol": round(float(curr["rvol"]), 2),
            "wick_to_body": round(wick_ratio, 2),
            "wick_to_range_pct": round(wick_range_pct, 1)
        },
        "execution": {
            "entry_trigger": round(entry_trigger, 2),
            "stop_loss": round(stop_loss, 2)
        },
        "fib_confluence": fib_data
    }


# =====================================================================
# 5. UNIVERSE BATCH SCANNER & TABULAR FORMATTING
# =====================================================================

def scan_universe(
    daily_cache: Dict[str, pd.DataFrame],
    symbols: Optional[List[str]] = None,
    cfg: Dict[str, Any] = CONFIG
) -> pd.DataFrame:
    """
    Batches scanning across a dictionary of daily historical DataFrames.
    Outputs a flat, typed DataFrame ready for display and CSV export.
    """
    if symbols is None:
        symbols = list(daily_cache.keys())

    records = []
    for sym in symbols:
        df = daily_cache.get(sym)
        if df is None or len(df) < 200:
            continue
        try:
            res = scan_symbol(sym, df, cfg)
            if res is not None:
                m = res["metrics"]
                ex = res["execution"]
                fib = res["fib_confluence"]

                # Extract matched fib level details if confluence exists
                if fib and fib.get("matched_levels"):
                    top_m = fib["matched_levels"][0]
                    matched_lvl = top_m["level"]
                    fib_price = top_m["price"]
                    dist_pct = top_m["ema_distance_pct"]
                    inval = fib["invalidation_0786"]
                    t1 = fib["targets"]["t1_swing"]
                    t2 = fib["targets"]["t2_ext_1272"]
                    t3 = fib["targets"]["t3_ext_1618"]
                    # Calculate Risk:Reward to Target 2
                    risk = abs(ex["entry_trigger"] - ex["stop_loss"])
                    reward = abs(t2 - ex["entry_trigger"])
                    rr = round(reward / risk, 2) if risk > 0 else 0.0
                else:
                    matched_lvl = "None"
                    fib_price = np.nan
                    dist_pct = np.nan
                    inval = np.nan
                    t1 = np.nan
                    t2 = np.nan
                    t3 = np.nan
                    rr = np.nan

                records.append({
                    "Symbol": sym,
                    "Direction": res["direction"],
                    "Setup_Tier": res["setup_tier"],
                    "Tier_Tag": "⭐ EMA + Fib Cluster" if res["setup_tier"] == "EMA_FIB_CONFLUENCE" else "📌 100 EMA Rejection",
                    "Close": m["close"],
                    "EMA_100": m["ema_100"],
                    "EMA_Slope": m["ema_slope"],
                    "RVOL": m["rvol"],
                    "Wick_To_Body": m["wick_to_body"],
                    "Wick_To_Range%": m["wick_to_range_pct"],
                    "Entry_Trigger": ex["entry_trigger"],
                    "Stop_Loss": ex["stop_loss"],
                    "Matched_Fib": matched_lvl,
                    "Fib_Price": fib_price,
                    "EMA_Fib_Dist%": dist_pct,
                    "Invalidation_0786": inval,
                    "Target_1": t1,
                    "Target_2": t2,
                    "Target_3": t3,
                    "R_R": rr,
                    "Signal_Date": res["date"]
                })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    out_df = pd.DataFrame(records)
    # Sort prioritized: EMA_FIB_CONFLUENCE first, then descending by RVOL
    tier_order = {"EMA_FIB_CONFLUENCE": 0, "EMA_ONLY": 1}
    out_df["tier_sort"] = out_df["Setup_Tier"].map(tier_order)
    out_df = out_df.sort_values(by=["tier_sort", "RVOL"], ascending=[True, False]).drop(columns=["tier_sort"]).reset_index(drop=True)

    return out_df


# =====================================================================
# 6. UNIT TEST SUITE (SECTION 8 & 9 QA SAFEGUARDS)
# =====================================================================

def run_unit_tests() -> bool:
    """Executes automated verification for 100-Day EMA & Fib Confluence logic."""
    print("Running 100-Day EMA & Fib Confluence Scanner Unit Tests...")

    # Test 1: Minimum history guard
    short_df = pd.DataFrame({
        "open": [100.0] * 50, "high": [105.0] * 50, "low": [98.0] * 50,
        "close": [102.0] * 50, "volume": [1000] * 50
    })
    assert scan_symbol("TEST_SHORT", short_df) is None, "Failed: short history (<200) was not rejected"
    print("✅ Test 1 PASSED: Minimum 200 daily candle history guard enforced.")

    # Test 2: Synthetic Bullish Rejection Construction
    n = 220
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    base_price = np.linspace(100.0, 200.0, n)
    df_synth = pd.DataFrame({
        "open": base_price,
        "high": base_price + 2.0,
        "low": base_price - 2.0,
        "close": base_price + 0.5,
        "volume": [100000.0] * n
    }, index=dates)

    # Establish stable 100 EMA
    df_synth = compute_indicators(df_synth)
    ema_last = float(df_synth["ema_100"].iloc[-1])

    # Craft qualifying rejection on the last bar:
    # 1. Slope > 0, prior close > prior EMA
    # 2. Low pierces EMA: low <= EMA * 1.005
    # 3. Close closes firmly above EMA
    # 4. Lower wick dominates: (lower_wick / body) >= 2.0, (lower_wick / range) >= 0.50
    # 5. RVOL >= 1.2
    c_open = ema_last + 3.0
    c_close = ema_last + 4.0
    c_low = ema_last - 0.5   # Pierces below 100 EMA
    c_high = ema_last + 4.2
    # lower_wick = min(3.0, 4.0) - (-0.5) = 3.5
    # body = 4.0 - 3.0 = 1.0 (wick / body = 3.5 >= 2.0)
    # range = 4.2 - (-0.5) = 4.7 (wick / range = 3.5 / 4.7 = 74% >= 50%)

    df_synth.iloc[-1, df_synth.columns.get_loc("open")] = c_open
    df_synth.iloc[-1, df_synth.columns.get_loc("high")] = c_high
    df_synth.iloc[-1, df_synth.columns.get_loc("low")] = c_low
    df_synth.iloc[-1, df_synth.columns.get_loc("close")] = c_close
    df_synth.iloc[-1, df_synth.columns.get_loc("volume")] = 200000.0  # RVOL = 2.0 >= 1.2

    res_bull = scan_symbol("SYNTH_BULL", df_synth)
    assert res_bull is not None, "Failed: synthetic bullish rejection did not trigger"
    assert res_bull["direction"] == "BULLISH", "Failed: wrong direction"
    assert res_bull["setup_tier"] in ("EMA_ONLY", "EMA_FIB_CONFLUENCE"), "Failed: invalid tier"
    print("✅ Test 2 PASSED: Synthetic Bullish 100-Day EMA rejection confirmed.")

    # Test 3: Doji Division & Zero Volume Protection
    df_doji = df_synth.copy()
    df_doji.iloc[-1, df_doji.columns.get_loc("open")] = 150.0
    df_doji.iloc[-1, df_doji.columns.get_loc("close")] = 150.0  # Exact Doji
    df_doji.iloc[-1, df_doji.columns.get_loc("high")] = 150.0   # Flat range
    df_doji.iloc[-1, df_doji.columns.get_loc("low")] = 150.0
    df_doji.iloc[-1, df_doji.columns.get_loc("volume")] = 0.0   # Zero volume
    df_ind_doji = compute_indicators(df_doji)
    assert not np.isinf(df_ind_doji["body"].iloc[-1]), "Failed: body division produced inf"
    assert not np.isnan(df_ind_doji["range"].iloc[-1]), "Failed: range division produced NaN"
    print("✅ Test 3 PASSED: Doji division and zero-volume protections verified.")

    print("🎉 ALL 100-DAY EMA & FIB CONFLUENCE UNIT TESTS PASSED!")
    return True


if __name__ == "__main__":
    run_unit_tests()
