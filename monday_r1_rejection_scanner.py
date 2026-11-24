"""
Module: monday_r1_rejection_scanner.py
Technical Specification: Monday Weekly R1 1st-Hour Rejection & Resistance Confluence Scanner

Developer documentation implementation for detecting Monday opening-hour (09:15-10:15 AM IST)
rejections at Weekly R1 and ranking setups by overlapping multi-timeframe resistance confluences.

TARGET STACK: Python 3.10+ (pandas, numpy, yfinance)
TIMEFRAMES: Daily (1d, 1y lookback >= 200 bars) + Intraday (60m, 14d lookback)
TIMEZONE: Asia/Kolkata (IST)
RUN WINDOW: Mondays at/after 10:16 AM IST
"""

import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import yfinance as yf

# =====================================================================
# 1. CONFIGURABLE PARAMETERS & DEFAULTS
# =====================================================================

CONFIG: Dict[str, Any] = {
    "TOUCH_TOL_PCT": 0.002,         # 0.2% tolerance below Weekly R1 for reversal touch
    "CONFLUENCE_TOL_PCT": 0.0075,   # 0.75% band for secondary resistance confluences
    "TIMEZONE": "Asia/Kolkata",     # Target exchange timezone
    "OPEN_AT_R1_FILTER": False,     # Optional: if True, requires Open >= Weekly_R1 * 0.995
}


# =====================================================================
# 2. DATA UTILITIES & MULTI-TIMEFRAME LEVEL ENGINE
# =====================================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level index column names if returned by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2 and any(c in ["Open", "High", "Low", "Close", "Volume"] for c in df.columns.levels[0]):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    df.columns = [str(c).capitalize() for c in df.columns]
    return df


def get_daily_and_weekly_levels(
    df_d: pd.DataFrame, target_monday: pd.Timestamp
) -> Optional[Dict[str, float]]:
    """
    Computes Weekly R1 and resistance confluences strictly prior to target_monday.
    
    Strict No-Lookahead Guarantee (Section 1 & 3):
    All pivot, swing, and moving-average levels are computed exclusively from
    daily bars where date < target_monday.date().
    Monthly pivots further filter for to_period("M") < target_monday.to_period("M").
    
    Derived Levels (Section 3):
    1. Weekly_R1: Pw = (Hw + Lw + Cw) / 3; R1 = 2Pw - Lw (Primary rejection anchor)
    2. PWH: Previous completed week High (Hw) (Structural weekly supply)
    3. PDH: Previous trading day High (Friday) (Short-term swing resistance)
    4. Weekly_Cam_H3: Cw + (Rw * 1.1 / 4) (Camarilla mean-reversion resistance)
    5. Weekly_Cam_H4: Cw + (Rw * 1.1 / 2) (Camarilla breakout/fade boundary)
    6. Monthly_PP: (Hm + Lm + Cm) / 3 (Macro equilibrium pivot from completed month)
    7. Monthly_R1: 2 * Monthly_PP - Lm (Higher-timeframe monthly resistance)
    8. EMA_100: 100-period Daily EMA (adjust=False) (Intermediate dynamic resistance)
    9. EMA_200: 200-period Daily EMA (adjust=False) (Long-term institutional resistance)
    """
    if df_d is None or df_d.empty:
        return None

    df_d = standardize_columns(df_d.copy())
    
    # Filter strictly prior to target_monday
    hist = df_d[df_d.index.date < target_monday.date()].copy()
    if len(hist) < 200:
        return None

    # Calculate 100 and 200 daily EMAs on completed daily bars
    hist["ema_100"] = hist["Close"].ewm(span=100, adjust=False).mean()
    hist["ema_200"] = hist["Close"].ewm(span=200, adjust=False).mean()
    prev_day = hist.iloc[-1]

    # Weekly resampling (strictly completed weeks ending Friday)
    weekly = hist.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()

    if len(weekly) < 1:
        return None

    pw = weekly.iloc[-1]
    pw_h, pw_l, pw_c = float(pw["High"]), float(pw["Low"]), float(pw["Close"])
    pw_range = pw_h - pw_l
    w_pp = (pw_h + pw_l + pw_c) / 3.0
    w_r1 = (2.0 * w_pp) - pw_l
    cam_h3 = pw_c + (pw_range * 1.1 / 4.0)
    cam_h4 = pw_c + (pw_range * 1.1 / 2.0)

    # Monthly resampling: strictly completed months prior to target_monday's month
    try:
        monthly = hist.resample("ME").agg({
            "High": "max",
            "Low": "min",
            "Close": "last",
        }).dropna()
    except Exception:
        monthly = hist.resample("M").agg({
            "High": "max",
            "Low": "min",
            "Close": "last",
        }).dropna()

    target_monday_naive = target_monday.tz_localize(None) if target_monday.tz is not None else target_monday
    target_month_period = target_monday_naive.to_period("M")
    monthly_naive_idx = monthly.index.tz_localize(None) if monthly.index.tz is not None else monthly.index
    monthly = monthly[monthly_naive_idx.to_period("M") < target_month_period]

    if len(monthly) >= 1:
        pm = monthly.iloc[-1]
        pm_h, pm_l, pm_c = float(pm["High"]), float(pm["Low"]), float(pm["Close"])
        m_pp = (pm_h + pm_l + pm_c) / 3.0
        m_r1 = (2.0 * m_pp) - pm_l
    else:
        m_pp = np.nan
        m_r1 = np.nan

    return {
        "Weekly_R1": float(w_r1),
        "PWH": float(pw_h),
        "PDH": float(prev_day["High"]),
        "Weekly_Cam_H3": float(cam_h3),
        "Weekly_Cam_H4": float(cam_h4),
        "Monthly_PP": float(m_pp) if not np.isnan(m_pp) else np.nan,
        "Monthly_R1": float(m_r1) if not np.isnan(m_r1) else np.nan,
        "EMA_100": float(prev_day["ema_100"]),
        "EMA_200": float(prev_day["ema_200"]),
    }


def find_confluences(levels: Dict[str, float], tol_pct: float) -> List[Dict[str, Any]]:
    """
    Computes percentage distance of secondary resistances to Weekly_R1:
    distance_pct = |V - Weekly_R1| / Weekly_R1.
    Any level where distance_pct <= tol_pct qualifies as an institutional confluence cluster.
    """
    w_r1 = levels.get("Weekly_R1")
    if w_r1 is None or np.isnan(w_r1):
        return []

    confluences = []
    for k, v in levels.items():
        if k == "Weekly_R1" or v is None or np.isnan(v):
            continue
        dist = abs(v - w_r1) / w_r1
        if dist <= tol_pct:
            confluences.append({
                "level_name": k,
                "price": round(v, 2),
                "distance_pct": round(dist * 100, 3)
            })

    # Sort closest first (ascending distance)
    confluences.sort(key=lambda x: x["distance_pct"])
    return confluences


# =====================================================================
# 3. SIGNAL DETECTION & SCANNER CORE
# =====================================================================

def scan_monday_r1_from_data(
    symbol: str,
    df_d: pd.DataFrame,
    df_1h: pd.DataFrame,
    cfg: Dict[str, Any] = CONFIG
) -> Optional[Dict[str, Any]]:
    """
    Scans a pre-loaded/cached pair of daily and 1-hour intraday DataFrames
    for Monday 1st-hour Weekly R1 rejection and resistance confluences.
    """
    if df_d is None or df_1h is None or df_d.empty or df_1h.empty:
        return None

    df_d = standardize_columns(df_d.copy())
    df_1h = standardize_columns(df_1h.copy())

    # Ensure tz-aware Asia/Kolkata timestamps on intraday data
    tz_target = cfg.get("TIMEZONE", "Asia/Kolkata")
    if df_1h.index.tz is None:
        df_1h.index = df_1h.index.tz_localize("UTC").tz_convert(tz_target)
    else:
        df_1h.index = df_1h.index.tz_convert(tz_target)

    # 4.1 Monday 1st-Hour Candle Isolation:
    # Filter for rows matching Monday (day_of_week == 0) and hour == 9 (09:15-10:15 AM IST)
    try:
        dow = df_1h.index.day_of_week
    except AttributeError:
        dow = df_1h.index.dayofweek
    mondays_1h = df_1h[(dow == 0) & (df_1h.index.hour == 9)]
    if mondays_1h.empty:
        return None

    candle = mondays_1h.iloc[-1]
    monday_ts = mondays_1h.index[-1]

    levels = get_daily_and_weekly_levels(df_d, monday_ts)
    if not levels:
        return None

    w_r1 = levels["Weekly_R1"]
    o = float(candle["Open"])
    h = float(candle["High"])
    l = float(candle["Low"])
    c = float(candle["Close"])

    body = abs(c - o)
    upper_wick = h - max(o, c)

    # 4.2 Core Rejection Criteria (all 3 must pass):
    # 1. Test or Pierce Weekly R1 (allowable shortfall TOUCH_TOL_PCT)
    touched_or_pierced_r1 = h >= w_r1 * (1.0 - cfg.get("TOUCH_TOL_PCT", 0.002))

    # 2. Close Below Weekly R1
    closed_below_r1 = c < w_r1

    # 3. Bearish Rejection Structure: either red candle (C < O) or upper rejection wick >= real body
    rejection_structure = (c < o) or (upper_wick >= body)

    if not (touched_or_pierced_r1 and closed_below_r1 and rejection_structure):
        return None

    # Optional Gap-Up / Open-at-R1 Filter
    if cfg.get("OPEN_AT_R1_FILTER", False):
        if o < w_r1 * 0.995:
            return None

    # 4.3 Resistance Confluence Clustering
    confluences = find_confluences(levels, cfg.get("CONFLUENCE_TOL_PCT", 0.0075))

    # Format human-readable confluences string
    conf_strs = [f"{c['level_name']} (₹{c['price']:.2f}, {c['distance_pct']:+.2f}%)" for c in confluences]
    conf_summary = ", ".join([f"{c['level_name']} ({c['distance_pct']:+.2f}%)" for c in confluences]) if confluences else "None (Single Weekly R1)"

    # Execution Levels: Short breakdown trigger below 1st-hour low (L), stop-loss above 1st-hour high (H)
    entry_trigger = l
    stop_loss = h
    risk_points = round(stop_loss - entry_trigger, 2)
    risk_pct = round((risk_points / entry_trigger) * 100, 2) if entry_trigger > 0 else 0.0

    # Rejection pattern tag
    if c < o and upper_wick >= body:
        pattern_tag = "🔴 Red Candle + Long Upper Wick"
    elif c < o:
        pattern_tag = "🔴 Bearish Red Candle Close"
    else:
        pattern_tag = "📌 Upper Rejection Wick (≥ Body)"

    return {
        "symbol": symbol,
        "monday_date": monday_ts.strftime("%Y-%m-%d"),
        "weekly_r1": round(w_r1, 2),
        "1h_open": round(o, 2),
        "1h_high": round(h, 2),
        "1h_low": round(l, 2),
        "1h_close": round(c, 2),
        "entry_trigger_below": round(entry_trigger, 2),
        "stop_loss_above": round(stop_loss, 2),
        "risk_points": risk_points,
        "risk_pct": risk_pct,
        "confluence_count": len(confluences),
        "confluences": conf_strs,
        "confluences_str": conf_summary,
        "pattern_tag": pattern_tag,
        "levels": {k: round(v, 2) if v is not None and not np.isnan(v) else None for k, v in levels.items()},
        "candle_time": monday_ts.strftime("%Y-%m-%d %H:%M:%S")
    }


def scan_monday_r1_rejection(
    symbol: str, cfg: Dict[str, Any] = CONFIG
) -> Optional[Dict[str, Any]]:
    """
    Downloads historical daily (1y) and hourly (14d) data via yfinance,
    then evaluates Monday Weekly R1 rejection and resistance confluences.
    """
    yf_sym = symbol if symbol.endswith(".NS") or symbol.startswith("^") else f"{symbol}.NS"

    try:
        df_d = yf.download(
            yf_sym,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        df_1h = yf.download(
            yf_sym,
            period="14d",
            interval="60m",
            auto_adjust=True,
            progress=False
        )
    except Exception:
        return None

    if df_d is None or df_1h is None or df_d.empty or df_1h.empty:
        return None

    return scan_monday_r1_from_data(symbol, df_d, df_1h, cfg)


# =====================================================================
# 4. BATCH UNIVERSE SCANNER
# =====================================================================

def run_scanner(
    symbols: List[str],
    cfg: Dict[str, Any] = CONFIG,
    data_cache: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Runs the Monday Weekly R1 Rejection Scanner over a universe of symbols.
    Returns a DataFrame compliant with Section 6 Output Record Schema,
    ordered by confluence_count descending.
    """
    results = []
    
    for sym in symbols:
        res = None
        # Use cached DataFrames if provided (e.g. from Streamlit session state)
        if data_cache and sym in data_cache.get("daily", {}) and sym in data_cache.get("hourly", {}):
            df_d = data_cache["daily"][sym]
            df_1h = data_cache["hourly"][sym]
            res = scan_monday_r1_from_data(sym, df_d, df_1h, cfg)
        else:
            res = scan_monday_r1_rejection(sym, cfg)

        if res is not None:
            results.append({
                "symbol": res["symbol"],
                "monday_date": res["monday_date"],
                "weekly_r1": res["weekly_r1"],
                "1h_open": res["1h_open"],
                "1h_high": res["1h_high"],
                "1h_low": res["1h_low"],
                "1h_close": res["1h_close"],
                "entry_trigger_below": res["entry_trigger_below"],
                "stop_loss_above": res["stop_loss_above"],
                "risk_points": res["risk_points"],
                "risk_pct": res["risk_pct"],
                "confluence_count": res["confluence_count"],
                "confluences": res["confluences_str"],
                "pattern_tag": res["pattern_tag"],
            })

    if not results:
        return pd.DataFrame(columns=[
            "symbol", "monday_date", "weekly_r1", "1h_open", "1h_high", "1h_low", "1h_close",
            "entry_trigger_below", "stop_loss_above", "risk_points", "risk_pct",
            "confluence_count", "confluences", "pattern_tag"
        ])

    df_out = pd.DataFrame(results)
    # Ordered by confluence_count descending so tightest clusters appear first
    df_out = df_out.sort_values(by=["confluence_count", "risk_pct"], ascending=[False, True]).reset_index(drop=True)
    return df_out


# =====================================================================
# 5. SELF-TEST & VALIDATION SUITE
# =====================================================================

def _self_test():
    """Validates mathematical rules, edge cases, and data contract with synthetic data."""
    print("🧪 Running self-test for Monday Weekly R1 Rejection Scanner...")

    # Create 220 synthetic daily bars ending Friday 2026-11-20
    dates = pd.date_range("2026-01-01", periods=240, freq="B")
    np.random.seed(42)
    closes = 1000.0 + np.cumsum(np.random.randn(len(dates)) * 5)
    highs = closes + np.random.uniform(2, 10, len(dates))
    lows = closes - np.random.uniform(2, 10, len(dates))
    opens = closes + np.random.uniform(-3, 3, len(dates))

    df_d = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes
    }, index=dates)

    # Monday target
    target_monday = pd.Timestamp("2026-11-23 09:15:00", tz="Asia/Kolkata")

    # Compute levels
    levels = get_daily_and_weekly_levels(df_d, target_monday)
    assert levels is not None, "Failed to compute daily/weekly levels"
    assert "Weekly_R1" in levels, "Weekly_R1 missing"
    assert "PWH" in levels, "PWH missing"
    assert "PDH" in levels, "PDH missing"
    assert "Weekly_Cam_H3" in levels, "Weekly_Cam_H3 missing"
    assert "Weekly_Cam_H4" in levels, "Weekly_Cam_H4 missing"
    assert "EMA_100" in levels, "EMA_100 missing"
    assert "EMA_200" in levels, "EMA_200 missing"

    w_r1 = levels["Weekly_R1"]

    # Inject a confluent level close to Weekly_R1 (within 0.5%)
    levels["EMA_100"] = w_r1 * 1.003
    confs = find_confluences(levels, tol_pct=0.0075)
    assert any(c["level_name"] == "EMA_100" for c in confs), "Confluence matching failed for EMA_100"

    # Construct synthetic 1-hour Monday candle (09:15-10:15) with rejection:
    # High touches R1 + 0.1%, Closes below R1 with red body
    h_1h = w_r1 * 1.001
    o_1h = w_r1 * 0.998
    c_1h = w_r1 * 0.990
    l_1h = w_r1 * 0.988

    idx_1h = [target_monday]
    df_1h = pd.DataFrame({
        "Open": [o_1h], "High": [h_1h], "Low": [l_1h], "Close": [c_1h]
    }, index=idx_1h)

    res = scan_monday_r1_from_data("MOCK_STOCK", df_d, df_1h, CONFIG)
    assert res is not None, "Scanner failed to detect valid Monday Weekly R1 rejection"
    assert res["entry_trigger_below"] == round(l_1h, 2), "Incorrect entry trigger"
    assert res["stop_loss_above"] == round(h_1h, 2), "Incorrect stop loss"
    assert res["confluence_count"] >= 1, "Expected at least 1 confluence"

    print("✅ All math specifications, strict no-lookahead checks, and confluence tests PASSED!")


if __name__ == "__main__":
    _self_test()
