"""
Core Screener Engine: Data Pipeline & Vectorized Trigger Logic
Implements Setup A (Bullish Breakout) & Setup B (Bearish Breakdown)
Enhanced with:
1. Volume Profile (VAH / VAL 70% Value Area)
2. Trailing 52-Week Extremes
3. CPR (Central Pivot Range / Money Zone)
4. Camarilla Pivots (H4/L4 & H3/L3)
5. Bollinger Bands Squeeze & Volatility Breakout (20 SMA, 2.0 StdDev, 120 Lookback)
"""

import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from fo_stocks import get_yf_symbol
from volume_profile import calculate_volume_profile, get_profile_histogram
from bollinger_screener import calculate_bollinger_bands


def fetch_and_screen(
    symbols: List[str], 
    timeframe: str = "5m", 
    high_proximity_pct: float = 0.95,
    low_proximity_pct: float = 1.05,
    enable_cpr_filter: bool = True,
    only_narrow_cpr: bool = False,
    narrow_cpr_max_pct: float = 0.35,
    camarilla_filter: str = "None",
    bollinger_filter: str = "None",
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Downloads intraday and 52-week data in batch, computes Volume Profile,
    Central Pivot Range (CPR), Camarilla Pivots, and Bollinger Bands Squeeze.
    
    Screens for Bullish Breakout and Bearish Breakdown triggers with
    multi-confluence validation.
    
    Returns:
    - (all_df, bullish_df, bearish_df, meta_info)
    """
    if not symbols:
        empty = pd.DataFrame()
        return empty, empty, empty, {}

    yf_tickers = [get_yf_symbol(s) for s in symbols]
    ticker_to_sym = {get_yf_symbol(s): s for s in symbols}

    # 1. Batch download 1-year daily data for 52W High & 52W Low & Bollinger Bands
    try:
        df_1y = yf.download(
            yf_tickers,
            period="1y",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True
        )
    except Exception:
        df_1y = pd.DataFrame()

    # 2. Batch download 5-day intraday data for Volume Profile
    try:
        df_intraday = yf.download(
            yf_tickers,
            period="5d",
            interval=timeframe,
            group_by="ticker",
            progress=False,
            threads=True
        )
    except Exception:
        df_intraday = pd.DataFrame()

    records = []
    ref_date_str = "N/A"
    live_date_str = "N/A"

    is_single_ticker = len(yf_tickers) == 1

    for t in yf_tickers:
        sym = ticker_to_sym.get(t, t)
        
        # Extract intraday sub-dataframe
        try:
            if is_single_ticker:
                sub_intra = df_intraday.dropna(how="all")
                if isinstance(sub_intra.columns, pd.MultiIndex):
                    sub_intra.columns = sub_intra.columns.get_level_values(0)
            else:
                if t not in df_intraday.columns.levels[0]:
                    continue
                sub_intra = df_intraday[t].dropna(how="all")
        except Exception:
            continue

        if sub_intra.empty or len(sub_intra) < 5:
            continue

        # Extract dates available in intraday data
        dates = sorted(list(set(sub_intra.index.date)))
        if len(dates) < 2:
            continue

        # Check if today is the active market day (IST)
        try:
            from zoneinfo import ZoneInfo
            today_date = datetime.datetime.now(ZoneInfo("Asia/Kolkata")).date()
        except Exception:
            today_date = datetime.date.today()

        if dates[-1] == today_date:
            curr_date = dates[-1]
            prev_date = dates[-2]
        else:
            # Weekend / After-market: last available date is the live/evaluation date
            curr_date = dates[-1]
            prev_date = dates[-2]

        ref_date_str = str(prev_date)
        live_date_str = str(curr_date)

        prev_candles = sub_intra[sub_intra.index.date == prev_date]
        curr_candles = sub_intra[sub_intra.index.date == curr_date]

        if prev_candles.empty or curr_candles.empty:
            continue

        # Compute Volume Profile for Previous Day
        vp = calculate_volume_profile(prev_candles)
        if np.isnan(vp["PD_VAH"]) or np.isnan(vp["PD_VAL"]):
            continue

        # Extract 52-Week High/Low and compute Bollinger Bands from daily data
        h52 = np.nan
        l52 = np.nan
        sub_1y = pd.DataFrame()
        try:
            if is_single_ticker:
                sub_1y = df_1y.dropna(how="all")
                if isinstance(sub_1y.columns, pd.MultiIndex):
                    sub_1y.columns = sub_1y.columns.get_level_values(0)
            else:
                sub_1y = df_1y[t].dropna(how="all") if t in df_1y.columns.levels[0] else pd.DataFrame()

            if not sub_1y.empty:
                h52 = float(sub_1y["High"].max())
                l52 = float(sub_1y["Low"].min())
        except Exception:
            pass

        # Fallback for 52W High/Low if daily download had issues
        if np.isnan(h52) or h52 <= 0:
            h52 = vp["Day_High"] * 1.10
        if np.isnan(l52) or l52 <= 0:
            l52 = vp["Day_Low"] * 0.90

        # Bollinger Bands & Squeeze calculation
        bb_info = calculate_bollinger_bands(sub_1y)

        # Current Market Data
        ltp = float(curr_candles["Close"].iloc[-1])
        day_open = float(curr_candles["Open"].iloc[0])
        day_high = float(curr_candles["High"].max())
        day_low = float(curr_candles["Low"].min())
        day_vol = float(curr_candles["Volume"].sum())
        prev_close = vp["Day_Close"]
        prev_h = vp["Day_High"]
        prev_l = vp["Day_Low"]

        chg_pct = ((ltp - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0

        # ----------------- CPR (Central Pivot Range / Money Zone) -----------------
        pd_pivot = (prev_h + prev_l + prev_close) / 3.0
        pd_bc = (prev_h + prev_l) / 2.0
        pd_tc = (2.0 * pd_pivot) - pd_bc
        pd_cpr_top = max(pd_tc, pd_bc)
        pd_cpr_bottom = min(pd_tc, pd_bc)

        # CPR Width & Narrow CPR Detection
        pd_cpr_width = abs(pd_tc - pd_bc)
        pd_cpr_width_pct = (pd_cpr_width / pd_pivot * 100.0) if pd_pivot > 0 else 0.0

        if pd_cpr_width_pct <= 0.15:
            cpr_type = "🎯 Ultra Narrow"
        elif pd_cpr_width_pct <= 0.35:
            cpr_type = "⚡ Narrow"
        elif pd_cpr_width_pct <= 0.70:
            cpr_type = "Average"
        else:
            cpr_type = "Wide"

        is_narrow_cpr = pd_cpr_width_pct <= narrow_cpr_max_pct

        if ltp > pd_cpr_top:
            cpr_status = "🟢 Above CPR"
        elif ltp < pd_cpr_bottom:
            cpr_status = "🔴 Below CPR"
        else:
            cpr_status = "🟡 Inside CPR"

        # ----------------- Camarilla Pivots (H4, H3, L3, L4) -----------------
        cam_range = prev_h - prev_l
        pd_cam_h4 = prev_close + (cam_range * 1.1 / 2.0)
        pd_cam_h3 = prev_close + (cam_range * 1.1 / 4.0)
        pd_cam_l3 = prev_close - (cam_range * 1.1 / 4.0)
        pd_cam_l4 = prev_close - (cam_range * 1.1 / 2.0)

        if ltp > pd_cam_h4:
            cam_status = "🚀 Above H4 (Breakout)"
        elif ltp > pd_cam_h3:
            cam_status = "⚡ Above H3"
        elif ltp < pd_cam_l4:
            cam_status = "🔻 Below L4 (Breakdown)"
        elif ltp < pd_cam_l3:
            cam_status = "⚡ Below L3"
        else:
            cam_status = "⚖️ Inside L3-H3"

        records.append({
            "Symbol": sym,
            "LTP": round(ltp, 2),
            "Chg%": round(chg_pct, 2),
            "PD_VAH": vp["PD_VAH"],
            "PD_VAL": vp["PD_VAL"],
            "PD_POC": vp["PD_POC"],
            "52W_High": round(h52, 2),
            "52W_Low": round(l52, 2),
            "Day_High": round(day_high, 2),
            "Day_Low": round(day_low, 2),
            "Day_Volume": int(day_vol),
            "Prev_Close": round(prev_close, 2),
            "PD_CPR_Top": round(pd_cpr_top, 2),
            "PD_CPR_Pivot": round(pd_pivot, 2),
            "PD_CPR_Bottom": round(pd_cpr_bottom, 2),
            "PD_CPR_Width": round(pd_cpr_width, 2),
            "PD_CPR_Width%": round(pd_cpr_width_pct, 3),
            "CPR_Type": cpr_type,
            "Is_Narrow_CPR": is_narrow_cpr,
            "CPR_Status": cpr_status,
            "PD_Cam_H4": round(pd_cam_h4, 2),
            "PD_Cam_H3": round(pd_cam_h3, 2),
            "PD_Cam_L3": round(pd_cam_l3, 2),
            "PD_Cam_L4": round(pd_cam_l4, 2),
            "Cam_Status": cam_status,
            "BB_Upper": bb_info["upper"],
            "BB_Middle": bb_info["sma20"],
            "BB_Lower": bb_info["lower"],
            "BB_Bandwidth%": bb_info["bandwidth"],
            "BB_Min_Bandwidth%": bb_info["min_bandwidth"],
            "BB_Is_Squeezing": bb_info["is_squeezing"],
            "BB_Was_Squeezing": bb_info["was_squeezing"],
            "BB_Status": bb_info["status"],
            "BB_Bull_Signal": bb_info["bullish_squeeze_signal"],
            "BB_Bear_Signal": bb_info["bearish_squeeze_signal"],
            "Ticker": t
        })

    if not records:
        empty = pd.DataFrame()
        return empty, empty, empty, {"ref_date": ref_date_str, "live_date": live_date_str}

    df = pd.DataFrame(records)

    # ----------------- VECTORIZED SCREENING LOGIC -----------------
    
    # 1. Core Value Area & 52-Week Breakout Conditions
    cond_bull_vah = df["LTP"] > df["PD_VAH"]
    cond_bull_52w = df["LTP"] >= (df["52W_High"] * high_proximity_pct)

    cond_bear_val = df["LTP"] < df["PD_VAL"]
    cond_bear_52w = df["LTP"] <= (df["52W_Low"] * low_proximity_pct)

    # 2. CPR Filter (Money Zone Criteria)
    if enable_cpr_filter:
        cond_bull_cpr = df["LTP"] > df["PD_CPR_Top"]
        cond_bear_cpr = df["LTP"] < df["PD_CPR_Bottom"]
    else:
        cond_bull_cpr = pd.Series(True, index=df.index)
        cond_bear_cpr = pd.Series(True, index=df.index)

    # 3. Camarilla Pivot Filter Criteria
    if camarilla_filter == "H4 / L4 Breakout (Strong)":
        cond_bull_cam = df["LTP"] > df["PD_Cam_H4"]
        cond_bear_cam = df["LTP"] < df["PD_Cam_L4"]
    elif camarilla_filter == "H3 / L3 Range Breakout":
        cond_bull_cam = df["LTP"] > df["PD_Cam_H3"]
        cond_bear_cam = df["LTP"] < df["PD_Cam_L3"]
    else:
        cond_bull_cam = pd.Series(True, index=df.index)
        cond_bear_cam = pd.Series(True, index=df.index)

    # 4. Bollinger Bands Squeeze & Breakout Filter
    if bollinger_filter == "Squeeze Breakout (Master Confluence)":
        cond_bull_bb = df["BB_Bull_Signal"] | (df["BB_Was_Squeezing"] & (df["LTP"] > df["BB_Upper"]))
        cond_bear_bb = df["BB_Bear_Signal"] | (df["BB_Was_Squeezing"] & (df["LTP"] < df["BB_Lower"]))
    elif bollinger_filter == "Currently Squeezing (Coil)":
        cond_bull_bb = df["BB_Is_Squeezing"]
        cond_bear_bb = df["BB_Is_Squeezing"]
    elif bollinger_filter == "Price Outside Bands (Upper/Lower)":
        cond_bull_bb = df["LTP"] > df["BB_Upper"]
        cond_bear_bb = df["LTP"] < df["BB_Lower"]
    else:
        cond_bull_bb = pd.Series(True, index=df.index)
        cond_bear_bb = pd.Series(True, index=df.index)

    # 5. Narrow CPR Filter
    if only_narrow_cpr:
        cond_narrow_cpr = df["Is_Narrow_CPR"]
    else:
        cond_narrow_cpr = pd.Series(True, index=df.index)

    # Final Combined Triggers
    df["Bullish_Breakout"] = cond_bull_vah & cond_bull_52w & cond_bull_cpr & cond_bull_cam & cond_bull_bb & cond_narrow_cpr
    df["Bearish_Breakdown"] = cond_bear_val & cond_bear_52w & cond_bear_cpr & cond_bear_cam & cond_bear_bb & cond_narrow_cpr

    # Distance Metrics
    df["Dist_VAH%"] = ((df["LTP"] - df["PD_VAH"]) / df["PD_VAH"] * 100.0).round(2)
    df["Dist_VAL%"] = ((df["LTP"] - df["PD_VAL"]) / df["PD_VAL"] * 100.0).round(2)
    df["Dist_52W_High%"] = ((df["LTP"] - df["52W_High"]) / df["52W_High"] * 100.0).round(2)
    df["Dist_52W_Low%"] = ((df["LTP"] - df["52W_Low"]) / df["52W_Low"] * 100.0).round(2)
    df["Dist_CPR_Top%"] = ((df["LTP"] - df["PD_CPR_Top"]) / df["PD_CPR_Top"] * 100.0).round(2)
    df["Dist_Cam_H4%"] = ((df["LTP"] - df["PD_Cam_H4"]) / df["PD_Cam_H4"] * 100.0).round(2)

    # Classify Signal Tag
    conditions = [
        df["Bullish_Breakout"],
        df["Bearish_Breakdown"],
        df["BB_Bull_Signal"],
        df["BB_Is_Squeezing"],
        df["LTP"] > df["PD_VAH"],
        df["LTP"] < df["PD_VAL"]
    ]
    choices = [
        "🚀 BULLISH BREAKOUT",
        "🔻 BEARISH BREAKDOWN",
        "💥 BB Squeeze Breakout",
        "🌀 BB Squeezing (Coil)",
        "⚡ Above VAH (Not Near 52W High)",
        "⚡ Below VAL (Not Near 52W Low)"
    ]
    df["Signal"] = np.select(conditions, choices, default="In Value Area")

    # Filtered DataFrames
    bullish_df = df[df["Bullish_Breakout"]].sort_values(by="Dist_52W_High%", ascending=False).copy()
    bearish_df = df[df["Bearish_Breakdown"]].sort_values(by="Dist_52W_Low%", ascending=True).copy()

    # Sort main dataframe by Symbol
    all_df = df.sort_values(by="Symbol").reset_index(drop=True)

    squeezing_count = int(df["BB_Is_Squeezing"].sum())
    narrow_cpr_count = int(df["Is_Narrow_CPR"].sum())

    meta = {
        "ref_date": ref_date_str,
        "live_date": live_date_str,
        "total_scanned": len(df),
        "bullish_count": len(bullish_df),
        "bearish_count": len(bearish_df),
        "squeezing_count": squeezing_count,
        "narrow_cpr_count": narrow_cpr_count,
        "only_narrow_cpr": only_narrow_cpr,
        "cpr_filter_active": enable_cpr_filter,
        "camarilla_filter_mode": camarilla_filter,
        "bollinger_filter_mode": bollinger_filter
    }

    return all_df, bullish_df, bearish_df, meta


def get_single_stock_chart_data(symbol: str, timeframe: str = "5m") -> Dict:
    """
    Fetches intraday data and computes Volume Profile, CPR, Camarilla, and Bollinger Bands for charting
    """
    ticker = get_yf_symbol(symbol)
    df = yf.download(ticker, period="5d", interval=timeframe, progress=False)
    if df.empty:
        return {}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    dates = sorted(list(set(df.index.date)))
    if len(dates) < 2:
        return {}

    prev_date = dates[-2]
    curr_date = dates[-1]

    prev_df = df[df.index.date == prev_date]
    curr_df = df[df.index.date == curr_date]

    vp = calculate_volume_profile(prev_df)
    bins, vols = get_profile_histogram(prev_df, num_bins=40)

    prev_h = vp["Day_High"]
    prev_l = vp["Day_Low"]
    prev_c = vp["Day_Close"]

    # CPR
    pd_pivot = (prev_h + prev_l + prev_c) / 3.0
    pd_bc = (prev_h + prev_l) / 2.0
    pd_tc = (2.0 * pd_pivot) - pd_bc
    pd_width = abs(pd_tc - pd_bc)
    pd_width_pct = (pd_width / pd_pivot * 100.0) if pd_pivot > 0 else 0.0
    cpr_type = "🎯 Ultra Narrow" if pd_width_pct <= 0.15 else ("⚡ Narrow" if pd_width_pct <= 0.35 else ("Average" if pd_width_pct <= 0.70 else "Wide"))
    cpr = {
        "top": round(max(pd_tc, pd_bc), 2),
        "pivot": round(pd_pivot, 2),
        "bottom": round(min(pd_tc, pd_bc), 2),
        "width": round(pd_width, 2),
        "width_pct": round(pd_width_pct, 3),
        "type": cpr_type
    }

    # Camarilla
    cam_rng = prev_h - prev_l
    cam = {
        "h4": round(prev_c + (cam_rng * 1.1 / 2.0), 2),
        "h3": round(prev_c + (cam_rng * 1.1 / 4.0), 2),
        "l3": round(prev_c - (cam_rng * 1.1 / 4.0), 2),
        "l4": round(prev_c - (cam_rng * 1.1 / 2.0), 2)
    }

    # Bollinger Bands for daily
    try:
        daily_df = yf.download(ticker, period="1y", interval="1d", progress=False)
        bb = calculate_bollinger_bands(daily_df)
    except Exception:
        bb = {}

    return {
        "prev_df": prev_df,
        "curr_df": curr_df,
        "vp": vp,
        "cpr": cpr,
        "cam": cam,
        "bb": bb,
        "bins": bins,
        "vols": vols,
        "prev_date": str(prev_date),
        "curr_date": str(curr_date)
    }
