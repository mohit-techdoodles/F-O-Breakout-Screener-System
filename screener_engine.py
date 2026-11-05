"""
Core Screener Engine: Data Pipeline & Vectorized Trigger Logic
Implements Setup A (Bullish Breakout) & Setup B (Bearish Breakdown)
"""

import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from fo_stocks import get_yf_symbol
from volume_profile import calculate_volume_profile, get_profile_histogram


def fetch_and_screen(
    symbols: List[str], 
    timeframe: str = "5m", 
    high_proximity_pct: float = 0.95,
    low_proximity_pct: float = 1.05
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Downloads intraday and 52-week data in batch, computes Volume Profile
    and screens for Bullish Breakout and Bearish Breakdown triggers.
    
    Returns:
    - (all_df, bullish_df, bearish_df, meta_info)
    """
    if not symbols:
        empty = pd.DataFrame()
        return empty, empty, empty, {}

    yf_tickers = [get_yf_symbol(s) for s in symbols]
    ticker_to_sym = {get_yf_symbol(s): s for s in symbols}

    # 1. Batch download 1-year daily data for 52W High & 52W Low
    try:
        df_1y = yf.download(
            yf_tickers,
            period="1y",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True
        )
    except Exception as e:
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
    except Exception as e:
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

        # Extract 52-Week High and Low from daily data
        h52 = np.nan
        l52 = np.nan
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

        # Current Market Data
        ltp = float(curr_candles["Close"].iloc[-1])
        day_open = float(curr_candles["Open"].iloc[0])
        day_high = float(curr_candles["High"].max())
        day_low = float(curr_candles["Low"].min())
        day_vol = float(curr_candles["Volume"].sum())
        prev_close = vp["Day_Close"]

        chg_pct = ((ltp - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0

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
            "Ticker": t
        })

    if not records:
        empty = pd.DataFrame()
        return empty, empty, empty, {"ref_date": ref_date_str, "live_date": live_date_str}

    df = pd.DataFrame(records)

    # Calculate Screening Triggers using Vectorized Logic
    # 1. Bullish Breakout Setup: LTP > PD_VAH AND LTP >= (52W_High * proximity)
    cond_bull_vah = df["LTP"] > df["PD_VAH"]
    cond_bull_52w = df["LTP"] >= (df["52W_High"] * high_proximity_pct)
    df["Bullish_Breakout"] = cond_bull_vah & cond_bull_52w

    # 2. Bearish Breakdown Setup: LTP < PD_VAL AND LTP <= (52W_Low * proximity)
    cond_bear_val = df["LTP"] < df["PD_VAL"]
    cond_bear_52w = df["LTP"] <= (df["52W_Low"] * low_proximity_pct)
    df["Bearish_Breakdown"] = cond_bear_val & cond_bear_52w

    # Distance Metrics
    df["Dist_VAH%"] = ((df["LTP"] - df["PD_VAH"]) / df["PD_VAH"] * 100.0).round(2)
    df["Dist_VAL%"] = ((df["LTP"] - df["PD_VAL"]) / df["PD_VAL"] * 100.0).round(2)
    df["Dist_52W_High%"] = ((df["LTP"] - df["52W_High"]) / df["52W_High"] * 100.0).round(2)
    df["Dist_52W_Low%"] = ((df["LTP"] - df["52W_Low"]) / df["52W_Low"] * 100.0).round(2)

    # Classify Setup Tag
    conditions = [
        df["Bullish_Breakout"],
        df["Bearish_Breakdown"],
        df["LTP"] > df["PD_VAH"],
        df["LTP"] < df["PD_VAL"]
    ]
    choices = [
        "🚀 BULLISH BREAKOUT",
        "🔻 BEARISH BREAKDOWN",
        "⚡ Above VAH (Not Near 52W High)",
        "⚡ Below VAL (Not Near 52W Low)"
    ]
    df["Signal"] = np.select(conditions, choices, default="In Value Area")

    # Filtered DataFrames
    bullish_df = df[df["Bullish_Breakout"]].sort_values(by="Dist_52W_High%", ascending=False).copy()
    bearish_df = df[df["Bearish_Breakdown"]].sort_values(by="Dist_52W_Low%", ascending=True).copy()

    # Sort main dataframe by Symbol
    all_df = df.sort_values(by="Symbol").reset_index(drop=True)

    meta = {
        "ref_date": ref_date_str,
        "live_date": live_date_str,
        "total_scanned": len(df),
        "bullish_count": len(bullish_df),
        "bearish_count": len(bearish_df)
    }

    return all_df, bullish_df, bearish_df, meta


def get_single_stock_chart_data(symbol: str, timeframe: str = "5m") -> Dict:
    """
    Fetches intraday data and computes Volume Profile for charting a single stock
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

    return {
        "prev_df": prev_df,
        "curr_df": curr_df,
        "vp": vp,
        "bins": bins,
        "vols": vols,
        "prev_date": str(prev_date),
        "curr_date": str(curr_date)
    }

