"""
Multi-Timeframe (MTF) Breakout Strategy
Tracking First-Hour Closes against PDH/PDL and 15-Minute VWAP
Implementation based on Developer Guide: MTF Breakout Algorithm
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def calculate_mtf_breakout(
    sub_intra: pd.DataFrame, 
    pdh_fallback: Optional[float] = None, 
    pdl_fallback: Optional[float] = None
) -> Dict[str, any]:
    """
    Computes the MTF Breakout Strategy metrics for an intraday stock dataset:
    1. Daily Timeframe: Previous Day High (PDH) and Previous Day Low (PDL)
    2. 1-Hour Timeframe: First 60-minute closing price (4th 15m candle close at 10:15 AM IST)
    3. 15-Minute Timeframe: Daily Anchored VWAP (Typical Price = (H + L + C) / 3)
    4. Confluence Trade Signals:
       - Bullish: First_1H_Close > PDH AND Current 15m Close > VWAP
       - Bearish: First_1H_Close < PDL AND Current 15m Close < VWAP
    """
    default_result = {
        "pdh": np.nan,
        "pdl": np.nan,
        "first_1h_close": np.nan,
        "prev_1h_close": np.nan,
        "curr_1h_close": np.nan,
        "prev_1h_high": np.nan,
        "prev_1h_low": np.nan,
        "curr_1h_high": np.nan,
        "curr_1h_low": np.nan,
        "vwap": np.nan,
        "dist_vwap_pct": 0.0,
        "mtf_1h_status": "No Data",
        "mtf_vwap_status": "No Data",
        "h1_breakout_type": "No Data",
        "is_1h_breakout_bull": False,
        "is_1h_breakout_bear": False,
        "is_1h_break_in": False,
        "opt_a_status": "No Data",
        "opt_a_bull": False,
        "opt_a_bear": False,
        "opt_a_breakin": False,
        "opt_b_status": "No Data",
        "opt_b_bull": False,
        "opt_b_bear": False,
        "opt_b_breakin": False,
        "opt_c_status": "No Data",
        "opt_c_bull": False,
        "opt_c_bear": False,
        "opt_c_breakin": False,
        "ltp_vs_curr_1h": "N/A",
        "ltp_vs_prev_1h": "N/A",
        "mtf_bullish": False,
        "mtf_bearish": False,
        "mtf_signal": "None",
        "vwap_series": pd.Series(dtype=float),
        "curr_15m_df": pd.DataFrame()
    }

    if sub_intra is None or sub_intra.empty or len(sub_intra) < 2:
        return default_result

    # Ensure clean DatetimeIndex
    df = sub_intra.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return default_result

    # Resample to 15-minute OHLCV candles
    try:
        df_15m = df.resample("15min", closed="left", label="left").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna(subset=["Close"])
    except Exception:
        return default_result

    # Filter to Regular Trading Hours (RTH: 09:15 to 15:30) for Indian / NSE markets
    try:
        df_15m = df_15m.between_time("09:15", "15:30")
    except Exception:
        pass

    if df_15m.empty:
        return default_result

    df_15m["Date"] = df_15m.index.date
    dates = sorted(list(set(df_15m["Date"])))
    curr_date = dates[-1]

    # Daily PDH and PDL calculation from 15m data with shift(1) to avoid lookahead bias
    daily_high_low = df_15m.groupby("Date").agg({"High": "max", "Low": "min"})
    
    if len(dates) >= 2:
        prev_date = dates[-2]
        pdh = float(daily_high_low.loc[prev_date, "High"])
        pdl = float(daily_high_low.loc[prev_date, "Low"])
    else:
        pdh = float(pdh_fallback) if pdh_fallback is not None and not np.isnan(pdh_fallback) else np.nan
        pdl = float(pdl_fallback) if pdl_fallback is not None and not np.isnan(pdl_fallback) else np.nan

    # Today's 15m session bars
    today_bars = df_15m[df_15m["Date"] == curr_date].copy()
    if today_bars.empty:
        return default_result

    # Calculate Daily Anchored VWAP using Typical Price formula: (H + L + C) / 3
    today_bars["Typical_Price"] = (today_bars["High"] + today_bars["Low"] + today_bars["Close"]) / 3.0
    today_bars["Vol_x_Price"] = today_bars["Volume"] * today_bars["Typical_Price"]
    cum_vol_price = today_bars["Vol_x_Price"].cumsum()
    cum_vol = today_bars["Volume"].cumsum()

    # VWAP Series
    vwap_series = np.where(cum_vol > 0, cum_vol_price / cum_vol, today_bars["Close"])
    today_bars["VWAP"] = vwap_series

    latest_vwap = float(today_bars["VWAP"].iloc[-1])
    ltp = float(today_bars["Close"].iloc[-1])

    # Previous Day 1H metrics
    prev_1h_close = np.nan
    prev_1h_high = np.nan
    prev_1h_low = np.nan
    if len(dates) >= 2:
        prev_date = dates[-2]
        prev_bars = df_15m[df_15m["Date"] == prev_date]
        if len(prev_bars) >= 4:
            prev_1h_close = float(prev_bars["Close"].iloc[3])
            prev_1h_high = float(prev_bars["High"].iloc[:4].max())
            prev_1h_low = float(prev_bars["Low"].iloc[:4].min())

    # Current Day 1H metrics
    first_1h_close = np.nan
    curr_1h_close = np.nan
    curr_1h_high = np.nan
    curr_1h_low = np.nan
    if len(today_bars) >= 4:
        curr_1h_close = float(today_bars["Close"].iloc[3])
        curr_1h_high = float(today_bars["High"].iloc[:4].max())
        curr_1h_low = float(today_bars["Low"].iloc[:4].min())
        first_1h_close = curr_1h_close

    # ----------------- Option A: 1H Close vs Prev 1H Close -----------------
    if np.isnan(curr_1h_close) or np.isnan(prev_1h_close):
        opt_a_status = "⏳ Pending (< 10:15)"
        opt_a_bull = False
        opt_a_bear = False
        opt_a_breakin = False
    else:
        if curr_1h_close > prev_1h_close:
            opt_a_status = "🟢 Breakout > Prev 1H Close"
            opt_a_bull = True
            opt_a_bear = False
            opt_a_breakin = False
        elif curr_1h_close < prev_1h_close:
            opt_a_status = "🔴 Breakdown < Prev 1H Close"
            opt_a_bull = False
            opt_a_bear = True
            opt_a_breakin = False
        else:
            opt_a_status = "⚪ Break-in / Equal"
            opt_a_bull = False
            opt_a_bear = False
            opt_a_breakin = True

    # ----------------- Option B: Live Market Price vs 1H Close -----------------
    day_high = float(today_bars["High"].max()) if not today_bars.empty else ltp
    day_low = float(today_bars["Low"].min()) if not today_bars.empty else ltp
    if np.isnan(curr_1h_close):
        opt_b_status = "⏳ Pending (< 10:15)"
        opt_b_bull = False
        opt_b_bear = False
        opt_b_breakin = False
    else:
        b_broke_above = (day_high > curr_1h_close)
        b_broke_below = (day_low < curr_1h_close)
        
        if b_broke_above and (ltp <= curr_1h_close):
            opt_b_status = "🔄 Break-in (Failed 1H Close Breakout)"
            opt_b_bull = False
            opt_b_bear = False
            opt_b_breakin = True
        elif b_broke_below and (ltp >= curr_1h_close):
            opt_b_status = "🔄 Break-in (Failed 1H Close Breakdown)"
            opt_b_bull = False
            opt_b_bear = False
            opt_b_breakin = True
        elif ltp > curr_1h_close:
            opt_b_status = "🟢 Live Price > 1H Close"
            opt_b_bull = True
            opt_b_bear = False
            opt_b_breakin = False
        elif ltp < curr_1h_close:
            opt_b_status = "🔴 Live Price < 1H Close"
            opt_b_bull = False
            opt_b_bear = True
            opt_b_breakin = False
        else:
            opt_b_status = "⚪ At 1H Close"
            opt_b_bull = False
            opt_b_bear = False
            opt_b_breakin = True

    # ----------------- Option C: Initial Balance (1H High / Low Range) -----------------
    if np.isnan(curr_1h_high) or np.isnan(curr_1h_low):
        opt_c_status = "⏳ Pending (< 10:15)"
        opt_c_bull = False
        opt_c_bear = False
        opt_c_breakin = False
    else:
        c_broke_above = (day_high > curr_1h_high)
        c_broke_below = (day_low < curr_1h_low)
        
        if ltp > curr_1h_high:
            opt_c_status = "🟢 Range Breakout > 1H High"
            opt_c_bull = True
            opt_c_bear = False
            opt_c_breakin = False
        elif ltp < curr_1h_low:
            opt_c_status = "🔴 Range Breakdown < 1H Low"
            opt_c_bull = False
            opt_c_bear = True
            opt_c_breakin = False
        elif (c_broke_above or c_broke_below) and (curr_1h_low <= ltp <= curr_1h_high):
            opt_c_status = "🔄 Range Break-in (Failed Breakout)"
            opt_c_bull = False
            opt_c_bear = False
            opt_c_breakin = True
        else:
            opt_c_status = "⚪ Inside 1H Range"
            opt_c_bull = False
            opt_c_bear = False
            opt_c_breakin = True

    # Current LTP vs 1H Closes
    ltp_vs_curr_1h = "🟢 Above Curr 1H" if (not np.isnan(curr_1h_close) and ltp > curr_1h_close) else ("🔴 Below Curr 1H" if (not np.isnan(curr_1h_close) and ltp < curr_1h_close) else "N/A")
    ltp_vs_prev_1h = "🟢 Above Prev 1H" if (not np.isnan(prev_1h_close) and ltp > prev_1h_close) else ("🔴 Below Prev 1H" if (not np.isnan(prev_1h_close) and ltp < prev_1h_close) else "N/A")

    # Status classification
    if np.isnan(first_1h_close):
        mtf_1h_status = "⏳ Pending (< 10:15)"
    elif not np.isnan(pdh) and first_1h_close > pdh:
        mtf_1h_status = "🟢 Breakout > PDH"
    elif not np.isnan(pdl) and first_1h_close < pdl:
        mtf_1h_status = "🔴 Breakdown < PDL"
    else:
        mtf_1h_status = "⚪ Inside Range"

    # VWAP Status
    if not np.isnan(latest_vwap) and latest_vwap > 0:
        dist_vwap_pct = round(((ltp - latest_vwap) / latest_vwap * 100.0), 2)
        if ltp > latest_vwap:
            mtf_vwap_status = "🟢 Above VWAP"
        elif ltp < latest_vwap:
            mtf_vwap_status = "🔴 Below VWAP"
        else:
            mtf_vwap_status = "⚪ At VWAP"
    else:
        dist_vwap_pct = 0.0
        mtf_vwap_status = "N/A"

    # Confluence Signals (Page 2 & Page 4)
    # Bullish: First 1H Close > PDH AND Current 15m Close > VWAP
    # Bearish: First 1H Close < PDL AND Current 15m Close < VWAP
    mtf_bullish = False
    mtf_bearish = False
    if not np.isnan(first_1h_close) and not np.isnan(pdh) and not np.isnan(pdl) and not np.isnan(latest_vwap):
        if (first_1h_close > pdh) and (ltp > latest_vwap):
            mtf_bullish = True
        elif (first_1h_close < pdl) and (ltp < latest_vwap):
            mtf_bearish = True

    if mtf_bullish:
        mtf_signal = "🚀 MTF Bullish"
    elif mtf_bearish:
        mtf_signal = "🩸 MTF Bearish"
    else:
        mtf_signal = "None"

    return {
        "pdh": round(pdh, 2) if not np.isnan(pdh) else np.nan,
        "pdl": round(pdl, 2) if not np.isnan(pdl) else np.nan,
        "first_1h_close": round(first_1h_close, 2) if not np.isnan(first_1h_close) else np.nan,
        "prev_1h_close": round(prev_1h_close, 2) if not np.isnan(prev_1h_close) else np.nan,
        "curr_1h_close": round(curr_1h_close, 2) if not np.isnan(curr_1h_close) else np.nan,
        "prev_1h_high": round(prev_1h_high, 2) if not np.isnan(prev_1h_high) else np.nan,
        "prev_1h_low": round(prev_1h_low, 2) if not np.isnan(prev_1h_low) else np.nan,
        "curr_1h_high": round(curr_1h_high, 2) if not np.isnan(curr_1h_high) else np.nan,
        "curr_1h_low": round(curr_1h_low, 2) if not np.isnan(curr_1h_low) else np.nan,
        "vwap": round(latest_vwap, 2) if not np.isnan(latest_vwap) else np.nan,
        "dist_vwap_pct": dist_vwap_pct,
        "mtf_1h_status": mtf_1h_status,
        "mtf_vwap_status": mtf_vwap_status,
        "h1_breakout_type": opt_a_status,
        "is_1h_breakout_bull": opt_a_bull,
        "is_1h_breakout_bear": opt_a_bear,
        "is_1h_break_in": opt_a_breakin,
        "opt_a_status": opt_a_status,
        "opt_a_bull": opt_a_bull,
        "opt_a_bear": opt_a_bear,
        "opt_a_breakin": opt_a_breakin,
        "opt_b_status": opt_b_status,
        "opt_b_bull": opt_b_bull,
        "opt_b_bear": opt_b_bear,
        "opt_b_breakin": opt_b_breakin,
        "opt_c_status": opt_c_status,
        "opt_c_bull": opt_c_bull,
        "opt_c_bear": opt_c_bear,
        "opt_c_breakin": opt_c_breakin,
        "ltp_vs_curr_1h": ltp_vs_curr_1h,
        "ltp_vs_prev_1h": ltp_vs_prev_1h,
        "mtf_bullish": mtf_bullish,
        "mtf_bearish": mtf_bearish,
        "mtf_signal": mtf_signal,
        "vwap_series": today_bars["VWAP"],
        "curr_15m_df": today_bars
    }


# TradingView Pine Script (v5) code snippet from Section 2 of Developer Guide
PINE_SCRIPT_V5_CODE = '''//@version=5
indicator("1H Breakout + 15m VWAP", overlay=true)

// 1. Get Previous Day High & Low (Avoids repainting during the current day)
pdh = request.security(syminfo.tickerid, "D", high[1], lookahead=barmerge.lookahead_on)
pdl = request.security(syminfo.tickerid, "D", low[1], lookahead=barmerge.lookahead_on)

// 2. Calculate Daily Anchored VWAP
vwapValue = ta.vwap(close)

// 3. Track First Hourly Close
var float firstHourClose = na
isNewDay = ta.change(time("D"))
if isNewDay
    firstHourClose := na

var int barsToday = 0
if isNewDay
    barsToday := 1
else
    barsToday += 1

// Lock in the closing price on the 4th 15-minute bar (10:15 AM IST)
if barsToday == 4
    firstHourClose := close

// 4. Evaluate Conditions
bullishCond = (barsToday > 4) and (firstHourClose > pdh) and (close > vwapValue)
bearishCond = (barsToday > 4) and (firstHourClose < pdl) and (close < vwapValue)

// 5. Plotting & Visuals
plot(pdh, title="PDH", color=color.green, style=plot.style_linebr, linewidth=2)
plot(pdl, title="PDL", color=color.red, style=plot.style_linebr, linewidth=2)
plot(vwapValue, title="VWAP", color=color.blue, linewidth=2)
plot(firstHourClose, title="1H Close", color=color.purple, style=plot.style_circles)

plotshape(bullishCond and not bullishCond[1], title="Bullish", style=shape.triangleup, location=location.belowbar, color=color.green, size=size.normal)
plotshape(bearishCond and not bearishCond[1], title="Bearish", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.normal)
'''
