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
from quantitative_filters import get_quant_metrics_for_stock
from mtf_breakout import calculate_mtf_breakout


def fetch_universe_data(
    symbols: List[str], 
    timeframe: str = "5m"
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Downloads market data and computes all indicators for all stocks.
    This is the heavy network + compute function that should be cached.
    """
    if not symbols:
        empty = pd.DataFrame()
        return empty, {}

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

        is_narrow_cpr = pd_cpr_width_pct <= 0.35

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

        # ----------------- Quantitative Filters (RVOL & BBW Contraction) -----------------
        quant_info = get_quant_metrics_for_stock(sub_1y, curr_day_vol=day_vol)

        # ----------------- MTF Breakout Algorithm (PDH/PDL + 1H Close + 15m VWAP) -----------------
        mtf_info = calculate_mtf_breakout(sub_intra, pdh_fallback=prev_h, pdl_fallback=prev_l)

        # ----------------- Tomorrow's CPR & Camarilla (Next Session Projection) -----------------
        next_pivot = (day_high + day_low + ltp) / 3.0
        next_bc = (day_high + day_low) / 2.0
        next_tc = (2.0 * next_pivot) - next_bc
        next_cpr_top = max(next_tc, next_bc)
        next_cpr_bottom = min(next_tc, next_bc)
        next_cpr_width = abs(next_tc - next_bc)
        next_cpr_width_pct = (next_cpr_width / next_pivot * 100.0) if next_pivot > 0 else 0.0

        if next_cpr_width_pct <= 0.15:
            next_cpr_type = "🎯 Ultra Narrow"
        elif next_cpr_width_pct <= 0.35:
            next_cpr_type = "⚡ Narrow"
        elif next_cpr_width_pct <= 0.70:
            next_cpr_type = "Average"
        else:
            next_cpr_type = "Wide"

        next_is_narrow = next_cpr_width_pct <= 0.35

        # Tomorrow's Camarilla Levels
        next_cam_rng = day_high - day_low
        next_cam_h4 = ltp + (next_cam_rng * 1.1 / 2.0)
        next_cam_h3 = ltp + (next_cam_rng * 1.1 / 4.0)
        next_cam_l3 = ltp - (next_cam_rng * 1.1 / 4.0)
        next_cam_l4 = ltp - (next_cam_rng * 1.1 / 2.0)

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
            # Quantitative Filters (RVOL & BBW Contraction)
            "RVOL": quant_info["rvol"],
            "Volume_SMA20": quant_info["volume_sma"],
            "BBW": quant_info["bbw"],
            "BBW_SMA20": quant_info["bbw_sma"],
            "BBW_Is_Contracting": quant_info["is_contracting"],
            "Prior_BBW_Contracting": quant_info["prior_is_contracting"],
            "RVOL_Status": f"🔥 {quant_info['rvol']}x" if quant_info['rvol'] >= 1.5 else f"{quant_info['rvol']}x",
            "BBW_Contract_Status": "⚡ Contracted" if quant_info["prior_is_contracting"] else "Normal",
            # NR7 & OI Buildup (Section 5)
            "Is_NR7": quant_info["is_nr7"],
            "Prior_NR7": quant_info["prior_is_nr7"],
            "NR7_Status": "⚡ NR7 (Coil)" if quant_info["is_nr7"] else ("Prior NR7" if quant_info["prior_is_nr7"] else "Normal"),
            "OI_Buildup": quant_info["oi_buildup"],
            # Weekly Context (Sections 6 & 7)
            "W_Pivot": quant_info["w_pivot"],
            "W_Top_CPR": quant_info["w_top_cpr"],
            "W_Bottom_CPR": quant_info["w_bottom_cpr"],
            "W_CPR_Bias": quant_info["w_cpr_bias"],
            "W_VAH": quant_info["w_vah"],
            "W_VAL": quant_info["w_val"],
            "W_POC": quant_info["w_poc"],
            # MTF Breakout Algorithm (PDH/PDL + 1H Close + 15m VWAP)
            "PDH": mtf_info["pdh"],
            "PDL": mtf_info["pdl"],
            "First_1H_Close": mtf_info["first_1h_close"],
            "VWAP": mtf_info["vwap"],
            "Dist_VWAP%": mtf_info["dist_vwap_pct"],
            "MTF_1H_Status": mtf_info["mtf_1h_status"],
            "MTF_VWAP_Status": mtf_info["mtf_vwap_status"],
            "MTF_Bullish": mtf_info["mtf_bullish"],
            "MTF_Bearish": mtf_info["mtf_bearish"],
            "MTF_Signal": mtf_info["mtf_signal"],
            # Initial Hour (1H) Breakout & Break-in Tracking
            "Prev_1H_Close": mtf_info["prev_1h_close"],
            "Curr_1H_Close": mtf_info["curr_1h_close"],
            "Prev_1H_High": mtf_info["prev_1h_high"],
            "Prev_1H_Low": mtf_info["prev_1h_low"],
            "Curr_1H_High": mtf_info["curr_1h_high"],
            "Curr_1H_Low": mtf_info["curr_1h_low"],
            "H1_Breakout_Type": mtf_info["h1_breakout_type"],
            "Is_1H_Breakout_Bull": mtf_info["is_1h_breakout_bull"],
            "Is_1H_Breakout_Bear": mtf_info["is_1h_breakout_bear"],
            "Is_1H_Break_In": mtf_info["is_1h_break_in"],
            "LTP_vs_Curr_1H": mtf_info["ltp_vs_curr_1h"],
            "LTP_vs_Prev_1H": mtf_info["ltp_vs_prev_1h"],
            # Options A, B, and C Initial Hour Status
            "Opt_A_Status": mtf_info.get("opt_a_status", "N/A"),
            "Opt_A_Bull": mtf_info.get("opt_a_bull", False),
            "Opt_A_Bear": mtf_info.get("opt_a_bear", False),
            "Opt_A_Breakin": mtf_info.get("opt_a_breakin", False),
            "Opt_B_Status": mtf_info.get("opt_b_status", "N/A"),
            "Opt_B_Bull": mtf_info.get("opt_b_bull", False),
            "Opt_B_Bear": mtf_info.get("opt_b_bear", False),
            "Opt_B_Breakin": mtf_info.get("opt_b_breakin", False),
            "Opt_C_Status": mtf_info.get("opt_c_status", "N/A"),
            "Opt_C_Bull": mtf_info.get("opt_c_bull", False),
            "Opt_C_Bear": mtf_info.get("opt_c_bear", False),
            "Opt_C_Breakin": mtf_info.get("opt_c_breakin", False),
            # Tomorrow's Projections
            "Next_CPR_Top": round(next_cpr_top, 2),
            "Next_CPR_Pivot": round(next_pivot, 2),
            "Next_CPR_Bottom": round(next_cpr_bottom, 2),
            "Next_CPR_Width": round(next_cpr_width, 2),
            "Next_CPR_Width%": round(next_cpr_width_pct, 3),
            "Next_CPR_Type": next_cpr_type,
            "Next_Is_Narrow": next_is_narrow,
            "Next_Cam_H4": round(next_cam_h4, 2),
            "Next_Cam_H3": round(next_cam_h3, 2),
            "Next_Cam_L3": round(next_cam_l3, 2),
            "Next_Cam_L4": round(next_cam_l4, 2),
            "Ticker": t
        })

    if not records:
        empty = pd.DataFrame()
        return empty, {"ref_date": ref_date_str, "live_date": live_date_str, "total_scanned": 0}

    df = pd.DataFrame(records)

    # Distance Metrics precalculated
    df["Dist_VAH%"] = ((df["LTP"] - df["PD_VAH"]) / df["PD_VAH"] * 100.0).round(2)
    df["Dist_VAL%"] = ((df["LTP"] - df["PD_VAL"]) / df["PD_VAL"] * 100.0).round(2)
    df["Dist_52W_High%"] = ((df["LTP"] - df["52W_High"]) / df["52W_High"] * 100.0).round(2)
    df["Dist_52W_Low%"] = ((df["LTP"] - df["52W_Low"]) / df["52W_Low"] * 100.0).round(2)
    df["Dist_CPR_Top%"] = ((df["LTP"] - df["PD_CPR_Top"]) / df["PD_CPR_Top"] * 100.0).round(2)
    df["Dist_Cam_H4%"] = ((df["LTP"] - df["PD_Cam_H4"]) / df["PD_Cam_H4"] * 100.0).round(2)
    df["Dist_W_VAH%"] = np.where(df["W_VAH"] > 0, ((df["LTP"] - df["W_VAH"]) / df["W_VAH"] * 100.0).round(2), 0.0)
    df["Dist_W_VAL%"] = np.where(df["W_VAL"] > 0, ((df["LTP"] - df["W_VAL"]) / df["W_VAL"] * 100.0).round(2), 0.0)

    base_meta = {
        "ref_date": ref_date_str,
        "live_date": live_date_str,
        "total_scanned": len(df)
    }

    return df, base_meta


def apply_screener_filters(
    master_df: pd.DataFrame,
    base_meta: Dict[str, any],
    high_proximity_pct: float = 0.95,
    low_proximity_pct: float = 1.05,
    ignore_52w: bool = False,
    h1_close_filter: str = "All",
    enable_cpr_filter: bool = True,
    only_narrow_cpr: bool = False,
    narrow_cpr_max_pct: float = 0.35,
    camarilla_filter: str = "None",
    bollinger_filter: str = "None",
    enable_rvol: bool = False,
    rvol_threshold: float = 1.5,
    enable_bbw_squeeze: bool = False,
    enable_weekly_filter: bool = False,
    enable_nr7_filter: bool = False,
    enable_mtf_filter: bool = False,
    mtf_mode: str = "Any / Both",
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Applies user filter conditions entirely in memory in < 1 millisecond.
    """
    if master_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, base_meta

    df = master_df.copy()

    # 1. Core Value Area & 52-Week Breakout Conditions
    cond_bull_vah = df["LTP"] > df["PD_VAH"]
    cond_bear_val = df["LTP"] < df["PD_VAL"]

    if ignore_52w:
        cond_bull_52w = pd.Series(True, index=df.index)
        cond_bear_52w = pd.Series(True, index=df.index)
    else:
        cond_bull_52w = df["LTP"] >= (df["52W_High"] * high_proximity_pct)
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

    # 5. Narrow CPR Filter (dynamic threshold)
    if only_narrow_cpr:
        cond_narrow_cpr = df["PD_CPR_Width%"] <= narrow_cpr_max_pct
    else:
        cond_narrow_cpr = pd.Series(True, index=df.index)

    # 6. Quantitative Filter: Relative Volume (RVOL)
    if enable_rvol:
        cond_rvol = df["RVOL"] >= rvol_threshold
    else:
        cond_rvol = pd.Series(True, index=df.index)

    # 7. Quantitative Filter: BBW Contraction (shift(1) check: contraction prior to breakout day)
    if enable_bbw_squeeze:
        cond_bbw_contract = df["Prior_BBW_Contracting"]
    else:
        cond_bbw_contract = pd.Series(True, index=df.index)

    # 8. Weekly Context Filters (Section 8 of PDF Specification)
    if enable_weekly_filter:
        cond_bull_weekly = (df["LTP"] > df["W_VAH"]) & df["W_CPR_Bias"].isin(["Bullish (Higher Value)", "Breakout (Inside Value)"])
        cond_bear_weekly = (df["LTP"] < df["W_VAL"]) & df["W_CPR_Bias"].isin(["Bearish (Lower Value)", "Breakout (Inside Value)"])
    else:
        cond_bull_weekly = pd.Series(True, index=df.index)
        cond_bear_weekly = pd.Series(True, index=df.index)

    # 9. NR7 Compression Filter (Section 5 of PDF Specification)
    if enable_nr7_filter:
        cond_nr7 = df["Prior_NR7"] | df["Is_NR7"]
    else:
        cond_nr7 = pd.Series(True, index=df.index)

    # 10. Multi-Timeframe (MTF) Breakout Strategy (PDH/PDL + 1H Close + 15m VWAP)
    if enable_mtf_filter:
        if mtf_mode == "Bullish (1H > PDH & Above VWAP)":
            cond_mtf_bull = df["MTF_Bullish"]
            cond_mtf_bear = pd.Series(False, index=df.index)
        elif mtf_mode == "Bearish (1H < PDL & Below VWAP)":
            cond_mtf_bull = pd.Series(False, index=df.index)
            cond_mtf_bear = df["MTF_Bearish"]
        else:  # "Any / Both"
            cond_mtf_bull = df["MTF_Bullish"]
            cond_mtf_bear = df["MTF_Bearish"]
    else:
        cond_mtf_bull = pd.Series(True, index=df.index)
        cond_mtf_bear = pd.Series(True, index=df.index)

    # 11. Initial Hour (1H) Filters: Option A, Option B, and Option C
    if h1_close_filter in ["Option A: Breakout > Prev 1H (Bullish)", "Breakout > Prev 1H Close"]:
        cond_h1_bull = df["Opt_A_Bull"]
        cond_h1_bear = pd.Series(False, index=df.index)
    elif h1_close_filter in ["Option A: Breakdown < Prev 1H (Bearish)", "Breakdown < Prev 1H Close"]:
        cond_h1_bull = pd.Series(False, index=df.index)
        cond_h1_bear = df["Opt_A_Bear"]
    elif h1_close_filter in ["Option A: Break-in (Inside Prev 1H)", "Break-in / Equal"]:
        cond_h1_bull = df["Opt_A_Breakin"]
        cond_h1_bear = df["Opt_A_Breakin"]

    elif h1_close_filter in ["Option B: Price > 1H Close (Bullish)", "Price > Curr 1H Close"]:
        cond_h1_bull = df["Opt_B_Bull"]
        cond_h1_bear = pd.Series(False, index=df.index)
    elif h1_close_filter in ["Option B: Price < 1H Close (Bearish)", "Price < Curr 1H Close"]:
        cond_h1_bull = pd.Series(False, index=df.index)
        cond_h1_bear = df["Opt_B_Bear"]
    elif h1_close_filter == "Option B: Break-in (Failed 1H Close Breakout)":
        cond_h1_bull = df["Opt_B_Breakin"]
        cond_h1_bear = df["Opt_B_Breakin"]

    elif h1_close_filter == "Option C: Range Breakout > 1H High (Bullish)":
        cond_h1_bull = df["Opt_C_Bull"]
        cond_h1_bear = pd.Series(False, index=df.index)
    elif h1_close_filter == "Option C: Range Breakdown < 1H Low (Bearish)":
        cond_h1_bull = pd.Series(False, index=df.index)
        cond_h1_bear = df["Opt_C_Bear"]
    elif h1_close_filter == "Option C: Range Break-in (Failed Breakout / Re-entered Range)":
        cond_h1_bull = df["Opt_C_Breakin"]
        cond_h1_bear = df["Opt_C_Breakin"]
    else:
        cond_h1_bull = pd.Series(True, index=df.index)
        cond_h1_bear = pd.Series(True, index=df.index)

    # Final Combined Triggers
    df["Bullish_Breakout"] = (
        cond_bull_vah & cond_bull_52w & cond_bull_cpr & cond_bull_cam & 
        cond_bull_bb & cond_narrow_cpr & cond_rvol & cond_bbw_contract & 
        cond_bull_weekly & cond_nr7 & cond_mtf_bull & cond_h1_bull
    )
    df["Bearish_Breakdown"] = (
        cond_bear_val & cond_bear_52w & cond_bear_cpr & cond_bear_cam & 
        cond_bear_bb & cond_narrow_cpr & cond_rvol & cond_bbw_contract & 
        cond_bear_weekly & cond_nr7 & cond_mtf_bear & cond_h1_bear
    )

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
    narrow_cpr_count = int((df["PD_CPR_Width%"] <= narrow_cpr_max_pct).sum())
    next_narrow_cpr_count = int(df["Next_Is_Narrow"].sum())
    rvol_pass_count = int((df["RVOL"] >= rvol_threshold).sum())
    bbw_contract_count = int(df["Prior_BBW_Contracting"].sum())
    nr7_count = int(df["Is_NR7"].sum())
    weekly_bull_count = int((df["W_CPR_Bias"] == "Bullish (Higher Value)").sum())
    weekly_bear_count = int((df["W_CPR_Bias"] == "Bearish (Lower Value)").sum())
    weekly_inside_count = int((df["W_CPR_Bias"] == "Breakout (Inside Value)").sum())
    mtf_bull_count = int(df["MTF_Bullish"].sum()) if "MTF_Bullish" in df.columns else 0
    mtf_bear_count = int(df["MTF_Bearish"].sum()) if "MTF_Bearish" in df.columns else 0
    mtf_1h_breakout_count = int((df["MTF_1H_Status"] == "🟢 Breakout > PDH").sum()) if "MTF_1H_Status" in df.columns else 0
    mtf_1h_breakdown_count = int((df["MTF_1H_Status"] == "🔴 Breakdown < PDL").sum()) if "MTF_1H_Status" in df.columns else 0
    h1_breakout_bull_count = int(df["Is_1H_Breakout_Bull"].sum()) if "Is_1H_Breakout_Bull" in df.columns else 0
    h1_breakout_bear_count = int(df["Is_1H_Breakout_Bear"].sum()) if "Is_1H_Breakout_Bear" in df.columns else 0
    h1_break_in_count = int(df["Is_1H_Break_In"].sum()) if "Is_1H_Break_In" in df.columns else 0

    meta = {
        "ref_date": base_meta.get("ref_date", "N/A"),
        "live_date": base_meta.get("live_date", "N/A"),
        "total_scanned": len(df),
        "bullish_count": len(bullish_df),
        "bearish_count": len(bearish_df),
        "squeezing_count": squeezing_count,
        "narrow_cpr_count": narrow_cpr_count,
        "next_narrow_cpr_count": next_narrow_cpr_count,
        "rvol_pass_count": rvol_pass_count,
        "bbw_contract_count": bbw_contract_count,
        "nr7_count": nr7_count,
        "weekly_bull_count": weekly_bull_count,
        "weekly_bear_count": weekly_bear_count,
        "weekly_inside_count": weekly_inside_count,
        "mtf_bull_count": mtf_bull_count,
        "mtf_bear_count": mtf_bear_count,
        "mtf_1h_breakout_count": mtf_1h_breakout_count,
        "mtf_1h_breakdown_count": mtf_1h_breakdown_count,
        "h1_breakout_bull_count": h1_breakout_bull_count,
        "h1_breakout_bear_count": h1_breakout_bear_count,
        "h1_break_in_count": h1_break_in_count,
        "ignore_52w": ignore_52w,
        "h1_close_filter": h1_close_filter,
        "only_narrow_cpr": only_narrow_cpr,
        "cpr_filter_active": enable_cpr_filter,
        "camarilla_filter_mode": camarilla_filter,
        "bollinger_filter_mode": bollinger_filter,
        "enable_rvol": enable_rvol,
        "rvol_threshold": rvol_threshold,
        "enable_bbw_squeeze": enable_bbw_squeeze,
        "enable_weekly_filter": enable_weekly_filter,
        "enable_nr7_filter": enable_nr7_filter,
        "enable_mtf_filter": enable_mtf_filter,
        "mtf_mode": mtf_mode
    }

    return all_df, bullish_df, bearish_df, meta


def fetch_and_screen(
    symbols: List[str], 
    timeframe: str = "5m", 
    high_proximity_pct: float = 0.95,
    low_proximity_pct: float = 1.05,
    ignore_52w: bool = False,
    h1_close_filter: str = "All",
    enable_cpr_filter: bool = True,
    only_narrow_cpr: bool = False,
    narrow_cpr_max_pct: float = 0.35,
    camarilla_filter: str = "None",
    bollinger_filter: str = "None",
    enable_rvol: bool = False,
    rvol_threshold: float = 1.5,
    enable_bbw_squeeze: bool = False,
    enable_weekly_filter: bool = False,
    enable_nr7_filter: bool = False,
    enable_mtf_filter: bool = False,
    mtf_mode: str = "Any / Both",
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Convenience wrapper that runs fetch_universe_data followed by apply_screener_filters.
    Kept for complete backward compatibility.
    """
    df, base_meta = fetch_universe_data(symbols, timeframe=timeframe)
    if df.empty:
        return df, df, df, base_meta
    return apply_screener_filters(
        df,
        base_meta,
        high_proximity_pct=high_proximity_pct,
        low_proximity_pct=low_proximity_pct,
        ignore_52w=ignore_52w,
        h1_close_filter=h1_close_filter,
        enable_cpr_filter=enable_cpr_filter,
        only_narrow_cpr=only_narrow_cpr,
        narrow_cpr_max_pct=narrow_cpr_max_pct,
        camarilla_filter=camarilla_filter,
        bollinger_filter=bollinger_filter,
        enable_rvol=enable_rvol,
        rvol_threshold=rvol_threshold,
        enable_bbw_squeeze=enable_bbw_squeeze,
        enable_weekly_filter=enable_weekly_filter,
        enable_nr7_filter=enable_nr7_filter,
        enable_mtf_filter=enable_mtf_filter,
        mtf_mode=mtf_mode,
        **kwargs
    )


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

    # Tomorrow's CPR & Camarilla Projection
    curr_h = float(curr_df["High"].max())
    curr_l = float(curr_df["Low"].min())
    curr_c = float(curr_df["Close"].iloc[-1])
    next_p = (curr_h + curr_l + curr_c) / 3.0
    next_bc = (curr_h + curr_l) / 2.0
    next_tc = (2.0 * next_p) - next_bc
    next_w = abs(next_tc - next_bc)
    next_w_pct = (next_w / next_p * 100.0) if next_p > 0 else 0.0
    next_type = "🎯 Ultra Narrow" if next_w_pct <= 0.15 else ("⚡ Narrow" if next_w_pct <= 0.35 else ("Average" if next_w_pct <= 0.70 else "Wide"))
    next_cpr = {
        "top": round(max(next_tc, next_bc), 2),
        "pivot": round(next_p, 2),
        "bottom": round(min(next_tc, next_bc), 2),
        "width": round(next_w, 2),
        "width_pct": round(next_w_pct, 3),
        "type": next_type
    }
    next_cam_rng = curr_h - curr_l
    next_cam = {
        "h4": round(curr_c + (next_cam_rng * 1.1 / 2.0), 2),
        "h3": round(curr_c + (next_cam_rng * 1.1 / 4.0), 2),
        "l3": round(curr_c - (next_cam_rng * 1.1 / 4.0), 2),
        "l4": round(curr_c - (next_cam_rng * 1.1 / 2.0), 2)
    }

    # Quantitative Filters & Bollinger Bands for daily
    quant = {}
    try:
        daily_df = yf.download(ticker, period="1y", interval="1d", progress=False)
        bb = calculate_bollinger_bands(daily_df)
        quant = get_quant_metrics_for_stock(daily_df, curr_day_vol=float(curr_df["Volume"].sum()))
    except Exception:
        bb = {}

    # MTF Breakout Algorithm (PDH, PDL, 1H Close, 15m VWAP)
    mtf = calculate_mtf_breakout(df, pdh_fallback=prev_h, pdl_fallback=prev_l)

    return {
        "prev_df": prev_df,
        "curr_df": curr_df,
        "vp": vp,
        "cpr": cpr,
        "cam": cam,
        "next_cpr": next_cpr,
        "next_cam": next_cam,
        "bb": bb,
        "quant": quant,
        "mtf": mtf,
        "bins": bins,
        "vols": vols,
        "prev_date": str(prev_date),
        "curr_date": str(curr_date)
    }
