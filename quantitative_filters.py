"""
Quantitative Filters: Volume, Volatility, Weekly CPR & Weekly Value Area
Implements Feature Implementation Guide (Volume & Volatility Quantitative Filters for F&O Breakout Screener)
Pages 1 - 6 Specification
"""

import pandas as pd
import numpy as np


# ----------------- 1. Relative Volume (RVOL) -----------------
def calculate_rvol(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculates Relative Volume (RVOL) using a 20-period volume SMA.
    Requires a 'Volume' column in the DataFrame.
    """
    df = df.copy()
    if "Volume" not in df.columns or df["Volume"].dropna().empty:
        df["Volume_SMA"] = np.nan
        df["RVOL"] = 1.0
        return df

    # Calculate the 20-period average volume
    df['Volume_SMA'] = df['Volume'].rolling(window=period).mean()

    # Calculate RVOL ratio
    df['RVOL'] = df['Volume'] / df['Volume_SMA']
    df['RVOL'] = df['RVOL'].fillna(1.0)
    return df


# ----------------- 2. Bollinger Bandwidth (BBW) Contraction -----------------
def calculate_bbw_contraction(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """
    Calculates BBW and its 20-period SMA to determine contraction state.
    Requires a 'Close' column in the DataFrame.
    """
    df = df.copy()
    if "Close" not in df.columns or df["Close"].dropna().empty:
        df['BB_SMA'] = np.nan
        df['BB_STD'] = np.nan
        df['BB_Upper'] = np.nan
        df['BB_Lower'] = np.nan
        df['BBW'] = np.nan
        df['BBW_SMA'] = np.nan
        df['Is_BBW_Contracting'] = False
        return df

    # Base Bollinger Band Calculations
    df['BB_SMA'] = df['Close'].rolling(window=period).mean()
    df['BB_STD'] = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_SMA'] + (df['BB_STD'] * std_dev)
    df['BB_Lower'] = df['BB_SMA'] - (df['BB_STD'] * std_dev)

    # Normalize Bollinger Bandwidth as a percentage
    df['BBW'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_SMA']) * 100
    df['BBW_SMA'] = df['BBW'].rolling(window=period).mean()

    # Boolean flag: True if current bandwidth is tighter than its recent average
    df['Is_BBW_Contracting'] = df['BBW'] < df['BBW_SMA']
    return df


# ----------------- 5. Open Interest (OI) Buildup & NR7 -----------------
def calculate_oi_buildup(df: pd.DataFrame, oi_threshold: float = 0.05) -> pd.DataFrame:
    """
    Calculates Open Interest percentage change and categorizes buildup.
    Requires 'Close' and standard OHLC columns.
    """
    df = df.copy()
    if "High" not in df.columns or "Low" not in df.columns or "Close" not in df.columns:
        df['Daily_Range'] = 0.0
        df['Min_7_Day_Range'] = 0.0
        df['Is_NR7'] = False
        df['OI_Buildup'] = "N/A"
        return df

    # Calculate percentage change in Open Interest and Price
    df['Price_Pct_Change'] = df['Close'].pct_change()

    # Flag as True if today's range is the minimum of the last 7 days (NR7)
    df['Daily_Range'] = df['High'] - df['Low']
    df['Min_7_Day_Range'] = df['Daily_Range'].rolling(window=7).min()
    df['Is_NR7'] = df['Daily_Range'] <= (df['Min_7_Day_Range'] * 1.001)

    # Check for Open_Interest column
    if 'Open_Interest' in df.columns and df['Open_Interest'].dropna().count() >= 5:
        df['OI_Pct_Change'] = df['Open_Interest'].pct_change()
        
        is_long_buildup = (df['Price_Pct_Change'] > 0) & (df['OI_Pct_Change'] >= oi_threshold)
        is_short_buildup = (df['Price_Pct_Change'] < 0) & (df['OI_Pct_Change'] >= oi_threshold)
        is_short_covering = (df['Price_Pct_Change'] > 0) & (df['OI_Pct_Change'] <= -oi_threshold)
        is_long_unwinding = (df['Price_Pct_Change'] < 0) & (df['OI_Pct_Change'] <= -oi_threshold)

        conditions = [is_long_buildup, is_short_buildup, is_short_covering, is_long_unwinding]
        choices = ["Long Buildup", "Short Buildup", "Short Covering", "Long Unwinding"]
        df['OI_Buildup'] = np.select(conditions, choices, default="Neutral")
    else:
        df['OI_Pct_Change'] = np.nan
        df['OI_Buildup'] = "N/A"

    return df


# ----------------- 6. Weekly CPR (WCPR) & Value Relationship Logic -----------------
def calculate_weekly_cpr_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Weekly CPR and determines the CPR value relationship (Bias)
    Requires a DatetimeIndex and standard OHLC columns.
    """
    if df.empty or len(df) < 10:
        df['W_Top_CPR'] = np.nan
        df['W_Bottom_CPR'] = np.nan
        df['W_Pivot'] = np.nan
        df['W_CPR_Bias'] = 'Overlapping/Neutral'
        return df

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            df['W_Top_CPR'] = np.nan
            df['W_Bottom_CPR'] = np.nan
            df['W_Pivot'] = np.nan
            df['W_CPR_Bias'] = 'Overlapping/Neutral'
            return df

    # Resample daily data to weekly to get previous week's H, L, C
    weekly_df = df.resample('W-FRI').agg({
        'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    if len(weekly_df) < 2:
        df['W_Top_CPR'] = np.nan
        df['W_Bottom_CPR'] = np.nan
        df['W_Pivot'] = np.nan
        df['W_CPR_Bias'] = 'Overlapping/Neutral'
        return df

    # Shift to use *previous* week's data for *current* week's CPR
    weekly_df['Prev_W_High'] = weekly_df['High'].shift(1)
    weekly_df['Prev_W_Low'] = weekly_df['Low'].shift(1)
    weekly_df['Prev_W_Close'] = weekly_df['Close'].shift(1)

    # Calculate Weekly CPR levels
    weekly_df['W_Pivot'] = (weekly_df['Prev_W_High'] + weekly_df['Prev_W_Low'] + weekly_df['Prev_W_Close']) / 3
    weekly_df['W_BC'] = (weekly_df['Prev_W_High'] + weekly_df['Prev_W_Low']) / 2
    weekly_df['W_TC'] = (weekly_df['W_Pivot'] - weekly_df['W_BC']) + weekly_df['W_Pivot']

    # Ensure TC is higher and BC is lower
    weekly_df['W_Top_CPR'] = np.where(weekly_df['W_TC'] > weekly_df['W_BC'], weekly_df['W_TC'], weekly_df['W_BC'])
    weekly_df['W_Bottom_CPR'] = np.where(weekly_df['W_TC'] < weekly_df['W_BC'], weekly_df['W_TC'], weekly_df['W_BC'])

    # Determine CPR Relationship (Bias)
    weekly_df['Prev_W_Top_CPR'] = weekly_df['W_Top_CPR'].shift(1)
    weekly_df['Prev_W_Bottom_CPR'] = weekly_df['W_Bottom_CPR'].shift(1)

    conditions = [
        (weekly_df['W_Bottom_CPR'] > weekly_df['Prev_W_Top_CPR']),
        (weekly_df['W_Top_CPR'] < weekly_df['Prev_W_Bottom_CPR']),
        ((weekly_df['W_Top_CPR'] < weekly_df['Prev_W_Top_CPR']) & (weekly_df['W_Bottom_CPR'] > weekly_df['Prev_W_Bottom_CPR']))
    ]
    choices = ['Bullish (Higher Value)', 'Bearish (Lower Value)', 'Breakout (Inside Value)']
    weekly_df['W_CPR_Bias'] = np.select(conditions, choices, default='Overlapping/Neutral')

    # Merge and forward-fill back to daily timeframe
    df = df.merge(weekly_df[['W_Top_CPR', 'W_Bottom_CPR', 'W_Pivot', 'W_CPR_Bias']], left_index=True, right_index=True, how='left')
    df[['W_Top_CPR', 'W_Bottom_CPR', 'W_Pivot', 'W_CPR_Bias']] = df[['W_Top_CPR', 'W_Bottom_CPR', 'W_Pivot', 'W_CPR_Bias']].ffill()
    return df


# ----------------- 7. Weekly Value Area (WVA) Logic -----------------
def calculate_weekly_value_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximates Weekly Value Area (VAH/VAL) based on weekly ranges.
    """
    if df.empty or len(df) < 10:
        df['W_VAH'] = np.nan
        df['W_VAL'] = np.nan
        df['W_POC'] = np.nan
        return df

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            df['W_VAH'] = np.nan
            df['W_VAL'] = np.nan
            df['W_POC'] = np.nan
            return df

    weekly_df = df.resample('W-FRI').agg({
        'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    if len(weekly_df) < 2:
        df['W_VAH'] = np.nan
        df['W_VAL'] = np.nan
        df['W_POC'] = np.nan
        return df

    weekly_df['Prev_W_High'] = weekly_df['High'].shift(1)
    weekly_df['Prev_W_Low'] = weekly_df['Low'].shift(1)
    weekly_df['Prev_W_Close'] = weekly_df['Close'].shift(1)

    # Approximate Value Area (70% of the core trading range around POC)
    weekly_df['W_POC'] = (weekly_df['Prev_W_High'] + weekly_df['Prev_W_Low'] + weekly_df['Prev_W_Close']) / 3
    weekly_df['W_Range'] = weekly_df['Prev_W_High'] - weekly_df['Prev_W_Low']
    weekly_df['W_VAH'] = weekly_df['W_POC'] + (weekly_df['W_Range'] * 0.35)
    weekly_df['W_VAL'] = weekly_df['W_POC'] - (weekly_df['W_Range'] * 0.35)

    # Merge and forward-fill back to daily timeframe
    df = df.merge(weekly_df[['W_VAH', 'W_VAL', 'W_POC']], left_index=True, right_index=True, how='left')
    df[['W_VAH', 'W_VAL', 'W_POC']] = df[['W_VAH', 'W_VAL', 'W_POC']].ffill()
    return df


# ----------------- Combined Helper: Single Stock Quant Metrics -----------------
def get_quant_metrics_for_stock(daily_df: pd.DataFrame, curr_day_vol: float = None, curr_day_range: float = None, period: int = 20) -> dict:
    """
    Extracts all quantitative metrics (RVOL, BBW Contraction, NR7, Weekly CPR & Weekly Value Area)
    for a single stock from its daily history.
    """
    if daily_df.empty or len(daily_df) < period:
        return {
            "rvol": 1.0,
            "volume_sma": 0.0,
            "bbw": 0.0,
            "bbw_sma": 0.0,
            "is_contracting": False,
            "prior_is_contracting": False,
            "is_nr7": False,
            "prior_is_nr7": False,
            "daily_range": 0.0,
            "min_7d_range": 0.0,
            "oi_buildup": "N/A",
            "w_pivot": 0.0,
            "w_top_cpr": 0.0,
            "w_bottom_cpr": 0.0,
            "w_cpr_bias": "Overlapping/Neutral",
            "w_vah": 0.0,
            "w_val": 0.0,
            "w_poc": 0.0
        }

    df_calc = daily_df.copy()
    if curr_day_vol is not None and curr_day_vol > 0 and len(df_calc) > 0 and "Volume" in df_calc.columns:
        vol_col_idx = df_calc.columns.get_loc("Volume")
        last_vol = df_calc.iloc[-1, vol_col_idx]
        df_calc.iloc[-1, vol_col_idx] = max(last_vol, curr_day_vol)

    df_calc = calculate_rvol(df_calc, period=period)
    df_calc = calculate_bbw_contraction(df_calc, period=period, std_dev=2)
    df_calc = calculate_oi_buildup(df_calc)
    df_calc = calculate_weekly_cpr_bias(df_calc)
    df_calc = calculate_weekly_value_area(df_calc)

    rvol = float(df_calc["RVOL"].iloc[-1]) if "RVOL" in df_calc.columns and not np.isnan(df_calc["RVOL"].iloc[-1]) else 1.0
    vol_sma = float(df_calc["Volume_SMA"].iloc[-1]) if "Volume_SMA" in df_calc.columns and not np.isnan(df_calc["Volume_SMA"].iloc[-1]) else 0.0
    bbw = float(df_calc["BBW"].iloc[-1]) if "BBW" in df_calc.columns and not np.isnan(df_calc["BBW"].iloc[-1]) else 0.0
    bbw_sma = float(df_calc["BBW_SMA"].iloc[-1]) if "BBW_SMA" in df_calc.columns and not np.isnan(df_calc["BBW_SMA"].iloc[-1]) else 0.0
    is_contracting = bool(df_calc["Is_BBW_Contracting"].iloc[-1]) if "Is_BBW_Contracting" in df_calc.columns else False

    # Shift(1) check: contraction prior to current breakout day
    prior_contracting = bool(df_calc["Is_BBW_Contracting"].shift(1).fillna(False).iloc[-1]) if "Is_BBW_Contracting" in df_calc.columns else False

    # NR7 metrics
    is_nr7 = bool(df_calc["Is_NR7"].iloc[-1]) if "Is_NR7" in df_calc.columns else False
    prior_is_nr7 = bool(df_calc["Is_NR7"].shift(1).fillna(False).iloc[-1]) if "Is_NR7" in df_calc.columns else False
    daily_range = float(df_calc["Daily_Range"].iloc[-1]) if "Daily_Range" in df_calc.columns and not np.isnan(df_calc["Daily_Range"].iloc[-1]) else 0.0
    min_7d_range = float(df_calc["Min_7_Day_Range"].iloc[-1]) if "Min_7_Day_Range" in df_calc.columns and not np.isnan(df_calc["Min_7_Day_Range"].iloc[-1]) else 0.0
    oi_buildup = str(df_calc["OI_Buildup"].iloc[-1]) if "OI_Buildup" in df_calc.columns else "N/A"

    # Weekly CPR & Value Area metrics
    w_pivot = float(df_calc["W_Pivot"].iloc[-1]) if "W_Pivot" in df_calc.columns and not np.isnan(df_calc["W_Pivot"].iloc[-1]) else 0.0
    w_top_cpr = float(df_calc["W_Top_CPR"].iloc[-1]) if "W_Top_CPR" in df_calc.columns and not np.isnan(df_calc["W_Top_CPR"].iloc[-1]) else 0.0
    w_bottom_cpr = float(df_calc["W_Bottom_CPR"].iloc[-1]) if "W_Bottom_CPR" in df_calc.columns and not np.isnan(df_calc["W_Bottom_CPR"].iloc[-1]) else 0.0
    w_cpr_bias = str(df_calc["W_CPR_Bias"].iloc[-1]) if "W_CPR_Bias" in df_calc.columns and not pd.isna(df_calc["W_CPR_Bias"].iloc[-1]) else "Overlapping/Neutral"
    w_vah = float(df_calc["W_VAH"].iloc[-1]) if "W_VAH" in df_calc.columns and not np.isnan(df_calc["W_VAH"].iloc[-1]) else 0.0
    w_val = float(df_calc["W_VAL"].iloc[-1]) if "W_VAL" in df_calc.columns and not np.isnan(df_calc["W_VAL"].iloc[-1]) else 0.0
    w_poc = float(df_calc["W_POC"].iloc[-1]) if "W_POC" in df_calc.columns and not np.isnan(df_calc["W_POC"].iloc[-1]) else 0.0

    return {
        "rvol": round(rvol, 2),
        "volume_sma": round(vol_sma, 0),
        "bbw": round(bbw, 2),
        "bbw_sma": round(bbw_sma, 2),
        "is_contracting": is_contracting,
        "prior_is_contracting": prior_contracting,
        "is_nr7": is_nr7,
        "prior_is_nr7": prior_is_nr7,
        "daily_range": round(daily_range, 2),
        "min_7d_range": round(min_7d_range, 2),
        "oi_buildup": oi_buildup,
        "w_pivot": round(w_pivot, 2),
        "w_top_cpr": round(w_top_cpr, 2),
        "w_bottom_cpr": round(w_bottom_cpr, 2),
        "w_cpr_bias": w_cpr_bias,
        "w_vah": round(w_vah, 2),
        "w_val": round(w_val, 2),
        "w_poc": round(w_poc, 2)
    }
