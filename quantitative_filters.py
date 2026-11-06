"""
Quantitative Filters: Relative Volume (RVOL) and Bollinger Bandwidth (BBW) Contraction
Implements Feature Implementation Guide (Volume & Volatility Quantitative Filters for F&O Breakout Screener)
"""

import pandas as pd
import numpy as np


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


def get_quant_metrics_for_stock(daily_df: pd.DataFrame, curr_day_vol: float = None, period: int = 20) -> dict:
    """
    Extracts RVOL and BBW contraction status for a single stock from its daily history.
    Uses shift(1) on Is_BBW_Contracting to verify prior-day compression.
    """
    if daily_df.empty or len(daily_df) < period:
        return {
            "rvol": 1.0,
            "volume_sma": 0.0,
            "bbw": 0.0,
            "bbw_sma": 0.0,
            "is_contracting": False,
            "prior_is_contracting": False
        }

    df_calc = daily_df.copy()
    if curr_day_vol is not None and curr_day_vol > 0 and len(df_calc) > 0:
        # Use current session's volume if provided
        vol_col_idx = df_calc.columns.get_loc("Volume")
        last_vol = df_calc.iloc[-1, vol_col_idx]
        df_calc.iloc[-1, vol_col_idx] = max(last_vol, curr_day_vol)

    df_calc = calculate_rvol(df_calc, period=period)
    df_calc = calculate_bbw_contraction(df_calc, period=period, std_dev=2)

    rvol = float(df_calc["RVOL"].iloc[-1]) if not np.isnan(df_calc["RVOL"].iloc[-1]) else 1.0
    vol_sma = float(df_calc["Volume_SMA"].iloc[-1]) if not np.isnan(df_calc["Volume_SMA"].iloc[-1]) else 0.0
    bbw = float(df_calc["BBW"].iloc[-1]) if not np.isnan(df_calc["BBW"].iloc[-1]) else 0.0
    bbw_sma = float(df_calc["BBW_SMA"].iloc[-1]) if not np.isnan(df_calc["BBW_SMA"].iloc[-1]) else 0.0
    is_contracting = bool(df_calc["Is_BBW_Contracting"].iloc[-1])

    # Critically: use shift(1) on BBW condition to verify contraction existed prior to current breakout day
    prior_contracting = bool(df_calc["Is_BBW_Contracting"].shift(1).fillna(False).iloc[-1])

    return {
        "rvol": round(rvol, 2),
        "volume_sma": round(vol_sma, 0),
        "bbw": round(bbw, 2),
        "bbw_sma": round(bbw_sma, 2),
        "is_contracting": is_contracting,
        "prior_is_contracting": prior_contracting
    }

