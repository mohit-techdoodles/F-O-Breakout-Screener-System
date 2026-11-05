"""
Volume Profile and Value Area (VAH, VAL, POC) Calculation Engine
Implements the CME / Steidlmayer 70% Value Area Algorithm
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def calculate_volume_profile(
    df_candles: pd.DataFrame, 
    num_bins: int = 70, 
    value_area_pct: float = 0.70
) -> Dict[str, float]:
    """
    Calculates Volume Profile metrics:
    - PD_POC (Point of Control): Price with the highest traded volume
    - PD_VAH (Value Area High): Upper price bound enclosing 70% of volume
    - PD_VAL (Value Area Low): Lower price bound enclosing 70% of volume
    - Total_Volume: Total volume of the session
    
    Parameters:
    - df_candles: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    - num_bins: Number of price discrete bins across the day's range (default: 70)
    - value_area_pct: Fraction of volume for Value Area (default: 0.70)
    """
    if df_candles.empty or len(df_candles) < 2:
        return {
            "PD_POC": np.nan,
            "PD_VAH": np.nan,
            "PD_VAL": np.nan,
            "Total_Volume": 0.0,
            "Day_High": np.nan,
            "Day_Low": np.nan,
            "Day_Close": np.nan,
        }

    # Ensure required columns exist
    cols = {c.lower(): c for c in df_candles.columns}
    high_col = cols.get("high", "High")
    low_col = cols.get("low", "Low")
    close_col = cols.get("close", "Close")
    vol_col = cols.get("volume", "Volume")

    day_high = float(df_candles[high_col].max())
    day_low = float(df_candles[low_col].min())
    day_close = float(df_candles[close_col].iloc[-1])
    total_volume = float(df_candles[vol_col].sum())

    if day_high <= day_low or total_volume <= 0:
        return {
            "PD_POC": day_close,
            "PD_VAH": day_high,
            "PD_VAL": day_low,
            "Total_Volume": total_volume,
            "Day_High": day_high,
            "Day_Low": day_low,
            "Day_Close": day_close,
        }

    # Create uniform price bins across the day's range
    bin_edges = np.linspace(day_low, day_high, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_volumes = np.zeros(num_bins, dtype=np.float64)

    # Distribute each candle's volume uniformly across the bins it spans
    highs = df_candles[high_col].values
    lows = df_candles[low_col].values
    vols = df_candles[vol_col].values

    for h, l, v in zip(highs, lows, vols):
        if v <= 0:
            continue
        if h <= l:
            # All volume goes to the single bin containing the price
            idx = np.searchsorted(bin_edges, l, side="right") - 1
            idx = np.clip(idx, 0, num_bins - 1)
            bin_volumes[idx] += v
        else:
            # Overlap distribution
            overlap_min = np.maximum(l, bin_edges[:-1])
            overlap_max = np.minimum(h, bin_edges[1:])
            overlap = np.maximum(0.0, overlap_max - overlap_min)
            fraction = overlap / (h - l)
            bin_volumes += v * fraction

    # Find POC (Point of Control)
    poc_idx = int(np.argmax(bin_volumes))
    poc_price = float(bin_centers[poc_idx])

    # Calculate 70% Value Area
    target_volume = value_area_pct * total_volume
    included_indices = {poc_idx}
    accumulated_volume = bin_volumes[poc_idx]

    up_idx = poc_idx + 1
    down_idx = poc_idx - 1

    while accumulated_volume < target_volume and (up_idx < num_bins or down_idx >= 0):
        vol_up = bin_volumes[up_idx] if up_idx < num_bins else -1.0
        vol_down = bin_volumes[down_idx] if down_idx >= 0 else -1.0

        if vol_up > vol_down:
            accumulated_volume += vol_up
            included_indices.add(up_idx)
            up_idx += 1
        elif vol_down > vol_up:
            accumulated_volume += vol_down
            included_indices.add(down_idx)
            down_idx -= 1
        else:
            # If equal, expand both (or whichever is valid)
            if up_idx < num_bins:
                accumulated_volume += vol_up
                included_indices.add(up_idx)
                up_idx += 1
            if down_idx >= 0 and accumulated_volume < target_volume:
                accumulated_volume += vol_down
                included_indices.add(down_idx)
                down_idx -= 1

    vah_idx = max(included_indices)
    val_idx = min(included_indices)

    # VAH is upper edge of highest bin, VAL is lower edge of lowest bin
    vah_price = float(bin_edges[vah_idx + 1])
    val_price = float(bin_edges[val_idx])

    return {
        "PD_POC": round(poc_price, 2),
        "PD_VAH": round(vah_price, 2),
        "PD_VAL": round(val_price, 2),
        "Total_Volume": round(total_volume, 0),
        "Day_High": round(day_high, 2),
        "Day_Low": round(day_low, 2),
        "Day_Close": round(day_close, 2),
    }


def get_profile_histogram(
    df_candles: pd.DataFrame, 
    num_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (price_bins, volumes) for visual charting
    """
    if df_candles.empty:
        return np.array([]), np.array([])
    
    cols = {c.lower(): c for c in df_candles.columns}
    day_high = float(df_candles[cols.get("high", "High")].max())
    day_low = float(df_candles[cols.get("low", "Low")].min())
    
    if day_high <= day_low:
        return np.array([day_low]), np.array([df_candles[cols.get("volume", "Volume")].sum()])

    bin_edges = np.linspace(day_low, day_high, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_volumes = np.zeros(num_bins, dtype=np.float64)

    highs = df_candles[cols.get("high", "High")].values
    lows = df_candles[cols.get("low", "Low")].values
    vols = df_candles[cols.get("volume", "Volume")].values

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

    return bin_centers, bin_volumes

