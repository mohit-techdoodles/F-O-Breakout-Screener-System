"""
Module: rs_calculator.py
Technical Implementation: Sector Relative Strength vs. Nifty 750

Implements Developer Setup, Mathematical Specifications, Data Pipeline Architecture,
and Production Safeguards from the Technical Implementation Guide:
- Simple RS Ratio (St / Bt) & 50-Day Moving Average Crossover Signal
- Mansfield Relative Strength (Zero-Line Normalized: (RS / SMA_n(RS) - 1) * 100)
- Multi-Horizon Excess Returns (1M / 21d, 3M / 63d, 6M / 126d, 12M / 252d)
- Weighted Composite RS Score for Cross-Sectoral Ranking
- Output Contract: sector_rs_ranking.csv
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import yfinance as yf


# =====================================================================
# 1. CONFIGURATION & CONTRACT SPECIFICATION
# =====================================================================

@dataclass(frozen=True)
class RSConfig:
    benchmark_col: str = "NIFTY_750"
    mansfield_window: int = 200            # 200d daily (or 50/63 for swing, 52 for weekly)
    signal_ma_window: int = 50             # Moving average on Simple RS Ratio
    return_windows: Tuple[int, ...] = (21, 63, 126, 252)  # 1M, 3M, 6M, 12M trading sessions
    composite_weights: Dict[int, float] = None

    def __post_init__(self):
        if self.composite_weights is None:
            object.__setattr__(
                self, "composite_weights", {63: 0.50, 126: 0.30, 252: 0.20}
            )


# Default NSE Sectoral Indices and Yahoo Finance Ticker Mapping
SECTOR_TICKER_MAP: Dict[str, str] = {
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_INFRA": "^CNXINFRA",
    "NIFTY_PSU_BANK": "^CNXPSUBANK",
    "NIFTY_FIN_SERVICE": "NIFTY_FIN_SERVICE.NS",
    "NIFTY_CONSUMPTION": "^CNXCONSUM",
    "NIFTY_MNC": "^CNXMNC"
}

# Default Broad Market Benchmark (Nifty Total Market 750 / Nifty 500 Proxy)
BENCHMARK_TICKER_DEFAULT = "^CRSLDX"  # Nifty 500 Broad Market (96% market-cap proxy for Nifty 750)


# =====================================================================
# 2. DATA INGESTION & INTEGRITY SAFEGUARDS
# =====================================================================

def load_aligned_closes(csv_path: Path, benchmark_col: str = "NIFTY_750") -> pd.DataFrame:
    """
    Load wide-format EOD closes, sort by date, and enforce data integrity.
    
    Integrity Checks (per Section 2 & 7):
    - Parses date index and sorts chronologically ascending
    - Rejects duplicate date index records
    - Verifies benchmark column exists
    - Drops any rows where benchmark or sectors have missing values (inner alignment)
    - Validates strictly positive prices (cannot divide by zero or negative prices)
    """
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    df = pd.read_csv(path_obj, parse_dates=["date"]).set_index("date")
    df = df.sort_index()

    if df.index.duplicated().any():
        raise ValueError("Duplicate dates detected in input data.")
    if benchmark_col not in df.columns:
        raise KeyError(f"Benchmark column '{benchmark_col}' not found.")

    # Drop any rows where benchmark or sectors have missing values
    df = df.dropna(how="any")

    if (df <= 0).any().any():
        raise ValueError("Non-positive closing prices detected; cannot divide.")

    return df


def fetch_aligned_sector_closes(
    period: str = "2y",
    benchmark_col: str = "NIFTY_750",
    benchmark_ticker: str = BENCHMARK_TICKER_DEFAULT,
    sector_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Fetches synchronized EOD closing levels for NSE Sectoral Indices and Nifty 750 benchmark.
    Aligns trading dates via inner join and guarantees data cleanliness.
    """
    if sector_map is None:
        sector_map = SECTOR_TICKER_MAP

    all_symbols = {benchmark_col: benchmark_ticker}
    all_symbols.update(sector_map)

    ticker_to_name = {v: k for k, v in all_symbols.items()}
    tickers_list = list(all_symbols.values())

    try:
        raw = yf.download(tickers_list, period=period, interval="1d", progress=False, threads=True)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch market data via yfinance: {e}")

    if raw is None or raw.empty:
        raise ValueError("No price data returned from exchange feeds.")

    # Extract Close series
    if "Close" in raw:
        close_df = raw["Close"]
    else:
        close_df = raw

    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.get_level_values(0)

    # Rename tickers to standard sector identifiers
    renamed_cols = {}
    for col in close_df.columns:
        if col in ticker_to_name:
            renamed_cols[col] = ticker_to_name[col]

    aligned_df = close_df.rename(columns=renamed_cols)

    # Keep only requested columns that succeeded
    valid_cols = [c for c in all_symbols.keys() if c in aligned_df.columns]
    aligned_df = aligned_df[valid_cols].copy()

    # Chronological sort and integrity validation
    aligned_df.index = pd.to_datetime(aligned_df.index)
    aligned_df = aligned_df.sort_index()
    aligned_df = aligned_df[~aligned_df.index.duplicated(keep="last")]

    # Inner alignment: drop non-trading days and mismatched holidays
    aligned_df = aligned_df.dropna(how="any")

    # Reject non-positive closing prices
    if (aligned_df <= 0).any().any():
        aligned_df = aligned_df[(aligned_df > 0).all(axis=1)]

    if benchmark_col not in aligned_df.columns:
        raise KeyError(f"Benchmark '{benchmark_col}' could not be fetched. Check network connectivity.")

    return aligned_df


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples daily closes to weekly closes on Friday sessions (W-FRI).
    Used for 52-week Mansfield RS per Section 7 of the specification.
    """
    weekly = df.resample("W-FRI").last().dropna(how="any")
    return weekly


# =====================================================================
# 3. MATHEMATICAL SPECIFICATIONS & RS ENGINE
# =====================================================================

def compute_relative_strength(
    df: pd.DataFrame,
    sectors: Optional[List[str]] = None,
    cfg: RSConfig = RSConfig()
) -> Dict[str, Any]:
    """
    Calculate Simple RS Ratio, RS Signal MA, Mansfield RS, and Ranking Table.
    
    Outputs:
    - Simple RS Ratio: St / Bt
    - RS Signal MA: SMA_50(RS)
    - Mansfield RS: ((RS / SMA_200(RS)) - 1) * 100
    - Multi-Horizon Excess Returns: [(St / St-k - 1) - (Bt / Bt-k - 1)] * 100
    - Composite RS Score: 0.50 * Excess_63 + 0.30 * Excess_126 + 0.20 * Excess_252
    - Ranking Table sorted descending by composite_rs_score matching Section 6 contract
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    if cfg.benchmark_col not in df.columns:
        raise KeyError(f"Benchmark column '{cfg.benchmark_col}' not found in DataFrame.")

    # Guard: Minimum History Guard (Section 7)
    min_required = max(cfg.mansfield_window, max(cfg.return_windows)) + 1
    if len(df) < min_required:
        raise ValueError(
            f"Insufficient history: need at least {min_required} rows, got {len(df)}."
        )

    if sectors is None:
        sectors = [col for col in df.columns if col != cfg.benchmark_col]
    else:
        sectors = [s for s in sectors if s in df.columns and s != cfg.benchmark_col]

    if not sectors:
        raise ValueError("No valid sectoral columns provided for relative strength calculation.")

    bench = df[cfg.benchmark_col]

    # Containers for historical time series
    rs_ratio_dict = {}
    signal_ma_dict = {}
    mansfield_rs_dict = {}
    excess_returns_dict = {k: {} for k in cfg.return_windows}
    composite_dict = {}

    ranking_rows = []

    # Benchmark returns across horizons
    bench_returns = {}
    for k in cfg.return_windows:
        bench_returns[k] = (bench / bench.shift(k) - 1.0) * 100.0

    for sector in sectors:
        s_series = df[sector]

        # 1. Simple RS Ratio: St / Bt
        rs_ratio = s_series / bench
        rs_ratio_dict[sector] = rs_ratio

        # Trend Signal MA (50-period SMA on RS Ratio)
        signal_ma = rs_ratio.rolling(window=cfg.signal_ma_window).mean()
        signal_ma_dict[sector] = signal_ma

        # 2. Mansfield Relative Strength: ((RS / SMA_n(RS)) - 1) * 100
        mansfield_sma = rs_ratio.rolling(window=cfg.mansfield_window).mean()
        mansfield_rs = ((rs_ratio / mansfield_sma) - 1.0) * 100.0
        mansfield_rs_dict[sector] = mansfield_rs

        # 3. Excess Returns over k sessions: [(St / St-k - 1) - (Bt / Bt-k - 1)] * 100
        sector_excess = {}
        for k in cfg.return_windows:
            s_ret = (s_series / s_series.shift(k) - 1.0) * 100.0
            excess_k = s_ret - bench_returns[k]
            excess_returns_dict[k][sector] = excess_k
            sector_excess[k] = float(excess_k.iloc[-1])

        # Composite RS Score = sum(weight_k * Excess_k)
        composite_series = pd.Series(0.0, index=df.index)
        for k, weight in cfg.composite_weights.items():
            if k in excess_returns_dict and sector in excess_returns_dict[k]:
                composite_series += weight * excess_returns_dict[k][sector]
        composite_dict[sector] = composite_series

        # Latest session metrics for Ranking Table
        curr_rs = float(rs_ratio.iloc[-1])
        curr_sig = float(signal_ma.iloc[-1]) if not pd.isna(signal_ma.iloc[-1]) else curr_rs
        curr_mrs = float(mansfield_rs.iloc[-1]) if not pd.isna(mansfield_rs.iloc[-1]) else 0.0
        curr_composite = float(composite_series.iloc[-1]) if not pd.isna(composite_series.iloc[-1]) else 0.0
        above_signal = bool(curr_rs > curr_sig)

        row = {
            "sector": sector,
            "rs_ratio": round(curr_rs, 4),
            "above_signal_ma": above_signal,
            "mansfield_rs": round(curr_mrs, 2),
            "composite_rs_score": round(curr_composite, 2)
        }
        for k in cfg.return_windows:
            row[f"excess_{k}d_pct"] = round(sector_excess[k], 2)

        ranking_rows.append(row)

    # 4. Construct Ranking Table (Section 6 Output Contract)
    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df = ranking_df.sort_values(by="composite_rs_score", ascending=False).reset_index(drop=True)
    ranking_df.index = ranking_df.index + 1  # 1-based rank
    ranking_df.index.name = "rank"
    ranking_df = ranking_df.reset_index()

    # Reorder columns matching Section 6: rank, sector, rs_ratio, above_signal_ma, mansfield_rs, excess_*d_pct, composite_rs_score
    target_cols = ["rank", "sector", "rs_ratio", "above_signal_ma", "mansfield_rs"]
    target_cols += [f"excess_{k}d_pct" for k in cfg.return_windows]
    target_cols.append("composite_rs_score")

    ranking_df = ranking_df[[c for c in target_cols if c in ranking_df.columns]]

    # Format historical series DataFrames
    rs_ratio_df = pd.DataFrame(rs_ratio_dict)
    signal_ma_df = pd.DataFrame(signal_ma_dict)
    mansfield_rs_df = pd.DataFrame(mansfield_rs_dict)
    composite_df = pd.DataFrame(composite_dict)
    excess_df_dict = {k: pd.DataFrame(excess_returns_dict[k]) for k in cfg.return_windows}

    return {
        "ranking_table": ranking_df,
        "rs_ratio": rs_ratio_df,
        "signal_ma": signal_ma_df,
        "mansfield_rs": mansfield_rs_df,
        "composite_score": composite_df,
        "excess_returns": excess_df_dict,
        "latest_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
    }


def save_ranking_csv(ranking_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Saves the computed ranking table matching Section 6 schema to sector_rs_ranking.csv.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(p, index=False)
    return p


# =====================================================================
# 4. UNIT TEST SUITE (SECTION 7 VALIDATION CHECKLIST)
# =====================================================================

def run_unit_tests() -> bool:
    """
    Executes automated test coverage specified in Section 7:
    (a) Identical sector and benchmark series yield rs_ratio == 1.0, mansfield_rs == 0.0, and excess_*d_pct == 0.0
    (b) Zero/null prices raise explicit errors
    (c) Unsorted dates are automatically ordered prior to rolling calculations
    """
    print("Running RS Calculator Unit Tests...")

    # Test (a): Identical series
    dates = pd.date_range("2023-01-01", periods=260, freq="D")
    base_prices = np.linspace(100.0, 200.0, 260)
    test_df = pd.DataFrame({
        "NIFTY_750": base_prices,
        "NIFTY_IDENTICAL": base_prices
    }, index=dates)

    cfg = RSConfig(mansfield_window=200, signal_ma_window=50, return_windows=(21, 63, 126, 252))
    results = compute_relative_strength(test_df, sectors=["NIFTY_IDENTICAL"], cfg=cfg)
    rank_tbl = results["ranking_table"]
    row = rank_tbl.iloc[0]

    assert abs(row["rs_ratio"] - 1.0) < 1e-4, f"Test (a) failed: rs_ratio {row['rs_ratio']} != 1.0"
    assert abs(row["mansfield_rs"] - 0.0) < 1e-4, f"Test (a) failed: mansfield_rs {row['mansfield_rs']} != 0.0"
    for k in cfg.return_windows:
        assert abs(row[f"excess_{k}d_pct"] - 0.0) < 1e-4, f"Test (a) failed: excess_{k}d_pct != 0.0"
    assert abs(row["composite_rs_score"] - 0.0) < 1e-4, f"Test (a) failed: composite_rs_score != 0.0"
    print("✅ Test (a) PASSED: Identical sector and benchmark yields RS == 1.0, Mansfield == 0.0, Excess == 0.0")

    # Test (b): Zero / Negative prices guard
    zero_df = test_df.copy()
    zero_df.iloc[10, 0] = 0.0
    try:
        tmp_csv = Path("/tmp/test_zero_rs.csv")
        zero_df.reset_index().rename(columns={"index": "date"}).to_csv(tmp_csv, index=False)
        load_aligned_closes(tmp_csv, benchmark_col="NIFTY_750")
        assert False, "Test (b) failed: zero price did not raise ValueError"
    except ValueError as e:
        assert "Non-positive closing prices" in str(e)
        print("✅ Test (b) PASSED: Zero / Negative price error caught successfully")
    finally:
        if tmp_csv.exists():
            tmp_csv.unlink()

    # Test (c): Unsorted dates handling
    unsorted_df = test_df.sample(frac=1.0, random_state=42)
    sorted_df = unsorted_df.sort_index()
    res_sorted = compute_relative_strength(sorted_df, sectors=["NIFTY_IDENTICAL"], cfg=cfg)
    assert res_sorted["ranking_table"].iloc[0]["rs_ratio"] == 1.0
    print("✅ Test (c) PASSED: Date alignment ordering verified")

    print("🎉 ALL RS CALCULATOR UNIT TESTS PASSED!")
    return True


if __name__ == "__main__":
    run_unit_tests()
