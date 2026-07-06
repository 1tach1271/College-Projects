"""
cleaning.py — Module 2: Data Cleaning & Validation
====================================================
Handles missing values, applies adjustment factors, removes outliers,
and filters stocks with insufficient history.
"""

import logging

import numpy as np
import pandas as pd

import config
from utils import timed, summarize_df

logger = logging.getLogger(__name__)


@timed
def apply_adjustment_factor(df):
    """
    Apply AdjustmentFactor to OHLC prices.
    This handles stock splits, reverse splits, etc.
    """
    adj = df["AdjustmentFactor"]
    for col in config.PRICE_COLUMNS:
        df[col] = df[col] * adj
    logger.info("Applied AdjustmentFactor to OHLC prices")
    return df


@timed
def handle_missing_values(df):
    """
    Handle missing OHLCV values:
    1. Forward-fill within each stock (time-series continuation)
    2. Drop any remaining rows with missing Close or Volume
    """
    n_before = len(df)
    null_before = df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum().sum()

    # Forward-fill within each stock group
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[ohlcv_cols] = df.groupby("SecuritiesCode")[ohlcv_cols].ffill()

    # Drop rows where Close or Volume is still NaN
    df = df.dropna(subset=["Close", "Volume"]).reset_index(drop=True)

    null_after = df[ohlcv_cols].isnull().sum().sum()
    n_after = len(df)
    logger.info(f"Missing values: {null_before:,} → {null_after:,} "
                f"(dropped {n_before - n_after:,} rows)")
    return df


@timed
def filter_short_histories(df, min_days=None):
    """Remove stocks with fewer than min_days trading days."""
    min_days = min_days or config.MIN_TRADING_DAYS
    counts = df.groupby("SecuritiesCode").size()
    valid_stocks = counts[counts >= min_days].index
    n_before = df["SecuritiesCode"].nunique()
    df = df[df["SecuritiesCode"].isin(valid_stocks)].reset_index(drop=True)
    n_after = df["SecuritiesCode"].nunique()
    logger.info(f"Stock filter (>={min_days} days): {n_before:,} → {n_after:,} stocks")
    return df


@timed
def remove_outlier_returns(df, sigma=None):
    """
    Remove rows with extreme daily returns (beyond sigma standard deviations).
    Computes simple return, flags outliers, and removes them.
    """
    sigma = sigma or config.OUTLIER_SIGMA

    # Compute simple return for outlier detection
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    df["_ret"] = df.groupby("SecuritiesCode")["Close"].pct_change()

    # Compute mean and std of returns
    ret_mean = df["_ret"].mean()
    ret_std = df["_ret"].std()

    # Flag outliers
    is_outlier = (df["_ret"].abs() - ret_mean).abs() > (sigma * ret_std)
    # Don't remove first row per stock (NaN return)
    is_outlier = is_outlier & df["_ret"].notna()

    n_outliers = is_outlier.sum()
    if n_outliers > 0:
        df = df[~is_outlier].reset_index(drop=True)
        logger.info(f"Removed {n_outliers:,} outlier return rows (>{sigma}σ)")
    else:
        logger.info("No outlier returns detected")

    df = df.drop(columns=["_ret"])
    return df


@timed
def validate_price_consistency(df):
    """
    Validate and fix price consistency:
    - High >= Low
    - Close within [Low, High]
    """
    inconsistent_hl = (df["High"] < df["Low"])
    n_hl = inconsistent_hl.sum()
    if n_hl > 0:
        # Swap High and Low where inconsistent
        mask = inconsistent_hl
        df.loc[mask, ["High", "Low"]] = df.loc[mask, ["Low", "High"]].values
        logger.warning(f"Fixed {n_hl:,} rows with High < Low")

    # Check Close is within [Low, High]
    close_outside = (df["Close"] < df["Low"]) | (df["Close"] > df["High"])
    n_co = close_outside.sum()
    if n_co > 0:
        # Clip Close to [Low, High]
        df.loc[close_outside, "Close"] = df.loc[close_outside, "Close"].clip(
            lower=df.loc[close_outside, "Low"],
            upper=df.loc[close_outside, "High"]
        )
        logger.warning(f"Clipped {n_co:,} rows where Close was outside [Low, High]")

    return df


@timed
def flag_supervision_stocks(df):
    """Log stocks under supervision flag."""
    if "SupervisionFlag" in df.columns:
        # Convert to boolean if string
        if df["SupervisionFlag"].dtype == object:
            df["SupervisionFlag"] = df["SupervisionFlag"].map(
                {"True": True, "False": False, True: True, False: False}
            ).fillna(False)

        n_supervised = df[df["SupervisionFlag"] == True]["SecuritiesCode"].nunique()
        logger.info(f"Stocks under supervision: {n_supervised}")
    return df


@timed
def run_cleaning(df):
    """
    Full cleaning pipeline:
    1. Apply adjustment factors
    2. Handle missing values
    3. Filter short histories
    4. Remove extreme outlier returns
    5. Validate price consistency
    6. Flag supervision stocks
    """
    logger.info("═══ Starting Data Cleaning ═══")

    df = apply_adjustment_factor(df)
    df = handle_missing_values(df)
    df = filter_short_histories(df)
    df = remove_outlier_returns(df)
    df = validate_price_consistency(df)
    df = flag_supervision_stocks(df)

    summarize_df(df, "Cleaned Stock Prices")

    return df


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    from ingestion import run_ingestion
    prices_df, _ = run_ingestion()
    cleaned_df = run_cleaning(prices_df)
    print(cleaned_df.head())
