"""
features.py — Module 3: GPU-Accelerated Feature Engineering
=============================================================
Creates advanced financial features: log returns, rolling volatility,
moving averages, momentum, drawdown, liquidity proxies, RSI,
cross-sectional features, and ML target variables.
"""

import logging
import warnings

import numpy as np
import pandas as pd

import config
from gpu_utils import is_gpu_available, gpu_context
from utils import timed, summarize_df

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════
# PER-ASSET FEATURES (computed within each SecuritiesCode group)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_log_returns(group):
    """Log returns: ln(Close_t / Close_{t-1})"""
    group["log_return"] = np.log(group["Close"] / group["Close"].shift(1))
    return group


def _compute_rolling_volatility(group):
    """Rolling volatility for multiple windows."""
    for w in config.ROLLING_WINDOWS:
        group[f"vol_{w}d"] = group["log_return"].rolling(w, min_periods=max(w // 2, 2)).std()
    return group


def _compute_moving_averages(group):
    """Simple moving averages and crossover signals."""
    for w in config.ROLLING_WINDOWS:
        group[f"sma_{w}d"] = group["Close"].rolling(w, min_periods=max(w // 2, 2)).mean()

    # MA crossover ratios
    if "sma_5d" in group.columns and "sma_63d" in group.columns:
        group["ma_crossover_5_63"] = group["sma_5d"] / group["sma_63d"] - 1
    if "sma_10d" in group.columns and "sma_126d" in group.columns:
        group["ma_crossover_10_126"] = group["sma_10d"] / group["sma_126d"] - 1

    return group


def _compute_momentum(group):
    """Price momentum over multiple periods."""
    for p in config.MOMENTUM_PERIODS:
        group[f"momentum_{p}d"] = group["Close"] / group["Close"].shift(p) - 1
    return group


def _compute_rsi(group):
    """Relative Strength Index (14-day)."""
    delta = group["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(config.RSI_PERIOD, min_periods=config.RSI_PERIOD).mean()
    avg_loss = loss.rolling(config.RSI_PERIOD, min_periods=config.RSI_PERIOD).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    group["rsi_14"] = 100 - (100 / (1 + rs))
    return group


def _compute_drawdown(group):
    """Current drawdown and rolling max drawdown."""
    rolling_max = group["Close"].cummax()
    group["drawdown"] = (group["Close"] - rolling_max) / (rolling_max + 1e-10)
    group["max_drawdown_63d"] = group["drawdown"].rolling(63, min_periods=10).min()
    return group


def _compute_liquidity_proxies(group):
    """Volume-based liquidity features."""
    vol_21_mean = group["Volume"].rolling(21, min_periods=5).mean()
    vol_21_std = group["Volume"].rolling(21, min_periods=5).std()

    group["volume_zscore"] = (group["Volume"] - vol_21_mean) / (vol_21_std + 1e-10)
    group["amihud_illiquidity"] = group["log_return"].abs() / (group["Volume"] + 1e-10)

    # Smooth Amihud
    group["amihud_21d"] = group["amihud_illiquidity"].rolling(21, min_periods=5).mean()
    return group


def _compute_intraday_volatility(group):
    """Parkinson estimator: ln(High/Low)"""
    group["parkinson_vol"] = np.log(group["High"] / (group["Low"] + 1e-10))
    group["parkinson_vol_21d"] = group["parkinson_vol"].rolling(21, min_periods=5).mean()
    return group


@timed
def compute_per_asset_features_gpu(df):
    """
    Stable feature engineering (CPU-optimized pandas version).

    WHY:
    - Avoids CUDA context crashes from cuDF → pandas conversion mid-pipeline
    - Uses pandas for full support of shift(), rolling(), groupby()
    - Keeps pipeline production-safe and deterministic
    """

    logger.info("Computing per-asset features (CPU mode)...")

    import pandas as pd
    import numpy as np

    # ✅ Convert ONCE at the beginning
    assert isinstance(df, pd.DataFrame), "❌ df must be pandas before feature engineering"

    # ✅ Sort for time-series correctness
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────
    # Basic Feature: Parkinson Volatility
    # ─────────────────────────────────────────────────────────────
    df["parkinson_vol"] = np.log(df["High"] / (df["Low"] + 1e-10))

    # ─────────────────────────────────────────────────────────────
    # Groupby Features
    # ─────────────────────────────────────────────────────────────
    grp = df.groupby("SecuritiesCode")

    # Log returns
    df["log_return"] = grp["Close"].transform(lambda x: np.log(x / x.shift(1)))

    # Rolling volatility
    for w in config.ROLLING_WINDOWS:
        df[f"vol_{w}d"] = grp["log_return"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).std()
        )

    # Moving averages
    for w in config.ROLLING_WINDOWS:
        df[f"sma_{w}d"] = grp["Close"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).mean()
        )

    # Momentum
    for p in config.MOMENTUM_PERIODS:
        df[f"momentum_{p}d"] = grp["Close"].transform(
            lambda x: x / x.shift(p) - 1
        )

    # Rolling Parkinson volatility
    df["parkinson_vol_21d"] = grp["parkinson_vol"].transform(
        lambda x: x.rolling(21, min_periods=5).mean()
    )

    # ─────────────────────────────────────────────────────────────
    # Volume Z-score
    # ─────────────────────────────────────────────────────────────
    vol_mean = grp["Volume"].transform(
        lambda x: x.rolling(21, min_periods=5).mean()
    )
    vol_std = grp["Volume"].transform(
        lambda x: x.rolling(21, min_periods=5).std()
    )
    df["volume_zscore"] = (df["Volume"] - vol_mean) / (vol_std + 1e-10)

    # ─────────────────────────────────────────────────────────────
    # RSI
    # ─────────────────────────────────────────────────────────────
    logger.info("Computing RSI, drawdown, and liquidity features...")

    delta = grp["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.groupby(df["SecuritiesCode"]).transform(
        lambda x: x.rolling(config.RSI_PERIOD, min_periods=config.RSI_PERIOD).mean()
    )
    avg_loss = loss.groupby(df["SecuritiesCode"]).transform(
        lambda x: x.rolling(config.RSI_PERIOD, min_periods=config.RSI_PERIOD).mean()
    )

    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ─────────────────────────────────────────────────────────────
    # Drawdown
    # ─────────────────────────────────────────────────────────────
    df["rolling_max_close"] = grp["Close"].cummax()

    df["drawdown"] = (
        (df["Close"] - df["rolling_max_close"]) /
        (df["rolling_max_close"] + 1e-10)
    )

    df["max_drawdown_63d"] = grp["drawdown"].transform(
        lambda x: x.rolling(63, min_periods=10).min()
    )

    df.drop(columns=["rolling_max_close"], inplace=True)

    # ─────────────────────────────────────────────────────────────
    # Amihud Illiquidity
    # ─────────────────────────────────────────────────────────────
    df["amihud_illiquidity"] = df["log_return"].abs() / (df["Volume"] + 1e-10)

    df["amihud_21d"] = grp["amihud_illiquidity"].transform(
        lambda x: x.rolling(21, min_periods=5).mean()
    )

    # ─────────────────────────────────────────────────────────────
    # Moving Average Crossovers
    # ─────────────────────────────────────────────────────────────
    if "sma_5d" in df.columns and "sma_63d" in df.columns:
        df["ma_crossover_5_63"] = df["sma_5d"] / df["sma_63d"] - 1

    if "sma_10d" in df.columns and "sma_126d" in df.columns:
        df["ma_crossover_10_126"] = df["sma_10d"] / df["sma_126d"] - 1

    logger.info("Feature engineering completed successfully.")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL FEATURES (per date)
# ═══════════════════════════════════════════════════════════════════════════

@timed
def compute_cross_sectional_features(df):
    """
    Cross-sectional (per-date) features:
    - Volatility percentile rank
    - Return percentile rank
    - Market breadth
    """
    logger.info("Computing cross-sectional features...")

    # Volatility rank (percentile within each date)
    if "vol_21d" in df.columns:
        df["vol_rank"] = df.groupby("Date")["vol_21d"].rank(pct=True)

    # Return rank
    if "log_return" in df.columns:
        df["return_rank"] = df.groupby("Date")["log_return"].rank(pct=True)

    # Market breadth (% of positive returns per date)
    if "log_return" in df.columns:
        breadth = df.groupby("Date")["log_return"].apply(
            lambda x: (x > 0).mean()
        ).rename("market_breadth")
        df = df.merge(breadth, on="Date", how="left")

    logger.info("Cross-sectional features computed")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TARGET VARIABLES
# ═══════════════════════════════════════════════════════════════════════════

@timed
def compute_targets(df):
    """
    Create ML target variables:
    1. regime_label: 0=Low, 1=Medium, 2=High volatility (terciles of 21d vol)
    2. vol_spike: Binary — 5-day forward vol > 2× current 21-day vol
    """
    logger.info("Computing target variables...")

    # ─── Regime Label (terciles of cross-sectional 21d vol) ───
    if "vol_21d" in df.columns:
        def safe_qcut(x):
            valid = x.dropna()
            if len(valid) < 3:
                return pd.Series(np.nan, index=x.index)
            try:
                return pd.qcut(x, q=3, labels=[0, 1, 2], duplicates="drop")
            except (ValueError, IndexError):
                return pd.Series(np.nan, index=x.index)

        df["regime_label"] = df.groupby("Date")["vol_21d"].transform(safe_qcut)
        df["regime_label"] = pd.to_numeric(df["regime_label"], errors="coerce").astype("Int64")

        regime_counts = df["regime_label"].value_counts().sort_index()
        logger.info(f"Regime distribution:\n{regime_counts.to_string()}")

    # ─── Volatility Spike (forward-looking) ───
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    # Forward vol: std of the next VOL_SPIKE_FORWARD_WINDOW days' returns
    fwd_window = config.VOL_SPIKE_FORWARD_WINDOW
    df["_fwd_vol"] = df.groupby("SecuritiesCode")["log_return"].transform(
        lambda x: x.shift(-fwd_window).rolling(fwd_window, min_periods=3).std()
    )

    if "vol_21d" in df.columns:
        df["vol_spike"] = (df["_fwd_vol"] > config.VOL_SPIKE_THRESHOLD * df["vol_21d"]).astype(int)
        spike_pct = df["vol_spike"].mean() * 100
        logger.info(f"Volatility spike prevalence: {spike_pct:.1f}%")
    df.drop(columns=["_fwd_vol"], inplace=True, errors="ignore")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FEATURE ENGINEERING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

@timed
def run_feature_engineering(df):
    """
    Full feature engineering pipeline:
    1. Per-asset features (GPU-accelerated)
    2. Cross-sectional features
    3. Target variables
    """
    logger.info("═══ Starting Feature Engineering ═══")

    import pandas as pd

    logger.info("Converting to pandas BEFORE feature engineering (safe zone)...")

    if not isinstance(df, pd.DataFrame):
        logger.info("Converting cuDF → pandas via Arrow (CUDA-safe)...")

        df = df.to_arrow().to_pandas()      

    df = compute_per_asset_features_gpu(df)
    df = compute_cross_sectional_features(df)
    df = compute_targets(df)

    # Drop rows with too many NaN features (beginning of each stock)
    key_features = [f"vol_{w}d" for w in config.ROLLING_WINDOWS] + ["log_return", "rsi_14"]
    existing_features = [f for f in key_features if f in df.columns]
    if existing_features:
        df = df.dropna(subset=existing_features[:3]).reset_index(drop=True)

    summarize_df(df, "Feature-Engineered Dataset")

    # List all new feature columns
    feature_cols = [c for c in df.columns if c not in [
        "RowId", "Date", "SecuritiesCode", "Open", "High", "Low", "Close",
        "Volume", "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag",
        "Target", "Name", "33SectorCode", "33SectorName", "17SectorCode",
        "17SectorName"
    ]]
    logger.info(f"Total feature columns: {len(feature_cols)}")

    return df


if __name__ == "__main__":
    from utils import setup_logging
    from ingestion import run_ingestion
    from cleaning import run_cleaning

    setup_logging()

    prices_df, _ = run_ingestion()
    cleaned_df = run_cleaning(prices_df)

    df = run_feature_engineering(cleaned_df)

    print(df.head())
