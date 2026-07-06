"""
signals.py — Module 4: FFT Signal Processing (GPU-Accelerated)
===============================================================
Applies FFT on per-asset return series to extract spectral features:
dominant frequencies, spectral entropy, energy band ratios,
spectral edge frequency. Detects volatility bursts and regime shifts.
"""

import logging

import numpy as np
import pandas as pd

import config
from gpu_utils import is_gpu_available, gpu_context
from utils import timed

logger = logging.getLogger(__name__)


def _fft_features_numpy(returns_array):
    """
    Compute FFT-based spectral features from a 1D array of returns.
    Uses NumPy/SciPy (CPU).
    """
    from scipy.fft import fft, fftfreq

    n = len(returns_array)
    if n < 16:
        return {}

    # Apply Hanning window to reduce spectral leakage
    windowed = returns_array * np.hanning(n)

    # FFT
    fft_vals = fft(windowed)
    power_spectrum = np.abs(fft_vals[:n // 2]) ** 2
    freqs = fftfreq(n, d=1.0)[:n // 2]

    # Normalize power spectrum
    total_power = power_spectrum.sum() + 1e-10
    norm_power = power_spectrum / total_power

    # 1. Dominant frequency
    dominant_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC
    dominant_freq = freqs[dominant_idx] if dominant_idx < len(freqs) else 0.0

    # 2. Spectral entropy (negative entropy → randomness measure)
    norm_power_clean = norm_power[norm_power > 0]
    spectral_entropy = -np.sum(norm_power_clean * np.log2(norm_power_clean + 1e-10))

    # 3. Energy band ratios
    nyquist = 0.5  # cycles per day
    low_mask = freqs < config.SPECTRAL_BANDS["low"][1] * nyquist
    mid_mask = (freqs >= config.SPECTRAL_BANDS["mid"][0] * nyquist) & \
               (freqs < config.SPECTRAL_BANDS["mid"][1] * nyquist)
    high_mask = freqs >= config.SPECTRAL_BANDS["high"][0] * nyquist

    low_energy = power_spectrum[low_mask].sum() / total_power
    mid_energy = power_spectrum[mid_mask].sum() / total_power
    high_energy = power_spectrum[high_mask].sum() / total_power

    # 4. Spectral edge frequency (95% cumulative power)
    cum_power = np.cumsum(power_spectrum) / total_power
    edge_idx = np.searchsorted(cum_power, 0.95)
    spectral_edge_freq = freqs[min(edge_idx, len(freqs) - 1)]

    return {
        "dominant_freq": float(dominant_freq),
        "spectral_entropy": float(spectral_entropy),
        "low_freq_energy": float(low_energy),
        "mid_freq_energy": float(mid_energy),
        "high_freq_energy": float(high_energy),
        "spectral_edge_freq": float(spectral_edge_freq),
    }


def _fft_features_cupy(returns_array):
    """
    Compute FFT-based spectral features using CuPy (GPU).
    """
    import cupy as cp

    n = len(returns_array)
    if n < 16:
        return {}

    # Transfer to GPU
    gpu_arr = cp.asarray(returns_array, dtype=cp.float64)

    # Apply Hanning window
    window = cp.hanning(n)
    windowed = gpu_arr * window

    # FFT on GPU
    fft_vals = cp.fft.fft(windowed)
    power_spectrum = cp.abs(fft_vals[:n // 2]) ** 2
    freqs = cp.fft.fftfreq(n, d=1.0)[:n // 2]

    total_power = power_spectrum.sum() + 1e-10
    norm_power = power_spectrum / total_power

    # Dominant freq
    dominant_idx = int(cp.argmax(power_spectrum[1:])) + 1
    freqs_np = cp.asnumpy(freqs)
    dominant_freq = float(freqs_np[dominant_idx]) if dominant_idx < len(freqs_np) else 0.0

    # Spectral entropy
    norm_power_np = cp.asnumpy(norm_power)
    mask = norm_power_np > 0
    spectral_entropy = float(-np.sum(norm_power_np[mask] * np.log2(norm_power_np[mask] + 1e-10)))

    # Energy bands
    nyquist = 0.5
    ps_np = cp.asnumpy(power_spectrum)
    tp = float(cp.asnumpy(total_power))

    low_mask = freqs_np < config.SPECTRAL_BANDS["low"][1] * nyquist
    mid_mask = (freqs_np >= config.SPECTRAL_BANDS["mid"][0] * nyquist) & \
               (freqs_np < config.SPECTRAL_BANDS["mid"][1] * nyquist)
    high_mask = freqs_np >= config.SPECTRAL_BANDS["high"][0] * nyquist

    low_energy = ps_np[low_mask].sum() / tp
    mid_energy = ps_np[mid_mask].sum() / tp
    high_energy = ps_np[high_mask].sum() / tp

    # Spectral edge
    cum_power = np.cumsum(ps_np) / tp
    edge_idx = np.searchsorted(cum_power, 0.95)
    spectral_edge_freq = float(freqs_np[min(edge_idx, len(freqs_np) - 1)])

    return {
        "dominant_freq": dominant_freq,
        "spectral_entropy": spectral_entropy,
        "low_freq_energy": float(low_energy),
        "mid_freq_energy": float(mid_energy),
        "high_freq_energy": float(high_energy),
        "spectral_edge_freq": spectral_edge_freq,
    }


@timed
def compute_spectral_features(df):
    """
    Compute FFT-based spectral features for each stock using sliding windows.
    Uses GPU if available, otherwise CPU.
    """
    logger.info("═══ Starting Signal Processing (FFT) ═══")

    fft_func = _fft_features_cupy if is_gpu_available() else _fft_features_numpy
    backend = "GPU (CuPy)" if is_gpu_available() else "CPU (NumPy/SciPy)"
    logger.info(f"FFT backend: {backend}")

    # Get unique stocks
    stocks = df["SecuritiesCode"].unique()
    logger.info(f"Processing FFT for {len(stocks)} stocks...")

    spectral_results = []
    processed = 0

    for stock_code in stocks:
        stock_data = df[df["SecuritiesCode"] == stock_code].sort_values("Date")
        returns = stock_data["log_return"].dropna().values

        if len(returns) < config.FFT_WINDOW:
            # Use whatever data is available
            window_returns = returns
            features = fft_func(window_returns)
            if features:
                features["SecuritiesCode"] = stock_code
                features["Date"] = stock_data["Date"].iloc[-1]
                spectral_results.append(features)
        else:
            # Sliding window FFT
            for start in range(0, len(returns) - config.FFT_WINDOW + 1, config.FFT_STEP):
                window_returns = returns[start:start + config.FFT_WINDOW]
                features = fft_func(window_returns)
                if features:
                    # Map to corresponding date
                    date_idx = stock_data.index[min(start + config.FFT_WINDOW - 1, len(stock_data) - 1)]
                    features["SecuritiesCode"] = stock_code
                    features["Date"] = stock_data.loc[date_idx, "Date"]
                    spectral_results.append(features)

        processed += 1
        if processed % 500 == 0:
            logger.info(f"  FFT processed: {processed}/{len(stocks)} stocks")

    spectral_df = pd.DataFrame(spectral_results)
    logger.info(f"Spectral features computed: {len(spectral_df):,} observations")

    return spectral_df


@timed
def detect_regime_shifts(spectral_df):
    """
    Detect regime shifts via rolling Z-score of spectral entropy.
    A sharp change in spectral entropy indicates structural market change.
    """
    if spectral_df.empty or "spectral_entropy" not in spectral_df.columns:
        return spectral_df

    spectral_df = spectral_df.sort_values(["SecuritiesCode", "Date"])

    # Rolling stats on spectral entropy per stock
    spectral_df["entropy_mean_21d"] = spectral_df.groupby("SecuritiesCode")["spectral_entropy"].transform(
        lambda x: x.rolling(5, min_periods=2).mean()
    )
    spectral_df["entropy_std_21d"] = spectral_df.groupby("SecuritiesCode")["spectral_entropy"].transform(
        lambda x: x.rolling(5, min_periods=2).std()
    )
    spectral_df["entropy_zscore"] = (
        (spectral_df["spectral_entropy"] - spectral_df["entropy_mean_21d"])
        / (spectral_df["entropy_std_21d"] + 1e-10)
    )

    # Regime shift flag: Z-score > 2
    spectral_df["regime_shift_signal"] = (spectral_df["entropy_zscore"].abs() > 2).astype(int)

    n_shifts = spectral_df["regime_shift_signal"].sum()
    logger.info(f"Detected {n_shifts:,} regime shift signals")

    return spectral_df


@timed
def detect_volatility_bursts(spectral_df):
    """
    Detect volatility bursts via sudden high-frequency energy spikes.
    """
    if spectral_df.empty or "high_freq_energy" not in spectral_df.columns:
        return spectral_df

    spectral_df = spectral_df.sort_values(["SecuritiesCode", "Date"])

    # Rolling mean of high-freq energy
    spectral_df["hf_energy_mean"] = spectral_df.groupby("SecuritiesCode")["high_freq_energy"].transform(
        lambda x: x.rolling(5, min_periods=2).mean()
    )
    spectral_df["hf_energy_std"] = spectral_df.groupby("SecuritiesCode")["high_freq_energy"].transform(
        lambda x: x.rolling(5, min_periods=2).std()
    )
    spectral_df["hf_energy_zscore"] = (
        (spectral_df["high_freq_energy"] - spectral_df["hf_energy_mean"])
        / (spectral_df["hf_energy_std"] + 1e-10)
    )

    # Volatility burst: high-freq energy Z-score > 2
    spectral_df["volatility_burst_signal"] = (spectral_df["hf_energy_zscore"] > 2).astype(int)

    n_bursts = spectral_df["volatility_burst_signal"].sum()
    logger.info(f"Detected {n_bursts:,} volatility burst signals")

    return spectral_df


@timed
def run_signal_processing(df):
    """
    Full signal processing pipeline:
    1. Compute spectral features via FFT
    2. Detect regime shifts
    3. Detect volatility bursts
    4. Merge back to main DataFrame
    """
    spectral_df = compute_spectral_features(df)
    spectral_df = detect_regime_shifts(spectral_df)
    spectral_df = detect_volatility_bursts(spectral_df)

    # Merge spectral features back to main df via regular merge
    # (merge_asof was failing on sort order; use left merge + ffill instead)
    spectral_merge_cols = [
        "SecuritiesCode", "Date",
        "dominant_freq", "spectral_entropy",
        "low_freq_energy", "mid_freq_energy", "high_freq_energy",
        "spectral_edge_freq", "entropy_zscore",
        "regime_shift_signal", "volatility_burst_signal",
    ]
    existing_cols = [c for c in spectral_merge_cols if c in spectral_df.columns]
    spectral_merge = spectral_df[existing_cols].drop_duplicates(subset=["SecuritiesCode", "Date"])

    # Merge on exact (SecuritiesCode, Date) matches
    merged = df.merge(spectral_merge, on=["SecuritiesCode", "Date"], how="left")

    # Forward-fill spectral features within each stock (since FFT uses a step of 21d)
    spectral_feat_cols = [c for c in existing_cols if c not in ["SecuritiesCode", "Date"]]
    merged = merged.sort_values(["SecuritiesCode", "Date"])
    merged[spectral_feat_cols] = merged.groupby("SecuritiesCode")[spectral_feat_cols].ffill()

    n_spectral = sum(1 for c in merged.columns if c in [
        "dominant_freq", "spectral_entropy", "low_freq_energy",
        "mid_freq_energy", "high_freq_energy", "spectral_edge_freq"
    ])
    logger.info(f"Merged {n_spectral} spectral features into main dataset")

    return merged, spectral_df


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    from ingestion import run_ingestion
    from cleaning import run_cleaning
    from features import run_feature_engineering
    prices_df, _ = run_ingestion()
    cleaned_df = run_cleaning(prices_df)
    featured_df = run_feature_engineering(cleaned_df)
    merged_df, spectral_df = run_signal_processing(featured_df)
    print(spectral_df.head())
