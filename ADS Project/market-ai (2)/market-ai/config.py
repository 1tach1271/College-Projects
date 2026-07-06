"""
config.py — Centralized Configuration for Market Intelligence Engine
=====================================================================
All paths, parameters, hyperparameters, and thresholds in one place.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories
for d in [OUTPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset Paths ───────────────────────────────────────────────────────────
STOCK_PRICES_PATH = DATA_DIR / "stock_prices.csv"
STOCK_LIST_PATH = DATA_DIR / "stock_list.csv"

# ─── Data Cleaning ──────────────────────────────────────────────────────────
MIN_TRADING_DAYS = 252          # Minimum days to keep a stock (1 year)
OUTLIER_SIGMA = 10.0            # Remove returns beyond 10σ
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

# ─── Feature Engineering ────────────────────────────────────────────────────
ROLLING_WINDOWS = [5, 10, 21, 63, 126]          # Trading day windows
SHORT_WINDOWS = [5, 10]                           # Short-term MAs
LONG_WINDOWS = [63, 126]                           # Long-term MAs
MOMENTUM_PERIODS = [5, 10, 21]                     # Momentum lookback
RSI_PERIOD = 14                                    # RSI window
VOLATILITY_WINDOW = 21                             # Primary vol window

# ─── Regime Labels ──────────────────────────────────────────────────────────
REGIME_LABELS = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}
VOL_SPIKE_THRESHOLD = 2.0       # Forward vol must be > 2× current vol
VOL_SPIKE_FORWARD_WINDOW = 5    # Days ahead for spike detection

# ─── Signal Processing ──────────────────────────────────────────────────────
FFT_WINDOW = 126                 # FFT sliding window (≈6 months)
FFT_STEP = 21                   # FFT step size
SPECTRAL_BANDS = {               # Frequency band boundaries (as fraction of Nyquist)
    "low": (0.0, 0.1),
    "mid": (0.1, 0.4),
    "high": (0.4, 1.0),
}

# ─── Correlation Graph ──────────────────────────────────────────────────────
GRAPH_TOP_N_STOCKS = 200          # Most liquid stocks for graph analysis
CORR_ROLLING_WINDOW = 63         # Rolling window for correlation
CORR_EDGE_THRESHOLD = 0.5       # Minimum |corr| to create edge
GRAPH_SNAPSHOT_DATES = 5         # Number of snapshots for time evolution

# ─── Machine Learning ───────────────────────────────────────────────────────
TRAIN_RATIO = 0.8                # 80/20 time-aware split
PURGE_GAP_DAYS = 5               # Gap between train and test
RANDOM_STATE = 42

# XGBoost GPU params
XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "tree_method": "gpu_hist",
    "device": "cuda",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Random Forest (cuML)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
}

# Logistic Regression (cuML)
LR_PARAMS = {
    "max_iter": 1000,
    "C": 1.0,
}

# ─── Risk Scoring ───────────────────────────────────────────────────────────
RISK_WEIGHTS = {
    "volatility_rank": 0.25,
    "drawdown_severity": 0.20,
    "graph_centrality": 0.15,
    "spike_probability": 0.25,
    "regime_score": 0.15,
}

# Decomposed Risk Categories (for Spider Chart)
RISK_FACTORS = {
    "Volatility Risk": ["volatility_rank", "regime_score"],
    "Structural Risk": ["graph_centrality"],
    "Tail Risk": ["drawdown_severity", "spike_probability"],
}

VOLATILITY_ALERT_THRESHOLD = 0.6   # Spike probability threshold for alert

# ─── Sentiment Proxy (Quant-based) ──────────────────────────────────────────
SENTIMENT_WEIGHTS = {
    "momentum": 0.4,       # Momentum 10d/21d
    "rsi": 0.3,            # RSI 14
    "breadth": 0.3,        # Market Breadth
}
SENTIMENT_LABELS = [
    (25, "Extreme Fear"),
    (45, "Fear"),
    (55, "Neutral"),
    (75, "Greed"),
    (100, "Extreme Greed")
]

# ─── Benchmarking ───────────────────────────────────────────────────────────
BENCHMARK_ITERATIONS = 3            # Repeat for stable timing

# ─── Dashboard ──────────────────────────────────────────────────────────────
DASH_HOST = "0.0.0.0"
DASH_PORT = 8050
DASH_DEBUG = False
