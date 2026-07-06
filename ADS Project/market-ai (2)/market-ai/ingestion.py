"""
ingestion.py — Module 1: Data Ingestion & Validation
=====================================================
GPU-FIRST pipeline using cuDF (RAPIDS) with safe CPU fallback.
"""

import logging

import config
from gpu_utils import is_gpu_available, gpu_context
from utils import timed, summarize_df

logger = logging.getLogger(__name__)

# =====================================================
# EXPECTED SCHEMAS
# =====================================================

EXPECTED_COLUMNS = [
    "RowId", "Date", "SecuritiesCode", "Open", "High", "Low", "Close",
    "Volume", "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag", "Target",
]

EXPECTED_STOCK_LIST_COLS = [
    "SecuritiesCode", "Name", "33SectorCode", "33SectorName",
    "17SectorCode", "17SectorName",
]


# =====================================================
# BACKEND SELECTION (GPU vs CPU)
# =====================================================

if is_gpu_available():
    import cudf as df_lib
    BACKEND = "GPU (cuDF)"
else:
    import pandas as df_lib
    BACKEND = "CPU (pandas)"


# =====================================================
# LOAD FUNCTIONS
# =====================================================

@timed
def load_stock_prices():
    """
    Load stock_prices.csv using GPU if available.
    """
    path = str(config.STOCK_PRICES_PATH)
    logger.info(f"[{BACKEND}] Loading stock prices from: {path}")

    if is_gpu_available():
        with gpu_context("CSV Load"):
            df = df_lib.read_csv(path)
    else:
        df = df_lib.read_csv(path)

    logger.info(f"Loaded {len(df):,} rows")
    return df


@timed
def load_stock_list():
    """
    Load stock_list.csv (small → CPU is fine, but we keep consistency).
    """
    path = str(config.STOCK_LIST_PATH)
    logger.info(f"[{BACKEND}] Loading stock list from: {path}")

    df = df_lib.read_csv(path)

    logger.info(f"Loaded {len(df):,} stock entries")
    return df


# =====================================================
# VALIDATION
# =====================================================

def validate_schema(df, expected_cols, name="DataFrame"):
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")

    logger.info(f"[{name}] Schema validated — {len(expected_cols)} columns present")


# =====================================================
# PREPROCESSING (GPU SAFE)
# =====================================================

@timed
def prepare_prices(df):
    """
    Clean and prepare dataset — GPU compatible.
    """

    # ---- Date parsing ----
    try:
        df["Date"] = df["Date"].astype("datetime64[ns]")
    except Exception:
        import pandas as pd
        df["Date"] = pd.to_datetime(df["Date"])

    # ---- Numeric columns ----
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = df[col].astype("float64")

    # ---- Adjustment Factor ----
    df["AdjustmentFactor"] = df["AdjustmentFactor"].fillna(1.0).astype("float64")

    # ---- Sorting ----
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    # ---- Stats ----
    try:
        n_stocks = df["SecuritiesCode"].nunique()
        date_min = df["Date"].min()
        date_max = df["Date"].max()

        logger.info(
            f"Data prepared: {n_stocks:,} stocks | "
            f"{str(date_min)[:10]} → {str(date_max)[:10]}"
        )
    except Exception:
        logger.warning("Could not compute summary stats (backend limitation)")

    return df


# =====================================================
# PIPELINE
# =====================================================

@timed
def run_ingestion():
    """
    Full ingestion pipeline.
    """

    # ---- Load ----
    prices_df = load_stock_prices()
    stock_list_df = load_stock_list()

    # ---- Validate ----
    validate_schema(prices_df, EXPECTED_COLUMNS, "StockPrices")

    available_sl_cols = [
        c for c in EXPECTED_STOCK_LIST_COLS if c in stock_list_df.columns
    ]

    if len(available_sl_cols) < 2:
        logger.warning("Stock list has limited column coverage")

    # ---- Prepare ----
    prices_df = prepare_prices(prices_df)

    # ---- Merge Sector Info ----
    sector_cols = ["SecuritiesCode"]

    for col in [
        "33SectorCode", "33SectorName",
        "17SectorCode", "17SectorName",
        "Name"
    ]:
        if col in stock_list_df.columns:
            sector_cols.append(col)

    if len(sector_cols) > 1:
        sector_info = stock_list_df[sector_cols].drop_duplicates(
            subset=["SecuritiesCode"]
        )

        prices_df = prices_df.merge(
            sector_info,
            on="SecuritiesCode",
            how="left"
        )

        logger.info(f"Merged sector info ({len(sector_cols) - 1} columns)")

    # ---- Summary ----
    summarize_df(prices_df, "Ingested Stock Prices")

    return prices_df, stock_list_df


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    from utils import setup_logging

    setup_logging()

    prices_df, stock_list_df = run_ingestion()

    print(prices_df.head())
    print(prices_df.dtypes)