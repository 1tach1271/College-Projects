"""
utils.py — Shared Utilities (PRODUCTION SAFE)
"""

import functools
import logging
import pickle
import sys
import time
from pathlib import Path

import config


# =========================================================
# 🔥 SAFE LOGGING SETUP (DOCKER SAFE)
# =========================================================
def setup_logging(level=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)-8s %(name)-20s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    try:
        # Ensure directory exists
        Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        log_path = config.OUTPUT_DIR / "pipeline.log"
        handlers.append(logging.FileHandler(log_path, mode="w"))

    except Exception as e:
        print(f"⚠️ File logging disabled: {e}")

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )

    # Reduce noise
    for lib in ["matplotlib", "PIL", "urllib3", "numba"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


# =========================================================
# ⏱ Timing Decorator
# =========================================================
def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info(f"▶ Starting {func.__name__}...")
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        logger.info(f"✓ Completed {func.__name__} in {elapsed:.2f}s")

        return result

    return wrapper


class Timer:
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

    def __repr__(self):
        return f"Timer({self.label}: {self.elapsed:.4f}s)"


# =========================================================
# 🔥 SAFE GPU → CPU CONVERSION (CRITICAL FIX)
# =========================================================
def safe_to_pandas(df, logger=None, name="DF"):
    try:
        if logger:
            logger.info(f"[SAFE CONVERT] {name}: cuDF → pandas via Arrow")

        return df.to_arrow().to_pandas()

    except Exception as e:
        if logger:
            logger.warning(f"[FALLBACK] Arrow failed for {name}: {e}")

        return pd.DataFrame(df.to_arrow().to_pylist())


# =========================================================
# 💾 Serialization
# =========================================================
def save_intermediate(obj, name: str):
    try:
        Path(config.INTERMEDIATE_DIR).mkdir(parents=True, exist_ok=True)

        path = config.INTERMEDIATE_DIR / f"{name}.pkl"

        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.getLogger(__name__).info(f"Saved intermediate: {path}")

    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save {name}: {e}")


def load_intermediate(name: str):
    path = config.INTERMEDIATE_DIR / f"{name}.pkl"

    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    return None


# =========================================================
# 📊 Data Summary
# =========================================================
def summarize_df(df, name: str = "DataFrame"):
    logger = logging.getLogger(__name__)

    logger.info(f"─── {name} Summary ───")
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    try:
        null_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        logger.info(f"  Null %: {null_pct:.2f}%")
    except Exception:
        pass

    try:
        mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
        logger.info(f"  Memory: {mem_mb:.1f} MB")
    except Exception:
        pass