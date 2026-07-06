"""
gpu_utils.py — GPU Infrastructure & Fallback Management
========================================================
CUDA verification, memory monitoring, and CPU/GPU dispatch utilities.
"""

import functools
import logging
import time
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)

# ─── GPU Availability Detection ─────────────────────────────────────────────

_GPU_AVAILABLE = False
_GPU_INFO = {}

try:
    import cupy as cp
    import cudf
    import cuml
    import cugraph

    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count > 0:
        _GPU_AVAILABLE = True
        props = cp.cuda.runtime.getDeviceProperties(0)
        _GPU_INFO = {
            "name": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
            "total_memory_gb": round(props["totalGlobalMem"] / (1024**3), 2),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "device_count": device_count,
        }
        logger.info(f"GPU detected: {_GPU_INFO['name']} "
                     f"({_GPU_INFO['total_memory_gb']} GB, "
                     f"CC {_GPU_INFO['compute_capability']})")
    else:
        logger.warning("CUDA runtime found but no GPU devices detected.")
except ImportError as e:
    logger.warning(f"RAPIDS/CuPy not available: {e}. Using CPU fallback.")
except Exception as e:
    logger.warning(f"GPU initialization error: {e}. Using CPU fallback.")


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE


def get_gpu_info() -> dict:
    """Return GPU device information."""
    return _GPU_INFO.copy()


# ─── Memory Monitoring ──────────────────────────────────────────────────────

def gpu_memory_usage() -> dict:
    """Return current GPU memory usage in MB."""
    if not _GPU_AVAILABLE:
        return {"used_mb": 0, "total_mb": 0, "free_mb": 0}
    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_total = cp.cuda.runtime.memGetInfo()[1]
    mem_used = mem_total - mem_free
    return {
        "used_mb": round(mem_used / (1024**2), 1),
        "total_mb": round(mem_total / (1024**2), 1),
        "free_mb": round(mem_free / (1024**2), 1),
    }


def log_gpu_memory(label: str = ""):
    """Log current GPU memory usage."""
    if _GPU_AVAILABLE:
        mem = gpu_memory_usage()
        logger.info(f"[GPU MEM {label}] Used: {mem['used_mb']}MB / "
                     f"{mem['total_mb']}MB (Free: {mem['free_mb']}MB)")


# ─── CPU/GPU Dispatch Decorator ─────────────────────────────────────────────

def gpu_accelerated(fallback_fn=None):
    """
    Decorator that attempts GPU execution, falls back to CPU on failure.

    Usage:
        @gpu_accelerated(fallback_fn=cpu_version)
        def my_gpu_function(data):
            # GPU code using cuDF/CuPy
            ...
    """
    def decorator(gpu_fn):
        @functools.wraps(gpu_fn)
        def wrapper(*args, **kwargs):
            if _GPU_AVAILABLE:
                try:
                    return gpu_fn(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"GPU execution failed for {gpu_fn.__name__}: {e}")
                    if fallback_fn:
                        logger.info(f"Falling back to CPU: {fallback_fn.__name__}")
                        return fallback_fn(*args, **kwargs)
                    raise
            elif fallback_fn:
                logger.info(f"No GPU — using CPU fallback: {fallback_fn.__name__}")
                return fallback_fn(*args, **kwargs)
            else:
                raise RuntimeError(f"GPU required for {gpu_fn.__name__} but not available")
        return wrapper
    return decorator


# ─── Conversion Utilities ───────────────────────────────────────────────────

def to_cudf(df):
    """Convert pandas DataFrame to cuDF DataFrame."""
    if _GPU_AVAILABLE:
        import cudf
        if isinstance(df, cudf.DataFrame):
            return df
        return cudf.from_pandas(df)
    return df


def to_pandas(df):
    """Convert cuDF DataFrame to pandas DataFrame."""
    if _GPU_AVAILABLE:
        import cudf
        if isinstance(df, cudf.DataFrame):
            return df.to_pandas()
    return df


def to_cupy(arr):
    """Convert numpy array to CuPy array."""
    if _GPU_AVAILABLE:
        if isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(arr)
    return arr


def to_numpy(arr):
    """Convert CuPy array to numpy array."""
    if _GPU_AVAILABLE:
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return np.asarray(arr)


# ─── Context Manager for GPU Operations ─────────────────────────────────────

@contextmanager
def gpu_context(label: str = "operation"):
    """Context manager that logs GPU memory and timing for a block."""
    start = time.perf_counter()
    log_gpu_memory(f"{label} START")
    try:
        yield
    finally:
        if _GPU_AVAILABLE:
            cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        log_gpu_memory(f"{label} END")
        logger.info(f"[TIMING] {label}: {elapsed:.3f}s")
