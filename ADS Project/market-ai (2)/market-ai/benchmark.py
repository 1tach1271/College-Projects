"""
benchmark.py — Module 8: CPU vs GPU Benchmarking
==================================================
Compares execution time and performance for key operations:
CSV loading, feature engineering, FFT, correlation, XGBoost, graph.
"""

import logging
import time
import tracemalloc

import numpy as np
import pandas as pd

import config
from gpu_utils import is_gpu_available
from utils import timed, Timer

logger = logging.getLogger(__name__)


def _measure(func, *args, iterations=None, **kwargs):
    """
    Measure execution time and peak memory of a function.
    Returns: (result, avg_time_seconds, peak_memory_mb)
    """
    iterations = iterations or config.BENCHMARK_ITERATIONS
    times = []
    result = None

    for i in range(iterations):
        tracemalloc.start()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)

    avg_time = np.mean(times)
    peak_mb = peak / (1024 ** 2)
    return result, avg_time, peak_mb


@timed
def benchmark_csv_loading():
    """Benchmark CSV loading: pandas vs cuDF."""
    path = str(config.STOCK_PRICES_PATH)
    results = {}

    # CPU: pandas
    def load_pandas():
        return pd.read_csv(path)

    _, cpu_time, cpu_mem = _measure(load_pandas, iterations=1)
    results["cpu"] = {"time_s": round(cpu_time, 3), "memory_mb": round(cpu_mem, 1)}
    logger.info(f"CSV Load CPU: {cpu_time:.3f}s, {cpu_mem:.1f}MB")

    # GPU: cuDF
    if is_gpu_available():
        import cudf

        def load_cudf():
            gdf = cudf.read_csv(path)
            return gdf

        _, gpu_time, gpu_mem = _measure(load_cudf, iterations=1)
        results["gpu"] = {"time_s": round(gpu_time, 3), "memory_mb": round(gpu_mem, 1)}
        results["speedup"] = round(cpu_time / max(gpu_time, 0.001), 1)
        logger.info(f"CSV Load GPU: {gpu_time:.3f}s (speedup: {results['speedup']}×)")

    return results


@timed
def benchmark_feature_engineering(df_sample):
    """Benchmark rolling window computations."""
    results = {}
    sample = df_sample.copy()

    # CPU: pandas groupby agg
    def cpu_features(df):
        return df.groupby("SecuritiesCode").agg({
            "Close": ["mean", "std", "max", "min"],
            "Volume": ["sum", "mean"]
        })

    _, cpu_time, cpu_mem = _measure(cpu_features, sample, iterations=1)
    results["cpu"] = {"time_s": round(cpu_time, 3), "memory_mb": round(cpu_mem, 1)}
    logger.info(f"Groupby Agg CPU: {cpu_time:.3f}s")

    # GPU: cuDF groupby agg
    if is_gpu_available():
        import cudf

        def gpu_features(df):
            gdf = cudf.from_pandas(df.copy())
            res = gdf.groupby("SecuritiesCode").agg({
                "Close": ["mean", "std", "max", "min"],
                "Volume": ["sum", "mean"]
            })
            return res.to_pandas()

        _, gpu_time, gpu_mem = _measure(gpu_features, sample, iterations=1)
        results["gpu"] = {"time_s": round(gpu_time, 3), "memory_mb": round(gpu_mem, 1)}
        results["speedup"] = round(cpu_time / max(gpu_time, 0.001), 1)
        logger.info(f"Features GPU: {gpu_time:.3f}s (speedup: {results['speedup']}×)")

    return results


@timed
def benchmark_fft(n_samples=10000):
    """Benchmark FFT: scipy vs cupy."""
    results = {}
    data = np.random.randn(n_samples).astype(np.float64)

    # CPU: scipy
    from scipy.fft import fft as scipy_fft

    def cpu_fft():
        return scipy_fft(data * np.hanning(n_samples))

    _, cpu_time, cpu_mem = _measure(cpu_fft)
    results["cpu"] = {"time_s": round(cpu_time, 5), "memory_mb": round(cpu_mem, 1)}
    logger.info(f"FFT CPU ({n_samples} samples): {cpu_time*1000:.2f}ms")

    # GPU: cupy
    if is_gpu_available():
        import cupy as cp

        gpu_data = cp.asarray(data)

        def gpu_fft():
            windowed = gpu_data * cp.hanning(n_samples)
            result = cp.fft.fft(windowed)
            cp.cuda.Stream.null.synchronize()
            return result

        _, gpu_time, gpu_mem = _measure(gpu_fft)
        results["gpu"] = {"time_s": round(gpu_time, 5), "memory_mb": round(gpu_mem, 1)}
        results["speedup"] = round(cpu_time / max(gpu_time, 0.00001), 1)
        logger.info(f"FFT GPU ({n_samples} samples): {gpu_time*1000:.2f}ms (speedup: {results['speedup']}×)")

    return results


@timed
def benchmark_correlation(n_stocks=200, n_days=63):
    """Benchmark correlation matrix computation."""
    results = {}
    data = np.random.randn(n_days, n_stocks).astype(np.float64)

    # CPU: numpy
    def cpu_corr():
        return np.corrcoef(data.T)

    _, cpu_time, cpu_mem = _measure(cpu_corr)
    results["cpu"] = {"time_s": round(cpu_time, 5), "memory_mb": round(cpu_mem, 1)}
    logger.info(f"Correlation CPU ({n_stocks}×{n_days}): {cpu_time*1000:.2f}ms")

    # GPU: cupy
    if is_gpu_available():
        import cupy as cp
        gpu_data = cp.asarray(data)

        def gpu_corr():
            result = cp.corrcoef(gpu_data.T)
            cp.cuda.Stream.null.synchronize()
            return result

        _, gpu_time, gpu_mem = _measure(gpu_corr)
        results["gpu"] = {"time_s": round(gpu_time, 5), "memory_mb": round(gpu_mem, 1)}
        results["speedup"] = round(cpu_time / max(gpu_time, 0.00001), 1)
        logger.info(f"Correlation GPU ({n_stocks}×{n_days}): {gpu_time*1000:.2f}ms (speedup: {results['speedup']}×)")

    return results


@timed
def benchmark_xgboost(n_samples=50000, n_features=20):
    """Benchmark XGBoost training: CPU vs GPU."""
    import xgboost as xgb
    results = {}

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (np.random.randn(n_samples) > 0).astype(np.int32)

    # CPU
    def cpu_xgb():
        dtrain = xgb.DMatrix(X, label=y)
        params = {"max_depth": 6, "learning_rate": 0.1, "tree_method": "hist",
                  "objective": "binary:logistic", "eval_metric": "logloss"}
        model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        return model

    _, cpu_time, cpu_mem = _measure(cpu_xgb, iterations=1)
    results["cpu"] = {"time_s": round(cpu_time, 3), "memory_mb": round(cpu_mem, 1)}
    logger.info(f"XGBoost CPU ({n_samples} samples): {cpu_time:.3f}s")

    # GPU
    if is_gpu_available():
        def gpu_xgb():
            dtrain = xgb.DMatrix(X, label=y)
            params = {"max_depth": 6, "learning_rate": 0.1, "tree_method": "gpu_hist",
                      "device": "cuda", "objective": "binary:logistic", "eval_metric": "logloss"}
            model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
            return model

        _, gpu_time, gpu_mem = _measure(gpu_xgb, iterations=1)
        results["gpu"] = {"time_s": round(gpu_time, 3), "memory_mb": round(gpu_mem, 1)}
        results["speedup"] = round(cpu_time / max(gpu_time, 0.001), 1)
        logger.info(f"XGBoost GPU ({n_samples} samples): {gpu_time:.3f}s (speedup: {results['speedup']}×)")

    return results


@timed
def benchmark_graph_community(n_nodes=200, n_edges=2000):
    """Benchmark graph community detection: NetworkX vs cuGraph."""
    results = {}

    # Generate random graph edges
    src = np.random.randint(0, n_nodes, n_edges).astype(np.int32)
    dst = np.random.randint(0, n_nodes, n_edges).astype(np.int32)
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    weights = np.random.rand(len(src)).astype(np.float32)

    # CPU: NetworkX
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    def cpu_community():
        G = nx.Graph()
        for i in range(len(src)):
            G.add_edge(int(src[i]), int(dst[i]), weight=float(weights[i]))
        communities = list(greedy_modularity_communities(G, weight="weight"))
        return len(communities)

    _, cpu_time, cpu_mem = _measure(cpu_community, iterations=1)
    results["cpu"] = {"time_s": round(cpu_time, 3), "memory_mb": round(cpu_mem, 1)}
    logger.info(f"Graph Community CPU ({n_nodes} nodes): {cpu_time:.3f}s")

    # GPU: cuGraph
    if is_gpu_available():
        import cudf
        import cugraph

        def gpu_community():
            g_edges = cudf.DataFrame({"src": src, "dst": dst, "weight": weights})
            G = cugraph.Graph()
            G.from_cudf_edgelist(g_edges, source="src", destination="dst", edge_attr="weight")
            parts, modularity = cugraph.louvain(G)
            return parts

        _, gpu_time, gpu_mem = _measure(gpu_community, iterations=1)
        results["gpu"] = {"time_s": round(gpu_time, 3), "memory_mb": round(gpu_mem, 1)}
        results["speedup"] = round(cpu_time / max(gpu_time, 0.001), 1)
        logger.info(f"Graph Community GPU ({n_nodes} nodes): {gpu_time:.3f}s (speedup: {results['speedup']}×)")

    return results


@timed
def run_benchmarks(df_sample=None):
    """
    Run all CPU vs GPU benchmarks.
    Returns: dict with results for each operation.
    """
    logger.info("═══ Starting CPU vs GPU Benchmarking ═══")

    benchmarks = {}

    benchmarks["csv_loading"] = benchmark_csv_loading()
    benchmarks["fft"] = benchmark_fft()
    benchmarks["correlation"] = benchmark_correlation()
    benchmarks["xgboost"] = benchmark_xgboost()
    benchmarks["graph_community"] = benchmark_graph_community()

    if df_sample is not None and len(df_sample) > 0:
        benchmarks["feature_engineering"] = benchmark_feature_engineering(df_sample)

    # Summary table
    logger.info("\n═══ Benchmark Summary ═══")
    summary_lines = []
    for op, res in benchmarks.items():
        cpu_time = res.get("cpu", {}).get("time_s", "N/A")
        gpu_time = res.get("gpu", {}).get("time_s", "N/A")
        speedup = res.get("speedup", "N/A")
        summary_lines.append(f"  {op:25s}  CPU: {str(cpu_time):>10s}s  GPU: {str(gpu_time):>10s}s  Speedup: {str(speedup):>6s}×")
    logger.info("\n".join(summary_lines))

    return benchmarks
