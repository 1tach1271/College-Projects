"""
main.py — Pipeline Orchestrator
=================================
Sequentially runs all modules of the Market Intelligence Engine.
"""

import logging
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import config
from utils import setup_logging, timed, save_intermediate, Timer

logger = logging.getLogger(__name__)


@timed
def run_pipeline():
    """Execute the full GPU-Accelerated Market Intelligence Pipeline."""

    print("=" * 70)
    print("  GPU-ACCELERATED FINANCIAL MARKET INTELLIGENCE ENGINE")
    print("  JPX Tokyo Stock Exchange — Regime & Risk Analysis")
    print("=" * 70)

    overall_start = time.perf_counter()

    # ─── Module 1: Data Ingestion ────────────────────────────────────────
    from ingestion import run_ingestion
    prices_df, stock_list_df = run_ingestion()
    save_intermediate(stock_list_df, "stock_list")

    # ─── Module 2: Data Cleaning ─────────────────────────────────────────
    from cleaning import run_cleaning
    cleaned_df = run_cleaning(prices_df)
    del prices_df  # Free memory

    # ─── Module 3: Feature Engineering ───────────────────────────────────
    from features import run_feature_engineering
    featured_df = run_feature_engineering(cleaned_df)
    del cleaned_df

    # ─── Module 4: Signal Processing ─────────────────────────────────────
    from signals import run_signal_processing
    featured_df, spectral_df = run_signal_processing(featured_df)
    save_intermediate(spectral_df, "spectral_features")

    # ─── Module 5: Graph Modeling ────────────────────────────────────────
    from graph import run_graph_modeling
    graph_results = run_graph_modeling(featured_df)
    save_intermediate(
        {k: v for k, v in graph_results.items() if k != "corr_matrix"},
        "graph_results_meta"
    )

    # ─── Module 6: Machine Learning ──────────────────────────────────────
    from ml import run_ml
    ml_results = run_ml(featured_df)

    # ─── Module 7: Risk Intelligence ─────────────────────────────────────
    from risk import run_risk_engine
    risk_results = run_risk_engine(featured_df, graph_results, ml_results)

    # ─── Module 8: Benchmarking ──────────────────────────────────────────
    from benchmark import run_benchmarks
    # Sample data for feature benchmark
    sample_stocks = featured_df["SecuritiesCode"].unique()[:50]
    df_sample = featured_df[featured_df["SecuritiesCode"].isin(sample_stocks)][
        ["SecuritiesCode", "Date", "Close", "Volume"]
    ].copy()
    benchmark_results = run_benchmarks(df_sample)

    # ─── Save Results ────────────────────────────────────────────────────
    all_results = {
        "featured_df": featured_df,
        "spectral_df": spectral_df,
        "graph_results": graph_results,
        "ml_results": ml_results,
        "risk_results": risk_results,
        "benchmark_results": benchmark_results,
        "stock_list_df": stock_list_df,
    }
    save_intermediate(all_results, "pipeline_results")

    overall_time = time.perf_counter() - overall_start

    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE — Total time: {overall_time:.1f}s")
    print("=" * 70)

    # ─── Print Summary ───────────────────────────────────────────────────
    print("\n📊 KEY RESULTS:")
    print(f"  • Data: {featured_df['SecuritiesCode'].nunique():,} stocks, "
          f"{len(featured_df):,} observations")
    print(f"  • Market Regime: {risk_results['market_regime']['regime']}")
    print(f"  • Systemic Risk: {risk_results['systemic_risk']['systemic_risk_score']}/100")
    print(f"  • Volatility Alerts: {len(risk_results['alerts'])} stocks")
    print(f"  • Graph Communities: {graph_results['n_communities']}")

    # ML summary
    for task in ["regime", "vol_spike"]:
        task_res = ml_results.get(task, {})
        for model in ["xgboost", "random_forest", "logistic_regression"]:
            if model in task_res and "f1_macro" in task_res[model]:
                print(f"  • {task}/{model}: F1={task_res[model]['f1_macro']:.4f}, "
                      f"ROC-AUC={task_res[model].get('roc_auc', 'N/A')}")

    # Benchmark summary
    print("\n⚡ GPU SPEEDUPS:")
    for op, res in benchmark_results.items():
        speedup = res.get("speedup", None)
        if speedup:
            print(f"  • {op}: {speedup}×")

    return all_results


if __name__ == "__main__":
    setup_logging(logging.INFO)
    try:
        results = run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
