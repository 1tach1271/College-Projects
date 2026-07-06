"""
risk.py — Module 7: Risk Intelligence Engine
==============================================
Generates per-asset risk scores, market regime labels,
volatility alerts, and systemic risk indicators.
"""

import logging

import numpy as np
import pandas as pd

import config
from utils import timed

logger = logging.getLogger(__name__)


@timed
def compute_risk_scores(df, graph_results, ml_results):
    """
    Compute composite risk score (0-100) per asset and its driving factors.
    """
    logger.info("═══ Computing Risk Scores ═══")

    # Get latest data per stock
    latest = df.sort_values("Date").groupby("SecuritiesCode").last().reset_index()

    risk_df = latest[["SecuritiesCode", "Date"]].copy()

    # Add sector info if available
    for col in ["Name", "17SectorName", "33SectorName"]:
        if col in latest.columns:
            risk_df[col] = latest[col].values

    # ─── Component 1: Volatility Rank (0-1) ───
    if "vol_rank" in latest.columns:
        risk_df["volatility_rank"] = latest["vol_rank"].values
    elif "vol_21d" in latest.columns:
        risk_df["volatility_rank"] = latest["vol_21d"].rank(pct=True).values
    else:
        risk_df["volatility_rank"] = 0.5

    # ─── Component 2: Drawdown Severity (0-1) ───
    if "max_drawdown_63d" in latest.columns:
        # Drawdown is negative, so negate and normalize
        dd = latest["max_drawdown_63d"].values
        dd_severity = np.clip(-dd, 0, 1)  # More negative = more severe
        risk_df["drawdown_severity"] = dd_severity
    else:
        risk_df["drawdown_severity"] = 0.0

    # ─── Component 3: Graph Centrality (0-1) ───
    if graph_results and "centrality_df" in graph_results:
        centrality_df = graph_results["centrality_df"]
        pr_map = centrality_df.set_index("SecuritiesCode")["pagerank"].to_dict()
        max_pr = max(pr_map.values()) if pr_map else 1.0
        risk_df["graph_centrality"] = risk_df["SecuritiesCode"].map(
            {k: v / max_pr for k, v in pr_map.items()}
        ).fillna(0.0)
    else:
        risk_df["graph_centrality"] = 0.0

    # ─── Component 4: Spike Probability (0-1) ───
    # Use the best ML model's prediction if available
    if "vol_spike" in latest.columns:
        risk_df["spike_probability"] = latest["vol_spike"].values.astype(float)
    else:
        risk_df["spike_probability"] = 0.0

    # ─── Component 5: Regime Score (0-1) ───
    if "regime_label" in latest.columns:
        risk_df["regime_score"] = latest["regime_label"].values.astype(float) / 2.0
    else:
        risk_df["regime_score"] = 0.5

    # ─── Decomposed Factors ───
    # Volatility Risk: combination of vol_rank and regime
    risk_df["factor_volatility"] = (
        (risk_df["volatility_rank"] + risk_df["regime_score"]) / 2.0 * 100
    ).round(1)

    # Structural Risk: based on graph centrality
    risk_df["factor_structural"] = (risk_df["graph_centrality"] * 100).round(1)

    # Tail Risk: combination of drawdown and spike probability
    risk_df["factor_tail"] = (
        (risk_df["drawdown_severity"] + risk_df["spike_probability"]) / 2.0 * 100
    ).round(1)

    # ─── Composite Risk Score (0-100) ───
    w = config.RISK_WEIGHTS
    risk_df["risk_score"] = (
        w["volatility_rank"] * risk_df["volatility_rank"] +
        w["drawdown_severity"] * risk_df["drawdown_severity"] +
        w["graph_centrality"] * risk_df["graph_centrality"] +
        w["spike_probability"] * risk_df["spike_probability"] +
        w["regime_score"] * risk_df["regime_score"]
    ) * 100

    risk_df["risk_score"] = risk_df["risk_score"].clip(0, 100).round(1)

    logger.info(f"Risk scores: mean={risk_df['risk_score'].mean():.1f}, "
                f"max={risk_df['risk_score'].max():.1f}, "
                f"min={risk_df['risk_score'].min():.1f}")

    return risk_df


@timed
def compute_market_sentiment(df):
    """
    Compute Quant-based Sentiment Proxy Index (0-100).
    Blends Momentum, RSI, and Market Breadth.
    """
    logger.info("═══ Computing Market Sentiment Proxy ═══")

    latest_date = df["Date"].max()
    latest_data = df[df["Date"] == latest_date]

    if latest_data.empty:
        return {"sentiment_score": 50, "label": "Neutral"}

    # 1. Momentum Component (0-1)
    if "momentum_21d" in latest_data.columns:
        # Scale momentum values (assumes returns are around 0)
        mom = latest_data["momentum_21d"].median()
        mom_score = np.clip((mom + 0.05) / 0.1, 0, 1)  # -5% to +5% range
    else:
        mom_score = 0.5

    # 2. RSI Component (0-1)
    if "rsi_14" in latest_data.columns:
        rsi = latest_data["rsi_14"].median()
        rsi_score = rsi / 100.0
    else:
        rsi_score = 0.5

    # 3. Breadth Component (0-1)
    if "market_breadth" in latest_data.columns:
        breadth = latest_data["market_breadth"].iloc[0]
        breadth_score = breadth
    else:
        breadth_score = 0.5

    # Weighted Average
    w = config.SENTIMENT_WEIGHTS
    score = (
        w["momentum"] * mom_score +
        w["rsi"] * rsi_score +
        w["breadth"] * breadth_score
    ) * 100

    # Get Label
    label = "Neutral"
    for threshold, l in config.SENTIMENT_LABELS:
        if score <= threshold:
            label = l
            break

    result = {
        "score": round(score, 1),
        "label": label,
        "momentum_median": round(mom_score, 3),
        "rsi_median": round(rsi_score, 3),
        "breadth": round(breadth_score, 3),
    }

    logger.info(f"Market Sentiment: {label} ({score:.1f}/100)")
    return result


@timed
def compute_market_regime(df):
    """
    Compute market-wide regime label based on aggregate volatility.
    """
    if "regime_label" not in df.columns:
        return {"regime": "Unknown", "regime_id": -1, "market_vol": 0.0}

    latest_date = df["Date"].max()
    latest_data = df[df["Date"] == latest_date]

    # Market regime = mode of individual regimes
    regime_mode = latest_data["regime_label"].mode()
    regime_id = int(regime_mode.iloc[0]) if len(regime_mode) > 0 else 1
    regime_label = config.REGIME_LABELS.get(regime_id, "Unknown")

    # Market-wide volatility
    market_vol = latest_data["vol_21d"].median() if "vol_21d" in latest_data.columns else 0.0

    result = {
        "regime": regime_label,
        "regime_id": regime_id,
        "market_vol": float(market_vol),
        "date": str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date),
    }

    logger.info(f"Market regime: {regime_label} (vol={market_vol:.4f})")
    return result


@timed
def generate_volatility_alerts(risk_df):
    """
    Generate volatility alerts for stocks with high spike probability.
    """
    threshold = config.VOLATILITY_ALERT_THRESHOLD
    alerts = risk_df[risk_df["spike_probability"] >= threshold].copy()
    alerts = alerts.sort_values("risk_score", ascending=False)

    logger.info(f"Volatility alerts: {len(alerts)} stocks (threshold={threshold})")

    return alerts


@timed
def compute_systemic_risk(graph_results, risk_df):
    """
    Compute market-level systemic risk indicator.
    Combines graph density, mean centrality, and market vol level.
    """
    if not graph_results:
        return {"systemic_risk_score": 0.0, "components": {}}

    density = graph_results.get("graph_density", 0)
    centrality_df = graph_results.get("centrality_df", pd.DataFrame())

    mean_centrality = centrality_df["pagerank"].mean() if len(centrality_df) > 0 else 0
    mean_vol_rank = risk_df["volatility_rank"].mean()

    # Systemic risk = normalized combination
    systemic = (
        0.4 * min(density / 0.5, 1.0) +  # High density → high risk
        0.3 * min(mean_centrality * 100, 1.0) +  # High centrality → concentrated risk
        0.3 * mean_vol_rank  # High vol → high risk
    ) * 100

    result = {
        "systemic_risk_score": round(systemic, 1),
        "graph_density": round(density, 4),
        "mean_centrality": round(mean_centrality, 6),
        "mean_vol_rank": round(mean_vol_rank, 4),
        "n_communities": graph_results.get("n_communities", 0),
    }

    logger.info(f"Systemic risk score: {systemic:.1f}/100")
    return result


@timed
def compute_sector_risk(risk_df):
    """
    Aggregate risk by sector (17-sector classification).
    """
    sector_col = "17SectorName" if "17SectorName" in risk_df.columns else \
                 "33SectorName" if "33SectorName" in risk_df.columns else None

    if sector_col is None:
        logger.warning("No sector column available for sector risk")
        return pd.DataFrame()

    sector_risk = risk_df.groupby(sector_col).agg(
        mean_risk=("risk_score", "mean"),
        max_risk=("risk_score", "max"),
        n_stocks=("SecuritiesCode", "count"),
        mean_vol_rank=("volatility_rank", "mean"),
        mean_drawdown=("drawdown_severity", "mean"),
    ).round(2).sort_values("mean_risk", ascending=False)

    logger.info(f"Sector risk computed for {len(sector_risk)} sectors")
    return sector_risk


@timed
def run_risk_engine(df, graph_results, ml_results):
    """
    Full risk intelligence pipeline:
    1. Per-asset risk scores (decomposed)
    2. Market sentiment proxy
    3. Market regime assessment
    4. Volatility alerts
    5. Systemic risk indicator
    6. Sector risk breakdown
    """
    logger.info("═══ Starting Risk Intelligence Engine ═══")

    risk_df = compute_risk_scores(df, graph_results, ml_results)
    sentiment = compute_market_sentiment(df)
    market_regime = compute_market_regime(df)
    alerts = generate_volatility_alerts(risk_df)
    systemic_risk = compute_systemic_risk(graph_results, risk_df)
    sector_risk = compute_sector_risk(risk_df)

    risk_results = {
        "risk_df": risk_df,
        "sentiment": sentiment,
        "market_regime": market_regime,
        "alerts": alerts,
        "systemic_risk": systemic_risk,
        "sector_risk": sector_risk,
    }

    # Summary
    logger.info("═══ Risk Intelligence Summary ═══")
    logger.info(f"  Market Sentiment: {sentiment['label']} ({sentiment['score']})")
    logger.info(f"  Market Regime: {market_regime['regime']}")
    logger.info(f"  Systemic Risk: {systemic_risk['systemic_risk_score']}/100")
    logger.info(f"  Stocks with alerts: {len(alerts)}")
    logger.info(f"  Highest risk stock score: {risk_df['risk_score'].max():.1f}")

    return risk_results
