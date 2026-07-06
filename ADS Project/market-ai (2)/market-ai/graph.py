"""
graph.py — Module 5: Correlation Graph Modeling (FIXED & STABLE)
"""

import logging
import numpy as np
import pandas as pd

import config
from gpu_utils import is_gpu_available, gpu_context
from utils import timed

logger = logging.getLogger(__name__)


# =========================================================
# 🔥 SAFE CONVERSION HELPER (CRITICAL FIX)
# =========================================================
def safe_to_pandas(df, name="DataFrame"):
    """Safe GPU → CPU conversion without CUDA context issues."""
    try:
        if hasattr(df, "to_pandas"):
            logger.info(f"[SAFE CONVERT] {name}: cuDF → pandas via Arrow")
            return df.to_pandas()
        return df

    except Exception as e:
        logger.warning(f"[FALLBACK] {name}: conversion failed ({e}), returning as-is")
        return df


# =========================================================

@timed
def select_liquid_stocks(df, top_n=None):
    top_n = top_n or config.GRAPH_TOP_N_STOCKS
    avg_vol = df.groupby("SecuritiesCode")["Volume"].mean().sort_values(ascending=False)
    selected = avg_vol.head(top_n).index.tolist()
    logger.info(f"Selected top {len(selected)} most liquid stocks for graph analysis")
    return selected


@timed
def compute_correlation_matrix(df, stock_list, window=None):
    window = window or config.CORR_ROLLING_WINDOW

    subset = df[df["SecuritiesCode"].isin(stock_list)].copy()

    returns_pivot = subset.pivot_table(
        index="Date",
        columns="SecuritiesCode",
        values="log_return"
    ).dropna(axis=1, thresh=window // 2)

    returns_window = returns_pivot.tail(window)

    if is_gpu_available():
        import cupy as cp
        with gpu_context("CorrelationMatrix"):
            gpu_data = cp.asarray(returns_window.values, dtype=cp.float64)

            col_means = cp.nanmean(gpu_data, axis=0)
            nan_mask = cp.isnan(gpu_data)

            for j in range(gpu_data.shape[1]):
                gpu_data[nan_mask[:, j], j] = col_means[j]

            corr_matrix = cp.corrcoef(gpu_data.T)
            corr_np = cp.asnumpy(corr_matrix)
    else:
        corr_np = returns_window.corr().values

    corr_np = np.nan_to_num(corr_np, nan=0.0)

    corr_df = pd.DataFrame(
        corr_np,
        index=returns_window.columns,
        columns=returns_window.columns,
    )

    logger.info(f"Correlation matrix: {corr_df.shape[0]}×{corr_df.shape[1]} stocks")
    return corr_df


@timed
def build_graph_from_correlation(corr_df, threshold=None):
    threshold = threshold or config.CORR_EDGE_THRESHOLD
    stocks = corr_df.index.tolist()

    edges = []
    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            weight = abs(corr_df.iloc[i, j])
            if weight > threshold:
                edges.append({
                    "src": stocks[i],
                    "dst": stocks[j],
                    "weight": float(corr_df.iloc[i, j]),
                    "abs_weight": float(weight),
                })

    edge_df = pd.DataFrame(edges)
    logger.info(f"Graph: {len(stocks)} nodes, {len(edge_df)} edges (threshold={threshold})")
    return edge_df, stocks


@timed
def run_community_detection(edge_df, stocks):
    community_map = {}

    if edge_df.empty:
        logger.warning("No edges in graph — cannot run community detection")
        return {s: 0 for s in stocks}

    if is_gpu_available():
        import cudf
        import cugraph

        with gpu_context("CommunityDetection"):

            stock_to_id = {s: i for i, s in enumerate(stocks)}
            id_to_stock = {i: s for s, i in stock_to_id.items()}

            g_edges = cudf.DataFrame({
                "src": edge_df["src"].map(stock_to_id).astype("int32"),
                "dst": edge_df["dst"].map(stock_to_id).astype("int32"),
                "weight": edge_df["abs_weight"].astype("float32"),
            })

            G = cugraph.Graph()
            G.from_cudf_edgelist(g_edges, source="src", destination="dst", edge_attr="weight")

            parts, modularity = cugraph.louvain(G)

            # 🔥 FIXED HERE
            parts_pd = safe_to_pandas(parts, "Louvain")

            # NOW SAFE
            for _, row in parts_pd.iterrows():  
                stock = id_to_stock.get(int(row["vertex"]), None)
                if stock is not None:
                    community_map[stock] = int(row["partition"])

            logger.info(f"Louvain communities: {len(set(community_map.values()))} "
                        f"(modularity: {modularity:.4f})")

    else:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities

        G = nx.Graph()
        for _, row in edge_df.iterrows():
            G.add_edge(row["src"], row["dst"], weight=row["abs_weight"])

        communities = list(greedy_modularity_communities(G, weight="weight"))
        for cid, comm in enumerate(communities):
            for stock in comm:
                community_map[stock] = cid

        logger.info(f"NetworkX communities: {len(communities)}")

    return community_map


@timed
def run_centrality_analysis(edge_df, stocks):

    if edge_df.empty:
        return pd.DataFrame({
            "SecuritiesCode": stocks,
            "pagerank": 0.0,
            "degree_centrality": 0.0
        })

    if is_gpu_available():
        import cudf
        import cugraph

        with gpu_context("CentralityAnalysis"):

            stock_to_id = {s: i for i, s in enumerate(stocks)}
            id_to_stock = {i: s for s, i in stock_to_id.items()}

            g_edges = cudf.DataFrame({
                "src": edge_df["src"].map(stock_to_id).astype("int32"),
                "dst": edge_df["dst"].map(stock_to_id).astype("int32"),
                "weight": edge_df["abs_weight"].astype("float32"),
            })

            G = cugraph.Graph()
            G.from_cudf_edgelist(g_edges, source="src", destination="dst", edge_attr="weight")

            pagerank_df = cugraph.pagerank(G)
            pagerank_pd = safe_to_pandas(pagerank_df, "PageRank")

            degree_gdf = G.degrees()

            # SAFE: convert using Arrow (no numba / no CUDA context issue)
            degree_df = degree_gdf.to_pandas()

            centrality = pd.DataFrame({"SecuritiesCode": stocks})

            pr_map = {
                id_to_stock[int(r["vertex"])]: r["pagerank"]
                for _, r in pagerank_pd.iterrows()
                if int(r["vertex"]) in id_to_stock
            }

            centrality["pagerank"] = centrality["SecuritiesCode"].map(pr_map).fillna(0)

            deg_map = {
                id_to_stock[int(r["vertex"])]: r["in_degree"] + r["out_degree"]
                for _, r in degree_df.iterrows()
                if int(r["vertex"]) in id_to_stock
            }

            max_deg = max(deg_map.values()) if deg_map else 1

            centrality["degree_centrality"] = centrality["SecuritiesCode"].map(
                {k: v / max_deg for k, v in deg_map.items()}
            ).fillna(0)

    else:
        import networkx as nx

        G = nx.Graph()
        for _, row in edge_df.iterrows():
            G.add_edge(row["src"], row["dst"], weight=row["abs_weight"])

        pagerank = nx.pagerank(G, weight="weight")
        degree_cent = nx.degree_centrality(G)

        centrality = pd.DataFrame({"SecuritiesCode": stocks})
        centrality["pagerank"] = centrality["SecuritiesCode"].map(pagerank).fillna(0)
        centrality["degree_centrality"] = centrality["SecuritiesCode"].map(degree_cent).fillna(0)

    logger.info(f"Centrality computed for {len(centrality)} stocks")
    return centrality


@timed
def compute_graph_density(edge_df, n_nodes):
    if n_nodes <= 1:
        return 0.0

    max_edges = n_nodes * (n_nodes - 1) / 2
    density = len(edge_df) / max_edges

    logger.info(f"Graph density: {density:.4f}")
    return density


@timed
def run_graph_modeling(df):

    logger.info("═══ Starting Correlation Graph Modeling ═══")

    liquid_stocks = select_liquid_stocks(df)
    corr_df = compute_correlation_matrix(df, liquid_stocks)
    edge_df, stocks = build_graph_from_correlation(corr_df)

    community_map = run_community_detection(edge_df, stocks)
    centrality_df = run_centrality_analysis(edge_df, stocks)
    density = compute_graph_density(edge_df, len(stocks))

    centrality_df["community"] = centrality_df["SecuritiesCode"].map(community_map)

    graph_results = {
        "corr_matrix": corr_df,
        "edge_df": edge_df,
        "community_map": community_map,
        "centrality_df": centrality_df,
        "graph_density": density,
        "liquid_stocks": liquid_stocks,
        "n_communities": len(set(community_map.values())),
    }

    top_nodes = centrality_df.nlargest(10, "pagerank")
    logger.info(f"Top systemic nodes:\n{top_nodes.to_string()}")

    return graph_results