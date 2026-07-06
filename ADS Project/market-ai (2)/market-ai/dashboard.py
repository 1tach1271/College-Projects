"""
dashboard.py — Financial Intelligence Dashboard V3
====================================================
Clean, user-friendly Plotly Dash dashboard with working sidebar navigation,
stock deep-dive, and clear risk factor visualization.
"""

import logging
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State
import dash

import config

logger = logging.getLogger(__name__)

# ─── Load Pipeline Results ──────────────────────────────────────────────────

def load_results():
    """Load pipeline results from intermediate storage."""
    path = config.INTERMEDIATE_DIR / "pipeline_results.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline results not found at {path}. Run main.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

# ─── Color Palette ──────────────────────────────────────────────────────────

C = {
    "bg":        "#0b0b12",
    "sidebar":   "#0f0f1a",
    "card":      "rgba(20, 20, 35, 0.85)",
    "blue":      "#3b82f6",
    "cyan":      "#06b6d4",
    "purple":    "#8b5cf6",
    "green":     "#10b981",
    "red":       "#ef4444",
    "amber":     "#f59e0b",
    "pink":      "#ec4899",
    "text":      "#f1f5f9",
    "text2":     "#94a3b8",
    "muted":     "#475569",
    "border":    "rgba(255,255,255,0.07)",
}

TEMPLATE = "plotly_dark"

# ─── Small Helpers ──────────────────────────────────────────────────────────

def _card(children, pad="24px"):
    """Glassmorphism card wrapper."""
    return html.Div(children, style={
        "background": C["card"], "backdropFilter": "blur(14px)",
        "borderRadius": "20px", "padding": pad,
        "border": f"1px solid {C['border']}",
        "boxShadow": "0 6px 24px rgba(0,0,0,0.6)",
        "marginBottom": "20px",
    })

def _kpi(icon, label, value, color=None, sub=""):
    """KPI metric box."""
    color = color or C["blue"]
    return html.Div([
        html.Div(icon, style={"fontSize": "22px", "marginBottom": "8px"}),
        html.Div(label, style={
            "fontSize": "10px", "color": C["muted"], "textTransform": "uppercase",
            "letterSpacing": "1.8px", "fontWeight": "700", "marginBottom": "4px",
        }),
        html.Div(str(value), style={
            "fontSize": "26px", "fontWeight": "800", "color": color,
            "textShadow": f"0 0 18px {color}33",
        }),
        html.Div(sub, style={"fontSize": "11px", "color": C["text2"], "marginTop": "2px"}),
    ], style={
        "background": C["card"], "borderRadius": "18px", "padding": "20px",
        "border": f"1px solid {C['border']}", "textAlign": "center",
        "flex": "1", "minWidth": "150px",
    })

def _heading(title, sub=""):
    return html.Div([
        html.H3(title, style={
            "fontSize": "20px", "fontWeight": "800", "color": C["text"],
            "margin": "0", "letterSpacing": "-0.3px",
        }),
        html.Div(style={
            "height": "3px", "width": "50px", "margin": "8px 0",
            "background": f"linear-gradient(90deg, {C['blue']}, transparent)",
            "borderRadius": "2px",
        }),
        html.P(sub, style={"fontSize": "12px", "color": C["text2"], "margin": "0"}),
    ], style={"marginBottom": "20px"})

def _fig_defaults(fig, height=350):
    """Apply standard dark theme defaults to a figure."""
    fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=height,
        margin=dict(l=45, r=15, t=35, b=40),
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    return fig

# ─── Chart Builders ─────────────────────────────────────────────────────────

def chart_volatility_timeline(featured_df):
    if "vol_21d" not in featured_df.columns:
        return _fig_defaults(go.Figure(), 280)
    daily = featured_df.groupby("Date")["vol_21d"].median().reset_index()
    fig = go.Figure(go.Scatter(
        x=daily["Date"], y=daily["vol_21d"], mode="lines", fill="tozeroy",
        line=dict(color=C["blue"], width=2), fillcolor="rgba(59,130,246,0.08)",
    ))
    fig.update_layout(yaxis_title="Median Volatility")
    return _fig_defaults(fig, 280)

def chart_breadth(featured_df):
    if "market_breadth" not in featured_df.columns:
        return _fig_defaults(go.Figure(), 240)
    b = featured_df.groupby("Date")["market_breadth"].first().reset_index()
    fig = go.Figure(go.Scatter(
        x=b["Date"], y=b["market_breadth"], mode="lines",
        line=dict(color=C["green"], width=1.5),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=C["muted"], opacity=0.4)
    fig.update_layout(yaxis=dict(title="Breadth", tickformat=".0%"))
    return _fig_defaults(fig, 240)

def chart_sentiment_gauge(sentiment):
    score = sentiment.get("score", 50)
    label = sentiment.get("label", "Neutral")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": label, "font": {"size": 16, "color": C["text2"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "dtick": 25},
            "bar": {"color": C["cyan"]},
            "steps": [
                {"range": [0, 30], "color": "rgba(239,68,68,0.15)"},
                {"range": [30, 70], "color": "rgba(255,255,255,0.03)"},
                {"range": [70, 100], "color": "rgba(16,185,129,0.15)"},
            ],
        }
    ))
    return _fig_defaults(fig, 220)

def chart_sector_risk(risk_results):
    sector_risk = risk_results.get("sector_risk", pd.DataFrame())
    if sector_risk.empty:
        return _fig_defaults(go.Figure(), 350)
    sr = sector_risk.reset_index()
    col = sr.columns[0]
    fig = go.Figure(go.Bar(
        x=sr["mean_risk"], y=sr[col], orientation="h",
        marker=dict(color=sr["mean_risk"], colorscale="RdYlGn_r"),
        text=sr["mean_risk"].round(1), textposition="outside",
    ))
    fig.update_layout(margin=dict(l=140, r=40, t=20, b=30), xaxis_title="Mean Risk")
    return _fig_defaults(fig, 350)

def chart_stock_price_vol(featured_df, stock_code):
    sd = featured_df[featured_df["SecuritiesCode"] == stock_code].sort_values("Date")
    if sd.empty:
        return _fig_defaults(go.Figure(), 400)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.65, 0.35])
    fig.add_trace(go.Scatter(
        x=sd["Date"], y=sd["Close"], name="Price",
        line=dict(color=C["blue"], width=2),
    ), row=1, col=1)
    if "vol_21d" in sd.columns:
        fig.add_trace(go.Scatter(
            x=sd["Date"], y=sd["vol_21d"], name="21d Volatility",
            line=dict(color=C["amber"], width=1.5), fill="tozeroy",
            fillcolor="rgba(245,158,11,0.08)",
        ), row=2, col=1)
    fig.update_layout(showlegend=True, legend=dict(x=0, y=1.12, orientation="h"))
    fig.update_yaxes(title_text="Price (¥)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", row=2, col=1)
    return _fig_defaults(fig, 420)

def chart_risk_radar(risk_df, stock_code):
    row = risk_df[risk_df["SecuritiesCode"] == stock_code]
    if row.empty:
        return _fig_defaults(go.Figure(), 300)
    r = row.iloc[0]
    cats = ["Volatility", "Structural", "Tail"]
    vals = [r.get("factor_volatility", 0), r.get("factor_structural", 0), r.get("factor_tail", 0)]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(59,130,246,0.15)",
        line=dict(color=C["blue"], width=2),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.06)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return _fig_defaults(fig, 300)

def chart_spectral(featured_df, stock_code):
    try:
        series = featured_df[featured_df["SecuritiesCode"] == stock_code]["Close"].values
        if len(series) < 64:
            return _fig_defaults(go.Figure(), 200)
        series = (series - np.mean(series)) / (np.std(series) + 1e-9)
        fft_vals = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(len(series))
        fig = go.Figure(go.Scatter(
            x=freqs[1:], y=fft_vals[1:], mode="lines", fill="tozeroy",
            line=dict(color=C["purple"], width=2),
            fillcolor="rgba(139,92,246,0.08)",
        ))
        fig.update_layout(xaxis_title="Frequency", yaxis=dict(showticklabels=False))
        return _fig_defaults(fig, 200)
    except Exception:
        return _fig_defaults(go.Figure(), 200)

def chart_network(graph_results):
    edge_df = graph_results.get("edge_df", pd.DataFrame())
    if edge_df.empty:
        return _fig_defaults(go.Figure(), 500)
    nodes = list(set(edge_df["src"].tolist() + edge_df["dst"].tolist()))
    np.random.seed(42)
    pos = {n: (np.random.rand(), np.random.rand()) for n in nodes}
    ex, ey = [], []
    for _, row in edge_df.iterrows():
        x0, y0 = pos[row["src"]]; x1, y1 = pos[row["dst"]]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
        line=dict(width=0.3, color="rgba(255,255,255,0.08)"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers", marker=dict(size=5, color=C["cyan"], opacity=0.7),
        text=[str(n) for n in nodes], hoverinfo="text",
    ))
    fig.update_layout(showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return _fig_defaults(fig, 550)

def chart_correlation(graph_results):
    corr = graph_results.get("corr_matrix", pd.DataFrame())
    if corr.empty:
        return _fig_defaults(go.Figure(), 450)
    n = min(35, len(corr))
    c = corr.iloc[:n, :n]
    fig = go.Figure(go.Heatmap(
        z=c.values, x=[str(x) for x in c.columns], y=[str(x) for x in c.index],
        colorscale="RdBu_r", zmid=0,
    ))
    return _fig_defaults(fig, 450)

def chart_benchmark(benchmark_results):
    if not benchmark_results:
        return _fig_defaults(go.Figure(), 320)
    ops = [op.replace("_", " ").title() for op in benchmark_results.keys()]
    speeds = [r.get("speedup", 1) for r in benchmark_results.values()]
    fig = go.Figure(go.Bar(
        x=ops, y=speeds, marker_color=C["cyan"],
        text=[f"{s:.1f}×" for s in speeds], textposition="outside",
    ))
    fig.update_layout(yaxis_title="GPU Speedup Factor")
    return _fig_defaults(fig, 320)

def chart_ml(ml_results):
    figs = {}
    for task_name, task in ml_results.items():
        if not isinstance(task, dict):
            continue
        models, f1s = [], []
        for m, v in task.items():
            if isinstance(v, dict) and "f1_macro" in v:
                models.append(m.replace("_", " ").title())
                f1s.append(v["f1_macro"])
        if models:
            fig = go.Figure(go.Bar(
                x=models, y=f1s, marker_color=C["purple"],
                text=[f"{f:.3f}" for f in f1s], textposition="outside",
            ))
            fig.update_layout(yaxis=dict(range=[0, 1], title="F1 Macro"))
            figs[task_name] = _fig_defaults(fig, 300)
    return figs

# ─── App Factory ────────────────────────────────────────────────────────────

def create_app():
    """Create the Dash application V3 with working sidebar navigation."""
    results = load_results()
    featured_df = results["featured_df"]
    graph_results = results.get("graph_results", {})
    ml_results = results.get("ml_results", {})
    risk_results = results.get("risk_results", {})
    benchmark_results = results.get("benchmark_results", {})

    market_regime = risk_results.get("market_regime", {})
    sentiment = risk_results.get("sentiment", {})
    systemic_risk = risk_results.get("systemic_risk", {})
    risk_df = risk_results.get("risk_df", pd.DataFrame())
    stock_codes = sorted(featured_df["SecuritiesCode"].unique().tolist())

    # Pre-build static charts
    fig_vol_timeline = chart_volatility_timeline(featured_df)
    fig_breadth = chart_breadth(featured_df)
    fig_sentiment = chart_sentiment_gauge(sentiment)
    fig_sector = chart_sector_risk(risk_results)
    fig_network = chart_network(graph_results)
    fig_corr = chart_correlation(graph_results)
    fig_benchmark = chart_benchmark(benchmark_results)
    ml_chart_figs = chart_ml(ml_results)

    app = Dash(__name__, title="Market Intelligence Hub",
               suppress_callback_exceptions=True)

    # ─── Define sidebar nav items (label, icon, path) ──────────────
    NAV_ITEMS = [
        ("Overview",        "📊", "/"),
        ("Stock Deep-Dive", "🎯", "/stock"),
        ("Network & Graph", "🔗", "/network"),
        ("ML & Compute",    "⚡", "/ml"),
    ]

    # Sidebar links — STATIC in layout, only style changes via callback
    def _sidebar_links():
        links = []
        for label, icon, href in NAV_ITEMS:
            links.append(dcc.Link(
                html.Div([
                    html.Span(icon, style={"marginRight": "12px", "fontSize": "17px"}),
                    html.Span(label, style={"fontSize": "13px"}),
                ], style={"display": "flex", "alignItems": "center"}),
                href=href,
                id=f"nav-{href.strip('/') or 'home'}",
                style={
                    "display": "block", "padding": "13px 22px",
                    "textDecoration": "none", "color": C["text2"],
                    "borderLeft": "3px solid transparent",
                    "transition": "all 0.2s ease", "marginBottom": "2px",
                },
            ))
        return links

    # ─── Layout ──────────────────────────────────────────────────────
    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),

        # Sidebar
        html.Div([
            # Logo
            html.Div([
                html.Div("⚡ Market Intel", style={
                    "fontSize": "18px", "fontWeight": "800", "color": C["text"],
                }),
                html.Div("GPU-Accelerated v3", style={
                    "fontSize": "10px", "color": C["muted"], "marginTop": "2px",
                }),
            ], style={"padding": "28px 22px 20px", "borderBottom": f"1px solid {C['border']}"}),

            # Stock selector
            html.Div([
                html.Label("SELECT STOCK", style={
                    "fontSize": "9px", "fontWeight": "800", "color": C["muted"],
                    "letterSpacing": "1.5px", "display": "block", "marginBottom": "8px",
                }),
                dcc.Dropdown(
                    id="stock-selector",
                    options=[{"label": str(s), "value": s} for s in stock_codes],
                    value=stock_codes[0] if stock_codes else None,
                    placeholder="Search stock code...",
                    searchable=True, clearable=False,
                    style={"fontSize": "13px"},
                ),
            ], style={"padding": "18px 16px", "borderBottom": f"1px solid {C['border']}"}),

            # Nav links
            html.Div(_sidebar_links(), style={"marginTop": "8px", "flex": "1"}),

            # GPU status
            html.Div([
                html.Div([
                    html.Div(style={
                        "width": "7px", "height": "7px", "borderRadius": "50%",
                        "backgroundColor": C["green"], "marginRight": "8px",
                    }),
                    html.Span("RAPIDS GPU ACTIVE", style={
                        "fontSize": "9px", "fontWeight": "800", "color": C["green"],
                        "letterSpacing": "0.5px",
                    }),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
                html.Div("RTX 4050 · 6 GB VRAM", style={
                    "fontSize": "10px", "color": C["muted"],
                }),
            ], style={
                "padding": "18px 22px", "borderTop": f"1px solid {C['border']}",
                "backgroundColor": "rgba(0,0,0,0.25)",
            }),

        ], style={
            "position": "fixed", "left": 0, "top": 0, "bottom": 0, "width": "240px",
            "backgroundColor": C["sidebar"],
            "borderRight": f"1px solid {C['border']}",
            "display": "flex", "flexDirection": "column",
            "fontFamily": "'Inter', 'Segoe UI', sans-serif",
            "zIndex": "100", "overflowY": "auto",
        }),

        # Main content
        html.Div(id="page-content", style={
            "marginLeft": "240px", "padding": "30px 36px", "minHeight": "100vh",
            "fontFamily": "'Inter', 'Segoe UI', sans-serif",
        }),

    ], style={"backgroundColor": C["bg"], "minHeight": "100vh"})

    # ═══ Page Builders ═══════════════════════════════════════════════

    def page_overview():
        return html.Div([
            html.H2("Market Executive Overview", style={
                "color": C["text"], "fontWeight": "800", "marginBottom": "6px",
            }),
            html.P("Consolidated market health, regime signals, and risk distribution",
                   style={"color": C["text2"], "fontSize": "13px", "marginBottom": "28px"}),

            # KPI row
            html.Div([
                _kpi("🧠", "Sentiment", f"{sentiment.get('score', 50):.0f}",
                     C["cyan"], sentiment.get("label", "")),
                _kpi("📊", "Systemic Risk",
                     f"{systemic_risk.get('systemic_risk_score', 0):.0f}%",
                     C["red"], "Market Connectivity"),
                _kpi("🚦", "Regime", market_regime.get("regime", "N/A"),
                     C["amber"], f"Vol: {market_regime.get('market_vol', 0):.3f}"),
                _kpi("🌊", "Universe",
                     f"{len(featured_df['SecuritiesCode'].unique()):,}",
                     C["green"], "Active Stocks"),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "28px"}),

            # Charts
            html.Div([
                html.Div([
                    _heading("Sentiment Gauge", "Quantitative fear/greed proxy"),
                    _card([dcc.Graph(figure=fig_sentiment, config={"displayModeBar": False})]),
                    _heading("Market Breadth", "Fraction of stocks above moving average"),
                    _card([dcc.Graph(figure=fig_breadth, config={"displayModeBar": False})]),
                ], style={"flex": "1", "minWidth": "300px"}),
                html.Div([
                    _heading("Volatility Timeline", "Median 21-day volatility across the market"),
                    _card([dcc.Graph(figure=fig_vol_timeline, config={"displayModeBar": False})]),
                    _heading("Sector Risk Ranking", "Average risk score by sector"),
                    _card([dcc.Graph(figure=fig_sector, config={"displayModeBar": False})]),
                ], style={"flex": "2", "minWidth": "400px"}),
            ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}),
        ])

    def page_stock(stock_code):
        if not stock_code or stock_code not in stock_codes:
            stock_code = stock_codes[0] if stock_codes else None
        if stock_code is None:
            return html.Div("No stocks available.", style={"color": C["text"]})

        info = risk_df[risk_df["SecuritiesCode"] == stock_code]
        r = info.iloc[0] if not info.empty else {}

        return html.Div([
            # Header
            html.Div([
                html.Div([
                    html.H2(f"Stock: {stock_code}", style={
                        "color": C["text"], "margin": "0", "fontWeight": "800",
                    }),
                    html.P(str(r.get("Name", "—")), style={
                        "color": C["cyan"], "fontWeight": "700", "margin": "2px 0 0",
                    }),
                ]),
                html.Div([
                    html.Span("Sector: ", style={"color": C["muted"]}),
                    html.Span(str(r.get("17SectorName", "N/A")), style={"color": C["text"]}),
                ], style={
                    "background": "rgba(255,255,255,0.04)", "padding": "8px 16px",
                    "borderRadius": "10px",
                }),
            ], style={
                "display": "flex", "justifyContent": "space-between",
                "alignItems": "flex-end", "marginBottom": "24px",
            }),

            # KPI row
            html.Div([
                _kpi("🎯", "Overall Risk", f"{r.get('risk_score', 0):.0f}",
                     C["red"] if r.get("risk_score", 0) > 50 else C["green"], "Composite Score"),
                _kpi("📈", "Volatility", f"{r.get('factor_volatility', 0):.0f}",
                     C["amber"], "Regime Variance"),
                _kpi("🔗", "Structural", f"{r.get('factor_structural', 0):.0f}",
                     C["purple"], "Network Centrality"),
                _kpi("⚠️", "Tail Risk", f"{r.get('factor_tail', 0):.0f}",
                     C["pink"], "Drawdown + Spike"),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "28px"}),

            # Charts
            html.Div([
                html.Div([
                    _heading("Price & Volatility", "Historical price and 21-day rolling volatility"),
                    _card([dcc.Graph(figure=chart_stock_price_vol(featured_df, stock_code),
                                     config={"displayModeBar": False})]),
                ], style={"flex": "2", "minWidth": "400px"}),
                html.Div([
                    _heading("Risk DNA", "Decomposed risk factor breakdown"),
                    _card([dcc.Graph(figure=chart_risk_radar(risk_df, stock_code),
                                     config={"displayModeBar": False})]),
                    _heading("Spectral Signature", "Unique frequency energy profile"),
                    _card([dcc.Graph(figure=chart_spectral(featured_df, stock_code),
                                     config={"displayModeBar": False})]),
                ], style={"flex": "1", "minWidth": "300px"}),
            ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}),
        ])

    def page_network():
        return html.Div([
            html.H2("Network Intelligence", style={
                "color": C["text"], "fontWeight": "800", "marginBottom": "6px",
            }),
            html.P("Stock correlation network and systemic connectivity",
                   style={"color": C["text2"], "fontSize": "13px", "marginBottom": "28px"}),
            _heading("Correlation Network", "Each node is a stock; edges show strong correlations"),
            _card([dcc.Graph(figure=fig_network, config={"displayModeBar": False})]),
            _heading("Correlation Heatmap", "Pairwise correlation matrix (top stocks)"),
            _card([dcc.Graph(figure=fig_corr, config={"displayModeBar": False})]),
        ])

    def page_ml():
        chart_elements = []
        for name, fig in ml_chart_figs.items():
            chart_elements.append(
                _card([dcc.Graph(figure=fig, config={"displayModeBar": False})])
            )
        return html.Div([
            html.H2("ML Performance & GPU Benchmarks", style={
                "color": C["text"], "fontWeight": "800", "marginBottom": "6px",
            }),
            html.P("Model accuracy metrics and hardware acceleration results",
                   style={"color": C["text2"], "fontSize": "13px", "marginBottom": "28px"}),
            _heading("Model Performance", "F1 Macro scores for regime and spike prediction"),
            html.Div(chart_elements, style={
                "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px",
            }) if chart_elements else html.Div("No ML results available.", style={"color": C["text2"]}),
            html.Div(style={"height": "24px"}),
            _heading("GPU Acceleration", "Speedup factors: GPU (RAPIDS) vs CPU"),
            _card([dcc.Graph(figure=fig_benchmark, config={"displayModeBar": False})]),
        ])

    # ═══ Callbacks ═══════════════════════════════════════════════════

    # Active nav highlight
    nav_ids = [f"nav-{href.strip('/') or 'home'}" for _, _, href in NAV_ITEMS]

    @app.callback(
        [Output(nid, "style") for nid in nav_ids],
        [Input("url", "pathname")]
    )
    def highlight_nav(pathname):
        path = (pathname or "/").rstrip("/") or "/"
        styles = []
        for _, _, href in NAV_ITEMS:
            active = (path == href) or (path == "" and href == "/")
            styles.append({
                "display": "block", "padding": "13px 22px",
                "textDecoration": "none", "marginBottom": "2px",
                "transition": "all 0.2s ease",
                "color": C["text"] if active else C["text2"],
                "fontWeight": "600" if active else "400",
                "background": "rgba(59,130,246,0.1)" if active else "transparent",
                "borderLeft": f"3px solid {C['blue']}" if active else "3px solid transparent",
            })
        return styles

    # Page routing
    @app.callback(
        Output("page-content", "children"),
        [Input("url", "pathname"), Input("stock-selector", "value")]
    )
    def route(pathname, selected_stock):
        path = (pathname or "/").rstrip("/") or "/"
        ctx = dash.callback_context
        trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        # If stock selector changed, go to stock page
        if trigger == "stock-selector" and selected_stock:
            return page_stock(selected_stock)

        if path == "/stock":
            return page_stock(selected_stock)
        elif path == "/network":
            return page_network()
        elif path == "/ml":
            return page_ml()
        else:
            return page_overview()

    return app

# ─── Entry Point ────────────────────────────────────────────────────────────

def run_dashboard():
    """Launch the Intelligence Hub."""
    app = create_app()
    print(f"\n🚀 Market Intelligence Hub launching at http://localhost:{config.DASH_PORT}")
    print(f"   Press Ctrl+C to stop\n")
    app.run(host=config.DASH_HOST, port=config.DASH_PORT, debug=config.DASH_DEBUG)

if __name__ == "__main__":
    run_dashboard()
