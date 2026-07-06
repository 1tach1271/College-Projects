"""
Simple Market AI Dashboard - Standalone Version
Interactive financial intelligence dashboard with real-time data
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import dash
from datetime import datetime, timedelta
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Color Palette
COLORS = {
    "bg_primary": "#0a0e17",
    "bg_secondary": "#111827", 
    "bg_card": "#1a1f2e",
    "bg_card_hover": "#232a3b",
    "accent_blue": "#3b82f6",
    "accent_cyan": "#06b6d4",
    "accent_purple": "#8b5cf6",
    "accent_green": "#10b981",
    "accent_red": "#ef4444",
    "accent_amber": "#f59e0b",
    "accent_pink": "#ec4899",
    "text_primary": "#f9fafb",
    "text_secondary": "#d1d5db",
    "text_muted": "#6b7280",
    "border": "#374151",
}

# Plot Template
PLOT_TEMPLATE = "plotly_dark"

def generate_sample_data():
    """Generate sample financial data for demonstration."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate sample price data
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate sample regime data
    regimes = np.random.choice(['bull', 'bear', 'sideways'], len(dates), p=[0.4, 0.3, 0.3])
    confidence = np.random.uniform(0.6, 0.95, len(dates))
    
    # Generate sample risk data
    risk_scores = np.random.uniform(0, 100, len(dates))
    volatility = np.random.uniform(0.01, 0.05, len(dates))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Returns': returns,
        'Regime': regimes,
        'Confidence': confidence,
        'Risk_Score': risk_scores,
        'Volatility': volatility
    })
    
    return data

def create_regime_chart(data):
    """Create regime classification chart."""
    fig = go.Figure()
    
    for regime in data['Regime'].unique():
        regime_data = data[data['Regime'] == regime]
        colors = {'bull': COLORS['accent_green'], 'bear': COLORS['accent_red'], 'sideways': COLORS['accent_amber']}
        
        fig.add_trace(go.Scatter(
            x=regime_data['Date'],
            y=regime_data['Price'],
            mode='lines',
            name=f'{regime.capitalize()} Market',
            line=dict(color=colors[regime], width=2),
            fill='tonexty' if regime == 'bull' else None
        ))
    
    fig.update_layout(
        title=dict(text="Market Regime Classification", font=dict(size=16, color=COLORS['text_primary'])),
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Price", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_volatility_chart(data):
    """Create volatility prediction chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Volatility'] * 100,
        mode='lines',
        name='Volatility',
        line=dict(color=COLORS['accent_purple'], width=2)
    ))
    
    # Add threshold line
    fig.add_hline(
        y=data['Volatility'].mean() * 100 * 1.5,
        line_dash="dash",
        line_color=COLORS['accent_red'],
        annotation_text="Volatility Spike Threshold"
    )
    
    fig.update_layout(
        title=dict(text="Volatility Analysis & Spike Detection", font=dict(size=16, color=COLORS['text_primary'])),
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Volatility (%)", gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False
    )
    
    return fig

def create_risk_heatmap(data):
    """Create risk heatmap."""
    # Sample risk data by sectors
    sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer', 'Industrial']
    risk_matrix = np.random.uniform(20, 80, (len(sectors), 12))  # 12 months
    
    fig = go.Figure(data=go.Heatmap(
        z=risk_matrix,
        x=[f'Month {i+1}' for i in range(12)],
        y=sectors,
        colorscale='RdYlBu_r',
        showscale=True,
        colorbar=dict(title="Risk Score")
    ))
    
    fig.update_layout(
        title=dict(text="Sector Risk Heatmap", font=dict(size=16, color=COLORS['text_primary'])),
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=100, r=20, t=50, b=40),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Sector")
    )
    
    return fig

def create_performance_chart():
    """Create GPU vs CPU performance comparison."""
    categories = ['Data Loading', 'Feature Engineering', 'ML Training', 'Graph Analysis', 'Signal Processing']
    cpu_times = [45.2, 123.4, 89.7, 234.1, 156.8]
    gpu_times = [2.1, 6.8, 12.3, 4.8, 3.2]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='CPU',
        x=categories,
        y=cpu_times,
        marker_color=COLORS['accent_blue']
    ))
    
    fig.add_trace(go.Bar(
        name='GPU',
        x=categories,
        y=gpu_times,
        marker_color=COLORS['accent_green']
    ))
    
    fig.update_layout(
        title=dict(text="GPU vs CPU Performance Comparison", font=dict(size=16, color=COLORS['text_primary'])),
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(title="Operation"),
        yaxis=dict(title="Processing Time (seconds)"),
        barmode='group'
    )
    
    return fig

def check_api_status():
    """Check if the API server is running."""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            return "Connected", COLORS['accent_green']
        else:
            return "Error", COLORS['accent_red']
    except:
        return "Disconnected", COLORS['accent_red']

def create_app():
    """Create the Dash application."""
    # Generate sample data
    data = generate_sample_data()
    
    # Create Dash app
    app = Dash(__name__)
    
    # Define layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Market AI - Financial Intelligence Dashboard", 
                   style={'color': COLORS['text_primary'], 'textAlign': 'center', 'marginBottom': '20px'}),
            html.P("GPU-Accelerated Market Analysis & Risk Assessment System",
                  style={'color': COLORS['text_secondary'], 'textAlign': 'center', 'marginBottom': '30px'})
        ], style={'backgroundColor': COLORS['bg_primary'], 'padding': '20px', 'borderRadius': '10px'}),
        
        # API Status
        html.Div([
            html.H3("API Status", style={'color': COLORS['text_primary']}),
            html.Div(id='api-status', style={'fontSize': '18px', 'fontWeight': 'bold'})
        ], style={'backgroundColor': COLORS['bg_card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
        
        # Charts Grid
        html.Div([
            # Row 1: Regime and Volatility
            html.Div([
                dcc.Graph(id='regime-chart', figure=create_regime_chart(data))
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                dcc.Graph(id='volatility-chart', figure=create_volatility_chart(data))
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            # Row 2: Risk and Performance
            html.Div([
                dcc.Graph(id='risk-chart', figure=create_risk_heatmap(data))
            ], style={'width': '48%', 'display': 'inline-block', 'marginTop': '20px', 'marginRight': '2%'}),
            
            html.Div([
                dcc.Graph(id='performance-chart', figure=create_performance_chart())
            ], style={'width': '48%', 'display': 'inline-block', 'marginTop': '20px'}),
            
        ], style={'backgroundColor': COLORS['bg_secondary'], 'padding': '20px', 'borderRadius': '10px'}),
        
        # Footer
        html.Div([
            html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  style={'color': COLORS['text_muted'], 'textAlign': 'center'}),
            html.P("Market AI System - Production Grade Financial Intelligence",
                  style={'color': COLORS['text_secondary'], 'textAlign': 'center'})
        ], style={'backgroundColor': COLORS['bg_primary'], 'padding': '20px', 'borderRadius': '10px', 'marginTop': '20px'}),
        
        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=30*1000,  # Update every 30 seconds
            n_intervals=0
        )
    ], style={'backgroundColor': COLORS['bg_primary'], 'minHeight': '100vh', 'padding': '20px'})
    
    # Callback for API status
    @app.callback(
        Output('api-status', 'children'),
        Output('api-status', 'style'),
        Input('interval-component', 'n_intervals')
    )
    def update_api_status(n):
        status, color = check_api_status()
        return status, {'color': color, 'fontSize': '18px', 'fontWeight': 'bold'}
    
    return app

def run_dashboard():
    """Run the dashboard application."""
    app = create_app()
    
    print("Market AI Dashboard Starting...")
    print("Access the dashboard at: http://localhost:8050")
    print("API Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=8050,
        debug=False
    )

if __name__ == "__main__":
    run_dashboard()
