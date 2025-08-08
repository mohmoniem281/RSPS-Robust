import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

def load_equity_curve_data(file_path: str) -> Dict:
    """Load equity curve data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading equity curve data from {file_path}: {e}")
        return {"curve": []}

def calculate_statistics(equity_curve: List[Dict]) -> Dict:
    """Get statistics directly from equity curve file summary instead of recalculating"""
    # This function is kept for backward compatibility but now returns empty dict
    # The actual statistics will be loaded from the file summary
    return {}

def load_statistics_from_file(file_path: str) -> Dict:
    """Load statistics directly from equity curve file summary"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get("summary", {})
    except Exception as e:
        print(f"Error loading statistics from {file_path}: {e}")
        return {}

def create_equity_curve_dashboard(config: Dict[str, Any]):
    """Create a comprehensive dashboard comparing both equity curves"""
    
    # Load both equity curves
    non_filtered_data = load_equity_curve_data(config["equity_curve_file"])
    filtered_data = load_equity_curve_data(config["equity_curve_file_filtered"])
    
    non_filtered_curve = non_filtered_data.get("curve", [])
    filtered_curve = filtered_data.get("curve", [])
    
    if not non_filtered_curve and not filtered_curve:
        print("No equity curve data found!")
        return
    
    # Convert to DataFrames
    df_non_filtered = pd.DataFrame(non_filtered_curve)
    df_filtered = pd.DataFrame(filtered_curve)
    
    # Treat identifiers as categorical variables (not datetime)
    if not df_non_filtered.empty:
        df_non_filtered['identifier'] = df_non_filtered['identifier'].astype('category')
    if not df_filtered.empty:
        df_filtered['identifier'] = df_filtered['identifier'].astype('category')
    
    # Calculate statistics
    stats_non_filtered = load_statistics_from_file(config["equity_curve_file"])
    stats_filtered = load_statistics_from_file(config["equity_curve_file_filtered"])
    
    # Create the dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Equity Curves Comparison',
            'Daily Returns Comparison',
            'Drawdown Analysis',
            'Performance Metrics',
            'Asset Allocation',
            'Risk Metrics'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "table"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1. Equity Curves Comparison
    if not df_non_filtered.empty:
        fig.add_trace(
            go.Scatter(
                x=df_non_filtered['identifier'],
                y=df_non_filtered['capital'],
                mode='lines',
                name='Non-Filtered',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Identifier:</b> %{x}<br><b>Capital:</b> $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if not df_filtered.empty:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['identifier'],
                y=df_filtered['capital'],
                mode='lines',
                name='Filtered',
                line=dict(color='red', width=2),
                hovertemplate='<b>Identifier:</b> %{x}<br><b>Capital:</b> $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Daily Returns Comparison
    if not df_non_filtered.empty:
        returns_non_filtered = df_non_filtered['return_pct'].values
        fig.add_trace(
            go.Bar(
                x=df_non_filtered['identifier'],
                y=returns_non_filtered,
                name='Non-Filtered Returns',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
    
    if not df_filtered.empty:
        returns_filtered = df_filtered['return_pct'].values
        fig.add_trace(
            go.Bar(
                x=df_filtered['identifier'],
                y=returns_filtered,
                name='Filtered Returns',
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=1, col=2
        )
    
    # 3. Drawdown Analysis
    if not df_non_filtered.empty:
        capitals = df_non_filtered['capital'].values
        peak = capitals[0]
        drawdowns = []
        
        for capital in capitals:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            drawdowns.append(drawdown)
        
        fig.add_trace(
            go.Scatter(
                x=df_non_filtered['identifier'],
                y=drawdowns,
                mode='lines',
                name='Non-Filtered Drawdown',
                line=dict(color='orange', width=2),
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.3)'
            ),
            row=2, col=1
        )
    
    # 4. Performance Metrics Table
    metrics_data = {
        'Metric': [
            'Total Return (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Omega Ratio',
            'Max Drawdown (%)',
            'Annualized Volatility (%)',
            'Annualized Return (%)',
            'Number of Trades'
        ],
        'Non-Filtered': [
            f"{stats_non_filtered.get('total_return', 0):.2f}",
            f"{stats_non_filtered.get('sharpe_ratio', 0):.2f}",
            f"{stats_non_filtered.get('sortino_ratio', 0):.2f}",
            f"{stats_non_filtered.get('omega_ratio', 0):.2f}",
            f"{stats_non_filtered.get('max_drawdown', 0):.2f}",
            f"{stats_non_filtered.get('annualized_volatility', 0):.2f}",
            f"{stats_non_filtered.get('annualized_return', 0):.2f}",
            f"{stats_non_filtered.get('number_of_trades', 0)}"
        ],
        'Filtered': [
            f"{stats_filtered.get('total_return', 0):.2f}",
            f"{stats_filtered.get('sharpe_ratio', 0):.2f}",
            f"{stats_filtered.get('sortino_ratio', 0):.2f}",
            f"{stats_filtered.get('omega_ratio', 0):.2f}",
            f"{stats_filtered.get('max_drawdown', 0):.2f}",
            f"{stats_filtered.get('annualized_volatility', 0):.2f}",
            f"{stats_filtered.get('annualized_return', 0):.2f}",
            f"{stats_filtered.get('number_of_trades', 0)}"
        ]
    }
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(metrics_data.keys()),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[metrics_data[key] for key in metrics_data.keys()],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        ),
        row=2, col=2
    )
    
    # 5. Asset Allocation (Winner Assets)
    if not df_non_filtered.empty:
        asset_counts = df_non_filtered['winner_asset'].value_counts()
        fig.add_trace(
            go.Bar(
                x=asset_counts.index,
                y=asset_counts.values,
                name='Non-Filtered Assets',
                marker_color='steelblue'
            ),
            row=3, col=1
        )
    
    # 6. Risk Metrics Comparison
    risk_metrics = ['Total Return (%)', 'Max Drawdown (%)', 'Annualized Volatility (%)', 'Sharpe Ratio']
    non_filtered_values = [
        stats_non_filtered.get('total_return', 0),
        stats_non_filtered.get('max_drawdown', 0),
        stats_non_filtered.get('annualized_volatility', 0),
        stats_non_filtered.get('sharpe_ratio', 0)
    ]
    filtered_values = [
        stats_filtered.get('total_return', 0),
        stats_filtered.get('max_drawdown', 0),
        stats_filtered.get('annualized_volatility', 0),
        stats_filtered.get('sharpe_ratio', 0)
    ]
    
    fig.add_trace(
        go.Bar(
            x=risk_metrics,
            y=non_filtered_values,
            name='Non-Filtered',
            marker_color='blue',
            opacity=0.7
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=risk_metrics,
            y=filtered_values,
            name='Filtered',
            marker_color='red',
            opacity=0.7
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Equity Curve Dashboard - Non-Filtered vs Filtered Performance',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1200,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Identifier", row=1, col=1)
    fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
    fig.update_xaxes(title_text="Identifier", row=1, col=2)
    fig.update_yaxes(title_text="Daily Return (%)", row=1, col=2)
    fig.update_xaxes(title_text="Identifier", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Asset", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="Metric", row=3, col=2)
    fig.update_yaxes(title_text="Value", row=3, col=2)
    
    # Save the combined dashboard to a single file
    output_path = config.get("equity_curve_visualization", "data/equity_curve.html")
    fig.write_html(output_path)
    print(f"✅ Combined equity curve dashboard saved to: {output_path}")
    
    return fig

def create_equity_curve_visualization(config: Dict[str, Any]):
    """Legacy function for backward compatibility"""
    return create_equity_curve_dashboard(config)

def create_filtered_equity_curve_visualization(config: Dict[str, Any]):
    """Legacy function for backward compatibility"""
    return create_equity_curve_dashboard(config)

if __name__ == "__main__":
    # Load config and create dashboard
    with open("app/config/config.json", "r") as f:
        config = json.load(f)
    
    create_equity_curve_dashboard(config)
