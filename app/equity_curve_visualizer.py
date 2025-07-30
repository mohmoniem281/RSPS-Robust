import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, Any, List, Tuple
import sys
import os
from collections import Counter
from datetime import datetime

# Configure Plotly to output to a file by default (headless mode)
pio.renderers.default = "browser"

def load_equity_curve_data(file_path: str) -> Dict[str, Any]:
    """Load equity curve data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Equity curve file not found at {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in equity curve file {file_path}")
        sys.exit(1)

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = np.array(returns) - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation only)."""
    if len(returns) == 0:
        return 0.0
    excess_returns = np.array(returns) - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)  # Annualized

def calculate_omega_ratio(returns: List[float], threshold: float = 0.0) -> float:
    """Calculate Omega ratio."""
    if len(returns) == 0:
        return 0.0
    excess_returns = np.array(returns) - threshold
    gains = excess_returns[excess_returns > 0].sum()
    losses = -excess_returns[excess_returns < 0].sum()
    if losses == 0:
        return float('inf') if gains > 0 else 1.0
    return gains / losses

def calculate_max_drawdown(capital_values: List[float]) -> Tuple[float, float]:
    """Calculate maximum drawdown and its percentage."""
    if len(capital_values) < 2:
        return 0.0, 0.0
    
    peak = capital_values[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    
    for value in capital_values:
        if value > peak:
            peak = value
        dd = peak - value
        dd_pct = (dd / peak) * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
    
    return max_dd, max_dd_pct

def analyze_asset_distribution(equity_curve: List[Dict]) -> Dict[str, Any]:
    """Analyze asset distribution and performance including cash periods."""
    asset_periods = {}
    asset_performance = {}
    total_periods = 0
    cash_periods = 0
    
    for entry in equity_curve:
        # Skip initial entry and current_signal entries
        if entry.get('identifier') and 'current_signal' not in entry.get('identifier', '') and entry.get('identifier') != '2025-01-01T00-00-00Z':
            total_periods += 1
            
            # Check if actually in cash (position is null) vs just having a winner_asset
            if pd.isna(entry.get('position')) or entry.get('position') is None:
                cash_periods += 1
            else:
                asset = entry.get('winner_asset', 'unknown')
                pnl = entry.get('pnl', 0)
                
                if asset not in asset_periods:
                    asset_periods[asset] = 0
                    asset_performance[asset] = []
                
                asset_periods[asset] += 1
                asset_performance[asset].append(pnl)
    
    # Calculate performance metrics per asset
    asset_stats = {}
    for asset in asset_periods:
        pnls = asset_performance[asset]
        asset_stats[asset] = {
            'periods': asset_periods[asset],
            'allocation_percentage': (asset_periods[asset] / total_periods * 100) if total_periods > 0 else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'win_rate': (len([p for p in pnls if p > 0]) / len(pnls) * 100) if pnls else 0
        }
    
    # Add cash to stats
    if cash_periods > 0:
        asset_stats['CASH'] = {
            'periods': cash_periods,
            'allocation_percentage': (cash_periods / total_periods * 100) if total_periods > 0 else 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'win_rate': 0
        }
    
    # Sort by allocation percentage
    sorted_assets = sorted(asset_stats.items(), key=lambda x: x[1]['allocation_percentage'], reverse=True)
    
    return {
        'asset_stats': dict(sorted_assets),
        'total_periods': total_periods,
        'cash_periods': cash_periods
    }

def create_performance_metrics(equity_curve: List[Dict], initial_capital: float) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    if not equity_curve:
        return {}
    
    # Filter out current_signal entries
    valid_entries = [e for e in equity_curve if e.get('identifier') != 'current_signal']
    
    if not valid_entries:
        return {}
    
    # Extract returns and capital values
    returns = [e.get('return_pct', 0) / 100 for e in valid_entries if e.get('return_pct', 0) != 0]
    capital_values = [e.get('capital', initial_capital) for e in valid_entries]
    
    # Basic metrics
    final_capital = capital_values[-1] if capital_values else initial_capital
    total_return = final_capital - initial_capital
    total_return_pct = (total_return / initial_capital) * 100
    
    # Risk metrics
    max_dd, max_dd_pct = calculate_max_drawdown(capital_values)
    
    # Ratio metrics
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    omega = calculate_omega_ratio(returns)
    
    # Trading metrics
    profitable_trades = len([r for r in returns if r > 0])
    total_trades = len(returns)
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Time metrics
    start_date = valid_entries[0].get('identifier', 'Unknown')
    end_date = valid_entries[-1].get('identifier', 'Unknown')
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'omega_ratio': omega,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'start_date': start_date,
        'end_date': end_date
    }

def create_equity_curve_visualization(config: Dict[str, Any]):
    """Create a comprehensive dual equity curve dashboard."""
    
    # Get paths
    equity_curve_path = config.get("equity_curve_file", "data/equity_curve.json")
    output_path = config.get("equity_curve_visualization", "data/equity_curve.html")
    
    print(f"📊 Creating comprehensive equity curve dashboard...")
    print(f"   Input: {equity_curve_path}")
    print(f"   Output: {output_path}")
    
    # Load data
    data = load_equity_curve_data(equity_curve_path)
    
    # Handle both single and dual equity curves
    if 'equity_curve' in data:
        # Legacy single curve format
        df = pd.DataFrame(data['equity_curve'])
        df = df[df['identifier'] != 'current_signal']
        reference_df = df.copy()
        actual_df = df.copy()
        curve_type = "Single"
        kalman_data = None
    else:
        # New dual curve format
        reference_data = data.get('reference_curve', {}).get('equity_curve', [])
        actual_data = data.get('actual_curve', {}).get('equity_curve', [])
        kalman_data = data.get('kalman_filter_data')
        
        reference_df = pd.DataFrame(reference_data) if reference_data else pd.DataFrame()
        actual_df = pd.DataFrame(actual_data) if actual_data else pd.DataFrame()
        curve_type = "Dual (Kalman TPI)" if data.get('kalman_tpi_enabled') else "Dual"
    
    if reference_df.empty or actual_df.empty:
        print("❌ No data to visualize")
        return
    
    # Filter out current_signal entries and convert timestamps
    reference_df = reference_df[reference_df['identifier'] != 'current_signal'].copy()
    actual_df = actual_df[actual_df['identifier'] != 'current_signal'].copy()
    
    # Convert identifiers to datetime
    reference_df['identifier'] = pd.to_datetime(reference_df['identifier'], errors='coerce')
    actual_df['identifier'] = pd.to_datetime(actual_df['identifier'], errors='coerce')
    
    # Calculate performance metrics
    initial_capital = 10000  # Default starting capital
    ref_metrics = create_performance_metrics(reference_data if 'reference_curve' in data else data['equity_curve'], initial_capital)
    actual_metrics = create_performance_metrics(actual_data if 'actual_curve' in data else data['equity_curve'], initial_capital)
    
    # Analyze asset distribution
    ref_asset_analysis = analyze_asset_distribution(reference_data if 'reference_curve' in data else data['equity_curve'])
    actual_asset_analysis = analyze_asset_distribution(actual_data if 'actual_curve' in data else data['equity_curve'])
    
    # Create subplots: 2 equity curves + Kalman filter + asset distribution charts
    if kalman_data:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Reference Curve (Always Allocated)',
                'Actual Curve (TPI Controlled)', 
                'Kalman Filter Signal',
                'Combined View',
                'Reference Asset Distribution',
                'Actual Asset Distribution'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"type": "pie"}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
    else:
        # Original layout for non-Kalman systems
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Reference Curve (Always Allocated)',
                'Actual Curve (TPI Controlled)',
                'Reference Asset Distribution',
                'Actual Asset Distribution'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"type": "pie"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
    
    # Add reference equity curve
    fig.add_trace(
        go.Scatter(
            x=reference_df['identifier'],
            y=reference_df['capital'],
            mode='lines+markers',
            name='Reference Curve',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Reference Curve</b><br>' +
                         'Date: %{x}<br>' +
                         'Capital: $%{y:,.2f}<br>' +
                         'Asset: %{customdata[0]}<br>' +
                         'PnL: $%{customdata[1]:,.2f}<br>' +
                         '<extra></extra>',
            customdata=list(zip(
                reference_df['winner_asset'].fillna('cash'),
                reference_df['pnl'].fillna(0)
            ))
        ),
        row=1, col=1
    )
    
    # Add actual equity curve with detailed asset information
    # First, let's create a combined trace that shows all points
    # Determine actual allocation based on position (null = cash, regardless of winner_asset)
    def get_allocation_type(row):
        if pd.isna(row.get('position')) or row.get('position') is None:
            return 'CASH'
        elif row.get('winner_asset'):
            return row['winner_asset'].upper()
        else:
            return 'CASH'
    
    actual_df['allocation_type'] = actual_df.apply(get_allocation_type, axis=1)
    actual_df['asset_display'] = actual_df['allocation_type']
    
    # Create color mapping for assets
    color_map = {
        'CASH': '#F18F01',
        'BTC': '#FF6B35', 
        'ETH': '#A23B72'
    }
    
    # Add all points with detailed hover information
    fig.add_trace(
        go.Scatter(
            x=actual_df['identifier'],
            y=actual_df['capital'],
            mode='lines+markers',
            name='Actual Curve',
            line=dict(color='#A23B72', width=2),
            marker=dict(
                size=6,
                color=[color_map.get(alloc, '#666666') for alloc in actual_df['allocation_type']],
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>Actual Curve</b><br>' +
                         'Date: %{x}<br>' +
                         'Capital: $%{y:,.2f}<br>' +
                         'Allocation: %{customdata[0]}<br>' +
                         'PnL: $%{customdata[1]:,.2f}<br>' +
                         'Return: %{customdata[2]:.2f}%<br>' +
                         'Entry: $%{customdata[3]}<br>' +
                         'Exit: $%{customdata[4]}<br>' +
                         '<extra></extra>',
            customdata=list(zip(
                actual_df['allocation_type'],
                actual_df['pnl'].fillna(0),
                actual_df['return_pct'].fillna(0),
                actual_df['entry_price'].fillna('N/A'),
                actual_df['exit_price'].fillna('N/A')
            ))
        ),
        row=1, col=2
    )
    
    # Add Kalman filter traces if available
    if kalman_data and kalman_data['filtered_values']:
        # Create datetime index for Kalman data
        kalman_dates = pd.to_datetime(kalman_data['identifiers'], errors='coerce')
        
        # Kalman filter line
        fig.add_trace(
            go.Scatter(
                x=kalman_dates,
                y=kalman_data['filtered_values'],
                mode='lines',
                name='Kalman Filter',
                line=dict(color='#F18F01', width=3),
                hovertemplate='<b>Kalman Filter</b><br>' +
                             'Date: %{x}<br>' +
                             'Filtered Value: $%{y:,.2f}<br>' +
                             'Signal: %{customdata}<br>' +
                             '<extra></extra>',
                customdata=[f"{'UP' if s > 0 else 'DOWN' if s < 0 else 'NEUTRAL'}" for s in kalman_data['trend_signals']]
            ),
            row=2, col=1
        )
        
        # Add reference curve to Kalman subplot for comparison
        fig.add_trace(
            go.Scatter(
                x=reference_df['identifier'],
                y=reference_df['capital'],
                mode='lines',
                name='Reference (on Kalman)',
                line=dict(color='#2E86AB', width=1, dash='dot'),
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Combined view showing all three lines
        fig.add_trace(
            go.Scatter(
                x=reference_df['identifier'],
                y=reference_df['capital'],
                mode='lines',
                name='Reference',
                line=dict(color='#2E86AB', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=actual_df['identifier'],
                y=actual_df['capital'],
                mode='lines',
                name='Actual',
                line=dict(color='#A23B72', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=kalman_dates,
                y=kalman_data['filtered_values'],
                mode='lines',
                name='Kalman',
                line=dict(color='#F18F01', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        pie_row = 3
    else:
        pie_row = 2
    
    # Add asset distribution pie charts
    if ref_asset_analysis['asset_stats']:
        ref_assets = list(ref_asset_analysis['asset_stats'].keys())
        ref_periods = [ref_asset_analysis['asset_stats'][asset]['periods'] for asset in ref_assets]
        
        fig.add_trace(
            go.Pie(
                labels=ref_assets,
                values=ref_periods,
                name="Reference Assets",
                textinfo='label+percent',
                textposition='auto',
                marker_colors=['#2E86AB', '#A23B72', '#F18F01', '#F3A712', '#C73E1D']
            ),
            row=pie_row, col=1
        )
    
    if actual_asset_analysis['asset_stats']:
        actual_assets = list(actual_asset_analysis['asset_stats'].keys())
        actual_periods = [actual_asset_analysis['asset_stats'][asset]['periods'] for asset in actual_assets]
    
    fig.add_trace(
            go.Pie(
                labels=actual_assets,
                values=actual_periods,
                name="Actual Assets",
                textinfo='label+percent',
                textposition='auto',
                marker_colors=['#2E86AB', '#A23B72', '#F18F01', '#F3A712', '#C73E1D']
            ),
            row=pie_row, col=2
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>RSPS-Robust Strategy Dashboard</b><br><span style="font-size:14px">Performance Analysis: {curve_type} Equity Curve System</span>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1400 if kalman_data else 1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(title_text="Date", row=1, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Capital ($)", row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Capital ($)", row=1, col=2, showgrid=True, gridcolor='lightgray')
    
    # Add axes for Kalman filter plots if they exist
    if kalman_data:
        fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="Date", row=2, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Filtered Value ($)", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Capital ($)", row=2, col=2, showgrid=True, gridcolor='lightgray')
    
    # Create performance metrics HTML
    def format_metric(value, format_type="number"):
        if format_type == "currency":
            return f"${value:,.2f}"
        elif format_type == "percentage":
            return f"{value:.2f}%"
        elif format_type == "ratio":
            return f"{value:.3f}"
        else:
            return f"{value:,.0f}"
    
    def create_metrics_table(metrics, title):
        return f"""
        <div style="margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
            <h3 style="margin-top: 0; color: #2E86AB;">{title}</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div><strong>Initial Capital:</strong> {format_metric(metrics.get('initial_capital', 0), 'currency')}</div>
                <div><strong>Final Capital:</strong> {format_metric(metrics.get('final_capital', 0), 'currency')}</div>
                <div><strong>Total Return:</strong> {format_metric(metrics.get('total_return_pct', 0), 'percentage')}</div>
                <div><strong>Max Drawdown:</strong> {format_metric(metrics.get('max_drawdown_pct', 0), 'percentage')}</div>
                <div><strong>Sharpe Ratio:</strong> {format_metric(metrics.get('sharpe_ratio', 0), 'ratio')}</div>
                <div><strong>Sortino Ratio:</strong> {format_metric(metrics.get('sortino_ratio', 0), 'ratio')}</div>
                <div><strong>Omega Ratio:</strong> {format_metric(metrics.get('omega_ratio', 0), 'ratio')}</div>
                <div><strong>Total Trades:</strong> {format_metric(metrics.get('total_trades', 0))}</div>
                <div><strong>Win Rate:</strong> {format_metric(metrics.get('win_rate', 0), 'percentage')}</div>
            </div>
        </div>
        """
    
    def create_asset_table(asset_analysis, title):
        if not asset_analysis['asset_stats']:
            return f"<div><h3>{title}</h3><p>No asset data available</p></div>"
        
        rows = ""
        for asset, stats in asset_analysis['asset_stats'].items():
            # For cash, show different display format
            if asset == 'CASH':
                rows += f"""
                <tr style="background-color: #fff3cd;">
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>{asset}</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['periods']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['allocation_percentage']:.1f}%</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">$0.00</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">$0.00</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">-</td>
                </tr>
                """
            else:
                rows += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{asset.upper()}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['periods']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['allocation_percentage']:.1f}%</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">${stats['total_pnl']:,.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">${stats['avg_pnl']:,.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['win_rate']:.1f}%</td>
                </tr>
                """
        
        return f"""
        <div style="margin: 20px 0;">
            <h3 style="color: #2E86AB;">{title}</h3>
            <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
                <thead>
                    <tr style="background-color: #f5f5f5;">
                        <th style="padding: 10px; border-bottom: 2px solid #ddd; text-align: left;">Asset</th>
                        <th style="padding: 10px; border-bottom: 2px solid #ddd; text-align: left;">Periods</th>
                        <th style="padding: 10px; border-bottom: 2px solid #ddd; text-align: left;">% Allocation</th>
                        <th style="padding: 10px; border-bottom: 2px solid #ddd; text-align: left;">Total PnL</th>
                        <th style="padding: 10px; border-bottom: 2px solid #ddd; text-align: left;">Avg PnL</th>
                        <th style="padding: 10px; border-bottom: 2px solid #ddd; text-align: left;">Win Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <p style="margin-top: 10px; color: #666; font-style: italic;">
                Cash Periods: {asset_analysis['cash_periods']} | Total Periods: {asset_analysis['total_periods']}
            </p>
        </div>
        """
    
    # Generate the complete HTML
    metrics_html = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px; background-color: white;">
        <h1 style="text-align: center; color: #2E86AB; margin-bottom: 30px;">
            🚀 RSPS-Robust Strategy Performance Dashboard
        </h1>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
            {create_metrics_table(ref_metrics, "📊 Reference Curve Metrics (Always Allocated)")}
            {create_metrics_table(actual_metrics, "🎯 Actual Curve Metrics (TPI Controlled)")}
        </div>
        
        <div style="margin: 30px 0;">
            <h2 style="color: #2E86AB; text-align: center;">Asset Performance Analysis</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                <div>{create_asset_table(ref_asset_analysis, "Reference Curve Asset Distribution")}</div>
                <div>{create_asset_table(actual_asset_analysis, "Actual Curve Asset Distribution")}</div>
            </div>
        </div>
        
        <div style="margin: 30px 0; padding: 20px; background-color: #e8f4f8; border-radius: 10px;">
            <h3 style="color: #2E86AB; margin-top: 0;">🎯 Strategy Performance Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; font-size: 16px;">
                <div><strong>Strategy Type:</strong> {curve_type} Equity Curve System</div>
                <div><strong>Analysis Period:</strong> {ref_metrics.get('start_date', 'N/A')} to {ref_metrics.get('end_date', 'N/A')}</div>
                <div><strong>TPI Performance Boost:</strong> {(actual_metrics.get('total_return_pct', 0) / ref_metrics.get('total_return_pct', 1) if ref_metrics.get('total_return_pct', 0) > 0 else 0):.1f}x</div>
                <div><strong>Capital Multiplier:</strong> {(actual_metrics.get('final_capital', 0) / actual_metrics.get('initial_capital', 1)):.2f}x</div>
            </div>
        </div>
    </div>
    """
    
    # Combine the plot with metrics
    html_string = fig.to_html(include_plotlyjs=True)
    
    # Insert metrics before the closing body tag
    html_string = html_string.replace('</body>', f'{metrics_html}</body>')
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(html_string)
    
    print(f"✅ Comprehensive equity curve dashboard saved to {output_path}")

if __name__ == "__main__":
    # Load config
    with open("app/config.json", "r") as f:
        config = json.load(f)
    
    create_equity_curve_visualization(config)
