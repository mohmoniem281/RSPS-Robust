import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import os

def load_equity_curve_data(equity_curve_path):
    """Load equity curve data from JSON file"""
    with open(equity_curve_path, 'r') as f:
        return json.load(f)

def parse_date(identifier):
    """Parse date from identifier string"""
    try:
        return datetime.strptime(identifier, '%Y-%m-%d')
    except:
        # For non-date identifiers like "hello", create a fallback date
        # Use the last valid date + 1 day, or a default date
        return None

def create_equity_curve_visualization(config):
    """Create comprehensive equity curve visualization using Plotly"""
    
    # Get paths from config
    equity_curve_path = config['equity_curve_file']
    output_path = config['equity_curve_visualization']
    
    data = load_equity_curve_data(equity_curve_path)
    
    # Extract summary statistics from the JSON data
    summary_stats = data.get('summary_statistics', {})
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data['equity_curve'])
    
    # Use identifiers as labels directly, no date parsing
    df['label'] = df['identifier']
    
    # Sort by the order they appear in the original data
    df = df.reset_index(drop=True)
    
    # Calculate additional metrics
    df['cumulative_return_pct'] = df['return_pct'].cumsum()
    df['drawdown'] = (df['capital'] - df['capital'].expanding().max()) / df['capital'].expanding().max() * 100
    
    # Create subplots - 2 columns: charts on left, summary table on right
    fig = make_subplots(
        rows=8, cols=2,
        subplot_titles=(
            'Equity Curve', 'Summary Statistics',
            'Daily Returns', '',
            'Cumulative Returns', '',
            'Drawdown', '',
            'Position Distribution', '',
            'Asset Performance', '',
            'PnL Distribution', '',
            'Monthly Returns', ''
        ),
        specs=[
            [{"secondary_y": False}, {"type": "table"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"type": "domain"}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}]
        ],
        column_widths=[0.7, 0.3],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    # 1. Equity Curve
    fig.add_trace(
        go.Scatter(
            x=df['label'],
            y=df['capital'],
            mode='lines+markers',
            name='Capital',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Identifier:</b> %{x}<br>' +
                         '<b>Capital:</b> $%{y:,.2f}<br>' +
                         '<b>Asset:</b> %{customdata[0]}<br>' +
                         '<b>PnL:</b> $%{customdata[1]:,.2f}<br>' +
                         '<b>Return:</b> %{customdata[2]:.2f}%<br>' +
                         '<b>Entry Price:</b> $%{customdata[3]:,.2f}<br>' +
                         '<b>Exit Price:</b> $%{customdata[4]:,.2f}<extra></extra>',
            customdata=df[['position', 'pnl', 'return_pct', 'entry_price', 'exit_price']].values
        ),
        row=1, col=1
    )
    
    # Add initial capital line using summary stats
    initial_capital = summary_stats.get('initial_capital', 10000)
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Initial Capital: ${initial_capital:,.0f}",
        row=1, col=1
    )
    
    # 2. Daily Returns
    colors = ['green' if x >= 0 else 'red' for x in df['return_pct']]
    fig.add_trace(
        go.Bar(
            x=df['label'],
            y=df['return_pct'],
            name='Daily Returns',
            marker_color=colors,
            hovertemplate='<b>Identifier:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 3. Cumulative Returns
    fig.add_trace(
        go.Scatter(
            x=df['label'],
            y=df['cumulative_return_pct'],
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#2ca02c', width=2),
            hovertemplate='<b>Identifier:</b> %{x}<br><b>Cumulative Return:</b> %{y:.2f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 4. Drawdown
    fig.add_trace(
        go.Scatter(
            x=df['label'],
            y=df['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='#d62728', width=2),
            fill='tonexty',
            fillcolor='rgba(214, 39, 40, 0.3)',
            hovertemplate='<b>Identifier:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>'
        ),
        row=4, col=1
    )
    
    # 5. Position Distribution
    # Replace None values with "cash" for proper labeling
    position_data = df['position'].fillna('cash')
    position_counts = position_data.value_counts()
    fig.add_trace(
        go.Pie(
            labels=position_counts.index,
            values=position_counts.values,
            name='Position Distribution',
            hovertemplate='<b>Asset:</b> %{label}<br><b>Count:</b> %{value}<extra></extra>'
        ),
        row=5, col=1
    )
    
    # 6. Asset Performance
    asset_performance = df.groupby('position').agg({
        'pnl': 'sum',
        'return_pct': 'sum'
    }).reset_index()
    # Replace None with "cash" for display
    asset_performance['position'] = asset_performance['position'].fillna('cash')
    
    fig.add_trace(
        go.Bar(
            x=asset_performance['position'],
            y=asset_performance['pnl'],
            name='Asset PnL',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            hovertemplate='<b>Asset:</b> %{x}<br><b>Total PnL:</b> $%{y:,.2f}<extra></extra>'
        ),
        row=6, col=1
    )
    
    # 7. PnL Distribution
    pnl_data = df[df['pnl'] != 0]['pnl']
    fig.add_trace(
        go.Histogram(
            x=pnl_data,
            nbinsx=20,
            name='PnL Distribution',
            marker_color='#9467bd',
            hovertemplate='<b>PnL Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
        ),
        row=7, col=1
    )
    
    # 8. Monthly Returns
    # Only include entries that can be parsed as dates for monthly grouping
    date_entries = df[df['label'].str.match(r'^\d{4}-\d{2}-\d{2}$')].copy()
    if not date_entries.empty:
        date_entries['month'] = date_entries['label'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m'))
        monthly_returns = date_entries.groupby('month')['return_pct'].sum().reset_index()
        monthly_returns['month'] = monthly_returns['month'].astype(str)
        
        colors_monthly = ['green' if x >= 0 else 'red' for x in monthly_returns['return_pct']]
        fig.add_trace(
            go.Bar(
                x=monthly_returns['month'],
                y=monthly_returns['return_pct'],
                name='Monthly Returns',
                marker_color=colors_monthly,
                hovertemplate='<b>Month:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=8, col=1
        )
    else:
        # If no valid dates, create a simple bar with total return
        fig.add_trace(
            go.Bar(
                x=['Total'],
                y=[df['return_pct'].sum()],
                name='Total Return',
                marker_color=['green' if df['return_pct'].sum() >= 0 else 'red'],
                hovertemplate='<b>Total Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=8, col=1
        )
    
    # Add Summary Statistics Table
    best_trade = summary_stats.get('best_trade', {})
    worst_trade = summary_stats.get('worst_trade', {})
    
    # Calculate additional statistics
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] < 0]
    total_trades = len(df[df['pnl'] != 0])
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')
    
    # Calculate volatility (standard deviation of returns)
    returns_std = df['return_pct'].std()
    
    # Calculate max consecutive wins and losses
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for pnl in df['pnl']:
        if pnl > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        elif pnl < 0:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='#1f77b4',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[
                    [
                        'Initial Capital',
                        'Final Capital', 
                        'Total Return',
                        'Total Return %',
                        'Number of Trades',
                        'Number of Periods',
                        'Win Rate',
                        'Profit Factor',
                        'Avg Win',
                        'Avg Loss',
                        'Avg Return/Trade',
                        'Volatility (Std Dev)',
                        'Sharpe Ratio',
                        'Sortino Ratio',
                        'Max Consecutive Wins',
                        'Max Consecutive Losses',
                        'Best Trade',
                        'Worst Trade',
                        'Max Drawdown'
                    ],
                    [
                        f"${summary_stats.get('initial_capital', 0):,.2f}",
                        f"${summary_stats.get('final_capital', 0):,.2f}",
                        f"${summary_stats.get('total_return', 0):,.2f}",
                        f"{summary_stats.get('total_return_pct', 0):.2f}%",
                        f"{summary_stats.get('number_of_trades', 0)}",
                        f"{summary_stats.get('number_of_periods', len(df))}",
                        f"{win_rate:.1f}%",
                        f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                        f"${avg_win:,.2f}",
                        f"${avg_loss:,.2f}",
                        f"{summary_stats.get('average_return_per_trade_pct', 0):.2f}%",
                        f"{returns_std:.2f}%",
                        f"{summary_stats.get('sharpe_ratio', 0):.3f}",
                        f"{summary_stats.get('sortino_ratio', 0):.3f}",
                        f"{max_consecutive_wins}",
                        f"{max_consecutive_losses}",
                        f"{best_trade.get('asset', 'N/A')} (${best_trade.get('pnl', 0):,.0f})",
                        f"{worst_trade.get('asset', 'N/A')} (${worst_trade.get('pnl', 0):,.0f})",
                        f"{df['drawdown'].min():.2f}%"
                    ]
                ],
                fill_color='rgba(240, 240, 240, 0.8)',
                font=dict(size=11),
                align='left',
                height=25
            )
        ),
        row=1, col=2
    )
    
    # Update layout using summary stats
    final_capital = summary_stats.get('final_capital', df['capital'].iloc[-1])
    total_return_pct = summary_stats.get('total_return_pct', 0)
    
    fig.update_layout(
        title={
            'text': f'Equity Curve Analysis<br><sub>Initial Capital: ${initial_capital:,.0f} | Final Capital: ${final_capital:,.2f} | Total Return: {total_return_pct:.2f}%</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=2000,  # Increased height for 8 stacked charts
        width=1400,   # Increased width for side-by-side layout
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes labels (only for left column charts)
    fig.update_xaxes(title_text="Identifier", row=1, col=1, type='category', showticklabels=False)
    fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
    fig.update_xaxes(title_text="Identifier", row=2, col=1, type='category', showticklabels=False)
    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Identifier", row=3, col=1, type='category', showticklabels=False)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=3, col=1)
    fig.update_xaxes(title_text="Identifier", row=4, col=1, type='category', showticklabels=False)
    fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)
    fig.update_xaxes(title_text="Asset", row=6, col=1, showticklabels=True)
    fig.update_yaxes(title_text="Total PnL ($)", row=6, col=1)
    fig.update_xaxes(title_text="PnL ($)", row=7, col=1, showticklabels=True)
    fig.update_yaxes(title_text="Frequency", row=7, col=1)
    fig.update_xaxes(title_text="Month", row=8, col=1, showticklabels=True)
    fig.update_yaxes(title_text="Monthly Return (%)", row=8, col=1)
    
    # Hide the empty subplots in the right column
    for row in range(2, 9):
        fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=row, col=2)
    
    # Save the visualization
    fig.write_html(output_path)
    print(f"Equity curve visualization saved to: {output_path}")
    
    # Print comprehensive summary statistics
    print("\n=== EQUITY CURVE SUMMARY ===")
    print(f"Initial Capital: ${summary_stats.get('initial_capital', 0):,.2f}")
    print(f"Final Capital: ${summary_stats.get('final_capital', 0):,.2f}")
    print(f"Total Return: ${summary_stats.get('total_return', 0):,.2f}")
    print(f"Total Return %: {summary_stats.get('total_return_pct', 0):.2f}%")
    print(f"Number of Trades: {summary_stats.get('number_of_trades', 0)}")
    print(f"Number of Periods: {summary_stats.get('number_of_periods', len(df))}")
    print(f"Average Return per Trade: {summary_stats.get('average_return_per_trade_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {summary_stats.get('sharpe_ratio', 0):.3f}")
    print(f"Sortino Ratio: {summary_stats.get('sortino_ratio', 0):.3f}")
    
    best_trade = summary_stats.get('best_trade', {})
    worst_trade = summary_stats.get('worst_trade', {})
    print(f"Best Trade: {best_trade.get('asset', 'N/A')} on {best_trade.get('identifier', 'N/A')} - ${best_trade.get('pnl', 0):,.2f}")
    print(f"Worst Trade: {worst_trade.get('asset', 'N/A')} on {worst_trade.get('identifier', 'N/A')} - ${worst_trade.get('pnl', 0):,.2f}")
    print(f"Max Drawdown: {df['drawdown'].min():.2f}%")
    
    return fig

def create_filtered_equity_curve_visualization(config):
    """Create comprehensive equity curve visualization for filtered data using Plotly"""
    
    # Get paths from config
    equity_curve_path = config['equity_curve_file_filtered']
    output_path = config['equity_curve_visualization_filtered']
    
    data = load_equity_curve_data(equity_curve_path)
    
    # Extract summary statistics from the JSON data
    summary_stats = data.get('summary_statistics', {})
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data['equity_curve'])
    
    # Use identifiers as labels directly, no date parsing
    df['label'] = df['identifier']
    
    # Sort by the order they appear in the original data
    df = df.reset_index(drop=True)
    
    # Calculate additional metrics
    df['cumulative_return_pct'] = df['return_pct'].cumsum()
    df['drawdown'] = (df['capital'] - df['capital'].expanding().max()) / df['capital'].expanding().max() * 100
    
    # Create subplots - 2 columns: charts on left, summary table on right
    fig = make_subplots(
        rows=8, cols=2,
        subplot_titles=(
            'Filtered Equity Curve', 'Summary Statistics',
            'Daily Returns', '',
            'Cumulative Returns', '',
            'Drawdown', '',
            'Position Distribution', '',
            'Asset Performance', '',
            'PnL Distribution', '',
            'Monthly Returns', ''
        ),
        specs=[
            [{"secondary_y": False}, {"type": "table"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"type": "domain"}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "scatter"}]
        ],
        column_widths=[0.7, 0.3],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    # 1. Equity Curve
    fig.add_trace(
        go.Scatter(
            x=df['label'],
            y=df['capital'],
            mode='lines+markers',
            name='Capital',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Identifier:</b> %{x}<br>' +
                         '<b>Capital:</b> $%{y:,.2f}<br>' +
                         '<b>Asset:</b> %{customdata[0]}<br>' +
                         '<b>PnL:</b> $%{customdata[1]:,.2f}<br>' +
                         '<b>Return:</b> %{customdata[2]:.2f}%<br>' +
                         '<b>Entry Price:</b> $%{customdata[3]:,.2f}<br>' +
                         '<b>Exit Price:</b> $%{customdata[4]:,.2f}<extra></extra>',
            customdata=df[['position', 'pnl', 'return_pct', 'entry_price', 'exit_price']].values
        ),
        row=1, col=1
    )
    
    # Add DEMA line
    fig.add_trace(
        go.Scatter(
            x=df['label'],
            y=df['dema_value'],
            mode='lines',
            name='DEMA',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='<b>Identifier:</b> %{x}<br>' +
                         '<b>DEMA:</b> $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add initial capital line using summary stats
    initial_capital = summary_stats.get('initial_capital', 10000)
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Initial Capital: ${initial_capital:,.0f}",
        row=1, col=1
    )
    
    # 2. Daily Returns
    colors = ['green' if x >= 0 else 'red' for x in df['return_pct']]
    fig.add_trace(
        go.Bar(
            x=df['label'],
            y=df['return_pct'],
            name='Daily Returns',
            marker_color=colors,
            hovertemplate='<b>Identifier:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 3. Cumulative Returns
    fig.add_trace(
        go.Scatter(
            x=df['label'],
            y=df['cumulative_return_pct'],
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#2ca02c', width=2),
            hovertemplate='<b>Identifier:</b> %{x}<br><b>Cumulative Return:</b> %{y:.2f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 4. Drawdown
    fig.add_trace(
        go.Scatter(
            x=df['label'],
            y=df['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='#d62728', width=2),
            fill='tonexty',
            fillcolor='rgba(214, 39, 40, 0.3)',
            hovertemplate='<b>Identifier:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>'
        ),
        row=4, col=1
    )
    
    # 5. Position Distribution
    # Replace None values with "cash" for proper labeling
    position_data = df['position'].fillna('cash')
    position_counts = position_data.value_counts()
    fig.add_trace(
        go.Pie(
            labels=position_counts.index,
            values=position_counts.values,
            name='Position Distribution',
            hovertemplate='<b>Asset:</b> %{label}<br><b>Count:</b> %{value}<extra></extra>'
        ),
        row=5, col=1
    )
    
    # 6. Asset Performance
    asset_performance = df.groupby('position').agg({
        'pnl': 'sum',
        'return_pct': 'sum'
    }).reset_index()
    # Replace None with "cash" for display
    asset_performance['position'] = asset_performance['position'].fillna('cash')
    
    fig.add_trace(
        go.Bar(
            x=asset_performance['position'],
            y=asset_performance['pnl'],
            name='Asset PnL',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            hovertemplate='<b>Asset:</b> %{x}<br><b>Total PnL:</b> $%{y:,.2f}<extra></extra>'
        ),
        row=6, col=1
    )
    
    # 7. PnL Distribution
    pnl_data = df[df['pnl'] != 0]['pnl']
    fig.add_trace(
        go.Histogram(
            x=pnl_data,
            nbinsx=20,
            name='PnL Distribution',
            marker_color='#9467bd',
            hovertemplate='<b>PnL Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
        ),
        row=7, col=1
    )
    
    # 8. Monthly Returns
    # Only include entries that can be parsed as dates for monthly grouping
    date_entries = df[df['label'].str.match(r'^\d{4}-\d{2}-\d{2}$')].copy()
    if not date_entries.empty:
        date_entries['month'] = date_entries['label'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m'))
        monthly_returns = date_entries.groupby('month')['return_pct'].sum().reset_index()
        monthly_returns['month'] = monthly_returns['month'].astype(str)
        
        colors_monthly = ['green' if x >= 0 else 'red' for x in monthly_returns['return_pct']]
        fig.add_trace(
            go.Bar(
                x=monthly_returns['month'],
                y=monthly_returns['return_pct'],
                name='Monthly Returns',
                marker_color=colors_monthly,
                hovertemplate='<b>Month:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=8, col=1
        )
    else:
        # If no valid dates, create a simple bar with total return
        fig.add_trace(
            go.Bar(
                x=['Total'],
                y=[df['return_pct'].sum()],
                name='Total Return',
                marker_color=['green' if df['return_pct'].sum() >= 0 else 'red'],
                hovertemplate='<b>Total Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=8, col=1
        )
    
    # Add Summary Statistics Table
    best_trade = summary_stats.get('best_trade', {})
    worst_trade = summary_stats.get('worst_trade', {})
    
    # Calculate additional statistics
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] < 0]
    total_trades = len(df[df['pnl'] != 0])
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')
    
    # Calculate volatility (standard deviation of returns)
    returns_std = df['return_pct'].std()
    
    # Calculate max consecutive wins and losses
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for pnl in df['pnl']:
        if pnl > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        elif pnl < 0:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='#1f77b4',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[
                    [
                        'Initial Capital',
                        'Final Capital', 
                        'Total Return',
                        'Total Return %',
                        'Number of Trades',
                        'Number of Periods',
                        'Win Rate',
                        'Profit Factor',
                        'Avg Win',
                        'Avg Loss',
                        'Avg Return/Trade',
                        'Volatility (Std Dev)',
                        'Sharpe Ratio',
                        'Sortino Ratio',
                        'Max Consecutive Wins',
                        'Max Consecutive Losses',
                        'Best Trade',
                        'Worst Trade',
                        'Max Drawdown'
                    ],
                    [
                        f"${summary_stats.get('initial_capital', 0):,.2f}",
                        f"${summary_stats.get('final_capital', 0):,.2f}",
                        f"${summary_stats.get('total_return', 0):,.2f}",
                        f"{summary_stats.get('total_return_pct', 0):.2f}%",
                        f"{summary_stats.get('number_of_trades', 0)}",
                        f"{summary_stats.get('number_of_periods', len(df))}",
                        f"{win_rate:.1f}%",
                        f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                        f"${avg_win:,.2f}",
                        f"${avg_loss:,.2f}",
                        f"{summary_stats.get('average_return_per_trade_pct', 0):.2f}%",
                        f"{returns_std:.2f}%",
                        f"{summary_stats.get('sharpe_ratio', 0):.3f}",
                        f"{summary_stats.get('sortino_ratio', 0):.3f}",
                        f"{max_consecutive_wins}",
                        f"{max_consecutive_losses}",
                        f"{best_trade.get('asset', 'N/A')} (${best_trade.get('pnl', 0):,.0f})",
                        f"{worst_trade.get('asset', 'N/A')} (${worst_trade.get('pnl', 0):,.0f})",
                        f"{df['drawdown'].min():.2f}%"
                    ]
                ],
                fill_color='rgba(240, 240, 240, 0.8)',
                font=dict(size=11),
                align='left',
                height=25
            )
        ),
        row=1, col=2
    )
    
    # Update layout using summary stats
    final_capital = summary_stats.get('final_capital', df['capital'].iloc[-1])
    total_return_pct = summary_stats.get('total_return_pct', 0)
    
    fig.update_layout(
        title={
            'text': f'Filtered Equity Curve Analysis<br><sub>Initial Capital: ${initial_capital:,.0f} | Final Capital: ${final_capital:,.2f} | Total Return: {total_return_pct:.2f}%</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=2000,  # Increased height for 8 stacked charts
        width=1400,   # Increased width for side-by-side layout
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes labels (only for left column charts)
    fig.update_xaxes(title_text="Identifier", row=1, col=1, type='category', showticklabels=False)
    fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
    fig.update_xaxes(title_text="Identifier", row=2, col=1, type='category', showticklabels=False)
    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Identifier", row=3, col=1, type='category', showticklabels=False)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=3, col=1)
    fig.update_xaxes(title_text="Identifier", row=4, col=1, type='category', showticklabels=False)
    fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)
    fig.update_xaxes(title_text="Asset", row=6, col=1, showticklabels=True)
    fig.update_yaxes(title_text="Total PnL ($)", row=6, col=1)
    fig.update_xaxes(title_text="PnL ($)", row=7, col=1, showticklabels=True)
    fig.update_yaxes(title_text="Frequency", row=7, col=1)
    fig.update_xaxes(title_text="Month", row=8, col=1, showticklabels=True)
    fig.update_yaxes(title_text="Monthly Return (%)", row=8, col=1)
    
    # Hide the empty subplots in the right column
    for row in range(2, 9):
        fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=row, col=2)
    
    # Save the visualization
    fig.write_html(output_path)
    print(f"Filtered equity curve visualization saved to: {output_path}")
    
    # Print comprehensive summary statistics
    print("\n=== FILTERED EQUITY CURVE SUMMARY ===")
    print(f"Initial Capital: ${summary_stats.get('initial_capital', 0):,.2f}")
    print(f"Final Capital: ${summary_stats.get('final_capital', 0):,.2f}")
    print(f"Total Return: ${summary_stats.get('total_return', 0):,.2f}")
    print(f"Total Return %: {summary_stats.get('total_return_pct', 0):.2f}%")
    print(f"Number of Trades: {summary_stats.get('number_of_trades', 0)}")
    print(f"Number of Periods: {summary_stats.get('number_of_periods', len(df))}")
    print(f"Average Return per Trade: {summary_stats.get('average_return_per_trade_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {summary_stats.get('sharpe_ratio', 0):.3f}")
    print(f"Sortino Ratio: {summary_stats.get('sortino_ratio', 0):.3f}")
    
    best_trade = summary_stats.get('best_trade', {})
    worst_trade = summary_stats.get('worst_trade', {})
    print(f"Best Trade: {best_trade.get('asset', 'N/A')} on {best_trade.get('identifier', 'N/A')} - ${best_trade.get('pnl', 0):,.2f}")
    print(f"Worst Trade: {worst_trade.get('asset', 'N/A')} on {worst_trade.get('identifier', 'N/A')} - ${worst_trade.get('pnl', 0):,.2f}")
    print(f"Max Drawdown: {df['drawdown'].min():.2f}%")
    
    return fig

if __name__ == "__main__":
    # Load config for standalone execution
    with open('app/config.json', 'r') as f:
        config = json.load(f)
    create_equity_curve_visualization(config)
