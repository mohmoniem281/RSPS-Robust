import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
import os

class EquityCurveVisualizer:
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 original_file: Optional[str] = None, 
                 filtered_file: Optional[str] = None,
                 output_file: Optional[str] = None):
        """
        Initialize the equity curve visualizer.
        
        Args:
            config: Configuration dictionary with file paths and output file
            original_file: Path to the original equity curve JSON file (overrides config)
            filtered_file: Path to the filtered equity curve JSON file (overrides config)
            output_file: Output HTML file path (overrides config)
        """
        self.config = config or {}
        
        # Set file paths (config takes precedence, then direct parameters, then defaults)
        self.original_file = (self.config.get('equity_curve_file') or 
                             original_file or 
                             "data/equity_curve.json")
        
        self.filtered_file = (self.config.get('equity_curve_file_filtered') or 
                              filtered_file or 
                              "data/equity_curve_filtered.json")
        
        # Set output file path
        self.output_file = (self.config.get('equity_curve_visualization') or 
                            output_file or 
                            "equity_curve_dashboard.html")
        
        self.original_data = None
        self.filtered_data = None
        
        print(f"Initialized visualizer with:")
        print(f"  Original file: {self.original_file}")
        print(f"  Filtered file: {self.filtered_file}")
        print(f"  Output file: {self.output_file}")
        
    def load_data(self) -> None:
        """Load both equity curve datasets."""
        try:
            with open(self.original_file, 'r') as f:
                self.original_data = json.load(f)
            print(f"Loaded original data: {len(self.original_data['curve'])} data points")
            
            with open(self.filtered_file, 'r') as f:
                self.filtered_data = json.load(f)
            print(f"Loaded filtered data: {len(self.filtered_data['curve'])} data points")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            raise
    
    def create_combined_curves_chart(self) -> go.Figure:
        """Create a chart showing both equity curves with detailed hover information."""
        fig = go.Figure()
        
        # Original equity curve
        original_curve = self.original_data['curve']
        identifiers = [point['identifier'] for point in original_curve]
        capital_values = [point['capital'] for point in original_curve]
        
        # Debug: Print some identifiers to verify they're being processed correctly
        print(f"Processing {len(identifiers)} data points")
        print(f"First few identifiers: {identifiers[:5]}")
        print(f"Last few identifiers: {identifiers[-5:]}")
        
        # Create detailed hover text for original curve
        hover_text_original = []
        for point in original_curve:
            # Handle None values safely
            entry_price_str = f"${point['entry_price']:,.2f}" if point['entry_price'] is not None else 'N/A'
            exit_price_str = f"${point['exit_price']:,.2f}" if point['exit_price'] is not None else 'N/A'
            position_str = str(point['position']) if point['position'] is not None else 'No Position'
            winner_asset_str = str(point['winner_asset']) if point['winner_asset'] is not None else 'N/A'
            
            hover_text = f"""
            <b>Original Strategy</b><br>
            <b>Period:</b> {point['identifier']}<br>
            <b>Capital:</b> ${point['capital']:,.2f}<br>
            <b>PnL:</b> ${point['pnl']:,.2f}<br>
            <b>Return:</b> {point['return_pct']:.2f}%<br>
            <b>Position:</b> {position_str}<br>
            <b>Entry Price:</b> {entry_price_str}<br>
            <b>Exit Price:</b> {exit_price_str}<br>
            <b>Winner Asset:</b> {winner_asset_str}
            """
            hover_text_original.append(hover_text)
        
        fig.add_trace(go.Scatter(
            x=identifiers,
            y=capital_values,
            mode='lines+markers',
            name='Original Strategy',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text_original
        ))
        
        # Filtered equity curve
        filtered_curve = self.filtered_data['curve']
        filtered_identifiers = [point['identifier'] for point in filtered_curve]
        filtered_capital_values = [point['capital'] for point in filtered_curve]
        
        # Create detailed hover text for filtered curve
        hover_text_filtered = []
        for point in filtered_curve:
            # Handle None values safely
            entry_price_str = f"${point['entry_price']:,.2f}" if point['entry_price'] is not None else 'N/A'
            exit_price_str = f"${point['exit_price']:,.2f}" if point['exit_price'] is not None else 'N/A'
            position_str = str(point['position']) if point['position'] is not None else 'No Position'
            winner_asset_str = str(point['winner_asset']) if point['winner_asset'] is not None else 'N/A'
            
            hover_text = f"""
            <b>Filtered Strategy</b><br>
            <b>Period:</b> {point['identifier']}<br>
            <b>Capital:</b> ${point['capital']:,.2f}<br>
            <b>PnL:</b> ${point['pnl']:,.2f}<br>
            <b>Return:</b> {point['return_pct']:.2f}%<br>
            <b>Position:</b> {position_str}<br>
            <b>Entry Price:</b> {entry_price_str}<br>
            <b>Exit Price:</b> {exit_price_str}<br>
            <b>Winner Asset:</b> {winner_asset_str}
            """
            hover_text_filtered.append(hover_text)
        
        fig.add_trace(go.Scatter(
            x=filtered_identifiers,
            y=filtered_capital_values,
            mode='lines+markers',
            name='Filtered Strategy',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=4),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text_filtered
        ))
        
        # Add baseline (initial capital)
        initial_capital = capital_values[0] if capital_values else 10000
        fig.add_hline(
            y=initial_capital, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Initial Capital: ${initial_capital:,.2f}"
        )
        
        fig.update_layout(
            title='Equity Curves Comparison',
            xaxis_title='Time Period',
            yaxis_title='Capital ($)',
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        # Force x-axis to be categorical to handle string identifiers like "Current Signal"
        fig.update_xaxes(type='category')
        
        # Make x-axis categorical and handle many data points
        if len(identifiers) > 50:
            n = max(1, len(identifiers) // 20)
            fig.update_xaxes(
                tickmode='array',
                tickvals=identifiers[::n],
                ticktext=identifiers[::n]
            )
        
        return fig
    
    def create_summary_table(self, data: Dict[str, Any], title: str = "Summary Statistics") -> go.Figure:
        """Create a table showing summary statistics."""
        if 'summary' not in data:
            return None
            
        summary = data['summary']
        
        # Define the metrics to display
        metrics = [
            ['Total Return (%)', f"{summary.get('total_return', 0):.2f}"],
            ['Annualized Return (%)', f"{summary.get('annualized_return', 0):.2f}"],
            ['Annualized Volatility (%)', f"{summary.get('annualized_volatility', 0):.2f}"],
            ['Sharpe Ratio', f"{summary.get('sharpe_ratio', 0):.2f}"],
            ['Sortino Ratio', f"{summary.get('sortino_ratio', 0):.2f}"],
            ['Omega Ratio', f"{summary.get('omega_ratio', 0):.2f}"],
            ['Max Drawdown (%)', f"{summary.get('max_drawdown', 0):.2f}"],
            ['Final Capital ($)', f"{summary.get('final_capital', 0):,.2f}"],
            ['Number of Trades', f"{summary.get('number_of_trades', 0)}"]
        ]
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                font=dict(size=14, color='white'),
                fill_color='#1f77b4',
                align='left'
            ),
            cells=dict(
                values=[[row[0] for row in metrics], [row[1] for row in metrics]],
                font=dict(size=12),
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title=title,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_drawdown_chart(self) -> go.Figure:
        """Create a chart showing drawdown for both curves."""
        fig = go.Figure()
        
        # Original curve drawdown
        original_curve = self.original_data['curve']
        identifiers = [point['identifier'] for point in original_curve]
        capital_values = [point['capital'] for point in original_curve]
        
        # Calculate drawdown for original
        peak = capital_values[0]
        drawdowns_original = []
        for capital in capital_values:
            if capital > peak:
                peak = capital
            drawdown = (capital - peak) / peak * 100
            drawdowns_original.append(drawdown)
        
        fig.add_trace(go.Scatter(
            x=identifiers,
            y=drawdowns_original,
            mode='lines',
            name='Original Strategy Drawdown',
            line=dict(color='#1f77b4', width=2),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        # Filtered curve drawdown
        filtered_curve = self.filtered_data['curve']
        filtered_capital_values = [point['capital'] for point in filtered_curve]
        
        # Calculate drawdown for filtered
        peak = filtered_capital_values[0]
        drawdowns_filtered = []
        for capital in filtered_capital_values:
            if capital > peak:
                peak = capital
            drawdown = (capital - peak) / peak * 100
            drawdowns_filtered.append(drawdown)
        
        fig.add_trace(go.Scatter(
            x=identifiers,
            y=drawdowns_filtered,
            mode='lines',
            name='Filtered Strategy Drawdown',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.3)'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Drawdown Comparison',
            xaxis_title='Time Period',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        # Force x-axis to be categorical to handle string identifiers like "Current Signal"
        fig.update_xaxes(type='category')
        
        # Handle x-axis for many data points
        if len(identifiers) > 50:
            n = max(1, len(identifiers) // 20)
            fig.update_xaxes(
                tickmode='array',
                tickvals=identifiers[::n],
                ticktext=identifiers[::n]
            )
        
        return fig
    
    def create_asset_allocation_chart(self) -> go.Figure:
        """Create a chart showing asset allocation for both curves."""
        # Create subplots for side-by-side comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original Strategy', 'Filtered Strategy'),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        # Original strategy asset allocation
        original_curve = self.original_data['curve']
        original_assets = [point['winner_asset'] if point['winner_asset'] else 'No Position' for point in original_curve]
        original_asset_counts = {}
        for asset in original_assets:
            if asset in original_asset_counts:
                original_asset_counts[asset] += 1
            else:
                original_asset_counts[asset] = 1
        
        fig.add_trace(
            go.Pie(
                labels=list(original_asset_counts.keys()),
                values=list(original_asset_counts.values()),
                hole=0.3,
                name="Original"
            ),
            row=1, col=1
        )
        
        # Filtered strategy asset allocation
        filtered_curve = self.filtered_data['curve']
        filtered_assets = [point['winner_asset'] if point['winner_asset'] else 'No Position' for point in filtered_curve]
        filtered_asset_counts = {}
        for asset in filtered_assets:
            if asset in filtered_asset_counts:
                filtered_asset_counts[asset] += 1
            else:
                filtered_asset_counts[asset] = 1
        
        fig.add_trace(
            go.Pie(
                labels=list(filtered_asset_counts.keys()),
                values=list(filtered_asset_counts.values()),
                hole=0.3,
                name="Filtered"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Asset Allocation Comparison',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_simple_html(self) -> str:
        """Create a simple HTML file with the requested charts and tables."""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Equity Curve Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .section-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }
        .tables-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 40px;
        }
        .table-section {
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: white;
        }
        .table-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Equity Curve Analysis</h1>
            <p>Simple and focused analysis of trading strategy performance</p>
        </div>
        
        <div class="section">
            <div class="section-title">📈 Combined Equity Curves</div>
            <div id="combined_curves"></div>
        </div>
        
        <div class="tables-container">
            <div class="table-section">
                <div class="table-title">📋 Original Strategy Summary</div>
                <div id="original_summary"></div>
            </div>
            <div class="table-section">
                <div class="table-title">📋 Filtered Strategy Summary</div>
                <div id="filtered_summary"></div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📉 Drawdown Comparison</div>
            <div id="drawdown"></div>
        </div>
        
        <div class="section">
            <div class="section-title">🥧 Asset Allocation Comparison</div>
            <div id="asset_allocation"></div>
        </div>
    </div>
    
    <script>
        // Charts will be loaded here
        const charts = {};
        
        // Function to load a chart
        function loadChart(containerId, chartData) {
            if (chartData && chartData.data) {
                Plotly.newPlot(containerId, chartData.data, chartData.layout);
            }
        }
        
        // Load all charts when page loads
        window.addEventListener('load', function() {
            if (charts.combined_curves) {
                loadChart('combined_curves', charts.combined_curves);
            }
            if (charts.original_summary) {
                loadChart('original_summary', charts.original_summary);
            }
            if (charts.filtered_summary) {
                loadChart('filtered_summary', charts.filtered_summary);
            }
            if (charts.drawdown) {
                loadChart('drawdown', charts.drawdown);
            }
            if (charts.asset_allocation) {
                loadChart('asset_allocation', charts.asset_allocation);
            }
        });
    </script>
</body>
</html>
"""
        return html_content
    
    def save_simple_html(self) -> None:
        """Save the simple HTML dashboard with only the requested components."""
        # Create the HTML content
        html_content = self.create_simple_html()
        
        # Get chart data as JSON
        charts_data = {
            'combined_curves': self.create_combined_curves_chart().to_dict(),
            'original_summary': self.create_summary_table(self.original_data, "Original Strategy Summary").to_dict() if self.original_data else None,
            'filtered_summary': self.create_summary_table(self.filtered_data, "Filtered Strategy Summary").to_dict() if self.filtered_data else None,
            'drawdown': self.create_drawdown_chart().to_dict(),
            'asset_allocation': self.create_asset_allocation_chart().to_dict()
        }
        
        # Convert charts data to JSON and embed in HTML
        charts_json = json.dumps(charts_data)
        
        # Replace the charts variable in HTML with actual data
        html_content = html_content.replace('const charts = {};', f'const charts = {charts_json};')
        
        # Save the HTML file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Simple HTML dashboard saved to: {self.output_file}")
    
    def save_charts(self, output_dir: Optional[str] = None) -> None:
        """Save charts as a single HTML file."""
        self.save_simple_html()
    
    def display_charts(self) -> None:
        """Display all charts in the notebook/browser."""
        print("Generating equity curve visualizations...")
        
        print("\n1. Combined Equity Curves Chart:")
        self.create_combined_curves_chart().show()
        
        print("\n2. Original Strategy Summary Table:")
        if self.original_data:
            self.create_summary_table(self.original_data, "Original Strategy Summary").show()
        
        print("\n3. Filtered Strategy Summary Table:")
        if self.filtered_data:
            self.create_summary_table(self.filtered_data, "Filtered Strategy Summary").show()
        
        print("\n4. Drawdown Comparison Chart:")
        self.create_drawdown_chart().show()
        
        print("\n5. Asset Allocation Comparison Chart:")
        self.create_asset_allocation_chart().show()

def create_equity_curve_dashboard(config):
    # Initialize visualizer with config
    visualizer = EquityCurveVisualizer(config=config)
    
    # Load data
    visualizer.load_data()
    
    # Save simple HTML dashboard
    visualizer.save_simple_html()
    
    print(f"\n✓ Equity curve dashboard created successfully!")
    print(f"✓ File saved to: {visualizer.output_file}")
    print(f"✓ Open this file in your web browser to view the dashboard")

def main():
    """Main function to run the visualizer with default settings."""
    # Initialize visualizer with default settings
    visualizer = EquityCurveVisualizer()
    
    # Load data
    visualizer.load_data()
    
    # Save simple HTML dashboard
    visualizer.save_simple_html()
    
    print(f"\n✓ Equity curve dashboard created successfully!")
    print(f"✓ File saved to: {visualizer.output_file}")
    print(f"✓ Open this file in your web browser to view the dashboard")

if __name__ == "__main__":
    main()

