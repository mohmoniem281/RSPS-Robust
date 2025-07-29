import json
import os
import math
import sys
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# try:
from trend_filters.dema import apply_dema_trend_filter, calculate_dema
from trend_filters.slope import apply_trend_slope_filter, calculate_trend_slope
from trend_filters.chandelier_exit import chandelier_exit_close_only
# except ImportError:
#     # Fallback: try absolute import
#     from app.trend_filters.dema import apply_dema_trend_filter, calculate_dema
#     from app.trend_filters.slope import apply_slope_trend_filter, calculate_slope

class EquityCurveBuilder:
    def __init__(self, config: Dict):
        """Initialize the equity curve builder with configuration."""
        self.config = config
        
        self.tournament_data_path = self.config["tournament_data_path"]
        self.equity_curve_file = self.config["equity_curve_file"]
        self.assets = self.config["assets"]
        self.process_from = self.config["process_from_identifier_from_included"]
        self.process_to = self.config["process_to_identifier_to_included"]
        
        # DEMA filtering configuration
        self.dema_filtering_enabled = self.config.get("equity_curve_dema_filtering_enabled", False)
        
        # Load data
        self.tournament_results = self._load_tournament_results()
        self.price_data = self._load_price_data()
        
        # Initialize equity curve
        self.initial_capital = 10000  # Starting with $10,000
        self.current_capital = self.initial_capital
        self.equity_curve = []
        
        # Track capital history for trend analysis
        self.capital_history = []
    
    def _load_tournament_results(self) -> Dict:
        """Load tournament results from JSON file."""
        with open(self.tournament_data_path, 'r') as f:
            return json.load(f)
    
    def _load_price_data(self) -> Dict[str, Dict[str, float]]:
        """Load price data for all assets."""
        price_data = {}
        
        for asset, paths in self.assets.items():
            price_data[asset] = {}
            
            # Load price history
            with open(paths["price_history"], 'r') as f:
                prices = json.load(f)
                
            for price_point in prices:
                time_str = price_point["time"]
                close_price = price_point["close"]
                price_data[asset][time_str] = close_price
                
        return price_data
    
    def _get_tournament_winner(self, identifier: str) -> Optional[str]:
        """Get the winner asset for a given identifier."""
        for tournament in self.tournament_results["tournaments"]:
            if tournament["tournament_identifier"] == identifier:
                winner = tournament["winner"]
                return winner if winner != "cash" else None
        return None
    
    def _get_price(self, asset: str, identifier: str) -> Optional[float]:
        """Get the price for an asset at a given identifier."""
        return self.price_data.get(asset, {}).get(identifier)
    
    def _calculate_pnl(self, entry_price: float, exit_price: float, position_size: float) -> float:
        """Calculate PnL for a trade."""
        return (exit_price - entry_price) * position_size
    
    def _should_trade_based_on_trend(self) -> Tuple[bool, Optional[float]]:
        """Determine if we should trade based on trend analysis."""
        if not self.dema_filtering_enabled or len(self.capital_history) < 2:
            return True, None  # Trade if no trend filters or insufficient history
        
        # Apply Chandelier Exit
        ce_trend = chandelier_exit_close_only(self.capital_history)
        print("Mody")
        print(ce_trend)
        signal = ce_trend.iloc[-1]
        
        if signal == 1:
            return True, signal
        return False, None
    
    def build_equity_curve(self):
        """Build the equity curve from tournament results."""
        print(f"Building equity curve from {self.process_from} to {self.process_to}")
        if self.dema_filtering_enabled:
            print("DEMA trend filtering is ENABLED - trades will only execute when trending")
        else:
            print("DEMA trend filtering is DISABLED - all trades will execute")
        
        # Get all tournament identifiers from tournament results
        tournament_identifiers = [t["tournament_identifier"] for t in self.tournament_results["tournaments"]]
        
        # Find the starting and ending indices in tournament identifiers
        try:
            start_idx = tournament_identifiers.index(self.process_from)
            end_idx = tournament_identifiers.index(self.process_to)
        except ValueError as e:
            print(f"Error: Could not find identifier in tournament results: {e}")
            print(f"Available tournament identifiers: {tournament_identifiers[:10]}...")  # Show first 10 for debugging
            return
        
        # Initialize first entry
        first_identifier = tournament_identifiers[start_idx]
        self.capital_history.append(self.current_capital)
        
        self.equity_curve.append({
            "identifier": first_identifier,
            "capital": self.current_capital,
            "pnl": 0.0,
            "return_pct": 0.0,
            "position": None,
            "entry_price": None,
            "exit_price": None,
            "winner_asset": None
        })
        
        # Process each tournament (except the first one since we need previous data)
        for i in range(start_idx + 1, end_idx + 1):
            current_identifier = tournament_identifiers[i]
            previous_identifier = tournament_identifiers[i - 1]
            
            print(f"Processing {current_identifier} using tournament result from {previous_identifier}")
            
            # Get the winner from the previous tournament
            winner_asset = self._get_tournament_winner(previous_identifier)
            
            # Check trend filter BEFORE making trading decision
            should_trade, dema_value = self._should_trade_based_on_trend()
            
            if winner_asset is None or not should_trade:
                # No trade (cash position) - either no winner or trend filter blocked trade
                self.equity_curve.append({
                    "identifier": current_identifier,
                    "capital": self.current_capital,
                    "pnl": 0.0,
                    "return_pct": 0.0,
                    "position": None,
                    "entry_price": None,
                    "exit_price": None,
                    "winner_asset": winner_asset if winner_asset else None
                })
                
                # Update capital history for trend analysis
                self.capital_history.append(self.current_capital)
                
                if not should_trade and winner_asset:
                    print(f"  Trend filter BLOCKED trade: {winner_asset} | Capital: ${self.current_capital:.2f}")
                continue
            
            # Get entry price (from previous identifier)
            entry_price = self._get_price(winner_asset, previous_identifier)
            if entry_price is None:
                print(f"Warning: No entry price found for {winner_asset} at {previous_identifier}")
                continue
            
            # Get exit price (from current identifier)
            exit_price = self._get_price(winner_asset, current_identifier)
            if exit_price is None:
                print(f"Warning: No exit price found for {winner_asset} at {current_identifier}")
                continue
            
            # Calculate position size (100% of capital)
            position_size = self.current_capital / entry_price
            
            # Calculate PnL
            pnl = self._calculate_pnl(entry_price, exit_price, position_size)
            
            # Update capital
            new_capital = self.current_capital + pnl
            return_pct = (pnl / self.current_capital) * 100 if self.current_capital > 0 else 0
            
            # Update current capital
            self.current_capital = new_capital
            
            # Update capital history for trend analysis
            self.capital_history.append(self.current_capital)
            
            # Add to equity curve
            self.equity_curve.append({
                "identifier": current_identifier,
                "capital": self.current_capital,
                "pnl": pnl,
                "return_pct": return_pct,
                "position": winner_asset,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "winner_asset": winner_asset
            })
            
            print(f"  Trade EXECUTED: {winner_asset} | Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f} | PnL: ${pnl:.2f} | Capital: ${self.current_capital:.2f}")
        
        # Save equity curve to file
        self._save_equity_curve()
        
        # Add current signal entry based on the latest tournament result
        self._add_current_signal_entry()
        
        # Print summary
        self._print_summary()
    
    def _add_current_signal_entry(self):
        """Add a current signal entry based on the latest tournament result."""
        # Get the latest tournament identifier
        latest_tournament_identifier = self.tournament_results["tournaments"][-1]["tournament_identifier"]
        
        # Get the winner from the latest tournament
        latest_winner = self._get_tournament_winner(latest_tournament_identifier)
        
        if latest_winner is None:
            # No current signal (cash position)
            current_signal_entry = {
                "identifier": "current_signal",
                "capital": self.current_capital,
                "pnl": 0.0,
                "return_pct": 0.0,
                "position": None,
                "entry_price": None,
                "exit_price": None,
                "winner_asset": None,
                "signal_source": latest_tournament_identifier
            }
        else:
            # Get entry price from the latest tournament identifier
            entry_price = self._get_price(latest_winner, latest_tournament_identifier)
            
            current_signal_entry = {
                "identifier": "current_signal",
                "capital": self.current_capital,
                "pnl": 0.0,  # No PnL yet since trade hasn't been executed
                "return_pct": 0.0,
                "position": latest_winner,
                "entry_price": entry_price,
                "exit_price": None,  # No exit price yet
                "winner_asset": latest_winner,
                "signal_source": latest_tournament_identifier
            }
        
        # Add to equity curve
        self.equity_curve.append(current_signal_entry)
        
        # Save updated equity curve
        self._save_equity_curve()
        
        print(f"\n📊 Current Signal Added:")
        print(f"  Signal Source: {latest_tournament_identifier}")
        print(f"  Current Position: {latest_winner if latest_winner else 'CASH'}")
        if latest_winner and entry_price:
            print(f"  Entry Price: ${entry_price:.2f}")
        print(f"  Current Capital: ${self.current_capital:.2f}")
    
    def _calculate_summary_statistics(self) -> Dict:
        """Calculate summary statistics for the equity curve."""
        if not self.equity_curve:
            return {}
        
        total_return = self.current_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Calculate some statistics
        returns = [entry["return_pct"] for entry in self.equity_curve if entry["return_pct"] != 0]
        trades_count = len([entry for entry in self.equity_curve if entry["position"] is not None])
        
        stats = {
            "initial_capital": self.initial_capital,
            "final_capital": self.current_capital,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "number_of_trades": trades_count,
            "number_of_periods": len(self.equity_curve)
        }
        
        if returns:
            avg_return = sum(returns) / len(returns)
            stats["average_return_per_trade_pct"] = avg_return
            
            # Convert percentage returns to decimal
            returns_decimal = [r / 100 for r in returns]
            
            # Calculate mean return
            mean_return = sum(returns_decimal) / len(returns_decimal)
            
            # Calculate standard deviation (for Sharpe ratio)
            variance = sum((r - mean_return) ** 2 for r in returns_decimal) / len(returns_decimal)
            std_dev = math.sqrt(variance)
            
            # Calculate downside deviation (for Sortino ratio)
            downside_returns = [r for r in returns_decimal if r < 0]
            if downside_returns:
                downside_variance = sum((r - mean_return) ** 2 for r in downside_returns) / len(returns_decimal)
                downside_deviation = math.sqrt(downside_variance)
            else:
                downside_deviation = 0
            
            # Risk-free rate (assuming 0% for simplicity, can be adjusted)
            risk_free_rate = 0.0
            
            # Calculate Sharpe ratio
            if std_dev > 0:
                sharpe_ratio = (mean_return - risk_free_rate) / std_dev
                stats["sharpe_ratio"] = sharpe_ratio
            else:
                stats["sharpe_ratio"] = None
            
            # Calculate Sortino ratio
            if downside_deviation > 0:
                sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
                stats["sortino_ratio"] = sortino_ratio
            else:
                stats["sortino_ratio"] = None
        
        # Find best and worst trades
        if self.equity_curve:
            best_trade = max(self.equity_curve, key=lambda x: x["pnl"])
            worst_trade = min(self.equity_curve, key=lambda x: x["pnl"])
            
            stats["best_trade"] = {
                "identifier": best_trade["identifier"],
                "asset": best_trade["winner_asset"],
                "pnl": best_trade["pnl"]
            }
            stats["worst_trade"] = {
                "identifier": worst_trade["identifier"],
                "asset": worst_trade["winner_asset"],
                "pnl": worst_trade["pnl"]
            }
        
        return stats

    def _save_equity_curve(self):
        """Save the equity curve to JSON file."""
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics()
        
        equity_curve_data = {
            "summary_statistics": summary_stats,
            "equity_curve": self.equity_curve
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.equity_curve_file), exist_ok=True)
        
        with open(self.equity_curve_file, 'w') as f:
            json.dump(equity_curve_data, f, indent=2)
        
        print(f"\nEquity curve saved to: {self.equity_curve_file}")
    
    def _print_summary(self):
        """Print a summary of the equity curve results."""
        summary_stats = self._calculate_summary_statistics()
        
        if not summary_stats:
            print("No equity curve data to summarize")
            return
        
        print(f"\n=== EQUITY CURVE SUMMARY ===")
        print(f"Initial Capital: ${summary_stats['initial_capital']:,.2f}")
        print(f"Final Capital: ${summary_stats['final_capital']:,.2f}")
        print(f"Total Return: ${summary_stats['total_return']:,.2f} ({summary_stats['total_return_pct']:.2f}%)")
        print(f"Number of Trades: {summary_stats['number_of_trades']}")
        print(f"Number of Periods: {summary_stats['number_of_periods']}")
        
        if 'average_return_per_trade_pct' in summary_stats:
            print(f"Average Return per Trade: {summary_stats['average_return_per_trade_pct']:.2f}%")
            
            # Print Sharpe ratio
            if summary_stats['sharpe_ratio'] is not None:
                print(f"Sharpe Ratio: {summary_stats['sharpe_ratio']:.3f}")
            else:
                print("Sharpe Ratio: N/A (no volatility)")
            
            # Print Sortino ratio
            if summary_stats['sortino_ratio'] is not None:
                print(f"Sortino Ratio: {summary_stats['sortino_ratio']:.3f}")
            else:
                print("Sortino Ratio: N/A (no downside volatility)")
        
        # Print best and worst trades
        if 'best_trade' in summary_stats:
            best = summary_stats['best_trade']
            worst = summary_stats['worst_trade']
            print(f"Best Trade: {best['identifier']} - {best['asset']} - ${best['pnl']:.2f}")
            print(f"Worst Trade: {worst['identifier']} - {worst['asset']} - ${worst['pnl']:.2f}")

def build_the_equity_curve(config: Dict):
    """Build equity curve using the provided config."""
    builder = EquityCurveBuilder(config)
    builder.build_equity_curve()
    return builder.equity_curve

# def apply_trend_filters(config: Dict):
#     """Apply trend filters to the equity curve and rebuild it properly."""
#     print("Applying trend filters to equity curve...")
    
#     # Load the equity curve data
#     with open(config["equity_curve_file"], 'r') as f:
#         equity_curve_data = json.load(f)

#     # Get the equity curve points
#     equity_points = equity_curve_data["equity_curve"]
    
#     # Create filtered file path
#     filtered_file_path = config["equity_curve_file"].replace(".json", "_filtered.json")
    
#     # Initialize capital tracking
#     initial_capital = equity_points[0]["capital"]
#     current_capital = initial_capital
    
#     # Create new equity curve with trend filters applied
#     filtered_equity_curve = []
    
#     # Add the first point (initial capital)
#     filtered_equity_curve.append({
#         "identifier": equity_points[0]["identifier"],
#         "capital": current_capital,
#         "pnl": 0.0,
#         "return_pct": 0.0,
#         "position": None,
#         "entry_price": None,
#         "exit_price": None,
#         "winner_asset": None,
#         "dema_value": None  # No DEMA for first point
#     })
    
#     # Process each point starting from the second one
#     for i in range(1, len(equity_points)):
#         point = equity_points[i]
        
#         # Get capital values from ORIGINAL equity curve for trend analysis
#         # Use data up to the previous day (i-1) for trend decision
#         historical_capitals = [p["capital"] for p in equity_points[:i-1]]
        
#         # Apply DEMA trend filter using historical data up to previous day
#         is_trending, dema_value = apply_dema_trend_filter(historical_capitals)
        
#         # For trend decision, compare previous day's capital with DEMA
#         # The trend decision should be based on whether we should trade on the current day
#         previous_capital = historical_capitals[-1] if historical_capitals else current_capital
        
#         if not is_trending:
#             # Not trending or capital below DEMA - set to cash position
#             new_point = {
#                 "identifier": point["identifier"],
#                 "capital": current_capital,  # No change in capital
#                 "pnl": 0.0,
#                 "return_pct": 0.0,
#                 "position": None,
#                 "entry_price": None,
#                 "exit_price": None,
#                 "winner_asset": None,
#                 "dema_value": dema_value
#             }
#         else:
#             # Trending  - check if original point has valid trade data
#             original_position = point.get("position")
#             original_entry_price = point.get("entry_price")
#             original_exit_price = point.get("exit_price")
#             original_winner_asset = point.get("winner_asset")
        
#             # Check if original point was a cash position (no trade)
#             if original_position is None or original_entry_price is None or original_exit_price is None:
#                 # Original point was cash - keep as cash even when trending
#                 new_point = {
#                     "identifier": point["identifier"],
#                     "capital": current_capital,  # No change in capital
#                     "pnl": 0.0,
#                     "return_pct": 0.0,
#                     "position": None,
#                     "entry_price": None,
#                     "exit_price": None,
#                     "winner_asset": None,
#                     "dema_value": dema_value
#                 }
#             else:
#                 # Original point had a valid trade - process it
#                 position_size = current_capital / original_entry_price
#                 pnl = (original_exit_price - original_entry_price) * position_size
#                 new_capital = current_capital + pnl
#                 return_pct = (pnl / current_capital) * 100 if current_capital > 0 else 0
                
#                 new_point = {
#                     "identifier": point["identifier"],
#                     "capital": new_capital,
#                     "pnl": pnl,
#                     "return_pct": return_pct,
#                     "position": original_position,
#                     "entry_price": original_entry_price,
#                     "exit_price": original_exit_price,
#                     "winner_asset": original_winner_asset,
#                     "dema_value": dema_value
#                 }
                
#                 # Update current capital
#                 current_capital = new_capital
        
#         filtered_equity_curve.append(new_point)
    
#     # Handle current signal entry if it exists
#     if filtered_equity_curve and filtered_equity_curve[-1]["identifier"] == "current_signal":
#         # Get capital values from original equity curve for trend analysis
#         historical_capitals = [p["capital"] for p in equity_points if p["identifier"] != "current_signal"]
        
#         # Apply DEMA trend filter
#         is_trending, dema_value = apply_dema_trend_filter(historical_capitals)
        
#         if not is_trending:
#             # Not trending - set current signal to cash
#             filtered_equity_curve[-1].update({
#                 "position": None,
#                 "entry_price": None,
#                 "exit_price": None,
#                 "winner_asset": None,
#                 "pnl": 0.0,
#                 "return_pct": 0.0,
#                 "dema_value": dema_value
#             })
#         else:
#             # Trending - update DEMA value only
#             filtered_equity_curve[-1].update({
#                 "dema_value": dema_value
#             })
    
#     # Create new equity curve data with recalculated summary statistics
#     new_equity_curve_data = {
#         "equity_curve": filtered_equity_curve
#     }
    
#     # Calculate new summary statistics
#     new_equity_curve_data["summary_statistics"] = _calculate_filtered_summary_statistics(
#         filtered_equity_curve, initial_capital
#     )
    
#     # Save the filtered equity curve to the new file
#     with open(filtered_file_path, 'w') as f:
#         json.dump(new_equity_curve_data, f, indent=2)
    
#     print(f"Trend filters applied. Final capital: ${current_capital:,.2f}")
#     print(f"Filtered equity curve saved to: {filtered_file_path}")

# def _calculate_filtered_summary_statistics(equity_curve: List[Dict], initial_capital: float) -> Dict:
#     """Calculate summary statistics for the filtered equity curve."""
#     if not equity_curve:
#         return {}
    
#     # Get final capital from the last non-current_signal entry
#     final_capital = initial_capital
#     for point in reversed(equity_curve):
#         if point["identifier"] != "current_signal":
#             final_capital = point["capital"]
#             break
    
#     total_return = final_capital - initial_capital
#     total_return_pct = (total_return / initial_capital) * 100
    
#     # Calculate statistics
#     returns = [entry["return_pct"] for entry in equity_curve if entry["return_pct"] != 0]
#     trades_count = len([entry for entry in equity_curve if entry["position"] is not None])
    
#     stats = {
#         "initial_capital": initial_capital,
#         "final_capital": final_capital,
#         "total_return": total_return,
#         "total_return_pct": total_return_pct,
#         "number_of_trades": trades_count,
#         "number_of_periods": len(equity_curve)
#     }
    
#     if returns:
#         avg_return = sum(returns) / len(returns)
#         stats["average_return_per_trade_pct"] = avg_return
        
#         # Convert percentage returns to decimal
#         returns_decimal = [r / 100 for r in returns]
        
#         # Calculate mean return
#         mean_return = sum(returns_decimal) / len(returns_decimal)
        
#         # Calculate standard deviation (for Sharpe ratio)
#         variance = sum((r - mean_return) ** 2 for r in returns_decimal) / len(returns_decimal)
#         std_dev = math.sqrt(variance)
        
#         # Calculate downside deviation (for Sortino ratio)
#         downside_returns = [r for r in returns_decimal if r < 0]
#         if downside_returns:
#             downside_variance = sum((r - mean_return) ** 2 for r in downside_returns) / len(returns_decimal)
#             downside_deviation = math.sqrt(downside_variance)
#         else:
#             downside_deviation = 0
        
#         # Risk-free rate (assuming 0% for simplicity, can be adjusted)
#         risk_free_rate = 0.0
        
#         # Calculate Sharpe ratio
#         if std_dev > 0:
#             sharpe_ratio = (mean_return - risk_free_rate) / std_dev
#             stats["sharpe_ratio"] = sharpe_ratio
#         else:
#             stats["sharpe_ratio"] = None
        
#         # Calculate Sortino ratio
#         if downside_deviation > 0:
#             sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
#             stats["sortino_ratio"] = sortino_ratio
#         else:
#             stats["sortino_ratio"] = None
    
#     # Find best and worst trades
#     if equity_curve:
#         best_trade = max(equity_curve, key=lambda x: x["pnl"])
#         worst_trade = min(equity_curve, key=lambda x: x["pnl"])
        
#         stats["best_trade"] = {
#             "identifier": best_trade["identifier"],
#             "asset": best_trade["winner_asset"],
#             "pnl": best_trade["pnl"]
#         }
#         stats["worst_trade"] = {
#             "identifier": worst_trade["identifier"],
#             "asset": worst_trade["winner_asset"],
#             "pnl": worst_trade["pnl"]
#         }
    
#     return stats

def main():
    """Main function to build the equity curve."""
    # Load config for standalone execution
    with open("app/config.json", 'r') as f:
        config = json.load(f)
    
    build_the_equity_curve(config)

if __name__ == "__main__":
    main() 