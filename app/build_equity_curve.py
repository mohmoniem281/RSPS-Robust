import json
import os
import math
import sys
from typing import Dict, List, Optional, Tuple

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trend_filters.dema import apply_dema_trend_filter
from trend_filters.slope import apply_trend_slope_filter, calculate_trend_slope
from equity_curve_tpi import EquityCurveTPI
from equity_curve_kalman_tpi import KalmanEquityCurveTPI

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
        
        # TPI configuration - now using Kalman filter instead of DEMA
        self.tpi_enabled = self.config.get("equity_curve_tpi_enabled", False)
        self.kalman_tpi_enabled = self.config.get("equity_curve_kalman_tpi_enabled", False)
        
        if self.kalman_tpi_enabled:
            self.tpi_analyzer = KalmanEquityCurveTPI()
        elif self.tpi_enabled:
            self.tpi_analyzer = EquityCurveTPI()
        
        # Load data
        self.tournament_results = self._load_tournament_results()
        self.price_data = self._load_price_data()
        
        # Initialize dual equity curves
        self.initial_capital = 10000  # Starting with $10,000
        
        # Reference curve (always trades)
        self.reference_capital = self.initial_capital
        self.reference_curve = []
        self.reference_history = []
        
        # Actual curve (TPI controlled)
        self.actual_capital = self.initial_capital
        self.equity_curve = []  # Keep this as main curve for compatibility
        
        # Track capital history for trend analysis (for legacy DEMA compatibility)
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
        """Determine if we should trade based on legacy DEMA trend analysis."""
        if not self.dema_filtering_enabled or len(self.capital_history) < 2:
            return True, None  # Trade if no trend filters or insufficient history
        
        # Apply DEMA trend filter using current capital history
        slope_value = apply_trend_slope_filter(self.capital_history)
        if slope_value >= 0:
            return True, slope_value
        return False, None
    
    def _should_trade_based_on_tpi(self) -> Tuple[bool, Dict]:
        """Determine if we should trade based on TPI analysis of reference curve."""
        if not (self.tpi_enabled or self.kalman_tpi_enabled) or len(self.reference_history) < self.tpi_analyzer.min_history_length:
            return True, {"reason": "insufficient_data", "should_trade": True}
        
        # Use TPI analyzer on reference curve
        should_trade, tpi_analysis = self.tpi_analyzer.should_trade_based_on_reference_curve(self.reference_history)
        
        # If using Kalman TPI, also store the filtered values for visualization
        if self.kalman_tpi_enabled and hasattr(self.tpi_analyzer, 'process_equity_curve'):
            filtered_values, trend_signals = self.tpi_analyzer.process_equity_curve(self.reference_history)
            tpi_analysis['kalman_filtered_values'] = filtered_values
            tpi_analysis['kalman_trend_signals'] = trend_signals
        
        return should_trade, tpi_analysis
    
    def _process_reference_curve_trade(self, current_identifier: str, signal_identifier: str, exit_price_identifier: str, winner_asset: str):
        """Process a trade for the reference curve (always trades)."""
        # Get entry and exit prices
        entry_price = self._get_price(winner_asset, signal_identifier)
        exit_price = self._get_price(winner_asset, exit_price_identifier)
        
        if entry_price is None or exit_price is None:
            # Can't execute trade - add cash entry
            self._add_reference_curve_cash_entry(current_identifier, winner_asset)
            return
        
        # Calculate position and PnL
        position_size = self.reference_capital / entry_price
        pnl = (exit_price - entry_price) * position_size
        new_capital = self.reference_capital + pnl
        return_pct = (pnl / self.reference_capital) * 100 if self.reference_capital > 0 else 0
        
        # Update reference curve
        self.reference_curve.append({
            "identifier": current_identifier,
            "capital": new_capital,
            "pnl": pnl,
            "return_pct": return_pct,
            "position": position_size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "winner_asset": winner_asset
        })
        
        # Update reference capital and history
        self.reference_capital = new_capital
        self.reference_history.append(new_capital)
        
        print(f"  REFERENCE Trade: {winner_asset} | Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f} | PnL: ${pnl:.2f} | Capital: ${new_capital:.2f}")
    
    def _process_actual_curve_trade(self, current_identifier: str, signal_identifier: str, exit_price_identifier: str, winner_asset: str, filter_reason: str):
        """Process a trade for the actual curve (TPI controlled)."""
        # Get entry and exit prices
        entry_price = self._get_price(winner_asset, signal_identifier)
        exit_price = self._get_price(winner_asset, exit_price_identifier)
        
        if entry_price is None or exit_price is None:
            # Can't execute trade - add cash entry
            self._add_actual_curve_cash_entry(current_identifier, winner_asset, f"No price data: {filter_reason}")
            return
        
        # Calculate position and PnL
        position_size = self.actual_capital / entry_price
        pnl = (exit_price - entry_price) * position_size
        new_capital = self.actual_capital + pnl
        return_pct = (pnl / self.actual_capital) * 100 if self.actual_capital > 0 else 0
        
        # Update actual curve
        tpi_signal = "trade_allowed" if self.tpi_enabled else "no_filter"
        self.equity_curve.append({
            "identifier": current_identifier,
            "capital": new_capital,
            "pnl": pnl,
            "return_pct": return_pct,
            "position": position_size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "winner_asset": winner_asset,
            "tpi_signal": tpi_signal,
            "reference_capital": self.reference_capital,
            "filter_reason": filter_reason
        })
        
        # Update actual capital and history
        self.actual_capital = new_capital
        self.capital_history.append(new_capital)
        
        print(f"  ACTUAL Trade: {winner_asset} | Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f} | PnL: ${pnl:.2f} | Capital: ${new_capital:.2f}")
    
    def _add_reference_curve_cash_entry(self, current_identifier: str, winner_asset: str):
        """Add a cash entry to the reference curve."""
        self.reference_curve.append({
            "identifier": current_identifier,
            "capital": self.reference_capital,
            "pnl": 0.0,
            "return_pct": 0.0,
            "position": None,
            "entry_price": None,
            "exit_price": None,
            "winner_asset": winner_asset
        })
        self.reference_history.append(self.reference_capital)
    
    def _add_actual_curve_cash_entry(self, current_identifier: str, winner_asset: str, filter_reason: str):
        """Add a cash entry to the actual curve."""
        tpi_signal = "trade_blocked" if self.tpi_enabled else "filter_blocked"
        self.equity_curve.append({
            "identifier": current_identifier,
            "capital": self.actual_capital,
            "pnl": 0.0,
            "return_pct": 0.0,
            "position": None,
            "entry_price": None,
            "exit_price": None,
            "winner_asset": winner_asset,
            "tpi_signal": tpi_signal,
            "reference_capital": self.reference_capital,
            "filter_reason": filter_reason
        })
        self.capital_history.append(self.actual_capital)
    
    def _add_cash_entry_both_curves(self, current_identifier: str, winner_asset: str):
        """Add cash entries to both curves when no tournament winner."""
        # Reference curve
        self._add_reference_curve_cash_entry(current_identifier, winner_asset)
        
        # Actual curve  
        self._add_actual_curve_cash_entry(current_identifier, winner_asset, "No tournament winner")
    
    def build_equity_curve(self):
        """Build dual equity curves from tournament results."""
        print(f"Building dual equity curves from {self.process_from} to {self.process_to}")
        
        if self.kalman_tpi_enabled:
            print("KALMAN TPI ENABLED - actual trades based on reference curve Kalman trend analysis")
        elif self.tpi_enabled:
            print("TPI ENABLED - actual trades based on reference curve trend")
        elif self.dema_filtering_enabled:
            print("DEMA trend filtering is ENABLED - trades will only execute when trending")
        else:
            print("NO FILTERING - all trades will execute")
        
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
        
        # Initialize first entry for both curves
        first_identifier = tournament_identifiers[start_idx]
        
        # Reference curve initialization
        self.reference_history.append(self.reference_capital)
        self.reference_curve.append({
            "identifier": first_identifier,
            "capital": self.reference_capital,
            "pnl": 0.0,
            "return_pct": 0.0,
            "position": None,
            "entry_price": None,
            "exit_price": None,
            "winner_asset": None
        })
        
        # Actual curve initialization
        self.capital_history.append(self.actual_capital)
        self.equity_curve.append({
            "identifier": first_identifier,
            "capital": self.actual_capital,
            "pnl": 0.0,
            "return_pct": 0.0,
            "position": None,
            "entry_price": None,
            "exit_price": None,
            "winner_asset": None,
            "tpi_signal": None,
            "reference_capital": self.reference_capital
        })
        
        # Process each tournament starting from the third one to avoid look-ahead bias
        # We need at least 2 previous days: one for tournament signal, one for exit price
        for i in range(start_idx + 2, end_idx + 1):
            current_identifier = tournament_identifiers[i]
            exit_price_identifier = tournament_identifiers[i - 1]  # Previous day for exit price
            signal_identifier = tournament_identifiers[i - 2]     # Two days ago for signal
            
            # CRITICAL: Use tournament result from 2 days ago to avoid look-ahead bias
            print(f"Processing {current_identifier}: Using signal from {signal_identifier}, exit price from {exit_price_identifier}")
            
            # Get the winner from the tournament 2 days ago
            winner_asset = self._get_tournament_winner(signal_identifier)
            
            if winner_asset is None:
                # No tournament winner - both curves stay in cash
                self._add_cash_entry_both_curves(current_identifier, None)
                continue
            
            # Process reference curve (ALWAYS trades the winner)
            # Use signal_identifier for entry price, exit_price_identifier for exit price
            self._process_reference_curve_trade(current_identifier, signal_identifier, exit_price_identifier, winner_asset)
            
            # Determine actual curve action based on filtering method
            if self.kalman_tpi_enabled or self.tpi_enabled:
                # Use TPI (Kalman or DEMA) to decide actual curve trades - based on historical data only
                should_trade, tpi_analysis = self._should_trade_based_on_tpi()
                tpi_type = "Kalman TPI" if self.kalman_tpi_enabled else "DEMA TPI"
                filter_reason = f"{tpi_type}: {tpi_analysis.get('decision_reason', 'Unknown')}"
            elif self.dema_filtering_enabled:
                # Use legacy DEMA filtering - based on historical data only
                should_trade, dema_value = self._should_trade_based_on_trend()
                filter_reason = f"DEMA: {'Allowed' if should_trade else 'Blocked'}"
            else:
                # No filtering
                should_trade = True
                filter_reason = "No filtering"
            
            # Process actual curve based on decision
            if should_trade:
                # Use signal_identifier for entry price, exit_price_identifier for exit price
                self._process_actual_curve_trade(current_identifier, signal_identifier, exit_price_identifier, winner_asset, filter_reason)
            else:
                self._add_actual_curve_cash_entry(current_identifier, winner_asset, filter_reason)
                print(f"  Filter BLOCKED trade: {winner_asset} | {filter_reason}")
        
        # Add current signal for both curves - using previous day's data
        self._add_current_signal_both_curves()
        
        # Save both equity curves
        self._save_dual_equity_curves()
        
        # Print summary for both curves
        self._print_dual_summary()
    
    def _add_current_signal_both_curves(self):
        """Add current signal entries to both curves."""
        # Get all tournament identifiers
        tournament_identifiers = [t["tournament_identifier"] for t in self.tournament_results["tournaments"]]
        
        if len(tournament_identifiers) < 1:
            # Not enough data for a proper signal
            return
        
        # CORRECTED LOGIC: Today's signal should be based on the most recent CLOSED tournament result
        # If our data goes up to process_to_identifier_to_included (e.g., 2025-07-25),
        # then today's signal should be based on that day's tournament result (the most recent closed data)
        # This is different from historical trade processing which uses N-2 for entry and N-1 for exit
        
        latest_tournament_identifier = tournament_identifiers[-1]  # Most recent closed data (e.g., 2025-07-25)
        signal_tournament_identifier = latest_tournament_identifier  # Use the most recent closed tournament result
        
        print(f"📊 Current signal logic: Using most recent closed tournament data from {signal_tournament_identifier}")
        
        # Get the winner from the signal tournament
        current_signal_winner = self._get_tournament_winner(signal_tournament_identifier)
        
        if current_signal_winner:
            # Get the entry price from the signal day (avoiding look-ahead bias)
            entry_price = self._get_price(current_signal_winner, signal_tournament_identifier)
        else:
            entry_price = None
        
        print(f"Current signal: Based on tournament {signal_tournament_identifier} -> {current_signal_winner}")
        
        # Add to reference curve
        self.reference_curve.append({
            "identifier": "current_signal",
            "capital": self.reference_capital,
            "pnl": 0.0,
            "return_pct": 0.0,
            "position": current_signal_winner,
            "entry_price": entry_price,
            "exit_price": None,
            "winner_asset": current_signal_winner,
            "signal_source": signal_tournament_identifier
        })
        
        # For the actual curve, we need to check if TPI would allow this trade
        if self.tpi_enabled and current_signal_winner:
            should_trade, tpi_analysis = self._should_trade_based_on_tpi()
            tpi_signal = "trade_allowed" if should_trade else "trade_blocked"
            actual_position = current_signal_winner if should_trade else None
            actual_entry_price = entry_price if should_trade else None
        else:
            tpi_signal = "no_filter"
            actual_position = current_signal_winner
            actual_entry_price = entry_price
        
        # Add to actual curve
        self.equity_curve.append({
            "identifier": "current_signal",
            "capital": self.actual_capital,
            "pnl": 0.0,
            "return_pct": 0.0,
            "position": actual_position,
            "entry_price": actual_entry_price,
            "exit_price": None,
            "winner_asset": current_signal_winner,
            "tpi_signal": tpi_signal,
            "reference_capital": self.reference_capital,
            "signal_source": signal_tournament_identifier
        })
    
    def _save_dual_equity_curves(self):
        """Save both equity curves to files."""
        # Calculate statistics for both curves
        actual_stats = self._calculate_summary_statistics(self.equity_curve, self.initial_capital)
        reference_stats = self._calculate_summary_statistics(self.reference_curve, self.initial_capital)
        
        # Generate Kalman filter data if enabled
        kalman_data = None
        if self.kalman_tpi_enabled and hasattr(self.tpi_analyzer, 'process_equity_curve'):
            filtered_values, trend_signals = self.tpi_analyzer.process_equity_curve(self.reference_history)
            kalman_data = {
                "filtered_values": filtered_values,
                "trend_signals": trend_signals,
                "identifiers": [entry["identifier"] for entry in self.reference_curve[:-1]]  # Exclude current_signal
            }
        
        # Create dual curve data structure
        dual_curve_data = {
            "actual_curve": {
                "equity_curve": self.equity_curve,
                "summary_statistics": actual_stats
            },
            "reference_curve": {
                "equity_curve": self.reference_curve,
                "summary_statistics": reference_stats
            },
            "tpi_enabled": self.tpi_enabled,
            "kalman_tpi_enabled": self.kalman_tpi_enabled,
            "dema_enabled": self.dema_filtering_enabled,
            "kalman_filter_data": kalman_data
        }
        
        # Save to main file
        with open(self.equity_curve_file, 'w') as f:
            json.dump(dual_curve_data, f, indent=2)
    
    def _print_dual_summary(self):
        """Print summary statistics for both curves."""
        print(f"\n=== DUAL EQUITY CURVE SUMMARY ===")
        print(f"Reference Curve (Always Allocated):")
        print(f"  Final Capital: ${self.reference_capital:,.2f}")
        print(f"  Total Return: ${self.reference_capital - self.initial_capital:,.2f}")
        print(f"  Return %: {((self.reference_capital - self.initial_capital) / self.initial_capital) * 100:.2f}%")
        
        print(f"\nActual Curve (TPI Controlled):")
        print(f"  Final Capital: ${self.actual_capital:,.2f}")
        print(f"  Total Return: ${self.actual_capital - self.initial_capital:,.2f}")
        print(f"  Return %: {((self.actual_capital - self.initial_capital) / self.initial_capital) * 100:.2f}%")
    

    

    
    def _calculate_summary_statistics(self, equity_curve: List[Dict], initial_capital: float) -> Dict:
        """Calculate summary statistics for the equity curve."""
        if not equity_curve:
            return {}
        
        # Get final capital and total return from the last non-current_signal entry
        final_capital = initial_capital
        for entry in reversed(equity_curve):
            if entry["identifier"] != "current_signal":
                final_capital = entry["capital"]
                break
        
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Calculate some statistics
        returns = [entry["return_pct"] for entry in equity_curve if entry["return_pct"] != 0]
        trades_count = len([entry for entry in equity_curve if entry["position"] is not None])
        
        stats = {
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "number_of_trades": trades_count,
            "number_of_periods": len(equity_curve)
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

def build_the_equity_curve(config: Dict):
    """Build equity curve using the provided config."""
    builder = EquityCurveBuilder(config)
    builder.build_equity_curve()
    return builder.equity_curve

def main():
    """Main function to build the equity curve."""
    # Load config for standalone execution
    with open("app/config.json", 'r') as f:
        config = json.load(f)
    
    build_the_equity_curve(config)

if __name__ == "__main__":
    main() 