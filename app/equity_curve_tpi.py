import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from trend_filters.dema import calculate_dema, calculate_ema


class EquityCurveTPI:
    """
    Simplified Trend Probability Indicator for Equity Curves
    
    Uses only 2-day DEMA on the equity curve for trend detection.
    Simple and responsive approach.
    """
    
    def __init__(self, config_file: str = "app/trend_filters/tpi_config.json"):
        """Initialize TPI with configurable parameters."""
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Load TPI configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Create default config if file doesn't exist
            config = self.create_default_config()
            self.save_config(config)
        
        # Simple DEMA Parameters
        self.dema_period = config.get("dema_period", 2)
        self.min_history_length = config.get("min_history_length", 5)
        
    def create_default_config(self) -> Dict:
        """Create simple default TPI configuration."""
        return {
            "dema_period": 2,
            "min_history_length": 5,
            "description": "Simplified TPI using only 2-day DEMA for trend detection"
        }
    
    def save_config(self, config: Dict):
        """Save configuration to file."""
        Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def analyze_trend(self, equity_values: List[float]) -> Tuple[bool, Dict]:
        """
        Simple trend analysis using only 2-day DEMA.
        
        Args:
            equity_values: List of equity curve capital values
            
        Returns:
            Tuple of (is_trending_up, analysis_details)
        """
        if len(equity_values) < self.min_history_length:
            return True, {
                "reason": "insufficient_data",
                "data_points": len(equity_values),
                "trending_up": True,
                "dema_value": None,
                "current_value": equity_values[-1] if equity_values else 0
            }
        
        current_value = equity_values[-1]
        
        # Calculate 2-day DEMA
        dema_values = calculate_dema(equity_values, self.dema_period)
        
        if not dema_values:
            # If DEMA calculation fails, default to trending up
            return True, {
                "reason": "dema_calculation_failed",
                "data_points": len(equity_values),
                "trending_up": True,
                "dema_value": None,
                "current_value": current_value
            }
        
        current_dema = dema_values[-1]
        
        # Simple rule: current equity value above DEMA = trending up
        is_trending_up = current_value > current_dema
        
        analysis = {
            "current_value": current_value,
            "dema_value": current_dema,
            "data_points": len(equity_values),
            "trending_up": is_trending_up,
            "above_dema": is_trending_up,
            "decision_reason": f"Current value ${current_value:,.2f} {'>' if is_trending_up else '<='} DEMA ${current_dema:,.2f}"
        }
        
        return is_trending_up, analysis
    
    def should_trade_based_on_reference_curve(self, reference_equity_values: List[float]) -> Tuple[bool, Dict]:
        """
        Determine if we should trade based on reference equity curve DEMA trend.
        
        Args:
            reference_equity_values: Capital values from always-allocated reference curve
            
        Returns:
            Tuple of (should_trade, tpi_analysis)
        """
        is_trending, analysis = self.analyze_trend(reference_equity_values)
        
        # Add decision logic
        analysis["should_trade"] = is_trending
        
        return is_trending, analysis


# Convenience function for integration
def create_tpi_analyzer(config_file: str = "app/trend_filters/tpi_config.json") -> EquityCurveTPI:
    """Create and return a TPI analyzer instance."""
    return EquityCurveTPI(config_file)


def analyze_equity_curve_trend(equity_values: List[float], 
                              config_file: str = "app/trend_filters/tpi_config.json") -> Tuple[bool, Dict]:
    """
    Quick function to analyze equity curve trend using simplified 2-day DEMA.
    
    Args:
        equity_values: List of equity curve capital values
        config_file: Path to TPI configuration file
        
    Returns:
        Tuple of (is_trending_up, analysis_details)
    """
    tpi = EquityCurveTPI(config_file)
    return tpi.analyze_trend(equity_values) 