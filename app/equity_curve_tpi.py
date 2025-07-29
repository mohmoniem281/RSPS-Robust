import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from trend_filters.dema import calculate_dema, calculate_ema


class EquityCurveTPI:
    """
    Trend Probability Indicator for Equity Curves
    
    Uses multiple indicators to determine equity curve trend:
    - DEMA crossover (primary trend detection)
    - Slope analysis (momentum confirmation)
    - Moving average (simple trend validation)
    
    Decision: 2 out of 3 indicators must agree for trend confirmation
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
        
        # TPI Parameters
        self.dema_period = config.get("dema_period", 10)
        self.slope_lookback = config.get("slope_lookback", 7)
        self.ma_period = config.get("ma_period", 15)
        self.min_history_length = config.get("min_history_length", 20)
        self.trend_agreement_threshold = config.get("trend_agreement_threshold", 2)  # 2 out of 3
        
    def create_default_config(self) -> Dict:
        """Create default TPI configuration."""
        return {
            "dema_period": 10,
            "slope_lookback": 7,
            "ma_period": 15,
            "min_history_length": 20,
            "trend_agreement_threshold": 2,
            "description": "TPI configuration for equity curve trend analysis"
        }
    
    def save_config(self, config: Dict):
        """Save configuration to file."""
        Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def calculate_slope(self, values: List[float], lookback: int) -> float:
        """Calculate slope (rate of change) over lookback periods."""
        if len(values) < lookback + 1:
            return 0.0
        
        # Use linear regression slope over lookback period
        recent_values = values[-lookback-1:]
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        # Linear regression: y = mx + b, we want slope m
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def calculate_moving_average(self, values: List[float], period: int) -> Optional[float]:
        """Calculate simple moving average."""
        if len(values) < period:
            return None
        
        recent_values = values[-period:]
        return sum(recent_values) / len(recent_values)
    
    def analyze_trend(self, equity_values: List[float]) -> Tuple[bool, Dict]:
        """
        Analyze equity curve trend using multiple indicators.
        
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
                "indicators": {}
            }
        
        current_value = equity_values[-1]
        indicators_agreeing = 0
        analysis = {
            "current_value": current_value,
            "data_points": len(equity_values),
            "indicators": {},
            "trending_up": True
        }
        
        # 1. DEMA Analysis
        dema_values = calculate_dema(equity_values, self.dema_period)
        if dema_values:
            current_dema = dema_values[-1]
            dema_bullish = current_value > current_dema
            analysis["indicators"]["dema"] = {
                "value": current_dema,
                "current_above_dema": dema_bullish,
                "signal": "bullish" if dema_bullish else "bearish"
            }
            if dema_bullish:
                indicators_agreeing += 1
        
        # 2. Slope Analysis
        slope = self.calculate_slope(equity_values, self.slope_lookback)
        slope_bullish = slope > 0
        analysis["indicators"]["slope"] = {
            "value": slope,
            "positive": slope_bullish,
            "signal": "bullish" if slope_bullish else "bearish"
        }
        if slope_bullish:
            indicators_agreeing += 1
        
        # 3. Moving Average Analysis
        ma_value = self.calculate_moving_average(equity_values, self.ma_period)
        if ma_value:
            ma_bullish = current_value > ma_value
            analysis["indicators"]["moving_average"] = {
                "value": ma_value,
                "current_above_ma": ma_bullish,
                "signal": "bullish" if ma_bullish else "bearish"
            }
            if ma_bullish:
                indicators_agreeing += 1
        
        # Final Decision
        total_indicators = len([k for k in analysis["indicators"] if analysis["indicators"][k]])
        is_trending_up = indicators_agreeing >= self.trend_agreement_threshold
        
        analysis.update({
            "indicators_agreeing": indicators_agreeing,
            "total_indicators": total_indicators,
            "agreement_threshold": self.trend_agreement_threshold,
            "trending_up": is_trending_up,
            "confidence": indicators_agreeing / total_indicators if total_indicators > 0 else 0.0
        })
        
        return is_trending_up, analysis
    
    def should_trade_based_on_reference_curve(self, reference_equity_values: List[float]) -> Tuple[bool, Dict]:
        """
        Determine if we should trade based on reference equity curve trend.
        
        Args:
            reference_equity_values: Capital values from always-allocated reference curve
            
        Returns:
            Tuple of (should_trade, tpi_analysis)
        """
        is_trending, analysis = self.analyze_trend(reference_equity_values)
        
        # Add decision logic
        analysis["should_trade"] = is_trending
        analysis["decision_reason"] = (
            f"Reference curve trending {'UP' if is_trending else 'DOWN'} "
            f"({analysis['indicators_agreeing']}/{analysis['total_indicators']} indicators agree)"
        )
        
        return is_trending, analysis


# Convenience function for integration
def create_tpi_analyzer(config_file: str = "app/trend_filters/tpi_config.json") -> EquityCurveTPI:
    """Create and return a TPI analyzer instance."""
    return EquityCurveTPI(config_file)


def analyze_equity_curve_trend(equity_values: List[float], 
                              config_file: str = "app/trend_filters/tpi_config.json") -> Tuple[bool, Dict]:
    """
    Quick function to analyze equity curve trend.
    
    Args:
        equity_values: List of equity curve capital values
        config_file: Path to TPI configuration file
        
    Returns:
        Tuple of (is_trending_up, analysis_details)
    """
    tpi = EquityCurveTPI(config_file)
    return tpi.analyze_trend(equity_values) 