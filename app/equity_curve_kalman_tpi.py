import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class KalmanEquityCurveTPI:
    """
    Kalman Filter-based Trend Performance Indicator for Equity Curves.
    
    Based on BackQuant Pine Script implementation with persistent trend state.
    """
    
    def __init__(self, config_file: str = "app/trend_filters/kalman_tpi_config.json"):
        """Initialize Kalman TPI with configurable parameters."""
        self.config_file = config_file
        self.load_config()
        
        # Initialize Kalman filter state variables
        self.reset_filter_state()
        
        # Persistent trend state (Pine Script: var Trend = 0)
        self.trend_state = 0
        self.previous_filtered_value = None
    
    def load_config(self):
        """Load Kalman TPI configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Create default config if file doesn't exist
            config = self.create_default_config()
            self.save_config(config)
        
        # Kalman Filter Parameters
        self.process_noise = config.get("process_noise", 0.01)
        self.measurement_noise = config.get("measurement_noise", 3.0)
        self.filter_order = config.get("filter_order", 5)
        self.min_history_length = config.get("min_history_length", 10)
        
    def create_default_config(self) -> Dict:
        """Create default Kalman TPI configuration matching Pine Script settings."""
        return {
            "process_noise": 0.01,
            "measurement_noise": 3.0,
            "filter_order": 5,
            "min_history_length": 10,
            "description": "Kalman Filter TPI based on BackQuant Pine Script implementation"
        }
    
    def save_config(self, config: Dict):
        """Save configuration to file."""
        Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def reset_filter_state(self):
        """Reset Kalman filter state variables."""
        self.state_estimate = [None] * self.filter_order
        self.error_covariance = [100.0] * self.filter_order
        self.initialized = False
        
        # Reset trend state when resetting filter
        self.trend_state = 0
        self.previous_filtered_value = None
    
    def initialize_filter(self, initial_value: float):
        """Initialize Kalman filter with first equity value (matches Pine Script f_init)."""
        for i in range(self.filter_order):
            self.state_estimate[i] = initial_value
            self.error_covariance[i] = 1.0
        self.initialized = True
    
    def kalman_filter_step(self, observation: float) -> float:
        """
        Perform one step of Kalman filtering (implements Pine Script f_kalman function).
        
        Args:
            observation: Current equity curve value
            
        Returns:
            Filtered value
        """
        if not self.initialized:
            self.initialize_filter(observation)
            return observation
        
        # Prediction Step
        predicted_state_estimate = self.state_estimate.copy()
        predicted_error_covariance = [
            cov + self.process_noise for cov in self.error_covariance
        ]
        
        # Update Step
        for i in range(self.filter_order):
            # Calculate Kalman gain
            kg = predicted_error_covariance[i] / (predicted_error_covariance[i] + self.measurement_noise)
            
            # Update state estimate
            self.state_estimate[i] = (
                predicted_state_estimate[i] + 
                kg * (observation - predicted_state_estimate[i])
            )
            
            # Update error covariance
            self.error_covariance[i] = (1 - kg) * predicted_error_covariance[i]
        
        return self.state_estimate[0]
    
    def update_trend_state(self, current_filtered: float):
        """Update persistent trend state (Pine Script: Trend := 1 if rising, -1 if falling)."""
        if self.previous_filtered_value is not None:
            if current_filtered > self.previous_filtered_value:
                self.trend_state = 1
            elif current_filtered < self.previous_filtered_value:
                self.trend_state = -1
            # Note: trend_state unchanged if values are equal
        
        self.previous_filtered_value = current_filtered
        return self.trend_state
    
    def process_equity_curve(self, equity_values: List[float]) -> Tuple[List[float], List[int]]:
        """
        Process entire equity curve through Kalman filter and detect trends.
        
        Args:
            equity_values: List of equity curve capital values
            
        Returns:
            Tuple of (filtered_values, trend_signals)
        """
        if len(equity_values) < self.min_history_length:
            return equity_values.copy(), [0] * len(equity_values)
        
        self.reset_filter_state()
        
        filtered_values = []
        trend_signals = []
        
        for equity_value in equity_values:
            filtered_value = self.kalman_filter_step(equity_value)
            filtered_values.append(filtered_value)
            
            trend_signal = self.update_trend_state(filtered_value)
            trend_signals.append(trend_signal)
        
        return filtered_values, trend_signals
    
    def analyze_trend(self, equity_values: List[float]) -> Tuple[bool, Dict]:
        """
        Analyze trend for the latest equity curve state.
        
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
                "kalman_filtered": None,
                "current_value": equity_values[-1] if equity_values else 0,
                "trend_signal": 0,
                "should_trade": True,
                "decision_reason": "Insufficient data - default to trade"
            }
        
        # Process through Kalman filter
        filtered_values, trend_signals = self.process_equity_curve(equity_values)
        
        if not filtered_values:
            return True, {
                "reason": "kalman_calculation_failed",
                "data_points": len(equity_values),
                "trending_up": True,
                "kalman_filtered": None,
                "current_value": equity_values[-1],
                "trend_signal": 0,
                "should_trade": True,
                "decision_reason": "Kalman calculation failed - default to trade"
            }
        
        current_value = equity_values[-1]
        current_filtered = filtered_values[-1]
        current_trend = trend_signals[-1] if trend_signals else 0
        
        # Trend decision: positive trend signal means trending up
        # Pine Script: Trend = 1 (uptrend), Trend = -1 (downtrend)
        is_trending_up = current_trend >= 0  # 1 (up) or 0 (neutral) = trade, -1 (down) = cash
        
        analysis = {
            "current_value": current_value,
            "kalman_filtered": current_filtered,
            "trend_signal": current_trend,
            "data_points": len(equity_values),
            "trending_up": is_trending_up,
            "should_trade": is_trending_up,
            "decision_reason": f"Trend signal: {current_trend} ({'UP' if current_trend > 0 else 'DOWN' if current_trend < 0 else 'NEUTRAL'})"
        }
        
        return is_trending_up, analysis
    
    def should_trade_based_on_reference_curve(self, reference_equity_values: List[float]) -> Tuple[bool, Dict]:
        """
        Determine if we should trade based on reference equity curve Kalman trend.
        
        Args:
            reference_equity_values: Capital values from always-allocated reference curve
            
        Returns:
            Tuple of (should_trade, kalman_analysis)
        """
        is_trending, analysis = self.analyze_trend(reference_equity_values)
        
        # Add decision logic
        analysis["should_trade"] = is_trending
        
        return is_trending, analysis


# Convenience function for integration
def create_kalman_tpi_analyzer(config_file: str = "app/trend_filters/kalman_tpi_config.json") -> KalmanEquityCurveTPI:
    """Create and return a Kalman TPI analyzer instance."""
    return KalmanEquityCurveTPI(config_file)


def analyze_equity_curve_kalman_trend(equity_values: List[float], 
                                     config_file: str = "app/trend_filters/kalman_tpi_config.json") -> Tuple[bool, Dict]:
    """
    Quick function to analyze equity curve trend using Kalman filter.
    
    Args:
        equity_values: List of equity curve capital values
        config_file: Path to Kalman TPI configuration file
        
    Returns:
        Tuple of (is_trending_up, analysis_details)
    """
    kalman_tpi = KalmanEquityCurveTPI(config_file)
    return kalman_tpi.analyze_trend(equity_values) 