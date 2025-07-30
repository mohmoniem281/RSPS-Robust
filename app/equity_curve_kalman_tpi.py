import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class KalmanEquityCurveTPI:
    """
    Kalman Filter-based Trend Performance Indicator for Equity Curves
    
    Based on the BackQuant Pine Script Kalman Price Filter implementation.
    Uses Kalman filtering for smooth trend detection on equity curves.
    """
    
    def __init__(self, config_file: str = "app/trend_filters/kalman_tpi_config.json"):
        """Initialize Kalman TPI with configurable parameters."""
        self.config_file = config_file
        self.load_config()
        
        # Initialize Kalman filter state variables
        self.reset_filter_state()
        
        # Initialize trend persistence state (like Pine Script's 'var Trend = 0')
        self.current_trend = 0  # Persists between calls
        self.last_filtered_value = None  # Track last filtered value
        self.processed_count = 0  # Track how many equity points we've processed
    
    def load_config(self):
        """Load Kalman TPI configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Create default config if file doesn't exist
            config = self.create_default_config()
            self.save_config(config)
        
        # Kalman Filter Parameters (from Pine Script)
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
    
    def initialize_filter(self, initial_value: float):
        """Initialize Kalman filter with first equity value."""
        for i in range(self.filter_order):
            self.state_estimate[i] = initial_value
            self.error_covariance[i] = 1.0
        self.initialized = True
    
    def kalman_filter_step(self, observation: float) -> float:
        """
        Perform one step of Kalman filtering.
        Based on the Pine Script f_kalman function.
        
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
        kalman_gain = []
        for i in range(self.filter_order):
            kg = predicted_error_covariance[i] / (predicted_error_covariance[i] + self.measurement_noise)
            kalman_gain.append(kg)
            
            # Update state estimate
            self.state_estimate[i] = (
                predicted_state_estimate[i] + 
                kg * (observation - predicted_state_estimate[i])
            )
            
            # Update error covariance
            self.error_covariance[i] = (1 - kg) * predicted_error_covariance[i]
        
        # Return the first element as the filtered value
        return self.state_estimate[0]
    
    def detect_trend(self, current_filtered: float, previous_filtered: float) -> int:
        """
        Detect trend direction based on Kalman filtered values.
        Based on Pine Script trend detection logic.
        
        Args:
            current_filtered: Current Kalman filtered value
            previous_filtered: Previous Kalman filtered value
            
        Returns:
            1 for uptrend, -1 for downtrend, 0 for neutral
        """
        if previous_filtered is None:
            return 0
        
        if current_filtered > previous_filtered:
            return 1  # Uptrend
        elif current_filtered < previous_filtered:
            return -1  # Downtrend
        else:
            return 0  # Neutral
    
    def process_equity_curve(self, equity_values: List[float]) -> Tuple[List[float], List[int]]:
        """
        Process entire equity curve through Kalman filter and detect trends.
        
        Args:
            equity_values: List of equity curve capital values
            
        Returns:
            Tuple of (filtered_values, trend_signals)
        """
        if len(equity_values) < self.min_history_length:
            # Not enough data for reliable filtering
            return equity_values.copy(), [0] * len(equity_values)
        
        # Reset filter state for clean processing
        self.reset_filter_state()
        
        filtered_values = []
        trend_signals = []
        
        for i, equity_value in enumerate(equity_values):
            # Apply Kalman filter
            filtered_value = self.kalman_filter_step(equity_value)
            filtered_values.append(filtered_value)
            
            # Detect trend (need at least 2 values)
            if i == 0:
                trend_signals.append(0)  # No trend on first value
            else:
                trend = self.detect_trend(filtered_value, filtered_values[i-1])
                trend_signals.append(trend)
        
        return filtered_values, trend_signals
    
    def analyze_trend(self, equity_values: List[float]) -> Tuple[bool, Dict]:
        """
        Analyze trend for the latest equity curve state.
        Uses incremental processing to maintain Kalman filter state continuity.
        
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
                "trend_signal": self.current_trend
            }
        
        # Process only new equity values incrementally (KEY FIX!)
        new_data_count = len(equity_values) - self.processed_count
        
        if new_data_count > 0:
            # Process only the new equity values we haven't seen before
            new_equity_values = equity_values[-new_data_count:]
            
            for equity_value in new_equity_values:
                # Apply Kalman filter to this single new point
                filtered_value = self.kalman_filter_step(equity_value)
                
                # Update trend based on filtered value comparison (like Pine Script)
                if self.last_filtered_value is not None:
                    if filtered_value > self.last_filtered_value:
                        self.current_trend = 1  # Uptrend
                    elif filtered_value < self.last_filtered_value:
                        self.current_trend = -1  # Downtrend
                    # If equal, trend persists (no change)
                
                self.last_filtered_value = filtered_value
            
            # Update processed count
            self.processed_count = len(equity_values)
        
        current_value = equity_values[-1]
        
        # Trading decision: Trade on uptrend and neutral, block only on downtrend
        is_trending_up = self.current_trend >= 0  # 1 (up) or 0 (neutral) = trade, -1 (down) = cash
        
        analysis = {
            "current_value": current_value,
            "kalman_filtered": self.last_filtered_value,
            "trend_signal": self.current_trend,
            "data_points": len(equity_values),
            "trending_up": is_trending_up,
            "should_trade": is_trending_up,
            "decision_reason": f"Trend signal: {self.current_trend} ({'UP' if self.current_trend > 0 else 'DOWN' if self.current_trend < 0 else 'NEUTRAL'})"
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