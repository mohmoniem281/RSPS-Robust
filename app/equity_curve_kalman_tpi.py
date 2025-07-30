import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class KalmanEquityCurveTPI:
    """
    Kalman Filter-based Trend Performance Indicator for Equity Curves
    
    Based on the BackQuant Pine Script Kalman Price Filter implementation.
    Uses Kalman filtering for smooth trend detection on equity curves.
    Implements EXACT Pine Script algorithm including persistent trend state.
    """
    
    def __init__(self, config_file: str = "app/trend_filters/kalman_tpi_config.json"):
        """Initialize Kalman TPI with configurable parameters."""
        self.config_file = config_file
        self.load_config()
        
        # Initialize Kalman filter state variables
        self.reset_filter_state()
        
        # Pine Script persistent trend state (var Trend = 0)
        self.trend_state = 0  # Persists across all filter operations
        self.previous_filtered_value = None  # For trend comparison
    
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
        # Using exact names from Pine Script: processNoise, measurementNoise, N
        self.process_noise = config.get("process_noise", 0.01)
        self.measurement_noise = config.get("measurement_noise", 3.0)
        self.filter_order = config.get("filter_order", 5)  # N in Pine Script
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
        """Reset Kalman filter state variables - matches Pine Script var arrays."""
        # Pine Script: var float[] stateEstimate = array.new_float(N, na)
        # Pine Script: var float[] errorCovariance = array.new_float(N, 100.0)
        self.state_estimate = [None] * self.filter_order
        self.error_covariance = [100.0] * self.filter_order
        self.initialized = False
        
        # Reset trend state when resetting filter
        self.trend_state = 0
        self.previous_filtered_value = None
    
    def initialize_filter(self, initial_value: float):
        """
        Initialize Kalman filter with first equity value.
        Matches Pine Script f_init function exactly.
        """
        # Pine Script f_init logic:
        # if na(array.get(stateEstimate, 0))
        #     for i = 0 to N-1
        #         array.set(stateEstimate, i, pricesource)
        #         array.set(errorCovariance, i, 1.0)
        
        for i in range(self.filter_order):
            self.state_estimate[i] = initial_value
            self.error_covariance[i] = 1.0
        self.initialized = True
    
    def kalman_filter_step(self, observation: float) -> float:
        """
        Perform one step of Kalman filtering.
        Implements EXACT Pine Script f_kalman function logic.
        
        Args:
            observation: Current equity curve value (pricesource in Pine Script)
            
        Returns:
            Filtered value (kalmanFilteredPrice in Pine Script)
        """
        # Pine Script f_init call
        if not self.initialized:
            self.initialize_filter(observation)
            return observation
        
        # Pine Script f_kalman function implementation:
        
        # Prediction Step
        # predictedStateEstimate = array.new_float(N)
        # predictedErrorCovariance = array.new_float(N)
        # for i = 0 to N-1
        #     array.set(predictedStateEstimate, i, array.get(stateEstimate, i)) // Simplified prediction
        #     array.set(predictedErrorCovariance, i, array.get(errorCovariance, i) + processNoise)
        
        predicted_state_estimate = self.state_estimate.copy()  # Simplified prediction
        predicted_error_covariance = [
            cov + self.process_noise for cov in self.error_covariance
        ]
        
        # Update Step
        # kalmanGain = array.new_float(N)
        # for i = 0 to N-1
        #     kg = array.get(predictedErrorCovariance, i) / (array.get(predictedErrorCovariance, i) + measurementNoise)
        #     array.set(kalmanGain, i, kg)
        #     array.set(stateEstimate, i, array.get(predictedStateEstimate, i) + kg * (pricesource - array.get(predictedStateEstimate, i)))
        #     array.set(errorCovariance, i, (1 - kg) * array.get(predictedErrorCovariance, i))
        
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
        
        # Return the first element as the filtered value
        # array.get(stateEstimate, 0)
        return self.state_estimate[0]
    
    def update_trend_state(self, current_filtered: float):
        """
        Update persistent trend state based on Pine Script logic.
        Implements EXACT Pine Script trend detection:
        
        var Trend = 0
        if kalmanFilteredPrice>kalmanFilteredPrice[1]
            Trend := 1
        if kalmanFilteredPrice<kalmanFilteredPrice[1] 
            Trend := -1
        """
        if self.previous_filtered_value is not None:
            # Pine Script logic: compare current vs previous
            if current_filtered > self.previous_filtered_value:
                self.trend_state = 1
            elif current_filtered < self.previous_filtered_value:
                self.trend_state = -1
            # Note: trend_state remains unchanged if current_filtered == previous_filtered_value
        
        # Update previous value for next comparison
        self.previous_filtered_value = current_filtered
        
        return self.trend_state
    
    def process_equity_curve(self, equity_values: List[float]) -> Tuple[List[float], List[int]]:
        """
        Process entire equity curve through Kalman filter and detect trends.
        Implements the complete Pine Script logic with persistent trend state.
        
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
            # Apply Kalman filter (f_kalman in Pine Script)
            filtered_value = self.kalman_filter_step(equity_value)
            filtered_values.append(filtered_value)
            
            # Update persistent trend state (Pine Script Trend variable logic)
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