import numpy as np
import json
import os
from pathlib import Path

def kalman_price_filter(prices):
    # Load configuration
    config_path = Path(__file__).parent / "kalman_filter.json"
    with open(config_path, 'r') as f:
        kalman_config = json.load(f)
    
    # Use provided parameters or fall back to config values
    process_noise = kalman_config.get('process_noise')
    measurement_noise = kalman_config.get('measurement_noise')
    filter_order = kalman_config.get('filter_order')
    
    n = len(prices)
    N = filter_order
    
    # Initialize arrays like Pine Script
    state_estimate = np.full(N, np.nan)  # Equivalent to array.new_float(N, na)
    error_covariance = np.full(N, 100.0)  # Equivalent to array.new_float(N, 100.0)
    
    # Initialize filtered prices array
    filtered_prices = np.zeros(n)
    trend_signals = np.zeros(n)
    
    # Initialize state estimates (equivalent to f_init function)
    if np.isnan(state_estimate[0]):
        for i in range(N):
            state_estimate[i] = prices[0]
            error_covariance[i] = 1.0
    
    # Main Kalman filter loop
    for t in range(n):
        current_price = prices[t]
        
        # Prediction Step (equivalent to f_kalman function)
        predicted_state_estimate = np.zeros(N)
        predicted_error_covariance = np.zeros(N)
        
        for i in range(N):
            predicted_state_estimate[i] = state_estimate[i]  # Simplified prediction
            predicted_error_covariance[i] = error_covariance[i] + process_noise
        
        # Update Step
        kalman_gain = np.zeros(N)
        for i in range(N):
            kg = predicted_error_covariance[i] / (predicted_error_covariance[i] + measurement_noise)
            kalman_gain[i] = kg
            state_estimate[i] = predicted_state_estimate[i] + kg * (current_price - predicted_state_estimate[i])
            error_covariance[i] = (1 - kg) * predicted_error_covariance[i]
        
        # Return first element like Pine Script: array.get(stateEstimate, 0)
        filtered_prices[t] = state_estimate[0]
        
        # Trend detection (equivalent to Pine Script trend logic)
        if t > 0:
            if filtered_prices[t] > filtered_prices[t-1]:
                trend_signals[t] = 1
            elif filtered_prices[t] < filtered_prices[t-1]:
                trend_signals[t] = -1
            else:
                trend_signals[t] = 0
    
    return filtered_prices, trend_signals

def kalman_trend_signal(prices):

    _, trend_signals = kalman_price_filter(prices)
    return trend_signals[-1],trend_signals