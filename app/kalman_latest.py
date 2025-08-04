import numpy as np
import pandas as pd

def kalman_filter_backquant(prices, process_noise=0.01, measurement_noise=3.0, filter_order=5):
    n = len(prices)
    state_estimates = np.full((n, filter_order), np.nan)
    error_covariances = np.full((n, filter_order), 1.0)

    # Initialization
    for i in range(filter_order):
        state_estimates[0, i] = prices[0]

    kalman_filtered = np.full(n, np.nan)
    trend_signal = np.full(n, 0)

    for t in range(1, n):
        predicted_state = state_estimates[t-1].copy()
        predicted_cov = error_covariances[t-1] + process_noise

        kalman_gain = predicted_cov / (predicted_cov + measurement_noise)
        new_state = predicted_state + kalman_gain * (prices[t] - predicted_state)
        new_cov = (1 - kalman_gain) * predicted_cov

        state_estimates[t] = new_state
        error_covariances[t] = new_cov

        kalman_filtered[t] = new_state[0]

        # Trend detection (same logic as Pine)
        if t > 1:
            if kalman_filtered[t] > kalman_filtered[t-1]:
                trend_signal[t] = 1
            elif kalman_filtered[t] < kalman_filtered[t-1]:
                trend_signal[t] = -1

    return kalman_filtered, trend_signal[-1]