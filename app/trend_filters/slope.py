import json
from pathlib import Path
import numpy as np

def calculate_trend_slope(values, slope_window):
    if len(values) < 2:
        return 0.0

    # Get the most recent slope_window values
    recent_values = values[-slope_window:] if len(values) >= slope_window else values

    if len(recent_values) < 2:
        return 0.0

    # Linear regression: fit y = mx + b, return m (slope)
    x = np.arange(len(recent_values))  # time steps: [0, 1, 2, ..., N]
    y = np.array(recent_values)

    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

    return slope

def get_slope_window_from_config():
    config_path = Path(__file__).parent / "slope.json"

    with open(config_path, 'r') as f:
        config = json.load(f)
        return config.get("trend_slope")

def apply_trend_slope_filter(values):
    slope_window = get_slope_window_from_config()
    return calculate_trend_slope(values, slope_window)