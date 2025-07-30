import numpy as np
import json
import os
from statsmodels.tsa.stattools import adfuller
from typing import List, Tuple, Union

def _load_config():
    """Load configuration from adf.json file."""
    config_path = os.path.join(os.path.dirname(__file__), 'adf.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('window'), config.get('p_threshold')


def is_mean_reverting(prices: Union[List[float], np.ndarray]) -> Tuple[bool, float]:
    # Load configuration
    window, p_threshold = _load_config()
    
    # Convert to numpy array if needed
    if isinstance(prices, list):
        prices = np.array(prices)
    
    # Validate inputs
    if len(prices) < window:
        raise ValueError(f"Not enough data. Need at least {window} values, got {len(prices)}.")
    
    if window < 10:
        raise ValueError("Window must be at least 10 for reliable ADF test.")
    
    # Take the last 'window' values
    recent_prices = prices[-window:]
    
    # Run ADF test
    result = adfuller(recent_prices)
    p_value = result[1]
    
    # Determine if mean-reverting (stationary)
    is_mean_reverting = p_value < p_threshold
    
    return is_mean_reverting, p_value