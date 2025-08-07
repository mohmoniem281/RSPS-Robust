import pandas as pd
import json
from pathlib import Path

def calculate_ema(prices):
    """
    Calculate the Exponential Moving Average (EMA) for a given price series.
    
    Parameters:
    - prices: List or Pandas Series of prices.
    - window: The EMA window (integer).
    
    Returns:
    - Pandas Series with EMA values.
    """
    # Load window from config if not provided
    config_path = Path(__file__).parent / "ema.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    window = config['window_size']
    
    prices_series = pd.Series(prices)
    ema = prices_series.ewm(span=window, adjust=False).mean()
    return ema