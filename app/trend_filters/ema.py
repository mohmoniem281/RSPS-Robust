import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any

def calculate_ema(prices, config: Dict[str, Any]):
    """
    Calculate the Exponential Moving Average (EMA) for a given price series.
    
    Parameters:
    - prices: List or Pandas Series of prices.
    - window: The EMA window (integer).
    
    Returns:
    - Pandas Series with EMA values.
    """

    window = config.get("trend_filters_settings", {}).get("ema", {}).get("window_size")
    
    prices_series = pd.Series(prices)
    ema = prices_series.ewm(span=window, adjust=False).mean()
    return ema