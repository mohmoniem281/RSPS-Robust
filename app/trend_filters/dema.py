import json
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import pandas_ta as ta
import numpy as np

def apply_dema_trend_filter(capital_values: List[float]) -> tuple[bool, float]:
    """
    Apply DEMA trend filter using pandas_ta DEMA calculation
    
    Args:
        capital_values: List of capital/price values
        
    Returns:
        tuple: (is_bullish, current_dema_value)
    """
    config_path = Path(__file__).parent / "dema.json"
    with open(config_path, 'r') as f:
        dema_config = json.load(f)
    dema_period = dema_config.get("dema_period")
    
    # Convert to pandas DataFrame for DEMA calculation
    df = pd.DataFrame({'close': capital_values})
    
    # Calculate DEMA using pandas_ta
    df['dema'] = ta.dema(df['close'], length=dema_period)
    
    # Get the most recent DEMA value
    current_dema = df['dema'].iloc[-1]
    current_price = capital_values[-1]
    
    # Handle insufficient data - if DEMA is NaN/None, assume bullish trend
    # This allows trading to continue while building up enough history
    if pd.isna(current_dema) or current_dema is None:
        print("Forcing dema to true sense there is no sufficient history")
        # When not enough history, assume bullish trend to allow trading
        # Use current price as DEMA value for consistency
        return True, current_price
    
    # Check if current value is above DEMA (bullish trend)
    return current_price > current_dema, current_dema
        