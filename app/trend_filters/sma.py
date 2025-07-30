import json
from typing import List, Optional
from pathlib import Path

def calculate_moving_average(values: List[float], period: int) -> Optional[float]:
    """Calculate simple moving average."""
    if len(values) < period:
        return None
    
    recent_values = values[-period:]
    return sum(recent_values) / len(recent_values)

def apply_sma_trend_filter(capital_values: List[float]) -> tuple[bool, float]:
    """Apply SMA trend filter by loading period from sma.json"""
    config_path = Path(__file__).parent / "sma.json"
    with open(config_path, 'r') as f:
        sma_config = json.load(f)
    sma_period = sma_config.get("period")
    
    sma = calculate_moving_average(capital_values, sma_period)
    if not sma:
        return True, 0.0  # Not enough data for SMA calculation, return true to enter trades
    
    # Check if current value is above SMA (bullish trend)
    return capital_values[-1] > sma, sma