import json
from typing import Dict, List, Optional
from pathlib import Path

def calculate_ema(values: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    ema = []
    k = 2 / (period + 1)

    for i, val in enumerate(values):
        if i == 0:
            ema.append(val)
        else:
            ema.append(val * k + ema[-1] * (1 - k))
    return ema

def calculate_dema(values: List[float], period: int) -> Optional[List[float]]:
    """Calculate Double Exponential Moving Average (DEMA)
    
    DEMA = 2 * EMA(period) - EMA(EMA(period))
    """
    if len(values) < period:
        return None  # Not enough data
    
    # Calculate first EMA
    ema1 = calculate_ema(values, period)
    
    # Calculate second EMA (EMA of EMA1)
    ema2 = calculate_ema(ema1, period)
    
    # DEMA formula: 2 * EMA1 - EMA2
    dema = [2 * e1 - e2 for e1, e2 in zip(ema1, ema2)]
    return dema

def apply_dema_trend_filter(capital_values: List[float], config: Dict[str, Any]) -> tuple[bool, float]:
    
    dema_period = config.get("trend_filters_settings", {}).get("dema", {}).get("dema_period")
    
    dema = calculate_dema(capital_values, dema_period)
    if not dema:
        return True, 0.0  # Not enough data for DEMA calculation IF there is not enough data for DEMA calculation we will just return true and enter trades
    
    # Check if current value is above DEMA (bullish trend)
    return capital_values[-1] > dema[-1], dema[-1]