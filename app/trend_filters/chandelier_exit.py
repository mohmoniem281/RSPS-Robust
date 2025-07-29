import pandas as pd
import numpy as np
import json
from pathlib import Path

def chandelier_exit_close_only(price_list):
    """
    Chandelier Exit trend filter using close-only data.

    Parameters:
    - close: pd.Series (e.g., equity curve or normalized prices)
    - atr_period: int (ATR-like lookback period)
    - atr_mult: float (volatility multiplier)

    Returns:
    - pd.Series of +1 (long), -1 (short)
    """
    # Load settings from chandelier_exit.json
    config_path = Path(__file__).parent / "chandelier_exit.json"
    with open(config_path) as f:
        config = json.load(f)
    atr_period = config.get("atr_period")
    atr_mult = config.get("atr_mult")

    close = pd.Series(price_list)

    returns = close.diff().abs()
    atr = returns.rolling(atr_period).mean() * atr_mult

    highest_close = close.rolling(atr_period).max()
    lowest_close = close.rolling(atr_period).min()

    long_stop = highest_close - atr
    short_stop = lowest_close + atr

    long_stop_smoothed = [np.nan]
    short_stop_smoothed = [np.nan]

    for i in range(1, len(close)):
        prev_long = long_stop_smoothed[-1] if not np.isnan(long_stop_smoothed[-1]) else long_stop.iloc[i-1]
        prev_short = short_stop_smoothed[-1] if not np.isnan(short_stop_smoothed[-1]) else short_stop.iloc[i-1]

        current_long = max(long_stop.iloc[i], prev_long) if close.iloc[i-1] > prev_long else long_stop.iloc[i]
        current_short = min(short_stop.iloc[i], prev_short) if close.iloc[i-1] < prev_short else short_stop.iloc[i]

        long_stop_smoothed.append(current_long)
        short_stop_smoothed.append(current_short)

    long_stop_smoothed = pd.Series(long_stop_smoothed, index=close.index)
    short_stop_smoothed = pd.Series(short_stop_smoothed, index=close.index)

    dir_series = pd.Series(index=close.index, dtype='float')
    dir_series.iloc[0] = 1  # Start long

    for i in range(1, len(close)):
        if close.iloc[i] > short_stop_smoothed.iloc[i-1]:
            dir_series.iloc[i] = 1
        elif close.iloc[i] < long_stop_smoothed.iloc[i-1]:
            dir_series.iloc[i] = -1
        else:
            dir_series.iloc[i] = dir_series.iloc[i-1]

    return dir_series

