import kalman
import math
import json
from pathlib import Path

def is_asset_trending_layer_1(config, identifier, asset_name):
    bool trending=False
    
    price_history_path = asset_data['price_history']
    # Load the actual price history data from the file
    with open(price_history_path, 'r') as f:
        price_history = json.load(f)
    # Filter price history up to the identifier
    filtered_history = [entry for entry in price_history if entry['time'] <= identifier]
    # Extract only the prices (close values)
    prices_only = [entry['close'] for entry in filtered_history]
    # prices_only now contains only the price values up to the identifier for this asset

    #logarithm the prices_only
    prices_only_log = [math.log(price) for price in prices_only]

    #apply kalman trend filter to the price series (logarithm)
    kalman_trend,kalman_signal = kalman.kalman_filter_backquant(prices_only_log, config)
    
    #apply r2 trend filter to the price series (logarithm)
    r2_trend, r2_trend_value = r2.rolling_r2_from_series(kalman_trend, config)
    r2_threshold = config.get("trend_filters_settings", {}).get("r2", {}).get("r2_threshold", 0.8)
    
    #check if the trend is bullish or bearish
    if kalman_signal > 0 and r2_trend_value > r2_threshold:
        trending = True

    return trending

    
def get_trending_ratios_with_strength(config, log_ratios):
    bool trending=False

    #apply kalman trend filter to the price series (logarithm)
    kalman_trend,kalman_signal = kalman.kalman_filter_backquant(log_ratios, config)
    
    #apply r2 trend filter to the price series (logarithm)
    r2_trend, r2_trend_value = r2.rolling_r2_from_series(kalman_trend, config)
    
    #check if the trend is bullish or bearish
    if kalman_signal > 0
        trending = True

    return trending, r2_trend_value