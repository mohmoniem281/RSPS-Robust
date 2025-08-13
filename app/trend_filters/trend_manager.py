import math
import json
from pathlib import Path
from . import slope_and_r2
from . import kalman

def is_asset_trending_layer_1(config, identifier, asset_name):
    
    asset_data = config.get("assets", {}).get(asset_name, {})
    price_history_path = asset_data['price_history']
    # Load the actual price history data from the file
    with open(price_history_path, 'r') as f:
        price_history = json.load(f)
    # Filter price history up to the identifier (inclusive)
    filtered_history = []
    for entry in price_history:
        filtered_history.append(entry)
        # Stop when we reach the target identifier
        if entry['time'] == identifier:
            break
    # Extract only the prices (close values)
    prices_only = [entry['close'] for entry in filtered_history]
    # prices_only now contains only the price values up to the identifier for this asset

    #logarithm the prices_only
    prices_only_log = [math.log(price) for price in prices_only]

    #apply kalman trend filter to the price series (logarithm)
    # WE SHOULD IGNORE KALMAN_SIGNAL HERE FORM THE INDICATOR, NOT RELIABE, WE WILL DO SLOPE AND R2 INSTEAD
    process_noise = config.get("trend_filters_settings", {}).get("layer_1", {}).get("kalman", {}).get("process_noise")
    measurement_noise = config.get("trend_filters_settings", {}).get("layer_1", {}).get("kalman", {}).get("measurement_noise")
    filter_order = config.get("trend_filters_settings", {}).get("layer_1", {}).get("kalman", {}).get("filter_order")
    kalman_trend,kalman_signal = kalman.kalman_filter_backquant(prices_only_log, process_noise, measurement_noise, filter_order)
    
    #apply slope and r2 trend filters to the kalman trend
    trend_slope_window = config.get("trend_filters_settings", {}).get("layer_1", {}).get("slope_and_r2", {}).get("trend_slope_window")

    slope_series, r2_series, last_slope, last_r2 = slope_and_r2.rolling_slope_and_r2(kalman_trend, trend_slope_window)

    min_slope = config.get("trend_filters_settings", {}).get("layer_1", {}).get("slope_and_r2", {}).get("slope_threshold")
    r2_threshold = config.get("trend_filters_settings", {}).get("layer_1", {}).get("slope_and_r2", {}).get("r2_threshold")
    
    #check if the trend is bullish or bearish
    if last_r2 < r2_threshold or last_slope < min_slope:
        return False, last_r2

    return True, last_r2

def is_ratio_trending_layer_2(config, identifier, asset_name):
    # Extract ratio data up to the identifier for this asset
    ratios_file_path = config.get("ratios_file_path")
    with open(ratios_file_path, 'r') as f:
        ratios_data = json.load(f)
    
    # Filter ratios data up to the identifier (inclusive)
    filtered_ratios = []
    for entry in ratios_data:
        filtered_ratios.append(entry["assets"][asset_name]["ratio"])
        # Stop when we reach the target identifier
        if entry["identifier"] == identifier:
            break
    
    # Apply logarithm to the ratios
    ratios_only_log = [math.log(ratio) for ratio in filtered_ratios]

    #apply kalman trend filter to the price series (logarithm)
    # WE SHOULD IGNORE KALMAN_SIGNAL HERE FORM THE INDICATOR, NOT RELIABE, WE WILL DO SLOPE AND R2 INSTEAD
    process_noise = config.get("trend_filters_settings", {}).get("layer_2", {}).get("kalman", {}).get("process_noise")
    measurement_noise = config.get("trend_filters_settings", {}).get("layer_2", {}).get("kalman", {}).get("measurement_noise")
    filter_order = config.get("trend_filters_settings", {}).get("layer_2", {}).get("kalman", {}).get("filter_order")
    kalman_trend,kalman_signal = kalman.kalman_filter_backquant(ratios_only_log, process_noise, measurement_noise, filter_order)
    
    #apply slope and r2 trend filters to the kalman trend
    trend_slope_window = config.get("trend_filters_settings", {}).get("layer_2", {}).get("slope_and_r2", {}).get("trend_slope_window")
    slope_series, r2_series, last_slope, last_r2 = slope_and_r2.rolling_slope_and_r2(kalman_trend, trend_slope_window)

    min_slope = config.get("trend_filters_settings", {}).get("layer_2", {}).get("slope_and_r2", {}).get("slope_threshold")
    r2_threshold = config.get("trend_filters_settings", {}).get("layer_2", {}).get("slope_and_r2", {}).get("r2_threshold")
    
    #check if the trend is bullish or bearish
    if last_r2 < r2_threshold or last_slope < min_slope:
        return False, last_r2

    return True, last_r2
