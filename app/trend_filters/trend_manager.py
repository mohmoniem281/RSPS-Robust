import kalman

import json
from pathlib import Path

def update_config_trending_assets_only(config, identifier):
    trending_assets = []
    for asset_name, asset_data in config['assets'].items():
        price_history_path = asset_data['price_history']
        # Load the actual price history data from the file
        with open(price_history_path, 'r') as f:
            price_history = json.load(f)
        # Filter price history up to the identifier
        filtered_history = [entry for entry in price_history if entry['time'] <= identifier]
        # Extract only the prices (close values)
        prices_only = [entry['close'] for entry in filtered_history]
        # prices_only now contains only the price values up to the identifier for this asset

        #apply kalman trend filter to the price series
        kalman_trend = kalman.kalman_filter_backquant(prices_only)
        # kalman_trend now contains the trend values for this asset

        #check if the trend is bullish or bearish
        if kalman_trend[-1] > 0:
            trending_assets.append(asset_name)

    #update the config to only include the trending assets
    config['assets'] = {asset_name: config['assets'][asset_name] for asset_name in trending_assets}

    return config

