import os
import json
from pathlib import Path
from trend_filters.slope import apply_trend_slope_filter
from trend_filters.kalman import kalman_filter_backquant

def get_normalized_price_for_identifier(asset_name, asset_config, identifier):
    # Ensure identifier is treated as string
    identifier = str(identifier)
    
    with open(asset_config["normalized_history"], 'r') as f:
        normalized_data = json.load(f)
    
    # Find the entry for this identifier
    for entry in normalized_data:
        # Ensure time field is also treated as string for comparison
        if str(entry["time"]) == identifier:
            return entry["normalized_price"]
    
    return None


def run_tournament_round(identifier, config, assets_to_consider):
    """Run a single tournament round for the given identifier and assets"""
    
    # Get normalized prices for all assets in this round
    asset_prices = {}
    for asset_name in assets_to_consider:
        asset_config = config["assets"][asset_name]
        normalized_price = get_normalized_price_for_identifier(asset_name, asset_config, identifier)
        asset_prices[asset_name] = normalized_price

    
    # Calculate the index (average of all normalized prices)
    index = sum(asset_prices.values()) / len(asset_prices)

    # Apply trend filtering
    if config["prices_slope_filtering_enabled"]:
        # Filter out assets that are not trending upward
        trending_assets = []
        
        for asset_name in assets_to_consider:
            asset_config = config["assets"][asset_name]
            
            # Load all normalized prices for this asset
            with open(asset_config["normalized_history"], 'r') as f:
                normalized_data = json.load(f)
            
            # Filter normalized prices up to the current identifier
            # Include every value until we hit the current identifier, then stop
            filtered_prices = []
            for entry in normalized_data:
                filtered_prices.append(entry["current_price"])
                # Stop when we reach the current identifier
                if str(entry["time"]) == str(identifier):
                    break
    
            
            # Apply slope filter to determine if asset is trending
            kalmans,kalman_signal = kalman_filter_backquant(filtered_prices)
            
            # Consider asset trending if slope is positive (upward trend)
            if kalman_signal > 0:
                trending_assets.append(asset_name)
            else:
                print(f"Asset {asset_name} filtered out - slope: {kalman_signal} (not trending upward)")
        
        # Update assets_to_consider to only include trending assets
        if trending_assets:
            assets_to_consider = trending_assets
            print(f"Trend filtering applied: {len(trending_assets)} assets remain after filtering")
            
            # Filter the existing asset_prices to only include trending assets
            asset_prices = {asset: asset_prices[asset] for asset in trending_assets}
            
            # Recalculate the index with the filtered assets
            index = sum(asset_prices.values()) / len(asset_prices)
        else:
            print("No assets passed trend filtering - declaring cash as winner")
            # Return cash as the winner and empty ratios
            return [None], {}
    # Calculate ratios and find assets that advance (ratio > 1)
    advancing_assets = []
    asset_ratios = {}
    if config["use_ema_filtering_price_normalization"]:
        # this means we are using the ema method to filter assets, we don't create ratios, we directly compare against the index
        for asset_name, normalized_price in asset_prices.items():
            if normalized_price > index:
                advancing_assets.append(asset_name)
        
        # If no assets have ratio > 1, advance the asset with highest ratio
        if not advancing_assets:
            # Compare normalized prices since asset_prices contains normalized values from the EMA calculation
            best_asset = max(asset_prices.items(), key=lambda x: x[1])  # x[1] is the normalized price
            advancing_assets = [best_asset[0]]
    else:
        # this is the standard original method
        for asset_name, normalized_price in asset_prices.items():
            ratio = normalized_price / index
            asset_ratios[asset_name] = ratio
            if ratio > 1:
                advancing_assets.append(asset_name)
        
        # If no assets have ratio > 1, advance the asset with highest ratio
        if not advancing_assets:
            best_asset = max(asset_ratios.items(), key=lambda x: x[1])
            advancing_assets = [best_asset[0]]
    
    return advancing_assets, asset_ratios

def run_tournament_for_identifier(identifier, config):
    # Start with all assets from config
    current_assets = list(config["assets"].keys())
    round_number = 1
    tournament_results = []
    
    # Run rounds until we have 1 or 0 assets left
    while len(current_assets) > 1:
        print(f"Round {round_number}: {len(current_assets)} assets competing")
        
        # Run the round
        advancing_assets, asset_ratios = run_tournament_round(identifier, config, current_assets)
        
        # Record round results
        round_result = {
            "round": round_number,
            "assets_competing": current_assets,
            "asset_ratios": asset_ratios,
            "advancing_assets": advancing_assets,
            "eliminated_assets": [asset for asset in current_assets if asset not in advancing_assets]
        }
        tournament_results.append(round_result)
        
        # Update assets for next round
        current_assets = advancing_assets
        round_number += 1
    
    # Final result
    final_result = {
        "tournament_identifier": identifier,
        "total_rounds": round_number - 1,
        "winner": current_assets[0] if current_assets else None,
        "rounds": tournament_results
    }
    
    print(f"Tournament completed! Winner: {final_result['winner']}")
    return final_result
    
def run_tournament(config):
    # Get all identifiers from the identifiers file
    with open(config["identifiers_file_path"], 'r') as f:
        identifiers = json.load(f)
    
    # Run tournament for each identifier and collect results
    all_tournament_results = []
    
    for identifier in identifiers:
        print(f"Running tournament for identifier: {identifier}")
        result = run_tournament_for_identifier(identifier, config)
        all_tournament_results.append(result)
    
    # Create aggregated summary
    aggregated_summary = {
        "total_identifiers": len(identifiers),
        "tournaments": all_tournament_results,
        "summary_stats": {
            "total_tournaments": len(all_tournament_results),
            "winners_by_asset": {}
        }
    }
    
    # Count winners by asset
    for result in all_tournament_results:
        winner = result["winner"]
        if winner:
            aggregated_summary["summary_stats"]["winners_by_asset"][winner] = \
                aggregated_summary["summary_stats"]["winners_by_asset"].get(winner, 0) + 1
    
    # Save aggregated results directly to the specified file
    with open(config["tournament_data_path"], 'w') as f:
        json.dump(aggregated_summary, f, indent=2)
    
    print(f"Aggregated tournament results saved to: {config['tournament_data_path']}")
    print(f"Summary: {len(identifiers)} tournaments completed")
    
    # Print winner summary
    print("\nWinner Summary:")
    for asset, wins in aggregated_summary["summary_stats"]["winners_by_asset"].items():
        print(f"  {asset}: {wins} wins")
    
    return aggregated_summary
    
