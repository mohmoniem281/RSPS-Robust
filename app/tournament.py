import os
import json
from pathlib import Path

def get_normalized_price_for_identifier(asset_name, asset_config, identifier):
    """Get the normalized price for a specific asset and identifier"""
    try:
        with open(asset_config["normalized_history"], 'r') as f:
            normalized_data = json.load(f)
        
        # Find the entry for this identifier
        for entry in normalized_data:
            if entry["time"] == identifier:
                return entry["normalized_price"]
        
        return None
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def run_tournament_round(identifier, config, assets_to_consider):
    """Run a single tournament round for the given identifier and assets"""
    
    # Get normalized prices for all assets in this round
    asset_prices = {}
    for asset_name in assets_to_consider:
        asset_config = config["assets"][asset_name]
        normalized_price = get_normalized_price_for_identifier(asset_name, asset_config, identifier)
        
        if normalized_price is not None:
            asset_prices[asset_name] = normalized_price
    
    if not asset_prices:
        return [], {}
    
    # Calculate the index (average of all normalized prices)
    index = sum(asset_prices.values()) / len(asset_prices)
    
    # Calculate ratios and find assets that advance (ratio > 1)
    advancing_assets = []
    asset_ratios = {}
    for asset_name, normalized_price in asset_prices.items():
        ratio = normalized_price / index
        asset_ratios[asset_name] = ratio
        if ratio > 1:
            advancing_assets.append(asset_name)
    
    return advancing_assets, asset_ratios

def run_tournament(identifier, config):
    # Create tournament data directory if it doesn't exist
    tournament_data_path = Path(config["tournament_data_path"])
    
    # Create a new directory for this specific tournament identifier
    tournament_dir = tournament_data_path / identifier
    tournament_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Save tournament results
    results_file = tournament_dir / "tournament_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_result, f, indent=2)
    
    print(f"Tournament completed! Winner: {final_result['winner']}")
    return final_result
    
    
    
