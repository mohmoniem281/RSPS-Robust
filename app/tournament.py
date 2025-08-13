import os
import json
from pathlib import Path
from trend_filters.kalman import kalman_filter_backquant
from trend_filters.trend_manager import is_asset_trending_layer_1,is_ratio_trending_layer_2
from ratios import get_ratios



def run_tournament_for_identifier(identifier, config):
    highest_r2_ratio=None
    highest_r2_asset=None
    winner_asset=None
    # Start with all assets from config
    current_assets = list(config["assets"].keys())
    round_number = 1
    tournament_results = []
    trending_assets = []
    trending_ratios = []
    # filter out non trending assets
    for asset_name in current_assets:
        is_trending, latest_r2 = is_asset_trending_layer_1(config,identifier,asset_name)
        if is_trending:
            trending_assets.append((asset_name, latest_r2))
   

    if len(trending_assets) == 0:
        print("No assets are trending. Declaring cash as winner for identifier: ", identifier)
        winner_asset = None
    else:
        #get the last r2 of the trending assets
        trending_assets.sort(key=lambda x: x[1])  # ascending by r2
        highest_r2_asset = trending_assets[-1][0]   
        
        for asset_tuple in trending_assets:
            asset_name = asset_tuple[0]  # Extract just the asset name from the tuple
            is_ratio_trending,ratio_r2 = is_ratio_trending_layer_2(config,identifier,asset_name)
            if is_ratio_trending:
                trending_ratios.append((asset_name, ratio_r2))

        if len(trending_ratios) == 0:
            print("No ratios are trending. Since there are trending assets, we will use the highest r2 asset as winner for identifier: ", identifier)
            winner_asset = highest_r2_asset
        else:
            trending_ratios.sort(key=lambda x: x[1])  # ascending by ratio_r2
            highest_r2_ratio = trending_ratios[-1][0] 
            winner_asset = highest_r2_ratio 
    
    
    
    # Record round results
    round_result = {
        "round": 0,
        "assets_competing": current_assets,
        "asset_ratios": trending_ratios,
        "advancing_assets": None,
        "eliminated_assets": None
    }
    tournament_results.append(round_result)
    
    
    # Final result
    final_result = {
        "tournament_identifier": identifier,
        "total_rounds": 1,
        "winner": winner_asset,
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
    
