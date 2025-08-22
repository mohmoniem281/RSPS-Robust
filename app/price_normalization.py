import json
import os
from pathlib import Path
from trend_filters.ema import calculate_ema
import math

def normalize_prices(config):
    """
    Normalize prices for all assets based on the normalization window.
    For each identifier, divide the current price by the price from normalization_window periods ago.
    """
    
    # Load identifiers
    with open(config["identifiers_file_path"], 'r') as f:
        identifiers = json.load(f)
    
    normalization_window = config["normalization_window"]
    
    # Process each asset
    for asset_name, asset_config in config["assets"].items():
        print(f"Processing asset: {asset_name}")
        
        # Load price history for this asset
        with open(asset_config["price_history"], 'r') as f:
            price_history = json.load(f)
        
        # Create a dictionary for quick price lookup by time
        price_lookup = {entry["time"]: entry["close"] for entry in price_history}
        
        # Initialize normalized prices list
        normalized_prices = []
        
        #using the normalization window (original method)
        # Process each identifier
        for i, identifier in enumerate(identifiers):
            current_price = price_lookup.get(identifier)
            
            if current_price is None:
                print(f"Warning: No price found for identifier {identifier} in asset {asset_name}")
                continue
            
            # Find the historical price (normalization_window periods ago)
            if i >= normalization_window:
                historical_identifier = identifiers[i - normalization_window]
                historical_price = price_lookup.get(historical_identifier)
                
                if historical_price is None:
                    print(f"Warning: No historical price found for identifier {historical_identifier} in asset {asset_name}")
                    continue
                
                if historical_price == 0:
                    print(f"Warning: Historical price is zero for identifier {historical_identifier} in asset {asset_name}")
                    continue
                
                # # Calculate normalized price
                # normalized_price = current_price / historical_price

                #normalized prices are now log
                normalized_price = math.log(current_price) - math.log(historical_price)
                
                normalized_prices.append({
                    "time": identifier,
                    "normalized_price": normalized_price,
                    "current_price": current_price,
                    "historical_price": historical_price,
                    "historical_identifier": historical_identifier
                })
            else:
                # For the first normalization_window entries, we can't normalize
                # We'll skip these or set them to 1.0 (no change)
                print(f"Skipping identifier {identifier} (index {i}) - insufficient history for normalization")
        
        
        # Save normalized prices to file
        output_path = asset_config["normalized_history"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(normalized_prices, f, indent=2)
        
        print(f"Saved {len(normalized_prices)} normalized prices for {asset_name} to {output_path}")

def main():
    """Main function to run price normalization"""
    # Load configuration
    with open('app/config.json', 'r') as f:
        config = json.load(f)
    
    # Run normalization
    normalize_prices(config)
    print("Price normalization completed!")

if __name__ == "__main__":
    main()
