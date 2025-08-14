import json

def create_index(config):
    
    index_data = []

    # Load identifiers from the identifiers file
    with open(config["identifiers_file_path"], 'r') as f:
        identifiers = json.load(f)
    
    for identifier in identifiers:
        normalized_prices = []
        
        # Load normalized price for each asset
        for asset_name, asset_config in config["assets"].items():
            with open(asset_config["normalized_history"], 'r') as f:
                normalized_data = json.load(f)
            
            # Find the normalized price for this identifier
            for entry in normalized_data:
                if str(entry["time"]) == str(identifier):
                    normalized_prices.append(entry["normalized_price"])
                    break
        
        # Calculate average if we have prices
        if normalized_prices:
            index_value = sum(normalized_prices) / len(normalized_prices)
            index_data.append({
                "identifier": identifier,
                "index": index_value
            })
    
    # Write index data to file
    with open(config["index_file_path"], 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"Index created with {len(index_data)} entries and saved to {config['index_file_path']}")
