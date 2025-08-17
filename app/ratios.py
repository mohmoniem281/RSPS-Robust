import json

def create_ratios(config):


        # Load identifiers from the identifiers file
    with open(config["identifiers_file_path"], 'r') as f:
        identifiers = json.load(f)

    # Load index data
    with open(config["index_file_path"], 'r') as f:
        index_data = json.load(f)
    
    # Create a dictionary for quick index lookup by identifier
    index_lookup = {str(entry["identifier"]): entry["index"] for entry in index_data}
    
    ratios_data = []
    
    for identifier in identifiers:
        # Get index value for this identifier
        index_value = index_lookup.get(str(identifier))
        
        # Process each asset for this identifier
        for asset_name, asset_config in config["assets"].items():
            # Load normalized price data for this asset
            with open(asset_config["normalized_history"], 'r') as f:
                normalized_data = json.load(f)
            
            # Find the normalized price for this identifier
            normalized_price = None
            for entry in normalized_data:
                if str(entry["time"]) == str(identifier):
                    normalized_price = entry["normalized_price"]
                    break
            
            # # Calculate ratio
            # ratio = normalized_price / index_value

            #calcualte ratio. now based on log 
            ratio = normalized_price - index_value
            
            # Find or create entry for this identifier
            identifier_entry = None
            for entry in ratios_data:
                if entry["identifier"] == identifier:
                    identifier_entry = entry
                    break
            
            if identifier_entry is None:
                identifier_entry = {
                    "identifier": identifier,
                    "index": index_value,
                    "assets": {}
                }
                ratios_data.append(identifier_entry)
            
            # Add asset data to this identifier entry
            identifier_entry["assets"][asset_name] = {
                "normalized_price": normalized_price,
                "ratio": ratio
            }
    
    # Write ratios data to file
    with open(config["ratios_file_path"], 'w') as f:
        json.dump(ratios_data, f, indent=2)
    
    print(f"Ratios created with {len(ratios_data)} entries and saved to {config['ratios_file_path']}")

def get_ratios(identifier, config, asset_name):
    with open(config["ratios_file_path"], 'r') as f:
        ratios_data = json.load(f)
    
    # Convert identifier to string for consistent comparison
    target_identifier = str(identifier)
    
    # Start from the top and collect ratios until we find the identifier
    ratios = []
    for entry in ratios_data:
    
        ratios.append(entry["assets"][asset_name]["ratio"])
        
        # Check if we've reached the target identifier
        if str(entry["identifier"]) == target_identifier:
            # Found the identifier, return all collected ratios and exit
            return ratios
    
    # Identifier not found, return empty list
    return []