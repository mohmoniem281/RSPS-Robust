import json
import os
import pandas as pd
from pathlib import Path
import glob

def extract_identifiers(config):
    # Get the first asset from the assets dictionary
    first_asset = list(config["assets"].keys())[0]
    asset_config = config["assets"][first_asset]
    
    # Read the price history file
    with open(asset_config["price_history"], 'r') as f:
        price_data = json.load(f)
    
    # Extract all time values from the array of objects
    time_values = [entry["time"] for entry in price_data]
    
    # Write to the identifiers file
    with open(config["identifiers_file_path"], 'w') as f:
        json.dump(time_values, f, indent=2)

def filter_identifiers(config):
    """
    Filter identifiers based on the from and to identifiers specified in the config.
    This function reads the identifiers file and filters it based on the range.
    """
    # Read the identifiers file
    with open(config["identifiers_file_path"], 'r') as f:
        identifiers = json.load(f)
    
    # Get the from and to identifiers from config
    from_identifier = config["process_from_identifier_from_included"]
    to_identifier = config["process_to_identifier_to_included"]
    
    # Find the indices of the from and to identifiers
    from_index = identifiers.index(from_identifier)
    to_index = identifiers.index(to_identifier)
    
    # Extract the range (inclusive of both ends)
    filtered_identifiers = identifiers[from_index:to_index + 1]
    
    # Write the filtered identifiers back to the file
    with open(config["identifiers_file_path"], 'w') as f:
        json.dump(filtered_identifiers, f, indent=2)
    