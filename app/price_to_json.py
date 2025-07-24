import csv
import json
import os
from pathlib import Path

def price_to_json(config):
    input_path = config["input_prices_path"]
    
    # Assets to convert (based on config structure)
    assets = list(config["assets"].keys())
    
    for asset in assets:
        csv_file = os.path.join(input_path, f"{asset}.csv")
        json_file = config["assets"][asset]["price_history"]
  
        # Convert CSV to JSON
        convert_csv_to_json(csv_file, json_file)

def convert_csv_to_json(csv_path, json_path):
    data = []
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Replace colons with hyphens in all field values
            for key, value in row.items():
                if isinstance(value, str):
                    row[key] = value.replace(':', '-')
            
            # Convert close price to float
            row['close'] = float(row['close'])
            data.append(row)
    
    # Write JSON file
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2)
    
    print(f"Converted {csv_path} to {json_path} ({len(data)} records)")
    

