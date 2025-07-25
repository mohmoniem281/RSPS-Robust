

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json
from typing import Optional
from identifiers import extract_identifiers, filter_identifiers
from price_to_json import price_to_json
from price_normalization import normalize_prices
from tournament import run_tournament
from build_equity_curve import build_the_equity_curve, apply_trend_filters
from equity_curve_visualizer import create_equity_curve_visualization, create_filtered_equity_curve_visualization
def load_config():

    config_path = Path(__file__).parent / "config.json"  
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
   
    
    print("🏆 RSPS-Robust DAILY TRADING STRATEGY")
    print("=" * 60)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load config
    config = load_config()
    print(f"📁 Loaded config successfully!")
    
    # Convert CSV to JSON
    price_to_json(config)
    print(f"✅ CSV converted to JSON successfully!")
    
    # Extract identifiers
    extract_identifiers(config)
    print(f"✅ Identifiers extracted successfully!")

    # Normalize prices
    normalize_prices(config)
    print(f"✅ Prices normalized successfully!")

    # Filter identifiers
    filter_identifiers(config)
    print(f"✅ Identifiers filtered successfully!")

    # Run tournament
    run_tournament(config)
    print(f"✅ Tournament completed successfully!")

    # Build equity curve
    build_the_equity_curve(config)
    print(f"✅ Equity curve built successfully!")

    # Build equity curve visualization
    create_equity_curve_visualization(config)
    print(f"✅ Equity curve visualization built successfully!")

    # Apply trend filters to the equity curve
    if config["equity_curve_dema_filtering_enabled"]:
        apply_trend_filters(config)
        print(f"✅ Trend filters applied successfully!")
        # Build equity curve visualization filtered
        create_filtered_equity_curve_visualization(config)
        print(f"✅ Equity curve visualization filtered built successfully!")


if __name__ == "__main__":
    main()
