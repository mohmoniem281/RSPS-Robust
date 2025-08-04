import json
from typing import Dict, List, Any
from equity_curve_kalman_tpi import analyze_equity_curve_kalman_trend
from kalman_latest import kalman_filter_backquant

def build_the_equity_curve(config: Dict):
    equity_curve_non_filtered = []
    equity_curve_filtered = []

    # Step : Load tournament data from config
    tournament_data = get_tournament_data(config)

    #step : get identifiers
    identifiers = get_identifiers(config)

    #step: instantiate the equity curves. we will open a cash trade on the first identifier since we don't have a pre-determined position from the day before. 
    instantiate_equity_curves(config, equity_curve_non_filtered, equity_curve_filtered, identifiers[0])
    #step  process each identifier
    for i in range (0, len(identifiers)):

        # here we close the trade on i 
        close_trade(equity_curve_non_filtered, identifiers[i], config) # non filtered (reference)

        close_trade(equity_curve_filtered, identifiers[i], config) # filtered (actual)

        # we trend filter after closing the trade
        # Extract capital values from equity curve for trend analysis
        capital_values = [entry["capital"] for entry in equity_curve_non_filtered]
        
        is_trending = True

        # we open a new trade based on the trend filter (or none) results on i+1 (for the non filtered curve)
        if i+1 < len(identifiers):   # this means we are still processing days which we have data for and can get i+1
            open_new_trade(equity_curve_non_filtered, tournament_data, identifiers[i+1], identifiers[i], config, is_trending)
        else: # this means we have reached the end of the identifiers array and i+1 is out of range, we will replace it with "Current Signal"
            open_new_trade(equity_curve_non_filtered, tournament_data, "Current Signal", identifiers[i], config, is_trending)

        # is_trending, analysis_data  = analyze_equity_curve_kalman_trend(capital_values)
        values , signal = kalman_filter_backquant(capital_values)
        if signal >= 1:
            is_trending = True
        else:
            is_trending = False

        # we open a new trade based on the trend filter (or none) results on i+1 (for the filtered curve)
        if i+1 < len(identifiers):   # this means we are still processing days which we have data for and can get i+1
            open_new_trade(equity_curve_filtered, tournament_data, identifiers[i+1], identifiers[i], config, is_trending)
        else: # this means we have reached the end of the identifiers array and i+1 is out of range, we will replace it with "Current Signal"
            print("DEBUG - NON _ FILTERED DATA!!!!!!!!!!!!")
            print([entry["capital"] for entry in equity_curve_non_filtered])
            print("DEBUG - FILTERED DATA!!!!!!!!!!!!")
            print([entry["capital"] for entry in equity_curve_filtered])
            
            open_new_trade(equity_curve_filtered, tournament_data, "Current Signal", identifiers[i], config, is_trending)


    
    #save the equity curve
    save_equity_curve(equity_curve_filtered, config)

def save_equity_curve(equity_curve: List[Dict], config: Dict):
    equity_curve_path = config["equity_curve_file"]
    with open(equity_curve_path, 'w') as f:
        json.dump(equity_curve, f)






def get_tournament_data(config: Dict):
    tournament_data_path = config["tournament_data_path"]
    
    with open(tournament_data_path, 'r') as f:
        tournament_data = json.load(f)
    
    return tournament_data

def get_identifiers(config: Dict):

    identifiers_file_path = config["identifiers_file_path"]
    
    with open(identifiers_file_path, 'r') as f:
        identifiers = json.load(f)
        
    return identifiers

def instantiate_equity_curves(config: Dict, equity_curve_non_filtered: List[Dict], equity_curve_filtered: List[Dict], signal_bar:str):
    initial_capital = config["initial_capital"]

    #open cash positions on the first day
    equity_curve_non_filtered.append({
        "identifier": signal_bar,
        "capital": initial_capital,
        "pnl": 0,
        "return_pct": 0,
        "position": None,
        "entry_price": None,
        "exit_price": None,
        "winner_asset": None
    })


    # add the filtered 
        #open cash positions on the first day
    equity_curve_filtered.append({
        "identifier": signal_bar,
        "capital": initial_capital,
        "pnl": 0,
        "return_pct": 0,
        "position": None,
        "entry_price": None,
        "exit_price": None,
        "winner_asset": None
    })

def close_trade(equity_curve: List[Dict], identifier: str, config: Dict):
    #find the trade in the equity curve
    trade = next((trade for trade in equity_curve if trade["identifier"] == identifier), None)
    
    if trade["winner_asset"] is None: #means this was a cash position
        return                        # everything stays the same
    
    # here the trade was not cash
    #get the price of the winner asset on the identifier
    winner_asset_price_close = get_asset_price_close(config, trade["winner_asset"], identifier)

    #calculate the pnl
    pnl = (winner_asset_price_close * trade["position"]) - (trade["entry_price"] * trade["position"])
    #calculate the return percentage
    return_pct = (pnl / trade["capital"]) * 100
    #calculate the capital
    capital = trade["capital"] + pnl
    #calculate the position this represents teh amount in the asset being held, based on the new capital 
    position = capital / winner_asset_price_close

    #update the equity curve
    trade["pnl"] = pnl
    trade["return_pct"] = return_pct
    trade["exit_price"] = winner_asset_price_close
    trade["capital"] = capital
    trade["position"] = position



def open_new_trade(equity_curve: List[Dict], tournament_data: Dict, signal_bar: str, closed_tournament_bar: str, config:Dict, is_trending:bool):
    winner_asset = get_tournament_winner_asset(tournament_data, closed_tournament_bar)

    #get the price of the winner asset on the signal bar which will be from the closed tournament bar
    winner_asset_price_close = get_asset_price_close(config, winner_asset, closed_tournament_bar)   # this is the same as the open price on the actual signal day

    #get the capital from the closed trade
    closed_trade = next((trade for trade in equity_curve if trade["identifier"] == closed_tournament_bar), None)
    capital = closed_trade["capital"] # this was already updated when we closed the trade

    #open a new trade

    if winner_asset is None or not is_trending:   # means this is a cash position 
        #open cash positions on the signal day
        equity_curve.append({
            "identifier": signal_bar,
            "capital": capital,
            "pnl": 0,
            "return_pct": 0,
            "position": None,
            "entry_price": None,
            "exit_price": None,
            "winner_asset": None
        })
        return 

    #there was an actual winner asset
    #open a new trade
    position = capital / winner_asset_price_close
    equity_curve.append({
        "identifier": signal_bar,
        "capital": capital,
        "pnl": 0,
        "return_pct": 0,
        "position": position,
        "entry_price": winner_asset_price_close,
        "exit_price": None,
        "winner_asset": winner_asset
    })


def get_tournament_winner_asset(tournament_data: Dict, tournament_bar: str):
    # Find the tournament for the given date
    tournament = next((t for t in tournament_data["tournaments"] if t["tournament_identifier"] == tournament_bar), None)
    
    if tournament is None:
        return None
        
    return tournament["winner"]

def get_asset_price_close(config: Dict, asset: str, bar: str):
    # Get the price history file path for the asset
    price_history_path = config["assets"][asset]["price_history"]
    
    # Load the price history
    with open(price_history_path, 'r') as f:
        price_history = json.load(f)
    
    # Find the price for the given bar
    price_data = next((p for p in price_history if p["time"] == bar), None)
    
    if price_data is None:
        return None
        
    return price_data["close"]
