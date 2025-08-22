import json
from typing import Dict, List, Any
from trend_filters.kalman import kalman_filter_backquant
from trend_filters.trend_manager import is_equity_curve_trending
import math

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


        is_trending, latest_r2  = is_equity_curve_trending(config, equity_curve_non_filtered)

        # we open a new trade based on the trend filter (or none) results on i+1 (for the filtered curve)
        if i+1 < len(identifiers):   # this means we are still processing days which we have data for and can get i+1
            open_new_trade(equity_curve_filtered, tournament_data, identifiers[i+1], identifiers[i], config, is_trending)
        else: # this means we have reached the end of the identifiers array and i+1 is out of range, we will replace it with "Current Signal"
            # print("DEBUG - NON _ FILTERED DATA!!!!!!!!!!!!")
            # print([entry["capital"] for entry in equity_curve_non_filtered])
            # print("DEBUG - FILTERED DATA!!!!!!!!!!!!")
            # print([entry["capital"] for entry in equity_curve_filtered])
            print("capital to check non filtered... ", equity_curve_filtered[-1]["capital"])    #the print language is wrong but we are doing this now to test the filtered
            # print("capital to check filtered... ", equity_curve_filtered[-1]["capital"])
            
            open_new_trade(equity_curve_filtered, tournament_data, "Current Signal", identifiers[i], config, is_trending)


    
    # #summarize the equity curve
    # equity_curve_non_filtered_summarized = summarize_equity_curve(equity_curve_non_filtered)
    # equity_curve_filtered_summarized = summarize_equity_curve(equity_curve_filtered)

    # #save the equity curve
    # save_equity_curve(equity_curve_non_filtered_summarized, config["equity_curve_file"])
    # save_equity_curve(equity_curve_filtered_summarized, config["equity_curve_file_filtered"])


from typing import List, Dict
import math

def summarize_equity_curve(equity_curve: List[Dict]) -> Dict:
    capitals = [entry["capital"] for entry in equity_curve]
    returns = [
        (capitals[i] - capitals[i - 1]) / capitals[i - 1]
        for i in range(1, len(capitals))
    ]
    
    if not returns:
        return {
            "curve": equity_curve,
            "summary": {
                "total_return": 0,
                "annualized_return": 0,
                "annualized_volatility": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "omega_ratio": 0,
                "max_drawdown": 0,
                "final_capital": capitals[-1] if capitals else 0,
                "number_of_trades": 0
            }
        }

    rf_rate = 0  # risk-free rate
    threshold = 0  # minimum acceptable return for Sortino & Omega

    # Basic stats
    avg_return = sum(returns) / len(returns)
    std_return = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))

    # Sortino (using deviation from threshold)
    downside_diff_sq = [(min(0, r - threshold))**2 for r in returns]
    downside_std = math.sqrt(sum(downside_diff_sq) / len(returns)) if any(downside_diff_sq) else 1e-6

    # Omega (sum of gains / abs(sum of losses))
    gains = sum(r for r in returns if r > threshold)
    losses = abs(sum(r for r in returns if r < threshold))
    omega = gains / losses if losses != 0 else float('inf')

    # Max drawdown
    peak = capitals[0]
    max_drawdown = 0
    for capital in capitals:
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak
        max_drawdown = max(max_drawdown, drawdown)

    summary = {
        "total_return": ((capitals[-1] - capitals[0]) / capitals[0]) * 100,
        "annualized_return": (avg_return * 252) * 100,
        "annualized_volatility": (std_return * math.sqrt(252)) * 100,
        "sharpe_ratio": ((avg_return - rf_rate) / std_return) * math.sqrt(252) if std_return != 0 else 0,
        "sortino_ratio": ((avg_return - threshold) / downside_std) * math.sqrt(252),
        "omega_ratio": omega,
        "max_drawdown": max_drawdown * 100,
        "final_capital": capitals[-1],
        "number_of_trades": sum(1 for entry in equity_curve if entry.get("pnl", 0) != 0)
    }

    return {"curve": equity_curve, "summary": summary}

def save_equity_curve(equity_curve: List[Dict], path: str):
    with open(path, 'w') as f:
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
    if winner_asset is not None:
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
