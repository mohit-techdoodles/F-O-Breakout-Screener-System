"""
List of NSE F&O Permitted Stocks and Universe Manager
"""

import re
from typing import List

# Curated, verified active NSE F&O symbols (180+ active liquid tickers)
VERIFIED_FO_STOCKS = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT", 
    "ADANIPORTS", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", 
    "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", 
    "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", 
    "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", 
    "BOSCHLTD", "BPCL", "BRITANNIA", "BSOFT", "CANBK", "CANFINHOME", "CHAMBLFERT", 
    "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", 
    "CROMPTON", "CUB", "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR", "DELHIVERY", 
    "DIVISLAB", "DIXON", "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", 
    "FEDERALBNK", "GAIL", "GLENMARK", "GMRAIRPORT", "GNFC", "GODREJCP", "GODREJPROP", 
    "GRANULES", "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", 
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDCOPPER", "HINDPETRO", 
    "HINDUNILVR", "ICICIBANK", "ICICIGI", "IDEA", "IDFCFIRSTB", "IEX", 
    "IGL", "INDHOTEL", "INDIACEM", "INDIAMART", "INDIGO", "INDUSINDBK", "INDUSTOWER", 
    "INFY", "IOC", "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", "JKCEMENT", "JSWSTEEL", 
    "JUBLFOOD", "KALYANKJIL", "KOTAKBANK", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", 
    "LICI", "LT", "LTF", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO", 
    "MARUTI", "MAXHEALTH", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS", 
    "MRF", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NESTLEIND", "NMDC", 
    "NTPC", "OBEROIRLTY", "OFSS", "ONGC", "PAGEIND", "PERSISTENT", "PETRONET", 
    "PFC", "PIDILITIND", "PIIND", "PNB", "POLYCAB", "POWERGRID", "PVRINOX", "RAMCOCEM", 
    "RBLBANK", "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", 
    "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "SRF", "SUNPHARMA", "SUNTV", "SYNGENE", 
    "TATACHEM", "TATACOMM", "TATACONSUM", "TATAPOWER", "TATASTEEL", 
    "TCS", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", 
    "UNITDSPR", "UPL", "VBL", "VEDL", "VOLTAS", "WIPRO", "ZYDUSLIFE"
]

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", 
    "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA", 
    "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", 
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", 
    "INDUSINDBK", "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", 
    "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", 
    "SBIN", "SUNPHARMA", "TATACONSUM", "TATASTEEL", "TCS", 
    "TECHM", "TITAN", "ULTRACEMCO", "WIPRO"
]

BANK_NIFTY = [
    "AUBANK", "AXISBANK", "BANDHANBNK", "BANKBARODA", "FEDERALBNK", "HDFCBANK", 
    "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK", "PNB", "SBIN"
]

TOP_30_LIQUID = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", 
    "ITC", "LT", "BAJFINANCE", "KOTAKBANK", "AXISBANK", "MARUTI", "SUNPHARMA", 
    "TITAN", "ULTRACEMCO", "NTPC", "POWERGRID", "M&M", "TATASTEEL", "JSWSTEEL", 
    "ADANIENT", "ADANIPORTS", "COALINDIA", "BAJAJFINSV", "HINDALCO", "EICHERMOT", 
    "DIVISLAB", "CIPLA", "TRENT"
]


def sanitize_symbol(symbol: str) -> bool:
    """Validates that a symbol contains only valid alphanumeric/ticker characters."""
    if not symbol or not isinstance(symbol, str):
        return False
    return bool(re.match(r"^[A-Z0-9&-]{2,15}$", symbol.strip().upper()))


def get_fo_universe() -> List[str]:
    """Returns the validated list of active F&O symbols."""
    return [s for s in sorted(list(set(VERIFIED_FO_STOCKS))) if sanitize_symbol(s)]


def get_yf_symbol(symbol: str) -> str:
    """Formats an NSE symbol for Yahoo Finance (e.g., RELIANCE -> RELIANCE.NS)"""
    clean_sym = symbol.strip().upper()
    return f"{clean_sym}.NS"
