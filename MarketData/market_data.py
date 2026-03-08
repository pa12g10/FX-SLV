# Market Data Module for FX-SLV
import pandas as pd
import QuantLib as ql

def get_eval_date():
    """
    Get the evaluation date for market data
    
    Returns:
        QuantLib.Date: Evaluation date (March 8, 2026)
    """
    return ql.Date(8, 3, 2026)

# ========================
# USD SOFR Curve Instruments
# ========================

def get_sofr_deposit_data():
    """
    Get SOFR Overnight Deposit data

    Returns:
        dict: Deposit instrument data with tenor, rate, and day count
    """
    return {
        'tenor': 'ON',
        'rate': 4.58,  # SOFR overnight rate
        'day_count': 'Actual/360'
    }


def get_sofr_futures_data():
    """
    Get SOFR Futures data (1M - 18M)
    
    Returns:
        pandas.DataFrame: Futures contracts with columns:
            - contract: Contract name
            - maturity: Maturity tenor
            - price: Futures price
            - rate: Implied rate (100 - price)
            - day_count: Day count convention
    """
    data = [
        ["SOFR-1M", "1M", 95.45, 4.55, "Actual/360"],
        ["SOFR-2M", "2M", 95.40, 4.60, "Actual/360"],
        ["SOFR-3M", "3M", 95.32, 4.68, "Actual/360"],
        ["SOFR-6M", "6M", 95.18, 4.82, "Actual/360"],
        ["SOFR-9M", "9M", 95.05, 4.95, "Actual/360"],
        ["SOFR-1Y", "1Y", 94.88, 5.12, "Actual/360"],
        ["SOFR-18M", "18M", 94.65, 5.35, "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['contract', 'maturity', 'price', 'rate', 'day_count'])


def get_sofr_swaps_data():
    """
    Get SOFR OIS Swaps data (2Y - 30Y)
    
    Returns:
        pandas.DataFrame: OIS swaps with columns:
            - tenor: Swap tenor
            - rate: Fixed rate (%)
            - fixed_freq: Fixed leg frequency
            - float_freq: Float leg frequency
            - day_count: Day count convention
    """
    data = [
        ["2Y", 5.52, "Annual", "Annual", "Actual/360"],
        ["3Y", 5.68, "Annual", "Annual", "Actual/360"],
        ["4Y", 5.78, "Annual", "Annual", "Actual/360"],
        ["5Y", 5.85, "Annual", "Annual", "Actual/360"],
        ["6Y", 5.90, "Annual", "Annual", "Actual/360"],
        ["7Y", 5.94, "Annual", "Annual", "Actual/360"],
        ["8Y", 5.97, "Annual", "Annual", "Actual/360"],
        ["9Y", 5.99, "Annual", "Annual", "Actual/360"],
        ["10Y", 6.00, "Annual", "Annual", "Actual/360"],
        ["12Y", 6.02, "Annual", "Annual", "Actual/360"],
        ["15Y", 6.03, "Annual", "Annual", "Actual/360"],
        ["20Y", 6.02, "Annual", "Annual", "Actual/360"],
        ["25Y", 5.98, "Annual", "Annual", "Actual/360"],
        ["30Y", 5.92, "Annual", "Annual", "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'rate', 'fixed_freq', 'float_freq', 'day_count'])


# ========================
# EUR ESTR Curve Instruments
# ========================

def get_estr_deposit_data():
    """
    Get ESTR Overnight Deposit data

    Returns:
        dict: Deposit instrument data with tenor, rate, and day count
    """
    return {
        'tenor': 'ON',
        'rate': 3.15,  # ESTR overnight rate
        'day_count': 'Actual/360'
    }


def get_estr_futures_data():
    """
    Get ESTR Futures data (1M - 18M)
    
    Returns:
        pandas.DataFrame: Futures contracts with columns:
            - contract: Contract name
            - maturity: Maturity tenor
            - price: Futures price
            - rate: Implied rate (100 - price)
            - day_count: Day count convention
    """
    data = [
        ["ESTR-1M", "1M", 96.88, 3.12, "Actual/360"],
        ["ESTR-2M", "2M", 96.82, 3.18, "Actual/360"],
        ["ESTR-3M", "3M", 96.75, 3.25, "Actual/360"],
        ["ESTR-6M", "6M", 96.58, 3.42, "Actual/360"],
        ["ESTR-9M", "9M", 96.42, 3.58, "Actual/360"],
        ["ESTR-1Y", "1Y", 96.22, 3.78, "Actual/360"],
        ["ESTR-18M", "18M", 95.95, 4.05, "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['contract', 'maturity', 'price', 'rate', 'day_count'])


def get_estr_swaps_data():
    """
    Get ESTR OIS Swaps data (2Y - 30Y)
    
    Returns:
        pandas.DataFrame: OIS swaps with columns:
            - tenor: Swap tenor
            - rate: Fixed rate (%)
            - fixed_freq: Fixed leg frequency
            - float_freq: Float leg frequency
            - day_count: Day count convention
    """
    data = [
        ["2Y", 4.22, "Annual", "Annual", "Actual/360"],
        ["3Y", 4.35, "Annual", "Annual", "Actual/360"],
        ["4Y", 4.42, "Annual", "Annual", "Actual/360"],
        ["5Y", 4.48, "Annual", "Annual", "Actual/360"],
        ["6Y", 4.52, "Annual", "Annual", "Actual/360"],
        ["7Y", 4.55, "Annual", "Annual", "Actual/360"],
        ["8Y", 4.57, "Annual", "Annual", "Actual/360"],
        ["9Y", 4.58, "Annual", "Annual", "Actual/360"],
        ["10Y", 4.59, "Annual", "Annual", "Actual/360"],
        ["12Y", 4.60, "Annual", "Annual", "Actual/360"],
        ["15Y", 4.59, "Annual", "Annual", "Actual/360"],
        ["20Y", 4.56, "Annual", "Annual", "Actual/360"],
        ["25Y", 4.51, "Annual", "Annual", "Actual/360"],
        ["30Y", 4.45, "Annual", "Annual", "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'rate', 'fixed_freq', 'float_freq', 'day_count'])


# ========================
# FX Market Data
# ========================

def get_fx_spot():
    """
    Get FX spot rate (EUR/USD)
    
    Returns:
        dict: FX spot with pair and rate
    """
    return {
        'pair': 'EUR/USD',
        'rate': 1.0850  # EUR/USD spot
    }


def get_fx_forwards_data():
    """
    Get FX Forward points (EUR/USD)
    
    Returns:
        pandas.DataFrame: FX forwards with columns:
            - tenor: Forward tenor
            - points: Forward points (pips)
            - outright: Outright forward rate
            - day_count: Day count convention
    """
    spot = get_fx_spot()['rate']
    data = [
        ["1M", -12.5, spot - 0.000125, "Actual/360"],
        ["2M", -24.8, spot - 0.000248, "Actual/360"],
        ["3M", -36.5, spot - 0.000365, "Actual/360"],
        ["6M", -71.2, spot - 0.000712, "Actual/360"],
        ["9M", -104.5, spot - 0.001045, "Actual/360"],
        ["1Y", -136.8, spot - 0.001368, "Actual/360"],
        ["18M", -198.5, spot - 0.001985, "Actual/360"],
        ["2Y", -258.2, spot - 0.002582, "Actual/360"],
        ["3Y", -372.5, spot - 0.003725, "Actual/360"],
        ["5Y", -585.0, spot - 0.005850, "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'points', 'outright', 'day_count'])


def get_ccy_swaps_data():
    """
    Get Cross-Currency Basis Swaps (EUR/USD)
    EUR 3M EURIBOR vs USD 3M SOFR + basis
    
    Returns:
        pandas.DataFrame: CCY basis swaps with columns:
            - tenor: Swap tenor
            - basis: Cross-currency basis (bps)
            - eur_leg_freq: EUR leg frequency
            - usd_leg_freq: USD leg frequency
            - day_count: Day count convention
    """
    data = [
        ["1Y", -8.5, "Quarterly", "Quarterly", "Actual/360"],
        ["2Y", -10.2, "Quarterly", "Quarterly", "Actual/360"],
        ["3Y", -12.5, "Quarterly", "Quarterly", "Actual/360"],
        ["4Y", -14.0, "Quarterly", "Quarterly", "Actual/360"],
        ["5Y", -15.2, "Quarterly", "Quarterly", "Actual/360"],
        ["7Y", -16.8, "Quarterly", "Quarterly", "Actual/360"],
        ["10Y", -18.5, "Quarterly", "Quarterly", "Actual/360"],
        ["15Y", -19.2, "Quarterly", "Quarterly", "Actual/360"],
        ["20Y", -19.5, "Quarterly", "Quarterly", "Actual/360"],
        ["30Y", -19.0, "Quarterly", "Quarterly", "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'basis', 'eur_leg_freq', 'usd_leg_freq', 'day_count'])


# ========================
# Volatility Surface Data
# ========================

def get_fx_vol_surface_data():
    """
    Get FX Volatility Surface (EUR/USD)
    
    Returns:
        pandas.DataFrame: Vol surface with columns:
            - strike: Strike as % of spot (ATM = 100)
            - expiry: Expiry in years
            - volatility: Implied volatility (decimal)
    """
    # Generate vol surface grid
    strikes = [90, 95, 100, 105, 110]  # % of spot
    expiries = [0.25, 0.5, 1.0, 2.0, 5.0]  # years
    
    data = []
    for expiry in expiries:
        for strike_pct in strikes:
            # ATM vol with smile
            atm_vol = 0.08 + 0.01 * expiry  # 8-13% ATM vol
            smile_adjustment = 0.002 * abs(strike_pct - 100)  # Vol smile
            vol = atm_vol + smile_adjustment
            
            spot = get_fx_spot()['rate']
            strike_level = spot * strike_pct / 100
            
            data.append([strike_level, expiry, vol])
    
    return pd.DataFrame(data, columns=['strike', 'expiry', 'volatility'])
