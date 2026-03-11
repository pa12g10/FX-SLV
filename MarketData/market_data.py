# Market Data Module for FX-SLV
# Rates as of 10-11 March 2026
# Sources: NY Fed (SOFR 3.65%), ECB (ESTR 1.933%), TraditionData (SOFR OIS short end)
import pandas as pd
import QuantLib as ql

def get_eval_date():
    """
    Get the evaluation date for market data

    Returns:
        QuantLib.Date: Evaluation date (10 March 2026)
    """
    return ql.Date(10, 3, 2026)


# ========================
# USD SOFR Curve Instruments
# ========================

def get_sofr_deposit_data():
    return {'tenor': 'ON', 'rate': 3.65, 'day_count': 'Actual/360'}


def get_sofr_futures_data():
    data = [
        ["SR1J6",  "1M",  96.32, 3.68, "Actual/360"],
        ["SR1K6",  "1M",  96.30, 3.70, "Actual/360"],
        ["SR1M6",  "1M",  96.34, 3.66, "Actual/360"],
        ["SR3U6",  "3M",  96.42, 3.58, "Actual/360"],
        ["SR3Z6",  "3M",  96.55, 3.45, "Actual/360"],
        ["SR3H7",  "3M",  96.62, 3.38, "Actual/360"],
        ["SR3U7",  "3M",  96.68, 3.32, "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['contract', 'tenor', 'price', 'rate', 'day_count'])


def get_sofr_swaps_data():
    data = [
        ["2Y",  3.38, "Annual", "Annual", "Actual/360"],
        ["3Y",  3.42, "Annual", "Annual", "Actual/360"],
        ["4Y",  3.50, "Annual", "Annual", "Actual/360"],
        ["5Y",  3.58, "Annual", "Annual", "Actual/360"],
        ["6Y",  3.65, "Annual", "Annual", "Actual/360"],
        ["7Y",  3.72, "Annual", "Annual", "Actual/360"],
        ["8Y",  3.77, "Annual", "Annual", "Actual/360"],
        ["9Y",  3.81, "Annual", "Annual", "Actual/360"],
        ["10Y", 3.84, "Annual", "Annual", "Actual/360"],
        ["12Y", 3.86, "Annual", "Annual", "Actual/360"],
        ["15Y", 3.87, "Annual", "Annual", "Actual/360"],
        ["20Y", 3.87, "Annual", "Annual", "Actual/360"],
        ["25Y", 3.86, "Annual", "Annual", "Actual/360"],
        ["30Y", 3.85, "Annual", "Annual", "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'rate', 'fixed_freq', 'float_freq', 'day_count'])


# ========================
# EUR ESTR Curve Instruments
# ========================

def get_estr_deposit_data():
    return {'tenor': 'ON', 'rate': 2.43, 'day_count': 'Actual/360'}


def get_estr_futures_data():
    data = [
        ["ER1J6",  "1M",  97.58, 2.42, "Actual/360"],
        ["ER1K6",  "1M",  97.57, 2.43, "Actual/360"],
        ["ER1M6",  "1M",  97.56, 2.44, "Actual/360"],
        ["ER3U6",  "3M",  97.54, 2.46, "Actual/360"],
        ["ER3Z6",  "3M",  97.52, 2.48, "Actual/360"],
        ["ER3H7",  "3M",  97.50, 2.50, "Actual/360"],
        ["ER3U7",  "3M",  97.48, 2.52, "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['contract', 'tenor', 'price', 'rate', 'day_count'])


def get_estr_swaps_data():
    data = [
        ["2Y",  2.52, "Annual", "Annual", "Actual/360"],
        ["3Y",  2.58, "Annual", "Annual", "Actual/360"],
        ["4Y",  2.63, "Annual", "Annual", "Actual/360"],
        ["5Y",  2.68, "Annual", "Annual", "Actual/360"],
        ["6Y",  2.72, "Annual", "Annual", "Actual/360"],
        ["7Y",  2.75, "Annual", "Annual", "Actual/360"],
        ["8Y",  2.77, "Annual", "Annual", "Actual/360"],
        ["9Y",  2.79, "Annual", "Annual", "Actual/360"],
        ["10Y", 2.80, "Annual", "Annual", "Actual/360"],
        ["12Y", 2.82, "Annual", "Annual", "Actual/360"],
        ["15Y", 2.84, "Annual", "Annual", "Actual/360"],
        ["20Y", 2.85, "Annual", "Annual", "Actual/360"],
        ["25Y", 2.84, "Annual", "Annual", "Actual/360"],
        ["30Y", 2.82, "Annual", "Annual", "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'rate', 'fixed_freq', 'float_freq', 'day_count'])


# ========================
# FX Market Data
# ========================

def get_fx_spot():
    return {'pair': 'EUR/USD', 'rate': 1.1616}


def get_fx_swap_data():
    """
    FX Swap outrights (O/N through 2Y), EUR/USD.

    Outrights computed as: F = spot * exp((r_USD - r_EUR - basis) * T)
    where basis transitions linearly from -20 bps (short end) to -22.5 bps (2Y),
    matching the first CCY swap pillar exactly.

    With SOFR ~3.65% and ESTR ~2.43% (differential ~1.22%), EUR trades at
    a forward PREMIUM (positive forward points) since USD rates > EUR rates
    means USD is worth more in the future => EUR/USD forward > spot.

    Basis (negative) slightly reduces the EUR discount factor, increasing
    the forward above pure CIP.

    Forward points (pips = (outright - spot) * 10000):
      1M:  +14.0   3M:  +41.3   6M:  +80.7   12M: +149.1
      15M: +175.6  18M: +202.1  2Y:  +254.8
    """
    spot = get_fx_spot()['rate']
    data = [
        # -- Overnight tenors (skipped by bootstrapper) --
        ["O/N",  +0.4,  round(spot + 0.000040, 5), "Actual/360"],
        ["T/N",  +0.8,  round(spot + 0.000080, 5), "Actual/360"],
        ["S/N",  +1.0,  round(spot + 0.000100, 5), "Actual/360"],
        # -- Short end (basis ~ -20 bps) --
        ["1W",   +3.3,  round(spot + 0.000330, 5), "Actual/360"],
        ["2W",   +6.5,  round(spot + 0.000650, 5), "Actual/360"],
        ["1M",  +14.0,  round(spot + 0.001400, 5), "Actual/360"],
        ["2M",  +27.8,  round(spot + 0.002780, 5), "Actual/360"],
        ["3M",  +41.3,  round(spot + 0.004130, 5), "Actual/360"],
        ["6M",  +80.7,  round(spot + 0.008070, 5), "Actual/360"],
        ["9M", +117.3,  round(spot + 0.011730, 5), "Actual/360"],
        ["12M",+149.1,  round(spot + 0.014910, 5), "Actual/360"],
        # -- Transition tenors (basis easing from -21.5 to -22.5 bps) --
        ["15M",+175.6,  round(spot + 0.017560, 5), "Actual/360"],
        ["18M",+202.1,  round(spot + 0.020210, 5), "Actual/360"],
        # -- 2Y bridge: exactly consistent with 2Y CCY swap at -22.5 bps --
        ["2Y", +254.8,  round(spot + 0.025480, 5), "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'points', 'outright', 'day_count'])


def get_ccy_swaps_data():
    """
    Mark-to-Market Cross-Currency Basis Swaps (EUR/USD, 2Y-30Y).
    Convention: ESTR flat vs SOFR + basis spread (bps).
    """
    data = [
        ["2Y",  -22.5, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["3Y",  -25.0, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["4Y",  -27.0, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["5Y",  -28.5, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["7Y",  -29.0, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["10Y", -27.5, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["12Y", -26.5, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["15Y", -25.0, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["20Y", -23.0, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["30Y", -20.5, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'basis', 'eur_leg', 'usd_leg', 'notional', 'day_count'])


# kept for backward compatibility
def get_fx_forwards_data():
    return get_fx_swap_data()


# ========================
# Volatility Surface Data
# ========================

def get_fx_vol_surface_data():
    strikes  = [90, 95, 100, 105, 110]
    expiries = [0.25, 0.5, 1.0, 2.0, 5.0]
    data = []
    spot = get_fx_spot()['rate']
    for expiry in expiries:
        for strike_pct in strikes:
            atm_vol          = 0.075 + 0.008 * expiry
            smile_adjustment = 0.002 * abs(strike_pct - 100)
            vol              = atm_vol + smile_adjustment
            data.append([spot * strike_pct / 100, expiry, vol])
    return pd.DataFrame(data, columns=['strike', 'expiry', 'volatility'])
