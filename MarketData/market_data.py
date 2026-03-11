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
    return {'tenor': 'ON', 'rate': 4.58, 'day_count': 'Actual/360'}


def get_sofr_futures_data():
    data = [
        ["SR1J6",  "1M",  95.45, 4.55, "Actual/360"],
        ["SR1K6",  "1M",  95.40, 4.60, "Actual/360"],
        ["SR1M6",  "1M",  95.32, 4.68, "Actual/360"],
        ["SR3U6",  "3M",  95.18, 4.82, "Actual/360"],
        ["SR3Z6",  "3M",  95.05, 4.95, "Actual/360"],
        ["SR3H7",  "3M",  94.88, 5.12, "Actual/360"],
        ["SR3U7",  "3M",  94.65, 5.35, "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['contract', 'tenor', 'price', 'rate', 'day_count'])


def get_sofr_swaps_data():
    data = [
        ["2Y",  5.52, "Annual", "Annual", "Actual/360"],
        ["3Y",  5.68, "Annual", "Annual", "Actual/360"],
        ["4Y",  5.78, "Annual", "Annual", "Actual/360"],
        ["5Y",  5.85, "Annual", "Annual", "Actual/360"],
        ["6Y",  5.90, "Annual", "Annual", "Actual/360"],
        ["7Y",  5.94, "Annual", "Annual", "Actual/360"],
        ["8Y",  5.97, "Annual", "Annual", "Actual/360"],
        ["9Y",  5.99, "Annual", "Annual", "Actual/360"],
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
    return {'tenor': 'ON', 'rate': 3.15, 'day_count': 'Actual/360'}


def get_estr_futures_data():
    data = [
        ["ER1J6",  "1M",  96.88, 3.12, "Actual/360"],
        ["ER1K6",  "1M",  96.82, 3.18, "Actual/360"],
        ["ER1M6",  "1M",  96.75, 3.25, "Actual/360"],
        ["ER3U6",  "3M",  96.58, 3.42, "Actual/360"],
        ["ER3Z6",  "3M",  96.42, 3.58, "Actual/360"],
        ["ER3H7",  "3M",  96.22, 3.78, "Actual/360"],
        ["ER3U7",  "3M",  95.95, 4.05, "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['contract', 'tenor', 'price', 'rate', 'day_count'])


def get_estr_swaps_data():
    data = [
        ["2Y",  4.22, "Annual", "Annual", "Actual/360"],
        ["3Y",  4.35, "Annual", "Annual", "Actual/360"],
        ["4Y",  4.42, "Annual", "Annual", "Actual/360"],
        ["5Y",  4.48, "Annual", "Annual", "Actual/360"],
        ["6Y",  4.52, "Annual", "Annual", "Actual/360"],
        ["7Y",  4.55, "Annual", "Annual", "Actual/360"],
        ["8Y",  4.57, "Annual", "Annual", "Actual/360"],
        ["9Y",  4.58, "Annual", "Annual", "Actual/360"],
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
    return {'pair': 'EUR/USD', 'rate': 1.0850}


def get_fx_swap_data():
    """
    Get FX Swap market data (O/N through 18M).

    FX swaps are the primary short-end instrument for CCY basis curve
    construction.  Each row represents a single swap tenor.

    Columns
    -------
    tenor        : Tenor label
    points       : Forward points (pips, 4th decimal place)
    outright     : Outright forward rate (spot + points/10000)
    day_count    : Day count convention
    """
    spot = get_fx_spot()['rate']
    data = [
        # ── Overnight tenors ───────────────────────────────────────────
        ["O/N",  -0.2,  spot - 0.000002, "Actual/360"],
        ["T/N",  -0.4,  spot - 0.000004, "Actual/360"],
        ["S/N",  -0.5,  spot - 0.000005, "Actual/360"],
        # ── Short end ─────────────────────────────────────────────────
        ["1W",   -2.8,  spot - 0.000028, "Actual/360"],
        ["2W",   -5.5,  spot - 0.000055, "Actual/360"],
        ["1M",   -12.5, spot - 0.000125, "Actual/360"],
        ["2M",   -24.8, spot - 0.000248, "Actual/360"],
        ["3M",   -36.5, spot - 0.000365, "Actual/360"],
        ["6M",   -71.2, spot - 0.000712, "Actual/360"],
        ["9M",  -104.5, spot - 0.001045, "Actual/360"],
        ["12M", -136.8, spot - 0.001368, "Actual/360"],
        # ── Last FX swap pillar before xccy swaps ───────────────────
        ["18M", -198.5, spot - 0.001985, "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'points', 'outright', 'day_count'])


def get_ccy_swaps_data():
    """
    Get Mark-to-Market Cross-Currency Basis Swaps (EUR/USD).

    Standard G3 instrument: ESTR flat vs SOFR + basis spread (bps).
    Notional resets on the EUR leg each coupon date at prevailing spot.
    Quoted as the spread added to the non-USD (EUR) leg.

    Tenors pick up from where FX swaps leave off (2Y+).

    Columns
    -------
    tenor        : Swap tenor
    basis        : EUR/USD xccy basis spread (bps, negative = pay less on EUR)
    eur_leg      : EUR floating index (ESTR)
    usd_leg      : USD floating index (SOFR)
    notional     : Notional reset convention
    day_count    : Day count convention
    """
    data = [
        ["2Y",  -10.2, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["3Y",  -12.5, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["4Y",  -14.0, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["5Y",  -15.2, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["7Y",  -16.8, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["10Y", -18.5, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["12Y", -18.8, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["15Y", -19.2, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["20Y", -19.5, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
        ["30Y", -19.0, "ESTR", "SOFR", "MtM Reset", "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'basis', 'eur_leg', 'usd_leg', 'notional', 'day_count'])


# kept for backward compatibility - now superseded by get_fx_swap_data() + get_ccy_swaps_data()
def get_fx_forwards_data():
    """
    Legacy helper — returns the same rows as get_fx_swap_data() for
    any code that still calls get_fx_forwards_data().
    """
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
            atm_vol          = 0.08 + 0.01 * expiry
            smile_adjustment = 0.002 * abs(strike_pct - 100)
            vol              = atm_vol + smile_adjustment
            data.append([spot * strike_pct / 100, expiry, vol])
    return pd.DataFrame(data, columns=['strike', 'expiry', 'volatility'])
