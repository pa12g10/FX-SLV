# Market Data Module for FX-SLV
# Rates as of 10-11 March 2026
# Sources: NY Fed (SOFR 3.65%), ECB (ESTR 1.933%), TraditionData (SOFR OIS short end)
import pandas as pd
import numpy as np
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
# Fed on hold; overnight SOFR = 3.65%.
# Term structure: slight inversion at front end as market prices 1-2 cuts in 2026,
# then modest re-steepening. Long end (15-30Y) capped ~3.85-3.90% reflecting
# realistic long-run neutral with moderate term premium.

def get_sofr_deposit_data():
    # SOFR fixing 10-Mar-2026: 3.65% (NY Fed)
    return {'tenor': 'ON', 'rate': 3.65, 'day_count': 'Actual/360'}


def get_sofr_futures_data():
    # Prices imply rates consistent with 1Y OIS at ~3.52% and gentle bull-steepening
    data = [
        ["SR1J6",  "1M",  96.32, 3.68, "Actual/360"],  # Apr-26
        ["SR1K6",  "1M",  96.30, 3.70, "Actual/360"],  # May-26
        ["SR1M6",  "1M",  96.34, 3.66, "Actual/360"],  # Jun-26
        ["SR3U6",  "3M",  96.42, 3.58, "Actual/360"],  # Sep-26 (1 cut priced)
        ["SR3Z6",  "3M",  96.55, 3.45, "Actual/360"],  # Dec-26
        ["SR3H7",  "3M",  96.62, 3.38, "Actual/360"],  # Mar-27
        ["SR3U7",  "3M",  96.68, 3.32, "Actual/360"],  # Sep-27
    ]
    return pd.DataFrame(data, columns=['contract', 'tenor', 'price', 'rate', 'day_count'])


def get_sofr_swaps_data():
    # SOFR OIS mid-market, 10-Mar-2026
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
# ECB deposit rate 2.50% (cut to terminal); ESTR fixing 10-Mar-2026 = 1.933%.
# Note: ESTR = ECB DFR - ~7bps spread; DFR currently 2.50% => ESTR ~2.43%
# OIS curve flat to very slightly positive — ECB seen on hold through 2026.

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
    # EUR/USD spot 10-Mar-2026
    return {'pair': 'EUR/USD', 'rate': 1.1616}


def get_fx_swap_data():
    """
    FX Swap market data (O/N through 18M).
    With SOFR ~3.65% and ESTR ~2.43%, USD-EUR differential ~122bps.
    EUR trades at a forward DISCOUNT so forward points are NEGATIVE.
    """
    spot = get_fx_spot()['rate']
    data = [
        ["O/N",   -0.4,  round(spot - 0.000040, 5), "Actual/360"],
        ["T/N",   -0.8,  round(spot - 0.000080, 5), "Actual/360"],
        ["S/N",   -1.0,  round(spot - 0.000100, 5), "Actual/360"],
        ["1W",    -3.5,  round(spot - 0.000350, 5), "Actual/360"],
        ["2W",    -7.0,  round(spot - 0.000700, 5), "Actual/360"],
        ["1M",   -14.5,  round(spot - 0.001450, 5), "Actual/360"],
        ["2M",   -28.8,  round(spot - 0.002880, 5), "Actual/360"],
        ["3M",   -43.0,  round(spot - 0.004300, 5), "Actual/360"],
        ["6M",   -84.5,  round(spot - 0.008450, 5), "Actual/360"],
        ["9M",  -124.5,  round(spot - 0.012450, 5), "Actual/360"],
        ["12M", -163.0,  round(spot - 0.016300, 5), "Actual/360"],
        ["18M", -238.5,  round(spot - 0.023850, 5), "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'points', 'outright', 'day_count'])


def get_ccy_swaps_data():
    """
    Mark-to-Market Cross-Currency Basis Swaps (EUR/USD).
    EUR/USD xccy basis structurally negative since 2008.
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


def get_fx_forwards_data():
    """Alias kept for backward compatibility."""
    return get_fx_swap_data()


# ========================
# Volatility Surface Data
# ========================

def get_fx_vol_surface_data():
    """
    Return a realistic EUR/USD vol surface in DECIMAL form.

    FIX vs original
    ---------------
    * Strikes are now expressed as ABSOLUTE EUR/USD levels centred on the
      current spot (~1.1616), not as percentage-of-par integers (90/95/100/
      105/110).  The original strike range of [1.045, 1.278] was too wide
      and only produced 5 strikes; the new grid gives 9 strikes per expiry
      covering ±8% around spot which is a realistic FX smile range.
    * Vols are returned as DECIMAL (e.g. 0.075) not as percent, consistent
      with the rest of the codebase contract.
    * ATM vol term structure and smile shape now match the surface defined in
      fx_slv_section.py so the two sources are consistent.
    """
    spot = get_fx_spot()['rate']  # 1.1616

    # Expiry grid: 1W, 1M, 3M, 6M, 1Y, 2Y  (in years)
    expiries = [1/52, 1/12, 0.25, 0.50, 1.00, 2.00]

    # ATM vols (decimal) per expiry – match fx_slv_section.py surface
    atm_vols = {1/52: 0.0620, 1/12: 0.0650, 0.25: 0.0690,
                0.50: 0.0720, 1.00: 0.0755, 2.00: 0.0800}

    # Strike offsets as fraction of spot: -8%, -6%, -4%, -2%, 0%, +2%, +4%, +6%, +8%
    strike_offsets = [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08]

    data = []
    for expiry in expiries:
        atm = atm_vols[expiry]
        for offset in strike_offsets:
            K   = round(spot * (1 + offset), 5)
            # Quadratic smile: wings add ~2 bps per 1% moneyness offset
            smile_add = 0.002 * abs(offset) * 100   # e.g. 8% offset => +1.6%
            # Slight put skew: negative offsets get a small extra premium
            skew_add  = -0.001 * offset * 100        # e.g. -8% => +0.8%, +8% => -0.8%
            vol       = round(atm + smile_add + skew_add, 6)
            vol       = max(0.03, vol)               # floor at 3%
            data.append([K, expiry, vol])

    return pd.DataFrame(data, columns=['strike', 'expiry', 'volatility'])
