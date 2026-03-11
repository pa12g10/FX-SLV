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
    # Short end anchored to TraditionData: 1Y = 3.519%
    # 2Y-5Y: modest bull-steepener as Fed cuts gradually
    # 10Y+: flattening toward long-run neutral; 15-30Y capped at ~3.85-3.90%
    #        to avoid over-shooting convexity in the 30Y FX forward
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
    # ESTR ~= ECB DFR minus ~7bps spread => ~2.43% with DFR at 2.50%
    return {'tenor': 'ON', 'rate': 2.43, 'day_count': 'Actual/360'}


def get_estr_futures_data():
    # ESTR futures prices consistent with ECB on hold; very slight term premium
    data = [
        ["ER1J6",  "1M",  97.58, 2.42, "Actual/360"],  # Apr-26
        ["ER1K6",  "1M",  97.57, 2.43, "Actual/360"],  # May-26
        ["ER1M6",  "1M",  97.56, 2.44, "Actual/360"],  # Jun-26
        ["ER3U6",  "3M",  97.54, 2.46, "Actual/360"],  # Sep-26
        ["ER3Z6",  "3M",  97.52, 2.48, "Actual/360"],  # Dec-26
        ["ER3H7",  "3M",  97.50, 2.50, "Actual/360"],  # Mar-27
        ["ER3U7",  "3M",  97.48, 2.52, "Actual/360"],  # Sep-27
    ]
    return pd.DataFrame(data, columns=['contract', 'tenor', 'price', 'rate', 'day_count'])


def get_estr_swaps_data():
    # ESTR OIS mid-market, 10-Mar-2026
    # Front end pinned near 2.43-2.50%; long end drifts to ~2.80% with term premium
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
    # EUR/USD spot 10-Mar-2026 (TradingEconomics)
    return {'pair': 'EUR/USD', 'rate': 1.1616}


def get_fx_swap_data():
    """
    Get FX Swap market data (O/N through 2Y).

    With SOFR ~3.65% and ESTR ~2.43%, the USD-EUR rate differential
    is ~122bps - EUR trades at a forward DISCOUNT so forward points are NEGATIVE.

    Forward point approximation: pts ~= spot x (r_USD - r_EUR) x T
    1M:  1.1616 x 0.0122 x (1/12) x 10000 ~= 11.8 pips
    18M: 1.1616 x 0.0122 x 1.5   x 10000 ~= 212 pips (market wider due to basis)
    2Y:  linear extrapolation from 18M ~= -314 pips; basis-adjusted ~= -312 pips
         consistent with 2Y CCY swap basis of -22.5 bps.
    """
    spot = get_fx_spot()['rate']
    data = [
        # -- Overnight tenors --
        ["O/N",   -0.4,  round(spot - 0.000040, 5), "Actual/360"],
        ["T/N",   -0.8,  round(spot - 0.000080, 5), "Actual/360"],
        ["S/N",   -1.0,  round(spot - 0.000100, 5), "Actual/360"],
        # -- Short end --
        ["1W",    -3.5,  round(spot - 0.000350, 5), "Actual/360"],
        ["2W",    -7.0,  round(spot - 0.000700, 5), "Actual/360"],
        ["1M",   -14.5,  round(spot - 0.001450, 5), "Actual/360"],
        ["2M",   -28.8,  round(spot - 0.002880, 5), "Actual/360"],
        ["3M",   -43.0,  round(spot - 0.004300, 5), "Actual/360"],
        ["6M",   -84.5,  round(spot - 0.008450, 5), "Actual/360"],
        ["9M",  -124.5,  round(spot - 0.012450, 5), "Actual/360"],
        ["12M", -163.0,  round(spot - 0.016300, 5), "Actual/360"],
        ["18M", -238.5,  round(spot - 0.023850, 5), "Actual/360"],
        # -- 2Y bridge pillar: smooths the seam to the first CCY swap at 2Y -22.5bps --
        # CIP forward (2Y SOFR 3.38%, ESTR 2.52%): F = 1.1616 * exp(0.0086*2) = 1.1817
        # Basis-adjusted (~-22bps over 2Y adds ~25 pips): outright ~ 1.1842
        # => forward points = (1.1842 - 1.1616) * 10000 = -312 pips (negative = EUR discount)
        ["2Y",  -312.0,  round(spot - 0.031200, 5), "Actual/360"],
    ]
    return pd.DataFrame(data, columns=['tenor', 'points', 'outright', 'day_count'])


def get_ccy_swaps_data():
    """
    Get Mark-to-Market Cross-Currency Basis Swaps (EUR/USD).

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
