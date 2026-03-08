# FX-SLV

FX Stochastic Local Volatility model implementation with GUI for calibration, simulation, and pricing of FX barrier options (single and double).

## Structure

- **Models/**: FX-SLV model implementation using QuantLib
- **Pricing/**: Pricing engines for FX barrier options (single and double barriers)
- **GUI/**: Streamlit-based user interface
- **MarketData/**: Market data inputs
- **Results/**: Output results and analysis
- **Unittest/**: Unit tests

## Features

- FX Stochastic Local Volatility model calibration to market instruments
- Monte Carlo simulation for model validation
- Pricing of single barrier FX options (Up-and-Out, Down-and-Out, Up-and-In, Down-and-In)
- Pricing of double barrier FX options (Double-Knock-Out, Double-Knock-In)
- Interactive GUI for calibration, simulation, and pricing

## Usage

```bash
python run_gui.py
```
