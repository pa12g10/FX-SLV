# FX-SLV

FX Stochastic Local Volatility (FX-SLV) model implementation with interactive GUI for calibration, simulation, and pricing of FX barrier options.

## Overview

This repository implements a complete FX-SLV framework using QuantLib, combining:
- **Stochastic Volatility**: Heston model for stochastic variance dynamics
- **Local Volatility**: Dupire local volatility surface
- **FX Barrier Options**: Single and double barrier option pricing engines

The implementation follows the same structure and patterns as the IR-HW1F repository, providing a professional-grade quantitative finance toolkit with an intuitive Streamlit-based GUI.

## Repository Structure

```
FX-SLV/
├── Models/                    # Model implementations
│   ├── __init__.py
│   ├── fx_curves.py           # FX yield curve bootstrapping
│   └── fx_slv.py              # FX Stochastic Local Volatility model
│
├── Pricing/                  # Pricing engines
│   ├── __init__.py
│   ├── single_barrier.py      # Single barrier FX options
│   └── double_barrier.py      # Double barrier FX options
│
├── GUI/                      # Streamlit GUI
│   ├── __init__.py
│   ├── main_tab.py            # Main GUI controller
│   └── sections/              # GUI sections
│       ├── __init__.py
│       ├── fx_curves_section.py       # FX curves calibration
│       ├── fx_slv_section.py          # FX-SLV model calibration
│       ├── single_barrier_section.py  # Single barrier pricing
│       └── double_barrier_section.py  # Double barrier pricing
│
├── MarketData/               # Market data inputs
├── Results/                  # Output results
├── Unittest/                 # Unit tests
├── run_gui.py                # GUI launcher
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features

### 1. FX Yield Curves
- Dual currency yield curve bootstrapping (domestic and foreign)
- Zero rate and discount factor curves
- Forward FX rate calculation using covered interest rate parity
- Interactive curve visualization

### 2. FX-SLV Model
- **Heston Stochastic Volatility**: Five-parameter model (v0, κ, θ, σ, ρ)
- **Local Volatility Surface**: Dupire formula implementation
- **Calibration**: Calibrate to market FX vanilla option volatilities
- **Simulation**: Monte Carlo path generation for spot FX and volatility
- **Validation**: Model validation against Black-Scholes prices

### 3. Single Barrier FX Options
Pricing support for:
- **Up-and-Out**: Option knocked out if spot rises above barrier
- **Down-and-Out**: Option knocked out if spot falls below barrier
- **Up-and-In**: Option activated if spot rises above barrier
- **Down-and-In**: Option activated if spot falls below barrier

Pricing engines:
- Finite Difference Heston engine (fast, accurate)
- Monte Carlo simulation (flexible, path-dependent)

### 4. Double Barrier FX Options
Pricing support for:
- **Double Knock-Out**: Option dies if spot hits either upper or lower barrier
- **Double Knock-In**: Option activates if spot hits either barrier

Features:
- Corridor analysis and breach probability calculations
- Upper/lower barrier breach statistics
- Payoff diagrams with barrier visualization

### 5. Interactive GUI
- **Streamlit-based**: Modern, responsive web interface
- **Workflow**: Guided workflow from curve calibration → model calibration → pricing
- **Visualization**: Plotly charts for all results
- **Real-time pricing**: Instant feedback on parameter changes

## Installation

### Prerequisites
- Python 3.8 or higher
- QuantLib Python bindings

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pa12g10/FX-SLV.git
cd FX-SLV
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Installing QuantLib

**Linux/Mac**:
```bash
pip install QuantLib
```

**Windows**:
```bash
pip install QuantLib-Python
```

For installation issues, see the [QuantLib documentation](https://www.quantlib.org/install.shtml).

## Usage

### Launch the GUI

```bash
python run_gui.py
```

This will start the Streamlit server and open the GUI in your default web browser (typically at `http://localhost:8501`).

### Workflow

1. **FX Yield Curves & Spot Rate**
   - Enter spot FX rate (e.g., USD/EUR = 1.10)
   - Input domestic and foreign yield curve data
   - Bootstrap curves
   - Visualize zero rates, discount factors, and forward FX rates

2. **FX Stochastic Local Volatility Model**
   - Input FX volatility surface (strikes, expiries, implied vols)
   - Configure Heston model parameters (v0, κ, θ, σ, ρ)
   - Calibrate model to market volatilities
   - View calibration quality and errors
   - Generate simulated FX spot and volatility paths
   - Validate model against Black-Scholes

3. **Single Barrier FX Options**
   - Select barrier type (Up-Out, Down-Out, Up-In, Down-In)
   - Specify strike, barrier, and expiry
   - Price using FD Heston or Monte Carlo
   - View Greeks and breach probabilities
   - Analyze payoff diagrams

4. **Double Barrier FX Options**
   - Select barrier type (Knock-Out, Knock-In)
   - Specify strike, upper/lower barriers, and expiry
   - Price using FD Heston or Monte Carlo
   - View corridor statistics and breach analysis
   - Analyze payoff in barrier corridor

## Model Details

### FX-SLV Model

The FX spot process under the Heston SLV model follows:

**Spot dynamics**:
```
dS(t) = (r_d - r_f) S(t) dt + √v(t) S(t) dW₁(t)
```

**Variance dynamics**:
```
dv(t) = κ(θ - v(t)) dt + σ √v(t) dW₂(t)
```

where:
- `S(t)` = FX spot rate at time t
- `v(t)` = instantaneous variance at time t
- `r_d` = domestic risk-free rate
- `r_f` = foreign risk-free rate
- `κ` = mean reversion speed
- `θ` = long-term variance
- `σ` = volatility of variance (vol-of-vol)
- `ρ = Cor(dW₁, dW₂)` = correlation between spot and variance

### Pricing Engines

**Finite Difference (FD) Heston Engine**:
- Solves the Heston PDE using finite difference methods
- Fast and accurate for European-style barrier options
- Grid-based: (time steps, spot grid, variance grid)

**Monte Carlo Engine**:
- Simulates spot paths using calibrated FX-SLV model
- Path-dependent monitoring for barrier breaches
- Provides breach probabilities and standard errors
- Flexible for complex payoff structures

## Technical Implementation

### Model Calibration

The FX-SLV model is calibrated by:
1. Building Black volatility surface from market data
2. Creating Heston model helpers (vanilla options)
3. Using Levenberg-Marquardt optimization to minimize pricing errors
4. Constructing local volatility surface from calibrated model

### Path Simulation

Monte Carlo paths use:
- Euler discretization for variance process with full truncation
- Log-normal spot process with variance-dependent diffusion
- Antithetic variates for variance reduction (optional)
- Correlated Brownian motions using Cholesky decomposition

## Comparison with IR-HW1F

This FX-SLV repository follows the identical structure of the IR-HW1F repository:

| Feature | IR-HW1F | FX-SLV |
|---------|---------|--------|
| **Model** | Hull-White 1-Factor (interest rates) | Heston SLV (FX) |
| **Calibration** | Swaption volatilities | FX vanilla option vols |
| **Simulation** | Short rate paths | FX spot + vol paths |
| **Derivatives** | Swaptions (European, American, Bermuda) | Barrier options (single, double) |
| **Framework** | QuantLib + Streamlit GUI | QuantLib + Streamlit GUI |
| **Structure** | Models/Pricing/GUI/sections | Models/Pricing/GUI/sections |

## Key Differences from IR-HW1F

1. **Two-factor process**: FX-SLV has both spot and volatility dynamics (vs. single short rate in HW1F)
2. **Stochastic volatility**: Captures volatility smile/skew (HW1F has deterministic vol)
3. **Dual curves**: Requires both domestic and foreign yield curves
4. **Barrier features**: Focus on barrier options vs. interest rate derivatives

## Dependencies

- **QuantLib**: Financial modeling library (C++ with Python bindings)
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **SciPy**: Scientific computing (optimization)

## Contributing

This is a personal research repository. For questions or suggestions, please open an issue.

## License

This project is for educational and research purposes.

## References

- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Dupire, B. (1994). "Pricing with a Smile"
- QuantLib Documentation: https://www.quantlib.org/
- Hull-White Model: IR-HW1F repository (same author)

## Author

pa12g10 - [GitHub Profile](https://github.com/pa12g10)

## Acknowledgments

- QuantLib community for the excellent open-source library
- Streamlit for the intuitive web framework
- IR-HW1F repository as the structural template
