# Changelog

## [2026-03-08] - Hotfixes

### Fixed
- **Curve Extrapolation**: Enabled extrapolation in FX yield curves to handle times beyond maximum curve time (fixes RuntimeError: "time is past max curve time")
- **Streamlit Deprecation**: Replaced deprecated `use_container_width=True` with `width='stretch'` in all GUI section files to comply with Streamlit 2026 API changes

### Changed
- Updated `Models/fx_curves.py` to call `enableExtrapolation()` on both domestic and foreign yield curves
- Updated all GUI section files (`fx_curves_section.py`, `fx_slv_section.py`, `single_barrier_section.py`, `double_barrier_section.py`) to use new Streamlit width parameter

## [2026-03-08] - Initial Release

### Added
- Complete FX Stochastic Local Volatility (FX-SLV) model implementation
- FX yield curve bootstrapping for dual currencies
- Heston stochastic volatility model calibration
- Single barrier FX options pricing (Up-Out, Down-Out, Up-In, Down-In)
- Double barrier FX options pricing (Knock-Out, Knock-In)
- Interactive Streamlit GUI with 4 workflow sections
- Comprehensive documentation and README
- Requirements file with all dependencies
