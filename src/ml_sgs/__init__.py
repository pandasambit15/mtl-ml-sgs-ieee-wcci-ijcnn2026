"""
ml_sgs – Physics-Guided Multi-Task Learning for SGS Turbulence Parameterisation
================================================================================

A research framework comparing physics integration strategies for predicting
subgrid-scale (SGS) viscosity and diffusivity coefficients in atmospheric LES
(MONC model), as described in:

    Panda et al. (2026). Physics-Guided Multi-Task Learning for Subgrid Scale
    Turbulence Parameterization: A Comparative Study of Physics Integration
    Strategies. IEEE WCCI IJCNN 2026, Maastricht, Netherlands.

Sub-packages
------------
models        Neural network architectures (baseline, Ri-conditioned)
data          MONC NetCDF data processing and feature engineering
training      Training loops for all model variants
inference     Unified and constrained inference engines
evaluation    Metrics, statistical tests, plotting, and HDF5 prediction storage
"""

__version__ = "1.0.0"
__author__ = "Sambit Kumar Panda, Todd Jones, Muhammad Shahzad, Bryan Lawrence, Anna-Louise Ellis"
__license__ = "MIT"
