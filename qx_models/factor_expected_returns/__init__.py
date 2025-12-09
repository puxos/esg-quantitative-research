"""
Factor Expected Returns Model

Calculates expected returns using factor model framework:
    E[R_i,t] = RF_t + Σ(β_i,k × λ_k)

where:
    - RF_t: Risk-free rate at time t
    - β_i,k: Stock i's exposure to factor k (from two_factor_regression)
    - λ_k: Factor k's risk premium (HAC-robust mean)

This model implements the Extended CAPM with ESG factor from:
    Pastor, Stambaugh & Taylor (2021): "Sustainable investing in equilibrium"
"""

from .model import FactorExpectedReturnsModel
from .schema import get_factor_expected_returns_contract

__all__ = ["FactorExpectedReturnsModel", "get_factor_expected_returns_contract"]
