"""
Two-Factor Regression Model

Performs OLS regression to estimate factor exposures (market beta and ESG beta):
    R_i,t - RF_t = α_i + β_market * (R_market,t - RF_t) + β_ESG * ESG_factor_t + ε_i,t
"""

from .model import MarketESGBetaModel
from .schema import get_contracts

__all__ = ["MarketESGBetaModel", "get_contracts"]
