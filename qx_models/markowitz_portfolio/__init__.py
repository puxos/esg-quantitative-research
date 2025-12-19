"""
Markowitz Portfolio Optimization Model

Mean-variance portfolio optimization with ESG controls:
    Maximize: μ'w - 0.5*γ*w'Σw

    Subject to:
        - Budget constraint: Σw = 1
        - Long-only: w ≥ 0
        - Position limits: w ≤ w_max
        - ESG exposure: L_ESG ≤ β_ESG'w ≤ U_ESG
        - Sector concentration: Σw[sector] ≤ cap

Academic foundation:
    - Markowitz (1952): Portfolio Selection
    - Ledoit & Wolf (2003): Covariance shrinkage estimation
    - Pastor, Stambaugh & Taylor (2021): ESG integration
"""

from .model import MarkowitzPortfolioModel
from .schema import get_contracts

__all__ = ["MarkowitzPortfolioModel", "get_contracts"]
