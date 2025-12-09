"""
ESG Factor Model

Constructs long-short factor portfolios based on ESG scores and momentum signals.
"""

from .model import ESGFactorModel
from .schema import get_esg_factors_contract

__all__ = ["ESGFactorModel", "get_esg_factors_contract"]
