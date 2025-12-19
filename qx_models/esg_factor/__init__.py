"""
ESG Factor Model

Constructs long-short factor portfolios based on ESG scores and momentum signals.
"""

from .model import ESGFactorModel
from .schema import get_contracts

__all__ = ["ESGFactorModel", "get_contracts"]
