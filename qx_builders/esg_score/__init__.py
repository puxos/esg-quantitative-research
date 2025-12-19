"""
ESG Score Builder Package

Provides ESG (Environmental, Social, Governance) score building functionality.

Components:
- schema: Dataset contract definition
- builder: ESGScoreBuilder class

Dependencies:
- Requires GVKEY mapping from gvkey_mapping package
"""

from .builder import ESGScoreBuilder
from .schema import get_contracts

__all__ = [
    "ESGScoreBuilder",
    "get_contracts",
]
