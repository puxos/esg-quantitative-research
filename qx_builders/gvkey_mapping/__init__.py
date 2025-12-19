"""
GVKEY Mapping Builder Package

Provides GVKEY-to-ticker mapping functionality for ESG and other data sources.

Components:
- schema: Dataset contract definition
- builder: GVKEYMappingBuilder class
- utils: Helper functions (clean_ticker_symbol, load_gvkey_mapping_from_curated)
"""

from .builder import GVKEYMappingBuilder
from .schema import get_contracts
from .utils import clean_ticker_symbol, load_gvkey_mapping_from_curated

__all__ = [
    "GVKEYMappingBuilder",
    "get_contracts",
    "clean_ticker_symbol",
    "load_gvkey_mapping_from_curated",
]
