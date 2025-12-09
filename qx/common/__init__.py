"""
Common utilities and shared components for Qx architecture

Modules:
- config_utils: Configuration loading helpers
- contracts: Dataset contract definitions and registry
- predefined: Pre-seeded dataset contracts
- schema_loader: YAML schema parsing
- ticker_mapper: Ticker symbol resolution for corporate actions
- types: Core type definitions
"""

from .ticker_mapper import TickerMapper, resolve_ticker

__all__ = [
    "TickerMapper",
    "resolve_ticker",
]
