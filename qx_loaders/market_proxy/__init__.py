"""
Market Proxy Loader
===================

Load market proxy returns (SPY by default) for benchmarking and factor models.

This loader provides a clean interface to fetch market returns as a time series,
commonly used in CAPM, Sharpe ratio calculations, and beta estimation.
"""

from qx_loaders.market_proxy.loader import MarketProxyLoader

__all__ = ["MarketProxyLoader"]
