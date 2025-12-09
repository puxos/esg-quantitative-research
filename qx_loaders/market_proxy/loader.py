"""
Market Proxy Loader Implementation
===================================

Loads market proxy OHLCV price data (e.g., SPY) for benchmarking and factor models.

This loader:
1. Loads OHLCV data for the specified market proxy symbol
2. Filters to the requested date range
3. Returns a DataFrame with OHLCV columns (date, open, high, low, close, volume, adj_close)

Usage in DAG:
-------------
```python
Task(
    id="LoadMarketProxy",
    run=run_loader(
        "qx_loaders/market_proxy",
        overrides={
            "start_date": "2014-01-01",
            "end_date": "2024-12-31",
            "proxy_symbol": "SPY"
        }
    ),
    deps=["BuildOHLCV"]
)
```
"""

import numpy as np
import pandas as pd

from qx.common.types import AssetClass, DatasetType, Domain, Frequency
from qx.foundation.base_loader import BaseLoader


class MarketProxyLoader(BaseLoader):
    """
    Load market proxy OHLCV price data for benchmarking and factor models.

    This loader provides a clean interface to fetch market OHLCV price data,
    commonly used in CAPM, Sharpe ratio calculations, beta estimation, and performance
    attribution.

    Parameters
    ----------
    start_date : str
        Start date for market proxy data (YYYY-MM-DD)
    end_date : str
        End date for market proxy data (YYYY-MM-DD)
    proxy_symbol : str, optional
        Market proxy ticker symbol (default: "SPY")
    frequency : str, optional
        Data frequency - "D", "W", or "M" (default: "D")
    exchange : str, optional
        Exchange for market proxy (default: "US")

    Returns
    -------
    pd.DataFrame
        Market proxy OHLCV price data with columns: date, symbol, open, high, low, close, volume, adj_close

    Examples
    --------
    >>> # Basic usage
    >>> loader = MarketProxyLoader(
    ...     curated_loader=curated_loader,
    ...     params={
    ...         "start_date": "2014-01-01",
    ...         "end_date": "2024-12-31",
    ...         "proxy_symbol": "SPY"
    ...     }
    ... )
    >>> market_returns = loader.load()
    """

    def load_impl(self) -> pd.DataFrame:
        """
        Load market proxy OHLCV price data.

        Returns
        -------
        pd.DataFrame
            Market proxy OHLCV price data with columns: date, symbol, open, high, low, close, volume, adj_close

        Raises
        ------
        ValueError
            If no data found for the specified market proxy symbol
        """
        # Get parameters
        start_date = pd.Timestamp(self.params["start_date"])
        end_date = pd.Timestamp(self.params["end_date"])
        proxy_symbol = self.params.get("proxy_symbol", "SPY")
        frequency = self.params.get("frequency", "D")
        exchange = self.params.get("exchange", "US")

        # Map frequency to enum
        freq_map = {"D": Frequency.DAILY, "W": Frequency.WEEKLY, "M": Frequency.MONTHLY}
        freq_enum = freq_map.get(frequency, Frequency.DAILY)

        # Load OHLCV data for market proxy
        ohlcv_type = DatasetType(
            domain=Domain.MARKET_DATA,
            asset_class=AssetClass.EQUITY,
            subdomain="ohlcv",
            region=None,
            frequency=freq_enum,
        )

        # Load data from curated
        # OHLCV data is partitioned by exchange, frequency, symbol, year
        # Build path manually and glob for all years
        from pathlib import Path

        base_path = Path("data/curated/market-data/ohlcv/schema_v1")
        symbol_path = (
            base_path
            / f"exchange={exchange}"
            / f"frequency={frequency}"
            / f"symbol={proxy_symbol}"
        )

        if not symbol_path.exists():
            raise ValueError(
                f"No data found for market proxy '{proxy_symbol}'. "
                f"Make sure it was included when building OHLCV data. "
                f"Expected path: {symbol_path}"
            )

        # Load all year partitions
        year_dirs = list(symbol_path.glob("year=*"))
        if not year_dirs:
            raise ValueError(
                f"No year partitions found for {proxy_symbol} at {symbol_path}"
            )

        dfs = []
        for year_dir in sorted(year_dirs):
            parquet_files = list(year_dir.glob("*.parquet"))
            for pq_file in parquet_files:
                df_year = pd.read_parquet(pq_file)
                dfs.append(df_year)

        if not dfs:
            raise ValueError(f"No parquet files found for {proxy_symbol}")

        all_data = pd.concat(dfs, ignore_index=True)

        # Ensure date column is datetime
        all_data["date"] = pd.to_datetime(all_data["date"])

        # Filter by date range
        proxy_df = all_data[
            (all_data["date"] >= pd.Timestamp(start_date))
            & (all_data["date"] <= pd.Timestamp(end_date))
        ]

        if proxy_df.empty:
            raise ValueError(
                f"No data found for market proxy '{proxy_symbol}' "
                f"in date range {start_date.date()} to {end_date.date()}"
            )

        # Sort by date
        proxy_df = proxy_df.sort_values("date").reset_index(drop=True)

        # Select relevant columns for OHLCV output
        ohlcv_columns = [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
        ]

        # Ensure all columns exist (use available columns)
        available_columns = [col for col in ohlcv_columns if col in proxy_df.columns]
        result_df = proxy_df[available_columns].copy()

        # Validate we have data
        if result_df.empty:
            raise ValueError(
                f"No price data available for market proxy '{proxy_symbol}'. "
                f"Check that OHLCV data exists for the requested period."
            )

        return result_df


# Example usage
if __name__ == "__main__":
    """
    Example standalone usage (requires curated OHLCV data).
    """
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.foundation.typed_loader import TypedCuratedLoader
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.pathing import PathResolver

    # Initialize storage components
    registry = DatasetRegistry()
    seed_registry(registry)
    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()

    # Create curated loader
    curated_loader = TypedCuratedLoader(
        backend=backend, resolver=resolver, registry=registry
    )

    # Load market proxy returns
    loader = MarketProxyLoader(
        curated_loader=curated_loader,
        params={
            "start_date": "2014-01-01",
            "end_date": "2024-12-31",
            "proxy_symbol": "SPY",
            "frequency": "D",
            "return_type": "simple",
        },
    )

    print("Loading market proxy returns...")
    market_returns = loader.load()

    print(f"\nMarket Proxy Returns Summary:")
    print(f"  Symbol: SPY")
    print(f"  Period: {market_returns.index[0]} to {market_returns.index[-1]}")
    print(f"  Observations: {len(market_returns)}")
    print(f"  Mean return: {market_returns.mean():.4%}")
    print(f"  Std dev: {market_returns.std():.4%}")
    print(
        f"  Sharpe (annualized): {(market_returns.mean() / market_returns.std()) * np.sqrt(252):.2f}"
    )
    print(f"\nFirst 5 returns:")
    print(market_returns.head())
