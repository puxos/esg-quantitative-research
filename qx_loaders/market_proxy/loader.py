"""
Market Proxy Loader
===================

Load market proxy OHLCV price data (e.g., SPY) for benchmarking and factor models.

This loader is a specialized single-symbol variant of ohlcv_panel, designed for
loading benchmark/market proxy data (like SPY) commonly used in:
- CAPM models (market returns)
- Beta estimation (benchmark returns)
- Performance attribution (benchmark comparison)
- Sharpe ratio calculations

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
            "proxy_symbol": "SPY",
            "frequency": "daily"
        }
    ),
    deps=["BuildOHLCV"]
)
```
"""

from pathlib import Path

import pandas as pd

from qx.foundation.base_loader import BaseLoader


class MarketProxyLoader(BaseLoader):
    """
    Load market proxy OHLCV price data for a single benchmark symbol.

    This is a specialized single-symbol OHLCV loader for benchmarks (SPY, VTI, etc.)
    used in factor models and performance analysis.

    Parameters (from loader.yaml):
        start_date: Period start (YYYY-MM-DD)
        end_date: Period end (YYYY-MM-DD)
        proxy_symbol: Market proxy ticker (default: "SPY")
        frequency: Data frequency - "daily", "weekly", "monthly" (default: "daily")
        exchange: Exchange filter (default: "US")

    Returns:
        pd.DataFrame: OHLCV data with columns:
            - date, symbol, open, high, low, close, volume, adj_close
    """

    def load_impl(self) -> pd.DataFrame:
        """
        Load market proxy OHLCV price data (single symbol).

        Returns:
            DataFrame with OHLCV data for the market proxy symbol

        Raises:
            ValueError: If no data found for the market proxy symbol
        """
        from qx.common.types import (
            AssetClass,
            DatasetType,
            Domain,
            Frequency,
            Subdomain,
        )

        # Get parameters
        start_date = pd.Timestamp(self.params["start_date"])
        end_date = pd.Timestamp(self.params["end_date"])
        proxy_symbol = self.params.get("proxy_symbol", "SPY")
        frequency = self.params.get("frequency", "daily")
        exchange = self.params.get("exchange", "US")

        print(f"📊 Loading market proxy OHLCV data")
        print(f"   Symbol: {proxy_symbol}")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print(f"   Frequency: {frequency}")
        print(f"   Exchange: {exchange}")

        # Map frequency string to Frequency enum
        freq_map = {
            "daily": Frequency.DAILY,
            "weekly": Frequency.WEEKLY,
            "monthly": Frequency.MONTHLY,
        }
        freq_enum = freq_map.get(frequency.lower(), Frequency.DAILY)

        # Define dataset type for OHLCV data
        ohlcv_type = DatasetType(
            domain=Domain.MARKET_DATA,
            asset_class=AssetClass.EQUITY,
            subdomain=Subdomain.BARS,
            region=None,
            frequency=freq_enum,
        )

        # Calculate which years are needed based on date range
        years = list(range(start_date.year, end_date.year + 1))

        # Load each year separately (OHLCV is partitioned by symbol and year)
        dfs = []
        for year in years:
            try:
                df_year = self.loader.load(
                    dataset_type=ohlcv_type,
                    partitions={
                        "exchange": exchange,
                        "frequency": frequency,
                        "symbol": proxy_symbol,
                        "year": str(year),
                    },
                )
                if not df_year.empty:
                    dfs.append(df_year)
            except FileNotFoundError:
                # Year not available, skip
                continue

        if not dfs:
            raise ValueError(
                f"No data found for market proxy '{proxy_symbol}'. "
                f"Make sure it was included when building OHLCV data (use market_proxy_symbols parameter)."
            )

        df = pd.concat(dfs, ignore_index=True)

        print(f"✅ Loaded {len(df):,} raw records for {proxy_symbol}")

        # Ensure date column is datetime
        if "date" not in df.columns:
            raise ValueError("No 'date' column found in OHLCV data")

        df["date"] = pd.to_datetime(df["date"])

        # Filter by date range
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

        print(f"✅ After date filter: {len(df):,} records")

        if df.empty:
            raise ValueError(
                f"No data found for market proxy '{proxy_symbol}' "
                f"in date range {start_date.date()} to {end_date.date()}"
            )

        # Sort by date for consistency
        df = df.sort_values("date").reset_index(drop=True)

        # Report final statistics
        date_range = df["date"].agg(["min", "max"])
        print(f"✅ Final dataset: {len(df):,} records")
        print(
            f"   Date range: {date_range['min'].date()} to {date_range['max'].date()}"
        )

        return df


# Example usage
if __name__ == "__main__":
    """
    Example standalone usage (requires curated OHLCV data).
    """
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.orchestration.factories import run_loader
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.pathing import PathResolver
    from qx.storage.table_format import TableFormatAdapter

    # Initialize storage components
    registry = DatasetRegistry()
    seed_registry(registry)
    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()
    adapter = TableFormatAdapter(backend)

    print("=" * 80)
    print("Market Proxy Loader - Example Usage")
    print("=" * 80)

    # Load market proxy OHLCV data
    task_fn = run_loader(
        package_path="qx_loaders/market_proxy",
        registry=registry,
        backend=backend,
        resolver=resolver,
        overrides={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "proxy_symbol": "SPY",
            "frequency": "daily",
            "exchange": "US",
        },
    )

    print("\nLoading market proxy OHLCV data...")
    result = task_fn()
    market_df = result["output"]

    print(f"\nMarket Proxy OHLCV Summary:")
    print(f"  Symbol: SPY")
    print(f"  Records: {len(market_df):,}")
    print(
        f"  Date range: {market_df['date'].min().date()} to {market_df['date'].max().date()}"
    )
    print(f"  Columns: {list(market_df.columns)}")

    print(f"\nFirst 5 records:")
    print(market_df.head())

    print(f"\nLast 5 records:")
    print(market_df.tail())
