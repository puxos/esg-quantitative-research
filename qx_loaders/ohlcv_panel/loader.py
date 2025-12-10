"""
OHLCV Panel Loader

Load OHLCV price data for a list of symbols with date range filtering.

This loader bridges universe selection (historic_members, continuous_universe)
with price-based research by:
1. Loading OHLCV data for specified symbols and date range
2. Filtering by date range
3. Optionally filtering to tradeable data (non-zero volume)

Use Cases:
    - Returns calculation: Load prices for factor models
    - Beta estimation: Get price data for regression
    - Portfolio backtesting: Historical prices for simulation
    - Momentum strategies: Price data for signal calculation

Example usage:
    # Load daily prices for universe members
    Task(
        id="LoadPrices",
        run=run_loader(
            package_path="qx_loaders/ohlcv_panel",
            registry=registry,
            backend=backend,
            resolver=resolver,
            overrides={
                "start_date": "2014-01-01",
                "end_date": "2024-12-31",
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "frequency": "daily",
                "exchange": "US",
                "require_volume": False
            }
        ),
        deps=[]
    )

    # Load prices from upstream loader (dynamic symbols)
    Task(
        id="LoadPrices",
        run=lambda ctx: run_loader(
            package_path="qx_loaders/ohlcv_panel",
            overrides={
                "symbols": ctx["GetHistoricMembers"]["output"],
                "start_date": "2014-01-01",
                "end_date": "2024-12-31"
            }
        )(),
        deps=["GetHistoricMembers"]
    )
"""

from typing import List

import pandas as pd

from qx.common.types import AssetClass, DatasetType, Domain, Frequency, Subdomain
from qx.foundation.base_loader import BaseLoader


class OHLCVPanelLoader(BaseLoader):
    """
    Load OHLCV panel data for specified symbols with date range filtering.

    Reads curated OHLCV price data and returns a DataFrame with price data
    for the specified symbols and date range.

    Parameters (from loader.yaml):
        start_date: Period start (YYYY-MM-DD)
        end_date: Period end (YYYY-MM-DD)
        symbols: List of ticker symbols (can be from upstream loader)
        frequency: Data frequency - "daily", "weekly", "monthly" (default: "daily")
        exchange: Exchange filter (default: "US")
        require_volume: Filter out zero/null volume rows (default: False)

    Returns:
        pd.DataFrame: OHLCV data with columns:
            - date, symbol, open, high, low, close, volume, adj_close
    """

    def load_impl(self) -> pd.DataFrame:
        """
        Load OHLCV panel data with date range filtering.

        Returns:
            DataFrame with OHLCV data for specified symbols
        """
        # Get parameters
        start_date = pd.Timestamp(self.params["start_date"])
        end_date = pd.Timestamp(self.params["end_date"])
        symbols = self.params["symbols"]
        frequency = self.params.get("frequency", "daily")
        exchange = self.params.get("exchange", "US")
        require_volume = self.params.get("require_volume", False)

        print(f"📊 Loading OHLCV panel data")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print(f"   Symbols: {len(symbols)} tickers")
        print(f"   Frequency: {frequency}")
        print(f"   Exchange: {exchange}")
        print(f"   Require volume: {require_volume}")

        # Handle empty symbol list
        if not symbols:
            print("⚠️  Empty symbol list, returning empty DataFrame")
            return pd.DataFrame()

        # Convert frequency string to enum
        # Frequency enum values ("daily", "weekly", "monthly") match storage paths
        freq_enum = Frequency(frequency)

        # Load OHLCV data from curated data
        ohlcv_type = DatasetType(
            domain=Domain.MARKET_DATA,
            asset_class=AssetClass.EQUITY,
            subdomain=Subdomain.OHLCV,
            region=None,
            frequency=freq_enum,
        )

        # Load data directly from file system (bypass typed_loader due to multi-level partitioning)
        # OHLCV data is partitioned by exchange, frequency, symbol, and year
        # We need to scan the file system to find all year directories for each symbol
        if not symbols:
            print(
                "   ⚠️  Cannot load OHLCV data without symbol list (symbol is partition key)"
            )
            return pd.DataFrame()

        print(f"   Loading {len(symbols)} symbols...")

        from pathlib import Path

        # Build base path: data/curated/market-data/ohlcv/schema_v1/exchange={exchange}/frequency={frequency}
        contract = self.registry.find(ohlcv_type)
        base_dir = (
            Path("data/curated/market-data/ohlcv")
            / contract.schema_version
            / f"exchange={exchange}"
            / f"frequency={frequency}"
        )

        if not base_dir.exists():
            print(f"   ⚠️  OHLCV data directory not found: {base_dir}")
            return pd.DataFrame()

        dfs = []
        errors = 0
        symbols_found = 0

        for symbol in symbols:
            symbol_dir = base_dir / f"symbol={symbol}"
            if not symbol_dir.exists():
                errors += 1
                continue

            # Find all year directories for this symbol
            year_dirs = list(symbol_dir.glob("year=*"))
            if not year_dirs:
                errors += 1
                continue

            symbols_found += 1

            # Load all parquet files from all years
            for year_dir in year_dirs:
                parquet_files = list(year_dir.glob("*.parquet"))
                for pq_file in parquet_files:
                    try:
                        df_year = pd.read_parquet(pq_file)
                        dfs.append(df_year)
                    except Exception as e:
                        if errors <= 5:
                            print(f"   ⚠️  Error loading {pq_file}: {e}")
                        errors += 1

        if not dfs:
            print(
                f"   ⚠️  No OHLCV data found (found {symbols_found}/{len(symbols)} symbols, {errors} errors)"
            )
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Deduplicate in case multiple builder runs created duplicate files
        if not df.empty:
            if "ingest_ts" in df.columns:
                df = df.sort_values("ingest_ts", ascending=False)
            df = df.drop_duplicates(
                subset=["date", "symbol", "exchange", "frequency"], keep="first"
            )

        if errors > 0 and errors <= 5:
            print(f"   ⚠️  {errors} errors during loading")

        print(f"✅ Loaded {len(df):,} raw OHLCV records")
        print(f"   Found {symbols_found}/{len(symbols)} symbols with data")

        # Ensure date column is datetime
        if "date" not in df.columns:
            print("   ⚠️  No 'date' column found in OHLCV data")
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])

        # Filter by date range
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

        print(f"✅ After date filter: {len(df):,} records")
        if df.empty:
            print(
                f"   ⚠️  No data in date range {start_date.date()} to {end_date.date()}"
            )
            return df

        # Filter by volume if required
        if require_volume and "volume" in df.columns:
            before_count = len(df)
            df = df[(df["volume"].notna()) & (df["volume"] > 0)].copy()
            after_count = len(df)
            filtered_count = before_count - after_count
            if filtered_count > 0:
                print(f"✅ Filtered {filtered_count:,} zero/null volume records")

        # Sort by date and symbol for consistency
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Report final statistics
        unique_symbols = df["symbol"].nunique()
        date_range = df["date"].agg(["min", "max"])

        print(f"✅ Final dataset: {len(df):,} records")
        print(f"   Unique symbols: {unique_symbols}")
        print(
            f"   Date range: {date_range['min'].date()} to {date_range['max'].date()}"
        )

        # Summary by symbol (show first 10)
        symbol_counts = df.groupby("symbol").size().sort_index()
        print(f"\n   Sample symbol coverage (first 10):")
        for symbol, count in symbol_counts.head(10).items():
            print(f"      {symbol}: {count} records")
        if len(symbol_counts) > 10:
            print(f"      ... and {len(symbol_counts) - 10} more symbols")

        return df
