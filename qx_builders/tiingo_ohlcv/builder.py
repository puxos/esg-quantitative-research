"""
Tiingo Price Data Builder

Fetches OHLCV price data from Tiingo API and builds curated datasets.
Migrated from src/market/price_manager.py to Qx architecture.

Supports both legacy and YAML-based initialization for DAG orchestration.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from tiingo import TiingoClient

from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.ticker_mapper import TickerMapper
from qx.foundation.base_builder import DataBuilderBase
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter
from qx.utils.date_utils import (
    adjust_fetch_dates,
    check_date_coverage,
    get_tolerance_for_frequency,
)


class TiingoOHLCVBuilder(DataBuilderBase):
    """
    Builder for equity price data from Tiingo API.

    Fetches OHLCV data including adjusted prices, dividends, and splits.
    Transforms to curated format with proper partitioning.

    Supports both legacy and YAML-based initialization:
    - Legacy: contract, adapter, resolver, tiingo_client, exchange, currency
    - YAML: package_dir, registry, adapter, resolver, overrides
    """

    def __init__(
        self,
        contract: Optional[DatasetContract] = None,
        adapter: Optional[TableFormatAdapter] = None,
        resolver: Optional[PathResolver] = None,
        tiingo_client: Optional[TiingoClient] = None,
        exchange: Optional[str] = None,
        currency: Optional[str] = None,
        ticker_mapper: Optional[TickerMapper] = None,
        package_dir: Optional[str] = None,
        registry: Optional[DatasetRegistry] = None,
        overrides: Optional[dict] = None,
    ):
        """
        Initialize Tiingo price builder.

        Supports both legacy and YAML-based modes.
        """
        # Call parent __init__ to handle YAML loading if package_dir provided
        super().__init__(
            contract=contract,
            adapter=adapter,
            resolver=resolver,
            package_dir=package_dir,
            registry=registry,
            overrides=overrides,
        )

        # Legacy mode: use provided parameters
        if package_dir is None:
            self.tiingo = tiingo_client
            self.exchange = exchange or "US"
            self.currency = currency or "USD"
            self.ticker_mapper = ticker_mapper or TickerMapper()
        # YAML mode: initialize from parameters
        else:
            # Get API key from params or environment
            api_key = self.params.get("tiingo_api_key") or os.environ.get(
                "TIINGO_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "Tiingo API key required. Set 'tiingo_api_key' parameter or TIINGO_API_KEY env var"
                )

            # Initialize Tiingo client
            self.tiingo = TiingoClient({"api_key": api_key})

            # Initialize other attributes from params
            self.exchange = self.params.get("exchange", "US")
            self.currency = self.params.get("currency", "USD")

            # Ticker mapper (for handling corporate actions like FB→META)
            use_ticker_mapper = self.params.get("use_ticker_mapper", True)
            self.ticker_mapper = TickerMapper() if use_ticker_mapper else None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def fetch_raw(self, **kwargs) -> pd.DataFrame:
        """
        Fetch raw price data from Tiingo API for multiple symbols.

        In YAML mode, reads symbols, start_date, end_date, frequency from kwargs.
        In legacy mode, expects individual symbol fetch.

        Returns:
            Combined DataFrame with data for all symbols
        """
        # Extract parameters
        symbols = kwargs.get("symbols", [])
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        frequency = kwargs.get("frequency", "daily")
        partitions = kwargs.get("partitions", {})
        fail_on_error = kwargs.get("fail_on_error", False)
        apply_date_correction = kwargs.get("apply_date_correction", True)
        tolerance_days_param = kwargs.get(
            "tolerance_days"
        )  # Can be None, string, or int
        # Convert tolerance_days to int if provided
        tolerance_days = None
        if tolerance_days_param is not None and tolerance_days_param != "null":
            tolerance_days = (
                int(tolerance_days_param)
                if isinstance(tolerance_days_param, str)
                else tolerance_days_param
            )

        # Apply date correction to fetch dates if enabled
        original_start = start_date
        original_end = end_date
        if apply_date_correction and start_date and end_date:
            start_date, end_date = adjust_fetch_dates(
                start_date, end_date, frequency, apply_correction=True
            )
            if start_date != original_start or end_date != original_end:
                print(f"\n📅 Date correction applied (±tolerance buffer):")
                print(f"   Requested: {original_start} to {original_end}")
                print(f"   Fetching:  {start_date} to {end_date}")
                tolerance = tolerance_days or get_tolerance_for_frequency(frequency)
                print(f"   Tolerance: ±{tolerance} days ({frequency})")

        # Legacy mode: single symbol fetch
        if "symbol" in kwargs:
            return self._fetch_single_symbol(
                symbol=kwargs["symbol"],
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
            )

        # YAML mode: batch fetch multiple symbols
        if not symbols:
            raise ValueError(
                "No symbols provided. Set 'symbols' parameter with list of tickers"
            )

        # Add market proxy symbols (SPY by default) to the batch
        market_proxy_symbols = kwargs.get("market_proxy_symbols", ["SPY"])
        if market_proxy_symbols:
            # Merge symbols and market proxies, removing duplicates
            all_symbols = list(set(symbols + market_proxy_symbols))
            if len(all_symbols) > len(symbols):
                added = set(all_symbols) - set(symbols)
                print(f"\n📈 Including market proxy symbols: {sorted(added)}")
            symbols = all_symbols

        print(f"\n📊 Fetching price data for {len(symbols)} symbols...")
        print(f"   Date range: {start_date} to {end_date or 'today'}")
        print(f"   Frequency: {frequency}")

        all_data = []
        failed_symbols = []

        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

                # Resolve ticker (handle corporate actions)
                actual_symbol = symbol
                if self.ticker_mapper:
                    actual_symbol = self.ticker_mapper.resolve(symbol)
                    if actual_symbol is None:
                        print(
                            f"⚠️  {symbol}: delisted/acquired with no successor, skipping"
                        )
                        continue
                    if actual_symbol != symbol:
                        print(f"📝 Resolved: {symbol} → {actual_symbol}")

                # Fetch data
                df = self._fetch_single_symbol(
                    symbol=actual_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    original_start=original_start,
                    original_end=original_end,
                    tolerance_days=tolerance_days,
                )

                if not df.empty:
                    # Store original symbol for consistency
                    df["symbol"] = actual_symbol
                    all_data.append(df)
                    print(f"✅ {symbol}: {len(df)} rows")
                else:
                    print(f"⚠️  {symbol}: No data returned")
                    failed_symbols.append(symbol)

            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
                failed_symbols.append(symbol)
                if fail_on_error:
                    raise RuntimeError(f"Failed to fetch {symbol}: {e}")

        if not all_data:
            print("\n⚠️  No data fetched for any symbol")
            return pd.DataFrame()

        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)

        print(f"\n✅ Fetched {len(combined):,} total rows for {len(all_data)} symbols")
        if failed_symbols:
            print(
                f"⚠️  Failed/skipped: {len(failed_symbols)} symbols - {failed_symbols}"
            )

        return combined

    def _fetch_single_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "daily",
        original_start: Optional[str] = None,
        original_end: Optional[str] = None,
        tolerance_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch raw price data for a single symbol from Tiingo API.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "MSFT")
            start_date: Start date in 'YYYY-MM-DD' format (possibly adjusted with buffer)
            end_date: End date in 'YYYY-MM-DD' format (possibly adjusted with buffer)
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            original_start: Original requested start date (before buffer adjustment)
            original_end: Original requested end date (before buffer adjustment)
            tolerance_days: Custom tolerance days (None = auto-calculate)

        Returns:
            Raw DataFrame from Tiingo with columns:
            [date, open, high, low, close, volume, adjOpen, adjHigh, adjLow,
             adjClose, adjVolume, divCash, splitFactor]
        """
        try:
            # Use tiingo-python library
            df = self.tiingo.get_dataframe(
                symbol, startDate=start_date, endDate=end_date, frequency=frequency
            )

            if df.empty:
                return pd.DataFrame()

            # Reset index to make date a column
            df = df.reset_index()

            # Check data completeness if original dates provided
            if original_start and original_end:
                actual_start = pd.to_datetime(df["date"].min()).date()
                actual_end = pd.to_datetime(df["date"].max()).date()
                req_start = pd.to_datetime(original_start).date()
                req_end = pd.to_datetime(original_end).date()

                coverage = check_date_coverage(
                    actual_start=actual_start,
                    actual_end=actual_end,
                    requested_start=req_start,
                    requested_end=req_end,
                    frequency=frequency,
                    apply_tolerance=True,
                )

                # Report completeness
                if coverage["is_complete"]:
                    status = "✅ COMPLETE"
                    if coverage["start_gap_days"] > 0 or coverage["end_gap_days"] > 0:
                        status += f" (±{coverage['tolerance_days']}d)"
                else:
                    status = f"⚠️  PARTIAL (missing: {coverage['start_gap_days']}d start, {coverage['end_gap_days']}d end)"

                # Don't print for every symbol, just summary in fetch_raw

            return df

        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return pd.DataFrame()
            raise RuntimeError(f"Error fetching {symbol} from Tiingo: {e}")

    def transform_to_curated(self, raw_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform raw Tiingo data to curated OHLCV format.

        Handles both single symbol (legacy) and multi-symbol (YAML) modes.

        Args:
            raw_df: Raw DataFrame from Tiingo API
            **kwargs: May contain symbol, frequency, partitions

        Returns:
            Curated DataFrame with canonical schema including:
            - Normalized column names (snake_case)
            - Proper data types
            - Exchange and currency metadata
            - Year partition for efficient storage
            - Source tracking
        """
        if raw_df.empty:
            return raw_df

        # Extract parameters
        frequency = kwargs.get("frequency", "daily")
        partitions = kwargs.get("partitions", {})

        # Map frequency codes (D → daily, W → weekly, M → monthly)
        frequency_map = {"D": "daily", "W": "weekly", "M": "monthly"}
        if frequency in frequency_map:
            frequency = frequency_map[frequency]

        # Convert date to date type (not datetime)
        raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.date

        # Create canonical DataFrame with all required columns
        curated = pd.DataFrame(
            {
                "date": raw_df["date"],
                "exchange": self.exchange,
                "symbol": (
                    raw_df["symbol"]
                    if "symbol" in raw_df.columns
                    else kwargs.get("symbol")
                ),
                "currency": self.currency,
                "frequency": frequency,
                "open": raw_df["open"].astype(float),
                "high": raw_df["high"].astype(float),
                "low": raw_df["low"].astype(float),
                "close": raw_df["close"].astype(float),
                "volume": raw_df["volume"].astype("int64"),
                "adj_open": raw_df["adjOpen"].astype(float),
                "adj_high": raw_df["adjHigh"].astype(float),
                "adj_low": raw_df["adjLow"].astype(float),
                "adj_close": raw_df["adjClose"].astype(float),
                "adj_volume": raw_df["adjVolume"].astype("int64"),
                "div_cash": raw_df["divCash"].astype(float),
                "split_factor": raw_df["splitFactor"].astype(float),
                "source": "tiingo",
            }
        )

        # Add year for partitioning
        curated["year"] = pd.to_datetime(curated["date"]).dt.year

        print(f"\n✅ Curated: {len(curated):,} price records")
        print(f"   Symbols: {curated['symbol'].nunique()} unique")
        print(f"   Date range: {curated['date'].min()} to {curated['date'].max()}")
        print(f"   Years: {sorted(curated['year'].unique())}")

        return curated

    def build_for_symbol(
        self,
        symbol: str,
        exchange: str,
        frequency: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[str]:
        """
        DEPRECATED: Use DAG orchestration with run_builder() instead.

        Legacy method for building price data for a single symbol.

        For new code, use:
            task = Task(
                id="BuildOHLCV",
                run=run_builder(
                    "qx_builders/tiingo_ohlcv",
                    partitions={"exchange": "US", "frequency": "D"},
                    overrides={"symbols": ["AAPL"], "start_date": "2020-01-01"}
                ),
                deps=[]
            )

        Args:
            symbol: Ticker symbol
            exchange: Exchange code (e.g., "US", "NYSE", "NASDAQ")
            frequency: Data frequency ("daily", "weekly", "monthly")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of output paths (one per year partition)
        """
        import warnings

        warnings.warn(
            "build_for_symbol() is deprecated. Use DAG orchestration with run_builder() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Resolve ticker symbol first (handle corporate actions)
        actual_symbol = self.ticker_mapper.resolve(symbol)

        if actual_symbol is None:
            print(f"⚠️  Skipping {symbol}: delisted/acquired with no successor")
            return []

        if actual_symbol != symbol:
            print(f"📝 Resolved ticker: {symbol} → {actual_symbol}")

        # Fetch and transform using resolved symbol
        raw_df = self.fetch_raw(
            symbol=actual_symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )

        if raw_df.empty:
            print(f"⚠️  No data to build for {symbol} (resolved: {actual_symbol})")
            return []

        # Transform using the ORIGINAL symbol to maintain consistency
        # (Tiingo stores historical data under the new ticker)
        curated = self.transform_to_curated(
            raw_df, symbol=actual_symbol, frequency=frequency
        )

        # Add schema metadata
        curated = curated.copy()
        curated["schema_version"] = self.contract.schema_version
        curated["ingest_ts"] = pd.Timestamp.utcnow()

        # Group by year and write separate partitions
        output_paths = []

        for year, year_df in curated.groupby("year"):
            partitions = {
                "exchange": exchange,
                "frequency": frequency,
                "symbol": symbol,
                "year": str(year),
            }

            rel_dir = self.resolver.curated_dir(self.contract, partitions)
            filename = f"part-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"

            output_path = self.adapter.write_batch(year_df, rel_dir, filename)
            output_paths.append(output_path)
            print(f"💾 Saved: {output_path} ({len(year_df)} rows)")

        return output_paths
