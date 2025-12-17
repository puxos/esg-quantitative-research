"""
Tiingo Price Data Builder (YAML-Only Version)

Fetches OHLCV price data from Tiingo API and builds curated datasets.

This is a simplified version that ONLY supports YAML-based initialization.
Legacy support has been removed for cleaner code.
"""

import os
from typing import Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from tiingo import TiingoClient

from qx.common.env_loader import get_env_var
from qx.common.ticker_mapper import TickerMapper
from qx.foundation.base_builder import DataBuilderBase
from qx.storage.curated_writer import CuratedWriter
from qx.utils.date_utils import (
    adjust_fetch_dates,
    check_date_coverage,
    get_tolerance_for_frequency,
)


class TiingoOHLCVBuilder(DataBuilderBase):
    """
    SOURCE BUILDER: Equity price data from Tiingo API.

    External Source: Tiingo REST API
    Authentication: Requires TIINGO_API_KEY environment variable or tiingo_api_key parameter

    Fetches OHLCV data including adjusted prices, dividends, and splits.
    Transforms to curated format with proper partitioning.

    YAML-only initialization - reads all config from builder.yaml.
    Use with run_builder() factory in DAG orchestration.
    """

    def __init__(
        self,
        package_dir: str,
        writer: CuratedWriter,
        overrides: Optional[dict] = None,
    ):
        """
        Initialize Tiingo price builder from YAML configuration.

        Args:
            package_dir: Path to builder package (e.g., "qx_builders/tiingo_ohlcv")
            writer: High-level curated data writer
            overrides: Parameter overrides (e.g., {"symbols": ["AAPL"], "start_date": "2020-01-01"})
        """
        # Call parent to load builder.yaml and initialize parameters
        # (.env file is auto-loaded by base class)
        super().__init__(
            package_dir=package_dir,
            writer=writer,
            overrides=overrides,
        )

        # Get API key (checks param, then .env, then raises error)
        api_key = get_env_var(
            key="TIINGO_API_KEY",
            param_name="tiingo_api_key",
            param_value=self.params.get("tiingo_api_key"),
            required=True,
        )

        # Initialize Tiingo client
        self.tiingo = TiingoClient({"api_key": api_key})

        # Initialize attributes from params
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

        Reads parameters from kwargs (populated by base class from builder.yaml).

        Returns:
            Combined DataFrame with data for all symbols
        """
        # Extract parameters
        symbols = kwargs.get("symbols", [])
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        frequency = kwargs.get("frequency", "daily")
        fail_on_error = kwargs.get("fail_on_error", False)
        apply_date_correction = kwargs.get("apply_date_correction", True)

        # Convert tolerance_days to int if provided
        tolerance_days_param = kwargs.get("tolerance_days")
        tolerance_days = None
        if tolerance_days_param is not None and tolerance_days_param != "null":
            tolerance_days = (
                int(tolerance_days_param)
                if isinstance(tolerance_days_param, str)
                else tolerance_days_param
            )

        if not symbols:
            raise ValueError(
                "No symbols provided. Set 'symbols' parameter with list of tickers"
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

                # Report completeness (summary only, not per-symbol details)
                if not coverage["is_complete"]:
                    gap_info = f"missing: {coverage['start_gap_days']}d start, {coverage['end_gap_days']}d end"
                    # Could log this for monitoring, but don't print for every symbol

            return df

        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return pd.DataFrame()
            raise RuntimeError(f"Error fetching {symbol} from Tiingo: {e}")

    def transform_to_curated(self, raw_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform raw Tiingo data to curated OHLCV format.

        Args:
            raw_df: Raw DataFrame from Tiingo API
            **kwargs: May contain frequency, partitions

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

        # Convert date to date type (not datetime)
        raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.date

        # Create canonical DataFrame with all required columns
        curated = pd.DataFrame(
            {
                "date": raw_df["date"],
                "exchange": self.exchange,
                "symbol": raw_df["symbol"],  # Always present in YAML mode
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
