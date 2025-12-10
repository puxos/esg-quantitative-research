"""
Historic Members Loader

Get ALL symbols that were members of a universe during a period, including removed members.

This loader eliminates survivorship bias by including:
- Current members (still in the index)
- Removed members (delisted, acquired, merged)
- Stocks that joined and left during the period

Example: For S&P 500 from 2014-2024, returns ~600+ symbols (not just current 500)
because companies get added/removed over the decade.

Use Cases:
    - Backtesting with proper historical universe (no survivorship bias)
    - ESG research requiring all historical constituents
    - Market beta calculation for all stocks that traded in period
    - Portfolio optimization with realistic historical constraints

Difference from continuous_universe loader:
    - continuous_universe: Returns ONLY symbols that were members for ENTIRE period
    - historic_members: Returns ALL symbols that were members at ANY point in period

Example usage:
    Task(
        id="GetHistoricMembers",
        run=run_loader(
            package_path="qx_loaders/historic_members",
            registry=registry,
            backend=backend,
            resolver=resolver,
            overrides={
                "start_date": "2014-01-01",
                "end_date": "2024-12-31",
                "universe": "sp500"
            }
        )
    )
"""

from typing import List

import pandas as pd

from qx.common.ticker_mapper import TickerMapper
from qx.common.types import DatasetType, Domain, Subdomain
from qx.foundation.base_loader import BaseLoader


class HistoricMembersLoader(BaseLoader):
    """
    Load all symbols that were members of universe during period (including removed).

    Reads membership interval data and returns ALL symbols whose membership
    overlaps with the analysis period, even if they were only members for
    part of the period.

    This eliminates survivorship bias by including stocks that:
    - Were members at start but removed during period
    - Joined during the period
    - Were members for only part of the period

    Algorithm:
        A symbol is included if its membership interval overlaps with [start_date, end_date]:
        (symbol_start <= period_end) AND (symbol_end >= period_start)

    Parameters (from loader.yaml):
        start_date: Period start (YYYY-MM-DD)
        end_date: Period end (YYYY-MM-DD)
        universe: Universe identifier (default: "sp500")
        use_ticker_mapper: Whether to consolidate ticker changes (default: True)

    Returns:
        List[str]: All ticker symbols that were members during any part of the period

    Ticker Mapping:
        When use_ticker_mapper=True (default), old tickers are mapped to current ones.
        Example: FB → META, CBS → PARA
        This ensures unique companies (not unique ticker symbols).
    """

    def load_impl(self) -> List[str]:
        """
        Get all historic universe members (eliminates survivorship bias).

        Returns:
            List of ticker symbols that were members at any point during the period
        """
        # Get parameters
        start_date = pd.Timestamp(self.params["start_date"])
        end_date = pd.Timestamp(self.params["end_date"])
        universe = self.params["universe"]
        use_mapper = self.params.get("use_ticker_mapper", True)

        print(f"📊 Loading historic {universe} members (including removed)")
        print(f"   Period: {start_date.date()} to {end_date.date()}")

        # Initialize ticker mapper if enabled
        ticker_mapper = TickerMapper() if use_mapper else None

        # Load membership intervals from curated data via direct file access
        from pathlib import Path

        membership_type = DatasetType(
            domain=Domain.INSTRUMENT_REFERENCE,
            asset_class=None,
            subdomain=Subdomain.INDEX_CONSTITUENTS,
            region=None,
            frequency=None,
        )

        # Construct partition path: data/curated/instrument-reference/index-constituents/schema_v1/universe={universe}/mode=intervals
        base_path = Path("data/curated/instrument-reference/index-constituents/schema_v1")
        partition_path = base_path / f"universe={universe}" / "mode=intervals"

        if not partition_path.exists():
            print(
                f"⚠️  No membership data found for universe '{universe}' at {partition_path}"
            )
            return []

        # Read all parquet files in partition
        files = list(partition_path.glob("*.parquet"))
        if not files:
            print(f"⚠️  No parquet files found for universe '{universe}'")
            return []

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        # Deduplicate in case multiple builder runs created duplicate files
        if not df.empty:
            if "ingest_ts" in df.columns:
                df = df.sort_values("ingest_ts", ascending=False)
            df = df.drop_duplicates(
                subset=["ticker", "start_date", "end_date"], keep="first"
            )

        if df.empty:
            print(f"⚠️  No membership data found for universe '{universe}'")
            return []

        # Convert dates to date objects for comparison
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
        df["end_date"] = pd.to_datetime(df["end_date"]).dt.date

        # Convert parameters to date objects
        start_date_val = (
            start_date.date() if isinstance(start_date, pd.Timestamp) else start_date
        )
        end_date_val = (
            end_date.date() if isinstance(end_date, pd.Timestamp) else end_date
        )

        # Find all symbols with overlapping membership periods
        # Include if: (symbol_start <= period_end) AND (symbol_end >= period_start)
        historic_members = df[
            (df["start_date"] <= end_date_val) & (df["end_date"] >= start_date_val)
        ]

        symbols = historic_members["ticker"].unique().tolist()

        # Apply ticker mapping if enabled
        if ticker_mapper:
            original_count = len(symbols)
            # Resolve each symbol, filter out None (delisted with no successor)
            resolved = [ticker_mapper.resolve(s) for s in symbols]
            symbols = [s for s in resolved if s is not None]
            # Remove duplicates (e.g., FB and META both resolve to META)
            symbols = list(set(symbols))
            symbols.sort()  # Keep consistent ordering

            if original_count != len(symbols):
                print(
                    f"📝 Ticker mapping: {original_count} ticker symbols → {len(symbols)} current symbols"
                )

        print(f"✅ Found {len(symbols)} historic members (includes removed/delisted)")

        # Provide transparency about composition
        if not df.empty:
            # Count current members (end_date is recent/future)
            current_members = df[df["end_date"] >= pd.Timestamp.now().date()]
            current_symbols = current_members["ticker"].unique()

            # Count removed members (not in current list)
            removed_count = len(
                [s for s in symbols if s not in current_symbols.tolist()]
            )

            if removed_count > 0:
                print(
                    f"   → {len(current_symbols)} current members + "
                    f"{removed_count} removed/changed during period"
                )
                print("   ℹ️  Survivorship bias eliminated!")

        if len(symbols) <= 20:
            print(f"   Symbols: {symbols}")
        else:
            print(f"   Sample: {symbols[:10]} ... (showing first 10 of {len(symbols)})")

        return symbols
