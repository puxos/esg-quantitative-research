"""
Universe at Date Loader

Selects symbols that were in a universe on a specific date (point-in-time).

This is useful for portfolio construction where you want the universe composition
on a specific rebalancing date.

Example usage:
    Task(
        id="SelectUniverse",
        run=run_loader(
            package_path="qx_loaders/universe_at_date",
            registry=registry,
            backend=backend,
            resolver=resolver,
            overrides={
                "date": "2024-12-31",
                "universe": "sp500"
            }
        )
    )
"""

from typing import List

import pandas as pd

from qx.common.types import DatasetType, Domain, Subdomain
from qx.foundation.base_loader import BaseLoader


class UniverseAtDateLoader(BaseLoader):
    """
    Load symbols that were in universe on a specific date.

    Reads membership interval data and filters for symbols whose membership
    includes the query date.

    Parameters (from loader.yaml):
        date: Query date (YYYY-MM-DD)
        universe: Universe identifier (default: "sp500")

    Returns:
        List[str]: Ticker symbols
    """

    def load_impl(self) -> List[str]:
        """
        Select universe members at specific date.

        Returns:
            List of ticker symbols that were in the universe on the date
        """
        # Get parameters
        query_date = pd.Timestamp(self.params["date"])
        universe = self.params["universe"]

        print(f"📊 Loading {universe} members at {query_date.date()}")

        # Load membership intervals from curated data via typed loader (contract-based)
        membership_type = DatasetType(
            domain=Domain.MEMBERSHIP,
            asset_class=None,
            subdomain=Subdomain.INTERVALS,
            region=None,
            frequency=None,
        )

        # Use typed loading instead of direct file access
        try:
            df = self.curated_loader.load(
                dataset_type=membership_type,
                partitions={"universe": universe, "mode": "intervals"},
            )
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            return []

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

        # Filter for members on query_date
        # Symbol must have start_date <= query_date <= end_date
        # Convert Timestamp to date for comparison
        query_date_val = (
            query_date.date() if isinstance(query_date, pd.Timestamp) else query_date
        )

        members_at_date = df[
            (df["start_date"] <= query_date_val) & (df["end_date"] >= query_date_val)
        ]

        symbols = members_at_date["ticker"].unique().tolist()

        print(f"✅ Found {len(symbols)} members on {query_date.date()}")
        if len(symbols) <= 10:
            print(f"   Symbols: {symbols}")
        else:
            print(f"   Sample: {symbols[:10]}...")

        return symbols
