"""
Continuous Universe Loader

Selects symbols that were continuously in a universe during a specified period.

This is useful for backtesting and research where you want to ensure symbols
had stable membership (no additions/removals) during the analysis period.

Example usage:
    Task(
        id="SelectUniverse",
        run=run_loader(
            package_path="qx_loaders/continuous_universe",
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

from qx.common.types import DatasetType, Domain, Subdomain
from qx.foundation.base_loader import BaseLoader


class ContinuousUniverseLoader(BaseLoader):
    """
    Load symbols that were continuously in universe during period.

    Reads membership interval data and filters for symbols whose membership
    spans the entire analysis period (start_date to end_date).

    Parameters (from loader.yaml):
        start_date: Period start (YYYY-MM-DD)
        end_date: Period end (YYYY-MM-DD)
        universe: Universe identifier (default: "sp500")

    Returns:
        List[str]: Ticker symbols
    """

    def load_impl(self) -> List[str]:
        """
        Select continuous universe members.

        Returns:
            List of ticker symbols that were continuously in the universe
        """
        # Get parameters
        start_date = pd.Timestamp(self.params["start_date"])
        end_date = pd.Timestamp(self.params["end_date"])
        universe = self.params["universe"]

        print(f"📊 Loading continuous {universe} members")
        print(f"   Period: {start_date.date()} to {end_date.date()}")

        # Load membership intervals from curated data via direct file access
        from pathlib import Path

        membership_type = DatasetType(
            domain=Domain.MEMBERSHIP,
            asset_class=None,
            subdomain=Subdomain.INTERVALS,
            region=None,
            frequency=None,
        )

        # Construct partition path
        base_path = Path("data/curated/membership/intervals/schema_v1")
        partition_path = base_path / f"universe={universe}" / "mode=intervals"

        if not partition_path.exists():
            print(f"⚠️  No membership data found for universe '{universe}'")
            return []

        # Read all parquet files
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

        # Filter for continuous membership
        # Symbol must have been member from start_date to end_date
        # Convert Timestamps to dates for comparison
        start_date_val = (
            start_date.date() if isinstance(start_date, pd.Timestamp) else start_date
        )
        end_date_val = (
            end_date.date() if isinstance(end_date, pd.Timestamp) else end_date
        )

        continuous = df[
            (df["start_date"] <= start_date_val) & (df["end_date"] >= end_date_val)
        ]

        symbols = continuous["ticker"].unique().tolist()

        print(f"✅ Selected {len(symbols)} continuous members")
        if len(symbols) <= 10:
            print(f"   Symbols: {symbols}")
        else:
            print(f"   Sample: {symbols[:10]}...")

        return symbols
