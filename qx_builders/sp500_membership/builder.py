"""
S&P 500 Membership Builder

Builds curated membership data (daily and intervals) from raw CSV.
Migrated from src/universe/sp500_universe.py to Qx architecture.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.foundation.base_builder import DataBuilderBase
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter


class SP500MembershipBuilder(DataBuilderBase):
    """
    Builder for S&P 500 historical membership data.

    Transforms raw CSV (date, tickers) into curated daily membership data.

    Supports both legacy and YAML-based initialization:
    - Legacy: contract, adapter, resolver, raw_data_root, membership_filename
    - YAML: package_dir, registry, adapter, resolver, overrides
    """

    def __init__(
        self,
        contract: Optional[DatasetContract] = None,
        adapter: Optional[TableFormatAdapter] = None,
        resolver: Optional[PathResolver] = None,
        raw_data_root: Optional[str] = None,
        membership_filename: Optional[str] = None,
        package_dir: Optional[str] = None,
        registry: Optional[DatasetRegistry] = None,
        overrides: Optional[dict] = None,
    ):
        """
        Initialize S&P 500 membership builder.

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
            self.raw_data_root = Path(raw_data_root or "./data/raw")
            self.membership_filename = (
                membership_filename
                or "S&P 500 Historical Components & Changes(11-16-2025).csv"
            )
        # YAML mode: parameters already loaded by parent into self.params
        else:
            self.raw_data_root = Path(self.params.get("raw_data_root", "./data/raw"))
            self.membership_filename = self.params.get(
                "membership_filename",
                "S&P 500 Historical Components & Changes(11-16-2025).csv",
            )

    def fetch_raw(self, **kwargs) -> pd.DataFrame:
        """
        Load raw S&P 500 membership CSV file.

        Returns:
            DataFrame with columns [date: str, tickers: str]
        """
        raw_csv_path = self.raw_data_root / self.membership_filename

        if not raw_csv_path.exists():
            # Fallback: try to find any matching file
            raw_csv_files = list(
                self.raw_data_root.glob("S&P 500 Historical Components*.csv")
            )
            if not raw_csv_files:
                raise FileNotFoundError(
                    f"No S&P 500 historical CSV found. Looking for: {raw_csv_path}"
                )
            raw_csv_path = raw_csv_files[0]
            print(f"⚠️  Configured file not found, using: {raw_csv_path.name}")

        print(f"📂 Loading raw CSV: {raw_csv_path}")

        df = pd.read_csv(raw_csv_path, engine="python")
        assert {"date", "tickers"}.issubset(
            df.columns
        ), f"CSV must have 'date' and 'tickers' columns, got {df.columns.tolist()}"

        return df

    def transform_to_curated(
        self, raw_df: pd.DataFrame, min_date: str = "2000-01-01", **kwargs
    ) -> pd.DataFrame:
        """
        Transform raw CSV to curated membership format (daily or intervals).

        Mode is determined by checking kwargs for 'mode' or from self.partition_spec.

        Daily mode:
            Explodes comma-separated tickers into individual rows:
            (date, "AAPL,MSFT,GOOGL") → [(date, "AAPL"), (date, "MSFT"), (date, "GOOGL")]

        Intervals mode:
            Synthesizes membership intervals from daily data:
            Daily records → [(ticker, start_date, end_date), ...]

        Args:
            raw_df: Raw DataFrame with [date, tickers]
            min_date: Minimum date to include (ISO format YYYY-MM-DD)
            **kwargs: May contain 'mode' or 'partitions' to determine output format

        Returns:
            Curated DataFrame with either:
            - Daily: [date: datetime, ticker: str]
            - Intervals: [ticker: str, start_date: date, end_date: date]
        """
        from qx_builders.sp500_membership.utils import synthesize_membership_intervals

        # Determine mode from kwargs or partition_spec
        mode = kwargs.get("mode")
        if mode is None:
            # Check if partitions were passed in kwargs
            partitions = kwargs.get("partitions", {})
            mode = partitions.get("mode", "daily")

        # Parse dates
        raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
        raw_df = raw_df.dropna(subset=["date"])

        # Explode comma-separated tickers into individual rows (always start with daily)
        records = []
        for date_val, tickers_str in zip(
            raw_df["date"].tolist(), raw_df["tickers"].astype(str).tolist()
        ):
            # Clean: strip quotes/spaces, split by comma
            tickers_clean = tickers_str.strip().strip('"').replace(" ", "")
            ticker_list = [t.upper() for t in tickers_clean.split(",") if t]
            for ticker in ticker_list:
                records.append((date_val, ticker))

        membership_daily = (
            pd.DataFrame(records, columns=["date", "ticker"])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Filter to minimum date
        min_date_ts = pd.Timestamp(min_date)
        membership_daily = membership_daily[membership_daily["date"] >= min_date_ts]

        print(f"✅ Curated: {len(membership_daily):,} daily membership records")
        print(
            f"   Date range: {membership_daily['date'].min().date()} to {membership_daily['date'].max().date()}"
        )
        print(f"   Unique tickers: {membership_daily['ticker'].nunique()}")

        # If intervals mode requested, synthesize intervals from daily
        if mode == "intervals":
            print(f"🔄 Synthesizing membership intervals from daily data...")
            membership_intervals = synthesize_membership_intervals(membership_daily)
            return membership_intervals

        return membership_daily
