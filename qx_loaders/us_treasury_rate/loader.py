"""
US Treasury Rate Loader Implementation
=======================================

Loads US Treasury rate data (risk-free rates) from curated storage.

This loader:
1. Loads treasury rate data for specified maturities
2. Filters to the requested date range
3. Returns a DataFrame with rate data

Usage in DAG:
-------------
```python
Task(
    id="LoadRiskFreeRate",
    run=run_loader(
        "qx_loaders/us_treasury_rate",
        overrides={
            "start_date": "2014-01-01",
            "end_date": "2024-12-31",
            "rate_types": ["3month"],
            "frequency": "daily"
        }
    ),
    deps=["BuildTreasuryRates"]
)
```
"""

from pathlib import Path

import pandas as pd

from qx.common.types import AssetClass, DatasetType, Domain, Frequency, Region
from qx.foundation.base_loader import BaseLoader


class USTreasuryRateLoader(BaseLoader):
    """
    Load US Treasury rate data for financial models.

    This loader provides a clean interface to fetch risk-free rates,
    commonly used in CAPM, Sharpe ratio calculations, discount rate models,
    and fixed income analysis.

    Parameters
    ----------
    start_date : str
        Start date for treasury rate data (YYYY-MM-DD)
    end_date : str
        End date for treasury rate data (YYYY-MM-DD)
    rate_types : list of str
        List of treasury maturities to load (e.g., ["3month", "10year"])
    frequency : str, optional
        Data frequency - "daily", "weekly", or "monthly" (default: "daily")

    Returns
    -------
    pd.DataFrame
        Treasury rate data with columns: date, rate_type, series_id, rate, frequency, source

    Examples
    --------
    >>> # Load 3-month T-bill rate for CAPM
    >>> loader = USTreasuryRateLoader(
    ...     package_dir="qx_loaders/us_treasury_rate",
    ...     registry=registry,
    ...     backend=backend,
    ...     resolver=resolver,
    ...     params={
    ...         "start_date": "2014-01-01",
    ...         "end_date": "2024-12-31",
    ...         "rate_types": ["3month"],
    ...         "frequency": "daily"
    ...     }
    ... )
    >>> rf_rates = loader.load()
    """

    def load_impl(self) -> pd.DataFrame:
        """
        Load US Treasury rate data.

        Returns
        -------
        pd.DataFrame
            Treasury rate data with columns: date, rate_type, series_id, rate, frequency, source

        Raises
        ------
        ValueError
            If no data found for the specified rate types and period
        """
        # Get parameters
        start_date = pd.Timestamp(self.params["start_date"])
        end_date = pd.Timestamp(self.params["end_date"])
        rate_types = self.params["rate_types"]
        frequency = self.params.get("frequency", "daily")

        print(f"📊 Loading US Treasury rate data")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print(f"   Rate types: {rate_types}")
        print(f"   Frequency: {frequency}")

        # Validate rate_types
        valid_rate_types = ["3month", "1year", "5year", "10year", "30year"]
        for rate_type in rate_types:
            if rate_type not in valid_rate_types:
                raise ValueError(
                    f"Invalid rate_type '{rate_type}'. "
                    f"Must be one of: {valid_rate_types}"
                )

        # Map frequency to enum
        freq_map = {
            "daily": Frequency.DAILY,
            "weekly": Frequency.WEEKLY,
            "monthly": Frequency.MONTHLY,
        }
        freq_enum = freq_map.get(frequency.lower(), Frequency.DAILY)

        # Load data from curated storage
        # Treasury data is partitioned by rate_type and frequency
        # Path: data/curated/risk-free/treasury_rate/schema_v1/rate_type={rate_type}/frequency={frequency}/
        base_path = Path("data/curated/risk-free/treasury_rate/schema_v1")

        all_dfs = []
        for rate_type in rate_types:
            rate_type_path = (
                base_path / f"rate_type={rate_type}" / f"frequency={frequency}"
            )

            if not rate_type_path.exists():
                print(f"   ⚠️  No data found for {rate_type} at {rate_type_path}")
                continue

            # Load all parquet files for this rate_type
            parquet_files = list(rate_type_path.glob("*.parquet"))
            if not parquet_files:
                print(f"   ⚠️  No parquet files found for {rate_type}")
                continue

            for pq_file in parquet_files:
                df_rate = pd.read_parquet(pq_file)
                all_dfs.append(df_rate)

            print(f"   ✅ Loaded {rate_type}: {len(parquet_files)} file(s)")

        if not all_dfs:
            raise ValueError(
                f"No data found for rate types {rate_types}. "
                f"Make sure data has been built using the us_treasury_rate builder. "
                f"Expected path: {base_path}"
            )

        # Combine all rate types
        all_data = pd.concat(all_dfs, ignore_index=True)

        # Ensure date column is datetime
        all_data["date"] = pd.to_datetime(all_data["date"])

        # Filter by date range
        treasury_df = all_data[
            (all_data["date"] >= pd.Timestamp(start_date))
            & (all_data["date"] <= pd.Timestamp(end_date))
        ].copy()

        if treasury_df.empty:
            raise ValueError(
                f"No treasury rate data found for rate types {rate_types} "
                f"in date range {start_date.date()} to {end_date.date()}"
            )

        # Sort by date and rate_type
        treasury_df = treasury_df.sort_values(["date", "rate_type"]).reset_index(
            drop=True
        )

        # Select output columns
        output_columns = [
            "date",
            "rate_type",
            "series_id",
            "rate",
            "frequency",
            "source",
        ]
        available_columns = [
            col for col in output_columns if col in treasury_df.columns
        ]
        result_df = treasury_df[available_columns].copy()

        print(f"✅ Loaded {len(result_df)} treasury rate observations")
        print(
            f"   Date range: {result_df['date'].min().date()} to {result_df['date'].max().date()}"
        )
        print(f"   Rate types: {sorted(result_df['rate_type'].unique().tolist())}")

        # Show summary statistics
        for rate_type in sorted(result_df["rate_type"].unique()):
            rate_data = result_df[result_df["rate_type"] == rate_type]["rate"]
            print(
                f"   {rate_type}: mean={rate_data.mean():.2f}%, std={rate_data.std():.2f}%, count={len(rate_data)}"
            )

        return result_df


# Example usage
if __name__ == "__main__":
    """
    Example standalone usage (requires curated treasury rate data).
    """
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.pathing import PathResolver

    print("=" * 80)
    print("US Treasury Rate Loader - Standalone Test")
    print("=" * 80)

    # Initialize infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()

    # Create loader
    loader = USTreasuryRateLoader(
        package_dir="qx_loaders/us_treasury_rate",
        registry=registry,
        backend=backend,
        resolver=resolver,
        params={
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "rate_types": ["3month", "10year"],
            "frequency": "daily",
        },
    )

    # Load data
    try:
        rf_data = loader.load()
        print(f"\n✅ Successfully loaded treasury rate data")
        print(f"\nFirst 10 rows:")
        print(rf_data.head(10))
        print(f"\nLast 10 rows:")
        print(rf_data.tail(10))
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: This example requires treasury rate data to be built first.")
        print("Run: python examples/test_treasury_builder.py")
