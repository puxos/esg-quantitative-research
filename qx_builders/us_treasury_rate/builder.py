"""
US Treasury Rate Builder

Fetches US Treasury rates from FRED API and builds curated datasets.
Migrated from src/market/risk_free_rate_builder.py to Qx architecture.

Naming Convention:
- USTreasuryRateBuilder: For US-specific implementation
- Future: HKTreasuryRateBuilder, UKTreasuryRateBuilder, etc.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.foundation.base_builder import DataBuilderBase
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter
from qx_builders.us_treasury_rate.utils import (
    FRED_SERIES,
    get_default_rate_types,
    get_fred_series_id,
)


class USTreasuryRateBuilder(DataBuilderBase):
    """
    Builder for US Treasury rates from FRED API.

    Fetches treasury constant maturity rates (3-month, 1-year, 5-year, 10-year, 30-year)
    from Federal Reserve Economic Data (FRED) and transforms to curated format.

    Supports two modes:
    1. YAML mode: Initialized from builder.yaml with run_builder()
    2. Legacy mode: Direct instantiation with explicit parameters

    Naming: USTreasuryRateBuilder (US-specific)
    Future: HKTreasuryRateBuilder, UKTreasuryRateBuilder for other countries
    """

    FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        contract: Optional[DatasetContract] = None,
        adapter: Optional[TableFormatAdapter] = None,
        resolver: Optional[PathResolver] = None,
        fred_api_key: Optional[str] = None,
        # YAML mode parameters
        package_dir: Optional[Path] = None,
        registry: Optional[DatasetRegistry] = None,
        overrides: Optional[Dict] = None,
    ):
        """
        Initialize US Treasury Rate builder.

        YAML mode (for DAG orchestration):
            builder = USTreasuryRateBuilder(
                package_dir=Path("qx_builders/us_treasury_rate"),
                registry=registry,
                adapter=adapter,
                resolver=resolver,
                overrides={"rate_types": ["10year"], "start_date": "2020-01-01"}
            )

        Legacy mode:
            builder = USTreasuryRateBuilder(
                contract=contract,
                adapter=adapter,
                resolver=resolver,
                fred_api_key="your_fred_api_key"
            )

        Args:
            contract: Dataset contract (legacy mode)
            adapter: Storage adapter for writing data
            resolver: Path resolver for output paths
            fred_api_key: FRED API key (legacy mode)
            package_dir: Path to builder package (YAML mode)
            registry: Contract registry (YAML mode)
            overrides: Parameter overrides (YAML mode)
        """
        if package_dir is not None:
            # YAML mode: Load builder.yaml and initialize from params
            super().__init__(
                package_dir=package_dir,
                registry=registry,
                adapter=adapter,
                resolver=resolver,
                overrides=overrides,
            )

            # Get FRED API key from params or environment
            api_key = self.params.get("fred_api_key") or os.environ.get("FRED_API_KEY")
            if not api_key:
                raise ValueError(
                    "FRED API key is required. "
                    "Set 'fred_api_key' parameter or FRED_API_KEY environment variable. "
                    "Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
                )

            self.fred_api_key = api_key

        else:
            # Legacy mode: Direct instantiation
            super().__init__(contract, adapter, resolver)

            if not fred_api_key:
                raise ValueError(
                    "FRED API key is required. "
                    "Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
                )

            self.fred_api_key = fred_api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def fetch_raw(self, **kwargs) -> pd.DataFrame:
        """
        Fetch raw treasury rate data from FRED API.

        Supports two modes:
        1. Batch mode: Fetch multiple rate_types in one call
        2. Single mode: Fetch one rate_type (legacy)

        Args:
            **kwargs: Keyword arguments
                rate_types (list): List of rate types to fetch (batch mode)
                rate_type (str): Single rate type (legacy mode)
                start_date (str): Start date in 'YYYY-MM-DD' format
                end_date (str): End date in 'YYYY-MM-DD' format
                fail_on_error (bool): If True, fail on any error; if False, skip failed rates

        Returns:
            Raw DataFrame with columns: [date, value, rate_type]
        """
        # Get parameters (YAML mode uses self.params, legacy uses kwargs)
        rate_types = kwargs.get("rate_types")
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        fail_on_error = kwargs.get("fail_on_error", False)

        # YAML mode: read from self.params
        if hasattr(self, "params"):
            rate_types = rate_types or self.params.get("rate_types", [])
            start_date = start_date or self.params.get("start_date")
            end_date = end_date or self.params.get("end_date")
            fail_on_error = fail_on_error or self.params.get("fail_on_error", False)

        # Legacy single rate_type support
        if not rate_types and "rate_type" in kwargs:
            rate_types = [kwargs["rate_type"]]

        # Default to all rate types if none specified
        if not rate_types:
            rate_types = get_default_rate_types()

        print(
            f"📡 Fetching {len(rate_types)} Treasury rate(s) from FRED API: {rate_types}"
        )

        all_data = []
        failed_rates = []

        for rate_type in rate_types:
            try:
                df = self._fetch_single_rate(rate_type, start_date, end_date)
                if not df.empty:
                    df["rate_type"] = rate_type
                    all_data.append(df)
                    print(f"  ✅ {rate_type}: {len(df)} observations")
                else:
                    print(f"  ⚠️  {rate_type}: No data returned")
                    failed_rates.append(rate_type)
            except Exception as e:
                print(f"  ❌ {rate_type}: Error - {e}")
                failed_rates.append(rate_type)
                if fail_on_error:
                    raise

        if not all_data:
            print(f"⚠️  No data fetched for any rate type")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)
        print(
            f"✅ Fetched {len(combined_df)} total observations across {len(all_data)} rate types"
        )

        if failed_rates:
            print(f"⚠️  Failed to fetch: {failed_rates}")

        return combined_df

    def _fetch_single_rate(
        self,
        rate_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch a single treasury rate from FRED API.

        Args:
            rate_type: Type of treasury rate ('3month', '1year', '5year', '10year', '30year')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            Raw DataFrame from FRED with columns: [date, value]
        """
        series_id = get_fred_series_id(rate_type)

        try:
            params = {
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
            }

            if start_date:
                params["observation_start"] = start_date
            if end_date:
                params["observation_end"] = end_date

            response = requests.get(self.FRED_API_URL, params=params)
            response.raise_for_status()

            data = response.json()

            if "observations" not in data:
                return pd.DataFrame()

            # Convert to DataFrame
            observations = data["observations"]
            df = pd.DataFrame(observations)

            return df

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error fetching {rate_type} from FRED: {e}")

    def transform_to_curated(self, raw_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform raw FRED data to curated treasury rate format.

        Handles both single and batch mode transformations.

        Args:
            raw_df: Raw DataFrame from FRED API with columns: [date, value, rate_type]
            **kwargs: Keyword arguments
                frequency (str): Target frequency ('daily', 'weekly', 'monthly')

        Returns:
            Curated DataFrame with canonical schema
        """
        if raw_df.empty:
            return raw_df

        # Get frequency parameter
        frequency = kwargs.get("frequency")
        if hasattr(self, "params"):
            frequency = frequency or self.params.get("frequency", "daily")
        if not frequency:
            frequency = "daily"

        # Parse date and rate
        raw_df = raw_df.copy()
        raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.date
        raw_df["rate"] = pd.to_numeric(raw_df["value"], errors="coerce")

        # Remove missing values (marked as '.' in FRED)
        raw_df = raw_df.dropna(subset=["rate"])

        # Resample to requested frequency if not daily
        if frequency != "daily":
            # Group by rate_type and resample each
            resampled_groups = []
            for rate_type, group in raw_df.groupby("rate_type"):
                resampled = self._resample_to_frequency(group, frequency)
                resampled["rate_type"] = rate_type
                resampled_groups.append(resampled)
            raw_df = pd.concat(resampled_groups, ignore_index=True)

        # Create canonical DataFrame
        curated = pd.DataFrame(
            {
                "date": raw_df["date"],
                "rate_type": raw_df["rate_type"],
                "series_id": raw_df["rate_type"].apply(get_fred_series_id),
                "rate": raw_df["rate"].astype(float),
                "frequency": frequency,
                "source": "FRED",
            }
        )

        print(f"✅ Curated: {len(curated)} {frequency} treasury rate observations")
        print(f"   Date range: {curated['date'].min()} to {curated['date'].max()}")
        print(
            f"   Rate stats: mean={curated['rate'].mean():.2f}%, std={curated['rate'].std():.2f}%"
        )

        # Show per-rate breakdown
        for rate_type in curated["rate_type"].unique():
            count = len(curated[curated["rate_type"] == rate_type])
            print(f"   {rate_type}: {count} observations")

        return curated

    def _resample_to_frequency(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Resample daily data to weekly or monthly frequency.

        Args:
            df: DataFrame with daily data [date, rate]
            frequency: Target frequency ('weekly', 'monthly')

        Returns:
            Resampled DataFrame
        """
        # Convert date to datetime for resampling
        df_temp = df.copy()
        df_temp["date"] = pd.to_datetime(df_temp["date"])
        df_temp = df_temp.set_index("date")

        # Resample (take last value of period)
        if frequency == "weekly":
            df_resampled = df_temp.resample("W-FRI").last()  # Week ending Friday
        elif frequency == "monthly":
            df_resampled = df_temp.resample("ME").last()  # Month end
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        # Reset index and convert back to date
        df_resampled = df_resampled.reset_index()
        df_resampled["date"] = df_resampled["date"].dt.date

        # Remove rows with no data (NaN rates)
        df_resampled = df_resampled.dropna(subset=["rate"])

        return df_resampled

    def build_for_rate_type(
        self,
        rate_type: str,
        frequency: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """
        Build treasury rate data for a specific rate type.

        This method handles the complete pipeline:
        1. Fetch from FRED API
        2. Transform to curated format
        3. Write to partitioned storage

        Args:
            rate_type: Type of treasury rate ('3month', '1year', '5year', '10year', '30year')
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Output path
        """
        # Fetch and transform
        raw_df = self.fetch_raw(
            rate_type=rate_type, start_date=start_date, end_date=end_date
        )

        if raw_df.empty:
            print(f"⚠️  No data to build for {rate_type}")
            return ""

        curated = self.transform_to_curated(
            raw_df, rate_type=rate_type, frequency=frequency
        )

        # Add schema metadata
        curated = curated.copy()
        curated["schema_version"] = self.contract.schema_version
        curated["ingest_ts"] = pd.Timestamp.utcnow()

        # Define partitions
        partitions = {"region": "US", "rate_type": rate_type, "frequency": frequency}

        rel_dir = self.resolver.curated_dir(self.contract, partitions)
        filename = f"part-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"

        output_path = self.adapter.write_batch(curated, rel_dir, filename)
        print(f"💾 Saved: {output_path} ({len(curated)} rows)")

        return output_path
