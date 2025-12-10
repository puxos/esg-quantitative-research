"""
GVKEY Mapping Builder

Builds GVKEY-to-ticker mapping from local Excel files.
This is metadata that ESG and other builders depend on.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from qx.common.contracts import DatasetContract
from qx.foundation.base_builder import DataBuilderBase
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter

if TYPE_CHECKING:
    from qx.common.contracts import DatasetRegistry


def clean_ticker_symbol(ticker: str) -> str:
    """
    Clean ticker symbol to normalized format.

    Handles:
    - Class shares (e.g., BRK.B → BRK-B)
    - Preferred shares (e.g., JPM-C)
    - Units and warrants (e.g., SPCE-WT)

    Args:
        ticker: Raw ticker string

    Returns:
        Cleaned ticker symbol

    Example:
        >>> clean_ticker_symbol("BRK.B")
        'BRK-B'
        >>> clean_ticker_symbol("aapl")
        'AAPL'
    """
    if pd.isna(ticker) or ticker == "":
        return ""

    ticker_str = str(ticker).strip().upper()

    # Replace period with hyphen for class shares (BRK.B → BRK-B)
    if "." in ticker_str:
        ticker_str = ticker_str.replace(".", "-")

    return ticker_str


class GVKEYMappingBuilder(DataBuilderBase):
    """
    Builder for GVKEY-ticker mapping metadata.

    GVKEY provides a stable company identifier that survives:
    - Ticker symbol changes
    - Corporate restructurings
    - Exchange migrations
    - Mergers and acquisitions

    This mapping is essential for:
    - ESG data (GVKEY → ticker)
    - Historical backfilling (resolve old tickers)
    - Cross-dataset linkage

    YAML-based initialization only - uses builder.yaml configuration.
    """

    def __init__(
        self,
        package_dir: str,
        registry: "DatasetRegistry",
        adapter: TableFormatAdapter,
        resolver: PathResolver,
        overrides: Optional[dict] = None,
    ):
        """
        Initialize GVKEY mapping builder from YAML configuration.

        Args:
            package_dir: Path to builder package containing builder.yaml
            registry: Dataset registry for resolving contracts
            adapter: Table format adapter for writing curated data
            resolver: Path resolver for output paths
            overrides: Parameter overrides
        """
        # Load YAML configuration
        super().__init__(
            package_dir=package_dir,
            registry=registry,
            adapter=adapter,
            resolver=resolver,
            overrides=overrides,
        )

        # Get raw file path from params (already resolved by base class)
        self.raw_file_path = self.params.get("crsp_file", "./raw/data_mapping.xlsx")

    def fetch_raw(self, **kwargs) -> pd.DataFrame:
        """
        Fetch raw GVKEY-ticker mapping from Excel file.

        Returns:
            Raw DataFrame with columns from Excel (gvkey, tic, etc.)
        """
        raw_path = Path(self.raw_file_path)

        if not raw_path.exists():
            raise FileNotFoundError(
                f"GVKEY mapping file not found: {raw_path}\n"
                f"Please ensure data_mapping.xlsx is in data/raw/"
            )

        print(f"📂 Loading GVKEY mapping from {raw_path}")

        try:
            df = pd.read_excel(raw_path)
            print(f"✅ Loaded {len(df):,} raw mappings")
            print(f"   Columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            raise RuntimeError(f"Error reading GVKEY mapping file: {e}")

    def transform_to_curated(self, raw_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform raw GVKEY mapping to curated format.

        Args:
            raw_df: Raw DataFrame from Excel

        Returns:
            Curated DataFrame with canonical schema
        """
        if raw_df.empty:
            return raw_df

        # Make a copy to avoid modifying original
        df = raw_df.copy()

        # Ensure required columns exist
        if "gvkey" not in df.columns:
            raise ValueError(
                f"Missing 'gvkey' column in mapping file. Found: {df.columns.tolist()}"
            )

        if "tic" not in df.columns:
            raise ValueError(
                f"Missing 'tic' column in mapping file. Found: {df.columns.tolist()}"
            )

        # Clean and normalize data
        df["ticker_raw"] = df["tic"].astype(str).str.strip()
        df["ticker"] = df["ticker_raw"].apply(clean_ticker_symbol)
        df["gvkey"] = df["gvkey"].astype(int)

        # Remove entries with empty tickers
        df = df[df["ticker"] != ""].copy()

        # Remove duplicates (prefer first occurrence for each GVKEY)
        df = df.drop_duplicates(subset=["gvkey"], keep="first")

        # Remove duplicates for each ticker (prefer first occurrence)
        # This handles cases where multiple GVKEYs map to same ticker (rare but possible)
        df = df.drop_duplicates(subset=["ticker"], keep="first")

        # Select required columns
        curated = df[["gvkey", "ticker", "ticker_raw"]].copy()

        print(f"✅ Curated: {len(curated):,} unique GVKEY-ticker mappings")
        print(f"   Unique GVKEYs: {curated['gvkey'].nunique():,}")
        print(f"   Unique tickers: {curated['ticker'].nunique():,}")

        # Show sample
        print(f"\nSample mappings:")
        print(curated.head(5).to_string(index=False))

        return curated

    def build(
        self, partitions: Optional[dict] = None, exchange: Optional[str] = None
    ) -> str:
        """
        Build GVKEY mapping data.

        Args:
            partitions: Partition values (e.g., {"exchange": "US"})
            exchange: Exchange code (legacy parameter, overridden by partitions)

        Returns:
            Output path
        """
        # Extract exchange from partitions or use parameter
        partitions = partitions or {}
        exchange_value = partitions.get("exchange", exchange or "US")

        # YAML mode: resolve contract with actual partition values
        if (
            self.contract is None
            and hasattr(self, "registry")
            and self.registry is not None
        ):
            from qx.common.types import DatasetType

            # Create actual output_dt with resolved values
            actual_dt = DatasetType(
                domain=self.output_dt_template.domain,
                asset_class=self.output_dt_template.asset_class,
                subdomain=self.output_dt_template.subdomain,
                region=None,
                frequency=None,
                dims=self.output_dt_template.dims,
            )

            # Find contract for actual dataset type
            self.contract = self.registry.find(actual_dt)
            if self.contract is None:
                raise ValueError(f"No contract found for output type: {actual_dt}")

        print(f"🏗️  Building GVKEY mapping for exchange={exchange_value}")

        # Fetch and transform
        raw_df = self.fetch_raw()
        curated = self.transform_to_curated(raw_df)

        if curated.empty:
            print("⚠️  No mappings to save")
            return ""

        # Add schema metadata
        curated = curated.copy()
        curated["schema_version"] = self.contract.schema_version
        curated["ingest_ts"] = pd.Timestamp.utcnow()

        # Define partitions (use the exchange value we extracted earlier)
        write_partitions = {"exchange": exchange_value}

        rel_dir = self.resolver.curated_dir(self.contract, write_partitions)
        filename = f"part-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"

        output_path = self.adapter.write_batch(curated, rel_dir, filename)
        print(f"💾 Saved: {output_path} ({len(curated)} rows)")

        return output_path
