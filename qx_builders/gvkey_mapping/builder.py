"""
GVKEY Mapping Builder

Builds GVKEY-to-ticker mapping from local Excel files.
This is metadata that ESG and other builders depend on.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from qx.foundation.base_builder import DataBuilderBase


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
    SOURCE BUILDER: GVKEY-ticker mapping metadata from local Excel file.

    External Source: Local Excel file (CRSP/Compustat data_mapping.xlsx)
    Transform: Excel (gvkey, tic) → Normalized mapping (gvkey, ticker, ticker_raw)
    Authentication: None (local file)

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

    def __init__(self, package_dir: str, writer, overrides: Optional[dict] = None):
        """
        Initialize GVKEY mapping builder from YAML configuration.

        Args:
            package_dir: Path to builder package containing builder.yaml
            writer: High-level curated data writer
            overrides: Parameter overrides
        """
        # Load YAML configuration
        super().__init__(package_dir=package_dir, writer=writer, overrides=overrides)

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
