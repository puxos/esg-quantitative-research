"""
GVKEY Mapping Utilities

Helper functions for GVKEY-ticker mapping operations.
"""

from pathlib import Path

import pandas as pd


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


def load_gvkey_mapping_from_curated(
    exchange: str = "US",
    base_path: str = "data/curated/metadata/gvkey_mapping/schema_v1",
) -> pd.DataFrame:
    """
    Load GVKEY mapping from curated storage.

    This is a convenience function for other builders (like ESGScoreBuilder)
    to load the mapping without going through the full builder pattern.

    Args:
        exchange: Exchange code (default: "US")
        base_path: Base path to curated mappings

    Returns:
        DataFrame with columns: gvkey, ticker, ticker_raw

    Raises:
        FileNotFoundError: If mapping file doesn't exist

    Example:
        >>> mapping = load_gvkey_mapping_from_curated()
        >>> print(mapping.head())
           gvkey ticker ticker_raw
        0   1004    AIR        AIR
        1   1045   ZEUS       ZEUS
    """
    full_path = Path(base_path) / f"exchange={exchange}"

    if not full_path.exists():
        raise FileNotFoundError(
            f"GVKEY mapping not found: {full_path}\n"
            f"Please run GVKEYMappingBuilder first to create the mapping."
        )

    # Find all parquet files
    parquet_files = list(full_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {full_path}")

    # Read most recent file (or combine all if multiple)
    if len(parquet_files) == 1:
        df = pd.read_parquet(parquet_files[0])
    else:
        # Combine all files and deduplicate
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["gvkey"], keep="last")

    # Select relevant columns
    return df[["gvkey", "ticker", "ticker_raw"]].copy()
