"""
S&P 500 Membership Schema

Dataset contracts for S&P 500 membership data (daily and intervals).
Loaded from YAML schema definitions.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract


# Paths to YAML schema files
SCHEMA_DAILY_PATH = Path(__file__).parent / "schema_daily.yaml"
SCHEMA_INTERVALS_PATH = Path(__file__).parent / "schema_intervals.yaml"


def get_sp500_daily_contract() -> DatasetContract:
    """
    Get contract for daily S&P 500 membership data.

    Loads contract from YAML schema definition (schema_daily.yaml).

    Returns:
        DatasetContract for daily membership

    Schema:
        - 2 columns: date, ticker
        - Partitioned by: universe, mode
        - Used for: Daily membership checks, point-in-time coverage

    Example:
        contract = get_sp500_daily_contract()
        # → DatasetContract for daily membership
    """
    return load_contract(SCHEMA_DAILY_PATH)


def get_sp500_intervals_contract() -> DatasetContract:
    """
    Get contract for interval-based S&P 500 membership data.

    Loads contract from YAML schema definition (schema_intervals.yaml).

    Returns:
        DatasetContract for interval membership

    Schema:
        - 3 columns: ticker, start_date, end_date
        - Partitioned by: universe, mode
        - Used for: Efficient membership queries, storage optimization

    Example:
        contract = get_sp500_intervals_contract()
        # → DatasetContract for interval membership
    """
    return load_contract(SCHEMA_INTERVALS_PATH)
