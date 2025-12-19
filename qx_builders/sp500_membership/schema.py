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


def _get_daily_contract() -> DatasetContract:
    """Get contract for daily membership (internal helper)."""
    return load_contract(SCHEMA_DAILY_PATH)


def _get_intervals_contract() -> DatasetContract:
    """Get contract for interval membership (internal helper)."""
    return load_contract(SCHEMA_INTERVALS_PATH)


def get_contracts() -> list[DatasetContract]:
    """
    Get contracts for S&P 500 membership data (both modes).

    Standard contract discovery function for auto-registration.
    Returns contracts for both daily and intervals modes.

    Returns:
        List of DatasetContract instances [daily, intervals]

    Schemas:
        - Daily: 2 columns (date, ticker), for point-in-time checks
        - Intervals: 3 columns (ticker, start_date, end_date), for continuity analysis
        - Both partitioned by: universe, mode

    Example:
        contracts = get_contracts()
        # → [daily contract, intervals contract]
    """
    return [_get_daily_contract(), _get_intervals_contract()]
