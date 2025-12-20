"""
S&P 500 Membership Schema

Dataset contracts for S&P 500 membership data (daily and intervals).
Loaded from unified builder.yaml definition.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contracts_from_builder_yaml

# Path to unified YAML file
BUILDER_YAML_PATH = Path(__file__).parent / "builder.yaml"


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
        # → [daily_contract, intervals_contract]
    """
    return load_contracts_from_builder_yaml(BUILDER_YAML_PATH)
