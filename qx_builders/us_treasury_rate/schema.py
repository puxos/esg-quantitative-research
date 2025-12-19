"""
US Treasury Rate Schema

Dataset contract for US Treasury rates from FRED API.
Loaded from YAML schema definition.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to YAML schema file
SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


def _get_base_contract() -> DatasetContract:
    """Get base contract from YAML schema (internal helper)."""
    return load_contract(SCHEMA_PATH)


def get_contracts() -> list[DatasetContract]:
    """
    Get contracts for all US Treasury rate frequencies.

    Standard contract discovery function for auto-registration.
    Returns frequency-specific contracts (daily, weekly, monthly).

    Returns:
        List of DatasetContract instances (daily, weekly, monthly)

    Schema:
        - 6 columns: date, rate_type, series_id, rate, frequency, source
        - Partitioned by: rate_type, frequency
        - Source: Federal Reserve Economic Data (FRED API)
        - Rate types: 3month, 1year, 5year, 10year, 30year
        - Region: US (implied)

    Example:
        contracts = get_contracts()
        # → [daily, weekly, monthly treasury contracts]
    """
    from dataclasses import replace

    from qx.common.types import DatasetType, Frequency

    base_contract = _get_base_contract()

    contracts = []
    for freq in (Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY):
        # Create contract with specific frequency in dataset_type
        updated_type = DatasetType(
            domain=base_contract.dataset_type.domain,
            asset_class=base_contract.dataset_type.asset_class,
            subdomain=base_contract.dataset_type.subdomain,
            subtype=base_contract.dataset_type.subtype,
            region=base_contract.dataset_type.region,
            frequency=freq,
        )
        contract = replace(base_contract, dataset_type=updated_type)
        contracts.append(contract)

    return contracts
