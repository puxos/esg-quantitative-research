"""
Tiingo Price Data Schema

Dataset contract for equity OHLCV data from Tiingo API.
Loaded from YAML schema definition.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract
from qx.common.types import Frequency

# Path to YAML schema file
SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


def _get_contract(exchange: str, frequency: Frequency) -> DatasetContract:
    """Get contract for specific exchange and frequency (internal helper)."""
    return load_contract(SCHEMA_PATH, exchange=exchange, frequency=frequency)


def get_contracts() -> list[DatasetContract]:
    """
    Get contracts for all Tiingo OHLCV frequencies (US exchange).

    Standard contract discovery function for auto-registration.
    Returns all frequency variants for US market.

    Returns:
        List of DatasetContract instances for US exchange (daily, weekly, monthly)

    Schema:
        - 18 columns: date, exchange, symbol, currency, frequency, OHLC prices,
          volumes, adjusted prices/volumes, dividends, splits
        - Partitioned by: exchange, frequency, symbol, year
        - Source: Tiingo API

    Example:
        contracts = get_contracts()
        # → [US daily, US weekly, US monthly]
    """
    return [
        _get_contract("US", Frequency.DAILY),
        _get_contract("US", Frequency.WEEKLY),
        _get_contract("US", Frequency.MONTHLY),
    ]
