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


def get_tiingo_ohlcv_contract(exchange: str, frequency: Frequency) -> DatasetContract:
    """
    Get contract for Tiingo OHLCV price data.

    Loads contract from YAML schema definition with parameterized exchange and frequency.

    Args:
        exchange: Stock exchange (e.g., "US", "NYSE", "NASDAQ", "HKEX")
        frequency: Data frequency (e.g., Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY)

    Returns:
        DatasetContract for OHLCV data

    Schema:
        Defined in schema.yaml:
        - 18 columns: date, exchange, symbol, currency, frequency, OHLC prices,
          volumes, adjusted prices/volumes, dividends, splits
        - Partitioned by: exchange, frequency, symbol, year
        - Source: Tiingo API

    Example:
        contract = get_tiingo_ohlcv_contract("US", Frequency.DAILY)
        # → DatasetContract for US daily OHLCV data
    """
    return load_contract(SCHEMA_PATH, exchange=exchange, frequency=frequency)
