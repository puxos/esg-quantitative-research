"""
GVKEY Mapping Schema

Dataset contract for GVKEY-to-ticker mapping metadata.
Loaded from YAML schema definition.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to YAML schema file
SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


def get_gvkey_mapping_contract(exchange: str = "US") -> DatasetContract:
    """
    Get GVKEY-to-ticker mapping contract.

    Loads contract from YAML schema definition with parameterized exchange.

    GVKEY (Global Company Key) is a unique identifier for companies in Compustat.
    Unlike ticker symbols, GVKEYs remain stable through ticker changes, mergers, etc.

    Args:
        exchange: Exchange for mapping (default: "US")

    Returns:
        DatasetContract for GVKEY mapping

    Schema:
        Defined in schema.yaml:
        - 3 columns: gvkey, ticker, ticker_raw
        - Partitioned by: exchange
        - Source: raw/data_mapping.xlsx (in package)
        - Used by: ESG builder, fundamentals (future)

    Example:
        contract = get_gvkey_mapping_contract("US")
        # → DatasetContract for US GVKEY mapping
    """
    return load_contract(SCHEMA_PATH, exchange=exchange)
