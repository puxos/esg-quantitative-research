"""
GVKEY Mapping Schema

Dataset contract for GVKEY-to-ticker mapping metadata.
Loaded from unified builder.yaml definition.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to unified YAML file
BUILDER_YAML_PATH = Path(__file__).parent / "builder.yaml"


def get_contracts() -> list[DatasetContract]:
    """
    Get GVKEY-to-ticker mapping contract.

    Standard contract discovery function for auto-registration.
    Returns contract for US exchange.

    GVKEY (Global Company Key) is a unique identifier for companies in Compustat.
    Unlike ticker symbols, GVKEYs remain stable through ticker changes, mergers, etc.

    Returns:
        List containing single DatasetContract for GVKEY mapping

    Schema:
        - 3 columns: gvkey, ticker, ticker_raw
        - Partitioned by: exchange
        - Source: raw/data_mapping.xlsx (in package)
        - Used by: ESG builder, fundamentals (future)

    Example:
        contracts = get_contracts()
        # → [DatasetContract for US GVKEY mapping]
    """
    return [load_contract(BUILDER_YAML_PATH)]
