"""
ESG Score Schema

Dataset contract for ESG (Environmental, Social, Governance) scores.
Loaded from YAML schema definition.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to YAML schema file
SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


def get_contracts() -> list[DatasetContract]:
    """
    Get ESG scores dataset contract.

    Standard contract discovery function for auto-registration.

    Returns:
        List containing single DatasetContract for ESG scores

    Schema:
        - 7 columns: ticker, gvkey, esg_year, esg_score,
          environmental_pillar_score, social_pillar_score, governance_pillar_score
        - Partitioned by: exchange, esg_year
        - Frequency: Annual (one record per company per ESG publication year)
        - Source: raw/data_matlab_ESG_withSIC.xlsx (in package)
        - Dependencies: gvkey_mapping

    Example:
        contracts = get_contracts()
        # → [DatasetContract for annual ESG scores]
    """
    return [load_contract(SCHEMA_PATH)]
