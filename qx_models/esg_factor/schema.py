"""
ESG Factor Model - Output Schema Definition

Defines the output contract for ESG factor returns.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to YAML schema definition
SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


def get_esg_factors_contract() -> DatasetContract:
    """
    Get the dataset contract for ESG factor returns.

    Output includes:
    - Level factors: ESG, E, S, G (long-short portfolios based on pillar scores)
    - Momentum factor: ESG_mom (long-short based on YoY ESG score changes)

    Returns:
        DatasetContract for processed/equity/factors
    """
    return load_contract(SCHEMA_PATH)
