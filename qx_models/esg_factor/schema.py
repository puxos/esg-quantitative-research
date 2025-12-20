"""
ESG Factor Model - Output Schema Definition

Defines the output contract for ESG factor returns.
Loaded from unified model.yaml configuration.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to unified model YAML file
MODEL_YAML_PATH = Path(__file__).parent / "model.yaml"


def get_contracts() -> list[DatasetContract]:
    """
    Get the dataset contract for ESG factor returns.

    Standard contract discovery function for auto-registration.

    Output includes:
    - Level factors: ESG, E, S, G (long-short portfolios based on pillar scores)
    - Momentum factor: ESG_mom (long-short based on YoY ESG score changes)

    Returns:
        List containing single DatasetContract for processed/equity/factors
    """
    return [load_contract(MODEL_YAML_PATH)]
