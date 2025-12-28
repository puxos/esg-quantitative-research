"""
Dataset Contract Schema for Portfolio Backtest Model

Defines the output schema for portfolio backtest results.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to YAML schema definition
MODEL_YAML_PATH = Path(__file__).parent / "model.yaml"


def get_contracts() -> list[DatasetContract]:
    """
    Get dataset contracts for portfolio backtest model.

    Standard contract discovery function for auto-registration.

    Returns:
        List of DatasetContract objects
    """
    return [load_contract(MODEL_YAML_PATH)]
