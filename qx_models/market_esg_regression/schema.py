"""
Market + ESG Two-Factor Regression Model - Output Schema Definition

Defines the output contract for factor exposures (market beta and ESG beta).
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to YAML schema definition
SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


def get_market_esg_regression_contract() -> DatasetContract:
    """
    Get the dataset contract for two-factor regression betas.

    Output includes:
    - Cross-sectional: One row per stock (full-sample estimates)
    - Time-series: Multiple rows per stock (rolling window estimates)

    Columns:
    - symbol: Stock ticker
    - date: Regression end date (cross-sectional: end of sample, time-series: window end)
    - alpha: Jensen's alpha (abnormal return, monthly)
    - beta_market: Market beta (sensitivity to market factor)
    - beta_esg: ESG beta (sensitivity to ESG factor)
    - Statistics: t-stats, p-values, R², F-stat, standard errors, observations

    Returns:
        DatasetContract for processed/equity/two_factor_betas
    """
    return load_contract(SCHEMA_PATH)
