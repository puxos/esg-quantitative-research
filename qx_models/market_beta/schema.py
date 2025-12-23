"""
Market Beta Model - Output Schema Definition

Defines the output contract for market beta estimates (CAPM single-factor regression).
Loaded from unified model.yaml configuration.
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to unified model YAML file
MODEL_YAML_PATH = Path(__file__).parent / "model.yaml"


def get_contracts() -> list[DatasetContract]:
    """
    Get the dataset contract for market beta estimates.

    Standard contract discovery function for auto-registration.

    Output includes:
    - symbol: Ticker symbol
    - date: Estimation date (end of rolling window)
    - alpha: Jensen's alpha (intercept)
    - beta: Market beta (systematic risk)
    - alpha_tstat: Alpha t-statistic (Newey-West HAC)
    - beta_tstat: Beta t-statistic (Newey-West HAC)
    - r_squared: R-squared (goodness of fit)
    - residual_vol: Residual volatility (idiosyncratic risk)
    - observations: Number of observations in regression
    - window_start: Start date of regression window
    - window_end: End date of regression window
    - model: Model identifier
    - model_version: Model version
    - run_id: Unique run identifier
    - run_ts: Run timestamp

    Returns:
        List containing single DatasetContract for processed/derived-metrics/factor-exposures
    """
    return [load_contract(MODEL_YAML_PATH)]
