"""
Schema definition for Factor Expected Returns Model output
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to YAML schema definition
MODEL_YAML_PATH = Path(__file__).parent / "model.yaml"


def get_contracts() -> list[DatasetContract]:
    """
    Schema contract for factor expected returns output.

    Standard contract discovery function for auto-registration.

    Output format:
        - One row per stock per forecast date (monthly frequency)
        - Contains factor exposures (betas), risk-free rate, and expected returns
        - Includes both original and capped betas (if capping applied)
        - Stores factor premia in metadata for reproducibility

    Columns:
        Identifiers:
            - symbol: Stock ticker symbol
            - date: Forecast date (monthly)
            - beta_date: Date when beta was estimated (for time-varying betas)

        Factor Exposures (Original):
            - beta_market: Market beta (original, uncapped)
            - beta_esg: ESG beta (original, uncapped)

        Factor Exposures (Capped):
            - beta_market_capped: Market beta after capping (if cap_betas=true)
            - beta_esg_capped: ESG beta after capping (if cap_betas=true)

        Risk-Free Rate:
            - RF: Risk-free rate (monthly decimal)

        Expected Returns:
            - ER_monthly: Expected return (monthly decimal)
            - ER_annual: Expected return (annualized, compound)

        Factor Premia (Metadata):
            - lambda_market: Market risk premium (monthly decimal)
            - lambda_esg: ESG risk premium (monthly decimal)

        Model Metadata:
            - model: Model ID ("factor_expected_returns")
            - model_version: Model version
            - featureset_id: Input featureset ID
            - run_id: Unique run identifier
            - run_ts: Run timestamp (ISO 8601)

    Partition Structure:
        data/processed/factor_expected_returns/model={model}/run_date={run_date}/

        Example:
            data/processed/factor_expected_returns/model=factor_expected_returns/run_date=2024-12-04/part-<run_id>.parquet

    Returns:
        List containing single DatasetContract for factor expected returns output
    """
    return [load_contract(MODEL_YAML_PATH)]
