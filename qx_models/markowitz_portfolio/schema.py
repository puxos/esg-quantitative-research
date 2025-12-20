"""
Schema definition for Markowitz Portfolio Optimization output
"""

from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

# Path to YAML schema definition
MODEL_YAML_PATH = Path(__file__).parent / "model.yaml"


def get_contracts() -> list[DatasetContract]:
    """
    Schema contract for optimal portfolio weights output.

    Standard contract discovery function for auto-registration.

    Output format:
        - One row per stock in optimal portfolio (active positions only)
        - Contains allocation weights, expected returns, and risk metrics
        - Portfolio-level statistics included as metadata

    Columns:
        Identifiers:
            - symbol: Stock ticker symbol
            - optimization_date: Date when portfolio was optimized

        Portfolio Allocation:
            - weight: Optimal allocation (0-1, sums to 1 across portfolio)

        Expected Metrics:
            - exp_return_monthly: Expected return (monthly decimal)
            - exp_return_annual: Expected return (annualized)

        Risk Metrics:
            - esg_beta: ESG exposure (β_ESG)
            - sector: GICS sector (if available)

        Portfolio-Level Statistics (same for all rows):
            - portfolio_return_monthly: Portfolio expected return (monthly)
            - portfolio_return_annual: Portfolio expected return (annual)
            - portfolio_vol_monthly: Portfolio volatility (monthly)
            - portfolio_vol_annual: Portfolio volatility (annualized)
            - portfolio_sharpe: Portfolio Sharpe ratio
            - portfolio_esg_exposure: Portfolio ESG exposure (β_ESG'w)
            - portfolio_concentration_top10: Weight in top 10 holdings
            - n_positions: Number of active positions

        Optimization Parameters (for reproducibility):
            - gamma: Risk aversion parameter used
            - esg_lower_bound: ESG lower bound constraint
            - esg_upper_bound: ESG upper bound constraint
            - solver: Optimization solver used

        Model Metadata:
            - model: Model ID ("markowitz_portfolio")
            - model_version: Model version
            - featureset_id: Input featureset ID
            - run_id: Unique run identifier
            - run_ts: Run timestamp (ISO 8601)

    Partition Structure:
        data/processed/portfolio_weights/model={model}/run_date={run_date}/

        Example:
            data/processed/portfolio_weights/model=markowitz_portfolio/run_date=2024-12-04/part-<run_id>.parquet

    Returns:
        List containing single DatasetContract for portfolio weights output
    """
    return [load_contract(MODEL_YAML_PATH)]
