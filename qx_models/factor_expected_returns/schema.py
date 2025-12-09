"""
Schema definition for Factor Expected Returns Model output
"""

from qx.common.contracts import DatasetContract
from qx.common.types import AssetClass, DatasetType, Domain


def get_factor_expected_returns_contract() -> DatasetContract:
    """
    Schema contract for factor expected returns output.

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
        DatasetContract for factor expected returns output
    """
    dt = DatasetType(
        domain=Domain.DERIVED_METRICS,
        asset_class=AssetClass.EQUITY,
        subdomain="factor_expected_returns",
        region=None,
        frequency=None,
    )

    return DatasetContract(
        dataset_type=dt,
        schema_version="schema_v1",
        required_columns=(
            # Identifiers
            "symbol",
            "date",
            "beta_date",  # When beta was estimated (for time-varying betas)
            # Factor exposures (original)
            "beta_market",
            "beta_esg",
            # Factor exposures (capped)
            "beta_market_capped",
            "beta_esg_capped",
            # Risk-free rate
            "RF",  # Monthly decimal
            # Expected returns
            "ER_monthly",  # Monthly decimal
            "ER_annual",  # Annualized (compound): (1 + ER_monthly)^12 - 1
            # Factor premia (for reproducibility and analysis)
            "lambda_market",  # Monthly decimal
            "lambda_esg",  # Monthly decimal
            # Model metadata
            "model",
            "model_version",
            "featureset_id",
            "run_id",
            "run_ts",
        ),
        partition_keys=("output_type", "model", "run_date"),
        path_template="data/processed/{output_type}/model={model}/run_date={run_date}",
    )
