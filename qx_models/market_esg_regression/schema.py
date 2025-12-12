"""
Market + ESG Two-Factor Regression Model - Output Schema Definition

Defines the output contract for factor exposures (market beta and ESG beta).
"""

from qx.common.contracts import DatasetContract
from qx.common.types import AssetClass, DatasetType, Domain, Subdomain


def get_market_esg_betas_contract() -> DatasetContract:
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
    dt = DatasetType(
        domain=Domain.DERIVED_METRICS,
        asset_class=AssetClass.EQUITY,
        subdomain=Subdomain.MODELS,
        subtype="market-esg-regression",
        region=None,
        frequency=None,
    )

    return DatasetContract(
        dataset_type=dt,
        schema_version="schema_v1",
        required_columns=(
            # Identifiers
            "symbol",  # Stock ticker
            "date",  # Regression end date
            # Factor exposures
            "alpha",  # Jensen's alpha (monthly)
            "beta_market",  # Market beta
            "beta_esg",  # ESG beta
            # Statistical inference (t-statistics)
            "alpha_tstat",  # Alpha t-statistic
            "beta_market_tstat",  # Market beta t-statistic
            "beta_esg_tstat",  # ESG beta t-statistic
            # Statistical inference (p-values)
            "alpha_pvalue",  # Alpha p-value
            "beta_market_pvalue",  # Market beta p-value
            "beta_esg_pvalue",  # ESG beta p-value
            # Model diagnostics
            "r_squared",  # R² (goodness of fit)
            "adj_r_squared",  # Adjusted R²
            "f_statistic",  # F-statistic (joint significance)
            "f_pvalue",  # F-statistic p-value
            "observations",  # Number of observations used
            # Standard errors (HAC-robust)
            "std_error_alpha",  # Standard error of alpha
            "std_error_beta_market",  # Standard error of market beta
            "std_error_beta_esg",  # Standard error of ESG beta
            # Metadata (auto-added by BaseModel)
            "model",  # "two_factor_regression"
            "model_version",  # e.g., "1.0.0"
            "featureset_id",  # e.g., "ohlcv_v1+esg_factors_v1+rf_v1"
            "run_id",  # Unique run identifier
            "run_ts",  # Run timestamp
        ),
        partition_keys=("output_type", "model", "run_date"),
        path_template="data/processed/{output_type}/model={model}/run_date={run_date}",
    )
