"""
ESG Factor Model - Output Schema Definition

Defines the output contract for ESG factor returns.
"""

from qx.common.contracts import DatasetContract
from qx.common.types import AssetClass, DatasetType, Domain


def get_esg_factors_contract() -> DatasetContract:
    """
    Get the dataset contract for ESG factor returns.

    Output includes:
    - Level factors: ESG, E, S, G (long-short portfolios based on pillar scores)
    - Momentum factor: ESG_mom (long-short based on YoY ESG score changes)

    Returns:
        DatasetContract for processed/equity/esg_factors
    """
    dt = DatasetType(
        domain=Domain.DERIVED_METRICS,
        asset_class=AssetClass.EQUITY,
        subdomain="esg_factors",
        region=None,
        frequency=None,
    )

    return DatasetContract(
        dataset_type=dt,
        schema_version="schema_v1",
        required_columns=(
            # Business columns
            "date",  # Monthly observation date
            "factor_name",  # Factor identifier (ESG, E, S, G, ESG_mom)
            "factor_return",  # Long-short excess return
            "long_return",  # Long leg excess return
            "short_return",  # Short leg excess return
            # Metadata (auto-added by BaseModel)
            "model",  # "esg_factor"
            "model_version",  # e.g., "1.0.0"
            "featureset_id",  # e.g., "ohlcv_v1+esg_v1+rf_v1"
            "run_id",  # Unique run identifier
            "run_ts",  # Run timestamp
        ),
        partition_keys=("output_type", "model", "run_date"),
        path_template="data/processed/{output_type}/model={model}/run_date={run_date}",
    )
