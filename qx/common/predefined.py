from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.types import AssetClass, DatasetType, Domain, Frequency
from qx_builders.esg_score import get_esg_scores_contract
from qx_builders.gvkey_mapping import get_gvkey_mapping_contract

# Import schema functions from builder packages
from qx_builders.sp500_membership import (
    get_sp500_daily_contract,
    get_sp500_intervals_contract,
)
from qx_builders.tiingo_ohlcv import get_tiingo_ohlcv_contract
from qx_builders.us_treasury_rate import get_us_treasury_rate_contract

# Import schema functions from model packages
from qx_models.esg_factor import get_esg_factors_contract
from qx_models.factor_expected_returns import get_factor_expected_returns_contract
from qx_models.markowitz_portfolio import get_portfolio_weights_contract
from qx_models.two_factor_regression import get_two_factor_betas_contract
from qx_models.universe_filter import get_universe_filter_contract


def seed_registry(reg: DatasetRegistry):
    # Equities OHLCV (multiple exchanges, daily/weekly/monthly frequencies)
    # Register contract once per frequency (exchange is in partitions, not in type)
    for freq in (Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY):
        reg.register(get_tiingo_ohlcv_contract(exchange="US", frequency=freq))

    # Risk-free rates - US Treasury yield curves
    # US: FRED Treasury rates (DGS3MO, DGS1, DGS5, DGS10, DGS30)
    # Region is hardcoded (US), frequency is partitioned (daily/weekly/monthly)
    # Register one contract per frequency for contract identity
    from dataclasses import replace

    for freq in (Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY):
        base_contract = get_us_treasury_rate_contract()
        # Create contract with specific frequency in dataset_type
        updated_type = DatasetType(
            domain=base_contract.dataset_type.domain,
            asset_class=base_contract.dataset_type.asset_class,
            subdomain=base_contract.dataset_type.subdomain,
            region=base_contract.dataset_type.region,  # US (hardcoded)
            frequency=freq,  # Vary by frequency
        )
        contract = replace(base_contract, dataset_type=updated_type)
        reg.register(contract)

    # Derived metrics predictions (generic)
    dt_out = DatasetType(
        Domain.DERIVED_METRICS, AssetClass.EQUITY, "predictions", None, None
    )
    reg.register(
        DatasetContract(
            dataset_type=dt_out,
            schema_version="schema_v1",
            required_columns=(
                "model",
                "model_version",
                "featureset_id",
                "run_id",
                "run_ts",
                "symbol",
                "horizon_d",
                "predicted_return",
                "predicted_price",
                "confidence",
            ),
            partition_keys=("output_type", "model", "run_date"),
            path_template="data/processed/{output_type}/model={model}/run_date={run_date}",
        )
    )

    # Derived metrics continuous_members (membership analysis)
    dt_continuous = DatasetType(
        Domain.DERIVED_METRICS, AssetClass.EQUITY, "continuous_members", None, None
    )
    reg.register(
        DatasetContract(
            dataset_type=dt_continuous,
            schema_version="schema_v1",
            required_columns=(
                "ticker",
                "start_date",
                "end_date",
                "total_snapshots",
                "snapshots_present",
                "continuity_pct",
                "snapshots_missing",
                "first_date",
                "last_date",
                "is_continuous",
                "model",
                "model_version",
                "featureset_id",
                "run_id",
                "run_ts",
            ),
            partition_keys=("output_type", "model", "run_date"),
            path_template="data/processed/{output_type}/model={model}/run_date={run_date}",
        )
    )

    # Universe membership - use schema functions from sp500_membership package
    reg.register(get_sp500_daily_contract())
    reg.register(get_sp500_intervals_contract())

    # GVKEY-ticker mapping (metadata for ESG and other data sources)
    # This maps Global Company Keys (GVKEY) to ticker symbols
    reg.register(get_gvkey_mapping_contract(exchange="US"))

    # ESG scores (annual ESG composite + pillar scores with monthly observations)
    # Source: Local Excel/CSV files (data_matlab_ESG_withSIC.xlsx)
    # Partitioned by exchange (US, HK) and year
    reg.register(get_esg_scores_contract())

    # ESG Factors (ESGFactorModel output)
    # Long-short factor portfolios based on ESG signals (ESG, E, S, G, ESG_mom)
    reg.register(get_esg_factors_contract())

    # Two-Factor Betas (TwoFactorRegressionModel output)
    # Market beta and ESG beta from OLS regression
    reg.register(get_two_factor_betas_contract())

    # Factor Expected Returns (FactorExpectedReturnsModel output)
    # Expected returns using factor model: E[R] = RF + β'λ
    reg.register(get_factor_expected_returns_contract())

    # Portfolio Weights (MarkowitzPortfolioModel output)
    # Optimal portfolio allocations from mean-variance optimization
    reg.register(get_portfolio_weights_contract())

    # Universe Filter (UniverseFilterModel output)
    # Filtered universe with continuous ESG score coverage
    reg.register(get_universe_filter_contract())
