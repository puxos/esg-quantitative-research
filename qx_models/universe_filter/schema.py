"""
Schema definition for Universe Filter output
"""

from qx.common.contracts import DatasetContract
from qx.common.types import AssetClass, DatasetType, Domain


def get_universe_filter_contract() -> DatasetContract:
    """
    Schema contract for universe filter output.

    Output format:
        - One row per ticker in input universe
        - Contains coverage metrics and filter decision
        - Used to identify tickers with continuous ESG score coverage

    Columns:
        Identifiers:
            - ticker: Stock ticker symbol
            - universe: Source universe (e.g., "sp500")

        Period:
            - start_date: Research period start
            - end_date: Research period end
            - trading_days: Number of trading days in period
            - expected_esg_points: Expected ESG publication points (monthly/quarterly)

        Coverage Metrics:
            - esg_count: Actual ESG score observations
            - esg_coverage_pct: ESG coverage percentage (0-1)
            - max_esg_gap_days: Maximum gap between ESG observations (days)
            - first_esg_date: First ESG score date
            - last_esg_date: Last ESG score date

        Filter Decision:
            - passed_filter: Boolean, whether ticker passed filter
            - filter_reason: Reason for pass/fail

        Filter Parameters (for reproducibility):
            - min_coverage_pct: Minimum coverage threshold used
            - max_gap_days: Maximum gap threshold used
            - require_continuous: Whether continuous requirement applied

        Model Metadata:
            - model: Model ID ("universe_filter")
            - model_version: Model version
            - run_id: Unique run identifier
            - run_ts: Run timestamp (ISO 8601)

    Partition Structure:
        data/processed/universe_filter/model={model}/run_date={run_date}/

        Example:
            data/processed/universe_filter/model=universe_filter/run_date=2024-12-04/part-<run_id>.parquet

    Returns:
        DatasetContract for universe filter output
    """
    dt = DatasetType(
        domain=Domain.DERIVED_METRICS,
        asset_class=AssetClass.EQUITY,
        subdomain="universe_filter",
        region=None,
        frequency=None,
    )

    return DatasetContract(
        dataset_type=dt,
        schema_version="schema_v1",
        required_columns=(
            # Identifiers
            "ticker",
            "universe",
            # Period
            "start_date",
            "end_date",
            "trading_days",
            "expected_esg_points",
            # Coverage metrics
            "esg_count",
            "esg_coverage_pct",
            "max_esg_gap_days",
            "first_esg_date",
            "last_esg_date",
            # Filter decision
            "passed_filter",
            "filter_reason",
            # Filter parameters
            "min_coverage_pct",
            "max_gap_days",
            "require_continuous",
            # Model metadata
            "model",
            "model_version",
            "run_id",
            "run_ts",
        ),
        partition_keys=("output_type", "model", "run_date"),
        path_template="data/processed/{output_type}/model={model}/run_date={run_date}",
    )
