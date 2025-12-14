from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


def get_capm_expected_returns_contract() -> DatasetContract:
    """
    Get the dataset contract for CAPM expected returns output.

    Standard CAPM model calculates expected returns using:
        E[R_i] = RF + β_market × (E[R_market] - RF)

    Returns:
        DatasetContract for processed/expected-returns/capm data
    """
    return load_contract(SCHEMA_PATH)
