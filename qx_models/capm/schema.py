from pathlib import Path

from qx.common.contracts import DatasetContract
from qx.common.schema_loader import load_contract

SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


def get_contracts() -> list[DatasetContract]:
    """
    Get the dataset contract for CAPM expected returns output.

    Standard contract discovery function for auto-registration.

    Standard CAPM model calculates expected returns using:
        E[R_i] = RF + β_market × (E[R_market] - RF)

    Returns:
        List containing single DatasetContract for processed/expected-returns/capm data
    """
    return [load_contract(SCHEMA_PATH)]
