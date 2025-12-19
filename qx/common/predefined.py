"""
Predefined Dataset Contracts Registry

Seeds the dataset registry with contracts from all builders and models.

Uses dynamic auto-discovery to avoid manual imports:
- Automatically discovers contracts from qx_builders/ and qx_models/ packages
- Each package's schema.py must have a get_contracts() function
- Function must return list[DatasetContract] (even for single contract)
"""

from qx.common.auto_registration import auto_register_contracts
from qx.common.contracts import DatasetRegistry


def seed_registry(reg: DatasetRegistry, verbose: bool = False) -> dict:
    """
    Seed the registry with all dataset contracts.

    Uses dynamic auto-discovery to register contracts from:
    - qx_builders/ packages (data ingestion builders)
    - qx_models/ packages (analytical models)

    Auto-discovery looks for schema.py files with:
    - get_contracts() → list[DatasetContract] (standard function name)

    Args:
        reg: DatasetRegistry to populate
        verbose: Print registration progress (default: False)

    Returns:
        Dict with registration statistics (builders, models, total)
    """

    # ==============================================================================
    # DYNAMIC AUTO-DISCOVERY
    # ==============================================================================
    # Automatically discover and register contracts from all packages
    # This eliminates the need to manually import each builder/model schema
    if verbose:
        print("=" * 80)
        print("DYNAMIC CONTRACT REGISTRATION")
        print("=" * 80)

    stats = auto_register_contracts(reg, verbose=verbose)

    # ==============================================================================
    # SUMMARY
    # ==============================================================================
    if verbose:
        print("\n" + "=" * 80)
        print(f"✅ Registration Complete")
        print(f"   Total: {stats['total']} contracts")
        print("=" * 80)

    return stats
