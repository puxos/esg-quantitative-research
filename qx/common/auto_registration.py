"""
Dynamic Contract Registration

Automatically discovers and registers dataset contracts from builder, loader, and model packages
without requiring manual imports in core qx code.

Convention:
- Each package must have a schema.py file with contract getter functions
- Function names must follow pattern: get_*_contract() or get_*_contracts()
- Functions must return DatasetContract or list[DatasetContract]
"""

import importlib
import inspect
from pathlib import Path
from typing import Callable, List, Optional

from qx.common.contracts import DatasetContract, DatasetRegistry


def discover_contract_functions(
    package_path: Path, package_name: str
) -> List[Callable]:
    """
    Discover contract getter functions from packages in a directory.

    Args:
        package_path: Path to package directory (e.g., qx_builders/)
        package_name: Python package name (e.g., "qx_builders")

    Returns:
        List of contract getter functions found

    Convention:
        - Looks for schema.py in each subdirectory
        - Imports functions matching get_*_contract() or get_*_contracts()
        - Skips __pycache__, __init__.py, and non-directory items
    """
    if not package_path.exists():
        return []

    contract_functions = []

    for subdir in package_path.iterdir():
        # Skip non-directories and special directories
        if not subdir.is_dir() or subdir.name.startswith(("__", ".")):
            continue

        # Look for schema.py
        schema_file = subdir / "schema.py"
        if not schema_file.exists():
            continue

        # Import the schema module
        module_name = f"{package_name}.{subdir.name}.schema"
        try:
            module = importlib.import_module(module_name)

            # Find contract getter functions
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # Match: get_*_contract or get_*_contracts
                if name.startswith("get_") and name.endswith(
                    ("_contract", "_contracts")
                ):
                    contract_functions.append(obj)

        except Exception as e:
            # Log error but continue (don't fail entire registration)
            print(f"⚠️  Warning: Failed to load {module_name}: {e}")
            continue

    return contract_functions


def register_contracts_from_functions(
    reg: DatasetRegistry, functions: List[Callable], verbose: bool = False
) -> int:
    """
    Call contract getter functions and register returned contracts.

    Args:
        reg: DatasetRegistry to register contracts into
        functions: List of contract getter functions
        verbose: Print registration progress

    Returns:
        Number of contracts registered

    Handles:
        - Functions that return single DatasetContract
        - Functions that return list[DatasetContract]
        - Functions that require parameters (calls with no args, skips if fails)
    """
    registered_count = 0

    for func in functions:
        try:
            # Try calling with no arguments first
            result = func()

            # Handle single contract
            if isinstance(result, DatasetContract):
                reg.register(result)
                registered_count += 1
                if verbose:
                    print(f"  ✓ {func.__module__}.{func.__name__}()")

            # Handle list of contracts
            elif isinstance(result, list):
                for contract in result:
                    if isinstance(contract, DatasetContract):
                        reg.register(contract)
                        registered_count += 1
                if verbose and result:
                    print(
                        f"  ✓ {func.__module__}.{func.__name__}() → {len(result)} contracts"
                    )

        except TypeError as e:
            # Function requires parameters - skip for now
            # These need to be registered manually or with explicit parameters
            if verbose:
                print(f"  ⊘ {func.__module__}.{func.__name__}() - requires parameters")
            continue

        except Exception as e:
            # Log error but continue
            if verbose:
                print(f"  ✗ {func.__module__}.{func.__name__}() - error: {e}")
            continue

    return registered_count


def auto_register_contracts(
    reg: DatasetRegistry,
    builder_packages: Optional[List[str]] = None,
    model_packages: Optional[List[str]] = None,
    verbose: bool = False,
) -> dict:
    """
    Auto-discover and register contracts from builder and model packages.

    Args:
        reg: DatasetRegistry to populate
        builder_packages: List of builder package names (default: ["qx_builders"])
        model_packages: List of model package names (default: ["qx_models"])
        verbose: Print discovery and registration progress

    Returns:
        Dict with registration statistics:
        {
            "builders": {"discovered": int, "registered": int},
            "models": {"discovered": int, "registered": int},
            "total": int
        }

    Example:
        registry = DatasetRegistry()
        stats = auto_register_contracts(registry, verbose=True)
        print(f"Registered {stats['total']} contracts")
    """
    if builder_packages is None:
        builder_packages = ["qx_builders"]
    if model_packages is None:
        model_packages = ["qx_models"]

    stats = {"builders": {}, "models": {}, "total": 0}

    # Discover and register builder contracts
    if verbose:
        print("📦 Discovering builder contracts...")

    builder_functions = []
    for pkg_name in builder_packages:
        # Get package path (assuming packages are siblings to qx/)
        import qx

        qx_path = Path(qx.__file__).parent.parent
        pkg_path = qx_path / pkg_name

        functions = discover_contract_functions(pkg_path, pkg_name)
        builder_functions.extend(functions)

    stats["builders"]["discovered"] = len(builder_functions)
    stats["builders"]["registered"] = register_contracts_from_functions(
        reg, builder_functions, verbose
    )

    # Discover and register model contracts
    if verbose:
        print("\n🧮 Discovering model contracts...")

    model_functions = []
    for pkg_name in model_packages:
        import qx

        qx_path = Path(qx.__file__).parent.parent
        pkg_path = qx_path / pkg_name

        functions = discover_contract_functions(pkg_path, pkg_name)
        model_functions.extend(functions)

    stats["models"]["discovered"] = len(model_functions)
    stats["models"]["registered"] = register_contracts_from_functions(
        reg, model_functions, verbose
    )

    stats["total"] = stats["builders"]["registered"] + stats["models"]["registered"]

    if verbose:
        print(f"\n✅ Total: {stats['total']} contracts registered")

    return stats
