"""
Unit tests for UniverseAtDateLoader.

Tests cover:
- YAML configuration loading
- Loading membership data
- Point-in-time member selection
- Output validation
- Error handling

This test file is co-located with the loader source code for easier maintenance.
"""

from pathlib import Path

import pandas as pd
import pytest

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.foundation.typed_curated_loader import TypedCuratedLoader
from qx.orchestration.dag import DAG, Task
from qx.orchestration.factories import run_builder, run_loader
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.curated_writer import CuratedWriter
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter

from .loader import UniverseAtDateLoader

# ============================================================================
# Test Configuration
# ============================================================================

QUERY_DATE = "2024-01-01"
UNIVERSE = "sp500"

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def storage_infrastructure():
    """Setup storage infrastructure for tests."""
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()
    writer = CuratedWriter(
        backend=backend, adapter=adapter, resolver=resolver, registry=registry
    )
    typed_loader = TypedCuratedLoader(
        backend=backend, registry=registry, resolver=resolver
    )

    return {
        "registry": registry,
        "backend": backend,
        "adapter": adapter,
        "resolver": resolver,
        "writer": writer,
        "loader": typed_loader,
    }


@pytest.fixture
def package_dir():
    """Get package directory path."""
    return str(Path(__file__).parent)


@pytest.fixture
def loader(package_dir, storage_infrastructure):
    """Create loader instance."""
    return UniverseAtDateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={"date": QUERY_DATE, "universe": UNIVERSE},
    )


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_loader_initialization(loader):
    """Test that loader initializes correctly from YAML."""
    assert loader is not None
    assert loader.info["id"] == "universe_at_date_loader"
    assert "version" in loader.info


def test_configuration_loading(loader):
    """Test YAML configuration is loaded properly."""
    assert hasattr(loader, "params")
    assert "date" in loader.params
    assert "universe" in loader.params
    assert loader.params["date"] == QUERY_DATE


def test_default_parameters(package_dir, storage_infrastructure):
    """Test default parameter values."""
    loader = UniverseAtDateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={"date": QUERY_DATE},
    )

    assert loader.params.get("universe") == "sp500"


# ============================================================================
# Unit Tests - Data Loading
# ============================================================================


@pytest.mark.integration
def test_load_universe_at_date(loader):
    """Test loading universe members at specific date."""
    try:
        symbols = loader.load()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(s, str) for s in symbols)

        print(f"✅ Loaded {len(symbols)} members at {QUERY_DATE}")

    except FileNotFoundError:
        pytest.skip("Membership data not available")


@pytest.mark.integration
def test_load_different_dates(package_dir, storage_infrastructure):
    """Test loading universe at different dates."""
    dates = ["2020-01-01", "2022-06-15", "2024-12-31"]

    for date in dates:
        loader = UniverseAtDateLoader(
            package_dir=package_dir,
            loader=storage_infrastructure["loader"],
            overrides={"date": date, "universe": UNIVERSE},
        )

        try:
            symbols = loader.load()
            assert isinstance(symbols, list)
            print(f"✅ {date}: {len(symbols)} members")

        except FileNotFoundError:
            pytest.skip(f"Membership data not available for {date}")


@pytest.mark.integration
def test_current_date_membership(package_dir, storage_infrastructure):
    """Test loading universe membership at recent date within data range."""
    # Use a date within the available data range (data goes to 2025-11-11)
    test_date = "2025-11-01"

    loader = UniverseAtDateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={"date": test_date, "universe": UNIVERSE},
    )

    try:
        symbols = loader.load()
        assert isinstance(symbols, list)
        # SP500 should have ~500 members
        assert 400 <= len(symbols) <= 600

        print(f"✅ Members on {test_date}: {len(symbols)}")

    except FileNotFoundError:
        pytest.skip("Membership data not available")


# ============================================================================
# Unit Tests - Output Validation
# ============================================================================


@pytest.mark.integration
def test_output_format(loader):
    """Test that output format matches YAML specification."""
    try:
        symbols = loader.load()

        # Should be a list of strings
        assert isinstance(symbols, list)
        assert all(isinstance(s, str) for s in symbols)

        # Symbols should be valid tickers
        for symbol in symbols[:10]:
            assert symbol.isupper() or "." in symbol or "-" in symbol

    except FileNotFoundError:
        pytest.skip("Membership data not available")


@pytest.mark.integration
def test_no_duplicates(loader):
    """Test that output contains no duplicate symbols."""
    try:
        symbols = loader.load()
        assert len(symbols) == len(set(symbols))

    except FileNotFoundError:
        pytest.skip("Membership data not available")


# ============================================================================
# Unit Tests - Error Handling
# ============================================================================


def test_missing_membership_data(package_dir, storage_infrastructure):
    """Test error handling when membership data is missing."""
    loader = UniverseAtDateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={"date": "1900-01-01", "universe": "nonexistent_universe"},
    )

    # Should return empty list when data not found
    symbols = loader.load()
    assert isinstance(symbols, list)
    assert len(symbols) == 0


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================


@pytest.mark.integration
def test_full_pipeline(storage_infrastructure, package_dir):
    """Test complete pipeline: BuildMembership → LoadUniverseAtDate."""
    dag = DAG(
        tasks=[
            Task(
                id="BuildMembership",
                run=run_builder(
                    package_path="qx_builders/sp500_membership",
                    registry=storage_infrastructure["registry"],
                    adapter=storage_infrastructure["adapter"],
                    resolver=storage_infrastructure["resolver"],
                    partitions={"universe": "sp500", "mode": "intervals"},
                    overrides={"min_date": "2000-01-01"},
                ),
                deps=[],
            ),
            Task(
                id="LoadUniverseAtDate",
                run=run_loader(
                    package_path="qx_loaders/universe_at_date",
                    registry=storage_infrastructure["registry"],
                    backend=storage_infrastructure["backend"],
                    resolver=storage_infrastructure["resolver"],
                    overrides={"date": QUERY_DATE, "universe": UNIVERSE},
                ),
                deps=["BuildMembership"],
            ),
        ]
    )

    results = dag.execute()

    if results is not None:
        assert "LoadUniverseAtDate" in results
        assert results["LoadUniverseAtDate"]["status"] == "success"

        symbols = results["LoadUniverseAtDate"]["output"]
        assert isinstance(symbols, list)
        assert len(symbols) > 0
    else:
        pytest.skip("DAG execution did not return results dictionary")

    print(f"✅ Pipeline completed: {len(symbols)} members at {QUERY_DATE}")


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use UniverseAtDateLoader.

    This is not a test, but serves as documentation.
    """
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.foundation.typed_curated_loader import TypedCuratedLoader
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.pathing import PathResolver

    # Setup infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()
    typed_loader = TypedCuratedLoader(
        backend=backend, registry=registry, resolver=resolver
    )

    # Create loader
    loader = UniverseAtDateLoader(
        package_dir="qx_loaders/universe_at_date",
        loader=typed_loader,
        overrides={"date": "2024-01-01", "universe": "sp500"},
    )

    # Load data
    symbols = loader.load()

    print(f"✅ Loaded {len(symbols)} members at 2024-01-01")


if __name__ == "__main__":
    # Run example
    example_usage()
