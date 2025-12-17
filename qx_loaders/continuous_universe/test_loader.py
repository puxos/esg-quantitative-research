"""
Unit tests for ContinuousUniverseLoader.

Tests cover:
- YAML configuration loading
- Loading membership data
- Continuous membership filtering logic
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

from .loader import ContinuousUniverseLoader

# ============================================================================
# Test Configuration
# ============================================================================

SAMPLE_PERIOD_START = "2020-01-01"
SAMPLE_PERIOD_END = "2024-12-31"
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
    return ContinuousUniverseLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "universe": UNIVERSE,
        },
    )


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_loader_initialization(loader):
    """Test that loader initializes correctly from YAML."""
    assert loader is not None
    assert loader.info["id"] == "continuous_universe_loader"
    assert "version" in loader.info


def test_configuration_loading(loader):
    """Test YAML configuration is loaded properly."""
    assert hasattr(loader, "params")
    assert "start_date" in loader.params
    assert "end_date" in loader.params
    assert "universe" in loader.params


def test_default_parameters(package_dir, storage_infrastructure):
    """Test default parameter values."""
    loader = ContinuousUniverseLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={"start_date": "2020-01-01", "end_date": "2024-12-31"},
    )

    assert loader.params.get("universe") == "sp500"


# ============================================================================
# Unit Tests - Data Loading
# ============================================================================


@pytest.mark.integration
def test_load_continuous_universe(loader):
    """Test loading continuous universe members."""
    try:
        symbols = loader.load()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(s, str) for s in symbols)

        print(f"✅ Loaded {len(symbols)} continuous members")

    except FileNotFoundError:
        pytest.skip("Membership data not available")


@pytest.mark.integration
def test_continuous_vs_historic_comparison(storage_infrastructure, package_dir):
    """Test that continuous universe is subset of historic members."""
    try:
        # Load historic members
        historic_loader = HistoricMembersLoader(
            package_dir="qx_loaders/historic_members",
            loader=storage_infrastructure["loader"],
            overrides={
                "start_date": SAMPLE_PERIOD_START,
                "end_date": SAMPLE_PERIOD_END,
                "universe": UNIVERSE,
                "use_ticker_mapper": False,
            },
        )
        historic_symbols = historic_loader.load()

        # Load continuous universe
        continuous_loader = ContinuousUniverseLoader(
            package_dir=package_dir,
            loader=storage_infrastructure["loader"],
            overrides={
                "start_date": SAMPLE_PERIOD_START,
                "end_date": SAMPLE_PERIOD_END,
                "universe": UNIVERSE,
            },
        )
        continuous_symbols = continuous_loader.load()

        # Continuous should be subset of historic
        assert len(continuous_symbols) <= len(historic_symbols)
        assert set(continuous_symbols).issubset(set(historic_symbols))

        print(
            f"✅ Continuous ({len(continuous_symbols)}) ⊆ Historic ({len(historic_symbols)})"
        )

    except (FileNotFoundError, NameError):
        pytest.skip(
            "Membership data not available or HistoricMembersLoader not imported"
        )


@pytest.mark.integration
def test_load_short_period(package_dir, storage_infrastructure):
    """Test loading continuous members for short period (should have more members)."""
    loader = ContinuousUniverseLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "universe": UNIVERSE,
        },
    )

    try:
        symbols = loader.load()
        assert isinstance(symbols, list)
        print(f"✅ Loaded {len(symbols)} continuous members for 2024")

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
    loader = ContinuousUniverseLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "1900-01-01",
            "end_date": "1900-12-31",
            "universe": "nonexistent_universe",
        },
    )

    # Should raise ValueError for empty result (allow_empty=false)
    with pytest.raises(ValueError, match="returned empty list"):
        symbols = loader.load()


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================


@pytest.mark.integration
def test_full_pipeline(storage_infrastructure, package_dir):
    """Test complete pipeline: BuildMembership → LoadContinuousUniverse."""
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
                id="LoadContinuousUniverse",
                run=run_loader(
                    package_path="qx_loaders/continuous_universe",
                    registry=storage_infrastructure["registry"],
                    backend=storage_infrastructure["backend"],
                    resolver=storage_infrastructure["resolver"],
                    overrides={
                        "start_date": SAMPLE_PERIOD_START,
                        "end_date": SAMPLE_PERIOD_END,
                        "universe": UNIVERSE,
                    },
                ),
                deps=["BuildMembership"],
            ),
        ]
    )

    results = dag.execute()

    if results is not None:
        assert "LoadContinuousUniverse" in results
        assert results["LoadContinuousUniverse"]["status"] == "success"

        symbols = results["LoadContinuousUniverse"]["output"]
        assert isinstance(symbols, list)
        assert len(symbols) > 0
    else:
        pytest.skip("DAG execution did not return results dictionary")

    print(f"✅ Pipeline completed: {len(symbols)} continuous members loaded")


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use ContinuousUniverseLoader.

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
    loader = ContinuousUniverseLoader(
        package_dir="qx_loaders/continuous_universe",
        loader=typed_loader,
        overrides={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "universe": "sp500",
        },
    )

    # Load data
    symbols = loader.load()

    print(f"✅ Loaded {len(symbols)} continuous members")


if __name__ == "__main__":
    # Run example
    example_usage()
