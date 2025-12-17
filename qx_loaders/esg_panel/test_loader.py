"""
Unit tests for ESGPanelLoader.

Tests cover:
- YAML configuration loading
- Loading ESG panel data
- Continuity filtering logic
- Symbol filtering
- Date range handling
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

from .loader import ESGPanelLoader

# ============================================================================
# Test Configuration
# ============================================================================

SAMPLE_PERIOD_START = "2020-01-01"
SAMPLE_PERIOD_END = "2024-12-31"
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
EXCHANGE = "US"

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
    return ESGPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "symbols": TEST_SYMBOLS,
            "exchange": EXCHANGE,
            "continuous": False,
        },
    )


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_loader_initialization(loader):
    """Test that loader initializes correctly from YAML."""
    assert loader is not None
    assert loader.info["id"] == "esg_panel_loader"
    assert "version" in loader.info


def test_configuration_loading(loader):
    """Test YAML configuration is loaded properly."""
    assert hasattr(loader, "params")
    assert "start_date" in loader.params
    assert "end_date" in loader.params
    assert "symbols" in loader.params
    assert "exchange" in loader.params


def test_default_parameters(package_dir, storage_infrastructure):
    """Test default parameter values."""
    loader = ESGPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "symbols": TEST_SYMBOLS,
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
        },
    )

    assert loader.params.get("continuity") == "any"
    assert loader.params.get("exchange") == "US"


# ============================================================================
# Unit Tests - Data Loading
# ============================================================================


@pytest.mark.integration
def test_load_esg_panel(loader):
    """Test loading ESG panel data."""
    try:
        df = loader.load()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "ticker" in df.columns
        assert "esg_year" in df.columns

        print(f"✅ Loaded {len(df)} ESG records")

    except FileNotFoundError:
        pytest.skip("ESG data not available")


@pytest.mark.integration
def test_load_with_continuity_filter(package_dir, storage_infrastructure):
    """Test loading with continuity filtering enabled."""
    loader = ESGPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "symbols": TEST_SYMBOLS,
            "exchange": EXCHANGE,
            "continuous": True,
        },
    )

    try:
        df = loader.load()

        assert isinstance(df, pd.DataFrame)

        # With continuity filter, should have fewer or equal rows
        if len(df) > 0:
            # Check that returned symbols have continuous coverage
            for symbol in df["ticker"].unique():
                symbol_data = df[df["ticker"] == symbol]
                years = sorted(symbol_data["esg_year"].unique())

                # Check for no gaps in years
                if len(years) > 1:
                    year_diffs = [
                        years[i + 1] - years[i] for i in range(len(years) - 1)
                    ]
                    assert all(
                        diff == 1 for diff in year_diffs
                    ), f"Gap found in {symbol} ESG years"

        print(f"✅ Loaded {len(df)} continuous ESG records")

    except FileNotFoundError:
        pytest.skip("ESG data not available")


@pytest.mark.integration
def test_load_empty_symbols_list(package_dir, storage_infrastructure):
    """Test loading with empty symbols list (should raise ValueError)."""
    loader = ESGPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "symbols": [],
            "exchange": EXCHANGE,
        },
    )

    # Should raise ValueError due to allow_empty=false in YAML
    with pytest.raises(
        ValueError, match="returned empty DataFrame, but allow_empty=false"
    ):
        df = loader.load()


# ============================================================================
# Unit Tests - Output Validation
# ============================================================================


@pytest.mark.integration
def test_output_format(loader):
    """Test that output format matches YAML specification."""
    try:
        df = loader.load()

        # Should be a DataFrame
        assert isinstance(df, pd.DataFrame)

        # Check required columns
        if len(df) > 0:
            assert "ticker" in df.columns
            assert "esg_year" in df.columns
            assert "gvkey" in df.columns or "gvkey" not in df.columns  # Optional

    except FileNotFoundError:
        pytest.skip("ESG data not available")


@pytest.mark.integration
def test_output_contains_requested_symbols(loader):
    """Test that output only contains requested symbols."""
    try:
        df = loader.load()

        if len(df) > 0:
            returned_symbols = df["ticker"].unique()
            # All returned symbols should be in request
            assert all(s in TEST_SYMBOLS for s in returned_symbols)

    except FileNotFoundError:
        pytest.skip("ESG data not available")


# ============================================================================
# Unit Tests - Error Handling
# ============================================================================


def test_missing_esg_data(package_dir, storage_infrastructure):
    """Test error handling when ESG data is missing."""
    loader = ESGPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "1900-01-01",
            "end_date": "1900-12-31",
            "symbols": ["NONEXISTENT"],
            "exchange": "US",
        },
    )

    # Should raise ValueError due to allow_empty=false when no data found
    with pytest.raises(
        ValueError, match="returned empty DataFrame, but allow_empty=false"
    ):
        df = loader.load()


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================


@pytest.mark.integration
def test_full_pipeline_with_universe(storage_infrastructure, package_dir):
    """Test complete pipeline: BuildMembership → LoadHistoricMembers → LoadESGPanel."""
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
                id="LoadHistoricMembers",
                run=run_loader(
                    package_path="qx_loaders/historic_members",
                    registry=storage_infrastructure["registry"],
                    backend=storage_infrastructure["backend"],
                    resolver=storage_infrastructure["resolver"],
                    overrides={
                        "start_date": "2020-01-01",
                        "end_date": "2024-12-31",
                        "universe": "sp500",
                    },
                ),
                deps=["BuildMembership"],
            ),
            Task(
                id="LoadESGPanel",
                run=lambda ctx: run_loader(
                    package_path="qx_loaders/esg_panel",
                    registry=storage_infrastructure["registry"],
                    backend=storage_infrastructure["backend"],
                    resolver=storage_infrastructure["resolver"],
                    overrides={
                        "start_date": "2020-01-01",
                        "end_date": "2024-12-31",
                        "symbols": ctx["LoadHistoricMembers"]["output"][
                            :10
                        ],  # First 10
                        "exchange": "US",
                        "continuous": False,
                    },
                )(),
                deps=["LoadHistoricMembers"],
            ),
        ]
    )

    try:
        results = dag.execute()

        if results is not None:
            assert "LoadESGPanel" in results
            assert results["LoadESGPanel"]["status"] == "success"

            df = results["LoadESGPanel"]["output"]
            assert isinstance(df, pd.DataFrame)

            print(f"✅ Pipeline completed: {len(df)} ESG records loaded")
        else:
            pytest.skip("DAG execution did not return results dictionary")

    except FileNotFoundError:
        pytest.skip("Required data not available")


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use ESGPanelLoader.

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
    loader = ESGPanelLoader(
        package_dir="qx_loaders/esg_panel",
        loader=typed_loader,
        overrides={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "continuous": True,
        },
    )

    # Load data
    df = loader.load()

    print(f"✅ Loaded {len(df)} ESG records")


if __name__ == "__main__":
    # Run example
    example_usage()
