"""
Unit tests for OHLCVPanelLoader.

Tests cover:
- YAML configuration loading
- Loading OHLCV price data
- Symbol filtering
- Date range handling
- Volume filtering
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

from .loader import OHLCVPanelLoader

# ============================================================================
# Test Configuration
# ============================================================================

SAMPLE_PERIOD_START = "2023-01-01"
SAMPLE_PERIOD_END = "2023-12-31"
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
FREQUENCY = "daily"
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
    return OHLCVPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "symbols": TEST_SYMBOLS,
            "frequency": FREQUENCY,
            "exchange": EXCHANGE,
            "require_volume": False,
        },
    )


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_loader_initialization(loader):
    """Test that loader initializes correctly from YAML."""
    assert loader is not None
    assert loader.info["id"] == "ohlcv_panel_loader"
    assert "version" in loader.info


def test_configuration_loading(loader):
    """Test YAML configuration is loaded properly."""
    assert hasattr(loader, "params")
    assert "start_date" in loader.params
    assert "end_date" in loader.params
    assert "symbols" in loader.params
    assert "frequency" in loader.params


def test_default_parameters(package_dir, storage_infrastructure):
    """Test default parameter values."""
    loader = OHLCVPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "symbols": TEST_SYMBOLS,
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
        },
    )

    assert loader.params.get("frequency") == "daily"
    assert loader.params.get("exchange") == "US"
    assert loader.params.get("require_volume") is False


# ============================================================================
# Unit Tests - Data Loading
# ============================================================================


@pytest.mark.integration
def test_load_ohlcv_panel(package_dir, storage_infrastructure):
    """Test loading OHLCV panel data for SPY which exists in daily data."""
    # Use SPY which exists in daily data
    loader = OHLCVPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "symbols": ["SPY"],  # Use symbol that exists in daily data
            "frequency": "daily",
        },
    )

    try:
        df = loader.load()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symbol" in df.columns
        assert "date" in df.columns
        assert "close" in df.columns

        print(f"✅ Loaded {len(df)} OHLCV records for SPY")

    except FileNotFoundError:
        pytest.skip("OHLCV data not available")


@pytest.mark.integration
def test_load_with_volume_filter(package_dir, storage_infrastructure):
    """Test loading with volume filtering enabled."""
    loader = OHLCVPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "symbols": TEST_SYMBOLS,
            "frequency": FREQUENCY,
            "exchange": EXCHANGE,
            "require_volume": True,
        },
    )

    try:
        df = loader.load()

        assert isinstance(df, pd.DataFrame)

        # With volume filter, all records should have volume > 0
        if len(df) > 0 and "volume" in df.columns:
            assert (df["volume"] > 0).all()

        print(f"✅ Loaded {len(df)} tradeable OHLCV records")

    except FileNotFoundError:
        pytest.skip("OHLCV data not available")


@pytest.mark.integration
def test_load_empty_symbols_list(package_dir, storage_infrastructure):
    """Test loading with empty symbols list."""
    loader = OHLCVPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "symbols": [],
            "frequency": FREQUENCY,
            "exchange": EXCHANGE,
        },
    )

    df = loader.load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@pytest.mark.integration
def test_load_monthly_frequency(package_dir, storage_infrastructure):
    """Test loading monthly frequency data."""
    loader = OHLCVPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "symbols": TEST_SYMBOLS,
            "frequency": "monthly",
            "exchange": EXCHANGE,
        },
    )

    try:
        df = loader.load()
        assert isinstance(df, pd.DataFrame)

        if len(df) > 0:
            # Monthly data should have fewer records than daily
            print(f"✅ Loaded {len(df)} monthly OHLCV records")

    except FileNotFoundError:
        pytest.skip("Monthly OHLCV data not available")


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
            assert "symbol" in df.columns
            assert "date" in df.columns
            assert "close" in df.columns

    except FileNotFoundError:
        pytest.skip("OHLCV data not available")


@pytest.mark.integration
def test_output_contains_requested_symbols(loader):
    """Test that output only contains requested symbols."""
    try:
        df = loader.load()

        if len(df) > 0:
            returned_symbols = df["symbol"].unique()
            # All returned symbols should be in request
            assert all(s in TEST_SYMBOLS for s in returned_symbols)

    except FileNotFoundError:
        pytest.skip("OHLCV data not available")


@pytest.mark.integration
def test_output_date_range(loader):
    """Test that output respects date range."""
    try:
        df = loader.load()

        if len(df) > 0:
            start = pd.Timestamp(SAMPLE_PERIOD_START)
            end = pd.Timestamp(SAMPLE_PERIOD_END)

            assert (df["date"] >= start).all()
            assert (df["date"] <= end).all()

    except FileNotFoundError:
        pytest.skip("OHLCV data not available")


# ============================================================================
# Unit Tests - Error Handling
# ============================================================================


def test_missing_ohlcv_data(package_dir, storage_infrastructure):
    """Test error handling when OHLCV data is missing."""
    loader = OHLCVPanelLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "1900-01-01",
            "end_date": "1900-12-31",
            "symbols": ["NONEXISTENT"],
            "frequency": "daily",
            "exchange": "US",
        },
    )

    # Should return empty DataFrame when data not found
    df = loader.load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================


@pytest.mark.integration
def test_full_pipeline_with_universe(storage_infrastructure, package_dir):
    """Test complete pipeline: BuildMembership → LoadHistoricMembers → BuildOHLCV → LoadOHLCVPanel."""
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
                        "start_date": "2023-01-01",
                        "end_date": "2023-12-31",
                        "universe": "sp500",
                    },
                ),
                deps=["BuildMembership"],
            ),
            Task(
                id="LoadOHLCVPanel",
                run=lambda ctx: run_loader(
                    package_path="qx_loaders/ohlcv_panel",
                    registry=storage_infrastructure["registry"],
                    backend=storage_infrastructure["backend"],
                    resolver=storage_infrastructure["resolver"],
                    overrides={
                        "start_date": "2023-01-01",
                        "end_date": "2023-12-31",
                        "symbols": ctx["LoadHistoricMembers"]["output"][:5],  # First 5
                        "frequency": "daily",
                        "exchange": "US",
                    },
                )(),
                deps=["LoadHistoricMembers"],
            ),
        ]
    )

    try:
        results = dag.execute()

        if results is not None:
            assert "LoadOHLCVPanel" in results
            assert results["LoadOHLCVPanel"]["status"] == "success"

            df = results["LoadOHLCVPanel"]["output"]
            assert isinstance(df, pd.DataFrame)

            print(f"✅ Pipeline completed: {len(df)} OHLCV records loaded")
        else:
            pytest.skip("DAG execution did not return results dictionary")

    except FileNotFoundError:
        pytest.skip("Required data not available")


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use OHLCVPanelLoader.

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
    loader = OHLCVPanelLoader(
        package_dir="qx_loaders/ohlcv_panel",
        loader=typed_loader,
        overrides={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "frequency": "daily",
        },
    )

    # Load data
    df = loader.load()

    print(f"✅ Loaded {len(df)} OHLCV records")


if __name__ == "__main__":
    # Run example
    example_usage()
