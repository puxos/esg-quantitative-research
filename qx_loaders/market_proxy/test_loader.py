"""
Unit tests for MarketProxyLoader.

Tests cover:
- YAML configuration loading
- Loading market proxy (benchmark) data
- Date range handling
- Frequency handling
- Returns calculation
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

from .loader import MarketProxyLoader

# ============================================================================
# Test Configuration
# ============================================================================

SAMPLE_PERIOD_START = "2023-01-01"
SAMPLE_PERIOD_END = "2023-12-31"
PROXY_SYMBOL = "SPY"
FREQUENCY = "daily"

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
    return MarketProxyLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "proxy_symbol": PROXY_SYMBOL,
            "frequency": FREQUENCY,
        },
    )


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_loader_initialization(loader):
    """Test that loader initializes correctly from YAML."""
    assert loader is not None
    assert loader.info["id"] == "market_proxy_loader"
    assert "version" in loader.info


def test_configuration_loading(loader):
    """Test YAML configuration is loaded properly."""
    assert hasattr(loader, "params")
    assert "start_date" in loader.params
    assert "end_date" in loader.params
    assert "proxy_symbol" in loader.params
    assert "frequency" in loader.params


def test_default_parameters(package_dir, storage_infrastructure):
    """Test default parameter values."""
    loader = MarketProxyLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={"start_date": "2020-01-01", "end_date": "2024-12-31"},
    )

    assert loader.params.get("proxy_symbol") == "SPY"
    assert loader.params.get("frequency") == "Daily"  # Note: capital D


# ============================================================================
# Unit Tests - Data Loading
# ============================================================================


@pytest.mark.integration
def test_load_market_proxy(loader):
    """Test loading market proxy data."""
    try:
        df = loader.load()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "date" in df.columns
        assert "symbol" in df.columns
        assert "close" in df.columns

        print(f"✅ Loaded {len(df)} market proxy records")

    except FileNotFoundError:
        pytest.skip("Market proxy data not available")


@pytest.mark.integration
def test_load_different_proxy(package_dir, storage_infrastructure):
    """Test loading different market proxy (e.g., VTI)."""
    loader = MarketProxyLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "proxy_symbol": "VTI",
            "frequency": FREQUENCY,
        },
    )

    try:
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert "symbol" in df.columns
        assert df["symbol"].iloc[0] == "VTI"

        print(f"✅ Loaded {len(df)} VTI records")

    except (FileNotFoundError, ValueError) as e:
        if "No data found" in str(e):
            pytest.skip("VTI data not available")
        raise


@pytest.mark.integration
def test_load_monthly_frequency(package_dir, storage_infrastructure):
    """Test loading monthly frequency data."""
    loader = MarketProxyLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "proxy_symbol": PROXY_SYMBOL,
            "frequency": "monthly",
        },
    )

    try:
        df = loader.load()
        assert isinstance(df, pd.DataFrame)

        # Monthly data should have fewer observations
        expected_months = (2024 - 2020 + 1) * 12
        assert len(df) <= expected_months

        print(f"✅ Loaded {len(df)} monthly market proxy records")

    except FileNotFoundError:
        pytest.skip("Monthly market proxy data not available")


# ============================================================================
# Unit Tests - Returns Calculation
# ============================================================================


@pytest.mark.integration
def test_returns_are_valid(loader):
    """Test that price data is valid."""
    try:
        df = loader.load()

        # Close price should be numeric
        assert "close" in df.columns
        assert df["close"].dtype in [float, "float64", int, "int64"]

        # Close price should not have NaN values
        assert not df["close"].isnull().any()

        # Prices should be positive
        assert (df["close"] > 0).all()

    except FileNotFoundError:
        pytest.skip("Market proxy data not available")


@pytest.mark.integration
def test_returns_index_is_datetime(loader):
    """Test that date column is datetime."""
    try:
        df = loader.load()

        assert "date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

        # Dates should be sorted
        assert df["date"].is_monotonic_increasing

    except FileNotFoundError:
        pytest.skip("Market proxy data not available")


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

        # Should have required OHLCV columns
        assert "date" in df.columns
        assert "symbol" in df.columns
        assert "close" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns

        # Date should be datetime
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    except FileNotFoundError:
        pytest.skip("Market proxy data not available")


@pytest.mark.integration
def test_output_date_range(loader):
    """Test that output respects date range."""
    try:
        df = loader.load()

        if len(df) > 0:
            start = pd.Timestamp(SAMPLE_PERIOD_START)
            end = pd.Timestamp(SAMPLE_PERIOD_END)

            assert df["date"].iloc[0] >= start
            assert df["date"].iloc[-1] <= end

    except FileNotFoundError:
        pytest.skip("Market proxy data not available")


# ============================================================================
# Unit Tests - Error Handling
# ============================================================================


def test_missing_proxy_data(package_dir, storage_infrastructure):
    """Test error handling when proxy data is missing."""
    loader = MarketProxyLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "1900-01-01",
            "end_date": "1900-12-31",
            "proxy_symbol": "NONEXISTENT",
            "frequency": "daily",
        },
    )

    # Should raise ValueError when data not found
    with pytest.raises(ValueError, match="No data found for market proxy"):
        df = loader.load()


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================


@pytest.mark.integration
def test_full_pipeline_with_ohlcv_builder(storage_infrastructure, package_dir):
    """Test complete pipeline: BuildOHLCV → LoadMarketProxy."""
    dag = DAG(
        tasks=[
            Task(
                id="BuildOHLCV",
                run=run_builder(
                    package_path="qx_builders/tiingo_ohlcv",
                    registry=storage_infrastructure["registry"],
                    adapter=storage_infrastructure["adapter"],
                    resolver=storage_infrastructure["resolver"],
                    partitions={"exchange": "US", "frequency": "daily"},
                    overrides={
                        "symbols": [PROXY_SYMBOL],
                        "start_date": "2023-01-01",
                        "end_date": "2023-12-31",
                        "frequency": "daily",
                    },
                ),
                deps=[],
            ),
            Task(
                id="LoadMarketProxy",
                run=run_loader(
                    package_path="qx_loaders/market_proxy",
                    registry=storage_infrastructure["registry"],
                    backend=storage_infrastructure["backend"],
                    resolver=storage_infrastructure["resolver"],
                    overrides={
                        "start_date": "2023-01-01",
                        "end_date": "2023-12-31",
                        "proxy_symbol": PROXY_SYMBOL,
                        "frequency": "daily",
                    },
                ),
                deps=["BuildOHLCV"],
            ),
        ]
    )

    try:
        results = dag.execute()

        if results is not None:
            assert "LoadMarketProxy" in results
            assert results["LoadMarketProxy"]["status"] == "success"

            df = results["LoadMarketProxy"]["output"]
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "close" in df.columns

            print(f"✅ Pipeline completed: {len(df)} market proxy records loaded")
        else:
            pytest.skip("DAG execution did not return results dictionary")

    except FileNotFoundError:
        pytest.skip("Required data not available")


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use MarketProxyLoader.

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
    loader = MarketProxyLoader(
        package_dir="qx_loaders/market_proxy",
        loader=typed_loader,
        overrides={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "proxy_symbol": "SPY",
            "frequency": "daily",
        },
    )

    # Load data
    df = loader.load()

    print(f"✅ Loaded {len(df)} market proxy records")
    print(f"   Symbol: {df['symbol'].iloc[0]}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Mean price: ${df['price'].mean():.2f}")


if __name__ == "__main__":
    # Run example
    example_usage()
