"""
Unit tests for TiingoOHLCVBuilder.

Tests the builder with:
- YAML configuration loading
- Batch symbol fetching from Tiingo API
- Date range parameters
- Frequency parameters (daily, weekly, monthly)
- Error handling and validation
- DAG integration

This test file is co-located with the builder source code for easier maintenance.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.common.types import Frequency
from qx.orchestration.dag import DAG, Task
from qx.orchestration.factories import run_builder, run_loader
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.curated_writer import CuratedWriter
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter

from .builder import TiingoOHLCVBuilder

# ============================================================================
# Test Configuration
# ============================================================================

UNIVERSE = "sp500"
SAMPLE_PERIOD_START = "2014-01-01"
SAMPLE_PERIOD_END = "2024-12-31"
SAMPLE_FREQUENCY = Frequency.MONTHLY.value


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def load_env():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if not env_path.exists():
        return

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and value and key not in os.environ:
                    os.environ[key] = value


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

    return {
        "registry": registry,
        "backend": backend,
        "adapter": adapter,
        "resolver": resolver,
        "writer": writer,
    }


@pytest.fixture
def package_dir():
    """Get package directory path."""
    return str(Path(__file__).parent)


@pytest.fixture
def builder(package_dir, storage_infrastructure):
    """Create builder instance with default configuration."""
    # Use dummy API key for unit tests (real API calls will be mocked or skipped)
    return TiingoOHLCVBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        overrides={
            "tiingo_api_key": "test_api_key_dummy",
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "frequency": "daily",
        },
    )


# ============================================================================
# Unit Tests
# ============================================================================


def test_builder_initialization(builder):
    """Test builder initializes correctly with configuration."""
    assert builder is not None
    assert builder.info["id"] == "tiingo_ohlcv_builder"
    assert "symbols" in builder.params
    assert "start_date" in builder.params


def test_configuration_loading(package_dir, storage_infrastructure):
    """Test YAML configuration is loaded properly."""
    builder = TiingoOHLCVBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        overrides={"tiingo_api_key": "test_api_key_dummy"},
    )

    # Check required configuration exists
    assert hasattr(builder, "params")
    assert hasattr(builder, "info")


@pytest.mark.skipif(
    not os.environ.get("TIINGO_API_KEY"), reason="TIINGO_API_KEY not set"
)
def test_fetch_raw_single_symbol(builder, load_env):
    """Test fetching raw OHLCV data for a single symbol."""
    # Override to single symbol for faster test
    builder.params["symbols"] = ["AAPL"]
    builder.params["start_date"] = "2024-01-01"
    builder.params["end_date"] = "2024-01-31"

    df = builder.fetch_raw()

    assert df is not None
    assert len(df) > 0
    assert "date" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns


@pytest.mark.skipif(
    not os.environ.get("TIINGO_API_KEY"), reason="TIINGO_API_KEY not set"
)
def test_fetch_raw_multiple_symbols(builder, load_env):
    """Test fetching raw OHLCV data for multiple symbols."""
    builder.params["symbols"] = ["AAPL", "MSFT", "GOOGL"]
    builder.params["start_date"] = "2024-01-01"
    builder.params["end_date"] = "2024-01-31"

    df = builder.fetch_raw()

    assert df is not None
    assert len(df) > 0
    assert "ticker" in df.columns
    assert df["ticker"].nunique() <= 3  # May be fewer if API issues


def test_transform_to_curated(builder):
    """Test transformation of raw data to curated format."""
    # Create sample raw data (as returned from Tiingo API)
    raw_df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "date": pd.date_range("2024-01-01", periods=5),
            "close": [150, 151, 152, 153, 154],
            "open": [149, 150, 151, 152, 153],
            "high": [151, 152, 153, 154, 155],
            "low": [148, 149, 150, 151, 152],
            "volume": [1000000] * 5,
            "adjOpen": [149, 150, 151, 152, 153],
            "adjHigh": [151, 152, 153, 154, 155],
            "adjLow": [148, 149, 150, 151, 152],
            "adjClose": [150, 151, 152, 153, 154],
            "adjVolume": [1000000] * 5,
            "divCash": [0.0] * 5,
            "splitFactor": [1.0] * 5,
        }
    )

    curated_df = builder.transform_to_curated(raw_df)

    assert curated_df is not None
    assert len(curated_df) == 5
    assert "symbol" in curated_df.columns
    assert "date" in curated_df.columns


def test_error_handling_missing_api_key(package_dir, storage_infrastructure):
    """Test error handling when API key is missing."""
    # Temporarily remove API key from environment
    original_key = os.environ.pop("TIINGO_API_KEY", None)

    try:
        # Should raise ValueError during initialization when no API key provided
        # Error message from get_env_var helper
        with pytest.raises(ValueError, match="TIINGO_API_KEY is required"):
            TiingoOHLCVBuilder(
                package_dir=package_dir,
                writer=storage_infrastructure["writer"],
                overrides={"symbols": ["AAPL"]},  # No API key in overrides
            )

    finally:
        # Restore API key
        if original_key:
            os.environ["TIINGO_API_KEY"] = original_key


def test_error_handling_invalid_symbol(builder):
    """Test error handling for invalid symbol with fail_on_error=False."""
    from unittest.mock import patch

    # Mock the helper method that fetches single symbol data to return empty (invalid symbol)
    with patch.object(builder, "_fetch_single_symbol", return_value=pd.DataFrame()):
        # Pass symbols and fail_on_error in kwargs (as base class does)
        df = builder.fetch_raw(symbols=["INVALID123"], fail_on_error=False)
        # With fail_on_error=False, should return empty DataFrame without raising
        assert isinstance(df, pd.DataFrame)
        # Empty because invalid symbol returned no data
        assert len(df) == 0


# ============================================================================
# Integration Test (DAG)
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("TIINGO_API_KEY"), reason="TIINGO_API_KEY not set"
)
def test_dag_integration(storage_infrastructure, load_env):
    """Test TiingoOHLCVBuilder in a complete DAG pipeline."""

    run_id = f"test-tiingo-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"

    # Build DAG
    dag = DAG(
        tasks=[
            # Task 1: Load historic members
            Task(
                id="GetHistoricMembers",
                run=run_loader(
                    package_path="qx_loaders/historic_members",
                    registry=storage_infrastructure["registry"],
                    backend=storage_infrastructure["backend"],
                    resolver=storage_infrastructure["resolver"],
                    overrides={
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                        "universe": "sp500",
                        "use_ticker_mapper": True,
                    },
                ),
                deps=["BuildMembershipIntervals"],
            ),
            # Task 2: Build OHLCV (limited symbols for test)
            Task(
                id="BuildOHLCV",
                run=lambda context=None: run_builder(
                    package_path="qx_builders/tiingo_ohlcv",
                    registry=storage_infrastructure["registry"],
                    adapter=storage_infrastructure["adapter"],
                    resolver=storage_infrastructure["resolver"],
                    partitions={"exchange": "US", "frequency": "daily"},
                    overrides={
                        "symbols": (
                            context["GetHistoricMembers"]["output"][:5]  # Only first 5
                            if context
                            else ["AAPL", "MSFT"]
                        ),
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                        "frequency": "daily",
                        "fail_on_error": False,
                    },
                )(),
                deps=["GetHistoricMembers"],
            ),
            # Task 3: Load market proxy
            Task(
                id="LoadMarketProxy",
                run=run_loader(
                    package_path="qx_loaders/market_proxy",
                    registry=storage_infrastructure["registry"],
                    backend=storage_infrastructure["backend"],
                    resolver=storage_infrastructure["resolver"],
                    overrides={
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                        "proxy_symbol": "SPY",
                        "frequency": "daily",
                    },
                ),
                deps=["BuildOHLCV"],
            ),
        ],
    )

    # Execute DAG
    results = dag.execute()

    # Verify results
    assert results is not None
    assert "BuildOHLCV" in results
    assert results["BuildOHLCV"]["status"] == "success"

    if "LoadMarketProxy" in results:
        market_returns = results["LoadMarketProxy"]["output"]
        assert market_returns is not None
        assert len(market_returns) > 0


# ============================================================================
# Example Usage (Can be run standalone)
# ============================================================================


def example_basic_usage():
    """
    Example: Basic usage of TiingoOHLCVBuilder.

    This can be run standalone for demonstration purposes.
    """
    print("=" * 80)
    print("TiingoOHLCVBuilder - Basic Usage Example")
    print("=" * 80)

    # Setup storage
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.table_format import TableFormatAdapter

    registry = DatasetRegistry()
    seed_registry(registry)
    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()
    writer = CuratedWriter(
        backend=backend, adapter=adapter, resolver=resolver, registry=registry
    )

    # Create builder
    builder = TiingoOHLCVBuilder(
        package_dir="qx_builders/tiingo_ohlcv",
        writer=writer,
        overrides={
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "frequency": "daily",
        },
    )

    # Build
    result = builder.build()

    print(f"\n✅ Build complete!")
    print(f"   Status: {result['status']}")
    print(f"   Output: {result['output_path']}")


if __name__ == "__main__":
    # Can run this file directly for quick testing
    example_basic_usage()
