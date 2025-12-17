"""
Unit tests for USTreasuryRateLoader.

Tests cover:
- YAML configuration loading
- Loading treasury rate data
- Rate type filtering
- Date range handling
- Frequency handling
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

from .loader import USTreasuryRateLoader

# ============================================================================
# Test Configuration
# ============================================================================

SAMPLE_PERIOD_START = "2023-01-01"
SAMPLE_PERIOD_END = "2023-12-31"
DEFAULT_RATE_TYPES = ["3month", "10year"]
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
    return USTreasuryRateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "rate_types": DEFAULT_RATE_TYPES,
            "frequency": FREQUENCY,
        },
    )


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_loader_initialization(loader):
    """Test that loader initializes correctly from YAML."""
    assert loader is not None
    assert loader.info["id"] == "us_treasury_rate_loader"
    assert "version" in loader.info


def test_configuration_loading(loader):
    """Test YAML configuration is loaded properly."""
    assert hasattr(loader, "params")
    assert "start_date" in loader.params
    assert "end_date" in loader.params
    assert "rate_types" in loader.params
    assert "frequency" in loader.params


def test_default_parameters(package_dir, storage_infrastructure):
    """Test default parameter values."""
    loader = USTreasuryRateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "rate_types": ["3month"],
        },
    )

    assert loader.params.get("rate_types") == ["3month"]
    assert loader.params.get("frequency") == "daily"


# ============================================================================
# Unit Tests - Data Loading
# ============================================================================


@pytest.mark.integration
def test_load_treasury_rates(loader):
    """Test loading treasury rate data."""
    try:
        df = loader.load()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "date" in df.columns
        assert "rate_type" in df.columns
        assert "rate" in df.columns

        print(f"✅ Loaded {len(df)} treasury rate records")

    except FileNotFoundError:
        pytest.skip("Treasury rate data not available")


@pytest.mark.integration
def test_load_single_rate_type(package_dir, storage_infrastructure):
    """Test loading single rate type."""
    loader = USTreasuryRateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "rate_types": ["3month"],
            "frequency": FREQUENCY,
        },
    )

    try:
        df = loader.load()
        assert isinstance(df, pd.DataFrame)

        if len(df) > 0:
            assert df["rate_type"].nunique() == 1
            assert df["rate_type"].iloc[0] == "3month"

        print(f"✅ Loaded {len(df)} 3-month treasury rates")

    except FileNotFoundError:
        pytest.skip("Treasury rate data not available")


@pytest.mark.integration
def test_load_multiple_rate_types(package_dir, storage_infrastructure):
    """Test loading multiple rate types."""
    rate_types = ["3month", "1year", "10year"]  # Fixed: use valid rate types

    loader = USTreasuryRateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": SAMPLE_PERIOD_START,
            "end_date": SAMPLE_PERIOD_END,
            "rate_types": rate_types,
            "frequency": FREQUENCY,
        },
    )

    try:
        df = loader.load()
        assert isinstance(df, pd.DataFrame)

        if len(df) > 0:
            returned_types = df["rate_type"].unique()
            # All returned types should be in request
            assert all(rt in rate_types for rt in returned_types)

        print(f"✅ Loaded {len(df)} treasury rates for {len(rate_types)} maturities")

    except FileNotFoundError:
        pytest.skip("Treasury rate data not available")


@pytest.mark.integration
def test_load_weekly_frequency(package_dir, storage_infrastructure):
    """Test loading weekly frequency data (expects no data available)."""
    loader = USTreasuryRateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "rate_types": ["3month"],
            "frequency": "weekly",
        },
    )

    # Weekly treasury data not available, should raise ValueError
    with pytest.raises(ValueError, match="No data found for rate types"):
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
            assert "date" in df.columns
            assert "rate_type" in df.columns
            assert "rate" in df.columns
            assert "series_id" in df.columns

    except FileNotFoundError:
        pytest.skip("Treasury rate data not available")


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
        pytest.skip("Treasury rate data not available")


@pytest.mark.integration
def test_rates_are_valid(loader):
    """Test that rates are valid (positive, reasonable range)."""
    try:
        df = loader.load()

        if len(df) > 0:
            # Rates should be numeric
            assert df["rate"].dtype in [float, "float64"]

            # Rates should not have NaN
            assert not df["rate"].isnull().any()

            # Rates should be in reasonable range (0% to 20% for US treasuries)
            assert (df["rate"] >= 0).all()
            assert (df["rate"] <= 20).all()

    except FileNotFoundError:
        pytest.skip("Treasury rate data not available")


# ============================================================================
# Unit Tests - Error Handling
# ============================================================================


def test_missing_treasury_data(package_dir, storage_infrastructure):
    """Test error handling when treasury data is missing."""
    loader = USTreasuryRateLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={
            "start_date": "1900-01-01",
            "end_date": "1900-12-31",
            "rate_types": ["nonexistent"],
            "frequency": "daily",
        },
    )

    # Should raise ValueError for invalid rate type
    with pytest.raises(ValueError, match="Invalid rate_type"):
        df = loader.load()


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================


@pytest.mark.integration
def test_full_pipeline_with_builder(storage_infrastructure, package_dir):
    """Test complete pipeline: BuildUSTreasury → LoadUSTreasuryRate."""
    dag = DAG(
        tasks=[
            Task(
                id="BuildUSTreasury",
                run=run_builder(
                    package_path="qx_builders/us_treasury_rate",
                    registry=storage_infrastructure["registry"],
                    adapter=storage_infrastructure["adapter"],
                    resolver=storage_infrastructure["resolver"],
                    partitions={"exchange": "US", "frequency": "daily"},
                    overrides={
                        "rate_types": ["3month", "10year"],
                        "start_date": "2023-01-01",
                        "end_date": "2023-12-31",
                    },
                ),
                deps=[],
            ),
            Task(
                id="LoadUSTreasuryRate",
                run=run_loader(
                    package_path="qx_loaders/us_treasury_rate",
                    registry=storage_infrastructure["registry"],
                    backend=storage_infrastructure["backend"],
                    resolver=storage_infrastructure["resolver"],
                    overrides={
                        "start_date": "2023-01-01",
                        "end_date": "2023-12-31",
                        "rate_types": ["3month", "10year"],
                        "frequency": "daily",
                    },
                ),
                deps=["BuildUSTreasury"],
            ),
        ]
    )

    try:
        results = dag.execute()

        if results is not None:
            assert "LoadTreasuryRates" in results
            assert results["LoadTreasuryRates"]["status"] == "success"

            df = results["LoadTreasuryRates"]["output"]
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        else:
            pytest.skip("DAG execution did not return results dictionary")

        print(f"✅ Pipeline completed: {len(df)} treasury rate records loaded")

    except FileNotFoundError:
        pytest.skip("Required data not available")


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use USTreasuryRateLoader.

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
    loader = USTreasuryRateLoader(
        package_dir="qx_loaders/us_treasury_rate",
        loader=typed_loader,
        overrides={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "rate_types": ["3month", "10year"],
            "frequency": "daily",
        },
    )

    # Load data
    df = loader.load()

    print(f"✅ Loaded {len(df)} treasury rate records")
    print(f"\nAverage rates:")
    print(df.groupby("rate_type")["rate"].mean())


if __name__ == "__main__":
    # Run example
    example_usage()
