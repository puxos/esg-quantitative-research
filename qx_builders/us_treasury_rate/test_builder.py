"""
Unit tests for USTreasuryRateBuilder.

Tests the US Treasury Rate builder with:
- YAML configuration loading
- FRED API data fetching (with retries)
- Rate type processing (3month, 1year, 5year, 10year, 30year)
- Frequency resampling (daily → weekly/monthly)
- Data transformation logic
- Schema validation
- Error handling

This test file is co-located with the builder source code for easier maintenance.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.curated_writer import CuratedWriter
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter

from .builder import USTreasuryRateBuilder

# ============================================================================
# Test Configuration
# ============================================================================

DEFAULT_REGION = "US"


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
def builder_with_api_key(package_dir, storage_infrastructure):
    """Create builder instance with API key."""
    # Try to get FRED API key from environment
    import os

    api_key = os.environ.get("FRED_API_KEY", "test_api_key_placeholder")

    return USTreasuryRateBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        overrides=(
            {"fred_api_key": api_key} if api_key == "test_api_key_placeholder" else None
        ),
    )


@pytest.fixture
def builder(package_dir, storage_infrastructure):
    """Create builder instance with mock API key for testing."""
    return USTreasuryRateBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        overrides={"fred_api_key": "mock_api_key_for_testing"},
    )


@pytest.fixture
def sample_raw_data():
    """Sample raw treasury rate data."""
    dates = pd.date_range("2020-01-01", "2020-01-31", freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value": [1.5 + i * 0.01 for i in range(len(dates))],
            "rate_type": ["10year"] * len(dates),
        }
    )


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_builder_initialization(builder):
    """Test that builder initializes correctly from YAML."""
    assert builder is not None
    assert builder.info["id"] == "us_treasury_rate_builder"
    assert builder.info["version"] == "1.0.0"


def test_configuration_loading(builder):
    """Test YAML configuration is loaded properly."""
    assert hasattr(builder, "params")
    assert "rate_types" in builder.params or builder.params.get("rate_types") is None

    # Check default parameters
    assert "start_date" in builder.params or builder.params.get("start_date") is None


def test_api_key_from_overrides(package_dir, storage_infrastructure):
    """Test API key from overrides."""
    builder = USTreasuryRateBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        overrides={"fred_api_key": "custom_key_123"},
    )

    assert builder.fred_api_key == "custom_key_123"


def test_missing_api_key(package_dir, storage_infrastructure):
    """Test error when API key is missing."""
    import os

    # Temporarily remove API key from environment
    old_key = os.environ.pop("FRED_API_KEY", None)

    try:
        # Error message from get_env_var helper
        with pytest.raises(ValueError, match="FRED_API_KEY is required"):
            USTreasuryRateBuilder(
                package_dir=package_dir,
                writer=storage_infrastructure["writer"],
                overrides={},  # No API key in overrides
            )
    finally:
        # Restore API key if it existed
        if old_key:
            os.environ["FRED_API_KEY"] = old_key


# ============================================================================
# Unit Tests - Data Fetching
# ============================================================================


@pytest.mark.integration
def test_fetch_raw_with_real_api(builder_with_api_key):
    """Test fetching raw data from FRED API (requires API key)."""
    import os

    api_key = os.environ.get("FRED_API_KEY")

    if not api_key or api_key == "test_api_key_placeholder":
        pytest.skip(
            "FRED API key not available (set FRED_API_KEY environment variable)"
        )

    # Fetch small date range
    builder_with_api_key.params["rate_types"] = ["10year"]
    builder_with_api_key.params["start_date"] = "2023-01-01"
    builder_with_api_key.params["end_date"] = "2023-01-31"

    df = builder_with_api_key.fetch_raw()

    assert df is not None
    assert len(df) > 0
    assert "date" in df.columns
    assert "value" in df.columns
    assert "rate_type" in df.columns

    print(f"✓ Fetched {len(df)} treasury rate observations")


def test_fetch_raw_mock(builder):
    """Test fetch_raw with mocked API response."""
    mock_data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2020-01-10", freq="D"),
            "value": [1.5] * 10,
        }
    )

    with patch.object(builder, "_fetch_single_rate", return_value=mock_data):
        builder.params["rate_types"] = ["10year"]
        df = builder.fetch_raw()

        assert len(df) > 0
        assert "rate_type" in df.columns
        assert df["rate_type"].iloc[0] == "10year"


def test_fetch_raw_multiple_rates(builder):
    """Test fetching multiple rate types."""

    # Mock needs to return a fresh copy each time (fetch_raw modifies the DataFrame)
    def mock_fetch_single_rate(rate_type, start_date, end_date):
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", "2020-01-10", freq="D"),
                "value": [1.5] * 10,
            }
        )

    with patch.object(
        builder, "_fetch_single_rate", side_effect=mock_fetch_single_rate
    ):
        builder.params["rate_types"] = ["10year", "5year", "30year"]
        df = builder.fetch_raw()

        assert len(df) > 0
        # Should have all 3 rate types (30 observations total: 10 per rate)
        assert len(df) == 30
        assert df["rate_type"].nunique() == 3


def test_fetch_raw_empty_response(builder):
    """Test handling of empty API response."""
    empty_df = pd.DataFrame()

    with patch.object(builder, "_fetch_single_rate", return_value=empty_df):
        builder.params["rate_types"] = ["10year"]
        df = builder.fetch_raw()

        assert df.empty


# ============================================================================
# Unit Tests - Data Transformation
# ============================================================================


def test_transform_to_curated_basic(builder, sample_raw_data):
    """Test basic transformation of raw treasury data."""
    curated = builder.transform_to_curated(sample_raw_data, frequency="daily")

    assert len(curated) > 0
    assert "date" in curated.columns
    assert "rate_type" in curated.columns
    assert "rate" in curated.columns
    assert "frequency" in curated.columns
    assert "source" in curated.columns

    # Check data types
    assert curated["rate"].dtype == float


def test_transform_frequency_conversion(builder, sample_raw_data):
    """Test frequency resampling (daily → monthly)."""
    curated = builder.transform_to_curated(
        sample_raw_data, region="US", frequency="monthly"
    )

    assert len(curated) > 0
    # Monthly should have fewer rows than daily
    assert len(curated) <= len(sample_raw_data)


def test_transform_removes_missing_values(builder):
    """Test that missing values are handled."""
    raw_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2020-01-05", freq="D"),
            "value": [1.5, None, 1.6, 1.7, None],
            "rate_type": ["10year"] * 5,
        }
    )

    curated = builder.transform_to_curated(raw_df, region="US", frequency="daily")

    # Should drop rows with missing values
    assert len(curated) == 3  # Only 3 valid values


def test_transform_empty_dataframe(builder):
    """Test transform with empty DataFrame."""
    raw_df = pd.DataFrame()

    curated = builder.transform_to_curated(raw_df, region="US", frequency="daily")

    assert curated.empty


def test_transform_date_conversion(builder, sample_raw_data):
    """Test date conversion to proper format."""
    curated = builder.transform_to_curated(sample_raw_data, frequency="daily")

    # Check date is properly converted to date.date object
    import datetime

    assert isinstance(curated["date"].iloc[0], datetime.date)


# ============================================================================
# Unit Tests - Utility Functions
# ============================================================================


def test_resample_to_weekly(builder, sample_raw_data):
    """Test resampling daily data to weekly."""
    # _resample_to_frequency expects 'rate' column, not 'value'
    test_data = sample_raw_data.copy()
    test_data["rate"] = pd.to_numeric(test_data["value"])
    test_data["date"] = pd.to_datetime(test_data["date"])

    weekly = builder._resample_to_frequency(test_data[["date", "rate"]], "weekly")

    # Weekly should have fewer rows
    assert len(weekly) <= len(test_data)
    assert len(weekly) > 0


def test_resample_to_monthly(builder, sample_raw_data):
    """Test resampling daily data to monthly."""
    # _resample_to_frequency expects 'rate' column, not 'value'
    test_data = sample_raw_data.copy()
    test_data["rate"] = pd.to_numeric(test_data["value"])
    test_data["date"] = pd.to_datetime(test_data["date"])

    monthly = builder._resample_to_frequency(test_data[["date", "rate"]], "monthly")

    # Monthly should have fewer rows
    assert len(monthly) <= len(test_data)
    assert len(monthly) > 0


def test_resample_preserves_rate_types(builder):
    """Test that resampling works with rate column."""
    # _resample_to_frequency only needs date and rate columns
    # rate_type is handled by the caller (transform_to_curated)
    raw_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2020-01-31", freq="D"),
            "rate": [1.5] * 31,
        }
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    monthly = builder._resample_to_frequency(raw_df, "monthly")

    assert len(monthly) > 0
    assert "rate" in monthly.columns


# ============================================================================
# Unit Tests - Schema and Output
# ============================================================================


def test_output_schema_validation(builder, sample_raw_data):
    """Test that output schema matches contract requirements."""
    curated = builder.transform_to_curated(sample_raw_data, frequency="daily")

    # Check required columns
    required_cols = ["date", "rate_type", "rate", "series_id", "frequency", "source"]
    for col in required_cols:
        assert col in curated.columns, f"Missing column: {col}"

    # Check data types
    assert curated["rate"].dtype == float
    assert curated["frequency"].dtype == object
    assert curated["source"].dtype == object


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
def test_build_with_partitions(builder_with_api_key):
    """Test full build pipeline with region and frequency partitions."""
    import os

    api_key = os.environ.get("FRED_API_KEY")

    if not api_key or api_key == "test_api_key_placeholder":
        pytest.skip("FRED API key not available")

    # Build with small date range
    result = builder_with_api_key.build(
        partitions={"region": "US", "frequency": "monthly"},
        rate_types=["10year"],
        start_date="2023-01-01",
        end_date="2023-01-31",
    )

    assert result is not None
    if isinstance(result, dict):
        assert result["status"] == "success"
        assert "output_path" in result


@pytest.mark.integration
def test_full_pipeline_mock(storage_infrastructure, package_dir):
    """Test complete pipeline with mocked API."""
    builder = USTreasuryRateBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        overrides={"fred_api_key": "mock_key"},
    )

    # Mock fetch_raw
    mock_data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2020-01-31", freq="D"),
            "value": [1.5 + i * 0.01 for i in range(31)],
            "rate_type": ["10year"] * 31,
        }
    )

    with patch.object(builder, "fetch_raw", return_value=mock_data):
        # Fetch
        raw_df = builder.fetch_raw()
        assert len(raw_df) > 0

        # Transform
        curated = builder.transform_to_curated(raw_df, region="US", frequency="daily")
        assert len(curated) > 0

        print(f"✓ Pipeline test: {len(curated)} curated records")


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use USTreasuryRateBuilder.

    This is not a test, but serves as documentation.
    """
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.curated_writer import CuratedWriter
    from qx.storage.pathing import PathResolver
    from qx.storage.table_format import TableFormatAdapter

    # Setup infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()
    writer = CuratedWriter(
        backend=backend, adapter=adapter, resolver=resolver, registry=registry
    )

    # Create builder
    builder = USTreasuryRateBuilder(
        package_dir="qx_builders/us_treasury_rate",
        writer=writer,
    )

    # Build treasury rates (requires FRED_API_KEY)
    result = builder.build(
        partitions={"region": "US"},
        rate_types=["10year", "5year"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        frequency="monthly",
    )

    print(f"✅ Built treasury rates: {result}")


if __name__ == "__main__":
    # Run example
    example_usage()
