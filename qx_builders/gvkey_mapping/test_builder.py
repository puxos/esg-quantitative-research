"""
Unit tests for GVKEYMappingBuilder.

Tests the GVKEY-ticker mapping builder with:
- YAML configuration loading
- Excel file reading (data_mapping.xlsx)
- Ticker normalization (BRK.B → BRK-B)
- Data transformation logic
- Schema validation
- Error handling
- Integration with ESG builder

This test file is co-located with the builder source code for easier maintenance.
"""

from pathlib import Path

import pandas as pd
import pytest

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.curated_writer import CuratedWriter
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter

from .builder import GVKEYMappingBuilder, clean_ticker_symbol

# ============================================================================
# Test Configuration
# ============================================================================

DEFAULT_EXCHANGE = "US"


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
def builder(package_dir, storage_infrastructure):
    """Create builder instance."""
    return GVKEYMappingBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
    )


# ============================================================================
# Unit Tests - Utility Functions
# ============================================================================


def test_clean_ticker_symbol_basic():
    """Test basic ticker symbol cleaning."""
    assert clean_ticker_symbol("AAPL") == "AAPL"
    assert clean_ticker_symbol("aapl") == "AAPL"
    assert clean_ticker_symbol("  aapl  ") == "AAPL"


def test_clean_ticker_symbol_class_shares():
    """Test cleaning of class share tickers (period to hyphen)."""
    assert clean_ticker_symbol("BRK.B") == "BRK-B"
    assert clean_ticker_symbol("BRK.A") == "BRK-A"
    assert clean_ticker_symbol("brk.b") == "BRK-B"


def test_clean_ticker_symbol_empty():
    """Test empty ticker handling."""
    assert clean_ticker_symbol("") == ""
    assert clean_ticker_symbol(None) == ""
    assert clean_ticker_symbol(pd.NA) == ""


def test_clean_ticker_symbol_already_hyphenated():
    """Test tickers that already have hyphens."""
    assert clean_ticker_symbol("JPM-C") == "JPM-C"
    assert clean_ticker_symbol("SPCE-WT") == "SPCE-WT"


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_builder_initialization(builder):
    """Test that builder initializes correctly from YAML."""
    assert builder is not None
    assert builder.info["id"] == "gvkey_mapping_builder"
    assert builder.info["version"] == "1.0.0"


def test_configuration_loading(builder):
    """Test YAML configuration is loaded properly."""
    assert hasattr(builder, "params")
    assert "exchange" in builder.params
    assert builder.params["exchange"] == "US"

    # Check default parameters
    assert "crsp_file" in builder.params
    assert "write_mode" in builder.params


def test_raw_file_path(builder):
    """Test that raw file path is set correctly."""
    assert hasattr(builder, "raw_file_path")
    assert "data_mapping.xlsx" in str(builder.raw_file_path)


# ============================================================================
# Unit Tests - Data Fetching
# ============================================================================


def test_fetch_raw_file_exists(builder):
    """Test fetching raw data from Excel file."""
    # Check if raw file exists
    raw_path = Path(builder.raw_file_path)

    if raw_path.exists():
        df = builder.fetch_raw()

        assert df is not None
        assert len(df) > 0
        assert "gvkey" in df.columns
        assert "tic" in df.columns

        print(f"✓ Loaded {len(df):,} raw GVKEY mappings")
    else:
        pytest.skip(f"Raw file not found: {raw_path}")


def test_fetch_raw_missing_file():
    """Test error handling when Excel file is missing."""
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.curated_writer import CuratedWriter
    from qx.storage.pathing import PathResolver
    from qx.storage.table_format import TableFormatAdapter

    # Create infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)
    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()
    writer = CuratedWriter(
        backend=backend, adapter=adapter, resolver=resolver, registry=registry
    )

    package_dir = str(Path(__file__).parent)

    builder = GVKEYMappingBuilder(
        package_dir=package_dir,
        writer=writer,
        overrides={"crsp_file": "./raw/nonexistent_file.xlsx"},
    )

    with pytest.raises(FileNotFoundError, match="GVKEY mapping file not found"):
        builder.fetch_raw()


# ============================================================================
# Unit Tests - Data Transformation
# ============================================================================


def test_transform_to_curated_basic(builder):
    """Test basic transformation of raw GVKEY mapping."""
    # Create sample raw data
    raw_df = pd.DataFrame(
        {
            "gvkey": [1690, 12141, 14593],
            "tic": ["AAPL", "MSFT", "GOOGL"],
            "other_col": ["data1", "data2", "data3"],
        }
    )

    curated = builder.transform_to_curated(raw_df)

    assert len(curated) == 3
    assert "gvkey" in curated.columns
    assert "ticker" in curated.columns
    assert "ticker_raw" in curated.columns

    # Check ticker normalization
    assert curated[curated["gvkey"] == 1690]["ticker"].values[0] == "AAPL"


def test_transform_ticker_normalization(builder):
    """Test ticker symbol normalization (period → hyphen)."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1234, 5678],
            "tic": ["BRK.B", "brk.a"],
        }
    )

    curated = builder.transform_to_curated(raw_df)

    assert curated[curated["gvkey"] == 1234]["ticker"].values[0] == "BRK-B"
    assert curated[curated["gvkey"] == 5678]["ticker"].values[0] == "BRK-A"


def test_transform_removes_empty_tickers(builder):
    """Test that empty tickers are removed."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1234, 5678, 9999],
            "tic": ["AAPL", "", "MSFT"],
        }
    )

    curated = builder.transform_to_curated(raw_df)

    # Should have 2 rows (AAPL and MSFT), empty ticker removed
    assert len(curated) == 2
    # GVKEY 5678 with empty ticker should be removed
    assert 5678 not in curated["gvkey"].values
    # GVKEY 1234 and 9999 should remain
    assert 1234 in curated["gvkey"].values
    assert 9999 in curated["gvkey"].values


def test_transform_deduplicates_gvkey(builder):
    """Test that duplicate GVKEYs are removed (keeping first)."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1234, 1234, 5678],
            "tic": ["AAPL", "MSFT", "GOOGL"],
        }
    )

    curated = builder.transform_to_curated(raw_df)

    assert len(curated) == 2
    assert curated[curated["gvkey"] == 1234]["ticker"].values[0] == "AAPL"


def test_transform_deduplicates_ticker(builder):
    """Test that duplicate tickers are removed (keeping first)."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1234, 5678],
            "tic": ["AAPL", "AAPL"],
        }
    )

    curated = builder.transform_to_curated(raw_df)

    assert len(curated) == 1
    assert curated["gvkey"].values[0] == 1234


def test_transform_missing_gvkey_column():
    """Test error handling when gvkey column is missing."""
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.curated_writer import CuratedWriter
    from qx.storage.pathing import PathResolver
    from qx.storage.table_format import TableFormatAdapter

    registry = DatasetRegistry()
    seed_registry(registry)
    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()
    writer = CuratedWriter(
        backend=backend, adapter=adapter, resolver=resolver, registry=registry
    )

    package_dir = str(Path(__file__).parent)
    builder = GVKEYMappingBuilder(package_dir=package_dir, writer=writer)

    raw_df = pd.DataFrame(
        {
            "tic": ["AAPL", "MSFT"],
        }
    )

    with pytest.raises(ValueError, match="Missing 'gvkey' column"):
        builder.transform_to_curated(raw_df)


def test_transform_missing_tic_column():
    """Test error handling when tic column is missing."""
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.curated_writer import CuratedWriter
    from qx.storage.pathing import PathResolver
    from qx.storage.table_format import TableFormatAdapter

    registry = DatasetRegistry()
    seed_registry(registry)
    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()
    writer = CuratedWriter(
        backend=backend, adapter=adapter, resolver=resolver, registry=registry
    )

    package_dir = str(Path(__file__).parent)
    builder = GVKEYMappingBuilder(package_dir=package_dir, writer=writer)

    raw_df = pd.DataFrame(
        {
            "gvkey": [1234, 5678],
        }
    )

    with pytest.raises(ValueError, match="Missing 'tic' column"):
        builder.transform_to_curated(raw_df)


def test_transform_empty_dataframe(builder):
    """Test transform with empty DataFrame."""
    raw_df = pd.DataFrame()

    curated = builder.transform_to_curated(raw_df)

    assert curated.empty


# ============================================================================
# Unit Tests - Build Pipeline
# ============================================================================


@pytest.mark.integration
def test_build_with_partitions(builder, storage_infrastructure):
    """Test full build pipeline with exchange partition."""
    # Check if raw file exists
    raw_path = Path(builder.raw_file_path)

    if not raw_path.exists():
        pytest.skip(f"Raw file not found: {raw_path}")

    # Build with explicit partition
    result = builder.build(partitions={"exchange": "US"})

    assert result is not None
    if isinstance(result, dict):
        assert result["status"] == "success"
        assert "output_path" in result
    else:
        # Legacy: result might be string path
        assert isinstance(result, str)


@pytest.mark.integration
def test_build_with_exchange_parameter(builder, storage_infrastructure):
    """Test build with legacy exchange parameter."""
    raw_path = Path(builder.raw_file_path)

    if not raw_path.exists():
        pytest.skip(f"Raw file not found: {raw_path}")

    # Build with exchange parameter (legacy)
    result = builder.build(exchange="US")

    assert result is not None


def test_build_empty_data(package_dir, storage_infrastructure):
    """Test build behavior when no data is returned."""
    # Create builder with invalid override to simulate no data
    builder = GVKEYMappingBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        overrides={"crsp_file": "./raw/data_mapping.xlsx"},
    )

    # Mock transform to return empty DataFrame
    original_transform = builder.transform_to_curated

    def mock_transform(*args, **kwargs):
        return pd.DataFrame(columns=["gvkey", "ticker", "ticker_raw"])

    builder.transform_to_curated = mock_transform

    result = builder.build(partitions={"exchange": "US"})

    # Base builder doesn't check for empty data - it writes it
    # So we just verify the build completes successfully
    if isinstance(result, dict):
        assert result["status"] == "success"
        assert "output_path" in result
    else:
        # Legacy: result might be string path
        assert isinstance(result, str)


# ============================================================================
# Unit Tests - Schema and Output
# ============================================================================


def test_output_schema_validation(builder):
    """Test that output schema matches contract requirements."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1690, 12141, 14593],
            "tic": ["AAPL", "MSFT", "GOOGL"],
        }
    )

    curated = builder.transform_to_curated(raw_df)

    # Check required columns
    assert "gvkey" in curated.columns
    assert "ticker" in curated.columns
    assert "ticker_raw" in curated.columns

    # Check data types
    assert curated["gvkey"].dtype in [pd.Int64Dtype(), "int64"]
    assert curated["ticker"].dtype == object
    assert curated["ticker_raw"].dtype == object


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
def test_full_pipeline(storage_infrastructure, package_dir):
    """Test complete pipeline: fetch → transform → build."""
    raw_path = Path(package_dir) / "raw" / "data_mapping.xlsx"

    if not raw_path.exists():
        pytest.skip(f"Raw file not found: {raw_path}")

    builder = GVKEYMappingBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
    )

    # Fetch
    raw_df = builder.fetch_raw()
    assert len(raw_df) > 0

    # Transform
    curated = builder.transform_to_curated(raw_df)
    assert len(curated) > 0
    assert len(curated) <= len(raw_df)  # May be fewer after deduplication

    # Build
    result = builder.build(partitions={"exchange": "US"})
    assert result is not None


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use GVKEYMappingBuilder.

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
    builder = GVKEYMappingBuilder(
        package_dir="qx_builders/gvkey_mapping",
        writer=writer,
    )

    # Build GVKEY mapping
    result = builder.build(partitions={"exchange": "US"})

    print(f"✅ Built GVKEY mapping: {result}")


if __name__ == "__main__":
    # Run example
    example_usage()
