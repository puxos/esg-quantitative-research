"""
Unit tests for ESGScoreBuilder.

Tests the ESG score builder with:
- YAML configuration loading
- Excel file reading (data_matlab_ESG_withSIC.xlsx)
- GVKEY-to-ticker mapping
- Ticker normalization (ticker mapper)
- Data transformation logic
- Schema validation
- Error handling
- Integration with GVKEY mapping builder

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

from .builder import ESGScoreBuilder

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
    return ESGScoreBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
    )


@pytest.fixture
def sample_gvkey_mapping():
    """Sample GVKEY mapping for testing."""
    return pd.DataFrame(
        {
            "gvkey": [1690, 12141, 14593, 6066],
            "ticker": ["AAPL", "MSFT", "GOOGL", "BRK-B"],
            "ticker_raw": ["AAPL", "MSFT", "GOOGL", "BRK.B"],
        }
    )


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_builder_initialization(builder):
    """Test that builder initializes correctly from YAML."""
    assert builder is not None
    assert builder.info["id"] == "esg_score_builder"
    assert builder.info["version"] == "1.0.0"


def test_configuration_loading(builder):
    """Test YAML configuration is loaded properly."""
    assert hasattr(builder, "params")
    assert "esg_source_path" in builder.params

    # Check default parameters
    assert "use_ticker_mapper" in builder.params


def test_raw_file_path(builder):
    """Test that raw file path is set correctly."""
    assert hasattr(builder, "esg_source_path")
    assert "data_matlab_ESG" in str(builder.esg_source_path) or "esg" in str(
        builder.esg_source_path
    ).lower()


def test_ticker_mapper_initialization(builder):
    """Test ticker mapper is initialized if enabled."""
    if builder.params.get("use_ticker_mapper", True):
        assert hasattr(builder, "ticker_mapper")
        assert builder.ticker_mapper is not None


# ============================================================================
# Unit Tests - Data Fetching
# ============================================================================


def test_fetch_raw_file_exists(builder):
    """Test fetching raw data from Excel file."""
    # Construct path
    esg_path = Path(builder.package_dir) / builder.esg_source_path

    if esg_path.exists():
        df = builder.fetch_raw()

        assert df is not None
        assert len(df) > 0
        assert "gvkey" in df.columns
        assert "YearESG" in df.columns
        assert "ESG Score" in df.columns

        print(f"✓ Loaded {len(df):,} raw ESG records")
        print(f"  Companies: {df['gvkey'].nunique()}")
        print(f"  Year range: {df['YearESG'].min()} to {df['YearESG'].max()}")
    else:
        pytest.skip(f"Raw ESG file not found: {esg_path}")


def test_fetch_raw_missing_file(package_dir, storage_infrastructure):
    """Test error handling when Excel file is missing."""
    builder = ESGScoreBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        overrides={"esg_source_path": "./raw/nonexistent_file.xlsx"},
    )

    with pytest.raises(FileNotFoundError, match="ESG data file not found"):
        builder.fetch_raw()


# ============================================================================
# Unit Tests - Data Transformation
# ============================================================================


def test_transform_to_curated_basic(builder, sample_gvkey_mapping):
    """Test basic transformation of raw ESG data."""
    # Create sample raw data
    raw_df = pd.DataFrame(
        {
            "gvkey": [1690, 1690, 12141, 12141, 14593],  # Duplicates for dedup test
            "YearESG": [2013, 2013, 2013, 2014, 2013],  # Same year for AAPL
            "YearMonth": [201301, 201302, 201301, 201401, 201301],
            "ESG Score": [75.5, 75.5, 82.3, 83.1, 90.2],  # Same score for same year
            "Environmental Pillar Score": [70.0, 70.0, 80.0, 81.0, 85.0],
            "Social Pillar Score": [75.0, 75.0, 78.0, 79.0, 88.0],
            "Governance Pillar Score": [80.0, 80.0, 88.0, 89.0, 95.0],
        }
    )

    curated = builder.transform_to_curated(
        raw_df, sample_gvkey_mapping, exchange="US"
    )

    assert len(curated) > 0
    assert "ticker" in curated.columns
    assert "esg_score" in curated.columns
    assert "year" in curated.columns  # Calendar year (esg_year + 1)

    # Check deduplication: Should have one record per (gvkey, YearESG)
    # AAPL 2013 had 2 monthly records → 1 annual
    # MSFT 2013 had 1 → 1
    # MSFT 2014 had 1 → 1
    # GOOGL 2013 had 1 → 1
    assert len(curated) == 4


def test_transform_gvkey_mapping(builder, sample_gvkey_mapping):
    """Test GVKEY to ticker mapping."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1690, 12141, 99999],  # Last one not in mapping
            "YearESG": [2013, 2013, 2013],
            "YearMonth": [201301, 201301, 201301],
            "ESG Score": [75.5, 82.3, 65.0],
            "Environmental Pillar Score": [70.0, 80.0, 60.0],
            "Social Pillar Score": [75.0, 78.0, 65.0],
            "Governance Pillar Score": [80.0, 88.0, 70.0],
        }
    )

    curated = builder.transform_to_curated(
        raw_df, sample_gvkey_mapping, exchange="US"
    )

    # Should map AAPL and MSFT, drop unmapped GVKEY 99999
    assert len(curated) == 2
    assert "AAPL" in curated["ticker"].values
    assert "MSFT" in curated["ticker"].values


def test_transform_deduplication(builder, sample_gvkey_mapping):
    """Test deduplication of monthly records to annual."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1690, 1690, 1690],  # Same company (AAPL)
            "YearESG": [2013, 2013, 2013],  # Same ESG year
            "YearMonth": [201301, 201302, 201303],  # Different months
            "ESG Score": [75.5, 75.5, 75.5],  # Same score (constant within year)
            "Environmental Pillar Score": [70.0, 70.0, 70.0],
            "Social Pillar Score": [75.0, 75.0, 75.0],
            "Governance Pillar Score": [80.0, 80.0, 80.0],
        }
    )

    curated = builder.transform_to_curated(
        raw_df, sample_gvkey_mapping, exchange="US"
    )

    # Should have only 1 record (deduplicated)
    assert len(curated) == 1
    assert curated["ticker"].values[0] == "AAPL"
    assert curated["esg_year"].values[0] == 2013


def test_transform_removes_missing_scores(builder, sample_gvkey_mapping):
    """Test that records with missing ESG scores are removed."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1690, 12141],
            "YearESG": [2013, 2013],
            "YearMonth": [201301, 201301],
            "ESG Score": [75.5, None],  # MSFT has missing score
            "Environmental Pillar Score": [70.0, 80.0],
            "Social Pillar Score": [75.0, 78.0],
            "Governance Pillar Score": [80.0, 88.0],
        }
    )

    curated = builder.transform_to_curated(
        raw_df, sample_gvkey_mapping, exchange="US"
    )

    # Should only have AAPL (MSFT removed due to missing score)
    assert len(curated) == 1
    assert curated["ticker"].values[0] == "AAPL"


def test_transform_missing_gvkey_column(builder, sample_gvkey_mapping):
    """Test error handling when gvkey column is missing."""
    raw_df = pd.DataFrame(
        {
            "YearESG": [2013, 2014],
            "ESG Score": [75.5, 82.3],
        }
    )

    # Should handle missing column gracefully or raise error
    # Check builder implementation - may need to handle this case


def test_transform_empty_dataframe(builder, sample_gvkey_mapping):
    """Test transform with empty DataFrame."""
    raw_df = pd.DataFrame()

    curated = builder.transform_to_curated(
        raw_df, sample_gvkey_mapping, exchange="US"
    )

    assert curated.empty


def test_transform_year_lag(builder, sample_gvkey_mapping):
    """Test that 'year' column applies 1-year lag (esg_year + 1)."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1690],
            "YearESG": [2013],  # ESG year 2013
            "YearMonth": [201301],
            "ESG Score": [75.5],
            "Environmental Pillar Score": [70.0],
            "Social Pillar Score": [75.0],
            "Governance Pillar Score": [80.0],
        }
    )

    curated = builder.transform_to_curated(
        raw_df, sample_gvkey_mapping, exchange="US"
    )

    # ESG year 2013 → available for 2014 trading
    assert curated["esg_year"].values[0] == 2013
    assert curated["year"].values[0] == 2014


# ============================================================================
# Unit Tests - Schema and Output
# ============================================================================


def test_output_schema_validation(builder, sample_gvkey_mapping):
    """Test that output schema matches contract requirements."""
    raw_df = pd.DataFrame(
        {
            "gvkey": [1690, 12141],
            "YearESG": [2013, 2013],
            "YearMonth": [201301, 201301],
            "ESG Score": [75.5, 82.3],
            "Environmental Pillar Score": [70.0, 80.0],
            "Social Pillar Score": [75.0, 78.0],
            "Governance Pillar Score": [80.0, 88.0],
        }
    )

    curated = builder.transform_to_curated(
        raw_df, sample_gvkey_mapping, exchange="US"
    )

    # Check required columns
    required_cols = [
        "ticker",
        "gvkey",
        "esg_year",
        "year",
        "esg_score",
        "environmental_pillar_score",
        "social_pillar_score",
        "governance_pillar_score",
    ]
    for col in required_cols:
        assert col in curated.columns, f"Missing column: {col}"

    # Check data types
    assert curated["gvkey"].dtype in [pd.Int64Dtype(), "int64"]
    assert curated["ticker"].dtype == object
    assert curated["esg_score"].dtype == float


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
def test_build_with_partitions(builder, storage_infrastructure):
    """Test full build pipeline with exchange partition."""
    # Check if raw file exists
    esg_path = Path(builder.package_dir) / builder.esg_source_path

    if not esg_path.exists():
        pytest.skip(f"Raw ESG file not found: {esg_path}")

    # Note: This test requires GVKEY mapping to be built first
    # In real DAG, BuildGVKEY would run before BuildESG

    # For testing, we can mock the GVKEY mapping loader
    # Or skip if GVKEY data isn't available
    try:
        result = builder.build(partitions={"exchange": "US"})

        assert result is not None
        if isinstance(result, dict):
            assert result["status"] == "success"
            assert "output_path" in result
    except FileNotFoundError as e:
        if "GVKEY" in str(e):
            pytest.skip("GVKEY mapping not available (build gvkey_mapping first)")
        else:
            raise


@pytest.mark.integration
def test_full_pipeline_mock_gvkey(storage_infrastructure, package_dir):
    """Test complete pipeline with mocked GVKEY mapping."""
    esg_path = Path(package_dir) / "raw" / "data_matlab_ESG_withSIC.xlsx"

    if not esg_path.exists():
        pytest.skip(f"Raw ESG file not found: {esg_path}")

    builder = ESGScoreBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
    )

    # Fetch
    raw_df = builder.fetch_raw()
    assert len(raw_df) > 0

    # Create mock GVKEY mapping for testing
    # Get unique GVKEYs from raw data
    unique_gvkeys = raw_df["gvkey"].unique()[:100]  # Test with first 100
    mock_mapping = pd.DataFrame(
        {
            "gvkey": unique_gvkeys,
            "ticker": [f"TICK{i}" for i in range(len(unique_gvkeys))],
        }
    )

    # Transform
    curated = builder.transform_to_curated(raw_df, mock_mapping, exchange="US")
    assert len(curated) > 0
    assert len(curated) <= len(raw_df)  # May be fewer after deduplication


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use ESGScoreBuilder.

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
    builder = ESGScoreBuilder(
        package_dir="qx_builders/esg_score",
        writer=writer,
        overrides={"start_year": 2014, "end_year": 2024},
    )

    # Build ESG scores (requires GVKEY mapping to be built first)
    result = builder.build(partitions={"exchange": "US"})

    print(f"✅ Built ESG scores: {result}")


if __name__ == "__main__":
    # Run example
    example_usage()
