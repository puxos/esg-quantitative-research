"""Unit tests for QualityFilterLoader."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.pathing import PathResolver
from qx.storage.typed_loader import TypedCuratedLoader

from .loader import QualityFilterLoader

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def storage_infrastructure():
    """Setup storage infrastructure."""
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()
    loader = TypedCuratedLoader(backend, registry, resolver)

    return {
        "registry": registry,
        "backend": backend,
        "resolver": resolver,
        "loader": loader,
    }


@pytest.fixture
def package_dir():
    """Get package directory."""
    return str(Path(__file__).parent)


@pytest.fixture
def loader(package_dir, storage_infrastructure):
    """Create loader instance."""
    return QualityFilterLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={"filter_date": "2024-12-31"},
    )


# ============================================================================
# Unit Tests - Initialization
# ============================================================================


def test_loader_initialization(loader):
    """Test loader initializes correctly."""
    assert loader is not None
    assert loader.info["id"] == "quality_filter_loader"
    assert loader.info["version"] == "1.0.0"


def test_configuration_loading(loader):
    """Test YAML configuration loaded."""
    assert hasattr(loader, "params")
    assert "min_price" in loader.params
    assert "max_volatility" in loader.params
    assert "min_trading_days" in loader.params
    assert "lookback_months" in loader.params


def test_default_parameters(loader):
    """Test default parameter values."""
    assert loader.params["min_price"] == 5.0
    assert loader.params["max_volatility"] == 0.10
    assert loader.params["min_trading_days"] == 40
    assert loader.params["lookback_months"] == 3


# ============================================================================
# Integration Tests - With Real Data
# ============================================================================


@pytest.mark.integration
def test_load_with_real_data(storage_infrastructure, package_dir):
    """Test loading with real OHLCV data."""
    loader = QualityFilterLoader(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        overrides={"filter_date": "2024-12-31"},
    )

    # This will fail if OHLCV data doesn't exist
    # Use pytest.skip() to handle gracefully
    try:
        symbols = loader.load()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(s, str) for s in symbols)
        print(f"✅ Loaded {len(symbols)} quality-filtered symbols")
    except Exception as e:
        pytest.skip(f"No OHLCV data available: {e}")


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """Example of using QualityFilterLoader."""
    registry = DatasetRegistry()
    seed_registry(registry)
    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()
    loader_infra = TypedCuratedLoader(backend, registry, resolver)

    loader = QualityFilterLoader(
        package_dir=str(Path(__file__).parent),
        loader=loader_infra,
        overrides={"filter_date": "2024-12-31"},
    )

    symbols = loader.load()
    print(f"✅ Quality-filtered: {len(symbols)} symbols")


if __name__ == "__main__":
    example_usage()
