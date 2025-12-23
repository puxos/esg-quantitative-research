"""
Unit tests for Market Beta Model.

Tests cover:
- Model initialization from YAML
- Return calculation
- Risk-free rate preparation
- Data merging
- Full-sample regression
- Rolling window regression
- Beta estimation with HAC errors
- Edge cases (insufficient data, singular matrix)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.common.types import (
    AssetClass,
    DatasetType,
    Domain,
    Frequency,
    Region,
    Subdomain,
)
from qx.engine.processed_writer import ProcessedWriterBase
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter

from .model import MarketBetaModel

# ============================================================================
# Test Configuration
# ============================================================================

DEFAULT_PARAMS = {
    "window": None,  # Full sample
    "min_observations": 24,
    "hac_lags": 6,
    "price_column": "adj_close",
    "annualization_factor": 12,
}

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
    writer = ProcessedWriterBase(adapter=adapter, resolver=resolver, registry=registry)

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
def model(package_dir, storage_infrastructure):
    """Create model instance."""

    # Create a mock loader with required attributes
    class MockLoader:
        def __init__(self):
            self.registry = storage_infrastructure["registry"]
            self.backend = storage_infrastructure["backend"]
            self.resolver = storage_infrastructure["resolver"]

    return MarketBetaModel(
        package_dir=package_dir,
        loader=MockLoader(),
        writer=storage_infrastructure["writer"],
    )


# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================


def test_model_initialization(model):
    """Test that model initializes correctly from YAML."""
    assert model is not None
    assert model.info["id"] == "market_beta_model"
    assert model.info["version"] == "1.0.0"


def test_configuration_loading(model):
    """Test YAML configuration is loaded properly."""
    assert hasattr(model, "params")
    assert "window" in model.params
    assert "min_observations" in model.params
    assert "hac_lags" in model.params


def test_io_types(model):
    """Test input/output types are defined."""
    assert hasattr(model, "inputs_cfg")
    assert hasattr(model, "output_dt")

    # Check required inputs
    input_names = [inp["name"] for inp in model.inputs_cfg]
    assert "equity_prices" in input_names
    assert "market_prices" in input_names
    assert "risk_free" in input_names


# ============================================================================
# Unit Tests - Data Processing
# ============================================================================


def test_compute_returns_single_symbol(model):
    """Test return computation for single symbol."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=5, freq="M"),
            "symbol": ["AAPL"] * 5,
            "adj_close": [100, 105, 103, 108, 110],
        }
    )

    returns = model._compute_returns(df, "adj_close")

    assert len(returns) == 4  # First observation dropped
    assert "return" in returns.columns
    assert returns["symbol"].nunique() == 1

    # Verify first return: (105 - 100) / 100 = 0.05
    assert abs(returns.iloc[0]["return"] - 0.05) < 1e-6


def test_compute_returns_multiple_symbols(model):
    """Test return computation for multiple symbols."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=6, freq="M").tolist() * 2,
            "symbol": ["AAPL"] * 6 + ["MSFT"] * 6,
            "adj_close": [100, 105, 103, 108, 110, 112]
            + [200, 210, 205, 215, 220, 218],
        }
    )

    returns = model._compute_returns(df, "adj_close")

    assert len(returns) == 10  # 5 per symbol (first dropped)
    assert returns["symbol"].nunique() == 2


def test_prepare_risk_free(model):
    """Test risk-free rate conversion from annual % to period return."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=3, freq="M"),
            "rate": [3.0, 3.5, 4.0],  # Annual percentage
        }
    )

    rf = model._prepare_risk_free(df, annualization=12)

    # 3% annual = 3/12 = 0.25% monthly = 0.0025
    assert abs(rf.iloc[0]["rf"] - 0.0025) < 1e-6
    assert abs(rf.iloc[1]["rf"] - (3.5 / 100 / 12)) < 1e-6


def test_merge_data(model):
    """Test merging equity, market, and risk-free data."""
    equity_returns = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=3, freq="M"),
            "symbol": ["AAPL"] * 3,
            "return": [0.05, -0.02, 0.03],
        }
    )

    market_returns = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=3, freq="M"),
            "symbol": ["SPY"] * 3,  # Will be dropped
            "return": [0.04, -0.01, 0.025],
        }
    )

    rf_returns = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=3, freq="M"),
            "rf": [0.001, 0.001, 0.0015],
        }
    )

    merged = model._merge_data(equity_returns, market_returns, rf_returns)

    assert len(merged) == 3
    assert "market_return" in merged.columns
    assert "rf" in merged.columns
    assert "symbol" in merged.columns  # Equity symbol preserved
    assert merged["symbol"].iloc[0] == "AAPL"


# ============================================================================
# Unit Tests - Beta Estimation
# ============================================================================


def test_estimate_beta_basic(model):
    """Test basic beta estimation."""
    # Create synthetic data: y = 0.01 + 1.2 * x + noise
    np.random.seed(42)
    x = np.random.normal(0, 0.02, 100)
    y = 0.01 + 1.2 * x + np.random.normal(0, 0.01, 100)

    stats = model._estimate_beta(y, x, hac_lags=6)

    assert stats is not None
    assert "alpha" in stats
    assert "beta" in stats
    assert "r_squared" in stats

    # Beta should be close to 1.2
    assert abs(stats["beta"] - 1.2) < 0.2  # Allow some deviation due to noise
    assert stats["r_squared"] > 0.5  # Should have decent fit


def test_estimate_beta_zero_beta(model):
    """Test beta estimation when beta is zero (no market exposure)."""
    np.random.seed(42)
    x = np.random.normal(0, 0.02, 100)
    y = np.random.normal(0.005, 0.01, 100)  # Independent of x

    stats = model._estimate_beta(y, x, hac_lags=6)

    assert stats is not None
    assert abs(stats["beta"]) < 0.3  # Should be close to zero
    assert stats["r_squared"] < 0.1  # Low R² expected


def test_estimate_beta_perfect_correlation(model):
    """Test beta estimation with perfect correlation."""
    x = np.linspace(-0.05, 0.05, 100)
    y = 0.5 * x  # Perfect linear relationship, beta = 0.5

    stats = model._estimate_beta(y, x, hac_lags=6)

    assert stats is not None
    assert abs(stats["beta"] - 0.5) < 1e-6
    assert stats["r_squared"] > 0.99


# ============================================================================
# Unit Tests - Regression Modes
# ============================================================================


def test_full_sample_regression(model):
    """Test full-sample regression."""
    # Create synthetic panel data
    dates = pd.date_range("2020-01-31", periods=50, freq="M")
    symbols = ["AAPL", "MSFT"]

    data = []
    for symbol in symbols:
        for date in dates:
            data.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "excess_return": np.random.normal(0.01, 0.05),
                    "excess_market": np.random.normal(0.008, 0.04),
                }
            )

    df = pd.DataFrame(data)

    results = model._full_sample_regression(df, min_obs=24, hac_lags=6)

    assert len(results) == 2  # One estimate per symbol
    assert "symbol" in results.columns
    assert "beta" in results.columns
    assert "alpha" in results.columns
    assert "observations" in results.columns
    assert results["observations"].min() == 50  # All observations used


def test_rolling_regression(model):
    """Test rolling window regression."""
    # Create synthetic data
    dates = pd.date_range("2020-01-31", periods=100, freq="M")

    data = []
    for date in dates:
        data.append(
            {
                "date": date,
                "symbol": "AAPL",
                "excess_return": np.random.normal(0.01, 0.05),
                "excess_market": np.random.normal(0.008, 0.04),
            }
        )

    df = pd.DataFrame(data)

    results = model._rolling_regression(df, window=60, min_obs=24, hac_lags=6)

    # Should have estimates from observation 24 onwards (when window >= min_obs)
    assert len(results) > 0
    assert "symbol" in results.columns
    assert "beta" in results.columns
    assert "window_start" in results.columns
    assert "window_end" in results.columns

    # Check window sizes
    assert results["observations"].max() == 60  # Window size
    assert results["observations"].min() >= 24  # Min observations


# ============================================================================
# Unit Tests - Edge Cases
# ============================================================================


def test_insufficient_data(model):
    """Test regression with insufficient data."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=10, freq="M"),
            "symbol": ["AAPL"] * 10,
            "excess_return": np.random.normal(0, 0.05, 10),
            "excess_market": np.random.normal(0, 0.04, 10),
        }
    )

    results = model._full_sample_regression(df, min_obs=24, hac_lags=6)

    # Should return empty DataFrame (insufficient observations)
    assert len(results) == 0


def test_estimate_beta_with_nans(model):
    """Test beta estimation handles NaN values gracefully."""
    x = np.array([0.01, 0.02, np.nan, 0.04, 0.05])
    y = np.array([0.015, 0.025, 0.03, np.nan, 0.055])

    # Should fail gracefully (return None)
    stats = model._estimate_beta(y, x, hac_lags=6)

    # Regression will fail due to NaN values
    assert stats is None or np.isnan(stats["beta"])


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
def test_model_with_mock_inputs(package_dir, storage_infrastructure):
    """Test model with mock input data."""
    # Create mock data
    dates = pd.date_range("2020-01-31", periods=60, freq="M")

    equity_df = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "symbol": ["AAPL"] * 60 + ["MSFT"] * 60,
            "adj_close": np.random.normal(100, 10, 120).cumsum(),
        }
    )

    market_df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "SPY",
            "adj_close": np.random.normal(300, 30, 60).cumsum(),
        }
    )

    rf_df = pd.DataFrame({"date": dates, "rate": np.random.uniform(2.0, 4.0, 60)})

    # Create mock loader with required attributes
    class MockLoader:
        def __init__(self):
            self.registry = storage_infrastructure["registry"]
            self.backend = storage_infrastructure["backend"]
            self.resolver = storage_infrastructure["resolver"]

    # Create model with mock inputs
    model = MarketBetaModel(
        package_dir=package_dir,
        loader=MockLoader(),
        writer=storage_infrastructure["writer"],
        overrides={"window": 36, "min_observations": 24},
    )

    # Inject mock inputs
    model.inputs = {
        "equity_prices": equity_df,
        "market_prices": market_df,
        "risk_free": rf_df,
    }

    # Run model
    results = model.run_impl()

    assert results is not None
    assert len(results) > 0
    assert "beta" in results.columns
    assert "alpha" in results.columns
    assert results["symbol"].nunique() == 2


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use MarketBetaModel.

    This is not a test, but serves as documentation.
    """
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.engine.processed_writer import ProcessedWriterBase
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.pathing import PathResolver
    from qx.storage.table_format import TableFormatAdapter

    # Setup infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()
    writer = ProcessedWriterBase(adapter=adapter, resolver=resolver, registry=registry)

    # Create mock loader
    class MockLoader:
        pass

    # Create model
    model = MarketBetaModel(
        package_dir="qx_models/market_beta",
        loader=MockLoader(),
        writer=writer,
        overrides={"window": 60},
    )

    # Model will auto-load inputs from curated storage
    # results = model.run()

    print(f"✅ Created market beta model: {model.info['id']}")


if __name__ == "__main__":
    # Run example
    example_usage()
