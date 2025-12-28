"""Unit tests for Portfolio Backtest Model."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.pathing import PathResolver
from qx.storage.processed_writer import ProcessedWriterBase
from qx.storage.typed_loader import TypedCuratedLoader

from .model import PortfolioBacktestModel

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def storage_infrastructure():
    """Setup storage infrastructure for tests."""
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()
    loader = TypedCuratedLoader(backend, registry, resolver)
    writer = ProcessedWriterBase(backend, resolver)

    return {
        "registry": registry,
        "backend": backend,
        "resolver": resolver,
        "loader": loader,
        "writer": writer,
    }


@pytest.fixture
def package_dir():
    """Get package directory path."""
    return str(Path(__file__).parent)


@pytest.fixture
def model(package_dir, storage_infrastructure):
    """Create model instance."""
    return PortfolioBacktestModel(
        package_dir=package_dir,
        loader=storage_infrastructure["loader"],
        writer=storage_infrastructure["writer"],
    )


@pytest.fixture
def sample_portfolio_weights():
    """Create sample portfolio weights (time series)."""
    dates = pd.date_range("2020-01-31", "2020-12-31", freq="M")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    data = []
    for date in dates:
        # Random weights that sum to 1
        weights = np.random.dirichlet(np.ones(len(symbols)))
        for symbol, weight in zip(symbols, weights):
            data.append(
                {
                    "symbol": symbol,
                    "optimization_date": date,
                    "weight": weight,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_prices():
    """Create sample price data (monthly)."""
    dates = pd.date_range("2019-12-31", "2021-01-31", freq="M")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    data = []
    for symbol in symbols:
        # Simulate random walk prices
        price = 100.0
        for date in dates:
            price = price * (1 + np.random.normal(0.01, 0.05))  # 1% drift, 5% vol
            data.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "adj_close": price,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1_000_000,
                }
            )

    return pd.DataFrame(data)


# ============================================================================
# Unit Tests - Initialization
# ============================================================================


def test_model_initialization(model):
    """Test that model initializes correctly from YAML."""
    assert model is not None
    assert model.info["id"] == "portfolio_backtest_model"
    assert model.info["version"] == "1.0.0"


def test_configuration_loading(model):
    """Test YAML configuration is loaded properly."""
    assert hasattr(model, "params")
    assert "transaction_cost_bps" in model.params
    assert model.params["transaction_cost_bps"] == 10.0
    assert model.params["initial_capital"] == 1_000_000.0


# ============================================================================
# Unit Tests - Data Preparation
# ============================================================================


def test_prepare_weights(model, sample_portfolio_weights):
    """Test portfolio weight preparation."""
    weights_by_date = model._prepare_weights(
        sample_portfolio_weights, min_threshold=0.01
    )

    assert len(weights_by_date) == 12  # Monthly for 2020
    assert all(isinstance(w, pd.Series) for w in weights_by_date.values())

    # Check normalization
    for date, weights in weights_by_date.items():
        assert abs(weights.sum() - 1.0) < 1e-6  # Sum to 1.0


def test_prepare_prices(model, sample_prices):
    """Test price panel preparation."""
    price_panel = model._prepare_prices(sample_prices)

    assert isinstance(price_panel, pd.DataFrame)
    assert (
        price_panel.index.name == "date" or price_panel.index.dtype == "datetime64[ns]"
    )
    assert "AAPL" in price_panel.columns
    assert len(price_panel.columns) == 5  # 5 symbols


def test_prepare_weights_filtering(model, sample_portfolio_weights):
    """Test that small weights are filtered out."""
    weights_by_date = model._prepare_weights(
        sample_portfolio_weights, min_threshold=0.15
    )

    # With high threshold, should have fewer positions
    avg_positions = np.mean([len(w) for w in weights_by_date.values()])
    assert avg_positions < 5  # Less than all 5 symbols


# ============================================================================
# Unit Tests - Backtest Simulation
# ============================================================================


def test_run_backtest_basic(model, sample_portfolio_weights, sample_prices):
    """Test basic backtest execution."""
    weights_by_date = model._prepare_weights(
        sample_portfolio_weights, min_threshold=0.01
    )
    price_panel = model._prepare_prices(sample_prices)

    backtest_df = model._run_backtest(
        weights_by_date=weights_by_date,
        price_panel=price_panel,
        initial_capital=1_000_000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    assert len(backtest_df) > 0
    assert "portfolio_value" in backtest_df.columns
    assert "portfolio_return" in backtest_df.columns
    assert "turnover" in backtest_df.columns
    assert "transaction_cost" in backtest_df.columns


def test_backtest_initial_value(model, sample_portfolio_weights, sample_prices):
    """Test that backtest starts with correct initial value."""
    weights_by_date = model._prepare_weights(
        sample_portfolio_weights, min_threshold=0.01
    )
    price_panel = model._prepare_prices(sample_prices)

    initial_capital = 1_000_000.0
    backtest_df = model._run_backtest(
        weights_by_date=weights_by_date,
        price_panel=price_panel,
        initial_capital=initial_capital,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    # First value should be close to initial capital (after transaction costs)
    first_value = backtest_df.iloc[0]["portfolio_value"]
    assert abs(first_value - initial_capital) / initial_capital < 0.01  # Within 1%


def test_backtest_transaction_costs(model, sample_portfolio_weights, sample_prices):
    """Test that transaction costs are applied correctly."""
    weights_by_date = model._prepare_weights(
        sample_portfolio_weights, min_threshold=0.01
    )
    price_panel = model._prepare_prices(sample_prices)

    backtest_df = model._run_backtest(
        weights_by_date=weights_by_date,
        price_panel=price_panel,
        initial_capital=1_000_000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    # Check that rebalance dates have transaction costs
    rebalance_rows = backtest_df[backtest_df["is_rebalance"]]
    assert (rebalance_rows["transaction_cost"] > 0).all()

    # Non-rebalance dates should have zero transaction costs
    non_rebalance_rows = backtest_df[~backtest_df["is_rebalance"]]
    assert (non_rebalance_rows["transaction_cost"] == 0).all()


# ============================================================================
# Unit Tests - Performance Metrics
# ============================================================================


def test_calculate_performance_metrics(model, sample_portfolio_weights, sample_prices):
    """Test performance metrics calculation."""
    weights_by_date = model._prepare_weights(
        sample_portfolio_weights, min_threshold=0.01
    )
    price_panel = model._prepare_prices(sample_prices)

    backtest_df = model._run_backtest(
        weights_by_date=weights_by_date,
        price_panel=price_panel,
        initial_capital=1_000_000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    performance_df = model._calculate_performance_metrics(
        backtest_df, calculate_drawdowns=True
    )

    assert "cumulative_return" in performance_df.columns
    assert "rolling_sharpe_12m" in performance_df.columns
    assert "drawdown" in performance_df.columns
    assert "max_drawdown_to_date" in performance_df.columns


def test_cumulative_return_calculation(model, sample_portfolio_weights, sample_prices):
    """Test that cumulative return is calculated correctly."""
    weights_by_date = model._prepare_weights(
        sample_portfolio_weights, min_threshold=0.01
    )
    price_panel = model._prepare_prices(sample_prices)

    backtest_df = model._run_backtest(
        weights_by_date=weights_by_date,
        price_panel=price_panel,
        initial_capital=1_000_000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    performance_df = model._calculate_performance_metrics(backtest_df)

    # First cumulative return should be ~0
    assert abs(performance_df.iloc[0]["cumulative_return"]) < 0.01

    # Cumulative return should match total return
    total_return = (
        performance_df.iloc[-1]["portfolio_value"]
        / performance_df.iloc[0]["portfolio_value"]
        - 1
    )
    assert abs(performance_df.iloc[-1]["cumulative_return"] - total_return) < 1e-6


def test_drawdown_calculation(model, sample_portfolio_weights, sample_prices):
    """Test drawdown calculation."""
    weights_by_date = model._prepare_weights(
        sample_portfolio_weights, min_threshold=0.01
    )
    price_panel = model._prepare_prices(sample_prices)

    backtest_df = model._run_backtest(
        weights_by_date=weights_by_date,
        price_panel=price_panel,
        initial_capital=1_000_000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    performance_df = model._calculate_performance_metrics(backtest_df)

    # Drawdown should always be <= 0
    assert (performance_df["drawdown"] <= 0).all()

    # Max drawdown to date should be monotonically decreasing (or equal)
    max_dd = performance_df["max_drawdown_to_date"]
    assert (max_dd.diff().dropna() <= 0).all()


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
def test_full_backtest(model, sample_portfolio_weights, sample_prices):
    """Test complete backtest pipeline."""
    inputs = {
        "portfolio_weights": sample_portfolio_weights,
        "equity_prices": sample_prices,
    }

    params = {
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
        "initial_capital": 1_000_000.0,
        "min_weight_threshold": 0.01,
        "calculate_drawdowns": True,
    }

    result_df = model.run_impl(inputs, params)

    assert result_df is not None
    assert len(result_df) > 0
    assert "portfolio_value" in result_df.columns
    assert "cumulative_return" in result_df.columns
    assert "sharpe" in result_df.columns or "rolling_sharpe_12m" in result_df.columns


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """
    Example of how to use PortfolioBacktestModel.
    """
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.pathing import PathResolver
    from qx.storage.processed_writer import ProcessedWriterBase
    from qx.storage.typed_loader import TypedCuratedLoader

    # Setup infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()
    loader = TypedCuratedLoader(backend, registry, resolver)
    writer = ProcessedWriterBase(backend, resolver)

    # Create model
    model = PortfolioBacktestModel(
        package_dir="qx_models/portfolio_backtest",
        loader=loader,
        writer=writer,
    )

    # Prepare inputs (would come from DAG)
    inputs = {
        "portfolio_weights": pd.DataFrame(),  # From Markowitz model
        "equity_prices": pd.DataFrame(),  # From builder
    }

    # Run backtest
    result = model.run_impl(inputs, model.params)

    print(f"✅ Backtest complete: {len(result)} observations")


if __name__ == "__main__":
    example_usage()
