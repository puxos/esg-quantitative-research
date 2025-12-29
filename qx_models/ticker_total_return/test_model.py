"""
Unit tests for Ticker Total Return Model
"""

from pathlib import Path

import pandas as pd
import pytest

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.engine.processed_writer import ProcessedWriterBase
from qx.foundation.typed_curated_loader import TypedCuratedLoader
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter

from .model import TickerTotalReturnModel


@pytest.fixture
def storage_infrastructure():
    """Setup storage infrastructure."""
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()

    writer = ProcessedWriterBase(adapter=adapter, resolver=resolver, registry=registry)
    loader = TypedCuratedLoader(backend=backend, registry=registry, resolver=resolver)

    return {
        "registry": registry,
        "backend": backend,
        "writer": writer,
        "loader": loader,
    }


@pytest.fixture
def package_dir():
    """Get package directory."""
    return str(Path(__file__).parent)


@pytest.fixture
def model(package_dir, storage_infrastructure):
    """Create model instance."""
    return TickerTotalReturnModel(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        loader=storage_infrastructure["loader"],
    )


@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    # AAPL: +50% return over year
    aapl = pd.DataFrame(
        {
            "symbol": "AAPL",
            "date": dates,
            "adj_close": 100 * (1 + 0.5 * (dates - dates[0]).days / 365),
        }
    )

    # MSFT: -20% return over year
    msft = pd.DataFrame(
        {
            "symbol": "MSFT",
            "date": dates,
            "adj_close": 100 * (1 - 0.2 * (dates - dates[0]).days / 365),
        }
    )

    # GOOGL: Flat
    googl = pd.DataFrame(
        {"symbol": "GOOGL", "date": dates, "adj_close": [100.0] * len(dates)}
    )

    return pd.concat([aapl, msft, googl], ignore_index=True)


def test_model_initialization(model):
    """Test model initializes correctly."""
    assert model is not None
    assert model.info["id"] == "ticker_total_return_model"


def test_basic_return_calculation(model, sample_price_data):
    """Test basic return calculation."""
    inputs = {"equity_prices": sample_price_data}
    params = {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "periods_per_year": 252,
    }

    result = model.run_impl(inputs, params)

    assert len(result) == 3
    assert set(result["symbol"].values) == {"AAPL", "MSFT", "GOOGL"}

    # Check AAPL (+50%)
    aapl_row = result[result["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["total_return"] == pytest.approx(0.5, rel=0.01)
    assert aapl_row["total_return_pct"] == pytest.approx(50.0, rel=0.01)

    # Check MSFT (-20%)
    msft_row = result[result["symbol"] == "MSFT"].iloc[0]
    assert msft_row["total_return"] == pytest.approx(-0.2, rel=0.01)

    # Check GOOGL (flat)
    googl_row = result[result["symbol"] == "GOOGL"].iloc[0]
    assert googl_row["total_return"] == pytest.approx(0.0, abs=0.001)


def test_date_range_filtering(model, sample_price_data):
    """Test date range filtering."""
    inputs = {"equity_prices": sample_price_data}
    params = {
        "symbols": ["AAPL"],
        "start_date": "2020-06-01",
        "end_date": "2020-09-30",
        "periods_per_year": 252,
    }

    result = model.run_impl(inputs, params)

    assert len(result) == 1
    aapl_row = result.iloc[0]
    assert aapl_row["start_date"] >= pd.Timestamp("2020-06-01")
    assert aapl_row["end_date"] <= pd.Timestamp("2020-09-30")


def test_min_observations_filter(model):
    """Test minimum observations filtering."""
    # Create data with only 1 observation
    sparse_data = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2020-01-01")],
            "adj_close": [100.0],
        }
    )

    inputs = {"equity_prices": sparse_data}
    params = {"min_observations": 2}

    result = model.run_impl(inputs, params)

    # Should return empty DataFrame (< min_observations)
    assert len(result) == 0


def test_volatility_calculation(model):
    """Test volatility calculation with volatile data."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    # Create volatile price series
    import numpy as np

    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02))

    volatile_data = pd.DataFrame({"symbol": "VOL", "date": dates, "adj_close": prices})

    inputs = {"equity_prices": volatile_data}
    params = {"periods_per_year": 252}

    result = model.run_impl(inputs, params)

    assert len(result) == 1
    vol_row = result.iloc[0]

    # Volatility should be positive
    assert vol_row["volatility"] > 0
    assert vol_row["volatility_pct"] > 0


def test_sharpe_ratio(model, sample_price_data):
    """Test Sharpe ratio calculation."""
    inputs = {"equity_prices": sample_price_data}
    params = {"symbols": ["AAPL"], "periods_per_year": 252}

    result = model.run_impl(inputs, params)

    aapl_row = result.iloc[0]

    # Sharpe = annualized_return / volatility
    expected_sharpe = aapl_row["annualized_return"] / aapl_row["volatility"]
    assert aapl_row["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=0.01)


def test_max_drawdown(model):
    """Test maximum drawdown calculation."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    # Create price series with clear drawdown
    # Goes from 100 → 150 → 75 → 120
    prices = [100.0] * 90 + [150.0] * 90 + [75.0] * 90 + [120.0] * (len(dates) - 270)

    dd_data = pd.DataFrame({"symbol": "DD", "date": dates, "adj_close": prices})

    inputs = {"equity_prices": dd_data}
    params = {"periods_per_year": 252}

    result = model.run_impl(inputs, params)

    dd_row = result.iloc[0]

    # Max drawdown should be (75 - 150) / 150 = -50%
    assert dd_row["max_drawdown"] == pytest.approx(-0.5, rel=0.01)
    assert dd_row["max_drawdown_pct"] == pytest.approx(-50.0, rel=0.01)


def test_missing_adj_close_fallback(model):
    """Test fallback to 'close' when 'adj_close' missing."""
    data = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 10,
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "close": [100 + i for i in range(10)],  # No adj_close
        }
    )

    inputs = {"equity_prices": data}
    params = {"periods_per_year": 252}

    result = model.run_impl(inputs, params)

    # Should work with 'close' column
    assert len(result) == 1
    assert result.iloc[0]["total_return"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
