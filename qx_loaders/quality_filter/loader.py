"""
Quality Filter Loader

Filters stocks based on quality metrics to exclude penny stocks, distressed
companies, and stocks with poor data quality.

Quality Checks:
- Minimum price: Exclude penny stocks (< $5)
- Maximum volatility: Exclude extremely volatile stocks (> 10% daily vol)
- Minimum trading days: Exclude stocks with sparse data
- Data availability: Exclude stocks missing in recent period

Usage:
    In DAG:
        Task(
            id="FilterQuality",
            run=run_loader(
                loader_module="qx_loaders.quality_filter",
                loader_class="QualityFilterLoader",
                overrides={"filter_date": "2024-12-31"}
            ),
            deps=["LoadOHLCVPanel"]
        )

    Pass filtered symbols to downstream tasks (portfolio optimization).
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from qx.foundation.base_loader import BaseLoader


class QualityFilterLoader(BaseLoader):
    """Filter stocks by quality metrics."""

    def load_impl(self) -> List[str]:
        """
        Apply quality filters to stock universe.

        Returns:
            List[str]: Symbols passing all quality filters
        """
        # Load equity prices
        prices_df = self.load_input("equity_prices")

        # Get parameters
        filter_date = pd.Timestamp(self.params["filter_date"])
        lookback_months = self.params["lookback_months"]
        min_price = self.params["min_price"]
        max_volatility = self.params["max_volatility"]
        min_trading_days = self.params["min_trading_days"]

        # Calculate lookback window
        start_date = filter_date - pd.DateOffset(months=lookback_months)

        print(f"\n{'='*70}")
        print(f"Quality Filtering ({filter_date.date()}):")
        print(f"  Lookback: {start_date.date()} to {filter_date.date()}")
        print(f"  Min price: ${min_price:.2f}")
        print(
            f"  Max volatility: {max_volatility*100:.1f}% daily ({max_volatility*np.sqrt(252)*100:.0f}% annual)"
        )
        print(f"  Min trading days: {min_trading_days}")
        print(f"{'='*70}\n")

        # Filter to lookback period
        window_df = prices_df[
            (prices_df["date"] >= start_date) & (prices_df["date"] <= filter_date)
        ].copy()

        initial_symbols = prices_df["symbol"].nunique()
        print(f"Initial universe: {initial_symbols} symbols")

        if len(window_df) == 0:
            print("⚠️ WARNING: No data in lookback window!")
            return []

        # Calculate quality metrics by symbol
        quality_metrics = []

        for symbol, group in window_df.groupby("symbol"):
            # Trading days check
            trading_days = len(group)

            # Price check (median price)
            median_price = group["close"].median()

            # Volatility check (annualized daily return std)
            if len(group) > 1:
                returns = group["close"].pct_change().dropna()
                daily_vol = returns.std()
            else:
                daily_vol = np.nan

            quality_metrics.append(
                {
                    "symbol": symbol,
                    "trading_days": trading_days,
                    "median_price": median_price,
                    "daily_vol": daily_vol,
                }
            )

        metrics_df = pd.DataFrame(quality_metrics)

        # Apply filters
        passed = metrics_df.copy()

        # Filter 1: Minimum trading days
        failed_trading = passed["trading_days"] < min_trading_days
        print(
            f"Filter 1 (min {min_trading_days} trading days): "
            f"{failed_trading.sum()} failed, {(~failed_trading).sum()} passed"
        )
        passed = passed[~failed_trading]

        # Filter 2: Minimum price
        failed_price = passed["median_price"] < min_price
        print(
            f"Filter 2 (min ${min_price:.2f} price): "
            f"{failed_price.sum()} failed, {(~failed_price).sum()} passed"
        )
        passed = passed[~failed_price]

        # Filter 3: Maximum volatility
        failed_vol = passed["daily_vol"] > max_volatility
        print(
            f"Filter 3 (max {max_volatility*100:.1f}% daily vol): "
            f"{failed_vol.sum()} failed, {(~failed_vol).sum()} passed"
        )
        passed = passed[~failed_vol]

        # Final list
        filtered_symbols = sorted(passed["symbol"].tolist())

        print(f"\n✅ Quality Filter Complete:")
        print(f"   Started with: {initial_symbols} symbols")
        print(
            f"   Passed filters: {len(filtered_symbols)} symbols ({len(filtered_symbols)/initial_symbols*100:.1f}%)"
        )
        print(f"   Filtered out: {initial_symbols - len(filtered_symbols)} symbols")

        # Show some examples of filtered out stocks
        failed_symbols = set(metrics_df["symbol"]) - set(filtered_symbols)
        if failed_symbols:
            failed_df = metrics_df[metrics_df["symbol"].isin(failed_symbols)].copy()

            # Categorize failures
            penny_stocks = failed_df[failed_df["median_price"] < min_price][
                "symbol"
            ].tolist()
            high_vol = failed_df[failed_df["daily_vol"] > max_volatility][
                "symbol"
            ].tolist()
            sparse_data = failed_df[failed_df["trading_days"] < min_trading_days][
                "symbol"
            ].tolist()

            print(f"\nFiltered out categories:")
            if penny_stocks:
                print(
                    f"   Penny stocks (<${min_price}): {len(penny_stocks)} - {', '.join(penny_stocks[:5])}"
                    + ("..." if len(penny_stocks) > 5 else "")
                )
            if high_vol:
                print(
                    f"   High volatility (>{max_volatility*100:.1f}%): {len(high_vol)} - {', '.join(high_vol[:5])}"
                    + ("..." if len(high_vol) > 5 else "")
                )
            if sparse_data:
                print(
                    f"   Sparse data (<{min_trading_days} days): {len(sparse_data)} - {', '.join(sparse_data[:5])}"
                    + ("..." if len(sparse_data) > 5 else "")
                )

        return filtered_symbols


def example_usage():
    """Example of using QualityFilterLoader."""
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.pathing import PathResolver
    from qx.storage.typed_loader import TypedCuratedLoader

    # Setup infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)
    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()
    loader_infra = TypedCuratedLoader(backend, registry, resolver)

    # Create loader
    loader = QualityFilterLoader(
        package_dir=str(Path(__file__).parent),
        loader=loader_infra,
        overrides={"filter_date": "2024-12-31"},
    )

    # Load
    symbols = loader.load()
    print(f"\n✅ Quality-filtered universe: {len(symbols)} symbols")
    print(f"Examples: {', '.join(symbols[:10])}")


if __name__ == "__main__":
    example_usage()
