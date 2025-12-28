"""
Portfolio Backtest Model Implementation

Backtests time series of portfolio weights using actual OHLCV data to calculate:
    - Realized portfolio returns (daily/monthly/cumulative)
    - Transaction costs and turnover
    - Performance metrics (Sharpe, Sortino, max drawdown)
    - Rolling statistics (volatility, tracking error)
    - Attribution analysis
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from qx.engine.base_model import BaseModel

logger = logging.getLogger(__name__)


class PortfolioBacktestModel(BaseModel):
    """
    Portfolio Backtest with Transaction Costs and Performance Attribution

    Takes a time series of portfolio weights and calculates:
        - Realized returns using actual OHLCV data
        - Turnover and transaction costs
        - Performance metrics (Sharpe, Sortino, Calmar ratios)
        - Maximum drawdown and recovery periods
        - Rolling volatility and correlations
        - Monthly/annual returns table

    Key Features:
        - Handles corporate actions (splits, dividends via adjusted close)
        - Transaction cost modeling (bps + slippage)
        - Rebalancing logic (monthly/quarterly/annual)
        - Survivorship bias handling (missing prices)
        - Performance attribution (factor exposures optional)
    """

    def run_impl(
        self,
        inputs: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Run backtest on portfolio time series.

        Args:
            inputs: Dictionary of input DataFrames
                - portfolio_weights: Optimizer output (symbol, optimization_date, weight)
                - equity_prices: Monthly OHLCV data (symbol, date, adj_close, ...)
            params: Model parameters from model.yaml
            **kwargs: Additional arguments

        Returns:
            DataFrame with backtest results:
                - date: Observation date
                - portfolio_value: Portfolio value after rebalancing
                - portfolio_return: Period return (%)
                - turnover: Two-way turnover (%)
                - transaction_cost: Cost in $ and bps
                - cumulative_return: Cumulative return since inception
                - sharpe_ratio: Rolling Sharpe (annualized)
                - max_drawdown: Maximum drawdown to date
                - positions_count: Number of active positions
        """
        logger.info("Starting Portfolio Backtest")

        # Extract inputs
        portfolio_weights_df = inputs["portfolio_weights"]
        equity_prices_df = inputs["equity_prices"]

        # DEBUG: Check what we received
        logger.info(f"DEBUG: portfolio_weights_df shape: {portfolio_weights_df.shape}")
        logger.info(
            f"DEBUG: portfolio_weights_df columns: {portfolio_weights_df.columns.tolist()}"
        )
        logger.info(f"DEBUG: equity_prices_df shape: {equity_prices_df.shape}")
        logger.info(
            f"DEBUG: equity_prices_df columns: {equity_prices_df.columns.tolist()}"
        )
        logger.info(f"DEBUG: equity_prices sample:\n{equity_prices_df.head()}")

        # Extract parameters
        transaction_cost_bps = params.get("transaction_cost_bps", 10.0)
        slippage_bps = params.get("slippage_bps", 5.0)
        initial_capital = params.get("initial_capital", 1_000_000.0)
        min_weight_threshold = params.get("min_weight_threshold", 0.001)
        calculate_drawdowns = params.get("calculate_drawdowns", True)

        logger.info(f"Parameters:")
        logger.info(f"  Initial capital: ${initial_capital:,.0f}")
        logger.info(f"  Transaction cost: {transaction_cost_bps} bps")
        logger.info(f"  Slippage: {slippage_bps} bps")
        logger.info(f"  Min weight threshold: {min_weight_threshold*100:.2f}%")

        # 1. Prepare portfolio weights
        logger.info("\nStep 1: Preparing portfolio weights")
        weights_by_date = self._prepare_weights(
            portfolio_weights_df, min_weight_threshold
        )

        # 2. Prepare price data
        logger.info("Step 2: Preparing price data")
        price_panel = self._prepare_prices(equity_prices_df)

        # 3. Run backtest
        logger.info("Step 3: Running backtest simulation")
        backtest_results = self._run_backtest(
            weights_by_date=weights_by_date,
            price_panel=price_panel,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )

        # 4. Calculate performance metrics
        logger.info("Step 4: Calculating performance metrics")
        performance_df = self._calculate_performance_metrics(
            backtest_results,
            calculate_drawdowns=calculate_drawdowns,
        )

        # 5. Generate summary statistics
        self._log_performance_summary(performance_df)

        return performance_df

    def _prepare_weights(
        self,
        portfolio_weights_df: pd.DataFrame,
        min_threshold: float,
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        Prepare portfolio weights dictionary keyed by date.

        Args:
            portfolio_weights_df: Optimizer output
            min_threshold: Minimum weight to consider active

        Returns:
            Dictionary: {date: Series(symbol -> weight)}
        """
        df = portfolio_weights_df.copy()
        df["optimization_date"] = pd.to_datetime(df["optimization_date"])

        # Filter to active positions
        df = df[df["weight"].abs() >= min_threshold].copy()

        # Ensure optimization_date is Timestamp for consistent lookups
        df["optimization_date"] = pd.to_datetime(df["optimization_date"])

        # Group by date
        weights_by_date = {}
        for date, group in df.groupby("optimization_date"):
            weights = group.set_index("symbol")["weight"]
            # Renormalize to ensure sum = 1.0
            weights = weights / weights.sum()
            weights_by_date[date] = weights

        logger.info(
            f"  Loaded {len(weights_by_date)} rebalancing dates "
            f"({min(weights_by_date.keys())} to {max(weights_by_date.keys())})"
        )
        logger.info(
            f"  Average positions per date: {np.mean([len(w) for w in weights_by_date.values()]):.1f}"
        )

        return weights_by_date

    def _prepare_prices(self, equity_prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare price panel (date x symbol) using adjusted close.

        Args:
            equity_prices_df: OHLCV data

        Returns:
            DataFrame with date index, symbol columns, adj_close values
        """
        df = equity_prices_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Pivot to wide format
        price_panel = df.pivot(index="date", columns="symbol", values="adj_close")
        price_panel = price_panel.sort_index()

        # Ensure index is Timestamp for consistent lookups
        if not isinstance(price_panel.index, pd.DatetimeIndex):
            price_panel.index = pd.to_datetime(price_panel.index)

        logger.info(
            f"  Price panel: {price_panel.shape[0]} dates x {price_panel.shape[1]} symbols"
        )
        logger.info(
            f"  Date range: {price_panel.index.min()} to {price_panel.index.max()}"
        )
        logger.info(
            f"  Missing values: {price_panel.isna().sum().sum()} / {price_panel.size} ({100*price_panel.isna().sum().sum()/price_panel.size:.2f}%)"
        )

        return price_panel

    def _run_backtest(
        self,
        weights_by_date: Dict[pd.Timestamp, pd.Series],
        price_panel: pd.DataFrame,
        initial_capital: float,
        transaction_cost_bps: float,
        slippage_bps: float,
    ) -> pd.DataFrame:
        """
        Simulate portfolio over time with rebalancing and transaction costs.

        Args:
            weights_by_date: Target weights at each rebalancing date
            price_panel: Price data (date x symbol)
            initial_capital: Starting capital
            transaction_cost_bps: One-way transaction cost (bps)
            slippage_bps: Market impact slippage (bps)

        Returns:
            DataFrame with columns:
                - date
                - portfolio_value
                - cash
                - portfolio_return (period)
                - turnover (two-way %)
                - transaction_cost ($)
                - positions_count
                - holdings (dict of symbol: shares)
        """
        rebalance_dates = sorted(weights_by_date.keys())
        all_dates = price_panel.index.tolist()

        # Filter to dates on or after first rebalance
        sim_dates = [d for d in all_dates if d >= rebalance_dates[0]]

        logger.info(f"  Simulation period: {sim_dates[0]} to {sim_dates[-1]}")
        logger.info(f"  Total dates: {len(sim_dates)}")
        logger.info(f"  Rebalance dates: {len(rebalance_dates)}")

        # Initialize state
        portfolio_value = initial_capital
        cash = initial_capital
        holdings = {}  # {symbol: shares}
        prev_weights = pd.Series(dtype=float)  # Empty initially

        results = []

        for i, date in enumerate(sim_dates):
            # Check if rebalancing date
            is_rebalance = date in rebalance_dates

            if is_rebalance:
                target_weights = weights_by_date[date]

                # Liquidate current holdings
                prices_today = price_panel.loc[date]
                liquidation_value = 0.0

                for symbol, shares in holdings.items():
                    if symbol in prices_today and not pd.isna(prices_today[symbol]):
                        sale_proceeds = shares * prices_today[symbol]
                        # Apply transaction cost
                        cost = (
                            sale_proceeds
                            * (transaction_cost_bps + slippage_bps)
                            / 10000.0
                        )
                        liquidation_value += sale_proceeds - cost

                # For first rebalance, use initial capital; otherwise use liquidation value
                if i == 0:
                    cash = initial_capital
                    portfolio_value = initial_capital
                else:
                    cash = liquidation_value
                    portfolio_value = cash

                # Calculate turnover
                if len(prev_weights) > 0:
                    # Align weights to common universe
                    all_symbols = sorted(
                        set(prev_weights.index) | set(target_weights.index)
                    )
                    prev_aligned = prev_weights.reindex(all_symbols, fill_value=0.0)
                    target_aligned = target_weights.reindex(all_symbols, fill_value=0.0)
                    turnover = (prev_aligned - target_aligned).abs().sum()
                else:
                    # First rebalance: 100% turnover
                    turnover = 1.0

                # Allocate to target weights
                new_holdings = {}
                total_transaction_cost = 0.0

                for symbol in target_weights.index:
                    weight = target_weights[symbol]
                    target_value = portfolio_value * weight

                    if symbol in prices_today and not pd.isna(prices_today[symbol]):
                        price = prices_today[symbol]
                        shares = target_value / price

                        # Transaction cost
                        cost = (
                            target_value
                            * (transaction_cost_bps + slippage_bps)
                            / 10000.0
                        )
                        total_transaction_cost += cost

                        new_holdings[symbol] = shares

                cash = 0.0  # Fully invested
                holdings = new_holdings
                portfolio_value -= total_transaction_cost
                prev_weights = target_weights.copy()

                results.append(
                    {
                        "date": date,
                        "portfolio_value": portfolio_value,
                        "cash": cash,
                        "portfolio_return": (
                            np.nan
                            if i == 0
                            else (portfolio_value / results[-1]["portfolio_value"] - 1)
                        ),
                        "turnover": turnover,
                        "transaction_cost": total_transaction_cost,
                        "transaction_cost_bps": (
                            total_transaction_cost / portfolio_value
                        )
                        * 10000,
                        "positions_count": len(holdings),
                        "is_rebalance": True,
                    }
                )

            else:
                # Mark-to-market
                prices_today = price_panel.loc[date]
                market_value = 0.0

                for symbol, shares in holdings.items():
                    if symbol in prices_today and not pd.isna(prices_today[symbol]):
                        market_value += shares * prices_today[symbol]

                prev_value = (
                    results[-1]["portfolio_value"] if results else initial_capital
                )
                portfolio_value = market_value + cash
                period_return = (
                    (portfolio_value / prev_value - 1) if prev_value > 0 else 0.0
                )

                results.append(
                    {
                        "date": date,
                        "portfolio_value": portfolio_value,
                        "cash": cash,
                        "portfolio_return": period_return,
                        "turnover": 0.0,
                        "transaction_cost": 0.0,
                        "transaction_cost_bps": 0.0,
                        "positions_count": len(holdings),
                        "is_rebalance": False,
                    }
                )

        backtest_df = pd.DataFrame(results)
        logger.info(f"  Backtest complete: {len(backtest_df)} observations")
        logger.info(
            f"  Final portfolio value: ${backtest_df.iloc[-1]['portfolio_value']:,.0f}"
        )
        logger.info(
            f"  Total return: {(backtest_df.iloc[-1]['portfolio_value']/initial_capital - 1)*100:.2f}%"
        )

        return backtest_df

    def _calculate_performance_metrics(
        self,
        backtest_df: pd.DataFrame,
        calculate_drawdowns: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics.

        Args:
            backtest_df: Backtest results from simulation
            calculate_drawdowns: Whether to calculate drawdown metrics

        Returns:
            DataFrame with performance metrics added
        """
        df = backtest_df.copy()

        # 1. Cumulative returns
        df["cumulative_return"] = (
            df["portfolio_value"] / df["portfolio_value"].iloc[0] - 1
        )

        # 2. Rolling volatility (12-month)
        returns = df["portfolio_return"].dropna()
        df["rolling_vol_12m"] = returns.rolling(12).std() * np.sqrt(12)

        # 3. Rolling Sharpe ratio (12-month, assume rf=0 for simplicity)
        df["rolling_sharpe_12m"] = (
            returns.rolling(12).mean() * 12 / (returns.rolling(12).std() * np.sqrt(12))
        )

        # 4. Drawdown calculations
        if calculate_drawdowns:
            cumulative_max = df["portfolio_value"].cummax()
            df["drawdown"] = (df["portfolio_value"] - cumulative_max) / cumulative_max
            df["max_drawdown_to_date"] = df["drawdown"].cummin()
        else:
            df["drawdown"] = np.nan
            df["max_drawdown_to_date"] = np.nan

        # 5. Cumulative turnover and transaction costs
        df["cumulative_turnover"] = df["turnover"].cumsum()
        df["cumulative_transaction_cost"] = df["transaction_cost"].cumsum()

        logger.info(f"  Performance metrics calculated")
        logger.info(f"  Total return: {df['cumulative_return'].iloc[-1]*100:.2f}%")
        logger.info(
            f"  Annualized volatility: {returns.std() * np.sqrt(12) * 100:.2f}%"
        )
        logger.info(
            f"  Sharpe ratio: {(returns.mean() * 12) / (returns.std() * np.sqrt(12)):.3f}"
        )
        logger.info(f"  Max drawdown: {df['max_drawdown_to_date'].min()*100:.2f}%")
        logger.info(f"  Total turnover: {df['cumulative_turnover'].iloc[-1]*100:.1f}%")
        logger.info(
            f"  Total transaction costs: ${df['cumulative_transaction_cost'].iloc[-1]:,.0f}"
        )

        return df

    def _log_performance_summary(self, performance_df: pd.DataFrame) -> None:
        """
        Log comprehensive performance summary.

        Args:
            performance_df: Performance metrics DataFrame
        """
        df = performance_df
        returns = df["portfolio_return"].dropna()

        # Overall statistics
        total_return = df["cumulative_return"].iloc[-1]
        n_months = len(df)
        n_years = n_months / 12
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        # Risk metrics
        monthly_vol = returns.std()
        annual_vol = monthly_vol * np.sqrt(12)
        sharpe = (returns.mean() * 12) / annual_vol if annual_vol > 0 else 0.0

        # Downside metrics
        downside_returns = returns[returns < 0]
        sortino = (
            (returns.mean() * 12) / (downside_returns.std() * np.sqrt(12))
            if len(downside_returns) > 0
            else 0.0
        )
        max_dd = df["max_drawdown_to_date"].min()

        # Transaction metrics
        total_turnover = df["cumulative_turnover"].iloc[-1]
        total_tx_cost = df["cumulative_transaction_cost"].iloc[-1]
        avg_tx_cost_bps = df[df["is_rebalance"]]["transaction_cost_bps"].mean()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns)

        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST PERFORMANCE SUMMARY")
        logger.info("=" * 70)
        logger.info(
            f"Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        )
        logger.info(f"Duration: {n_months} months ({n_years:.1f} years)")
        logger.info(f"Rebalances: {df['is_rebalance'].sum()}")
        logger.info("")
        logger.info("RETURNS:")
        logger.info(f"  Total Return:        {total_return*100:>8.2f}%")
        logger.info(f"  Annualized Return:   {annualized_return*100:>8.2f}%")
        logger.info(f"  Best Month:          {returns.max()*100:>8.2f}%")
        logger.info(f"  Worst Month:         {returns.min()*100:>8.2f}%")
        logger.info(f"  Win Rate:            {win_rate*100:>8.2f}%")
        logger.info("")
        logger.info("RISK:")
        logger.info(f"  Monthly Volatility:  {monthly_vol*100:>8.2f}%")
        logger.info(f"  Annual Volatility:   {annual_vol*100:>8.2f}%")
        logger.info(f"  Sharpe Ratio:        {sharpe:>8.3f}")
        logger.info(f"  Sortino Ratio:       {sortino:>8.3f}")
        logger.info(f"  Max Drawdown:        {max_dd*100:>8.2f}%")
        logger.info(
            f"  Calmar Ratio:        {(annualized_return / abs(max_dd)) if max_dd != 0 else 0:>8.3f}"
        )
        logger.info("")
        logger.info("TURNOVER & COSTS:")
        logger.info(f"  Total Turnover:      {total_turnover*100:>8.1f}%")
        logger.info(
            f"  Avg Turnover/Rebal:  {(total_turnover/df['is_rebalance'].sum())*100:>8.1f}%"
        )
        logger.info(f"  Total TX Costs:      ${total_tx_cost:>12,.0f}")
        logger.info(f"  Avg TX Cost (bps):   {avg_tx_cost_bps:>8.1f}")
        logger.info(
            f"  TX Costs % of AUM:   {(total_tx_cost/df['portfolio_value'].mean())*100:>8.3f}%"
        )
        logger.info("")
        logger.info("PORTFOLIO:")
        logger.info(f"  Avg Positions:       {df['positions_count'].mean():>8.1f}")
        logger.info(f"  Max Positions:       {df['positions_count'].max():>8.0f}")
        logger.info(f"  Min Positions:       {df['positions_count'].min():>8.0f}")
        logger.info("=" * 70)


if __name__ == "__main__":
    """
    Standalone testing and example usage.
    """
    import sys

    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.pathing import PathResolver
    from qx.storage.processed_writer import ProcessedWriterBase
    from qx.storage.typed_loader import TypedCuratedLoader

    print("Portfolio Backtest Model - Standalone Test")
    print("=" * 60)

    # Setup storage infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    resolver = PathResolver()
    loader = TypedCuratedLoader(backend, registry, resolver)
    writer = ProcessedWriterBase(backend, resolver)

    # Create model
    model = PortfolioBacktestModel(
        package_dir=str(Path(__file__).parent),
        loader=loader,
        writer=writer,
    )

    print(f"\n✅ Model initialized: {model.info['id']} v{model.info['version']}")
    print(f"   Description: {model.info['description']}")
    print(f"   Parameters: {list(model.params.keys())}")
