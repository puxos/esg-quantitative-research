"""
Ticker Total Return Model Implementation

Calculates total return metrics for individual ticker(s) including:
    - Total return over period
    - Annualized return (CAGR)
    - Volatility (annualized standard deviation)
    - Sharpe ratio
    - Maximum drawdown
    - Percentage of positive return days
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from qx.engine.base_model import BaseModel

logger = logging.getLogger(__name__)


class TickerTotalReturnModel(BaseModel):
    """
    Calculate Total Return Metrics for Individual Tickers

    Takes OHLCV data and calculates comprehensive return metrics for each ticker:
        - Total return: (end_price / start_price - 1)
        - Annualized return: CAGR using actual time period
        - Volatility: Annualized standard deviation of returns
        - Sharpe ratio: Annualized return / volatility (assuming rf=0)
        - Maximum drawdown: Largest peak-to-trough decline
        - Positive days: Percentage of periods with positive returns

    Key Features:
        - Handles multiple tickers in parallel
        - Uses adjusted close prices (accounts for splits/dividends)
        - Flexible date range filtering
        - Configurable annualization periods
        - Optional time series output for visualization
    """

    def run_impl(
        self,
        inputs: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Calculate total return metrics for tickers.

        Args:
            inputs: Dictionary of input DataFrames
                - equity_prices: OHLCV data (symbol, date, adj_close, ...)
            params: Model parameters from model.yaml
            **kwargs: Additional arguments

        Returns:
            DataFrame with total return metrics per ticker:
                - symbol: Ticker symbol
                - start_date, end_date: Date range
                - observations: Number of price points
                - start_price, end_price: First and last prices
                - total_return: Total return (decimal)
                - annualized_return: CAGR
                - volatility: Annualized volatility
                - sharpe_ratio: Return/risk ratio
                - max_drawdown: Maximum drawdown
                - positive_days_pct: % of positive return periods
        """
        logger.info("Starting Ticker Total Return Calculation")

        # Extract inputs
        equity_prices_df = inputs["equity_prices"]

        # Extract parameters
        symbols = params.get("symbols", [])
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        periods_per_year = params.get("periods_per_year", 252)
        min_observations = params.get("min_observations", 2)
        include_time_series = params.get("include_time_series", False)

        logger.info(f"Parameters:")
        logger.info(f"  Symbols: {symbols if symbols else 'All symbols in data'}")
        logger.info(f"  Date range: {start_date or 'Auto'} to {end_date or 'Auto'}")
        logger.info(f"  Periods per year: {periods_per_year}")
        logger.info(f"  Min observations: {min_observations}")
        logger.info(f"  Include time series: {include_time_series}")

        # 1. Prepare price data
        logger.info("\nStep 1: Preparing price data")
        price_df = self._prepare_prices(
            equity_prices_df,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

        # 2. Calculate returns for each ticker
        logger.info("Step 2: Calculating returns for each ticker")
        results = []

        for symbol in price_df["symbol"].unique():
            symbol_df = price_df[price_df["symbol"] == symbol].sort_values("date")

            if len(symbol_df) < min_observations:
                logger.warning(
                    f"  ⚠️  {symbol}: Only {len(symbol_df)} observations (< {min_observations}), skipping"
                )
                continue

            metrics = self._calculate_ticker_metrics(
                symbol=symbol,
                price_df=symbol_df,
                periods_per_year=periods_per_year,
            )

            results.append(metrics)

        if not results:
            logger.warning("No tickers met minimum observation requirements")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # 3. Log summary
        self._log_summary(results_df)

        return results_df

    def _prepare_prices(
        self,
        equity_prices_df: pd.DataFrame,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """
        Prepare and filter price data.

        Args:
            equity_prices_df: Raw OHLCV data
            symbols: List of symbols to filter (empty = all)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            Filtered DataFrame with [symbol, date, adj_close]
        """
        df = equity_prices_df.copy()

        # Ensure date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            logger.error("No 'date' column found in equity_prices")
            raise ValueError("equity_prices must have a 'date' column")

        # Use adj_close if available, otherwise close
        if "adj_close" in df.columns:
            price_col = "adj_close"
        elif "close" in df.columns:
            price_col = "close"
            logger.warning(
                "No 'adj_close' column, using 'close' (may not account for splits/dividends)"
            )
        else:
            logger.error("No 'adj_close' or 'close' column found")
            raise ValueError("equity_prices must have 'adj_close' or 'close' column")

        # Select relevant columns
        df = df[["symbol", "date", price_col]].rename(columns={price_col: "adj_close"})

        # Filter symbols
        if symbols:
            df = df[df["symbol"].isin(symbols)]
            logger.info(f"  Filtered to {len(symbols)} symbols")

        # Filter dates
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df["date"] >= start_dt]
            logger.info(f"  Filtered to dates >= {start_date}")

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df["date"] <= end_dt]
            logger.info(f"  Filtered to dates <= {end_date}")

        # Drop NaN prices
        initial_rows = len(df)
        df = df.dropna(subset=["adj_close"])
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.warning(f"  Dropped {dropped} rows with NaN prices")

        logger.info(
            f"  Prepared {len(df):,} price observations for {df['symbol'].nunique()} symbols"
        )
        logger.info(
            f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}"
        )

        return df

    def _calculate_ticker_metrics(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        periods_per_year: int,
    ) -> Dict[str, Any]:
        """
        Calculate return metrics for a single ticker.

        Args:
            symbol: Ticker symbol
            price_df: Price data (date, adj_close) sorted by date
            periods_per_year: Annualization factor

        Returns:
            Dictionary of return metrics
        """
        # Basic info
        start_date = price_df["date"].iloc[0]
        end_date = price_df["date"].iloc[-1]
        observations = len(price_df)
        start_price = price_df["adj_close"].iloc[0]
        end_price = price_df["adj_close"].iloc[-1]

        # Calculate returns
        price_df = price_df.copy()
        price_df["return"] = price_df["adj_close"].pct_change()

        # Total return
        total_return = (end_price / start_price) - 1

        # Annualized return (CAGR)
        # Calculate actual years elapsed
        days_elapsed = (end_date - start_date).days
        years_elapsed = days_elapsed / 365.25

        if years_elapsed > 0:
            annualized_return = (1 + total_return) ** (1 / years_elapsed) - 1
        else:
            annualized_return = total_return

        # Volatility (annualized)
        returns = price_df["return"].dropna()
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(periods_per_year)
        else:
            volatility = np.nan

        # Sharpe ratio (assuming risk-free rate = 0)
        if volatility > 0 and not np.isnan(volatility):
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = np.nan

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else np.nan

        # Positive days percentage
        if len(returns) > 0:
            positive_days_pct = (returns > 0).sum() / len(returns) * 100
        else:
            positive_days_pct = np.nan

        logger.info(
            f"  ✓ {symbol}: Total return {total_return*100:+.2f}%, "
            f"Ann. return {annualized_return*100:+.2f}%, "
            f"Volatility {volatility*100:.2f}%, "
            f"Sharpe {sharpe_ratio:.2f}"
        )

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "observations": observations,
            "start_price": start_price,
            "end_price": end_price,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "volatility": volatility,
            "volatility_pct": volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "positive_days_pct": positive_days_pct,
        }

    def _log_summary(self, results_df: pd.DataFrame) -> None:
        """Log summary statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("Total Return Summary")
        logger.info("=" * 80)
        logger.info(f"Tickers analyzed: {len(results_df)}")
        logger.info(f"\nTop 5 performers (by total return):")

        top5 = results_df.nlargest(5, "total_return")
        for _, row in top5.iterrows():
            logger.info(
                f"  {row['symbol']:6s}: {row['total_return_pct']:+8.2f}% "
                f"(Ann: {row['annualized_return_pct']:+7.2f}%, "
                f"Vol: {row['volatility_pct']:6.2f}%, "
                f"Sharpe: {row['sharpe_ratio']:5.2f})"
            )

        logger.info(f"\nBottom 5 performers (by total return):")
        bottom5 = results_df.nsmallest(5, "total_return")
        for _, row in bottom5.iterrows():
            logger.info(
                f"  {row['symbol']:6s}: {row['total_return_pct']:+8.2f}% "
                f"(Ann: {row['annualized_return_pct']:+7.2f}%, "
                f"Vol: {row['volatility_pct']:6.2f}%, "
                f"Sharpe: {row['sharpe_ratio']:5.2f})"
            )

        logger.info(f"\nAggregate statistics:")
        logger.info(
            f"  Mean total return: {results_df['total_return_pct'].mean():+.2f}%"
        )
        logger.info(
            f"  Median total return: {results_df['total_return_pct'].median():+.2f}%"
        )
        logger.info(
            f"  Mean annualized return: {results_df['annualized_return_pct'].mean():+.2f}%"
        )
        logger.info(f"  Mean volatility: {results_df['volatility_pct'].mean():.2f}%")
        logger.info(f"  Mean Sharpe ratio: {results_df['sharpe_ratio'].mean():.2f}")
        logger.info("=" * 80)
