"""
Market Beta Model - Single Factor Regression (CAPM)

Estimates market beta via OLS regression:
    Excess_Return_i = α + β * Excess_Return_market + ε

Where:
    - Excess_Return_i = Return_i - Risk_Free_Rate
    - Excess_Return_market = Market_Return - Risk_Free_Rate
    - β = Market beta (systematic risk)
    - α = Jensen's alpha (risk-adjusted outperformance)

Supports:
    - Rolling window estimation (e.g., 60-month)
    - Full sample estimation (window=null)
    - Newey-West HAC standard errors
    - Per-symbol beta estimates
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

from qx.engine.base_model import BaseModel


class MarketBetaModel(BaseModel):
    """
    Single-factor regression model for market beta estimation.

    Implements CAPM: Excess_Return_i = α + β * Excess_Return_market + ε
    """

    def __init__(self, package_dir: str, loader, writer, overrides: dict = None):
        """
        Initialize market beta model.

        Args:
            package_dir: Path to model package directory
            loader: TypedCuratedLoader instance
            writer: ProcessedWriterBase instance
            overrides: Parameter overrides
        """
        super().__init__(package_dir, loader, writer, overrides)

    def run_impl(self, inputs: dict, params: dict, **kwargs) -> pd.DataFrame:
        """
        Execute market beta model.

        Args:
            inputs: Dictionary of input DataFrames
            params: Model parameters
            **kwargs: Additional keyword arguments

        Returns:
            DataFrame with columns:
                - symbol: Equity ticker
                - date: Observation date (rolling) or sample end date (full)
                - alpha: Intercept (Jensen's alpha)
                - beta: Market beta
                - alpha_tstat: t-statistic for alpha
                - beta_tstat: t-statistic for beta
                - r_squared: R² of regression
                - residual_vol: Residual standard deviation (idiosyncratic risk)
                - observations: Number of observations used
                - window_start: Window start date
                - window_end: Window end date
        """
        # Load inputs via auto-injection
        equity_df = inputs["equity_prices"]
        market_df = inputs["market_prices"]
        rf_df = inputs["risk_free"]

        print(f"\n📊 Market Beta Model")
        print(f"   Equity data: {len(equity_df):,} rows")
        print(f"   Market data: {len(market_df):,} rows")
        print(f"   Risk-free data: {len(rf_df):,} rows")

        # Extract parameters
        window = params.get("window")
        min_obs = params.get("min_observations", 24)
        hac_lags = params.get("hac_lags", 6)
        price_col = params.get("price_column", "adj_close")
        annualization = params.get("annualization_factor", 12)

        print(f"   Window: {window if window else 'Full sample'}")
        print(f"   Min observations: {min_obs}")
        print(f"   HAC lags: {hac_lags}")
        print(f"   Price column: {price_col}")

        # Check for empty equity data
        if equity_df.empty:
            print(f"\n⚠️  No equity data available - returning empty results")
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "date",
                    "alpha",
                    "beta",
                    "alpha_tstat",
                    "beta_tstat",
                    "r_squared",
                    "residual_vol",
                    "observations",
                    "window_start",
                    "window_end",
                ]
            )

        # Prepare data
        equity_returns = self._compute_returns(equity_df, price_col)
        market_returns = self._compute_returns(market_df, price_col)
        rf_returns = self._prepare_risk_free(rf_df, annualization)

        print(f"\n✅ Computed returns")
        print(f"   Equity: {len(equity_returns):,} observations")
        print(f"   Market: {len(market_returns):,} observations")
        print(f"   Risk-free: {len(rf_returns):,} observations")

        # Merge data
        merged = self._merge_data(equity_returns, market_returns, rf_returns)

        print(f"✅ Merged data: {len(merged):,} observations")
        print(f"   Symbols: {merged['symbol'].nunique()}")

        # Handle both datetime.date and pd.Timestamp for display
        min_date = merged["date"].min()
        max_date = merged["date"].max()
        min_date_str = min_date.date() if hasattr(min_date, "date") else min_date
        max_date_str = max_date.date() if hasattr(max_date, "date") else max_date
        print(f"   Date range: {min_date_str} to {max_date_str}")

        # Compute excess returns
        merged["excess_return"] = merged["return"] - merged["rf"]
        merged["excess_market"] = merged["market_return"] - merged["rf"]

        # Run regressions
        if window:
            print(f"\n🔄 Running rolling {window}-month regressions...")
            results = self._rolling_regression(merged, window, min_obs, hac_lags)
        else:
            print(f"\n📈 Running full-sample regressions...")
            results = self._full_sample_regression(merged, min_obs, hac_lags)

        print(f"\n✅ Regressions complete: {len(results):,} estimates")
        print(f"   Symbols: {results['symbol'].nunique()}")
        print(f"   Mean beta: {results['beta'].mean():.3f}")
        print(f"   Mean R²: {results['r_squared'].mean():.3f}")

        return results

    def _compute_returns(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Compute returns from OHLCV data."""
        if df.empty:
            # Return empty DataFrame with expected schema
            return (
                pd.DataFrame(columns=["date", "symbol", "return"])
                if "symbol" in df.columns
                else pd.DataFrame(columns=["date", "return"])
            )

        df = df.copy()
        df = (
            df.sort_values(["symbol", "date"])
            if "symbol" in df.columns
            else df.sort_values("date")
        )

        if "symbol" in df.columns:
            df["return"] = df.groupby("symbol")[price_col].pct_change()
        else:
            df["return"] = df[price_col].pct_change()

        return (
            df[["date", "symbol", "return"]].dropna()
            if "symbol" in df.columns
            else df[["date", "return"]].dropna()
        )

    def _prepare_risk_free(self, df: pd.DataFrame, annualization: int) -> pd.DataFrame:
        """Convert risk-free rate from annual percentage to period return."""
        df = df.copy()
        # Assume 'rate' is in annual percentage (e.g., 3.5 for 3.5%)
        df["rf"] = (df["rate"] / 100) / annualization
        return df[["date", "rf"]]

    def _merge_data(
        self,
        equity_returns: pd.DataFrame,
        market_returns: pd.DataFrame,
        rf_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge equity, market, and risk-free data."""
        # Rename market return column
        market_returns = market_returns.rename(columns={"return": "market_return"})
        if "symbol" in market_returns.columns:
            market_returns = market_returns.drop(columns=["symbol"])

        # Merge
        merged = equity_returns.merge(market_returns, on="date", how="inner")
        merged = merged.merge(rf_returns, on="date", how="inner")

        return merged

    def _rolling_regression(
        self, df: pd.DataFrame, window: int, min_obs: int, hac_lags: int
    ) -> pd.DataFrame:
        """Run rolling window regressions."""
        results = []

        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].sort_values("date")

            for i in range(len(symbol_df)):
                # Define window
                window_end_idx = i + 1
                window_start_idx = max(0, window_end_idx - window)

                window_data = symbol_df.iloc[window_start_idx:window_end_idx]

                if len(window_data) < min_obs:
                    continue

                # Run regression
                beta_stats = self._estimate_beta(
                    window_data["excess_return"].values,
                    window_data["excess_market"].values,
                    hac_lags,
                )

                if beta_stats is None:
                    continue

                results.append(
                    {
                        "symbol": symbol,
                        "date": symbol_df.iloc[i]["date"],
                        "window_start": window_data["date"].min(),
                        "window_end": window_data["date"].max(),
                        "observations": len(window_data),
                        **beta_stats,
                    }
                )

        return pd.DataFrame(results)

    def _full_sample_regression(
        self, df: pd.DataFrame, min_obs: int, hac_lags: int
    ) -> pd.DataFrame:
        """Run full-sample regressions per symbol."""
        results = []

        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].sort_values("date")

            if len(symbol_df) < min_obs:
                continue

            # Run regression
            beta_stats = self._estimate_beta(
                symbol_df["excess_return"].values,
                symbol_df["excess_market"].values,
                hac_lags,
            )

            if beta_stats is None:
                continue

            results.append(
                {
                    "symbol": symbol,
                    "date": symbol_df["date"].max(),
                    "window_start": symbol_df["date"].min(),
                    "window_end": symbol_df["date"].max(),
                    "observations": len(symbol_df),
                    **beta_stats,
                }
            )

        return pd.DataFrame(results)

    def _estimate_beta(
        self, y: np.ndarray, x: np.ndarray, hac_lags: int
    ) -> Optional[dict]:
        """
        Estimate beta via OLS with HAC standard errors.

        Args:
            y: Excess returns (dependent variable)
            x: Excess market returns (independent variable)
            hac_lags: Number of lags for Newey-West HAC

        Returns:
            Dictionary with regression statistics or None if failed
        """
        try:
            # Add intercept
            X = np.column_stack([np.ones(len(x)), x])

            # OLS regression
            model = OLS(y, X)
            results = model.fit()

            # Newey-West HAC covariance
            cov_hac_matrix = cov_hac(results, nlags=hac_lags)

            # Extract coefficients and standard errors
            alpha = results.params[0]
            beta = results.params[1]
            alpha_se = np.sqrt(cov_hac_matrix[0, 0])
            beta_se = np.sqrt(cov_hac_matrix[1, 1])

            # T-statistics
            alpha_tstat = alpha / alpha_se if alpha_se > 0 else np.nan
            beta_tstat = beta / beta_se if beta_se > 0 else np.nan

            # R-squared and residual volatility
            r_squared = results.rsquared
            residual_vol = np.std(results.resid)

            return {
                "alpha": alpha,
                "beta": beta,
                "alpha_tstat": alpha_tstat,
                "beta_tstat": beta_tstat,
                "r_squared": r_squared,
                "residual_vol": residual_vol,
            }

        except Exception as e:
            # Regression failed (e.g., singular matrix, insufficient data)
            return None


# Example usage (for testing)
if __name__ == "__main__":
    print("Market Beta Model - Single Factor Regression")
    print("=" * 80)
    print("This model estimates market beta via CAPM regression.")
    print()
    print("Usage:")
    print("  1. Add to DAG with dependencies on:")
    print("     - LoadOHLCVPanel (equity prices)")
    print("     - LoadMarketProxy (market returns)")
    print("     - LoadTreasuryRates (risk-free rate)")
    print()
    print("  2. Configure parameters:")
    print("     - window: Rolling window (null = full sample)")
    print("     - min_observations: Minimum data points")
    print("     - hac_lags: Newey-West lags")
    print()
    print("  3. Output: Market betas with statistics")
    print("=" * 80)
