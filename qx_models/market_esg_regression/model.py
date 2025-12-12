"""
Market + ESG Two-Factor OLS Regression Model Implementation

Migrated from legacy TwoFactorRegression (src/programs/two_factor_regression.py)
to Qx BaseModel architecture.

Performs OLS regression to estimate factor exposures:
    R_i,t - RF_t = α_i + β_market * (R_market,t - RF_t) + β_ESG * ESG_factor_t + ε_i,t
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from qx.engine.base_model import BaseModel

logger = logging.getLogger(__name__)


class MarketESGRegressionModel(BaseModel):
    """
    Market + ESG Two-Factor OLS Regression Model

    Estimates factor exposures (betas) and alpha for each stock using OLS regression.
    Supports both cross-sectional (full sample) and time-series (rolling window) analysis.
    """

    def run_impl(
        self, inputs: Dict[str, pd.DataFrame], params: Dict, **kwargs
    ) -> pd.DataFrame:
        """
        Run two-factor OLS regression for all stocks

        Args:
            inputs: Dict with keys:
                - equity_prices: Daily stock OHLCV data (multiple symbols)
                - market_prices: Daily SPY OHLCV data (market proxy)
                - esg_factors: Monthly ESG factor returns
                - risk_free: Daily risk-free rates
            params: Dict with keys:
                - esg_factor_name: Which ESG factor to use (default: "ESG")
                - window_months: Rolling window size (null = full sample)
                - min_observations: Minimum observations required
                - use_hac_se: Use Newey-West HAC standard errors
                - hac_maxlags: Maximum lags for HAC covariance
            **kwargs: Additional args (e.g., run_id)

        Returns:
            DataFrame with columns: symbol, date, alpha, beta_market, beta_esg, statistics

            BaseModel will add: model, model_version, featureset_id, run_id, run_ts
        """
        logger.info("=" * 80)
        logger.info("Two-Factor Regression Model - Execution Started")
        logger.info("=" * 80)

        # Extract inputs
        equity_prices = inputs["equity_prices"]
        market_prices = inputs["market_prices"]
        esg_factors = inputs["esg_factors"]
        rf_df = inputs["risk_free"]

        # Extract parameters
        esg_factor_name = params["esg_factor_name"]
        window_months = params["window_months"]
        min_observations = params["min_observations"]
        use_hac_se = params["use_hac_se"]
        hac_maxlags = params["hac_maxlags"]

        logger.info(f"Parameters:")
        logger.info(f"  ESG factor: {esg_factor_name}")
        logger.info(f"  Window: {window_months if window_months else 'Full sample'}")
        logger.info(f"  Min observations: {min_observations}")
        logger.info(f"  HAC standard errors: {use_hac_se}")
        if use_hac_se:
            logger.info(f"  HAC max lags: {hac_maxlags}")

        # Prepare data
        logger.info("\n📊 Preparing data...")

        # Resample daily data to monthly
        logger.info("  Resampling daily prices to monthly...")
        stock_returns = self._prepare_stock_returns(equity_prices, rf_df)
        market_excess = self._prepare_market_returns(market_prices, rf_df)

        # Prepare ESG factor (already monthly from ESGFactorModel)
        logger.info(f"  Extracting {esg_factor_name} factor...")
        esg_factor = esg_factors[esg_factors["factor_name"] == esg_factor_name][
            ["date", "factor_return"]
        ].copy()
        esg_factor = esg_factor.rename(columns={"factor_return": "esg_factor"})

        logger.info(f"    Stock returns: {len(stock_returns)} observations")
        logger.info(f"    Market excess: {len(market_excess)} observations")
        logger.info(f"    ESG factor: {len(esg_factor)} observations")

        # Get unique symbols
        symbols = stock_returns["symbol"].unique()
        logger.info(f"\n🏗️ Running regressions for {len(symbols)} stocks...")

        # Run regressions for each stock
        all_results = []
        success_count = 0
        fail_count = 0

        for i, symbol in enumerate(symbols, 1):
            if i % 50 == 0:
                logger.info(
                    f"  Progress: {i}/{len(symbols)} ({success_count} success, {fail_count} failed)"
                )

            try:
                # Get stock-specific returns
                stock_data = stock_returns[stock_returns["symbol"] == symbol].copy()

                # Merge with factors
                regression_data = stock_data.merge(
                    market_excess, on="date", how="inner"
                )
                regression_data = regression_data.merge(
                    esg_factor, on="date", how="inner"
                )

                if len(regression_data) < min_observations:
                    logger.debug(
                        f"  Insufficient data for {symbol}: {len(regression_data)} < {min_observations}"
                    )
                    fail_count += 1
                    continue

                # Run regression
                if window_months is None:
                    # Full sample (cross-sectional)
                    results = self._run_single_regression(
                        regression_data,
                        symbol,
                        use_hac_se=use_hac_se,
                        hac_maxlags=hac_maxlags,
                    )
                    if results:
                        all_results.append(results)
                        success_count += 1
                else:
                    # Rolling window (time-series)
                    results = self._run_rolling_regression(
                        regression_data,
                        symbol,
                        window_months=window_months,
                        use_hac_se=use_hac_se,
                        hac_maxlags=hac_maxlags,
                    )
                    if results:
                        all_results.extend(results)
                        success_count += 1

            except Exception as e:
                logger.warning(f"  Error processing {symbol}: {e}")
                fail_count += 1
                continue

        if not all_results:
            logger.error("No regression results generated")
            return pd.DataFrame()

        # Combine results
        result_df = pd.DataFrame(all_results)

        logger.info(f"\n✅ Completed regressions:")
        logger.info(f"   Success: {success_count} stocks")
        logger.info(f"   Failed: {fail_count} stocks")
        logger.info(f"   Total results: {len(result_df)} rows")
        if window_months:
            logger.info(
                f"   Time-series: {result_df['date'].min()} to {result_df['date'].max()}"
            )

        logger.info("=" * 80)
        logger.info("Two-Factor Regression Model - Execution Completed")
        logger.info("=" * 80)

        return result_df

    @staticmethod
    def _prepare_stock_returns(
        prices_df: pd.DataFrame, rf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare monthly stock excess returns"""
        # Resample daily to monthly (end of month)
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        monthly_prices = []
        for symbol, group in prices_df.groupby("symbol"):
            group_sorted = group.sort_values("date").set_index("date")
            monthly_group = group_sorted.resample("ME").last()
            monthly_group["symbol"] = symbol
            monthly_prices.append(monthly_group.reset_index())

        monthly = pd.concat(monthly_prices, ignore_index=True)
        monthly = monthly.dropna(subset=["close"])

        # Calculate returns
        monthly["return"] = monthly.groupby("symbol")["close"].pct_change()

        # Resample RF to monthly
        rf_monthly = MarketESGRegressionModel._resample_rf_to_monthly(rf_df)

        # Merge and calculate excess returns
        stock_excess = monthly[["date", "symbol", "return"]].dropna()
        stock_excess = stock_excess.merge(rf_monthly, on="date", how="inner")
        stock_excess["excess_return"] = stock_excess["return"] - stock_excess["RF"]

        return stock_excess[["date", "symbol", "excess_return"]]

    @staticmethod
    def _prepare_market_returns(
        market_prices: pd.DataFrame, rf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare monthly market excess returns (SPY)"""
        market_prices = market_prices.copy()
        market_prices["date"] = pd.to_datetime(market_prices["date"])

        # Resample to monthly
        market_monthly = market_prices.sort_values("date").set_index("date")
        market_monthly = market_monthly.resample("ME").last().reset_index()
        market_monthly = market_monthly.dropna(subset=["close"])

        # Calculate returns
        market_monthly["market_return"] = market_monthly["close"].pct_change()

        # Resample RF to monthly
        rf_monthly = MarketESGRegressionModel._resample_rf_to_monthly(rf_df)

        # Calculate excess returns
        market_excess = market_monthly[["date", "market_return"]].dropna()
        market_excess = market_excess.merge(rf_monthly, on="date", how="inner")
        market_excess["market_excess"] = (
            market_excess["market_return"] - market_excess["RF"]
        )

        return market_excess[["date", "market_excess"]]

    @staticmethod
    def _resample_rf_to_monthly(rf_df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily risk-free rates to monthly (average)"""
        rf_df = rf_df.copy()
        rf_df["date"] = pd.to_datetime(rf_df["date"])
        rf_df = rf_df.sort_values("date").set_index("date")

        # Resample to month-end, taking average rate for the month
        rf_monthly = rf_df["rate"].resample("ME").mean().reset_index()

        # Normalize to monthly decimal (from annual percentage)
        rf_monthly["RF"] = rf_monthly["rate"] / 100 / 12

        return rf_monthly[["date", "RF"]]

    @staticmethod
    def _run_single_regression(
        data: pd.DataFrame,
        symbol: str,
        use_hac_se: bool = True,
        hac_maxlags: int = 12,
    ) -> Optional[Dict]:
        """
        Run single OLS regression for full sample

        Args:
            data: DataFrame with excess_return, market_excess, esg_factor
            symbol: Stock ticker
            use_hac_se: Use Newey-West HAC standard errors
            hac_maxlags: Maximum lags for HAC covariance

        Returns:
            Dictionary with regression results (monthly units)
        """
        try:
            # Prepare regression
            y = data["excess_return"]
            X = data[["market_excess", "esg_factor"]]
            X = sm.add_constant(X)

            # Run OLS
            if use_hac_se:
                # HAC-robust standard errors (corrects for autocorrelation and heteroskedasticity)
                model = sm.OLS(y, X).fit(
                    cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
                )
            else:
                # Standard OLS (assumes i.i.d. errors)
                model = sm.OLS(y, X).fit()

            # Extract results
            results = {
                "symbol": symbol,
                "date": data["date"].max(),  # End of sample
                "alpha": model.params["const"],
                "beta_market": model.params["market_excess"],
                "beta_esg": model.params["esg_factor"],
                "alpha_tstat": model.tvalues["const"],
                "beta_market_tstat": model.tvalues["market_excess"],
                "beta_esg_tstat": model.tvalues["esg_factor"],
                "alpha_pvalue": model.pvalues["const"],
                "beta_market_pvalue": model.pvalues["market_excess"],
                "beta_esg_pvalue": model.pvalues["esg_factor"],
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "f_statistic": model.fvalue,
                "f_pvalue": model.f_pvalue,
                "observations": int(model.nobs),
                "std_error_alpha": model.bse["const"],
                "std_error_beta_market": model.bse["market_excess"],
                "std_error_beta_esg": model.bse["esg_factor"],
            }

            return results

        except Exception as e:
            logger.debug(f"Regression failed for {symbol}: {e}")
            return None

    @staticmethod
    def _run_rolling_regression(
        data: pd.DataFrame,
        symbol: str,
        window_months: int,
        use_hac_se: bool = True,
        hac_maxlags: int = 12,
    ) -> list:
        """
        Run rolling window regressions to generate time-series of betas

        Args:
            data: DataFrame with excess_return, market_excess, esg_factor
            symbol: Stock ticker
            window_months: Rolling window size
            use_hac_se: Use Newey-West HAC standard errors
            hac_maxlags: Maximum lags for HAC covariance

        Returns:
            List of dictionaries with rolling regression results
        """
        results = []

        # Sort by date
        data = data.sort_values("date")

        # Rolling windows
        for i in range(window_months, len(data) + 1):
            window_data = data.iloc[i - window_months : i]

            result = MarketESGRegressionModel._run_single_regression(
                window_data, symbol, use_hac_se=use_hac_se, hac_maxlags=hac_maxlags
            )

            if result:
                # Update date to window end
                result["date"] = window_data["date"].iloc[-1]
                results.append(result)

        return results
