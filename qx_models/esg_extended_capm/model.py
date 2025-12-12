"""
Factor Expected Returns Model Implementation

Calculates expected returns using factor model framework:
    E[R_i,t] = RF_t + Σ(β_i,k × λ_k)

This implementation focuses on two-factor model (market + ESG):
    E[R_i,t] = RF_t + β_market × λ_market + β_ESG × λ_ESG
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from qx.engine.base_model import BaseModel

logger = logging.getLogger(__name__)


class FactorExpectedReturnsModel(BaseModel):
    """
    Factor Expected Returns Model (Extended CAPM with ESG)

    Calculates expected returns using factor exposures (betas) and factor premia (lambdas).

    Process:
        1. Load pre-computed factor betas from two_factor_regression model
        2. Estimate factor premia using HAC-robust means (Newey-West)
        3. Optionally apply shrinkage to premia (correct sample period bias)
        4. Optionally cap extreme betas (prevent outlier leverage)
        5. Calculate expected returns: E[R] = RF + β'λ
        6. Compound annualize for reporting

    Key Features:
        - HAC-robust factor premia (autocorrelation + heteroskedasticity correction)
        - Shrinkage toward historical long-term means (sample period bias correction)
        - Beta capping (prevent extreme leverage from outlier betas)
        - Time-varying betas support (rolling window estimates)
        - Compound annualization (realistic return projections)
    """

    def run_impl(
        self,
        inputs: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Calculate factor-based expected returns.

        Args:
            inputs: Dictionary of input DataFrames
                - two_factor_betas: Factor betas (symbol, date, beta_market, beta_esg)
                - market_prices: Market portfolio prices (SPY)
                - risk_free: Risk-free rate time-series
                - esg_factors: ESG factor returns
            params: Model parameters from model.yaml
            **kwargs: Additional arguments (e.g., date_range for forecast period)

        Returns:
            DataFrame with expected returns for each stock and forecast date
        """
        logger.info("Starting Factor Expected Returns calculation")

        # Extract inputs
        betas_df = inputs["two_factor_betas"]
        market_prices = inputs["market_prices"]
        rf_df = inputs["risk_free"]
        esg_factors = inputs["esg_factors"]

        # Extract parameters
        apply_shrinkage = params.get("apply_shrinkage", True)
        shrinkage_weight = params.get("shrinkage_weight", 0.5)
        historical_market_premium = params.get("historical_market_premium", 0.005)
        historical_esg_premium = params.get("historical_esg_premium", 0.0)
        hac_maxlags = params.get("hac_maxlags", 12)
        cap_betas = params.get("cap_betas", True)
        beta_market_cap = params.get("beta_market_cap", 3.0)
        beta_esg_cap = params.get("beta_esg_cap", 5.0)
        use_latest_betas = params.get("use_latest_betas", True)

        # Get date range from kwargs (if not provided, use RF date range)
        date_range = kwargs.get("date_range")
        if date_range is None:
            start_date = rf_df["date"].min()
            end_date = rf_df["date"].max()
        else:
            start_date, end_date = date_range

        logger.info(f"Forecast period: {start_date} to {end_date}")
        logger.info(
            f"Parameters: shrinkage={apply_shrinkage} (w={shrinkage_weight:.2f}), "
            f"beta_capping={cap_betas} (market=±{beta_market_cap:.1f}, ESG=±{beta_esg_cap:.1f})"
        )

        # 1. Prepare data (resample to monthly)
        logger.info("Step 1: Preparing data (resampling to monthly)")
        market_excess = self._prepare_market_returns(
            market_prices, rf_df, start_date, end_date
        )
        esg_factor_series = self._prepare_esg_factors(esg_factors, start_date, end_date)
        rf_monthly = self._resample_rf_to_monthly(rf_df, start_date, end_date)

        # 2. Estimate factor premia
        logger.info("Step 2: Estimating factor premia (HAC-robust)")
        lambda_market, lambda_esg = self._estimate_factor_premia(
            market_excess=market_excess,
            esg_factor=esg_factor_series,
            shrinkage_weight=shrinkage_weight,
            historical_market_premium=historical_market_premium,
            historical_esg_premium=historical_esg_premium,
            apply_shrinkage=apply_shrinkage,
            hac_maxlags=hac_maxlags,
        )

        # 3. Process betas (handle latest vs time-varying)
        logger.info("Step 3: Processing factor betas")
        betas_processed = self._process_betas(betas_df, use_latest_betas)

        # 4. Calculate expected returns
        logger.info("Step 4: Calculating expected returns")
        results = self._calculate_expected_returns(
            betas_df=betas_processed,
            rf_df=rf_monthly,
            lambda_market=lambda_market,
            lambda_esg=lambda_esg,
            start_date=start_date,
            end_date=end_date,
            cap_betas=cap_betas,
            beta_market_cap=beta_market_cap,
            beta_esg_cap=beta_esg_cap,
        )

        logger.info(
            f"Generated expected returns for {len(results['symbol'].unique())} stocks, "
            f"{len(results)} total observations"
        )

        return results

    def _prepare_market_returns(
        self,
        market_prices: pd.DataFrame,
        rf_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """
        Prepare market excess returns (monthly).

        Args:
            market_prices: Market portfolio prices (SPY, daily)
            rf_df: Risk-free rate (daily)
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Series of monthly market excess returns
        """
        # Filter to date range
        df = market_prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df = df.sort_values("date")

        # Resample to monthly (end of month)
        df = df.set_index("date")
        monthly = df["adjClose"].resample("ME").last()
        market_returns = monthly.pct_change().dropna()

        # Get monthly RF rate
        rf_monthly = self._resample_rf_to_monthly(rf_df, start_date, end_date)

        # Calculate excess returns
        excess = market_returns.to_frame("market_return")
        excess = excess.merge(rf_monthly, left_index=True, right_on="date", how="inner")
        excess["market_excess"] = excess["market_return"] - excess["RF"]

        logger.info(f"Prepared {len(excess)} monthly market excess returns")
        logger.info(
            f"  Mean (monthly): {excess['market_excess'].mean():.6f} "
            f"({excess['market_excess'].mean()*12*100:.2f}% annualized)"
        )

        return excess.set_index("date")["market_excess"]

    def _prepare_esg_factors(
        self,
        esg_factors: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """
        Prepare ESG factor returns (monthly).

        Args:
            esg_factors: ESG factor returns (monthly)
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Series of monthly ESG factor returns
        """
        df = esg_factors.copy()

        # Filter to ESG factor (composite)
        if "factor_name" in df.columns:
            df = df[df["factor_name"] == "ESG"]

        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df = df.sort_values("date")

        # Extract returns
        if "long_short_return" in df.columns:
            esg_series = df.set_index("date")["long_short_return"]
        elif "return" in df.columns:
            esg_series = df.set_index("date")["return"]
        else:
            raise ValueError("ESG factors DataFrame missing return column")

        logger.info(f"Prepared {len(esg_series)} monthly ESG factor returns")
        logger.info(
            f"  Mean (monthly): {esg_series.mean():.6f} "
            f"({esg_series.mean()*12*100:.2f}% annualized)"
        )

        return esg_series

    def _resample_rf_to_monthly(
        self,
        rf_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Resample risk-free rate to monthly frequency.

        Args:
            rf_df: Risk-free rate (daily, annual %)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with columns [date, RF] (monthly decimal)
        """
        df = rf_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        # Convert annual % to monthly decimal
        df["RF"] = df["rate"] / 100 / 12

        # Resample to monthly (end of month)
        df = df.set_index("date")
        monthly = df["RF"].resample("ME").last()

        result = monthly.reset_index()
        result.columns = ["date", "RF"]

        logger.info(f"Resampled {len(result)} monthly RF observations")
        logger.info(
            f"  Mean (monthly): {result['RF'].mean():.6f} "
            f"({result['RF'].mean()*12*100:.2f}% annualized)"
        )

        return result

    def _hac_factor_mean(
        self,
        x: pd.Series,
        lags: int = 12,
    ) -> Tuple[float, float, float]:
        """
        Estimate factor mean using HAC-robust standard errors (Newey-West).

        Args:
            x: Time-series of factor returns (monthly decimal)
            lags: Number of lags for Newey-West correction

        Returns:
            Tuple of (mean, standard_error, t_statistic)
        """
        X = np.ones((len(x), 1))  # Constant only (intercept-only regression)
        model = sm.OLS(x.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

        mean = float(model.params[0])
        std_err = float(model.bse[0])
        t_stat = float(model.tvalues[0])

        return mean, std_err, t_stat

    def _estimate_factor_premia(
        self,
        market_excess: pd.Series,
        esg_factor: pd.Series,
        shrinkage_weight: float,
        historical_market_premium: float,
        historical_esg_premium: float,
        apply_shrinkage: bool,
        hac_maxlags: int,
    ) -> Tuple[float, float]:
        """
        Estimate factor premia using HAC-robust means with optional shrinkage.

        Args:
            market_excess: Market excess returns (monthly decimal)
            esg_factor: ESG factor returns (monthly decimal)
            shrinkage_weight: Weight on historical mean (0-1)
            historical_market_premium: Long-term equity premium
            historical_esg_premium: Long-term ESG premium
            apply_shrinkage: Whether to apply shrinkage
            hac_maxlags: Lags for Newey-West correction

        Returns:
            Tuple of (lambda_market, lambda_esg) in monthly decimals
        """
        # Align data
        df = pd.concat(
            [market_excess.rename("MKT"), esg_factor.rename("ESG")], axis=1
        ).dropna()

        if df.empty:
            raise ValueError("No overlapping dates for factor premia estimation")

        # Estimate HAC-robust means (sample estimates)
        lambda_market_sample, se_market, t_market = self._hac_factor_mean(
            df["MKT"], lags=hac_maxlags
        )
        lambda_esg_sample, se_esg, t_esg = self._hac_factor_mean(
            df["ESG"], lags=hac_maxlags
        )

        logger.info(f"Factor Premia (Sample, HAC-robust, monthly decimals):")
        logger.info(
            f"  λ_market (sample) = {lambda_market_sample:.6f} "
            f"({lambda_market_sample*12*100:.2f}% annualized)"
        )
        logger.info(f"    SE = {se_market:.6f}, t-stat = {t_market:.2f}")
        logger.info(
            f"  λ_ESG (sample)    = {lambda_esg_sample:.6f} "
            f"({lambda_esg_sample*12*100:.2f}% annualized)"
        )
        logger.info(f"    SE = {se_esg:.6f}, t-stat = {t_esg:.2f}")
        logger.info(f"  Based on {len(df)} monthly observations")

        # Apply shrinkage toward historical long-term mean
        if apply_shrinkage:
            w = shrinkage_weight
            lambda_market = (
                w * historical_market_premium + (1 - w) * lambda_market_sample
            )
            lambda_esg = w * historical_esg_premium + (1 - w) * lambda_esg_sample

            logger.info(f"\nShrinkage Applied (weight={w:.2f}):")
            logger.info(
                f"  λ_market (adjusted) = {lambda_market:.6f} "
                f"({lambda_market*12*100:.2f}% annualized)"
            )
            logger.info(
                f"    Historical: {historical_market_premium:.6f}, "
                f"Sample: {lambda_market_sample:.6f}"
            )
            logger.info(
                f"  λ_ESG (adjusted)    = {lambda_esg:.6f} "
                f"({lambda_esg*12*100:.2f}% annualized)"
            )
            logger.info(
                f"    Historical: {historical_esg_premium:.6f}, "
                f"Sample: {lambda_esg_sample:.6f}"
            )
        else:
            lambda_market = lambda_market_sample
            lambda_esg = lambda_esg_sample
            logger.info("\nNo shrinkage applied (using raw sample estimates)")

        return lambda_market, lambda_esg

    def _process_betas(
        self,
        betas_df: pd.DataFrame,
        use_latest: bool,
    ) -> pd.DataFrame:
        """
        Process betas (handle latest vs time-varying).

        Args:
            betas_df: Factor betas from two_factor_regression
            use_latest: Use latest estimate (True) or time-series (False)

        Returns:
            Processed betas DataFrame
        """
        df = betas_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Rename columns to match expected format
        rename_map = {}
        if "ticker" in df.columns and "symbol" not in df.columns:
            rename_map["ticker"] = "symbol"
        if "beta_esg" in df.columns and "beta_ESG" not in df.columns:
            rename_map["beta_esg"] = "beta_ESG"

        if rename_map:
            df = df.rename(columns=rename_map)

        if use_latest:
            # Use only latest beta estimate for each stock
            df = df.sort_values(["symbol", "date"])
            df = df.groupby("symbol").tail(1).reset_index(drop=True)
            logger.info(f"Using latest betas for {len(df)} stocks")
        else:
            logger.info(f"Using time-varying betas ({len(df)} total observations)")

        return df

    def _calculate_expected_returns(
        self,
        betas_df: pd.DataFrame,
        rf_df: pd.DataFrame,
        lambda_market: float,
        lambda_esg: float,
        start_date: str,
        end_date: str,
        cap_betas: bool,
        beta_market_cap: float,
        beta_esg_cap: float,
    ) -> pd.DataFrame:
        """
        Calculate expected returns using factor model formula.

        E[R_i,t] = RF_t + β_market × λ_market + β_ESG × λ_ESG

        Args:
            betas_df: Factor betas (symbol, date, beta_market, beta_ESG)
            rf_df: Risk-free rate (date, RF)
            lambda_market: Market risk premium
            lambda_esg: ESG risk premium
            start_date: Forecast start date
            end_date: Forecast end date
            cap_betas: Whether to cap extreme betas
            beta_market_cap: Cap for market beta
            beta_esg_cap: Cap for ESG beta

        Returns:
            DataFrame with expected returns
        """
        # Filter RF to forecast range
        rf = rf_df.copy()
        rf["date"] = pd.to_datetime(rf["date"])
        rf = rf[(rf["date"] >= start_date) & (rf["date"] <= end_date)]

        results = []

        for symbol in betas_df["symbol"].unique():
            symbol_betas = betas_df[betas_df["symbol"] == symbol].copy()

            if len(symbol_betas) == 1:
                # Constant betas across all forecast dates
                beta_m = symbol_betas["beta_market"].iloc[0]
                beta_esg = symbol_betas["beta_ESG"].iloc[0]
                beta_date = symbol_betas["date"].iloc[0]

                # Cap betas if requested
                beta_m_capped = (
                    np.clip(beta_m, -beta_market_cap, beta_market_cap)
                    if cap_betas
                    else beta_m
                )
                beta_esg_capped = (
                    np.clip(beta_esg, -beta_esg_cap, beta_esg_cap)
                    if cap_betas
                    else beta_esg
                )

                # Apply to all forecast dates
                for _, row in rf.iterrows():
                    er_monthly = (
                        row["RF"]
                        + beta_m_capped * lambda_market
                        + beta_esg_capped * lambda_esg
                    )
                    er_annual = (1 + er_monthly) ** 12 - 1  # Compound annualization

                    results.append(
                        {
                            "symbol": symbol,
                            "date": row["date"],
                            "beta_date": beta_date,
                            "beta_market": beta_m,
                            "beta_esg": beta_esg,
                            "beta_market_capped": beta_m_capped,
                            "beta_esg_capped": beta_esg_capped,
                            "RF": row["RF"],
                            "ER_monthly": er_monthly,
                            "ER_annual": er_annual,
                            "lambda_market": lambda_market,
                            "lambda_esg": lambda_esg,
                        }
                    )
            else:
                # Time-varying betas (use most recent beta for each date)
                symbol_betas = symbol_betas.set_index("date")
                for _, row in rf.iterrows():
                    # Find most recent beta at or before this date
                    valid_betas = symbol_betas[symbol_betas.index <= row["date"]]
                    if len(valid_betas) == 0:
                        continue  # No beta available yet

                    latest_beta = valid_betas.iloc[-1]
                    beta_m = latest_beta["beta_market"]
                    beta_esg = latest_beta["beta_ESG"]
                    beta_date = latest_beta.name

                    # Cap betas if requested
                    beta_m_capped = (
                        np.clip(beta_m, -beta_market_cap, beta_market_cap)
                        if cap_betas
                        else beta_m
                    )
                    beta_esg_capped = (
                        np.clip(beta_esg, -beta_esg_cap, beta_esg_cap)
                        if cap_betas
                        else beta_esg
                    )

                    er_monthly = (
                        row["RF"]
                        + beta_m_capped * lambda_market
                        + beta_esg_capped * lambda_esg
                    )
                    er_annual = (1 + er_monthly) ** 12 - 1  # Compound annualization

                    results.append(
                        {
                            "symbol": symbol,
                            "date": row["date"],
                            "beta_date": beta_date,
                            "beta_market": beta_m,
                            "beta_esg": beta_esg,
                            "beta_market_capped": beta_m_capped,
                            "beta_esg_capped": beta_esg_capped,
                            "RF": row["RF"],
                            "ER_monthly": er_monthly,
                            "ER_annual": er_annual,
                            "lambda_market": lambda_market,
                            "lambda_esg": lambda_esg,
                        }
                    )

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(["symbol", "date"])

        # Log capping statistics
        if cap_betas:
            n_market_capped = (
                results_df["beta_market"] != results_df["beta_market_capped"]
            ).sum()
            n_esg_capped = (
                results_df["beta_esg"] != results_df["beta_esg_capped"]
            ).sum()
            if n_market_capped > 0 or n_esg_capped > 0:
                logger.info(f"Beta capping applied:")
                logger.info(f"  Market betas capped: {n_market_capped} observations")
                logger.info(f"  ESG betas capped: {n_esg_capped} observations")

        logger.info(f"Calculated expected returns:")
        logger.info(f"  Mean ER (annual): {results_df['ER_annual'].mean()*100:.2f}%")
        logger.info(
            f"  Median ER (annual): {results_df['ER_annual'].median()*100:.2f}%"
        )
        logger.info(
            f"  Range: [{results_df['ER_annual'].min()*100:.2f}%, "
            f"{results_df['ER_annual'].max()*100:.2f}%]"
        )

        return results_df
