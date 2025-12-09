"""
Markowitz Portfolio Optimization Model Implementation

Mean-variance portfolio optimization with ESG controls:
    Maximize: μ'w - 0.5*γ*w'Σw - λ*||w - w_prev||₁
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from qx.engine.base_model import BaseModel

logger = logging.getLogger(__name__)


class MarkowitzPortfolioModel(BaseModel):
    """
    Markowitz Mean-Variance Portfolio Optimization with ESG Control

    Solves the classic Markowitz problem with additional ESG constraints:

        Maximize: μ'w - 0.5*γ*w'Σw - λ*||w - w_prev||₁

        Subject to:
            - Budget: Σw = 1
            - Long-only: w ≥ 0
            - Position limits: w ≤ w_max
            - ESG exposure: L_ESG ≤ β_ESG'w ≤ U_ESG
            - Sector concentration: Σw[sector] ≤ cap

    Key Features:
        - Ledoit-Wolf shrinkage covariance estimation
        - ESG-neutral or ESG-tilted portfolio construction
        - Sector diversification constraints
        - Turnover cost modeling
        - Efficient frontier generation
        - CVXPY-based convex optimization
    """

    def run_impl(
        self,
        inputs: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Optimize portfolio weights using mean-variance framework.

        Args:
            inputs: Dictionary of input DataFrames
                - expected_returns: Expected returns (symbol, date, ER_monthly)
                - two_factor_betas: Factor betas (symbol, beta_esg)
                - equity_prices: Historical prices for covariance estimation
                - risk_free: Risk-free rate time-series
                - sector_mapping (optional): Sector classification
            params: Model parameters from model.yaml
            **kwargs: Additional arguments (e.g., optimization_date)

        Returns:
            DataFrame with optimal portfolio weights and statistics
        """
        logger.info("Starting Markowitz Portfolio Optimization")

        # Extract inputs
        expected_returns_df = inputs["expected_returns"]
        two_factor_betas_df = inputs["two_factor_betas"]
        equity_prices = inputs["equity_prices"]
        risk_free = inputs["risk_free"]
        sector_mapping_df = inputs.get("sector_mapping")

        # Extract parameters
        gamma = params.get("gamma", 4.0)
        long_only = params.get("long_only", True)
        position_max = params.get("position_max", 0.10)
        esg_neutral = params.get("esg_neutral", False)
        esg_lower_bound = params.get("esg_lower_bound")
        esg_upper_bound = params.get("esg_upper_bound")
        sector_constraints = params.get("sector_constraints", False)
        sector_cap = params.get("sector_cap", 0.30)
        lookback_months = params.get("lookback_months", 36)
        shrinkage_intensity = params.get("shrinkage_intensity", 0.25)
        turnover_penalty = params.get("turnover_penalty", 0.0)
        compute_frontier = params.get("compute_frontier", False)

        # Get optimization date from kwargs
        optimization_date = kwargs.get("optimization_date")
        if optimization_date is None:
            optimization_date = expected_returns_df["date"].max()

        logger.info(f"Optimization date: {optimization_date}")
        logger.info(
            f"Parameters: gamma={gamma}, position_max={position_max}, "
            f"esg_neutral={esg_neutral}"
        )

        # 1. Prepare expected returns
        logger.info("Step 1: Preparing expected returns")
        exp_ret_series = self._prepare_expected_returns(
            expected_returns_df, optimization_date
        )

        # 2. Prepare ESG betas
        logger.info("Step 2: Preparing ESG betas")
        esg_beta_series = self._prepare_esg_betas(
            two_factor_betas_df, exp_ret_series.index.tolist()
        )

        # 3. Build covariance matrix
        logger.info("Step 3: Building covariance matrix")
        cov_matrix = self._build_covariance_matrix(
            equity_prices=equity_prices,
            risk_free=risk_free,
            tickers=exp_ret_series.index.tolist(),
            lookback_months=lookback_months,
            shrinkage_intensity=shrinkage_intensity,
            optimization_date=optimization_date,
        )

        # 4. Prepare sector mapping (if applicable)
        sector_map = None
        sector_caps_dict = None
        if sector_constraints and sector_mapping_df is not None:
            logger.info("Step 4: Preparing sector constraints")
            sector_map = self._prepare_sector_mapping(
                sector_mapping_df, exp_ret_series.index.tolist()
            )
            # Apply sector cap to all sectors
            sector_caps_dict = {
                sector: sector_cap for sector in sector_map.unique() if pd.notna(sector)
            }
            logger.info(
                f"  Applying sector cap of {sector_cap:.1%} to {len(sector_caps_dict)} sectors"
            )

        # 5. Determine ESG bounds
        esg_bounds = self._determine_esg_bounds(
            esg_neutral, esg_lower_bound, esg_upper_bound
        )

        # 6. Get current risk-free rate (for Sharpe calculation)
        rf_monthly = self._get_current_rf_monthly(risk_free, optimization_date)

        # 7. Optimize portfolio
        if compute_frontier:
            logger.info("Step 7: Computing efficient frontier")
            gammas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
            results_list = []

            for g in gammas:
                weights = self._optimize_portfolio(
                    exp_ret=exp_ret_series,
                    cov_matrix=cov_matrix,
                    esg_beta=esg_beta_series,
                    gamma=g,
                    long_only=long_only,
                    position_max=position_max,
                    esg_bounds=esg_bounds,
                    sector_map=sector_map,
                    sector_caps=sector_caps_dict,
                )

                result_df = self._format_output(
                    weights=weights,
                    exp_ret=exp_ret_series,
                    cov_matrix=cov_matrix,
                    esg_beta=esg_beta_series,
                    sector_map=sector_map,
                    optimization_date=optimization_date,
                    gamma=g,
                    esg_bounds=esg_bounds,
                    rf_monthly=rf_monthly,
                )
                results_list.append(result_df)

            result_df = pd.concat(results_list, ignore_index=True)
            logger.info(f"Generated {len(gammas)} frontier portfolios")
        else:
            logger.info(f"Step 7: Optimizing portfolio (gamma={gamma})")
            weights = self._optimize_portfolio(
                exp_ret=exp_ret_series,
                cov_matrix=cov_matrix,
                esg_beta=esg_beta_series,
                gamma=gamma,
                long_only=long_only,
                position_max=position_max,
                esg_bounds=esg_bounds,
                sector_map=sector_map,
                sector_caps=sector_caps_dict,
            )

            result_df = self._format_output(
                weights=weights,
                exp_ret=exp_ret_series,
                cov_matrix=cov_matrix,
                esg_beta=esg_beta_series,
                sector_map=sector_map,
                optimization_date=optimization_date,
                gamma=gamma,
                esg_bounds=esg_bounds,
                rf_monthly=rf_monthly,
            )

        logger.info(f"Portfolio optimization completed: {len(result_df)} positions")

        return result_df

    def _prepare_expected_returns(
        self,
        expected_returns_df: pd.DataFrame,
        optimization_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Prepare expected returns series for optimization date.

        Args:
            expected_returns_df: Expected returns from factor model
            optimization_date: Target optimization date

        Returns:
            Series with symbol → expected monthly return
        """
        df = expected_returns_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Filter to optimization date (or nearest)
        df_date = df[df["date"] == optimization_date]
        if df_date.empty:
            # Use nearest date
            nearest_date = df["date"].iloc[
                (df["date"] - optimization_date).abs().argmin()
            ]
            df_date = df[df["date"] == nearest_date]
            logger.warning(
                f"No data for {optimization_date}, using nearest: {nearest_date}"
            )

        # Rename columns to standard format
        if "ticker" in df_date.columns and "symbol" not in df_date.columns:
            df_date = df_date.rename(columns={"ticker": "symbol"})

        exp_ret = df_date.set_index("symbol")["ER_monthly"]

        logger.info(f"Prepared expected returns for {len(exp_ret)} stocks")
        logger.info(
            f"  Mean ER: {exp_ret.mean():.4f} ({exp_ret.mean()*12*100:.2f}% annual)"
        )
        logger.info(f"  Range: [{exp_ret.min():.4f}, {exp_ret.max():.4f}]")

        return exp_ret

    def _prepare_esg_betas(
        self,
        two_factor_betas_df: pd.DataFrame,
        tickers: list,
    ) -> pd.Series:
        """
        Prepare ESG betas for optimization.

        Args:
            two_factor_betas_df: Two-factor betas from regression
            tickers: List of tickers to include

        Returns:
            Series with symbol → ESG beta
        """
        df = two_factor_betas_df.copy()

        # Rename columns to standard format
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename(columns={"ticker": "symbol"})

        # Use latest beta estimate for each stock
        df = df.sort_values(["symbol", "date"])
        df = df.groupby("symbol").tail(1)

        esg_beta = df.set_index("symbol")["beta_esg"]
        esg_beta = esg_beta.reindex(tickers).fillna(0.0)

        logger.info(f"Prepared ESG betas for {len(esg_beta)} stocks")
        logger.info(f"  Mean β_ESG: {esg_beta.mean():.4f}")
        logger.info(f"  Range: [{esg_beta.min():.4f}, {esg_beta.max():.4f}]")

        return esg_beta

    def _build_covariance_matrix(
        self,
        equity_prices: pd.DataFrame,
        risk_free: pd.DataFrame,
        tickers: list,
        lookback_months: int,
        shrinkage_intensity: float,
        optimization_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Build shrinkage covariance matrix from historical returns.

        Uses Ledoit-Wolf style shrinkage: Σ = (1-δ)*Σ_sample + δ*Σ_diagonal

        Args:
            equity_prices: Historical equity prices
            risk_free: Risk-free rate time-series
            tickers: List of tickers to include
            lookback_months: Number of months for covariance estimation
            shrinkage_intensity: Shrinkage parameter δ ∈ [0, 1]
            optimization_date: End date for covariance estimation

        Returns:
            Covariance matrix (DataFrame, ticker × ticker)
        """
        # Calculate lookback start date
        start_date = optimization_date - pd.DateOffset(months=lookback_months)

        # Load and prepare returns for each ticker
        returns_dict = {}

        for ticker in tickers:
            # Filter prices for this ticker
            ticker_prices = equity_prices[equity_prices["symbol"] == ticker].copy()
            ticker_prices["date"] = pd.to_datetime(ticker_prices["date"])
            ticker_prices = ticker_prices.sort_values("date")
            ticker_prices = ticker_prices[
                (ticker_prices["date"] >= start_date)
                & (ticker_prices["date"] <= optimization_date)
            ]

            if len(ticker_prices) < 10:
                continue  # Need minimum observations

            # Resample to monthly
            ticker_prices = ticker_prices.set_index("date")
            monthly = ticker_prices["adjClose"].resample("ME").last()
            returns = monthly.pct_change().dropna()

            if len(returns) > 0:
                returns_dict[ticker] = returns

        if not returns_dict:
            raise ValueError("No returns data found for covariance estimation")

        # Create returns panel
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()  # Drop dates with any missing data

        logger.info(
            f"Built returns panel: {len(returns_df)} months, {len(returns_df.columns)} stocks"
        )

        if len(returns_df) < 12:
            raise ValueError(
                f"Insufficient data for covariance: {len(returns_df)} months < 12 minimum"
            )

        # Calculate sample covariance
        cov_sample = returns_df.cov()

        # Build diagonal target (variances only, zero correlations)
        cov_diagonal = pd.DataFrame(
            np.diag(np.diag(cov_sample.values)),
            index=cov_sample.index,
            columns=cov_sample.columns,
        )

        # Apply shrinkage
        cov_shrunk = (
            1.0 - shrinkage_intensity
        ) * cov_sample + shrinkage_intensity * cov_diagonal

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(cov_shrunk.values)
        if eigvals.min() < 0:
            eps = 1e-8 - eigvals.min()
            cov_shrunk.values[:] = cov_shrunk.values + np.eye(cov_shrunk.shape[0]) * eps
            logger.info(f"  Applied PSD correction: eps={eps:.2e}")

        logger.info(f"Built covariance matrix: {cov_shrunk.shape[0]} assets")
        logger.info(f"  Shrinkage intensity: {shrinkage_intensity:.2f}")
        logger.info(f"  Eigenvalues: [{eigvals.min():.6f}, {eigvals.max():.6f}]")

        return cov_shrunk

    def _prepare_sector_mapping(
        self,
        sector_mapping_df: pd.DataFrame,
        tickers: list,
    ) -> pd.Series:
        """
        Prepare sector mapping for constraints.

        Args:
            sector_mapping_df: Sector metadata
            tickers: List of tickers to include

        Returns:
            Series with symbol → sector
        """
        df = sector_mapping_df.copy()

        # Rename columns to standard format
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename(columns={"ticker": "symbol"})

        sector_map = df.set_index("symbol")["sector"]
        sector_map = sector_map.reindex(tickers)

        logger.info(f"Prepared sector mapping for {sector_map.notna().sum()} stocks")
        logger.info(f"  Sectors: {sector_map.nunique()} unique")

        return sector_map

    def _determine_esg_bounds(
        self,
        esg_neutral: bool,
        esg_lower_bound: Optional[float],
        esg_upper_bound: Optional[float],
    ) -> Optional[Tuple[float, float]]:
        """
        Determine ESG bounds for constraint.

        Args:
            esg_neutral: Whether to enforce ESG neutrality
            esg_lower_bound: Custom lower bound
            esg_upper_bound: Custom upper bound

        Returns:
            Tuple (L_ESG, U_ESG) or None if no constraint
        """
        if esg_lower_bound is not None and esg_upper_bound is not None:
            bounds = (esg_lower_bound, esg_upper_bound)
            logger.info(f"ESG bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}] (custom)")
            return bounds

        if esg_neutral:
            bounds = (-0.05, 0.05)
            logger.info(f"ESG bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}] (neutral)")
            return bounds

        logger.info("ESG bounds: None (unconstrained)")
        return None

    def _get_current_rf_monthly(
        self,
        risk_free: pd.DataFrame,
        optimization_date: pd.Timestamp,
    ) -> float:
        """
        Get current monthly risk-free rate.

        Args:
            risk_free: Risk-free rate DataFrame
            optimization_date: Optimization date

        Returns:
            Monthly risk-free rate (decimal)
        """
        df = risk_free.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= optimization_date]

        if df.empty:
            logger.warning("No risk-free rate data available, using 0.0")
            return 0.0

        latest_rf = df.iloc[-1]["rate"]  # Annual percentage
        rf_monthly = latest_rf / 100 / 12  # Convert to monthly decimal

        logger.info(f"Current RF: {latest_rf:.2f}% annual = {rf_monthly:.6f} monthly")

        return rf_monthly

    def _optimize_portfolio(
        self,
        exp_ret: pd.Series,
        cov_matrix: pd.DataFrame,
        esg_beta: pd.Series,
        gamma: float,
        long_only: bool,
        position_max: float,
        esg_bounds: Optional[Tuple[float, float]],
        sector_map: Optional[pd.Series],
        sector_caps: Optional[dict],
    ) -> pd.Series:
        """
        Solve Markowitz optimization problem using CVXPY.

        Args:
            exp_ret: Expected returns (symbol → ER)
            cov_matrix: Covariance matrix (symbol × symbol)
            esg_beta: ESG betas (symbol → β_ESG)
            gamma: Risk aversion parameter
            long_only: Restrict to long-only positions
            position_max: Maximum position size
            esg_bounds: ESG exposure bounds (L, U) or None
            sector_map: Sector mapping (symbol → sector)
            sector_caps: Sector caps {sector: cap}

        Returns:
            Optimal weights (Series, symbol → weight)
        """
        # Align all inputs
        tickers = exp_ret.index.tolist()
        cov_matrix = cov_matrix.loc[tickers, tickers]
        esg_beta = esg_beta.reindex(tickers).fillna(0.0)
        return self._optimize_cvxpy(
            exp_ret,
            cov_matrix,
            esg_beta,
            gamma,
            long_only,
            position_max,
            esg_bounds,
            sector_map,
            sector_caps,
        )

    def _optimize_cvxpy(
        self,
        exp_ret: pd.Series,
        cov_matrix: pd.DataFrame,
        esg_beta: pd.Series,
        gamma: float,
        long_only: bool,
        position_max: float,
        esg_bounds: Optional[Tuple[float, float]],
        sector_map: Optional[pd.Series],
        sector_caps: Optional[dict],
    ) -> pd.Series:
        """CVXPY optimization implementation"""
        tickers = exp_ret.index.tolist()
        n = len(tickers)

        mu = exp_ret.values
        Sigma = cov_matrix.values
        beta_esg = esg_beta.values

        # Decision variable
        w = cp.Variable(n)

        # Objective: μ'w - 0.5*γ*w'Σw
        quad_form = cp.quad_form(w, Sigma)
        objective = mu @ w - 0.5 * gamma * quad_form

        # Constraints
        constraints = [cp.sum(w) == 1]  # Budget

        if long_only:
            constraints += [w >= 0]

        if position_max is not None:
            constraints += [w <= position_max]

        if esg_bounds is not None:
            L_esg, U_esg = esg_bounds
            constraints += [beta_esg @ w >= L_esg, beta_esg @ w <= U_esg]

        if sector_map is not None and sector_caps is not None:
            sec = sector_map.reindex(tickers)
            for sector, cap in sector_caps.items():
                idx = np.where(sec.values == sector)[0]
                if len(idx) > 0:
                    constraints += [cp.sum(w[idx]) <= cap]

        # Solve with ECOS (default CVXPY solver for QP)
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if w.value is None:
            raise RuntimeError("Optimization failed with CVXPY")

        weights = pd.Series(np.array(w.value).ravel(), index=tickers)

        logger.info(f"  Solver: ECOS (CVXPY)")
        logger.info(f"  Objective: {prob.value:.6f}")
        logger.info(f"  Active positions: {(weights.abs() > 1e-6).sum()}")

        return weights

    def _format_output(
        self,
        weights: pd.Series,
        exp_ret: pd.Series,
        cov_matrix: pd.DataFrame,
        esg_beta: pd.Series,
        sector_map: Optional[pd.Series],
        optimization_date: pd.Timestamp,
        gamma: float,
        esg_bounds: Optional[Tuple[float, float]],
        rf_monthly: float,
    ) -> pd.DataFrame:
        """
        Format optimization results as output DataFrame.

        Args:
            weights: Optimal portfolio weights
            exp_ret: Expected returns
            cov_matrix: Covariance matrix
            esg_beta: ESG betas
            sector_map: Sector mapping
            optimization_date: Optimization date
            gamma: Risk aversion parameter
            esg_bounds: ESG bounds
            rf_monthly: Risk-free rate (monthly)

        Returns:
            DataFrame with portfolio weights and statistics
        """
        # Filter to active positions
        active_weights = weights[weights.abs() > 1e-6].copy()
        tickers = active_weights.index.tolist()

        # Calculate portfolio statistics
        w = weights.values
        mu = exp_ret.reindex(weights.index).values
        Sigma = cov_matrix.loc[weights.index, weights.index].values
        beta_esg_vec = esg_beta.reindex(weights.index).values

        portfolio_return_monthly = float(mu @ w)
        portfolio_var_monthly = float(w @ Sigma @ w)
        portfolio_vol_monthly = np.sqrt(portfolio_var_monthly)
        portfolio_sharpe = (
            portfolio_return_monthly - rf_monthly
        ) / portfolio_vol_monthly
        portfolio_esg_exposure = float(beta_esg_vec @ w)

        # Annualize
        portfolio_return_annual = portfolio_return_monthly * 12
        portfolio_vol_annual = portfolio_vol_monthly * np.sqrt(12)

        # Concentration
        top10_weight = active_weights.nlargest(10).sum()
        n_positions = len(active_weights)

        # Build output DataFrame
        results = []
        for ticker in tickers:
            row = {
                # Identifiers
                "symbol": ticker,
                "optimization_date": optimization_date,
                # Allocation
                "weight": active_weights[ticker],
                # Expected metrics (per stock)
                "exp_return_monthly": exp_ret.loc[ticker],
                "exp_return_annual": exp_ret.loc[ticker] * 12,
                # Risk metrics (per stock)
                "esg_beta": esg_beta.loc[ticker],
                "sector": sector_map.loc[ticker] if sector_map is not None else None,
                # Portfolio-level statistics
                "portfolio_return_monthly": portfolio_return_monthly,
                "portfolio_return_annual": portfolio_return_annual,
                "portfolio_vol_monthly": portfolio_vol_monthly,
                "portfolio_vol_annual": portfolio_vol_annual,
                "portfolio_sharpe": portfolio_sharpe,
                "portfolio_esg_exposure": portfolio_esg_exposure,
                "portfolio_concentration_top10": top10_weight,
                "n_positions": n_positions,
                # Optimization parameters
                "gamma": gamma,
                "esg_lower_bound": esg_bounds[0] if esg_bounds else None,
                "esg_upper_bound": esg_bounds[1] if esg_bounds else None,
            }
            results.append(row)

        result_df = pd.DataFrame(results)

        logger.info(f"Portfolio statistics:")
        logger.info(
            f"  Expected return: {portfolio_return_monthly:.4f} monthly "
            f"({portfolio_return_annual*100:.2f}% annual)"
        )
        logger.info(
            f"  Volatility: {portfolio_vol_monthly:.4f} monthly "
            f"({portfolio_vol_annual*100:.2f}% annual)"
        )
        logger.info(f"  Sharpe ratio: {portfolio_sharpe:.4f}")
        logger.info(f"  ESG exposure: {portfolio_esg_exposure:.4f}")
        logger.info(f"  Active positions: {n_positions}")
        logger.info(f"  Top 10 concentration: {top10_weight*100:.2f}%")

        return result_df
