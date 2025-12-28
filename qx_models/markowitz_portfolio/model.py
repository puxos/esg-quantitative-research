"""
Markowitz Portfolio Optimization Model Implementation

Mean-variance portfolio optimization using DSL-based constraints:
    Minimize: ½γw'Σw - μ'w

This model uses the ConstraintCompiler to transform high-level constraint
specifications (ConstraintDSL) into canonical QP matrices.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from qx.engine.base_model import BaseModel
from qx.engine.constraint_compiler import ConstraintCompiler
from qx.engine.constraint_dsl import ConstraintDSL
from qx.engine.qp_problem import QPProblem

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
        Optimize portfolio weights using mean-variance framework with DSL constraints.

        Args:
            inputs: Dictionary of input DataFrames
                - expected_returns: Expected returns (symbol, date, ER_monthly)
                - equity_prices: Historical prices for covariance estimation
                - risk_free: Risk-free rate time-series
            params: Model parameters from model.yaml
            **kwargs: Additional arguments
                - constraints: ConstraintDSL object (required, from constraint loaders)
                - optimization_date: Specific date or None for all dates

        Returns:
            DataFrame with optimal portfolio weights and statistics
        """
        logger.info("Starting Markowitz Portfolio Optimization (DSL-based)")

        # Extract inputs
        expected_returns_df = inputs["expected_returns"]
        equity_prices = inputs["equity_prices"]
        risk_free = inputs["risk_free"]

        # Extract ConstraintDSL from kwargs (passed via DAG context)
        constraint_dsl: ConstraintDSL = kwargs.get("constraint_dsl")

        # Validate required constraints input
        if constraint_dsl is None:
            raise ValueError(
                "Missing required 'constraint_dsl' input. "
                "Provide ConstraintDSL via constraint loader(s) in your DAG pipeline. "
                "Example loaders: basic_constraints, esg_constraints, sector_constraints"
            )

        if not isinstance(constraint_dsl, ConstraintDSL):
            raise TypeError(
                f"Expected ConstraintDSL, got {type(constraint_dsl).__name__}. "
                "Ensure your constraint loader returns ConstraintDSL."
            )

        num_symbols = (
            len(constraint_dsl.symbols)
            if constraint_dsl.symbols is not None
            else "<inferred>"
        )
        logger.info(
            f"✅ Received {len(constraint_dsl)} constraint(s) for {num_symbols} symbols"
        )
        logger.info(f"   {constraint_dsl.summary()}")

        # Extract parameters
        gamma = params.get("gamma", 4.0)
        lookback_months = params.get("lookback_months", 36)
        shrinkage_intensity = params.get("shrinkage_intensity", 0.25)
        turnover_penalty = params.get("turnover_penalty", 0.0)
        compute_frontier = params.get("compute_frontier", False)

        # Get optimization dates (all unique dates or specific date)
        # Check params first, then kwargs for backward compatibility
        print(f"🔍 DEBUG: params keys = {list(params.keys())}")
        print(f"🔍 DEBUG: kwargs keys = {list(kwargs.keys())}")
        print(
            f"🔍 DEBUG: optimization_date in params = {params.get('optimization_date')}"
        )
        print(
            f"🔍 DEBUG: optimization_date in kwargs = {kwargs.get('optimization_date')}"
        )

        optimization_date = params.get("optimization_date") or kwargs.get(
            "optimization_date"
        )
        if optimization_date is not None and isinstance(optimization_date, str):
            # Convert string date to Timestamp
            optimization_date = pd.Timestamp(optimization_date)

        if optimization_date is None:
            # Optimize for all available dates (monthly time series)
            all_dates = sorted(expected_returns_df["date"].unique())

            # Filter dates to ensure sufficient lookback for covariance estimation
            # Need at least lookback_months of data before optimization date
            min_date = equity_prices["date"].min()
            min_optimization_date = min_date + pd.DateOffset(months=lookback_months)

            optimization_dates = [d for d in all_dates if d >= min_optimization_date]

            logger.info(
                f"Filtered {len(all_dates)} dates to {len(optimization_dates)} dates with sufficient lookback"
            )
            logger.info(
                f"First optimization date: {optimization_dates[0] if optimization_dates else 'N/A'}"
            )
            logger.info(
                f"Last optimization date: {optimization_dates[-1] if optimization_dates else 'N/A'}"
            )
        else:
            # Single date optimization
            optimization_dates = [optimization_date]
            logger.info(f"Optimizing for single date: {optimization_date}")

        logger.info(f"Parameters: gamma={gamma}, lookback_months={lookback_months}")

        # Initialize constraint compiler
        compiler = ConstraintCompiler()

        # Loop through each optimization date
        all_results = []

        for opt_date in optimization_dates:
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimization date: {opt_date}")
            logger.info(f"{'='*60}")

            # 1. Prepare expected returns for this date
            logger.info("Step 1: Preparing expected returns")
            exp_ret_series = self._prepare_expected_returns(
                expected_returns_df, opt_date
            )

            # Validate universe alignment with constraints
            # ✅ DSL: Constraints are now universe-agnostic (symbols=None)
            # Symbol ordering comes from expected returns index
            symbols = exp_ret_series.index.tolist()

            # 2. Build covariance matrix
            logger.info("Step 2: Building covariance matrix")
            cov_matrix = self._build_covariance_matrix(
                equity_prices=equity_prices,
                risk_free=risk_free,
                tickers=exp_ret_series.index.tolist(),
                lookback_months=lookback_months,
                shrinkage_intensity=shrinkage_intensity,
                optimization_date=opt_date,
            )

            # Filter expected returns to match covariance matrix universe
            # (some stocks may be excluded due to insufficient price history)
            cov_symbols = cov_matrix.index.tolist()
            exp_ret_series = exp_ret_series.loc[cov_symbols]
            symbols = exp_ret_series.index.tolist()

            logger.info(
                f"Universe after covariance filtering: {len(symbols)} stocks (down from {len(exp_ret_series)+len(symbols)-len(cov_symbols)})"
            )

            # 3. Get current risk-free rate (for Sharpe calculation)
            rf_monthly = self._get_current_rf_monthly(risk_free, opt_date)

            # 4. Compile constraints → canonical QP
            logger.info("Step 3: Compiling constraints to canonical QP")

            if compute_frontier:
                logger.info("Step 4: Computing efficient frontier")
                gammas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

                for g in gammas:
                    qp_problem = compiler.compile_qp(
                        dsl=constraint_dsl,
                        mu=exp_ret_series.values,
                        Sigma=cov_matrix.values,
                        risk_aversion=g,
                        symbols=symbols,  # ✅ DSL: Pass symbols from expected returns
                    )

                    weights = self._solve_qp(qp_problem)

                    result_df = self._format_output(
                        weights=weights,
                        exp_ret=exp_ret_series,
                        cov_matrix=cov_matrix,
                        qp_problem=qp_problem,
                        optimization_date=opt_date,
                        gamma=g,
                        rf_monthly=rf_monthly,
                    )
                    all_results.append(result_df)

                logger.info(
                    f"Generated {len(gammas)} frontier portfolios for {opt_date}"
                )
            else:
                logger.info(f"Step 4: Optimizing portfolio (gamma={gamma})")

                qp_problem = compiler.compile_qp(
                    dsl=constraint_dsl,
                    mu=exp_ret_series.values,
                    Sigma=cov_matrix.values,
                    risk_aversion=gamma,
                    symbols=symbols,  # ✅ DSL: Pass symbols from expected returns
                )

                weights = self._solve_qp(qp_problem)

                result_df = self._format_output(
                    weights=weights,
                    exp_ret=exp_ret_series,
                    cov_matrix=cov_matrix,
                    qp_problem=qp_problem,  # ✅ DSL: Pass QPProblem for provenance
                    optimization_date=opt_date,
                    gamma=gamma,
                    rf_monthly=rf_monthly,
                )
                all_results.append(result_df)

        # Concatenate all results
        if len(all_results) > 1:
            final_result = pd.concat(all_results, ignore_index=True)
            logger.info(
                f"\n✅ Optimization complete: {len(all_results)} time periods, {len(final_result)} total positions"
            )
        else:
            final_result = all_results[0]
            logger.info(f"\n✅ Optimization complete: {len(final_result)} positions")

        return final_result

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

    def _prepare_market_betas_as_esg(
        self,
        market_betas_df: pd.DataFrame,
        tickers: list,
    ) -> pd.Series:
        """
        Prepare market betas as ESG proxy (fallback when no ESG betas available).

        Args:
            market_betas_df: Market betas from market_beta model
            tickers: List of tickers to include

        Returns:
            Series with symbol → neutral ESG beta (1.0)
        """
        df = market_betas_df.copy()

        # Rename columns to standard format
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename(columns={"ticker": "symbol"})

        # Use latest beta estimate for each stock
        df = df.sort_values(["symbol", "date"])
        df = df.groupby("symbol").tail(1)

        # Create neutral ESG betas (1.0) since we don't have actual ESG data
        # This makes portfolio ESG-neutral by default
        esg_beta = pd.Series(1.0, index=tickers)

        logger.info(
            f"Using market betas as fallback (neutral ESG = 1.0) for {len(esg_beta)} stocks"
        )
        logger.info(f"  All ESG betas set to 1.0 (neutral)")

        return esg_beta

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

        print(f"\n🔍 Building covariance matrix:")
        print(f"   Optimization date: {optimization_date}")
        print(f"   Lookback period: {lookback_months} months")
        print(f"   Start date: {start_date}")
        print(f"   Number of tickers: {len(tickers)}")

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
            monthly = ticker_prices["adj_close"].resample("ME").last()
            returns = monthly.pct_change().dropna()

            if len(returns) > 0:
                returns_dict[ticker] = returns

        if not returns_dict:
            raise ValueError("No returns data found for covariance estimation")

        # Create returns panel
        returns_df = pd.DataFrame(returns_dict)

        print(
            f"   Initial panel: {len(returns_df)} dates × {len(returns_df.columns)} stocks"
        )
        print(
            f"   Missing values: {returns_df.isna().sum().sum()} / {returns_df.size} ({returns_df.isna().sum().sum()/returns_df.size*100:.1f}%)"
        )

        # Drop stocks with too many missing values (>50% missing)
        missing_pct_by_stock = returns_df.isna().sum() / len(returns_df)
        stocks_to_keep = missing_pct_by_stock[missing_pct_by_stock < 0.5].index
        returns_df = returns_df[stocks_to_keep]

        print(
            f"   After dropping sparse stocks: {len(returns_df.columns)} stocks (removed {len(returns_dict) - len(stocks_to_keep)} stocks)"
        )

        # Fill remaining missing values with 0 (assumes no return when data missing)
        returns_df = returns_df.fillna(0.0)

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

    def _solve_qp(self, qp: QPProblem) -> pd.Series:
        """
        Solve canonical QP problem using CVXPY.

        Args:
            qp: Canonical QP problem with matrices (P,q,G,h,A,b,l,u)

        Returns:
            Optimal weights as pandas Series

        Raises:
            RuntimeError: If optimization fails
        """
        n = qp.n_vars

        # Decision variable
        w = cp.Variable(n)

        # Convert matrices to dense and create CVXPY parameters (not constants)
        # This avoids scipy sparse matrix .A attribute issues in CVXPY
        P_dense = np.array(qp.P, dtype=np.float64)
        q_dense = np.array(qp.q, dtype=np.float64).ravel()

        # Ensure P is symmetric
        P_dense = 0.5 * (P_dense + P_dense.T)

        # Create parameters (not embedded constants)
        P_param = cp.Parameter(P_dense.shape, value=P_dense, PSD=True)
        q_param = cp.Parameter(q_dense.shape, value=q_dense)

        # Objective: minimize ½w'Pw + q'w
        objective = 0.5 * cp.quad_form(w, P_param) + q_param @ w

        # Constraints
        constraints = []

        # Equality constraints: Aw = b
        if qp.n_eq > 0:
            A_dense = np.array(qp.A, dtype=np.float64)
            b_dense = np.array(qp.b, dtype=np.float64).ravel()
            A_param = cp.Parameter(A_dense.shape, value=A_dense)
            b_param = cp.Parameter(b_dense.shape, value=b_dense)
            constraints.append(A_param @ w == b_param)

        # Inequality constraints: Gw ≤ h
        if qp.n_ineq > 0:
            G_dense = np.array(qp.G, dtype=np.float64)
            h_dense = np.array(qp.h, dtype=np.float64).ravel()
            G_param = cp.Parameter(G_dense.shape, value=G_dense)
            h_param = cp.Parameter(h_dense.shape, value=h_dense)
            constraints.append(G_param @ w <= h_param)

        # Box constraints: l ≤ w ≤ u
        l_dense = np.array(qp.l, dtype=np.float64).ravel()
        u_dense = np.array(qp.u, dtype=np.float64).ravel()
        l_param = cp.Parameter(l_dense.shape, value=l_dense)
        u_param = cp.Parameter(u_dense.shape, value=u_dense)
        constraints.append(w >= l_param)
        constraints.append(w <= u_param)

        # Solve
        prob = cp.Problem(cp.Minimize(objective), constraints)
        try:
            # Try OSQP first (faster for QP)
            prob.solve(solver=cp.OSQP, verbose=False)
        except (AttributeError, Exception) as e:
            # Fallback to SCS (more robust)
            logger.warning(
                f"OSQP failed ({str(e)[:50]}...), falling back to SCS solver"
            )
            prob.solve(solver=cp.SCS, verbose=False)

        if w.value is None:
            logger.error(f"  ❌ Optimization failed!")
            logger.error(f"  Problem status: {prob.status}")
            logger.error(f"  {qp.summary()}")
            raise RuntimeError(f"Optimization failed: {prob.status}")

        weights = pd.Series(np.array(w.value).ravel(), index=qp.symbols)

        # Log optimization stats
        portfolio_return = -float(qp.q @ weights.values)  # q = -mu
        portfolio_vol = np.sqrt(
            weights.values @ qp.P @ weights.values / qp.meta.get("risk_aversion", 1.0)
        )
        logger.info(
            f"  Portfolio return: {portfolio_return:.4f} ({portfolio_return*12*100:.2f}% annual)"
        )
        logger.info(
            f"  Portfolio vol: {portfolio_vol:.4f} ({portfolio_vol*np.sqrt(12)*100:.2f}% annual)"
        )
        logger.info(f"  Num positions: {(weights > 0.001).sum()}")

        return weights

    def _format_output(
        self,
        weights: pd.Series,
        exp_ret: pd.Series,
        cov_matrix: pd.DataFrame,
        qp_problem: QPProblem,
        optimization_date: pd.Timestamp,
        gamma: float,
        rf_monthly: float,
    ) -> pd.DataFrame:
        """
        Format optimization output.

        Args:
            weights: Optimal weights
            exp_ret: Expected returns
            cov_matrix: Covariance matrix
            qp_problem: Compiled QP problem (for provenance)
            optimization_date: Optimization date
            gamma: Risk aversion parameter
            rf_monthly: Risk-free rate (monthly)

        Returns:
            DataFrame with portfolio positions and statistics
        """
        # Filter to non-zero positions
        non_zero_weights = weights[weights > 1e-6].copy()

        if len(non_zero_weights) == 0:
            logger.warning("  ⚠️  No non-zero positions found!")
            return pd.DataFrame()

        # Build output DataFrame
        output = pd.DataFrame(
            {
                "symbol": non_zero_weights.index,
                "weight": non_zero_weights.values,
                "optimization_date": optimization_date,
                "exp_return_monthly": [
                    exp_ret[ticker] for ticker in non_zero_weights.index
                ],
                "exp_return_annual": [
                    (1 + exp_ret[ticker]) ** 12 - 1 for ticker in non_zero_weights.index
                ],
            }
        )

        # Portfolio-level statistics
        portfolio_return = weights @ exp_ret
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        portfolio_sharpe = (
            (portfolio_return - rf_monthly) / portfolio_vol
            if portfolio_vol > 0
            else 0.0
        )

        output["portfolio_return"] = portfolio_return
        output["portfolio_vol"] = portfolio_vol
        output["portfolio_sharpe"] = portfolio_sharpe
        output["gamma"] = gamma

        # ✅ DSL: No legacy ESG/sector extraction needed
        # Constraint provenance is in qp_problem.meta

        return output

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
            logger.error(f"  ❌ Optimization failed!")
            logger.error(f"  Problem status: {prob.status}")
            logger.error(f"  Num assets: {len(tickers)}")
            logger.error(f"  Expected returns shape: {mu.shape}")
            logger.error(f"  Covariance matrix shape: {Sigma.shape}")
            logger.error(
                f"  Covariance matrix condition number: {np.linalg.cond(Sigma):.2e}"
            )
            logger.error(f"  Risk aversion (gamma): {gamma}")
            logger.error(f"  Long only: {long_only}")
            logger.error(f"  Position max: {position_max}")
            logger.error(f"  ESG bounds: {esg_bounds}")
            logger.error(f"  ESG beta: {beta_esg}")
            raise RuntimeError(f"Optimization failed: {prob.status}")

        weights = pd.Series(np.array(w.value).ravel(), index=tickers)

        logger.info(f"  Solver: ECOS (CVXPY)")
        logger.info(f"  Objective: {prob.value:.6f}")
        logger.info(f"  Active positions: {(weights.abs() > 1e-6).sum()}")

        return weights

    def _format_output_legacy(
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
        Format optimization results as output DataFrame (LEGACY path).

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
