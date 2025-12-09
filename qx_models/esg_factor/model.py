"""
ESG Factor Model Implementation

Migrated from legacy ESGFactorBuilder (src/esg/esg_factor.py)
to Qx BaseModel architecture.

Key Changes:
- Extends BaseModel (automatic input loading, output writing, metadata)
- Uses model.yaml for configuration (no __init__ needed)
- run_impl() implements core business logic only
- Returns DataFrame with output columns (BaseModel adds metadata)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from qx.engine.base_model import BaseModel

logger = logging.getLogger(__name__)


class ESGFactorModel(BaseModel):
    """
    ESG Factor Model

    Constructs long-short factor portfolios based on ESG signals:
    - Level factors: ESG, E, S, G scores
    - Momentum factor: Changes in ESG scores

    Features:
    - Proper signal lagging to avoid look-ahead bias
    - Sector-neutral ranking (optional)
    - Value-weighted or equal-weighted portfolios
    - Automatic risk-free rate adjustment
    - Excess returns (risk-free adjusted)
    """

    def run_impl(
        self, inputs: Dict[str, pd.DataFrame], params: Dict, **kwargs
    ) -> pd.DataFrame:
        """
        Build ESG factors from curated inputs

        Args:
            inputs: Dict with keys:
                - equity_prices: Daily OHLCV data
                - esg_scores: Annual ESG scores with monthly observations
                - risk_free: Daily risk-free rates
            params: Dict with keys:
                - quantile: Quantile for long/short legs (default: 0.2)
                - sector_neutral: Whether to rank within sectors (default: False)
                - lag_signal: Number of periods to lag signal (default: 1)
                - weighting: "equal" or "value" (default: "equal")
                - esg_annual_lag: Months to lag ESG scores (default: 12)
                - return_legs: Whether to return long/short legs (default: True)
            **kwargs: Additional args (e.g., run_id)

        Returns:
            DataFrame with columns: date, factor_name, factor_return[, long_return, short_return]

            BaseModel will add: model, model_version, featureset_id, run_id, run_ts
        """
        logger.info("=" * 80)
        logger.info("ESG Factor Model - Execution Started")
        logger.info("=" * 80)

        # Extract inputs
        prices_df = inputs["equity_prices"]
        esg_df = inputs["esg_scores"]
        rf_df = inputs["risk_free"]
        universe_filter_df = inputs.get("universe_filter")  # Optional

        # Filter to universe if provided
        if universe_filter_df is not None:
            logger.info("\n🔍 Filtering to universe_filter (passed tickers only)...")
            filtered_tickers = universe_filter_df[
                universe_filter_df["passed_filter"] == True
            ]["ticker"].unique()
            logger.info(f"  Universe filter: {len(filtered_tickers)} tickers passed")

            # Filter prices and ESG scores
            prices_before = len(prices_df["symbol"].unique())
            esg_before = len(esg_df["ticker"].unique())

            prices_df = prices_df[prices_df["symbol"].isin(filtered_tickers)].copy()
            esg_df = esg_df[esg_df["ticker"].isin(filtered_tickers)].copy()

            prices_after = len(prices_df["symbol"].unique())
            esg_after = len(esg_df["ticker"].unique())

            logger.info(f"  Filtered prices: {prices_before} → {prices_after} tickers")
            logger.info(f"  Filtered ESG: {esg_before} → {esg_after} tickers")
        else:
            logger.info("\nℹ️  No universe_filter provided, using all tickers")

        # Extract parameters
        quantile = params["quantile"]
        sector_neutral = params["sector_neutral"]
        lag_signal = params["lag_signal"]
        weighting = params["weighting"]
        esg_annual_lag = params["esg_annual_lag"]
        return_legs = params["return_legs"]

        logger.info(f"Parameters:")
        logger.info(f"  Quantile: {quantile} (top/bottom {quantile*100:.0f}%)")
        logger.info(f"  Sector neutral: {sector_neutral}")
        logger.info(f"  Signal lag: {lag_signal} periods")
        logger.info(f"  Weighting: {weighting}")
        logger.info(f"  ESG annual lag: {esg_annual_lag} months")
        logger.info(f"  Return legs: {return_legs}")

        # Resample daily data to monthly (end of month)
        logger.info("\n📅 Resampling daily data to monthly...")
        prices_monthly = self._resample_to_monthly(prices_df)
        rf_monthly = self._resample_rf_to_monthly(rf_df)

        logger.info(f"  Prices: {len(prices_monthly)} observations")
        logger.info(f"  Risk-free: {len(rf_monthly)} observations")

        # Compute monthly returns
        logger.info("\n💹 Computing monthly returns...")
        returns_df = self._compute_monthly_returns(prices_monthly)
        logger.info(f"  Returns: {len(returns_df)} observations")

        # Compute weights for value-weighting if requested
        weights_df = None
        if weighting == "value":
            logger.info("\n⚖️ Computing market cap weights...")
            weights_df = self._compute_market_cap_weights(prices_monthly)
            if weights_df is not None:
                logger.info(f"  Weights: {len(weights_df)} observations")
            else:
                logger.warning(
                    "  No volume data available, falling back to equal weighting"
                )

        # Convert to excess returns
        logger.info("\n📊 Converting to excess returns...")
        panel_excess = self._to_excess_returns(returns_df, rf_monthly)
        logger.info(f"  Excess returns: {len(panel_excess)} observations")

        # Prepare ESG scores (already monthly from builder)
        logger.info("\n🌿 Preparing ESG scores...")
        esg_prepared = self._prepare_esg_scores(esg_df)

        # Apply annual ESG lag
        logger.info(f"\n⏳ Applying {esg_annual_lag}-month ESG lag...")
        esg_lagged = self._apply_annual_esg_lag(esg_prepared, lag_months=esg_annual_lag)
        logger.info(f"  After lag: {len(esg_lagged)} observations")

        # Load sector mapping if sector-neutral
        sector_map = None
        if sector_neutral:
            logger.info("\n🏭 Loading sector mapping...")
            sector_map = self._get_sector_mapping(esg_lagged)
            if sector_map is not None:
                logger.info(
                    f"  Mapped {len(sector_map)} tickers to {sector_map.nunique()} sectors"
                )
            else:
                logger.warning(
                    "  No sector data available, using cross-sectional ranking"
                )

        # Build level factors (ESG, E, S, G)
        logger.info("\n🏗️ Building level factors...")
        factors = []
        for col in [
            "esg_score",
            "environmental_pillar_score",
            "social_pillar_score",
            "governance_pillar_score",
        ]:
            factor_short_name = {
                "esg_score": "ESG",
                "environmental_pillar_score": "E",
                "social_pillar_score": "S",
                "governance_pillar_score": "G",
            }[col]

            logger.info(f"  Building {factor_short_name} factor...")
            sig = esg_lagged[[col]].rename(columns={col: factor_short_name})

            factor_df = self._build_long_short_factor(
                panel_excess=panel_excess,
                signal_df=sig,
                weights_df=weights_df,
                sector_map=sector_map,
                quantile=quantile,
                lag_signal=lag_signal,
                return_legs=return_legs,
            )

            factors.append(factor_df)

        # Build momentum factor
        logger.info("\n🚀 Building ESG momentum factor...")
        esg_mom_sig = self._build_esg_momentum_signal(esg_lagged[["esg_score"]])

        if not esg_mom_sig.empty:
            esg_mom_factor = self._build_long_short_factor(
                panel_excess=panel_excess,
                signal_df=esg_mom_sig,
                weights_df=weights_df,
                sector_map=sector_map,
                quantile=quantile,
                lag_signal=lag_signal,
                return_legs=return_legs,
            )
            factors.append(esg_mom_factor)
        else:
            logger.warning("  No ESG momentum signal available")

        # Combine all factors
        logger.info("\n🔗 Combining factors...")
        result_df = pd.concat(factors, ignore_index=True)

        # Drop any NaN rows
        result_df = result_df.dropna()

        logger.info(
            f"✅ Built {result_df['factor_name'].nunique()} factors with {len(result_df)} observations"
        )
        logger.info(
            f"   Date range: {result_df['date'].min()} to {result_df['date'].max()}"
        )
        logger.info(f"   Factors: {result_df['factor_name'].unique().tolist()}")

        logger.info("=" * 80)
        logger.info("ESG Factor Model - Execution Completed")
        logger.info("=" * 80)

        return result_df

    @staticmethod
    def _resample_to_monthly(prices_df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily prices to monthly (end of month)"""
        # Ensure date is datetime
        if "date" not in prices_df.columns:
            raise ValueError("prices_df must have 'date' column")

        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        # Group by symbol and resample to month-end
        monthly = []
        for symbol, group in prices_df.groupby("symbol"):
            group_sorted = group.sort_values("date").set_index("date")
            # Resample to month-end, taking last observation
            monthly_group = group_sorted.resample("ME").last()
            monthly_group["symbol"] = symbol
            monthly.append(monthly_group.reset_index())

        result = pd.concat(monthly, ignore_index=True)
        result = result.dropna(subset=["close"])

        return result

    @staticmethod
    def _resample_rf_to_monthly(rf_df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily risk-free rates to monthly (average)"""
        rf_df = rf_df.copy()
        rf_df["date"] = pd.to_datetime(rf_df["date"])
        rf_df = rf_df.sort_values("date").set_index("date")

        # Resample to month-end, taking average rate for the month
        rf_monthly = rf_df["rate"].resample("ME").mean().reset_index()

        return rf_monthly

    @staticmethod
    def _compute_monthly_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
        """Compute monthly returns from prices"""
        returns = prices_df.copy()
        returns = returns.sort_values(["symbol", "date"])
        returns["ret"] = returns.groupby("symbol")["close"].pct_change()
        returns = returns.dropna(subset=["ret"])

        # Create MultiIndex for compatibility
        returns = returns.set_index(["date", "symbol"]).sort_index()

        return returns[["ret"]]

    @staticmethod
    def _compute_market_cap_weights(prices_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute market cap weights from price × volume"""
        if "volume" not in prices_df.columns:
            return None

        weights = prices_df.copy()

        # Market cap proxy = price × volume
        weights["mktcap"] = weights["close"] * weights["volume"]

        # Handle zeros/negatives
        weights["mktcap"] = weights["mktcap"].replace(0, np.nan)
        weights["mktcap"] = weights["mktcap"].clip(lower=0)

        # Normalize within each date to sum to 1
        weights["weight"] = weights.groupby("date")["mktcap"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x)
        )

        # Create MultiIndex
        weights = weights.set_index(["date", "symbol"]).sort_index()

        return weights[["weight"]]

    @staticmethod
    def _to_excess_returns(
        returns_df: pd.DataFrame, rf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert returns to excess returns (subtract risk-free rate)"""
        # Reset index for merge
        returns_with_date = returns_df.reset_index()

        # Normalize RF to monthly decimal (from annual percentage)
        rf_copy = rf_df.copy()
        rf_copy["RF"] = rf_copy["rate"] / 100 / 12

        # Merge
        panel = returns_with_date.merge(rf_copy[["date", "RF"]], on="date", how="left")
        panel["excess"] = panel["ret"] - panel["RF"]

        # Restore MultiIndex
        panel = panel.set_index(["date", "symbol"]).sort_index()

        return panel[["excess"]]

    @staticmethod
    def _prepare_esg_scores(esg_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare ESG scores for factor construction"""
        esg = esg_df.copy()

        # Ensure required columns
        required_cols = [
            "date",
            "ticker",
            "esg_score",
            "environmental_pillar_score",
            "social_pillar_score",
            "governance_pillar_score",
        ]
        missing = [col for col in required_cols if col not in esg.columns]
        if missing:
            raise ValueError(f"ESG data missing required columns: {missing}")

        # Rename ticker to symbol for consistency
        if "ticker" in esg.columns and "symbol" not in esg.columns:
            esg = esg.rename(columns={"ticker": "symbol"})

        # Ensure date is datetime
        esg["date"] = pd.to_datetime(esg["date"])

        # Create MultiIndex
        esg = esg.set_index(["date", "symbol"]).sort_index()

        return esg

    @staticmethod
    def _apply_annual_esg_lag(
        esg_df: pd.DataFrame, lag_months: int = 12
    ) -> pd.DataFrame:
        """Apply lag to ESG scores to avoid look-ahead bias"""
        esg_cols = [
            "esg_score",
            "environmental_pillar_score",
            "social_pillar_score",
            "governance_pillar_score",
        ]

        # Shift ESG scores forward by lag_months
        lagged = esg_df.groupby(level="symbol")[esg_cols].shift(lag_months)

        return lagged.dropna()

    @staticmethod
    def _build_esg_momentum_signal(esg_df: pd.DataFrame) -> pd.DataFrame:
        """Build ESG momentum signal (z-scored YoY changes)"""
        # Calculate year-over-year ESG changes (12-month lag)
        d_esg = esg_df.groupby(level="symbol")["esg_score"].diff(12).to_frame("dESG")

        # Z-score cross-section by month
        mom = []
        for dt, df in d_esg.groupby(level="date"):
            x = df.droplevel(0)
            mu = x["dESG"].mean()
            sd = x["dESG"].std(ddof=0)
            z = (x["dESG"] - mu) / (sd if sd and sd != 0 else 1)

            mom_df = pd.DataFrame({"ESG_mom": z}, index=x.index)
            mom_df["date"] = dt
            mom_df["symbol"] = mom_df.index
            mom.append(mom_df)

        if not mom:
            return pd.DataFrame()

        result = pd.concat(mom, ignore_index=True)
        result = result.set_index(["date", "symbol"]).sort_index()

        return result[["ESG_mom"]]

    @staticmethod
    def _get_sector_mapping(esg_df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract sector mapping from ESG data (if available)"""
        # Check if sic_code is available
        esg_reset = esg_df.reset_index()

        if "sic_code" not in esg_reset.columns:
            return None

        # Get most recent SIC code for each symbol
        sector_map = {}
        for symbol, group in esg_reset.groupby("symbol"):
            sic_codes = group["sic_code"].dropna()
            if not sic_codes.empty:
                sic_code = sic_codes.iloc[-1]
                sector_map[symbol] = ESGFactorModel._sic_to_sector(int(sic_code))

        if not sector_map:
            return None

        return pd.Series(sector_map)

    @staticmethod
    def _sic_to_sector(sic_code: int) -> str:
        """Map SIC code to broad sector"""
        if sic_code < 1000:
            return "Agriculture"
        elif sic_code < 1500:
            return "Mining"
        elif sic_code < 1800:
            return "Construction"
        elif sic_code < 4000:
            return "Manufacturing"
        elif sic_code < 5000:
            return "Transportation & Utilities"
        elif sic_code < 5200:
            return "Wholesale Trade"
        elif sic_code < 6000:
            return "Retail Trade"
        elif sic_code < 6800:
            return "Finance & Real Estate"
        elif sic_code < 9000:
            return "Services"
        else:
            return "Public Administration"

    def _build_long_short_factor(
        self,
        panel_excess: pd.DataFrame,
        signal_df: pd.DataFrame,
        weights_df: Optional[pd.DataFrame],
        sector_map: Optional[pd.Series],
        quantile: float,
        lag_signal: int,
        return_legs: bool,
    ) -> pd.DataFrame:
        """Build long-short factor from signal"""
        # Lag signals
        sig_lag = signal_df.groupby(level="symbol").shift(lag_signal)

        # Lag weights to match signal timing
        if weights_df is not None:
            weights_lag = weights_df.groupby(level="symbol").shift(lag_signal)
        else:
            weights_lag = None

        # Merge signal and excess returns
        panel = panel_excess[["excess"]].join(sig_lag, how="inner").dropna()

        factor_name = signal_df.columns[0]
        results = []

        for dt, df in panel.groupby(level="date"):
            x = df.droplevel(0)  # index=symbol
            score_col = signal_df.columns[0]

            # Rank stocks by signal
            x = self._rank_within(x, score_col=score_col, sector_map=sector_map)

            # Form long/short legs
            long = x[x["rank_pct"] >= (1 - quantile)]
            short = x[x["rank_pct"] <= quantile]

            # Get weights for this date
            w_long = None
            w_short = None
            if weights_lag is not None and dt in weights_lag.index.get_level_values(0):
                w_on_dt = (
                    weights_lag.loc[dt].reindex(long.index)["weight"]
                    if len(long)
                    else None
                )
                w_on_ds = (
                    weights_lag.loc[dt].reindex(short.index)["weight"]
                    if len(short)
                    else None
                )
                w_long = (
                    w_on_dt
                    if w_on_dt is not None and not w_on_dt.isna().all()
                    else None
                )
                w_short = (
                    w_on_ds
                    if w_on_ds is not None and not w_on_ds.isna().all()
                    else None
                )

            # Calculate leg returns
            r_long = (
                self._value_weighted_return(long, weights=w_long)
                if len(long)
                else np.nan
            )
            r_short = (
                self._value_weighted_return(short, weights=w_short)
                if len(short)
                else np.nan
            )

            if np.isnan(r_long) or np.isnan(r_short):
                continue

            if return_legs:
                results.append(
                    {
                        "date": dt,
                        "factor_name": factor_name,
                        "factor_return": r_long - r_short,
                        "long_return": r_long,
                        "short_return": r_short,
                    }
                )
            else:
                results.append(
                    {
                        "date": dt,
                        "factor_name": factor_name,
                        "factor_return": r_long - r_short,
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def _rank_within(
        df: pd.DataFrame, score_col: str, sector_map: Optional[pd.Series]
    ) -> pd.DataFrame:
        """Rank stocks by signal (cross-sectional or sector-neutral)"""
        x = df.copy()

        if sector_map is not None:
            x["sector"] = sector_map.reindex(x.index)

            n_with_sector = x["sector"].notna().sum()
            if n_with_sector == 0:
                x["rank_pct"] = x[score_col].rank(pct=True)
            else:
                ranks = []
                for sec, g in x.groupby("sector", dropna=False):
                    if pd.notna(sec):
                        ranks.append(g[score_col].rank(pct=True))
                x["rank_pct"] = pd.concat(ranks).sort_index()

            x.drop(columns=["sector"], inplace=True)
        else:
            x["rank_pct"] = x[score_col].rank(pct=True)

        return x

    @staticmethod
    def _value_weighted_return(
        sub_df: pd.DataFrame, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate value-weighted or equal-weighted return"""
        if weights is None or weights.sum() == 0 or weights.isnull().all():
            return sub_df["excess"].mean()

        w = weights.fillna(0)
        w = w / (w.sum() if w.sum() != 0 else 1)
        return float(np.dot(sub_df["excess"].to_numpy(), w.to_numpy()))
