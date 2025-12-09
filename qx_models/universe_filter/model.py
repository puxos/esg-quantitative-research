"""
Universe Filter Model Implementation

Filters universe to tickers with continuous ESG score coverage.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from qx.engine.base_model import BaseModel

logger = logging.getLogger(__name__)


class UniverseFilterModel(BaseModel):
    """
    Universe Filter Model: Filter to continuous ESG coverage.

    This model analyzes ESG score availability for a universe of tickers
    and filters to those meeting data quality requirements.

    Key Features:
        - Configurable coverage thresholds (percentage + gap limits)
        - Multiple membership modes (any_time, continuous, end_of_period)
        - Auto-detect ESG publication frequency
        - Detailed coverage reporting for audit trail
        - Reproducible filtering with run_id tracking

    Use Case:
        Before building ESG factors, filter the universe to ensure
        all tickers have sufficient ESG score history to avoid
        lookahead bias and missing data issues.
    """

    def run_impl(
        self,
        inputs: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Filter universe to tickers with continuous ESG coverage.

        Args:
            inputs: Dictionary of input DataFrames
                - esg_scores: ESG scores (ticker, date, esg_score)
                - membership_data: Universe membership (date, ticker, universe)
            params: Model parameters from model.yaml
            **kwargs: Additional arguments

        Returns:
            DataFrame with coverage metrics and filter decisions
        """
        logger.info("Starting Universe Filter Model")

        # Extract inputs
        esg_scores = inputs["esg_scores"]
        membership_data = inputs["membership_data"]

        # Extract parameters
        start_date = pd.Timestamp(params["start_date"])
        end_date = pd.Timestamp(params["end_date"])
        universe = params.get("universe", "sp500")
        min_coverage_pct = params.get("min_coverage_pct", 0.90)
        max_gap_days = params.get("max_gap_days", 90)
        require_continuous = params.get("require_continuous", True)
        esg_frequency = params.get("esg_frequency", "monthly")
        membership_mode = params.get("membership_mode", "any_time")
        include_failed = params.get("include_failed", True)
        verbose_logging = params.get("verbose_logging", True)

        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Universe: {universe}")
        logger.info(f"Min coverage: {min_coverage_pct:.1%}")
        logger.info(f"Max gap: {max_gap_days} days")
        logger.info(f"Require continuous: {require_continuous}")

        # 1. Get universe tickers
        logger.info("Step 1: Getting universe tickers")
        universe_tickers = self._get_universe_tickers(
            membership_data, universe, start_date, end_date, membership_mode
        )
        logger.info(f"  Found {len(universe_tickers)} tickers in universe")

        # 2. Calculate expected ESG points
        trading_days = self._calculate_trading_days(start_date, end_date)
        expected_esg_points = self._calculate_expected_esg_points(
            start_date, end_date, esg_frequency, esg_scores
        )
        logger.info(f"  Trading days: {trading_days}")
        logger.info(f"  Expected ESG points: {expected_esg_points}")

        # 3. Analyze ESG coverage for each ticker
        logger.info("Step 2: Analyzing ESG coverage")
        coverage_results = []

        for ticker in universe_tickers:
            coverage = self._analyze_ticker_coverage(
                ticker=ticker,
                esg_scores=esg_scores,
                start_date=start_date,
                end_date=end_date,
                expected_esg_points=expected_esg_points,
                min_coverage_pct=min_coverage_pct,
                max_gap_days=max_gap_days,
                require_continuous=require_continuous,
            )

            coverage["universe"] = universe
            coverage["trading_days"] = trading_days
            coverage["min_coverage_pct"] = min_coverage_pct
            coverage["max_gap_days"] = max_gap_days
            coverage["require_continuous"] = require_continuous

            coverage_results.append(coverage)

            if verbose_logging and not coverage["passed_filter"]:
                logger.info(
                    f"  {ticker}: FAILED - {coverage['filter_reason']} "
                    f"(coverage: {coverage['esg_coverage_pct']:.1%}, "
                    f"max gap: {coverage['max_esg_gap_days']} days)"
                )

        result_df = pd.DataFrame(coverage_results)

        # 4. Summary statistics
        passed_count = result_df["passed_filter"].sum()
        failed_count = len(result_df) - passed_count
        pass_rate = passed_count / len(result_df) if len(result_df) > 0 else 0

        logger.info(f"Filtering complete:")
        logger.info(f"  Total tickers: {len(result_df)}")
        logger.info(f"  Passed filter: {passed_count} ({pass_rate:.1%})")
        logger.info(f"  Failed filter: {failed_count}")

        if failed_count > 0:
            # Show failure reasons
            failure_reasons = (
                result_df[~result_df["passed_filter"]]
                .groupby("filter_reason")
                .size()
                .sort_values(ascending=False)
            )
            logger.info(f"  Failure reasons:")
            for reason, count in failure_reasons.items():
                logger.info(f"    - {reason}: {count}")

        # 5. Filter output if requested
        if not include_failed:
            logger.info(f"Filtering to passed tickers only (include_failed=False)")
            result_df = result_df[result_df["passed_filter"]].copy()

        return result_df

    def _get_universe_tickers(
        self,
        membership_data: pd.DataFrame,
        universe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        membership_mode: str,
    ) -> List[str]:
        """
        Get tickers for universe based on membership mode.

        Args:
            membership_data: Membership DataFrame
            universe: Universe name
            start_date: Period start
            end_date: Period end
            membership_mode: 'any_time', 'continuous', 'end_of_period'

        Returns:
            List of ticker symbols
        """
        df = membership_data.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Filter to universe and period
        df = df[
            (df["universe"] == universe)
            & (df["date"] >= start_date)
            & (df["date"] <= end_date)
        ]

        if membership_mode == "any_time":
            # All tickers present at ANY point during period
            tickers = df["ticker"].unique().tolist()

        elif membership_mode == "continuous":
            # Only tickers present for ALL dates in period
            dates_in_period = df["date"].nunique()
            ticker_counts = df.groupby("ticker")["date"].nunique()
            continuous_tickers = ticker_counts[ticker_counts == dates_in_period].index
            tickers = continuous_tickers.tolist()

        elif membership_mode == "end_of_period":
            # Only tickers present at end of period
            end_date_data = df[df["date"] == df["date"].max()]
            tickers = end_date_data["ticker"].unique().tolist()

        else:
            raise ValueError(f"Unknown membership_mode: {membership_mode}")

        return sorted(tickers)

    def _calculate_trading_days(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> int:
        """
        Calculate approximate trading days in period.

        Args:
            start_date: Period start
            end_date: Period end

        Returns:
            Approximate number of trading days
        """
        total_days = (end_date - start_date).days
        # Approximate: 252 trading days per 365 calendar days
        trading_days = int(total_days * 252 / 365)
        return max(1, trading_days)  # At least 1 day

    def _calculate_expected_esg_points(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        esg_frequency: str,
        esg_scores: pd.DataFrame,
    ) -> int:
        """
        Calculate expected number of ESG observations.

        Args:
            start_date: Period start
            end_date: Period end
            esg_frequency: 'monthly', 'quarterly', 'annual', 'auto'
            esg_scores: ESG scores DataFrame (for auto-detection)

        Returns:
            Expected number of ESG points
        """
        years = (end_date - start_date).days / 365

        if esg_frequency == "monthly":
            expected = int(years * 12)
        elif esg_frequency == "quarterly":
            expected = int(years * 4)
        elif esg_frequency == "annual":
            expected = int(years * 1)
        elif esg_frequency == "auto":
            # Auto-detect from median gap in data
            df = esg_scores.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            gaps = df["date"].diff().dt.days.dropna()
            if len(gaps) > 0:
                median_gap = gaps.median()
                if median_gap <= 40:  # ~Monthly
                    expected = int(years * 12)
                elif median_gap <= 120:  # ~Quarterly
                    expected = int(years * 4)
                else:  # ~Annual
                    expected = int(years * 1)
            else:
                expected = int(years * 12)  # Default to monthly
        else:
            raise ValueError(f"Unknown esg_frequency: {esg_frequency}")

        return max(1, expected)  # At least 1 point

    def _analyze_ticker_coverage(
        self,
        ticker: str,
        esg_scores: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        expected_esg_points: int,
        min_coverage_pct: float,
        max_gap_days: int,
        require_continuous: bool,
    ) -> Dict[str, Any]:
        """
        Analyze ESG coverage for a single ticker.

        Args:
            ticker: Ticker symbol
            esg_scores: ESG scores DataFrame
            start_date: Period start
            end_date: Period end
            expected_esg_points: Expected number of ESG observations
            min_coverage_pct: Minimum coverage threshold
            max_gap_days: Maximum allowed gap
            require_continuous: Whether to enforce gap limit

        Returns:
            Dictionary with coverage metrics and filter decision
        """
        # Filter to ticker and period
        ticker_esg = esg_scores[esg_scores["ticker"] == ticker].copy()
        ticker_esg["date"] = pd.to_datetime(ticker_esg["date"])
        ticker_esg = ticker_esg[
            (ticker_esg["date"] >= start_date) & (ticker_esg["date"] <= end_date)
        ]
        ticker_esg = ticker_esg.sort_values("date")

        # Calculate metrics
        esg_count = len(ticker_esg)
        esg_coverage_pct = (
            esg_count / expected_esg_points if expected_esg_points > 0 else 0
        )

        # Calculate gaps
        if len(ticker_esg) > 1:
            gaps = ticker_esg["date"].diff().dt.days.dropna()
            max_esg_gap_days = int(gaps.max())
        else:
            max_esg_gap_days = 999 if len(ticker_esg) == 0 else 0

        # First and last ESG dates
        first_esg_date = ticker_esg["date"].min() if len(ticker_esg) > 0 else None
        last_esg_date = ticker_esg["date"].max() if len(ticker_esg) > 0 else None

        # Determine pass/fail
        passed_filter = True
        filter_reason = "PASSED"

        if esg_count == 0:
            passed_filter = False
            filter_reason = "No ESG data in period"

        elif esg_coverage_pct < min_coverage_pct:
            passed_filter = False
            filter_reason = (
                f"Coverage {esg_coverage_pct:.1%} < {min_coverage_pct:.1%} threshold"
            )

        elif require_continuous and max_esg_gap_days > max_gap_days:
            passed_filter = False
            filter_reason = (
                f"Gap {max_esg_gap_days} days > {max_gap_days} day threshold"
            )

        return {
            "ticker": ticker,
            "start_date": start_date.date(),
            "end_date": end_date.date(),
            "expected_esg_points": expected_esg_points,
            "esg_count": esg_count,
            "esg_coverage_pct": esg_coverage_pct,
            "max_esg_gap_days": max_esg_gap_days,
            "first_esg_date": first_esg_date.date() if first_esg_date else None,
            "last_esg_date": last_esg_date.date() if last_esg_date else None,
            "passed_filter": passed_filter,
            "filter_reason": filter_reason,
        }
