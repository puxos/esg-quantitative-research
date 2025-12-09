"""
Continuous Membership Model

Identifies tickers with continuous membership in a universe over a specified period.

This model addresses survivorship bias by analyzing historical membership data
and filtering for tickers that were continuously present (no gaps > threshold).

Adapted from legacy logic:
- src/programs/check_esg_continuity.py
- src/universe/universe.py::get_all_historical_members()
"""

import pandas as pd

from qx.engine.base_model import BaseModel


class ContinuousMembershipModel(BaseModel):
    """
    Model to identify tickers with continuous membership.

    Input: Daily membership data (curated/membership)
    Output: Continuous members list with metadata (processed/continuous_members)
    """

    def run_impl(self, inputs: dict, params: dict, **kwargs) -> pd.DataFrame:
        """
        Run continuous membership analysis.

        Args:
            inputs: Dict with 'membership_daily' DataFrame [date, ticker]
            params: Dict with start_date, end_date, min_continuity_pct
            **kwargs: Additional parameters

        Returns:
            DataFrame with continuous members and metadata:
                - ticker: Ticker symbol
                - start_date: Analysis start date
                - end_date: Analysis end date
                - total_snapshots: Total unique snapshot dates in period
                - snapshots_present: Number of snapshots where ticker appears
                - continuity_pct: Percentage of snapshots where ticker is present
                - snapshots_missing: Number of missing snapshots
                - first_date: First observed date
                - last_date: Last observed date
                - is_continuous: Boolean flag (True if >= min_continuity_pct)
        """
        # Extract inputs
        membership_df = inputs["membership_daily"]

        # Extract parameters
        start_date = params["start_date"]
        end_date = params["end_date"]
        min_continuity_pct = params["min_continuity_pct"]

        print(f"Running {self.info['id']} v{self.info['version']}")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Min continuity: {min_continuity_pct*100:.1f}%")
        print()

        # Validate inputs
        if "date" not in membership_df.columns or "ticker" not in membership_df.columns:
            raise ValueError("membership_df must have 'date' and 'ticker' columns")

        # Convert dates
        membership_df = membership_df.copy()
        membership_df["date"] = pd.to_datetime(membership_df["date"])
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Filter to analysis period
        period_df = membership_df[
            (membership_df["date"] >= start) & (membership_df["date"] <= end)
        ]

        if len(period_df) == 0:
            print(f"⚠️  No membership data in period {start_date} to {end_date}")
            return pd.DataFrame()

        print(f"  Total membership records: {len(period_df):,}")
        print(f"  Unique tickers: {period_df['ticker'].nunique()}")
        print()

        # Calculate expected snapshot dates (unique dates in data, not all business days)
        # This is critical: membership data may be weekly or sparse snapshots
        unique_snapshot_dates = period_df["date"].drop_duplicates().sort_values()
        expected_days = len(unique_snapshot_dates)
        min_days_required = int(expected_days * min_continuity_pct)

        print(f"  Unique snapshot dates: {expected_days}")
        print(
            f"  Min snapshots required: {min_days_required} ({min_continuity_pct*100:.1f}%)"
        )
        print()

        # Group by ticker and calculate continuity metrics
        # For each ticker, count unique dates they appear in
        ticker_stats = []

        for ticker, group in period_df.groupby("ticker"):
            # Count unique snapshot dates this ticker appears in
            unique_dates_present = group["date"].nunique()
            days_present = unique_dates_present  # Renamed for clarity
            continuity_pct = days_present / expected_days
            gap_count = expected_days - days_present
            is_continuous = days_present >= min_days_required

            ticker_stats.append(
                {
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_snapshots": expected_days,
                    "snapshots_present": days_present,
                    "continuity_pct": continuity_pct,
                    "snapshots_missing": gap_count,
                    "first_date": group["date"].min().strftime("%Y-%m-%d"),
                    "last_date": group["date"].max().strftime("%Y-%m-%d"),
                    "is_continuous": is_continuous,
                }
            )

        result_df = pd.DataFrame(ticker_stats).sort_values(
            "continuity_pct", ascending=False
        )

        # Summary statistics
        continuous_count = result_df["is_continuous"].sum()
        total_count = len(result_df)

        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Total tickers analyzed: {total_count}")
        print(
            f"Continuous tickers: {continuous_count} ({continuous_count/total_count*100:.1f}%)"
        )
        print(
            f"Discontinuous tickers: {total_count - continuous_count} ({(total_count - continuous_count)/total_count*100:.1f}%)"
        )
        print()

        # Show continuity distribution
        if len(result_df) > 0:
            print("Continuity distribution:")
            print(f"  Min: {result_df['continuity_pct'].min()*100:.2f}%")
            print(f"  Median: {result_df['continuity_pct'].median()*100:.2f}%")
            print(f"  Mean: {result_df['continuity_pct'].mean()*100:.2f}%")
            print(f"  Max: {result_df['continuity_pct'].max()*100:.2f}%")
            print()

        # Show sample of continuous tickers
        continuous_df = result_df[result_df["is_continuous"]]
        if len(continuous_df) > 0:
            print(f"Sample continuous tickers (top 10 by continuity):")
            print(
                continuous_df[
                    [
                        "ticker",
                        "snapshots_present",
                        "continuity_pct",
                        "snapshots_missing",
                    ]
                ]
                .head(10)
                .to_string(index=False)
            )
            print()

        # Show sample of discontinuous tickers
        discontinuous_df = result_df[~result_df["is_continuous"]]
        if len(discontinuous_df) > 0:
            print(f"Sample discontinuous tickers (worst 5 by continuity):")
            print(
                discontinuous_df[
                    [
                        "ticker",
                        "snapshots_present",
                        "continuity_pct",
                        "snapshots_missing",
                    ]
                ]
                .tail(5)
                .to_string(index=False)
            )
            print()

        print("=" * 80)

        return result_df
