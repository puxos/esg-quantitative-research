"""
S&P 500 Membership Utilities

Helper functions for membership data transformations.
"""

import pandas as pd


def synthesize_membership_intervals(membership_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Transform daily membership to interval representation.

    Detects runs of consecutive presence across recorded dates to build intervals.

    Algorithm:
    1. For each ticker, mark which dates it appears on
    2. Find runs where ticker is present continuously
    3. Record (ticker, start_date, end_date) for each run

    Args:
        membership_daily: DataFrame with [date, ticker]

    Returns:
        DataFrame with [ticker, start_date, end_date]

    Example:
        Input:
            date        ticker
            2020-01-01  AAPL
            2020-01-02  AAPL
            2020-01-03  AAPL
            2020-01-05  AAPL  # Gap on 2020-01-04

        Output:
            ticker  start_date  end_date
            AAPL    2020-01-01  2020-01-03
            AAPL    2020-01-05  2020-01-05
    """
    # Get sorted unique dates across entire dataset
    all_dates = pd.Index(membership_daily["date"].sort_values().unique())

    interval_rows = []
    for ticker, group in membership_daily.groupby("ticker"):
        # Dates where this ticker is present
        present_dates = pd.Index(group["date"].unique())

        # Create binary series across all dates
        present = pd.Series(0, index=all_dates, dtype=int)
        present.loc[present_dates] = 1

        # Find run boundaries
        # Start: present=1 and previous=0
        starts = present.index[(present == 1) & (present.shift(1, fill_value=0) == 0)]
        # End: present=1 and next=0
        ends = present.index[(present == 1) & (present.shift(-1, fill_value=0) == 0)]

        for start, end in zip(starts, ends):
            interval_rows.append((ticker, start.date(), end.date()))

    membership_intervals = (
        pd.DataFrame(interval_rows, columns=["ticker", "start_date", "end_date"])
        .sort_values(["ticker", "start_date"])
        .reset_index(drop=True)
    )

    print(f"✅ Synthesized: {len(membership_intervals):,} membership intervals")

    return membership_intervals
