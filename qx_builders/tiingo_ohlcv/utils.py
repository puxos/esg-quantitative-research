"""
Tiingo Price Data Utilities

Helper functions for date alignment and frequency handling.
"""

import pandas as pd


def align_start_date_to_frequency(
    start_date: pd.Timestamp, frequency: str
) -> pd.Timestamp:
    """
    Align start date to frequency period.

    For monthly data: 2014-01-01 → 2014-01-31 (end of month)
    For weekly data: Align to Friday (end of week)
    For daily data: No adjustment

    Args:
        start_date: Requested start date
        frequency: Data frequency ('daily', 'weekly', 'monthly')

    Returns:
        Aligned start date

    Example:
        >>> align_start_date_to_frequency(pd.Timestamp('2014-01-01'), 'monthly')
        Timestamp('2014-01-31 00:00:00')

        >>> align_start_date_to_frequency(pd.Timestamp('2014-01-06'), 'weekly')
        Timestamp('2014-01-10 00:00:00')  # Next Friday
    """
    frequency = frequency.lower()

    if frequency == "monthly":
        # Align to end of month
        if start_date.month == 12:
            next_month = start_date.replace(year=start_date.year + 1, month=1, day=1)
        else:
            next_month = start_date.replace(month=start_date.month + 1, day=1)
        return next_month - pd.Timedelta(days=1)

    elif frequency == "weekly":
        # Align to end of week (Friday)
        days_until_friday = (4 - start_date.weekday()) % 7
        if days_until_friday == 0 and start_date.weekday() != 4:
            days_until_friday = 7
        return start_date + pd.Timedelta(days=days_until_friday)

    else:  # daily
        return start_date


def get_tolerance_for_frequency(frequency: str) -> int:
    """
    Get appropriate tolerance in days based on frequency.

    Tolerance defines how many days of missing data are acceptable
    before considering data stale or incomplete.

    Args:
        frequency: Data frequency ('daily', 'weekly', 'monthly')

    Returns:
        Tolerance in days

    Tolerances:
        - Daily: 2 days (weekends)
        - Weekly: 6 days (almost a full week)
        - Monthly: 3 days (end-of-month buffer)

    Example:
        >>> get_tolerance_for_frequency('daily')
        2
        >>> get_tolerance_for_frequency('monthly')
        3
    """
    TOLERANCES = {
        "daily": 2,
        "weekly": 6,
        "monthly": 3,
    }
    return TOLERANCES.get(frequency.lower(), 2)
