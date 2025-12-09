"""
Date Utilities for Data Builders
==================================

Utilities for handling date alignment, tolerance buffers, and trading day
adjustments in data builders.

These utilities help avoid false "missing data" warnings when actual trading
dates don't align exactly with requested dates due to:
- Weekends and holidays (daily data)
- Week start/end variations (weekly data)
- Month-end vs last trading day (monthly data)
"""

from datetime import date, timedelta
from typing import Optional

import pandas as pd

# Tolerance constants (in days)
DAILY_TOLERANCE_DAYS = 2
WEEKLY_TOLERANCE_DAYS = 6
MONTHLY_TOLERANCE_DAYS = 3


def align_start_date_to_frequency(start_date: date, frequency: str) -> date:
    """
    Align start date to frequency period to avoid false "missing data" warnings.

    For monthly data: A request starting 2014-01-01 should expect data from 2014-01-31 (end of month)
    For weekly data: A request starting Monday should expect data from Friday (end of week)
    For daily data: No adjustment needed

    Args:
        start_date: Requested start date
        frequency: Data frequency ('daily', 'weekly', 'monthly', 'D', 'W', 'M')

    Returns:
        Aligned start date appropriate for the frequency

    Examples:
        >>> align_start_date_to_frequency(date(2014, 1, 1), 'monthly')
        date(2014, 1, 31)  # End of January
        >>> align_start_date_to_frequency(date(2014, 1, 1), 'daily')
        date(2014, 1, 1)   # No change
        >>> align_start_date_to_frequency(date(2014, 1, 6), 'weekly')
        date(2014, 1, 10)  # Next Friday
    """
    frequency = frequency.lower()

    if frequency in ("monthly", "m"):
        # Align to end of month
        # Move to first day of next month, then back one day
        if start_date.month == 12:
            next_month = start_date.replace(year=start_date.year + 1, month=1, day=1)
        else:
            next_month = start_date.replace(month=start_date.month + 1, day=1)
        return next_month - timedelta(days=1)

    elif frequency in ("weekly", "w"):
        # Align to end of week (Friday)
        # If start date is not Friday, move to next Friday
        days_until_friday = (4 - start_date.weekday()) % 7  # 4 = Friday
        if days_until_friday == 0 and start_date.weekday() != 4:
            # Already Friday, no change
            return start_date
        return start_date + timedelta(
            days=days_until_friday if days_until_friday > 0 else 7
        )

    else:  # daily or other
        return start_date


def get_tolerance_for_frequency(frequency: str) -> int:
    """
    Get appropriate tolerance in days based on data frequency.

    For daily data: Allow ±2 days tolerance (weekends, holidays)
    For weekly data: Allow ±6 days tolerance (data might be on any weekday)
    For monthly data: Allow ±3 days tolerance (month-end vs first/last trading day)

    Args:
        frequency: Data frequency ('daily', 'weekly', 'monthly', 'D', 'W', 'M')

    Returns:
        Tolerance in days

    Examples:
        >>> get_tolerance_for_frequency('daily')
        2
        >>> get_tolerance_for_frequency('weekly')
        6
        >>> get_tolerance_for_frequency('monthly')
        3
        >>> get_tolerance_for_frequency('D')
        2
    """
    frequency = frequency.lower()

    if frequency in ("daily", "d"):
        return DAILY_TOLERANCE_DAYS
    elif frequency in ("weekly", "w"):
        return WEEKLY_TOLERANCE_DAYS
    elif frequency in ("monthly", "m"):
        return MONTHLY_TOLERANCE_DAYS
    else:
        # Unknown frequency, use daily default
        return DAILY_TOLERANCE_DAYS


def adjust_fetch_dates(
    start_date: str, end_date: str, frequency: str, apply_correction: bool = True
) -> tuple[str, str]:
    """
    Adjust fetch dates to account for trading day alignment.

    When fetching data, we want to ensure we get the first/last available
    trading days even if they don't exactly match the requested dates.

    Args:
        start_date: Requested start date (YYYY-MM-DD)
        end_date: Requested end date (YYYY-MM-DD)
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        apply_correction: Whether to apply date correction

    Returns:
        Tuple of (adjusted_start, adjusted_end) as strings

    Examples:
        >>> adjust_fetch_dates('2014-01-01', '2024-12-31', 'daily')
        ('2013-12-30', '2025-01-02')  # Buffer for weekends/holidays

        >>> adjust_fetch_dates('2014-01-01', '2024-12-31', 'monthly')
        ('2013-12-28', '2025-01-03')  # Buffer for month-end alignment
    """
    if not apply_correction:
        return start_date, end_date

    # Get tolerance for frequency
    tolerance = get_tolerance_for_frequency(frequency)

    # Convert to date objects
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    # Add buffer before start and after end
    adjusted_start = start - timedelta(days=tolerance)
    adjusted_end = end + timedelta(days=tolerance)

    # Convert back to strings
    return adjusted_start.strftime("%Y-%m-%d"), adjusted_end.strftime("%Y-%m-%d")


def check_date_coverage(
    actual_start: date,
    actual_end: date,
    requested_start: date,
    requested_end: date,
    frequency: str,
    apply_tolerance: bool = True,
) -> dict:
    """
    Check if actual date coverage meets requested range within tolerance.

    Args:
        actual_start: Actual start date in data
        actual_end: Actual end date in data
        requested_start: Requested start date
        requested_end: Requested end date
        frequency: Data frequency
        apply_tolerance: Whether to apply tolerance buffer

    Returns:
        Dictionary with coverage information:
        - is_complete: bool
        - start_gap_days: int
        - end_gap_days: int
        - tolerance_days: int
        - aligned_start: date (frequency-aligned requested start)
    """
    # Align requested start to frequency
    aligned_start = align_start_date_to_frequency(requested_start, frequency)

    # Get tolerance
    tolerance_days = get_tolerance_for_frequency(frequency) if apply_tolerance else 0

    # Calculate gaps
    start_gap_days = (
        (actual_start - aligned_start).days if actual_start > aligned_start else 0
    )
    end_gap_days = (
        (requested_end - actual_end).days if actual_end < requested_end else 0
    )

    # Check completeness
    is_complete = (start_gap_days <= tolerance_days) and (
        end_gap_days <= tolerance_days
    )

    return {
        "is_complete": is_complete,
        "start_gap_days": start_gap_days,
        "end_gap_days": end_gap_days,
        "tolerance_days": tolerance_days,
        "aligned_start": aligned_start,
        "actual_start": actual_start,
        "actual_end": actual_end,
    }
