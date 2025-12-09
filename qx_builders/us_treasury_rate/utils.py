"""
US Treasury Rate Utilities

Helper functions and constants for treasury rate data.
"""

from typing import Dict

# FRED Series IDs for US Treasury rates
FRED_SERIES: Dict[str, str] = {
    "3month": "DGS3MO",  # 3-Month Treasury Constant Maturity
    "1year": "DGS1",  # 1-Year Treasury Constant Maturity
    "5year": "DGS5",  # 5-Year Treasury Constant Maturity
    "10year": "DGS10",  # 10-Year Treasury Constant Maturity
    "30year": "DGS30",  # 30-Year Treasury Constant Maturity
}


def get_default_rate_types() -> list[str]:
    """
    Get list of standard US Treasury rate types.

    Returns:
        List of rate type identifiers

    Example:
        >>> get_default_rate_types()
        ['3month', '1year', '5year', '10year', '30year']
    """
    return list(FRED_SERIES.keys())


def get_fred_series_id(rate_type: str) -> str:
    """
    Get FRED series ID for a given rate type.

    Args:
        rate_type: Rate type ('3month', '1year', '5year', '10year', '30year')

    Returns:
        FRED series ID (e.g., 'DGS10' for 10-year)

    Raises:
        ValueError: If rate_type is not recognized

    Example:
        >>> get_fred_series_id('10year')
        'DGS10'
    """
    if rate_type not in FRED_SERIES:
        raise ValueError(
            f"Invalid rate_type: {rate_type}. "
            f"Must be one of {list(FRED_SERIES.keys())}"
        )
    return FRED_SERIES[rate_type]
