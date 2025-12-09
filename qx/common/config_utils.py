"""
Configuration Utilities for Qx

Helpers for loading configuration and initializing external clients.
"""

from pathlib import Path
from typing import Any, Dict

import yaml
from tiingo import TiingoClient


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to settings.yaml

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_nested_config(config: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "fetcher.tiingo.api_key")
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> config = {"fetcher": {"tiingo": {"api_key": "abc123"}}}
        >>> get_nested_config(config, "fetcher.tiingo.api_key")
        'abc123'
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def create_tiingo_client(config_path: str = "config/settings.yaml") -> TiingoClient:
    """
    Create TiingoClient from configuration file.

    Args:
        config_path: Path to settings.yaml

    Returns:
        Initialized TiingoClient

    Raises:
        ValueError: If API key not found in config

    Example:
        >>> tiingo = create_tiingo_client()
        >>> df = tiingo.get_dataframe('AAPL')
    """
    config = load_config(config_path)
    api_key = get_nested_config(config, "fetcher.tiingo.api_key")

    if not api_key:
        raise ValueError(
            "Tiingo API key not found in config. "
            "Please set 'fetcher.tiingo.api_key' in config/settings.yaml"
        )

    return TiingoClient({"api_key": api_key, "session": True})


def get_fred_api_key(config_path: str = "config/settings.yaml") -> str:
    """
    Get FRED API key from configuration file.

    Args:
        config_path: Path to settings.yaml

    Returns:
        FRED API key string

    Raises:
        ValueError: If API key not found in config

    Example:
        >>> fred_key = get_fred_api_key()
        >>> builder = USTreasuryRateBuilder(..., fred_api_key=fred_key)

    Note:
        Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """
    config = load_config(config_path)
    api_key = get_nested_config(config, "fetcher.fred.api_key")

    if not api_key:
        raise ValueError(
            "FRED API key not found in config. "
            "Please set 'fetcher.fred.api_key' in config/settings.yaml\n"
            "Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    return api_key
