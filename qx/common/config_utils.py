"""
Configuration Utilities for Qx

Helpers for loading configuration, resolving parameters, and initializing external clients.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from tiingo import TiingoClient


def resolve_parameters(spec: Dict, overrides: Optional[Dict] = None) -> Dict:
    """
    Resolve parameters from spec with type conversion and overrides.

    Shared utility used by builders, loaders, and models to parse YAML parameters
    and apply runtime overrides with appropriate type conversions.

    Args:
        spec: Parameter specification from YAML (e.g., builder.yaml "parameters" section)
        overrides: Runtime parameter overrides (e.g., from DAG or factory)

    Returns:
        Resolved parameters dictionary with type conversions applied

    Example:
        >>> spec = {"count": {"type": "int", "default": 10}, "name": {"type": "str", "default": "test"}}
        >>> resolve_parameters(spec, {"count": "20"})
        {'count': 20, 'name': 'test'}
    """
    overrides = overrides or {}
    params = {}

    for key, param_spec in spec.items():
        value = overrides.get(key, param_spec.get("default"))
        param_type = param_spec.get("type")

        # Skip type conversion if value is None
        if value is None:
            params[key] = value
            continue

        # Apply type conversions
        if param_type == "int":
            value = int(value)
        elif param_type == "float":
            value = float(value)
        elif param_type == "bool":
            value = (
                value
                if isinstance(value, bool)
                else str(value).lower() in ("1", "true", "yes")
            )
        elif param_type == "enum":
            allowed = param_spec.get("allowed", [])
            if value not in allowed:
                raise ValueError(
                    f"Parameter '{key}' must be one of {allowed}, got '{value}'"
                )
        # For "str" or unspecified type, keep value as-is

        params[key] = value

    return params


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
