"""
Enum Validation Utilities

Provides strict validation and conversion between YAML strings and Python enums.
Helps catch typos and invalid values during builder/loader/model initialization.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from qx.common.types import AssetClass, Domain, Frequency, Region, Subdomain


class EnumValidationError(ValueError):
    """Raised when YAML contains invalid enum values."""

    pass


def validate_enum_value(
    enum_class: Type[Enum], value: Any, field_name: str, allow_none: bool = True
) -> Optional[Enum]:
    """
    Validate and convert a value to an enum member.

    Args:
        enum_class: The enum class to validate against (Domain, AssetClass, etc.)
        value: The value to validate (string or None)
        field_name: Name of the field being validated (for error messages)
        allow_none: Whether None is an acceptable value

    Returns:
        Enum member or None if value is None and allow_none=True

    Raises:
        EnumValidationError: If value is invalid

    Example:
        >>> domain = validate_enum_value(Domain, "market-data", "domain")
        >>> frequency = validate_enum_value(Frequency, "daily", "frequency")
    """
    # Handle None values
    if value is None:
        if allow_none:
            return None
        raise EnumValidationError(
            f"{field_name} cannot be None. Must be one of: {get_valid_values(enum_class)}"
        )

    # Handle "null" string (common in YAML)
    if isinstance(value, str) and value.lower() == "null":
        if allow_none:
            return None
        raise EnumValidationError(
            f"{field_name} cannot be 'null'. Must be one of: {get_valid_values(enum_class)}"
        )

    # Try to find matching enum member
    # First try by value (recommended: "daily", "market-data", etc.)
    for member in enum_class:
        if member.value == value:
            return member

    # Also try by name (allows "DAILY", "MARKET_DATA" for flexibility)
    for member in enum_class:
        if member.name == value:
            return member

    # Not found - raise detailed error
    valid_values = get_valid_values(enum_class)
    raise EnumValidationError(
        f"Invalid {field_name}: '{value}'\n"
        f"Valid values are: {valid_values}\n"
        f"Did you mean one of: {suggest_similar(value, valid_values)}"
    )


def get_valid_values(enum_class: Type[Enum]) -> List[str]:
    """
    Get list of all valid values for an enum class.

    Args:
        enum_class: The enum class

    Returns:
        List of valid string values

    Example:
        >>> get_valid_values(Frequency)
        ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
    """
    return [member.value for member in enum_class]


def suggest_similar(
    value: str, valid_values: List[str], max_suggestions: int = 3
) -> List[str]:
    """
    Suggest similar valid values based on string similarity.

    Args:
        value: The invalid value
        valid_values: List of valid values
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of suggested valid values

    Example:
        >>> suggest_similar("daiy", ["daily", "weekly", "monthly"])
        ['daily']
    """
    if not isinstance(value, str):
        return []

    value_lower = value.lower()

    # Check for exact prefix matches
    prefix_matches = [v for v in valid_values if v.startswith(value_lower)]
    if prefix_matches:
        return prefix_matches[:max_suggestions]

    # Check for substring matches
    substring_matches = [
        v for v in valid_values if value_lower in v or v in value_lower
    ]
    if substring_matches:
        return substring_matches[:max_suggestions]

    # Return first few valid values as fallback
    return valid_values[:max_suggestions]


def validate_dataset_type_config(config: Dict[str, Any]) -> Dict[str, Optional[Enum]]:
    """
    Validate all enum fields in a dataset type configuration.

    Args:
        config: Configuration dict from builder.yaml io.output.type section

    Returns:
        Dict with validated enum members

    Raises:
        EnumValidationError: If any field is invalid

    Example:
        >>> config = {
        ...     "domain": "market-data",
        ...     "asset_class": "equity",
        ...     "subdomain": "ohlcv",
        ...     "region": "US",
        ...     "frequency": null
        ... }
        >>> validated = validate_dataset_type_config(config)
        >>> validated["domain"]
        <Domain.MARKET_DATA: 'market-data'>
    """
    validated = {}

    # Validate domain (required)
    validated["domain"] = validate_enum_value(
        Domain, config.get("domain"), "domain", allow_none=False
    )

    # Validate asset_class (required for most domains)
    validated["asset_class"] = validate_enum_value(
        AssetClass, config.get("asset_class"), "asset_class", allow_none=True
    )

    # Validate subdomain (required)
    validated["subdomain"] = validate_enum_value(
        Subdomain, config.get("subdomain"), "subdomain", allow_none=False
    )

    # Validate region (optional, used in partitions)
    validated["region"] = validate_enum_value(
        Region, config.get("region"), "region", allow_none=True
    )

    # Validate frequency (optional, used in partitions)
    validated["frequency"] = validate_enum_value(
        Frequency, config.get("frequency"), "frequency", allow_none=True
    )

    return validated


def validate_frequency_parameter(
    value: Any, allow_none: bool = False
) -> Optional[Frequency]:
    """
    Validate a frequency parameter value.

    Convenience wrapper for frequency validation with better error messages.

    Args:
        value: The frequency value to validate
        allow_none: Whether None is acceptable

    Returns:
        Frequency enum member or None

    Raises:
        EnumValidationError: If value is invalid

    Example:
        >>> freq = validate_frequency_parameter("daily")
        >>> freq
        <Frequency.DAILY: 'daily'>
    """
    try:
        return validate_enum_value(Frequency, value, "frequency", allow_none=allow_none)
    except EnumValidationError as e:
        # Add more context for frequency errors
        raise EnumValidationError(
            f"{str(e)}\n"
            f"Note: Use lowercase frequency values like 'daily', 'weekly', 'monthly'"
        )


def validate_partition_values(
    partitions: Dict[str, Any], expected_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate partition values, converting enum strings to enum members.

    Args:
        partitions: Partition dict (e.g., {"exchange": "US", "frequency": "daily"})
        expected_keys: List of expected partition keys (optional)

    Returns:
        Dict with validated partition values

    Raises:
        EnumValidationError: If any value is invalid

    Example:
        >>> partitions = {"region": "US", "frequency": "daily"}
        >>> validated = validate_partition_values(partitions)
        >>> validated["frequency"]
        <Frequency.DAILY: 'daily'>
    """
    validated = {}

    # Validate each partition
    for key, value in partitions.items():
        if key in ("region", "exchange"):
            # Validate region/exchange values
            validated[key] = validate_enum_value(Region, value, key, allow_none=False)
        elif key == "frequency":
            # Validate frequency
            validated[key] = validate_enum_value(
                Frequency, value, key, allow_none=False
            )
        else:
            # Keep other partition values as-is (symbol, date, etc.)
            validated[key] = value

    # Check for missing expected keys
    if expected_keys:
        missing = set(expected_keys) - set(validated.keys())
        if missing:
            raise EnumValidationError(
                f"Missing required partition keys: {missing}\n"
                f"Provided: {list(partitions.keys())}\n"
                f"Expected: {expected_keys}"
            )

    return validated


def print_valid_enum_values():
    """
    Print all valid enum values for reference.

    Useful for debugging and documentation.
    """
    print("=" * 80)
    print("Valid Enum Values for Qx Framework")
    print("=" * 80)

    print("\n📊 Domain:")
    for member in Domain:
        print(f"  - {member.value:20} ({member.name})")

    print("\n💼 AssetClass:")
    for member in AssetClass:
        print(f"  - {member.value:20} ({member.name})")

    print("\n🔖 Subdomain (selected):")
    # Show first 15 subdomains (there are many)
    for i, member in enumerate(Subdomain):
        if i < 15:
            print(f"  - {member.value:20} ({member.name})")
        else:
            print(f"  ... and {len(list(Subdomain)) - 15} more")
            break

    print("\n🌍 Region:")
    for member in Region:
        print(f"  - {member.value:20} ({member.name})")

    print("\n📅 Frequency:")
    for member in Frequency:
        print(f"  - {member.value:20} ({member.name})")

    print("\n" + "=" * 80)


# Convenience exports
__all__ = [
    "EnumValidationError",
    "validate_enum_value",
    "validate_dataset_type_config",
    "validate_frequency_parameter",
    "validate_partition_values",
    "get_valid_values",
    "print_valid_enum_values",
]
