from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class Domain(Enum):
    MARKET_DATA = "market-data"
    FUNDAMENTALS = "fundamentals"
    REFERENCE_RATES = "reference-rates"
    CORPORATE_ACTIONS = "corporate-actions"
    ESG = "esg"
    INSTRUMENT_REFERENCE = "instrument-reference"
    DERIVED_METRICS = "derived-metrics"
    PORTFOLIO = "portfolio"
    DATA_PRODUCTS = "data-products"
    METADATA = "metadata"


class AssetClass(Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed-income"
    FX = "fx"
    COMMODITY = "commodity"
    DERIVATIVE = "derivative"
    CRYPTO = "crypto"
    MULTI_ASSET = "multi-asset"


class Subdomain(Enum):
    # Market Data
    QUOTES = "quotes"
    TRADES = "trades"
    BARS = "bars"
    ORDER_BOOK = "order-book"
    AUCTION_IMBALANCE = "auction-imbalance"
    VOL_SURFACE = "vol-surface"

    # Reference Rates
    YIELD_CURVES = "yield-curves"
    BENCHMARK_RATES = "benchmark-rates"
    INDEX_LEVELS = "index-levels"

    # Corporate Actions
    DIVIDENDS = "dividends"
    SPLITS = "splits"
    MERGERS = "mergers"
    SYMBOL_CHANGES = "symbol-changes"

    # Fundamentals
    FIN_STATEMENTS = "financial-statements"
    RATIOS = "ratios"

    # ESG
    ESG_SCORES = "esg-scores"
    DISCLOSURES = "disclosures"

    # Instrument Reference
    IDENTIFIERS = "identifiers"
    EXCHANGES = "exchanges"
    CALENDARS = "calendars"
    INDEX_CONSTITUENTS = "index-constituents"

    # Derived Metrics
    FACTORS = "factors"
    RISK_MODELS = "risk-models"
    MODELS = "models"
    EVENT_STUDIES = "event-studies"
    EXPECTED_RETURNS = "expected-returns"
    FACTOR_RETURNS = "factor-returns"
    FACTOR_EXPOSURES = "factor-exposures"

    # Portfolio
    POSITIONS = "positions"
    TRANSACTIONS = "transactions"
    PNL = "pnl"
    PORTFOLIO_WEIGHTS = "portfolio-weights"

    # Data Products
    RESEARCH_EXPORTS = "research-exports"
    DASHBOARD_FEEDS = "dashboard-feeds"


class Region(Enum):
    """Geographic/market regions (used in DatasetType for contract identity)"""

    US = "US"
    HK = "HK"
    GLOBAL = "GLOBAL"


class Exchange(Enum):
    """Specific trading venues (used in partition keys for storage organization)"""

    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    HKEX = "HKEX"
    US = "US"  # Aggregate US exchanges
    HK = "HK"  # Aggregate HK exchanges


class Frequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass(frozen=True)
class DatasetType:
    """
    Identity of a dataset for contract lookup.

    Fields:
    - domain: Data category (market-data, reference-rates, esg, derived-metrics, etc.)
    - asset_class: Asset type (equity, fx, fixed-income, commodity, etc.)
    - subdomain: Specific data subdomain (bars, benchmark-rates, esg-scores, factors, etc.)
    - subtype: Custom subtype string (optional, not enum-restricted, for fine-grained classification)
    - region: Geographic/market region (US, HK, GLOBAL) - used for contract identity
    - frequency: Temporal frequency (daily, weekly, monthly, etc.)
    - dims: Additional dimensions for typed datasets

    Note: 'exchange' at partition level (NYSE, NASDAQ) is separate from 'region' here.
    """

    domain: Domain
    asset_class: Optional[AssetClass]
    subdomain: Optional[Subdomain]
    subtype: Optional[str] = None  # Custom string, not enum-restricted
    region: Optional[Region] = None  # Contract-level: US, HK, GLOBAL
    frequency: Optional[Frequency] = None
    dims: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


# ============================================================================
# Enum Conversion Utilities
# ============================================================================


def to_enum(enum_class: type[Enum], value: any) -> Optional[Enum]:
    """
    Convert string or enum value to enum instance.

    Shared utility for converting YAML config values to enums.
    Handles both value-based ("market-data") and name-based ("MARKET_DATA") lookups.

    Args:
        enum_class: Enum class to convert to
        value: String value, enum instance, or None

    Returns:
        Enum instance or None if value is None

    Raises:
        ValueError: If value doesn't match any enum member

    Example:
        >>> to_enum(Domain, "market-data")
        <Domain.MARKET_DATA: 'market-data'>
        >>> to_enum(Frequency, None)
        None
    """
    if value is None:
        return None

    if isinstance(value, enum_class):
        return value

    # Try value match first (e.g., "market-data")
    for member in enum_class:
        if member.value == value:
            return member

    # Try name match (e.g., "MARKET_DATA")
    for member in enum_class:
        if member.name == value:
            return member

    raise ValueError(f"Unknown {enum_class.__name__}: {value}")


def dataset_type_from_config(config: dict) -> DatasetType:
    """
    Create DatasetType from YAML configuration dictionary.

    Shared utility for creating DatasetType instances from builder.yaml, model.yaml, etc.
    Performs enum conversion and validation.

    Args:
        config: Configuration dict with domain, asset_class, subdomain, etc.

    Returns:
        DatasetType instance

    Raises:
        ValueError: If domain is missing or enum values are invalid

    Example:
        >>> config = {"domain": "market-data", "subdomain": "bars", "frequency": "daily"}
        >>> dataset_type_from_config(config)
        DatasetType(domain=<Domain.MARKET_DATA: 'market-data'>, ...)
    """
    domain = to_enum(Domain, config.get("domain"))
    if domain is None:
        raise ValueError("Domain is required in dataset type configuration")

    return DatasetType(
        domain=domain,
        asset_class=to_enum(AssetClass, config.get("asset_class")),
        subdomain=to_enum(Subdomain, config.get("subdomain")),
        subtype=config.get("subtype"),  # Custom string, no enum conversion
        region=to_enum(Region, config.get("region")),
        frequency=to_enum(Frequency, config.get("frequency")),
    )
