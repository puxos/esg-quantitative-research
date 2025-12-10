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
    MEMBERSHIP = "membership"


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

    # Derived Metrics
    FACTORS = "factors"
    RISK_MODELS = "risk-models"
    EVENT_STUDIES = "event-studies"

    # Portfolio
    POSITIONS = "positions"
    TRANSACTIONS = "transactions"
    PNL = "pnl"

    # Data Products
    RESEARCH_EXPORTS = "research-exports"
    DASHBOARD_FEEDS = "dashboard-feeds"

    # Legacy/Generic subdomains (for backward compatibility)
    OHLCV = "ohlcv"  # Legacy: maps to BARS
    TREASURY_RATE = "treasury_rate"  # Legacy: maps to BENCHMARK_RATES
    ZERO_RATE = "zero_rate"  # Legacy: maps to YIELD_CURVES
    PREDICTIONS = "predictions"  # Legacy: maps to RISK_MODELS
    CONTINUOUS_MEMBERS = "continuous_members"  # Legacy
    ESG_FACTORS = "esg_factors"  # Legacy: maps to FACTORS
    TWO_FACTOR_BETAS = "two_factor_betas"  # Legacy: maps to RISK_MODELS
    FACTOR_EXPECTED_RETURNS = "factor_expected_returns"  # Legacy
    PORTFOLIO_WEIGHTS = "portfolio_weights"  # Legacy: maps to POSITIONS
    UNIVERSE_FILTER = "universe_filter"  # Legacy
    RETURNS = "returns"  # Legacy: maps to RISK_MODELS
    DAILY = "daily"  # Legacy: membership data
    INTERVALS = "intervals"  # Legacy: membership data
    GVKEY_MAPPING = "gvkey_mapping"  # Legacy: maps to IDENTIFIERS
    ESG_SCORES_LEGACY = "esg_scores"  # Legacy: maps to ESG_SCORES


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
    - region: Geographic/market region (US, HK, GLOBAL) - used for contract identity
    - frequency: Temporal frequency (daily, weekly, monthly, etc.)
    - dims: Additional dimensions for typed datasets

    Note: 'exchange' at partition level (NYSE, NASDAQ) is separate from 'region' here.
    """

    domain: Domain
    asset_class: Optional[AssetClass]
    subdomain: Optional[Subdomain]
    region: Optional[Region]  # Contract-level: US, HK, GLOBAL
    frequency: Optional[Frequency]
    dims: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
