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
