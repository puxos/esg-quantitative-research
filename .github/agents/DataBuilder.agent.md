---
description: QuantX AI Stock Data Fetcher - Expert mode for building a containerized financial data pipeline
tools: ['edit', 'runNotebooks', 'search', 'new', 'runCommands', 'runTasks', 'Azure MCP/*', 'usages', 'vscodeAPI', 'problems', 'changes', 'testFailure', 'openSimpleBrowser', 'fetch', 'githubRepo', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-toolsai.jupyter/configureNotebook', 'ms-toolsai.jupyter/listNotebookPackages', 'ms-toolsai.jupyter/installNotebookPackages', 'extensions', 'todos', 'runTests']
model: Claude Sonnet 4.5
---

# QuantX Financial Data Builder — Copilot Chat Mode

Enterprise-grade financial data pipeline development with Python. Focus on building a robust, containerized Python worker that downloads historical daily OHLCV data for various stock markets, writes partitioned Parquet files, and maintains manifest tracking for incremental updates.

## Mission Statement

Build a reliable batch stock data ingestion system that runs as Azure Container Apps Scheduled Jobs, featuring:

- Universal support for S&P500
- Historical backfill with resumable operations from 2000 to present
- Automated daily incremental updates
- Idempotent appends to existing Parquet files
- Comprehensive error handling and logging
- ESG data integration framework for future expansion

## Market Universe Evolution

Multiple market-specific universes, following the industry standard. The market-specific logic is encapsulated in separate builder classes (US, HK, JP, EU, etc.), allowing easy extension to new markets.
Phased approach to gradually expand coverage:

Storing price data once per ticker and managing universe membership separately.

### Design Goals

- Single source of truth for prices per exchange+ticker (no duplication across universes).
- Point‑in‑time membership to support “as‑of” backtests.
- Scalable partitioning for efficient scans/filtering in Spark/Polars/Pandas/SQL engines.
- Extensible to multiple exchanges, frequencies (daily/minute), adjusted vs. raw.

### Data Lake folder structure (Parquet)

Use Hive‑style folders to encode partition columns in the path. Keep prices separate from membership and metadata.

```plain
/data/curated/
├── tickers/                        # FACT: immutable price history
    ├── exchange=us/
        ├── ticker=AAPL/
          ├── prices/
            ├──freq=daily/
              ├── year=2000/           # coarse time partition
                ├── part-000.parquet
              ├── year=2001/
                ├── part-000.parquet
            ├── freq=monthly/
              ├── year=2024/
                ├── part-000.parquet
          ├── esg/
            ├── year=2006/
              ├── part-000.parquet
            ├── year=2007/
              ├── part-000.parquet
          ├── fundamentals/
            ├── statement=income/
              ├── year=2020/
                ├── part-000.parquet
              ├── year=2021/
                ├── part-000.parquet
├── membership/                    # DIM: index/sector universes
    ├── universe=sp500/
        ├── mode=daily/
            sp500_membership_daily.parquet         # (date, ticker_id)
        ├── mode=intervals/
            sp500_membership_intervals.parquet    # (ticker_id, start_date, end_date)
    ├── universe=nasdaq100/
        ├── mode=intervals/
            nasdaq100_membership_intervals.parquet
├── metadata/                      # DIM tables (slowly changing)
    symbols.parquet              # ticker_id, exchange, symbol, issuer_id, currency, status, dates...
    issuers.parquet              # issuer_id, legal_name, country, sector (optional)
    exchanges.parquet            # exchange_code, timezone, currency
    corporate_actions.parquet    # ticker_id, action_type, ex_date, split_ratio, dividend, etc.
    load_logs.parquet            # ingestion audits
├── indexes/                       # optional materialized “views” for fast reads
    latest_daily_prices/         # compact index for last trading day snapshot by ticker_id
```

### Why this layout?

- Prices are partitioned by exchange, ticker, freq, adj, and year—good balance between performance and file management (avoids too many tiny files, yet enables time‑bounded reads).
- Membership lives separately, partitioned by universe and mode (daily / intervals). No price duplication.
- Metadata centralizes symbol and issuer mapping to handle renames and cross‑listings.

### Canonical IDs & symbol handling

To avoid confusion around ticker changes and cross‑listings:

- ticker_id (surrogate key): unique internal integer/uuid per exchange+symbol pair.
  Example: US:AAPL → ticker_id=1001; HK:9988.HK → ticker_id=8023.
- issuer_id: maps multiple tickers to one underlying company (e.g., cross‑listings: US:BABA and HK:9988.HK share issuer_id).
- Symbols table (metadata/symbols.parquet)

```plain
ticker_id: int
exchange: string  # 'us', 'hk', 'jp', ...
symbol: string    # 'AAPL', '0005.HK'
issuer_id: int    # nullable
currency: string  # 'USD', 'HKD'
status: string    # 'active' | 'delisted'
first_trade_date: date
delisted_date: date|null
primary_listing: bool
```

**Keep prices keyed by ticker_id internally (even if the path uses ticker=SYMBOL for readability). This makes joins robust if a symbol renames.**

### Price schema (Parquet)

Daily — the common case for backtests:

```plain
date: date
gvkey: int        # GVKEY identifier
open: double
high: double
low: double
close: double
volume: int64
adj_open: double
adj_high: double
adj_low: double
adj_close: double
adj_volume: int64
div_cash: double
split_factor: double
exchange: string         # redundant but handy in file
currency: string         # e.g., 'USD'
freq: string             # 'daily'
year: int                # materialized for partition pruning
```

- Compression: use snappy for balanced speed/size (default).

## Universe membership modeling

Two complementary datasets per universe:

1. Daily membership (snapshot table)

- Schema (compact table)

```plain
date: date
gvkey: int
universe: string   # 'sp500'
```

- Purpose: exact constituents for a given as‑of date.

2. Intervals (timeline) (compact table)

- Schema

```plain
gvkey: int
universe: string   # 'sp500'
start_date: date
end_date: date
```

- Purpose: quick range joins (e.g., “was this stock in SP500 on D?”).

### Query patterns

1. Get constituents as‑of a date (intervals table)

```python
import pandas as pd

m_int = pd.read_parquet("data/curated/membership/universe=sp500/mode=intervals/sp500_membership_intervals.parquet")

def constituents_asof(universe: str, asof: str):
    D = pd.to_datetime(asof).date()
    df = m_int[(m_int['universe']==universe) &
               (m_int['start_date'] <= D) &
               (m_int['end_date'] >= D)]
    return df['ticker_id'].unique().tolist()
```

2. Load prices for a universe over a backtest window

```python
import os
import pandas as pd

def load_universe_prices(universe: str, start: str, end: str):
    ids = constituents_asof(universe, end)        # final-day membership or compute rolling windows
    years = range(pd.to_datetime(start).year, pd.to_datetime(end).year + 1)

    all_parts = []
    # resolve symbols to path-friendly names if needed
    sym = pd.read_parquet("data/metadata/symbols.parquet")
    sym = sym[sym['ticker_id'].isin(ids)]

    for _, row in sym.iterrows():
        exch = row['exchange']
        symbol = row['symbol']  # path component
        for y in years:
            p = f"data/prices/exchange={exch}/ticker={symbol}/freq=daily/adj=true/year={y}/part-000.parquet"
            if os.path.exists(p):
                all_parts.append(pd.read_parquet(p))

    prices = pd.concat(all_parts, ignore_index=True)
    msk = (prices['date'] >= pd.to_datetime(start)) & (prices['date'] <= pd.to_datetime(end))
    return prices[msk]
```

3. Point‑in‑time filter (join by date)

For strict “as‑of every date” portfolios, do a date‑wise join using daily membership:

```python
m_daily = pd.read_parquet("data/curated/membership/universe=sp500/mode=daily/sp500_membership_daily.parquet")
prices  = load_universe_prices("sp500", "2015-01-01", "2015-12-31")

# Join on (date, ticker_id) to keep only members on each day
pt_prices = prices.merge(m_daily, how="inner",
                         left_on=['date','ticker_id'],
                         right_on=['date','ticker_id'])
```

## Architecture & Technology Stack

### Core Technologies

- **Python 3.12+** with type hints and mypy compliance
- **Typer** for CLI interface with rich help and structured commands
- **Pydantic** for configuration management and data validation
- **Tenacity** for retry logic and exponential backoff
- **Structlog** for JSON structured logging with correlation IDs
- **Pandas/PyArrow** for data processing and Parquet I/O
- **Azure Storage Blob** + **Azure Identity** for cloud storage
- **Pytest** for comprehensive testing

### Data Sources & Extensibility

- **Primary:** tiingo (robust, free, comprehensive)
- **Future:** CRSP/WRDS/HKEX direct APIs
- **Universe Sources:** CSV, Wikipedia, official exchange listings
- **ESG:** CSV/JSON imports, 3rd party APIs

## CLI Interface & Commands

### Historical Backfill

```bash
python -m quantx-data fetch \
  --universe "SP500" \
  --out-root /data \
  --start 2014-01-01 \
  --end 2024-12-31 \
  --max-workers 10 \
  --log-level INFO

python -m quantx-data backfill \
  --universe "SP500" \
  --out-root /data \
  --manifest meta/symbols_manifest.csv \
  --start 2014-01-01 \
  --end 2024-12-31 \
  --max-workers 10 \
  --fetch-actions \
  --log-level INFO
```

### Daily Incremental Update

```bash
python -m quantx-data update-daily \
  --universe "sp500" \
  --out-root /data \
  --manifest meta/symbols_manifest.csv \
  --max-workers 5 \
  --fetch-actions \
  --log-level INFO
```


### CLI Behavior Patterns

- **Resume Logic:** Check manifest for last_date, fetch incrementally from that point
- **Append Operations:** Add new data with automatic deduplication
- **Rate Limiting:** Respect provider limits with polite delays between requests
- **Error Handling:** Exponential backoff with structured logging and correlation IDs

## Azure Deployment

**Use Azure Functions for:**

- Daily scheduled updates (cheap, automatic scaling)
- Weekly universe refresh
- Trigger-based operations

**Use Container Apps Jobs for:**

- Heavy backfill operations (2-4 hours runtime)
- High parallelism needs (10-20 workers)

## Proposed Architecture

Shared Core Logic -> Two Deployment Targets:

1. **Azure Functions:** (daily updates, universe refresh) - lightweight, scheduled
2. **Container Apps Jobs:** (heavy backfill) - containerized, manual triggers

### Scheduling Strategy

- **Daily Updates:** `0 6 * * 1-5` (6 AM weekdays for T-1 data)
- **Weekly Backfill:** `0 2 * * 6` (2 AM Saturdays for catch-up operations)
- **Manual Triggers:** On-demand execution for ad-hoc data requests

## Implementation Plan

1. Shared core module structure

```plain
quantx-data-builder/
├── config/                # Overall settings, shared by all targets
│   └── settings.yaml
├── src/                   # Shared business logic
│   ├── fetcher/           # Fetchers for different markets
│   ├── universe/          # Universe builders
│   ├── processor/         # Data processing logic
│   ├── storage/           # Storage abstractions
│   └── requirements.txt
├── azure_functions/              # Functions-specific code
│   ├── function_app.py
│   ├── host.json
│   └── requirements.txt
├── container/                    # Container-specific code
│   ├── Dockerfile
│   ├── cli.py                   # CLI entry point
│   └── requirements.txt
└── deploy/
    ├── deploy_functions.sh
    └── deploy_container.sh
```

2. Benefits of Shared Codebase

- Single codebase for all business logic
- Shared dependencies and configurations
- Easy testing - test once, deploy twice
- Cost optimized - right tool for each job
- No code duplication - DRY principle

3. How it works

- Azure Functions imports src/ and wraps in function triggers
- Container Apps imports src/ and wraps in CLI commands
- Both use the same DataFetcher, UniverseBuilder, and Storage modules

## References & Documentation

### Azure Container Apps Best Practices

- [Jobs in Azure Container Apps](https://learn.microsoft.com/en-us/azure/container-apps/jobs?tabs=azure-cli)
- [Manual Job Creation](https://learn.microsoft.com/en-us/azure/container-apps/jobs-get-started-cli?pivots=container-apps-job-manual#create-and-run-a-scheduled-job)
- [Scheduled Job Creation](https://learn.microsoft.com/en-us/azure/container-apps/jobs-get-started-cli?pivots=container-apps-job-scheduled#create-and-run-a-scheduled-job)

### Infrastructure Templates & Samples

- [Python Azure Container Apps Jobs](https://github.com/Azure-Samples/container-apps-jobs)

**Focus:** Build production-ready, maintainable code that scales from MVP to enterprise-grade data pipeline, with robust daily update capabilities and extensibility for future ESG data integration.
