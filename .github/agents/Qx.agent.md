---
description: Qx - A local-first, swappable architecture for financial modeling with three layers (Data/Loader/Model), strict dataset typing, YAML-driven packages, co-located tests, and storage abstraction. Comprehensive guide covering Builders (SOURCE/TRANSFORM), Loaders (package-based), Models, DAG orchestration, design patterns, and best practices.
tools:
  [
    "vscode",
    "execute",
    "read",
    "edit",
    "search",
    "web",
    "agent",
    "pylance-mcp-server/*",
    "ms-python.python/getPythonEnvironmentInfo",
    "ms-python.python/getPythonExecutableCommand",
    "ms-python.python/installPythonPackage",
    "ms-python.python/configurePythonEnvironment",
    "ms-toolsai.jupyter/configureNotebook",
    "ms-toolsai.jupyter/listNotebookPackages",
    "ms-toolsai.jupyter/installNotebookPackages",
    "todo",
  ]
model: Claude Sonnet 4.5 (copilot)
---

# Qx Framework - Comprehensive Guide

**Local-first, swappable architecture for quantitative research** with three layers—**Builders**, **Loaders**, and **Models**—orchestrated via DAGs, with strict dataset typing and storage abstraction.

## Core Architecture Layers

### 🏗️ Builders - Data Ingestion

**Purpose**: Fetch/create curated datasets  
**Input**: External APIs, raw files, databases  
**Output**: Curated data (typed, partitioned, persisted)  
**Usage**: Standalone OR in-pipeline  
**Package**: `builder.py` + `builder.yaml` + `schema.yaml` + `test_builder.py`

**Two Types**:

- **SOURCE Builder**: Fetches from external sources (APIs, files, databases)
  - Examples: Tiingo OHLCV (API), SP500 Membership (CSV), ESG Scores (Excel)
  - Requires authentication, network I/O, may fail
  - No dependencies on other builders
- **TRANSFORM Builder**: Transforms existing curated data
  - Examples: Daily→Monthly aggregation, Membership intervals synthesis
  - No authentication, local disk I/O, deterministic
  - Depends on other builders

### 📊 Loaders - Data Bridging

**Purpose**: Read curated → transform → lightweight outputs  
**Input**: Curated datasets (via `TypedCuratedLoader`)  
**Output**: Python objects (List, Dict, DataFrame) - **NOT persisted**  
**Usage**: Only in DAG pipelines  
**Package**: `loader.py` + `loader.yaml` + `test_loader.py`

**Examples**:

- Select continuous SP500 members → `List[symbols]`
- Load ESG panel for period → `DataFrame`
- Filter universe by market cap → `List[symbols]`

### 🎯 Models - Analytics & Predictions

**Purpose**: Process curated → predictions/analytics  
**Input**: Curated datasets (via Loaders)  
**Output**: Processed datasets (with run_id, persisted)  
**Usage**: Only in DAG pipelines  
**Package**: `model.py` + `model.yaml` + `test_model.py`

**Examples**:

- ESG Factor Model (ESG scores → factor returns)
- CAPM Model (prices + risk-free → expected returns)
- Portfolio Optimizer (returns → optimal weights)

### ⚙️ Orchestration - DAG Execution

**Purpose**: Coordinate tasks via Directed Acyclic Graph  
**Components**: `Task`, `DAG`, `run_builder()`, `run_loader()`, `run_model()`  
**Benefits**: Explicit dependencies, automatic ordering, cycle detection, resumable, auditable

---

## Core Principles

### 1. Package-Based Architecture

All components (builders, loaders, models) are **packages** with:

- **YAML config**: Declares IO types, parameters, validation rules
- **Python implementation**: Extends base classes (`DataBuilderBase`, `BaseLoader`, `BaseModel`)
- **Co-located tests**: `test_builder.py`, `test_loader.py`, `test_model.py` in same directory
- **Schema definition**: `schema.yaml` for contract definitions

### 2. Typed Datasets

Every dataset has a `DatasetType`:

```python
DatasetType(
    domain=Domain.MARKET_DATA,        # Required
    asset_class=AssetClass.EQUITY,    # Optional
    subdomain=Subdomain.BARS,         # Required
    region=Region.US,                 # Optional
    frequency=Frequency.DAILY         # Optional
)
```

### 3. Contract-Driven Storage

`DatasetContract` binds type → schema:

- **Required columns**: Enforced at write time
- **Partition keys**: Organize data in lake-ready structure
- **Path template**: Renders to `data/curated/{domain}/{subdomain}/...`
- **Schema version**: Immutable versioning

### 4. High-Level Abstractions

- **CuratedWriter**: Encapsulates backend + adapter + resolver + registry
- **TypedCuratedLoader**: Type-safe loading via contracts
- **No raw infrastructure**: Builders/loaders/models use abstractions only

### 5. Storage-Agnostic Design

- **Today**: LocalParquetBackend (file system)
- **Tomorrow**: Azure Data Lake + Delta tables
- **No code changes**: Swap backend in `conf/storage.yaml`

### 6. DAG Orchestration

- **Explicit dependencies**: `Task(id="A", deps=["B", "C"])`
- **Automatic ordering**: DAG executor handles topological sort
- **Cycle detection**: Prevents circular dependencies
- **Resumable execution**: Restart from failed task

---

## Builder Implementation Guide

### Builder Types: SOURCE vs TRANSFORM

**Identify by asking**: Where does `fetch_raw()` get data?

- **External API/file** → SOURCE Builder
- **Curated storage** → TRANSFORM Builder

### SOURCE Builder Example

```python
# qx_builders/tiingo_ohlcv/builder.py
class TiingoOHLCVBuilder(DataBuilderBase):
    """
    SOURCE BUILDER: Fetch OHLCV data from Tiingo API.

    External Source: Tiingo REST API
    Authentication: API key required
    """

    def __init__(self, package_dir: str, writer, overrides=None):
        super().__init__(package_dir, writer, overrides)
        self.api_key = self.params["api_key"]

    def fetch_raw(self, **kwargs) -> pd.DataFrame:
        """Fetch from external API."""
        symbols = self.params["symbols"]
        response = requests.get(
            f"https://api.tiingo.com/iex/{symbol}/prices",
            headers={"Authorization": f"Token {self.api_key}"}
        )
        return pd.DataFrame(response.json())

    def transform_to_curated(self, raw_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Clean and standardize."""
        df = raw_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"close": "price"})
        return df[["date", "symbol", "open", "high", "low", "price", "volume"]]
```

### TRANSFORM Builder Example

```python
# qx_builders/sp500_membership/builder.py (intervals mode)
class SP500MembershipBuilder(DataBuilderBase):
    """
    DUAL-MODE BUILDER:
    - Daily mode (SOURCE): CSV → daily membership
    - Intervals mode (TRANSFORM): Daily → continuous intervals
    """

    def fetch_raw(self, **kwargs) -> pd.DataFrame:
        mode = kwargs.get("partitions", {}).get("mode", "daily")

        if mode == "daily":
            # SOURCE: Load from CSV file
            return pd.read_csv(self.csv_path)
        else:
            # TRANSFORM: Load from curated daily data
            return self.loader.load(
                dt=DatasetType(
                    domain=Domain.INSTRUMENT_REFERENCE,
                    subdomain=Subdomain.INDEX_CONSTITUENTS
                ),
                partitions={"mode": "daily"}
            )

    def transform_to_curated(self, raw_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        mode = kwargs.get("partitions", {}).get("mode", "daily")

        if mode == "intervals":
            # Transform daily → intervals
            return synthesize_membership_intervals(raw_df)
        else:
            return raw_df
```

### Builder YAML Structure

```yaml
# builder.yaml
builder:
  id: my_builder
  version: 1.0.0
  description: "One-line description"

io:
  output:
    type:
      domain: market-data
      asset_class: equity
      subdomain: bars
      frequency: daily

parameters:
  symbols:
    type: list
    default: ["AAPL", "MSFT"]
    description: "Ticker symbols"

  start_date:
    type: str
    required: true
    description: "Start date (YYYY-MM-DD)"

partitions:
  required: [exchange, frequency]
  defaults:
    exchange: US
```

---

## Loader Implementation Guide

### Loader Package Structure

```
qx_loaders/historic_members/
├── loader.py           # Implementation
├── loader.yaml         # Config (IO, parameters, validation)
└── test_loader.py      # Co-located tests
```

### Loader Example

```python
# qx_loaders/historic_members/loader.py
from qx.foundation.base_loader import BaseLoader
from qx.common.types import DatasetType, Domain

class HistoricMembersLoader(BaseLoader):
    """Load continuous SP500 members for period."""

    def load_impl(self) -> List[str]:
        # Use self.loader (TypedCuratedLoader injected)
        df = self.loader.load(
            dt=DatasetType(
                domain=Domain.INSTRUMENT_REFERENCE,
                subdomain="index-constituents"
            ),
            partitions={
                "universe": self.params["universe"],
                "mode": "intervals"
            }
        )

        # Filter continuous members
        start = pd.Timestamp(self.params["start_date"])
        end = pd.Timestamp(self.params["end_date"])

        continuous = df[
            (df['start_date'] <= start) &
            (df['end_date'] >= end)
        ]

        symbols = continuous['ticker'].unique().tolist()
        print(f"✅ Found {len(symbols)} continuous members")
        return symbols
```

### Loader YAML Structure

```yaml
# loader.yaml
loader:
  id: historic_members_loader
  version: 1.0.0
  description: "Load continuous index members"

# ============================================================================
# INPUTS: Curated datasets (resolved via contracts)
# ============================================================================
inputs:
  - dataset_type:
      domain: instrument-reference
      subdomain: index-constituents
    description: "Membership intervals"

# ============================================================================
# PARAMETERS: Configuration values
# ============================================================================
parameters:
  universe:
    type: string
    default: "sp500"
  start_date:
    type: str
    required: true
  end_date:
    type: str
    required: true

# ============================================================================
# OUTPUT: Validation rules
# ============================================================================
output:
  type: list
  item_type: str
  validation:
    allow_empty: false
    min_length: 1
    item_pattern: "^[A-Z0-9.\\-]+$"
```

---

## Model Implementation Guide

### Model Package Structure

```
qx_models/esg_factor/
├── model.py            # Implementation
├── model.yaml          # Config (IO, parameters)
└── test_model.py       # Co-located tests
```

### Model Example

```python
# qx_models/esg_factor/model.py
from qx.engine.base_model import BaseModel

class ESGFactorModel(BaseModel):
    """Build ESG factor portfolios."""

    def run_impl(self) -> pd.DataFrame:
        # Load inputs via self.loader
        esg_scores = self.loader.load(
            dt=DatasetType(
                domain=Domain.ESG,
                subdomain="esg_scores"
            ),
            partitions={"exchange": self.params["exchange"]}
        )

        prices = self.loader.load(
            dt=DatasetType(
                domain=Domain.MARKET_DATA,
                subdomain="bars",
                frequency=Frequency.MONTHLY
            )
        )

        # Process
        factors = self._compute_factor_returns(esg_scores, prices)

        return factors
```

### Model YAML Structure

```yaml
# model.yaml
model:
  id: esg_factor
  version: 1.0.0
  description: "ESG factor model"

io:
  inputs:
    - name: esg_scores
      required: true
      type:
        domain: esg
        subdomain: esg-scores

    - name: prices
      required: true
      type:
        domain: market-data
        subdomain: bars
        frequency: monthly

  output:
    type:
      domain: derived-metrics
      subdomain: factor-returns

parameters:
  sector_neutral:
    type: bool
    default: true

  formation_method:
    type: enum
    default: "tercile"
    allowed: ["tercile", "quintile", "decile"]
```

---

## DAG Orchestration

### DAG Basics

```python
from qx.orchestration.dag import DAG, Task
from qx.orchestration.factories import run_builder, run_loader, run_model

# Define tasks
tasks = [
    Task(
        id="BuildMembership",
        run=run_builder("qx_builders/sp500_membership"),
        deps=None  # Root task
    ),
    Task(
        id="SelectUniverse",
        run=run_loader(
            loader_module="qx_loaders.historic_members",
            loader_class="HistoricMembersLoader",
            overrides={"start_date": "2014-01-01", "end_date": "2024-12-31"}
        ),
        deps=["BuildMembership"]
    ),
    Task(
        id="BuildOHLCV",
        run=lambda ctx: run_builder(
            "qx_builders/tiingo_ohlcv",
            overrides={"symbols": ctx["SelectUniverse"]["output"]}
        )(),
        deps=["SelectUniverse"]
    ),
    Task(
        id="CalculateBetas",
        run=run_model("qx_models/market_beta"),
        deps=["BuildOHLCV"]
    ),
]

# Execute
dag = DAG(tasks=tasks)
results = dag.execute()
```

### Task Manifest Structure

Every task returns a manifest:

```python
{
    "status": "success",           # or "failed"
    "builder": "my_builder",       # Component ID
    "version": "1.0.0",
    "output_path": "data/...",     # Where data was written
    "rows": 1000,                  # Metadata
    "layer": "curated"             # or "processed"
}
```

### Dependency Rules

- **No cycles**: `A→B→C→A` is invalid (detected at compile time)
- **Direct deps only**: Only list immediate dependencies
- **Multiple deps allowed**: `deps=["A", "B", "C"]` waits for all

### Data Flow in DAG

```
BuildMembership (SOURCE)
    ↓ writes curated
SelectUniverse (LOADER)
    ↓ reads curated, returns list
BuildOHLCV (SOURCE, uses loader output)
    ↓ writes curated
LoadPricePanel (LOADER)
    ↓ reads curated, returns DataFrame
CalculateBetas (MODEL, uses loader output)
    ↓ writes processed
```

---

## Design Patterns & Best Practices

### 1. Standardized Initialization

**✅ Use High-Level Abstractions**:

```python
# Builders
class MyBuilder(DataBuilderBase):
    def __init__(self, package_dir, writer, overrides=None):
        super().__init__(package_dir, writer, overrides)

# Loaders
class MyLoader(BaseLoader):
    def __init__(self, package_dir, loader, overrides=None):
        super().__init__(package_dir, loader, overrides)

# Models
class MyModel(BaseModel):
    def __init__(self, package_dir, loader, writer, overrides=None):
        super().__init__(package_dir, loader, writer, overrides)
```

**❌ Don't Use Low-Level Infrastructure**:

```python
# OLD (deprecated)
def __init__(self, package_dir, registry, adapter, resolver, ...):
```

### 2. TypedCuratedLoader Usage

**✅ Type-Safe Loading**:

```python
def load_impl(self):
    df = self.loader.load(
        dt=DatasetType(
            domain=Domain.MARKET_DATA,
            subdomain=Subdomain.BARS
        ),
        partitions={"exchange": "US", "frequency": "daily"},
        columns=["date", "symbol", "price"],
        filters={"date_range": ("2024-01-01", "2024-12-31")}
    )
```

**❌ Don't Hardcode Paths**:

```python
# OLD (deprecated)
df = pd.read_parquet("data/curated/market-data/bars/...")
```

### 3. Co-Located Tests

**✅ Test Structure**:

```
qx_builders/my_builder/
├── builder.py
├── builder.yaml
└── test_builder.py      ✅ Tests live with code
```

**Test Template**:

```python
"""Unit tests for MyBuilder."""

import pytest
from .builder import MyBuilder  # Relative import

@pytest.fixture
def storage_infrastructure():
    registry = DatasetRegistry()
    seed_registry(registry)
    backend = LocalParquetBackend(base_uri="file://.")
    writer = CuratedWriter(backend, ...)
    return {"writer": writer}

@pytest.fixture
def builder(storage_infrastructure):
    return MyBuilder(
        package_dir=str(Path(__file__).parent),
        writer=storage_infrastructure["writer"]
    )

def test_initialization(builder):
    assert builder.info["id"] == "my_builder"

def test_fetch_raw(builder):
    df = builder.fetch_raw()
    assert len(df) > 0

@pytest.mark.integration
def test_full_pipeline(builder):
    result = builder.build(partitions={"exchange": "US"})
    assert result["status"] == "success"
```

### 4. Enum Validation

**✅ Validate YAML Files**:

```bash
# Validate single builder
python -m qx.tools.validate_builder_yaml qx_builders/my_builder/builder.yaml

# Validate all builders
python -m qx.tools.validate_builder_yaml qx_builders/*/builder.yaml

# Show valid enum values
python -m qx.tools.validate_builder_yaml --show-enums
```

**Common Enums**:

- **Domain**: `market-data`, `esg`, `fundamentals`, `reference-rates`, `instrument-reference`, `derived-metrics`
- **AssetClass**: `equity`, `fixed-income`, `fx`, `commodity`, `derivative`
- **Subdomain**: `bars`, `esg-scores`, `yield-curves`, `index-constituents`, etc.
- **Frequency**: `daily`, `weekly`, `monthly`, `quarterly`, `yearly`
- **Region**: `US`, `HK`, `GLOBAL`

### 5. Output Validation

**✅ Define Validation Rules in YAML**:

```yaml
output:
  type: list
  item_type: str
  validation:
    allow_empty: false
    min_length: 1
    max_length: 5000
    item_pattern: "^[A-Z0-9.\\-]+$"
```

```yaml
output:
  type: dataframe
  validation:
    allow_empty: false
    required_columns: ["ticker", "date", "price"]
    min_rows: 1
    max_rows: 1000000
```

### 6. YAML Section Organization

**✅ Clear Structure**:

```yaml
# ============================================================================
# INPUTS: Curated datasets (resolved via contracts)
# ============================================================================
inputs:
  - dataset_type: { ... }

# ============================================================================
# PARAMETERS: Configuration values
# ============================================================================
parameters:
  param1: { ... }

# ============================================================================
# RUNTIME_DEPENDENCIES: Optional task outputs
# ============================================================================
runtime_dependencies:
  - task_id: "BuildData"

# ============================================================================
# OUTPUT: Validation rules
# ============================================================================
output:
  type: list
```

---

## Testing Guide

### Overview

All Qx components (builders, loaders, models) follow a **co-located testing pattern** with comprehensive test coverage organized by functionality. Tests live in the same directory as the source code for easier maintenance and discovery.

### Test File Structure

```
qx_builders/my_builder/
├── builder.py           # Implementation
├── builder.yaml         # Configuration
└── test_builder.py      # ✅ Tests (co-located)
```

### Test Organization Pattern

Based on the GVKEY mapping builder implementation, organize tests into logical sections:

```python
"""
Unit tests for MyBuilder.

Tests cover:
- YAML configuration loading
- Data fetching (external sources or files)
- Data transformation logic
- Schema validation
- Error handling
- Integration with storage layer

This test file is co-located with the builder source code for easier maintenance.
"""

from pathlib import Path
import pandas as pd
import pytest

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.curated_writer import CuratedWriter
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter

from .builder import MyBuilder  # Relative import

# ============================================================================
# Test Configuration
# ============================================================================

DEFAULT_EXCHANGE = "US"

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def storage_infrastructure():
    """Setup storage infrastructure for tests."""
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()
    writer = CuratedWriter(
        backend=backend, adapter=adapter, resolver=resolver, registry=registry
    )

    return {
        "registry": registry,
        "backend": backend,
        "adapter": adapter,
        "resolver": resolver,
        "writer": writer,
    }

@pytest.fixture
def package_dir():
    """Get package directory path."""
    return str(Path(__file__).parent)

@pytest.fixture
def builder(package_dir, storage_infrastructure):
    """Create builder instance."""
    return MyBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
    )

# ============================================================================
# Unit Tests - Utility Functions
# ============================================================================

def test_utility_function_basic():
    """Test basic utility function behavior."""
    # Example: Testing a ticker cleaning function
    assert clean_ticker_symbol("AAPL") == "AAPL"
    assert clean_ticker_symbol("aapl") == "AAPL"
    assert clean_ticker_symbol("  aapl  ") == "AAPL"

def test_utility_function_edge_cases():
    """Test utility function edge cases."""
    assert clean_ticker_symbol("") == ""
    assert clean_ticker_symbol(None) == ""
    assert clean_ticker_symbol(pd.NA) == ""

# ============================================================================
# Unit Tests - Initialization and Configuration
# ============================================================================

def test_builder_initialization(builder):
    """Test that builder initializes correctly from YAML."""
    assert builder is not None
    assert builder.info["id"] == "my_builder"
    assert builder.info["version"] == "1.0.0"

def test_configuration_loading(builder):
    """Test YAML configuration is loaded properly."""
    assert hasattr(builder, "params")
    assert "exchange" in builder.params
    assert builder.params["exchange"] == "US"

def test_raw_file_path(builder):
    """Test that raw file path is set correctly."""
    assert hasattr(builder, "raw_file_path")
    assert "expected_file.xlsx" in str(builder.raw_file_path)

# ============================================================================
# Unit Tests - Data Fetching
# ============================================================================

def test_fetch_raw_file_exists(builder):
    """Test fetching raw data from external source."""
    raw_path = Path(builder.raw_file_path)

    if raw_path.exists():
        df = builder.fetch_raw()

        assert df is not None
        assert len(df) > 0
        assert "expected_column" in df.columns

        print(f"✓ Loaded {len(df):,} raw records")
    else:
        pytest.skip(f"Raw file not found: {raw_path}")

def test_fetch_raw_missing_file(package_dir, storage_infrastructure):
    """Test error handling when source file is missing."""
    builder = MyBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
        overrides={"source_file": "./raw/nonexistent_file.xlsx"},
    )

    with pytest.raises(FileNotFoundError, match="Source file not found"):
        builder.fetch_raw()

# ============================================================================
# Unit Tests - Data Transformation
# ============================================================================

def test_transform_to_curated_basic(builder):
    """Test basic transformation of raw data."""
    raw_df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": ["A", "B", "C"],
        "other_col": ["data1", "data2", "data3"],
    })

    curated = builder.transform_to_curated(raw_df)

    assert len(curated) == 3
    assert "id" in curated.columns
    assert "value" in curated.columns

def test_transform_data_cleaning(builder):
    """Test data cleaning logic in transformation."""
    raw_df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": ["BRK.B", "brk.a", "AAPL"],
    })

    curated = builder.transform_to_curated(raw_df)

    # Test normalization: period → hyphen
    assert curated[curated["id"] == 1]["value"].values[0] == "BRK-B"
    assert curated[curated["id"] == 2]["value"].values[0] == "BRK-A"

def test_transform_removes_invalid_data(builder):
    """Test that invalid data is removed."""
    raw_df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": ["AAPL", "", "MSFT"],
    })

    curated = builder.transform_to_curated(raw_df)

    # Should have 2 rows (AAPL and MSFT), empty value removed
    assert len(curated) == 2
    assert 2 not in curated["id"].values

def test_transform_deduplicates_data(builder):
    """Test that duplicate data is removed (keeping first)."""
    raw_df = pd.DataFrame({
        "id": [1, 1, 2],
        "value": ["AAPL", "MSFT", "GOOGL"],
    })

    curated = builder.transform_to_curated(raw_df)

    assert len(curated) == 2
    assert curated[curated["id"] == 1]["value"].values[0] == "AAPL"

def test_transform_missing_required_column(package_dir, storage_infrastructure):
    """Test error handling when required column is missing."""
    builder = MyBuilder(package_dir=package_dir, writer=storage_infrastructure["writer"])

    raw_df = pd.DataFrame({
        "other_col": ["data1", "data2"],
    })

    with pytest.raises(ValueError, match="Missing 'id' column"):
        builder.transform_to_curated(raw_df)

def test_transform_empty_dataframe(builder):
    """Test transform with empty DataFrame."""
    raw_df = pd.DataFrame()

    curated = builder.transform_to_curated(raw_df)

    assert curated.empty

# ============================================================================
# Unit Tests - Build Pipeline
# ============================================================================

@pytest.mark.integration
def test_build_with_partitions(builder):
    """Test full build pipeline with partitions."""
    raw_path = Path(builder.raw_file_path)

    if not raw_path.exists():
        pytest.skip(f"Raw file not found: {raw_path}")

    # Build with explicit partition
    result = builder.build(partitions={"exchange": "US"})

    assert result is not None
    if isinstance(result, dict):
        assert result["status"] == "success"
        assert "output_path" in result
    else:
        # Legacy: result might be string path
        assert isinstance(result, str)

@pytest.mark.integration
def test_build_with_parameters(builder):
    """Test build with custom parameters."""
    raw_path = Path(builder.raw_file_path)

    if not raw_path.exists():
        pytest.skip(f"Raw file not found: {raw_path}")

    # Build with custom parameter
    result = builder.build(exchange="US")

    assert result is not None

def test_build_empty_data(package_dir, storage_infrastructure):
    """Test build behavior when no data is returned."""
    builder = MyBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
    )

    # Mock transform to return empty DataFrame
    def mock_transform(*args, **kwargs):
        return pd.DataFrame(columns=["id", "value"])

    builder.transform_to_curated = mock_transform

    result = builder.build(partitions={"exchange": "US"})

    # Base builder doesn't check for empty data - it writes it
    if isinstance(result, dict):
        assert result["status"] == "success"
        assert "output_path" in result

# ============================================================================
# Unit Tests - Schema and Output
# ============================================================================

def test_output_schema_validation(builder):
    """Test that output schema matches contract requirements."""
    raw_df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": ["AAPL", "MSFT", "GOOGL"],
    })

    curated = builder.transform_to_curated(raw_df)

    # Check required columns
    assert "id" in curated.columns
    assert "value" in curated.columns

    # Check data types
    assert curated["id"].dtype in [pd.Int64Dtype(), "int64"]
    assert curated["value"].dtype == object

# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
def test_full_pipeline(storage_infrastructure, package_dir):
    """Test complete pipeline: fetch → transform → build."""
    raw_path = Path(package_dir) / "raw" / "source_file.xlsx"

    if not raw_path.exists():
        pytest.skip(f"Raw file not found: {raw_path}")

    builder = MyBuilder(
        package_dir=package_dir,
        writer=storage_infrastructure["writer"],
    )

    # Fetch
    raw_df = builder.fetch_raw()
    assert len(raw_df) > 0

    # Transform
    curated = builder.transform_to_curated(raw_df)
    assert len(curated) > 0
    assert len(curated) <= len(raw_df)  # May be fewer after cleaning

    # Build
    result = builder.build(partitions={"exchange": "US"})
    assert result is not None

# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """
    Example of how to use MyBuilder.

    This is not a test, but serves as documentation.
    """
    from qx.common.contracts import DatasetRegistry
    from qx.common.predefined import seed_registry
    from qx.storage.backend_local import LocalParquetBackend
    from qx.storage.curated_writer import CuratedWriter
    from qx.storage.pathing import PathResolver
    from qx.storage.table_format import TableFormatAdapter

    # Setup infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = LocalParquetBackend(base_uri="file://.")
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()
    writer = CuratedWriter(
        backend=backend, adapter=adapter, resolver=resolver, registry=registry
    )

    # Create builder
    builder = MyBuilder(
        package_dir="qx_builders/my_builder",
        writer=writer,
    )

    # Build data
    result = builder.build(partitions={"exchange": "US"})

    print(f"✅ Built data: {result}")

if __name__ == "__main__":
    # Run example
    example_usage()
```

### Test Execution

**Run all tests in package**:

```bash
pytest qx_builders/my_builder/test_builder.py -v
```

**Run specific test category**:

```bash
# Run only unit tests (exclude integration)
pytest qx_builders/my_builder/test_builder.py -v -m "not integration"

# Run only integration tests
pytest qx_builders/my_builder/test_builder.py -v -m integration
```

**Run with coverage**:

```bash
pytest qx_builders/my_builder/test_builder.py --cov=qx_builders/my_builder --cov-report=html
```

### Test Coverage Guidelines

**Minimum Coverage for All Components**:

1. ✅ **Initialization Tests** (2-3 tests)

   - Builder/loader/model initializes correctly
   - YAML configuration loaded
   - Paths/parameters set correctly

2. ✅ **Data Fetching Tests** (2-4 tests)

   - Successful data fetch
   - Missing file/API error handling
   - Invalid credentials handling (for SOURCE builders)
   - Empty response handling

3. ✅ **Transformation Tests** (5-8 tests)

   - Basic transformation logic
   - Data cleaning/normalization
   - Invalid data removal
   - Deduplication
   - Missing column errors
   - Edge cases (empty input, null values)

4. ✅ **Build/Run Pipeline Tests** (2-3 tests)

   - Build with partitions
   - Build with parameters
   - Empty data handling

5. ✅ **Schema Validation Tests** (1-2 tests)

   - Output schema matches contract
   - Required columns present
   - Data types correct

6. ✅ **Integration Tests** (1-2 tests)
   - Full pipeline (fetch → transform → build)
   - End-to-end with real data (if available)

### Best Practices

**✅ DO**:

- Use `pytest.skip()` when test data isn't available (don't fail)
- Test edge cases: empty data, null values, missing columns
- Use `@pytest.mark.integration` for tests requiring external resources
- Mock external dependencies in unit tests
- Test error handling with `pytest.raises()`
- Include helpful print statements for successful tests
- Organize tests with clear section comments
- Use descriptive test names: `test_transform_removes_empty_tickers`

**❌ DON'T**:

- Call real APIs in unit tests (expensive, flaky)
- Put tests in separate directory far from source
- Test everything in one giant function
- Ignore error cases
- Assume data is always valid
- Skip testing utility functions

### Common Testing Patterns

**Pattern 1: File Existence Checks**:

```python
def test_fetch_raw_file_exists(builder):
    raw_path = Path(builder.raw_file_path)

    if raw_path.exists():
        df = builder.fetch_raw()
        assert len(df) > 0
    else:
        pytest.skip(f"Raw file not found: {raw_path}")
```

**Pattern 2: Error Handling**:

```python
def test_missing_required_column(builder):
    raw_df = pd.DataFrame({"wrong_col": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing 'expected_col' column"):
        builder.transform_to_curated(raw_df)
```

**Pattern 3: Mock Transformations**:

```python
def test_build_empty_data(builder):
    # Mock transform to return empty DataFrame
    def mock_transform(*args, **kwargs):
        return pd.DataFrame(columns=["id", "value"])

    builder.transform_to_curated = mock_transform

    result = builder.build(partitions={"exchange": "US"})
    assert result["status"] == "success"
```

**Pattern 4: Integration Testing**:

```python
@pytest.mark.integration
def test_full_pipeline(storage_infrastructure, package_dir):
    builder = MyBuilder(package_dir, storage_infrastructure["writer"])

    # Fetch → Transform → Build
    raw_df = builder.fetch_raw()
    curated = builder.transform_to_curated(raw_df)
    result = builder.build(partitions={"exchange": "US"})

    assert result is not None
```

---

## Storage Abstraction

### Local Development (Current)

**Config (`conf/storage.yaml`):**

```yaml
storage:
  backend: local
  base_uri: "file://."
  table_format: parquet
  write_mode: append
```

**Code Usage (abstracted)**:

```python
# Components never see storage details
writer = CuratedWriter(backend, adapter, resolver, registry)
builder = MyBuilder(package_dir, writer)  # Just works

loader = TypedCuratedLoader(backend, registry, resolver)
model = MyModel(package_dir, loader, writer)  # Just works
```

### Future Azure Data Lake

**Config Swap** (no code changes):

```yaml
storage:
  backend: adls
  base_uri: "abfs://lake@myaccount.dfs.core.windows.net"
  table_format: delta # or iceberg
  write_mode: append
  small_file_compaction_mb: 128
```

**Benefits**:

- **No model code changes**: All components continue using `writer.write()` and `loader.load()`
- **No path logic changes**: PathResolver still uses contracts
- **Add features**: Delta compaction, time travel, ACID transactions

---

## Why DAGs?

**Benefits of DAG Orchestration**:

1. **Explicit Dependencies**: No hidden coupling between tasks
2. **Automatic Ordering**: Topological sort ensures correct execution sequence
3. **Cycle Detection**: Compile-time error if `A→B→C→A` detected
4. **Resumable**: Skip completed tasks, only run failed/pending
5. **Auditable**: Every task returns manifest with status/version/output
6. **Parallelizable** (future): Execute independent branches concurrently
7. **Testable**: Mock individual task functions, validate full DAG structure

**Local Runner Now, Cloud Later**:

- **Today**: Lightweight Python DAG class (`qx.orchestration.dag`)
- **Future**: Swap to Azure Data Factory / AML Pipelines / Durable Functions (no task code changes)

---

## Folder Structure

```
qx/                              # Core framework (infrastructure)
  ├── common/                   # Types, contracts, predefined registry
  │   ├── dataset_type.py       # Domain, AssetClass, Subdomain enums
  │   ├── contracts.py          # DatasetContract class
  │   └── registry.py           # DatasetRegistry
  ├── storage/                  # Storage abstraction
  │   ├── backend.py            # LocalParquetBackend (+ future ADLS)
  │   ├── table_format.py       # TableFormatAdapter (Parquet/Delta/Iceberg)
  │   ├── path_resolver.py      # Renders paths from contracts
  │   ├── curated_writer.py     # High-level write abstraction
  │   └── typed_loader.py       # TypedCuratedLoader (high-level read)
  ├── foundation/               # Base classes
  │   ├── base_builder.py       # DataBuilderBase
  │   └── base_loader.py        # BaseLoader
  ├── engine/                   # Model layer
  │   ├── base_model.py         # BaseModel
  │   └── processed_writer.py   # ProcessedWriterBase
  ├── orchestration/            # DAG orchestration
  │   ├── dag.py                # DAG, Task classes
  │   └── factories.py          # run_builder, run_loader, run_model
  └── tools/                    # CLI utilities
      └── validate_builder_yaml.py  # YAML validation tool

qx_builders/                    # Builder implementations (packages)
  ├── sp500_membership/
  │   ├── builder.py            # Membership builder (dual-mode)
  │   ├── builder.yaml          # Config (parameters, output validation)
  │   └── test_builder.py       # ✅ Co-located tests
  ├── tiingo_ohlcv/
  │   ├── builder.py            # OHLCV bars builder (SOURCE)
  │   ├── builder.yaml
  │   └── test_builder.py       # ✅ Co-located tests
  ├── gvkey_mapping/
  │   ├── builder.py            # GVKEY-ticker mapping (TRANSFORM)
  │   ├── builder.yaml
  │   └── test_builder.py       # ✅ Co-located tests
  └── esg_score/
      ├── builder.py            # ESG scores builder (SOURCE)
      ├── builder.yaml
      └── test_builder.py       # ✅ Co-located tests

qx_loaders/                     # Loader implementations (packages)
  ├── historic_members/
  │   ├── loader.py             # Historic SP500 members loader
  │   ├── loader.yaml           # Config (inputs, parameters, validation)
  │   └── test_loader.py        # ✅ Co-located tests
  └── price_panel/
      ├── loader.py             # Price panel loader
      ├── loader.yaml
      └── test_loader.py        # ✅ Co-located tests

qx_models/                      # Model implementations (packages)
  ├── market_beta/
  │   ├── model.py              # Beta calculation model
  │   ├── model.yaml            # Config (IO constraints, parameters)
  │   └── test_model.py         # ✅ Co-located tests
  ├── esg_factor/
  │   ├── model.py              # ESG factor model
  │   ├── model.yaml
  │   └── test_model.py         # ✅ Co-located tests
  └── factor_expected_returns/
      ├── model.py              # Expected returns model
      ├── model.yaml
      └── test_model.py         # ✅ Co-located tests

conf/                           # Configuration
  ├── storage.yaml              # Storage backend/format selection
  └── settings.yaml             # API keys, environment settings

data/                           # Data lake structure
  ├── curated/                  # Builder outputs (typed datasets)
  │   ├── market-data/
  │   │   └── bars/
  │   │       └── schema_v1/
  │   │           └── region=US/
  │   │               └── frequency=daily/
  │   │                   └── date=2024-01-01/
  │   │                       └── exchange=NYSE/
  │   │                           └── part-20250101-001.parquet
  │   ├── esg/
  │   └── fundamentals/
  └── processed/                # Model outputs (processed datasets)
      └── derived-metrics/
          └── predictions/
              └── model=market_beta/
                  └── run_date=2025-01-15/
                      └── part-<run_id>.parquet

docs/                           # Comprehensive documentation
  ├── ARCHITECTURE_LAYERS_QUICK_REF.md     # Framework overview
  ├── BUILDER_TYPES_QUICK_REF.md           # SOURCE vs TRANSFORM
  ├── LOADER_PATTERN_GUIDE.md              # Package-based loaders
  ├── TEST_ORGANIZATION_QUICK_REF.md       # Co-located tests
  ├── DESIGN_REVIEW_QUICK_REF.md           # Design review summary
  ├── STANDARDIZED_INITIALIZATION_QUICK_REF.md  # High-level abstractions
  ├── ENUM_VALIDATION_QUICK_REF.md         # Enum validation guide
  └── DAG_ORCHESTRATION_GUIDE.md           # DAG patterns

examples/                       # Working examples
  ├── builder_dag_with_loader.py           # Complete DAG example
  └── standalone_builder.py                # Standalone builder execution

tests/                          # Integration tests
  ├── integration/
  └── fixtures/
```

**Key Organizational Principles**:

- **Framework vs Implementation**: `qx/` is infrastructure, `qx_builders/` / `qx_loaders/` / `qx_models/` are user code
- **Co-Located Tests**: Every component package has `test_*.py` in same directory as source
- **Package Structure**: Every builder/loader/model is a package (YAML + implementation + tests)
- **Data Lake Hierarchy**: `data/curated/` (builders) and `data/processed/` (models)

---

## Migration Path (Local → Lakehouse → Cloud)

### Phase 1: Local Development (Current)

**What We Have**:

- Parquet files on local filesystem (`file://.`)
- Local DAG runner (`qx.orchestration.dag`)
- Fast iteration: run builders/loaders/models in seconds
- Full testing infrastructure: unit + integration tests

**Benefits**:

- Zero cloud dependencies
- Instant feedback loop
- Easy debugging (files on disk)
- No network latency

### Phase 2: Lakehouse Upgrade (Future)

**Config Change Only**:

```yaml
# conf/storage.yaml
storage:
  backend: adls
  base_uri: "abfs://lake@account.dfs.core.windows.net"
  table_format: delta # Upgrade from parquet
```

**New Capabilities**:

- Delta Lake features: ACID transactions, time travel, schema evolution
- OPTIMIZE/VACUUM commands for compaction
- Metadata management with Delta Log
- Lakehouse SQL query support

**No Code Changes**:

- Builders still call `writer.write()`
- Loaders still call `loader.load()`
- Models unchanged
- Path templates work identically

### Phase 3: Cloud Orchestration (Future)

**Orchestration Options**:

- **Azure Data Factory**: Visual pipeline designer, scheduled triggers
- **AML Pipelines**: ML-focused workflows, experiment tracking
- **Durable Functions**: Code-first, serverless, complex workflows

**Task Code Unchanged**:

```python
# Same task definitions work in cloud
Task(
    id="BuildOHLCV",
    run=run_builder("qx_builders/tiingo_ohlcv"),
    deps=["BuildMembership"]
)
```

**What Changes**:

- DAG definition → Cloud pipeline definition
- Local runner → Cloud orchestrator
- Manual trigger → Scheduled/event-driven

### Design Philosophy

**Swap Infrastructure, Keep Logic**:

- Storage backend swappable → `LocalParquetBackend` to `ADLSBackend`
- Table format swappable → `Parquet` to `Delta` or `Iceberg`
- Orchestration swappable → Local DAG to Cloud pipelines
- **Component code never changes** → Builders/Loaders/Models stay identical

---

## Glossary

### Data Concepts

- **Curated Data**: Cleaned, typed, partitioned datasets produced by Builders. Ready for modeling (e.g., OHLCV bars, ESG scores, fundamentals). Written to `data/curated/`.
- **Processed Data**: Model outputs (predictions, portfolios, risk metrics) produced by Models. Includes run lineage and metadata. Written to `data/processed/`.

- **DatasetType**: Identity tuple describing a dataset: `(domain, asset_class, subdomain, region, frequency)`. Used to uniquely identify and discover datasets.

- **DatasetContract**: Binds a DatasetType to its schema, partition keys, and path template. Defines how data is stored and accessed. Auto-discovered from `schema.yaml` files.

### Component Concepts

- **Builder**: Component that ingests data from external sources (SOURCE) or transforms existing curated data (TRANSFORM). Writes to `data/curated/`.

- **Loader**: Component that reads curated data and transforms it into lightweight outputs (lists, dicts, DataFrames). Memory-only, not persisted. Bridges data for downstream tasks.

- **Model**: Component that consumes curated datasets (via TypedCuratedLoader) and produces processed outputs (predictions, analytics). Writes to `data/processed/`.

### Architecture Concepts

- **DAG (Directed Acyclic Graph)**: Task dependency graph for orchestration. No cycles allowed. Enables automatic ordering, resumability, and auditability.

- **Task**: Unit of work in a DAG with unique ID, run function, and dependencies. Returns manifest upon completion.

- **Manifest**: Dictionary describing task execution result: `{status, builder/loader/model, version, output_path, rows, layer}`.

- **CuratedWriter**: High-level abstraction for writing curated datasets. Encapsulates backend + adapter + resolver + registry. Used by Builders.

- **TypedCuratedLoader**: High-level abstraction for reading curated datasets by type. Encapsulates backend + registry + resolver. Used by Loaders and Models.

### Storage Concepts

- **Backend**: Storage implementation (e.g., LocalParquetBackend, future ADLSBackend). Handles actual I/O operations.

- **TableFormatAdapter**: Format-level operations (Parquet, Delta, Iceberg). Handles append/overwrite, future compaction.

- **PathResolver**: Renders lake-ready file paths from DatasetContracts. Ensures consistent path structure across storage backends.

### Type System Concepts

- **Domain**: Top-level category (e.g., `market-data`, `esg`, `fundamentals`). Enum with 11 values.

- **AssetClass**: Asset category (e.g., `equity`, `fixed-income`, `fx`). Enum with 7 values.

- **Subdomain**: Granular data type (e.g., `bars`, `esg-scores`, `yield-curves`). Enum with 50+ values.

- **Frequency**: Temporal granularity (e.g., `daily`, `monthly`, `yearly`). Enum with 5 values.

- **Region**: Geographic scope (e.g., `US`, `HK`, `GLOBAL`). Enum with 3 values.

---

## Architecture Rules & Patterns

### ✅ Allowed Data Flows

```
SOURCE BUILDER → CURATED DATA (external API/file → curated)
TRANSFORM BUILDER → CURATED DATA (curated → curated)
LOADER → parameters (curated → List/Dict/DataFrame, memory only)
LOADER → BUILDER (pass parameters like symbols, date ranges)
LOADER → MODEL (pass data like DataFrames, universes)
BUILDER → CURATED → LOADER (builder writes, loader reads)
MODEL → PROCESSED → MODEL (chain models, processed → processed)
```

### ❌ Prohibited Flows

```
MODEL → BUILDER ❌ (violates layer separation)
LOADER → CURATED ❌ (loaders don't persist, memory only)
BUILDER → PROCESSED ❌ (builders write curated only, not processed)
BUILDER → BUILDER ❌ (use TRANSFORM builder instead)
```

### Layer Separation Principles

1. **Builders**: Write curated data only. Never read processed data.
2. **Loaders**: Read curated data only. Never write any data.
3. **Models**: Read curated data, write processed data. Never write curated.
4. **Orchestration**: Coordinates tasks. Never directly writes data.

### Builder Type Rules

**SOURCE Builders**:

- ✅ Fetch from external APIs, files, databases
- ✅ Require authentication (API keys, credentials)
- ✅ Network I/O expected
- ✅ Document in docstring: `"""SOURCE Builder: Fetches from Tiingo API..."""`
- ❌ Never read curated datasets as inputs

**TRANSFORM Builders**:

- ✅ Read curated datasets (via TypedCuratedLoader)
- ✅ Transform/aggregate/enrich data
- ✅ Local disk I/O only
- ✅ Deterministic (same inputs → same outputs)
- ✅ Document in docstring: `"""TRANSFORM Builder: Aggregates daily bars to monthly..."""`
- ❌ Never fetch from external sources

### YAML Section Rules

**Builders**:

- ✅ SOURCE: Only `parameters` and `output` sections
- ✅ TRANSFORM: Add `inputs` section for curated datasets
- ❌ Never have `io` section (that's for models)

**Loaders**:

- ✅ Have `inputs` section (curated datasets to load)
- ✅ Have `parameters` section (filters, date ranges, etc.)
- ✅ Have `output` section with validation rules
- ❌ Never have `io` section (that's for models)

**Models**:

- ✅ Have `io` section with `inputs` (curated datasets) and `output` (processed dataset type)
- ✅ Have `parameters` section (model hyperparameters)
- ❌ Never have standalone `inputs` section (use `io.inputs` instead)

---

## Quick Tips & Common Patterns

### Naming Conventions

✅ **DO**:

- Keep `subdomain` names stable and descriptive: `bars`, `yield-curves`, `esg-scores`
- Use snake_case for IDs: `sp500_membership`, `tiingo_ohlcv`
- Version schemas in paths: `schema_v1`, `schema_v2` (never mutate history)

❌ **DON'T**:

- Use generic names: `data`, `output`, `temp`
- Mix naming styles: `camelCase` with `snake_case`
- Mutate existing schema versions

### Data Immutability

✅ **DO**:

- Write immutable processed outputs (new `run_id` partitions)
- Append to curated data (new date partitions)
- Create new schema versions for breaking changes

❌ **DON'T**:

- Overwrite existing curated/processed data
- Modify data in-place
- Delete partitions without proper archival

### Type Safety

✅ **DO**:

- Enforce strict input types in `model.yaml` for lego-style composition
- Use enum validation: `python -m qx.tools.validate_builder_yaml`
- Define output validation rules in YAML

❌ **DON'T**:

- Skip YAML validation before committing
- Use hardcoded paths or type assumptions
- Mix different DatasetTypes without explicit contracts

### Data Flow Patterns

✅ **DO**:

- Use Loaders to bridge curated data → builder parameters (avoid hardcoding symbols/universes)
- Pass loader outputs to builders via DAG context
- Chain models via processed data
- Document builder type (SOURCE/TRANSFORM) in docstring

❌ **DON'T**:

- Pass model outputs to builders (violates layer separation)
- Hardcode universes/symbols in builder code
- Read processed data in builders/loaders

### Testing Strategies

✅ **DO**:

- Co-locate tests: `test_builder.py` in same directory as `builder.py`
- Test initialization, fetch, transform, and build separately
- Use `@pytest.mark.integration` for end-to-end tests
- Mock external APIs in unit tests

❌ **DON'T**:

- Put tests in separate `tests/` directory far from source
- Test everything in one giant function
- Call real APIs in unit tests (expensive, flaky)

### Performance Optimization

✅ **DO**:

- Filter early: use `columns` and `filters` in `loader.load()`
- Partition smartly: balance partition size (target: 50-500 MB per file)
- Use appropriate data types: `category` for low-cardinality strings
- Leverage Parquet column pruning

❌ **DON'T**:

- Load entire datasets when you need a subset
- Create thousands of tiny files (< 1 MB)
- Use `object` dtype when categorical works

### Debugging Workflows

✅ **DO**:

- Run builders standalone for quick testing: `python -m qx_builders.my_builder.builder`
- Use `--dev-mode` flags for small data samples
- Check task manifests for execution metadata
- Inspect Parquet files with `pyarrow` or `pandas`

❌ **DON'T**:

- Debug in full DAG runs (too slow)
- Ignore manifest status codes
- Assume data is correct without inspection

### Common Mistakes to Avoid

1. **Wrong initialization pattern**: Using old `(registry, adapter, resolver)` instead of `(writer)` / `(loader)`
2. **Missing builder type**: Not documenting SOURCE vs TRANSFORM in docstring
3. **Loader persistence**: Trying to write data from a loader (loaders are read-only)
4. **Model → Builder flow**: Passing model outputs back to builders (violates architecture)
5. **Hardcoded paths**: Using `pd.read_parquet("data/...")` instead of `loader.load()`
6. **Enum typos**: Using `"equity"` vs `"equities"` (validate with CLI tool)
7. **Missing tests**: Committing packages without co-located test files

---

## Complete Workflow Examples

### Example 1: SP500 Historical Analysis

**Objective**: Calculate market betas for continuous SP500 members (2014-2024)

```python
from qx.orchestration.dag import DAG, Task
from qx.orchestration.factories import run_builder, run_loader, run_model

tasks = [
    # Step 1: Build SP500 membership data (SOURCE builder)
    Task(
        id="BuildMembership",
        run=run_builder("qx_builders/sp500_membership", overrides={"mode": "intervals"}),
        deps=None  # Root task
    ),

    # Step 2: Select continuous members (LOADER)
    Task(
        id="SelectUniverse",
        run=run_loader(
            loader_module="qx_loaders.historic_members",
            loader_class="HistoricMembersLoader",
            overrides={"start_date": "2014-01-01", "end_date": "2024-12-31", "continuity": "full"}
        ),
        deps=["BuildMembership"]
    ),

    # Step 3: Build OHLCV data for selected symbols (SOURCE builder with loader output)
    Task(
        id="BuildOHLCV",
        run=lambda ctx: run_builder(
            "qx_builders/tiingo_ohlcv",
            overrides={
                "symbols": ctx["SelectUniverse"]["output"],  # List from loader
                "start_date": "2014-01-01",
                "end_date": "2024-12-31"
            }
        )(),
        deps=["SelectUniverse"]
    ),

    # Step 4: Build risk-free rate data (SOURCE builder)
    Task(
        id="BuildRiskFree",
        run=run_builder("qx_builders/us_treasury_rate"),
        deps=None  # Can run in parallel with membership
    ),

    # Step 5: Calculate market betas (MODEL)
    Task(
        id="CalculateBetas",
        run=run_model(
            "qx_models/market_beta",
            overrides={"lookback_days": 252, "min_observations": 200}
        ),
        deps=["BuildOHLCV", "BuildRiskFree"]
    ),
]

# Execute DAG
dag = DAG(tasks=tasks)
results = dag.execute()

print(f"Beta calculations completed: {results['CalculateBetas']['status']}")
print(f"Output: {results['CalculateBetas']['output_path']}")
```

### Example 2: ESG Factor Construction

**Objective**: Build ESG factor returns for portfolio optimization

```python
tasks = [
    # Step 1: Build ESG scores (SOURCE builder)
    Task(
        id="BuildESG",
        run=run_builder("qx_builders/esg_score", overrides={"years": list(range(2014, 2025))}),
        deps=None
    ),

    # Step 2: Build GVKEY mapping (TRANSFORM builder, needs membership)
    Task(
        id="BuildGVKEY",
        run=run_builder("qx_builders/gvkey_mapping"),
        deps=None  # Reads raw Excel file
    ),

    # Step 3: Load ESG panel with continuity filtering (LOADER)
    Task(
        id="LoadESGPanel",
        run=run_loader(
            loader_module="qx_loaders.esg_panel",
            loader_class="ESGPanelLoader",
            overrides={"min_years": 5, "min_coverage": 0.8}
        ),
        deps=["BuildESG", "BuildGVKEY"]
    ),

    # Step 4: Build monthly OHLCV (SOURCE builder)
    Task(
        id="BuildMonthlyOHLCV",
        run=run_builder("qx_builders/tiingo_ohlcv", overrides={"frequency": "monthly"}),
        deps=None
    ),

    # Step 5: Calculate ESG factor returns (MODEL)
    Task(
        id="CalculateESGFactor",
        run=run_model(
            "qx_models/esg_factor",
            overrides={"portfolio_method": "quintile", "rebalance_frequency": "monthly"}
        ),
        deps=["LoadESGPanel", "BuildMonthlyOHLCV"]
    ),
]

dag = DAG(tasks=tasks)
results = dag.execute()
```

### Example 3: Data Quality Reporting

**Objective**: Generate coverage report for ESG data

```python
tasks = [
    Task(
        id="BuildESG",
        run=run_builder("qx_builders/esg_score"),
        deps=None
    ),
    Task(
        id="BuildMembership",
        run=run_builder("qx_builders/sp500_membership"),
        deps=None
    ),
    Task(
        id="AnalyzeCoverage",
        run=run_loader(
            loader_module="qx_loaders.esg_coverage",
            loader_class="ESGCoverageAnalyzer",
            overrides={"universe": "sp500", "generate_report": True}
        ),
        deps=["BuildESG", "BuildMembership"]
    ),
]

dag = DAG(tasks=tasks)
results = dag.execute()
# Report saved to data/results/esg_coverage_report.csv
```

---

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                       │
│                     (DAG: Tasks, Dependencies)                   │
└─────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   BUILDERS    │       │    LOADERS    │       │    MODELS     │
│  (Write Only) │       │  (Read Only)  │       │  (Read/Write) │
├───────────────┤       ├───────────────┤       ├───────────────┤
│ • SOURCE      │       │ • Package-    │       │ • Package-    │
│   (External)  │──────▶│   based       │──────▶│   based       │
│ • TRANSFORM   │       │ • Type-safe   │       │ • IO-strict   │
│   (Curated)   │       │   loading     │       │ • Lineage     │
└───────┬───────┘       └───────────────┘       └───────┬───────┘
        │                                               │
        │ writes                                 writes │
        ▼                                               ▼
┌──────────────────┐                       ┌──────────────────┐
│  CURATED DATA    │                       │ PROCESSED DATA   │
│  data/curated/   │                       │ data/processed/  │
│                  │                       │                  │
│ • market-data    │                       │ • predictions    │
│ • esg            │                       │ • portfolios     │
│ • fundamentals   │                       │ • risk-metrics   │
│ • reference      │                       │ • factors        │
└──────────────────┘                       └──────────────────┘
        │                                               │
        └─────────────────┬─────────────────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ STORAGE BACKEND  │
                │  (Swappable)     │
                ├──────────────────┤
                │ • Local Parquet  │ ◀── Now
                │ • Azure Delta    │ ◀── Future
                │ • Iceberg        │ ◀── Future
                └──────────────────┘
```

---

## Key Documentation References

**Essential Reading**:

- [ARCHITECTURE_LAYERS_QUICK_REF.md](docs/ARCHITECTURE_LAYERS_QUICK_REF.md) - Framework overview
- [BUILDER_TYPES_QUICK_REF.md](docs/BUILDER_TYPES_QUICK_REF.md) - SOURCE vs TRANSFORM
- [LOADER_PATTERN_GUIDE.md](docs/LOADER_PATTERN_GUIDE.md) - Package-based loaders
- [TEST_ORGANIZATION_QUICK_REF.md](docs/TEST_ORGANIZATION_QUICK_REF.md) - Co-located tests
- [DAG_ORCHESTRATION_GUIDE.md](docs/DAG_ORCHESTRATION_GUIDE.md) - Task dependencies

**Design Review**:

- [DESIGN_REVIEW_QUICK_REF.md](docs/DESIGN_REVIEW_QUICK_REF.md) - All 6 recommendations
- [STANDARDIZED_INITIALIZATION_QUICK_REF.md](docs/STANDARDIZED_INITIALIZATION_QUICK_REF.md) - High-level abstractions
- [ENUM_VALIDATION_QUICK_REF.md](docs/ENUM_VALIDATION_QUICK_REF.md) - Type safety

**Working Examples**:

- [examples/builder_dag_with_loader.py](examples/builder_dag_with_loader.py) - Complete DAG
- [examples/standalone_builder.py](examples/standalone_builder.py) - Builder execution

---

## Summary

Qx is a **local-first, swappable architecture** for financial data pipelines:

✅ **Three Layers**: Builders (data ingestion), Loaders (data bridging), Models (analytics)  
✅ **Package-Based**: Every component is a package (YAML + Python + co-located tests)  
✅ **Contract-Driven**: Strict typing via DatasetType + DatasetContract  
✅ **Storage-Agnostic**: Swap Parquet → Delta → Iceberg with config change  
✅ **DAG Orchestration**: Explicit dependencies, automatic ordering, cycle detection  
✅ **Builder Types**: SOURCE (external APIs) vs TRANSFORM (curated → curated)  
✅ **High-Level Abstractions**: CuratedWriter, TypedCuratedLoader (no raw infrastructure)  
✅ **Co-Located Tests**: Tests live with source code for easy maintenance

**Start Here**:

1. Read [ARCHITECTURE_LAYERS_QUICK_REF.md](docs/ARCHITECTURE_LAYERS_QUICK_REF.md)
2. Explore working builders: [qx_builders/sp500_membership/](qx_builders/sp500_membership/)
3. Run validation: `python -m qx.tools.validate_builder_yaml`
4. Build your first DAG: [examples/builder_dag_with_loader.py](examples/builder_dag_with_loader.py)
