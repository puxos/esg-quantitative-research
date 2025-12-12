# Qx - Finance Modeling Platform

A local-first, swappable architecture for financial modeling, built around three layers—**Data**, **Model**, and **Orchestration**—with strict dataset typing, package-based models, and storage abstraction

## High-Level Overview

- **Data Layer:** Ingests raw sources, builds **curated** datasets (Parquet today), and loads curated data by **typed contracts**.
- **Loader Layer:** Reads curated data and transforms to lightweight outputs (lists, parameters, DataFrames) for downstream tasks. **Not persisted**.
- **Model Layer:** Each model is a **package** (`model.py` + `model.yaml`), consumes curated datasets (strict types), generates **processed** outputs with lineage.
- **Orchestration Layer:** Coordinates tasks via a **DAG** (Directed Acyclic Graph). Manages dependencies, runs, and manifests—local runner now; Azure pipelines later.

## 📁 Project Structure

```plain
qx/                          # Core framework (infrastructure)
  ├── common/               # Types, contracts, predefined registry
  ├── storage/              # Backends, format adapter, path resolver
  ├── foundation/           # Base classes (base_builder, typed_loader)
  ├── engine/               # Model base, processed writer
  ├── orchestration/        # DAG, tasks, factories (run_builder, run_loader, run_model)
  └── utils/                # Utilities (universe.py - loader functions)

qx_builders/                # Builder implementations (outside framework)
  ├── sp500_membership/    # builder.py + builder.yaml
  ├── tiingo_ohlcv/        # builder.py + builder.yaml
  ├── us_treasury_rate/    # builder.py + builder.yaml
  └── esg_score/           # builder.py + builder.yaml

qx_loaders/                 # Loader implementations (outside framework)
  ├── historic_members     # loader.py + loader.yaml
  ├── universe_at_date     # loader.py + loader.yaml
  ├── us_treasury_rate     # loader.py + loader.yaml
  ├── ohlcv_panel          # loader.py + loader.yaml
  ├── esg_panel
  └── market_proxy        

qx_models/                  # Model implementations (outside framework)
  ├── esg_factor/          # model.py + model.yaml
  ├── market_beta/         # model.py + model.yaml
  └── factor_expected_returns/  # model.py + model.yaml

conf/
  └── storage.yaml          # Select backend/format
```

## Layer Responsibilities

### Data Layer (Builders)

- **Builders** (`DataBuilderBase`):  
  `raw → transform → curated` (write Parquet under partitioned path templates).
  - Can run **standalone** (populate data lake) OR **in-pipeline** (fetch on demand)
  - Examples: SP500 Membership, Tiingo OHLCV, US Treasury Rates, ESG Scores
- **Contracts & Types**:
  - `DatasetType`: `(domain, asset_class?, subdomain, exchange?, frequency?)`
  - `DatasetContract`: `(type, schema_version, required_columns, partition_keys, path_template)`
- **Storage & Paths**:
  - `LocalParquetBackend`: local filesystem Parquet IO.
  - `TableFormatAdapter`: format-level writes (append/overwrite, future compaction).
  - `PathResolver`: renders lake-ready paths from contracts.

### Loader Layer

- **Loaders** (lightweight functions):  
  Read curated datasets by `DatasetType` + partition filters, transform to Python objects.
  - **Input**: Curated datasets (via `TypedCuratedLoader`)
  - **Output**: Lists, Dicts, DataFrames (memory only, **NOT persisted**)
  - **Usage**: Only in DAG pipelines (no standalone use)
  - **Factory**: `run_loader(load_fn, registry, backend, resolver)`
  - **Examples**:
    - `get_continuous_sp500_members()` → List[symbols]
    - `filter_universe_by_market_cap()` → List[symbols]
    - `load_esg_panel()` → DataFrame
- **Key Principle**: Loaders bridge curated data and task parameters. They enable **data-driven pipelines** where Builder inputs come from curated data, not hardcoded values.

### Model Layer

- **BaseModel** (config-driven):  
  Loads `model.yaml`, validates **input types** and **parameters**, runs `run_impl()`, and persists **processed outputs** (Parquet) with run metadata.
- **Model package example**: `qx_models/capm/`
  - `model.yaml`: IO type constraints, parameter defaults/ranges.
  - `model.py`: `CAPMModel(BaseModel)` with feature prep and prediction logic.
- **Processed writer** (`ProcessedWriterBase`):  
  `data/processed/{output_type}/model={model}/run_date=YYYY-MM-DD/part-<run_id>.parquet`.

### Orchestration Layer

- **DAG runner (local)**:
  - `Task(id, run, deps)` executes when dependencies are satisfied.
  - `DAG(tasks).execute()` processes the graph.
- **Three factory functions**:
  - `run_builder()` → Execute builder (write curated data)
  - `run_loader()` → Execute loader (read curated → produce parameters)
  - `run_model()` → Execute model (read curated → write processed)
- **Typical flow**:  
  `BuildMembership → SelectUniverse (Loader) → BuildOHLCV → RunCAPM → Portfolio`.

## Dataset Typing — Examples

- **Curated market data (equities OHLCV):**
  - `DatasetType`:  
    `domain=market-data, asset_class=equity, subdomain=bars, region=US|HK, frequency=daily|weekly|monthly`
  - Partitions: `(region, frequency, date, exchange)`
- **Curated risk-free (zero curve):**
  - `DatasetType`:  
    `domain=reference-rates, subdomain=yield-curves, region=US|HK, frequency=D`
  - Partitions: `(region, date, curve_id)`
- **Processed predictions (generic):**
  - `DatasetType`:  
    `domain=derived-metrics, asset_class=equity, subdomain=predictions`
