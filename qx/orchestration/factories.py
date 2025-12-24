"""
Dynamic Builder, Loader, and Model Factories for DAG Orchestration

Provides three factory functions for DAG task construction:
- run_builder(): Fetch/create curated data (writes to data lake)
- run_loader(): Read curated data and transform to parameters (memory only)
- run_model(): Process curated data into predictions/analytics (writes processed)

Architecture with package-based loaders:
    BUILDERS (standalone)     → Populate curated data
    LOADERS (packages)        → Read curated → produce parameters/datasets
    BUILDERS (in pipeline)    → Fetch data using Loader outputs
    MODELS (in pipeline)      → Process data using Loader/Model outputs

Example usage:
    # Loader: Get continuous SP500 members
    task_select_universe = Task(
        id="SelectUniverse",
        run=run_loader(
            package_path="qx_loaders/continuous_universe",
            registry=registry,
            backend=backend,
            resolver=resolver,
            overrides={"start_date": "2014-01-01", "end_date": "2024-12-31"}
        )
    )

    # Builder: Fetch OHLCV using Loader output
    task_build_ohlcv = Task(
        id="BuildOHLCV",
        run=lambda context: run_builder(
            "qx_builders/tiingo_ohlcv",
            overrides=lambda ctx: {"symbols": ctx["SelectUniverse"]["output"]}
        ),
        deps=["SelectUniverse"]
    )

    # Model: Process data
    task_run_capm = Task(
        id="RunCAPM",
        run=run_model("qx_models/capm"),
        deps=["BuildOHLCV"]
    )
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from qx.common.contracts import DatasetRegistry
from qx.common.types import AssetClass, DatasetType, Domain, Frequency, Region
from qx.engine.base_model import BaseModel
from qx.engine.processed_writer import ProcessedWriterBase
from qx.foundation.base_builder import DataBuilderBase
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter


def run_loader(
    package_path: str,
    registry: DatasetRegistry,
    backend: LocalParquetBackend,
    resolver: PathResolver,
    overrides: Optional[Dict] = None,
) -> Callable[[], Dict]:
    """
    Create a DAG task callable for a loader package.

    Dynamically loads a loader package (loader.yaml + loader.py) and returns
    a callable that executes the loader when invoked.

    Loaders read curated data and produce lightweight outputs (lists, dicts, DataFrames)
    that can be consumed by downstream Builders or Models. Unlike Builders/Models, Loaders
    do NOT persist their outputs - they exist only in memory and in the DAG context.

    Args:
        package_path: Path to loader package (e.g., "qx_loaders/continuous_universe")
        registry: Dataset registry for resolving contracts
        backend: Storage backend for reading curated data
        resolver: Path resolver for locating data
        overrides: Parameter overrides (e.g., {"start_date": "2014-01-01", "end_date": "2024-12-31"})

    Returns:
        Callable that executes loader and returns task manifest dict with output

    Example:
        task_select_universe = Task(
            id="SelectUniverse",
            run=run_loader(
                package_path="qx_loaders/continuous_universe",
                registry=registry,
                backend=backend,
                resolver=resolver,
                overrides={"start_date": "2014-01-01", "end_date": "2024-12-31"}
            )
        )
    """
    pkg_dir = Path(package_path)

    if not pkg_dir.exists():
        raise FileNotFoundError(f"Loader package not found: {package_path}")

    loader_yaml = pkg_dir / "loader.yaml"
    loader_py = pkg_dir / "loader.py"

    if not loader_yaml.exists():
        raise FileNotFoundError(f"loader.yaml not found in {package_path}")
    if not loader_py.exists():
        raise FileNotFoundError(f"loader.py not found in {package_path}")

    # Extract input requirements from loader.yaml (if any)
    import yaml

    from qx.common.types import dataset_type_from_config

    with open(loader_yaml, "r") as f:
        loader_config = yaml.safe_load(f)

    # Check if loader has curated data inputs
    input_requirements = []
    if "inputs" in loader_config:
        for inp in loader_config["inputs"]:
            input_requirements.append(
                {
                    "name": inp["name"],
                    "type": dataset_type_from_config(inp["type"]),
                    "required": inp.get("required", True),
                    "description": inp.get("description", ""),
                }
            )

    # Determine if this loader needs auto-injection
    needs_auto_injection = len(input_requirements) > 0

    def task_callable(available_types: Optional[List[DatasetType]] = None) -> Dict:
        """Execute loader package and return manifest with output."""
        # Dynamic import of loader module
        from qx.foundation.base_loader import BaseLoader

        module_name = f"loader_mod_{pkg_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, loader_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        # Find loader class (first class extending BaseLoader)
        loader_cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseLoader)
                and obj is not BaseLoader
            ):
                loader_cls = obj
                break

        if loader_cls is None:
            raise RuntimeError(f"No loader class found in {package_path}/loader.py")

        # Create high-level TypedCuratedLoader abstraction
        from qx.foundation.typed_curated_loader import TypedCuratedLoader

        typed_loader = TypedCuratedLoader(backend, registry, resolver)

        # Instantiate loader with high-level abstraction
        loader = loader_cls(
            package_dir=str(pkg_dir),
            loader=typed_loader,
            overrides=overrides or {},
        )

        # Execute loader (pass available_types if loader needs auto-injection)
        if needs_auto_injection and available_types is not None:
            output = loader.load(available_types=available_types)
        else:
            output = loader.load()

        # Determine output type for manifest
        output_type = type(output).__name__
        output_size = None

        if isinstance(output, list):
            output_size = len(output)
        elif isinstance(output, dict):
            output_size = len(output)
        elif hasattr(output, "__len__"):
            output_size = len(output)

        return {
            "status": "success",
            "loader": loader.loader_id,
            "version": loader.version,
            "layer": "loader",
            "output_type": output_type,
            "output_size": output_size,
            "output": output,  # Store actual output for downstream tasks
        }

    # Attach metadata for DAG validation (if loader has curated inputs)
    if input_requirements:
        task_callable.input_requirements = input_requirements
        task_callable._loader_package_path = package_path

    # ENHANCEMENT: Loaders emit verified dataset types
    # Extract dataset types from loader's inputs - these represent datasets
    # the loader verifies as existing (by successfully reading them)
    verified_types = []

    # Extract runtime type parameters from overrides
    # These override null values in YAML (e.g., frequency: null → frequency: daily)
    runtime_params = {
        k: v
        for k, v in (overrides or {}).items()
        if k in ("frequency", "region", "asset_class")
    }

    # Extract inputs from top-level
    inputs_list = loader_config.get("inputs", [])

    for inp in inputs_list:
        # Loader YAML uses "type" key, not "dataset_type"
        if "type" in inp:
            # Pass runtime params to properly resolve parameterized types
            dataset_type = dataset_type_from_config(inp["type"], runtime_params)
            verified_types.append(dataset_type)

    # Attach verified types so DAG can collect them for models
    if verified_types:
        task_callable.output_types = verified_types
        task_callable.type_source = "verified"  # vs "produced" for builders

    return task_callable


def run_builder(
    package_path: str,
    registry: DatasetRegistry,
    adapter: TableFormatAdapter,
    resolver: PathResolver,
    partitions: Optional[Dict] = None,
    overrides: Optional[Dict] = None,
) -> Callable[[], Dict]:
    """
    Create a DAG task callable for a builder package.

    Dynamically loads a builder package (builder.yaml + builder.py) and returns
    a callable that executes the builder when invoked.

    Args:
        package_path: Path to builder package directory (e.g., "qx_builders/sp500_membership")
        registry: Dataset registry for resolving contracts
        adapter: Table format adapter for writing curated data
        resolver: Path resolver for output paths
        partitions: Partition values for output (e.g., {"universe": "sp500", "mode": "daily"})
        overrides: Parameter overrides (e.g., {"min_date": "2014-01-01"})

    Returns:
        Callable that executes builder and returns task manifest dict

    Example:
        task_build_membership = Task(
            id="BuildMembership",
            run=run_builder(
                "qx_builders/sp500_membership",
                registry=registry,
                adapter=adapter,
                resolver=resolver,
                partitions={"universe": "sp500", "mode": "daily"},
                overrides={"min_date": "2014-01-01"}
            )
        )
    """
    pkg_dir = Path(package_path)

    if not pkg_dir.exists():
        raise FileNotFoundError(f"Builder package not found: {package_path}")

    builder_yaml = pkg_dir / "builder.yaml"
    builder_py = pkg_dir / "builder.py"

    if not builder_yaml.exists():
        raise FileNotFoundError(f"builder.yaml not found in {package_path}")
    if not builder_py.exists():
        raise FileNotFoundError(f"builder.py not found in {package_path}")

    # Extract output type and input requirements from builder.yaml
    import yaml

    from qx.common.types import dataset_type_from_config

    with open(builder_yaml, "r") as f:
        builder_config = yaml.safe_load(f)

    output_type = dataset_type_from_config(builder_config["output"]["dataset"])

    # Merge partition values into output type (e.g., frequency from partitions)
    if partitions:
        # Create a new DatasetType with partition values merged
        updated_fields = {}

        # Map partition keys to DatasetType fields
        if "frequency" in partitions and output_type.frequency is None:
            # Convert frequency string to Frequency enum
            freq_str = partitions["frequency"]
            freq_enum = Frequency[freq_str.upper()] if freq_str else None
            updated_fields["frequency"] = freq_enum

        if "region" in partitions and output_type.region is None:
            # Convert region string to Region enum
            region_str = partitions["region"]
            region_enum = Region[region_str.upper()] if region_str else None
            updated_fields["region"] = region_enum

        # Create new DatasetType with merged fields
        if updated_fields:
            output_type = DatasetType(
                domain=output_type.domain,
                asset_class=output_type.asset_class,
                subdomain=output_type.subdomain,
                subtype=output_type.subtype,
                region=updated_fields.get("region", output_type.region),
                frequency=updated_fields.get("frequency", output_type.frequency),
            )

    # Check if builder has curated data inputs (TRANSFORM builders)
    input_requirements = []
    if "inputs" in builder_config:
        for inp in builder_config["inputs"]:
            input_requirements.append(
                {
                    "name": inp["name"],
                    "type": dataset_type_from_config(inp["type"]),
                    "required": inp.get("required", True),
                    "description": inp.get("description", ""),
                }
            )

    # Determine if this builder needs auto-injection
    needs_auto_injection = len(input_requirements) > 0

    def task_callable(available_types: Optional[List[DatasetType]] = None) -> Dict:
        """Execute builder and return task manifest."""
        # Dynamic import of builder module
        module_name = f"builder_mod_{pkg_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, builder_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        # Find builder class (first class extending DataBuilderBase)
        builder_cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, DataBuilderBase)
                and obj is not DataBuilderBase
            ):
                builder_cls = obj
                break

        if builder_cls is None:
            raise RuntimeError(f"No builder class found in {package_path}/builder.py")

        # Create high-level CuratedWriter abstraction
        from qx.storage.curated_writer import CuratedWriter

        curated_writer = CuratedWriter(
            backend=adapter.backend,
            adapter=adapter,
            resolver=resolver,
            registry=registry,
        )

        # Instantiate builder with high-level abstraction
        builder = builder_cls(
            package_dir=str(pkg_dir),
            writer=curated_writer,
            overrides=overrides or {},
        )

        # Execute build (pass available_types if builder needs auto-injection)
        if needs_auto_injection and available_types is not None:
            output_path = builder.build(
                partitions=partitions or {}, available_types=available_types
            )
        else:
            output_path = builder.build(partitions=partitions or {})

        return {
            "status": "success",
            "builder": builder.info.get("id", "unknown"),
            "version": builder.info.get("version", "unknown"),
            "output_path": output_path,
            "layer": "curated",
        }

    # Attach metadata for DAG validation
    task_callable.output_types = [output_type]
    if input_requirements:
        task_callable.input_requirements = input_requirements
    task_callable._builder_package_path = package_path

    return task_callable


def run_model(
    package_path: str,
    registry: DatasetRegistry,
    backend: LocalParquetBackend,
    resolver: PathResolver,
    writer: ProcessedWriterBase,
    partitions: Optional[Dict[str, Dict[str, str]]] = None,
    run_id: Optional[str] = None,
    overrides: Optional[Dict] = None,
    input_mappings: Optional[Dict[str, List[str]]] = None,
) -> Callable[[], Dict]:
    """
    Create a DAG task callable for a model package.

    Dynamically loads a model package (model.yaml + model.py) and returns
    a callable that executes the model when invoked.

    Dataset types are automatically injected by the DAG from dependency outputs.

    Args:
        package_path: Path to model package directory (e.g., "qx_models/capm")
        registry: Dataset registry for resolving contracts
        backend: Storage backend for reading curated data
        resolver: Path resolver for constructing file paths
        writer: Processed data writer
        partitions: Partition filters by input name (e.g., {"risk_free": {"region": "US"}})
        run_id: Run identifier for output tracking
        overrides: Parameter overrides (e.g., {"horizon_d": 252})
        input_mappings: Optional dict mapping input names to task IDs for auto-injection
                       e.g., {"equity_prices": ["LoadOHLCVPanel", "LoadPrices"],
                              "expected_returns": ["BuildCAPM"]}
                       If None, auto-mapping is disabled and inputs loaded from storage.

    Returns:
        Callable that executes model and returns task manifest dict

    Example:
        task_run_capm = Task(
            id="RunCAPM",
            run=run_model(
                "qx_models/capm",
                registry=registry,
                backend=backend,
                resolver=resolver,
                writer=writer,
                partitions={"risk_free": {"region": "US"}},
                run_id="run-001",
                overrides={"horizon_d": 252}
            ),
            deps=["BuildOHLCV", "BuildRiskFree"]
        )
    """
    pkg_dir = Path(package_path)

    if not pkg_dir.exists():
        raise FileNotFoundError(f"Model package not found: {package_path}")

    model_yaml = pkg_dir / "model.yaml"
    model_py = pkg_dir / "model.py"

    if not model_yaml.exists():
        raise FileNotFoundError(f"model.yaml not found in {package_path}")
    if not model_py.exists():
        raise FileNotFoundError(f"model.py not found in {package_path}")

    # Extract output type and input requirements from model.yaml for DAG validation
    import yaml

    from qx.common.types import dataset_type_from_config

    with open(model_yaml, "r") as f:
        model_config = yaml.safe_load(f)

    output_type = dataset_type_from_config(model_config["output"]["type"])

    # Extract input requirements for DAG validation
    input_requirements = [
        {
            "name": inp["name"],
            "type": dataset_type_from_config(inp["type"]),
            "required": inp.get("required", True),
            "description": inp.get("description", ""),
        }
        for inp in model_config.get("inputs", [])
    ]

    def task_callable(available_types: List[DatasetType], ctx: Dict = None) -> Dict:
        """
        Execute model with auto-injected dataset types and optional loader outputs.

        Args:
            available_types: Dataset types injected by DAG from dependencies
            ctx: DAG context containing outputs from dependency tasks (loader outputs)
        """
        # Dynamic import of model module
        module_name = f"model_mod_{pkg_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, model_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        # Find model class (first class extending BaseModel)
        model_cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
            ):
                model_cls = obj
                break

        if model_cls is None:
            raise RuntimeError(f"No model class found in {package_path}/model.py")

        # Create high-level TypedCuratedLoader abstraction
        from qx.foundation.typed_curated_loader import TypedCuratedLoader

        typed_loader = TypedCuratedLoader(backend, registry, resolver)

        # Instantiate model with high-level abstractions
        model = model_cls(
            package_dir=str(pkg_dir),
            loader=typed_loader,
            writer=writer,
            overrides=overrides or {},
        )

        # Extract pre-loaded inputs from loader/model outputs in context (if available)
        inputs_df = None
        if ctx is not None and input_mappings is not None:
            # Map input names to loader/model outputs from dependencies
            # Model YAML declares inputs with names (e.g., "equity_prices", "expected_returns", "market_betas")
            # Match these to task outputs in context
            inputs_df = {}

            for inp in model.inputs_cfg:
                input_name = inp["name"]
                potential_tasks = input_mappings.get(input_name, [])

                # Try to find matching loader/model output in context
                for task_id in potential_tasks:
                    if task_id in ctx and isinstance(ctx[task_id], dict):
                        task_result = ctx[task_id]
                        if "output" in task_result:
                            inputs_df[input_name] = task_result["output"]
                            print(
                                f"  ✓ Mapped {task_id} → {input_name} ({len(task_result['output'])} rows)"
                            )
                            break

            if inputs_df:
                # Only use inputs_df if we found all required inputs
                required_inputs = [
                    i["name"] for i in model.inputs_cfg if i.get("required", True)
                ]
                if len(inputs_df) < len(required_inputs):
                    missing = set(required_inputs) - set(inputs_df.keys())
                    print(f"  ⚠ Missing required inputs: {missing}")
                    print(f"  ⚠ Falling back to loading from curated storage")
                    inputs_df = None  # Fall back to loading from storage
                else:
                    print(
                        f"  ✅ All {len(inputs_df)} required inputs mapped from loader/model outputs"
                    )
            else:
                # No mappings found, fall back to storage
                inputs_df = None

        # Execute model with auto-injected types and optional pre-loaded inputs
        output_df = model.run(
            available_types=available_types,
            partitions_by_input=partitions or {},
            inputs_df=inputs_df,
            run_id=run_id,
        )

        return {
            "status": "success",
            "model": model.info.get("id", "unknown"),
            "version": model.info.get("version", "unknown"),
            "rows": int(len(output_df)),
            "layer": "processed",
            "output": output_df,  # Include output DataFrame for downstream models
        }

    # Attach metadata for DAG validation and auto-injection
    task_callable.output_types = [output_type]
    task_callable.input_requirements = input_requirements
    task_callable.is_model = True  # Mark as model task for DAG
    task_callable._model_package_path = package_path  # For error messages

    return task_callable
