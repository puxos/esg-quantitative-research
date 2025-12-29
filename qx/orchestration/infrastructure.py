"""
Infrastructure Factory - Configuration-driven infrastructure setup.

Provides factory functions to create all necessary infrastructure components
(registry, backend, writers, loaders) from configuration files.
"""

from pathlib import Path
from typing import Dict, Optional

import yaml

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.engine.processed_writer import ProcessedWriterBase
from qx.foundation.typed_curated_loader import TypedCuratedLoader
from qx.orchestration.pipeline_context import PipelineContext
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.curated_writer import CuratedWriter
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter


def create_infrastructure(
    config_path: Optional[str] = None,
    run_id: Optional[str] = None,
    base_uri_override: Optional[str] = None,
) -> PipelineContext:
    """
    Create all infrastructure components from configuration.

    This factory function centralizes infrastructure setup, making it easy to:
    - Switch backends via configuration (local → Azure → S3)
    - Override settings for testing
    - Maintain consistent setup across pipelines

    Args:
        config_path: Path to storage config file. Defaults to "conf/storage.yaml"
        run_id: Unique run identifier. If None, a timestamp-based ID is generated
        base_uri_override: Override base_uri from config (useful for testing)

    Returns:
        PipelineContext with all infrastructure pre-configured

    Example:
        >>> # Standard usage (reads from conf/storage.yaml)
        >>> ctx = create_infrastructure(run_id="my-pipeline-20250129")
        >>>
        >>> # Testing with custom base_uri
        >>> ctx = create_infrastructure(
        ...     run_id="test-run",
        ...     base_uri_override="file://./test_data"
        ... )
        >>>
        >>> # Use in pipeline
        >>> Task(id="LoadData", run=ctx.loader(...))
    """
    # Load configuration
    if config_path is None:
        config_path = "conf/storage.yaml"

    config = _load_storage_config(config_path)

    # Generate run_id if not provided
    if run_id is None:
        import pandas as pd

        run_id = f"pipeline-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"

    # Override base_uri if specified
    base_uri = base_uri_override or config.get("base_uri", "file://.")

    # Create core infrastructure
    registry = DatasetRegistry()
    seed_registry(registry)

    backend = _create_backend(config, base_uri)
    adapter = TableFormatAdapter(backend)
    resolver = PathResolver()

    # Create writers
    curated_writer = CuratedWriter(
        backend=backend, adapter=adapter, resolver=resolver, registry=registry
    )
    processed_writer = ProcessedWriterBase(
        adapter=adapter, resolver=resolver, registry=registry
    )

    # Create PipelineContext (encapsulates everything)
    return PipelineContext(
        registry=registry,
        backend=backend,
        resolver=resolver,
        curated_writer=curated_writer,
        processed_writer=processed_writer,
        run_id=run_id,
    )


def create_typed_curated_loader(
    ctx: PipelineContext,
) -> TypedCuratedLoader:
    """
    Create a TypedCuratedLoader from PipelineContext.

    Some pipelines need direct access to TypedCuratedLoader (e.g., for custom
    loading logic outside the DAG). This helper creates it from existing context.

    Args:
        ctx: Pipeline context with infrastructure

    Returns:
        TypedCuratedLoader configured with context's infrastructure

    Example:
        >>> ctx = create_infrastructure(run_id="my-pipeline")
        >>> loader = create_typed_curated_loader(ctx)
        >>> df = loader.load(dataset_type=..., partitions=...)
    """
    return TypedCuratedLoader(
        backend=ctx.backend,
        registry=ctx.registry,
        resolver=ctx.resolver,
    )


def _load_storage_config(config_path: str) -> Dict:
    """Load storage configuration from YAML file."""
    path = Path(config_path)

    if not path.exists():
        # Return default configuration if file doesn't exist
        return {
            "backend": "local",
            "base_uri": "file://.",
            "table_format": "parquet",
        }

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Extract storage section if nested
    if "storage" in config:
        return config["storage"]
    return config


def _create_backend(config: Dict, base_uri: str):
    """
    Create storage backend from configuration.

    Future: Support multiple backends (ADLS, S3, etc.)
    """
    backend_type = config.get("backend", "local")

    if backend_type == "local":
        return LocalParquetBackend(base_uri=base_uri)
    elif backend_type == "adls":
        # Future: Azure Data Lake Storage
        raise NotImplementedError("Azure Data Lake backend not yet implemented")
    elif backend_type == "s3":
        # Future: AWS S3
        raise NotImplementedError("S3 backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
