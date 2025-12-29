"""
Pipeline Context - Reusable infrastructure wrapper for pipeline construction.

Provides factory methods for creating tasks without repetitive infrastructure parameters.
"""

from typing import Callable, Dict, Optional

from qx.common.contracts import DatasetRegistry
from qx.engine.processed_writer import ProcessedWriterBase
from qx.orchestration.factories import run_builder, run_loader, run_model
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.curated_writer import CuratedWriter
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter


class PipelineContext:
    """
    Encapsulates infrastructure for pipeline construction.

    Provides factory methods to create loader and model tasks without
    repetitive registry/backend/resolver parameters.

    Example:
        >>> # Setup infrastructure
        >>> ctx = PipelineContext(
        ...     registry=registry,
        ...     backend=backend,
        ...     adapter=adapter,
        ...     resolver=resolver,
        ...     curated_writer=curated_writer,
        ...     processed_writer=processed_writer,
        ...     run_id="my-pipeline-20250129"
        ... )
        >>>
        >>> # Create tasks cleanly
        >>> Task(
        ...     id="LoadData",
        ...     run=ctx.loader(
        ...         package_path="qx_loaders/my_loader",
        ...         overrides={"param": "value"}
        ...     ),
        ...     deps=[]
        ... )
    """

    def __init__(
        self,
        registry: DatasetRegistry,
        backend: LocalParquetBackend,
        adapter: TableFormatAdapter,
        resolver: PathResolver,
        curated_writer: CuratedWriter,
        processed_writer: ProcessedWriterBase,
        run_id: str,
    ):
        """
        Initialize pipeline context with infrastructure.

        Args:
            registry: Dataset registry for contract resolution
            backend: Storage backend (e.g., LocalParquetBackend)
            adapter: Table format adapter for writing data
            resolver: Path resolver for generating storage paths
            curated_writer: Writer for curated datasets
            processed_writer: Writer for processed datasets
            run_id: Unique run identifier for this pipeline execution
        """
        self.registry = registry
        self.backend = backend
        self.adapter = adapter
        self.resolver = resolver
        self.curated_writer = curated_writer
        self.processed_writer = processed_writer
        self.run_id = run_id

    def builder(
        self,
        package_path: str,
        partitions: Optional[Dict] = None,
        overrides: Optional[Dict] = None,
    ) -> Callable:
        """
        Create a builder task with infrastructure pre-configured.

        Args:
            package_path: Path to builder package (e.g., "qx_builders/sp500_membership")
            partitions: Partition specifications for builder output
            overrides: Parameter overrides for the builder

        Returns:
            Callable that executes the builder and returns results

        Example:
            >>> run = ctx.builder(
            ...     package_path="qx_builders/sp500_membership",
            ...     partitions={"universe": "sp500", "mode": "intervals"},
            ...     overrides={"source_file": "custom.csv"}
            ... )
        """
        return run_builder(
            package_path=package_path,
            registry=self.registry,
            adapter=self.adapter,
            resolver=self.resolver,
            partitions=partitions or {},
            overrides=overrides or {},
        )

    def loader(
        self,
        package_path: str,
        overrides: Optional[Dict] = None,
    ) -> Callable:
        """
        Create a loader task with infrastructure pre-configured.

        Args:
            package_path: Path to loader package (e.g., "qx_loaders/historic_members")
            overrides: Parameter overrides for the loader

        Returns:
            Callable that executes the loader and returns results

        Example:
            >>> run = ctx.loader(
            ...     package_path="qx_loaders/historic_members",
            ...     overrides={"universe": "sp500", "start_date": "2020-01-01"}
            ... )
        """
        return run_loader(
            package_path=package_path,
            registry=self.registry,
            backend=self.backend,
            resolver=self.resolver,
            overrides=overrides or {},
        )

    def model(
        self,
        package_path: str,
        partitions: Optional[Dict] = None,
        input_mappings: Optional[Dict] = None,
        overrides: Optional[Dict] = None,
    ) -> Callable:
        """
        Create a model task with infrastructure pre-configured.

        Args:
            package_path: Path to model package (e.g., "qx_models/esg_factor")
            partitions: Partition specifications for model inputs
            input_mappings: Task ID mappings for model inputs
            overrides: Parameter overrides for the model

        Returns:
            Callable that executes the model and returns results

        Example:
            >>> run = ctx.model(
            ...     package_path="qx_models/esg_factor",
            ...     partitions={"esg_scores": {"exchange": "US"}},
            ...     input_mappings={"esg_scores": ["LoadESGPanel"]},
            ...     overrides={"quantile": 0.2}
            ... )
        """
        return run_model(
            package_path=package_path,
            registry=self.registry,
            backend=self.backend,
            resolver=self.resolver,
            writer=self.processed_writer,
            partitions=partitions or {},
            input_mappings=input_mappings or {},
            run_id=self.run_id,
            overrides=overrides or {},
        )

    def __repr__(self) -> str:
        """String representation of pipeline context."""
        return f"PipelineContext(run_id='{self.run_id}')"
