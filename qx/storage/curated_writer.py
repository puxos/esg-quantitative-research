"""
High-level abstraction for writing curated data.

This provides a uniform interface for builders to write curated data without
directly depending on low-level infrastructure (backend, adapter, resolver).

Design Review Recommendation #4: Standardize Initialization
- Use high-level abstractions (CuratedWriter) instead of low-level components
- All components operate at same abstraction level
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.types import DatasetType
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter


class CuratedWriter:
    """
    High-level interface for writing curated datasets.

    Encapsulates backend + adapter + resolver to provide a clean API
    for builders. Handles path resolution, partitioning, and persistence.
    """

    def __init__(
        self,
        backend: LocalParquetBackend,
        adapter: TableFormatAdapter,
        resolver: PathResolver,
        registry: DatasetRegistry,
    ):
        """
        Initialize curated writer.

        Args:
            backend: Storage backend for file operations
            adapter: Table format adapter for writing data
            resolver: Path resolver for dataset paths
            registry: Dataset registry for contract resolution
        """
        self.backend = backend
        self.adapter = adapter
        self.resolver = resolver
        self.registry = registry

    def write(
        self,
        dataset_type: DatasetType,
        data: pd.DataFrame,
        partitions: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
        write_mode: str = "append",
    ) -> str:
        """
        Write curated dataset using type + contract.

        Args:
            dataset_type: Dataset type identifier
            data: DataFrame to write
            partitions: Partition values (e.g., {"exchange": "US", "frequency": "daily"})
            run_id: Optional run identifier for filename
            write_mode: Write mode ("append" or "overwrite")

        Returns:
            Path where data was written

        Raises:
            ValueError: If contract not found or data doesn't match schema
        """
        # Resolve contract
        contract = self.registry.find(dataset_type)
        if not contract:
            raise ValueError(f"No contract found for {dataset_type}")

        # Validate required columns
        missing_cols = set(contract.required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"Data missing required columns for {dataset_type}: {missing_cols}"
            )

        # Resolve output path
        output_path = self.resolver.resolve_path(
            contract=contract,
            partitions=partitions or {},
            run_id=run_id,
        )

        # Write data
        self.adapter.write(
            data=data,
            path=output_path,
            write_mode=write_mode,
        )

        return str(output_path)

    def write_to_path(
        self,
        data: pd.DataFrame,
        path: Union[str, Path],
        write_mode: str = "append",
    ) -> str:
        """
        Write data to explicit path (for non-contract writes).

        Args:
            data: DataFrame to write
            path: Output path
            write_mode: Write mode ("append" or "overwrite")

        Returns:
            Path where data was written
        """
        self.adapter.write(
            data=data,
            path=Path(path),
            write_mode=write_mode,
        )
        return str(path)
