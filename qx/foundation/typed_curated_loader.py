"""
Typed Curated Data Loader

Provides type-safe loading of curated datasets using DatasetType contracts.

This abstraction:
1. Maps DatasetType → DatasetContract via registry
2. Resolves storage paths using resolver
3. Loads data from backend
4. Ensures loaders and models use contract-based loading (not hardcoded paths)

Example usage:
    loader = TypedCuratedLoader(backend=backend, registry=registry, resolver=resolver)

    # Load membership data by type
    df = loader.load(
        dataset_type=DatasetType(domain=Domain.MEMBERSHIP, subdomain="sp500_membership_intervals"),
        partitions={"universe": "sp500", "mode": "intervals"}
    )
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.types import DatasetType
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.pathing import PathResolver


class TypedCuratedLoader:
    """
    Type-safe loader for curated datasets.

    Uses DatasetType contracts to resolve storage locations and load data.
    Prevents loaders from using hardcoded paths and ensures type safety.
    """

    def __init__(
        self,
        backend: LocalParquetBackend,
        registry: DatasetRegistry,
        resolver: PathResolver,
    ):
        """
        Initialize typed curated loader.

        Args:
            backend: Storage backend for reading Parquet files
            registry: Dataset registry for resolving types → contracts
            resolver: Path resolver for generating storage paths
        """
        self.backend = backend
        self.registry = registry
        self.resolver = resolver

    def load(
        self,
        dataset_type: DatasetType,
        partitions: Optional[Dict[str, str]] = None,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
    ) -> pd.DataFrame:
        """
        Load curated dataset by type and partitions.

        Args:
            dataset_type: The type of dataset to load (resolved via registry)
            partitions: Partition key-value pairs for filtering (e.g., {"universe": "sp500", "mode": "intervals"})
            columns: Optional list of columns to load (None = all columns)
            filters: Optional PyArrow filter expressions as list of tuples
                     Format: [(column, operator, value), ...]
                     Operators: '=', '!=', '<', '>', '<=', '>=', 'in', 'not in'
                     Example: [('ticker', 'in', ['AAPL', 'MSFT'])]

        Returns:
            DataFrame with requested data

        Raises:
            KeyError: If dataset_type not registered
            FileNotFoundError: If no data files found for the partitions

        Example:
            # Load membership intervals for SP500
            df = loader.load(
                dataset_type=DatasetType(
                    domain=Domain.MEMBERSHIP,
                    subdomain="sp500_membership_intervals"
                ),
                partitions={"universe": "sp500", "mode": "intervals"}
            )

            # Load with filters
            df = loader.load(
                dataset_type=esg_type,
                partitions={"exchange": "US", "esg_year": "2020"},
                filters=[('ticker', 'in', ['AAPL', 'MSFT', 'GOOGL'])]
            )
        """
        # Resolve contract from registry
        contract = self.registry.find(dataset_type)

        # Build storage path from contract + partitions
        # May contain wildcards (*) for missing partition keys
        partitions = partitions or {}
        base_dir = self.resolver.curated_dir(contract, partitions)

        # Find all parquet files (handle wildcards in path)
        if "*" in base_dir:
            # Wildcard path - use glob from root to find matching files
            parquet_files = list(self.backend.root.glob(f"{base_dir}/*.parquet"))
            if not parquet_files:
                # Try recursive glob for nested partitions
                parquet_files = list(self.backend.root.glob(f"{base_dir}/**/*.parquet"))
        else:
            # Exact path - check if exists
            base_path = self.backend.root / base_dir
            if not base_path.exists():
                raise FileNotFoundError(
                    f"No data found for dataset type {dataset_type} at {base_path}. "
                    f"Partitions: {partitions}"
                )
            parquet_files = list(base_path.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found for dataset type {dataset_type}. "
                f"Path pattern: {base_dir}, Partitions: {partitions}"
            )

        # Load and concatenate all files
        dfs = []
        for pq_file in parquet_files:
            # Get relative path for backend (base_dir is already relative)
            rel_path = pq_file.relative_to(self.backend.root)
            df = self.backend.read_parquet(
                str(rel_path), columns=columns, filters=filters
            )
            dfs.append(df)

        result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        return result

    def __repr__(self):
        return f"TypedCuratedLoader(backend={self.backend}, resolver_mode={self.resolver.mode})"
