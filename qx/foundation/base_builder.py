import abc
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml

from qx.common.config_utils import resolve_parameters
from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.enum_validator import EnumValidationError, validate_dataset_type_config
from qx.common.env_loader import load_env_file
from qx.common.types import DatasetType, Frequency, dataset_type_from_config
from qx.storage.curated_writer import CuratedWriter
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter


class DataBuilderBase(abc.ABC):
    """
    Base class for data builders (raw → curated).

    YAML-based initialization only - reads configuration from builder.yaml.
    Use with run_builder() factory for DAG orchestration.

    Builder Types:
    --------------
    Builders fall into two categories based on their data source:

    1. SOURCE BUILDERS:
       - Fetch data from EXTERNAL sources (APIs, files, databases)
       - No dependencies on other builders
       - Require authentication (API keys, credentials)
       - Handle network errors, rate limiting
       - Examples: USTreasuryRateBuilder (FRED API), TiingoOHLCVBuilder (Tiingo API)

    2. TRANSFORM BUILDERS:
       - Transform EXISTING curated data into new curated datasets
       - Depend on other builders (must run after dependencies)
       - No authentication required (local disk I/O)
       - Deterministic transformations
       - Examples: MembershipIntervalsBuilder (daily → intervals)

    Implementation Pattern:
    ----------------------
    All builders implement two key methods:

    - fetch_raw(**kwargs) -> pd.DataFrame:
        * SOURCE: Fetch from external API/file/database
        * TRANSFORM: Load from curated storage using self.loader

    - transform_to_curated(raw_df, **kwargs) -> pd.DataFrame:
        * Clean, standardize, and prepare data for curated storage
        * Add metadata (schema_version, ingest_ts)
        * Apply business logic and transformations

    See docs/BUILDER_TYPES.md for complete guide on builder types.
    """

    def __init__(
        self,
        package_dir: str,
        writer: CuratedWriter,
        overrides: Optional[Dict] = None,
    ):
        """
        Initialize builder from YAML configuration.

        Args:
            package_dir: Path to builder package containing builder.yaml
            writer: High-level curated data writer (contains registry, adapter, resolver)
            overrides: Parameter overrides (e.g., {"symbols": ["AAPL"], "start_date": "2020-01-01"})
        """
        # Load .env file automatically (idempotent - only loads once)
        load_env_file()

        # Load builder.yaml from package_dir
        self.package_dir = Path(package_dir)
        yaml_path = self.package_dir / "builder.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"builder.yaml not found in {package_dir}")

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.info = cfg["builder"]

        try:
            # Validate enum values before conversion
            output_cfg = cfg.get("output", {})
            validate_dataset_type_config(output_cfg["dataset"])
            self.output_dt_template = dataset_type_from_config(output_cfg["dataset"])
        except (EnumValidationError, ValueError) as e:
            raise ValueError(
                f"Invalid dataset type in builder.yaml:\n{str(e)}\n"
                f"Check your output.dataset section"
            )
        except KeyError as e:
            raise ValueError(
                f"Missing output.dataset in builder.yaml\n"
                f"Add output.dataset section with domain, subdomain, etc."
            )

        self.params = resolve_parameters(cfg.get("parameters", {}), overrides)
        self.partition_spec = cfg.get("partitions", {})

        # Resolve relative paths in parameters (relative to package directory)
        self._resolve_relative_paths()

        # Store high-level writer and extract components
        self.writer = writer
        self.registry = writer.registry
        self.adapter = writer.adapter

        # Set write mode if specified in parameters
        write_mode = self.params.get("write_mode", "append")
        if write_mode not in ["append", "overwrite"]:
            raise ValueError(
                f"Invalid write_mode '{write_mode}'. Must be 'append' or 'overwrite'"
            )
        self.write_mode = write_mode

        # Use resolver from writer
        self.resolver = writer.resolver

        # Contract will be resolved in build() with actual partition values
        self.contract = None

    def _resolve_relative_paths(self):
        """
        Resolve relative file/directory paths to be relative to package directory.

        This enables package-local raw data storage. Any parameter ending with
        '_path', '_root', '_file', or '_dir' that contains a relative path will
        be resolved relative to the package directory.

        Example:
            Parameter: raw_data_root = "./raw"
            Package: qx_builders/sp500_membership
            Resolved: /absolute/path/to/qx_builders/sp500_membership/raw
        """
        path_suffixes = ("_path", "_root", "_file", "_dir")

        for param_name, param_value in self.params.items():
            # Check if parameter looks like a file/directory path
            if (
                param_name.endswith(path_suffixes)
                and isinstance(param_value, str)
                and param_value
            ):
                path = Path(param_value)
                # Only resolve if it's a relative path
                if not path.is_absolute():
                    resolved_path = (self.package_dir / path).resolve()
                    self.params[param_name] = str(resolved_path)

    @abc.abstractmethod
    def fetch_raw(self, **kwargs) -> pd.DataFrame: ...

    @abc.abstractmethod
    def transform_to_curated(self, raw_df: pd.DataFrame, **kwargs) -> pd.DataFrame: ...

    def build(
        self, partitions: dict, available_types: Optional[List] = None, **kwargs
    ) -> Union[str, List[str]]:
        """
        Execute the build pipeline: fetch → transform → write.

        Args:
            partitions: Partition values (e.g., {"exchange": "US", "frequency": "daily"})
            available_types: Optional list of available DatasetTypes for builders
                with curated data inputs (auto-injected by DAG)
            **kwargs: Additional arguments passed to fetch_raw and transform_to_curated

        Returns:
            Path(s) to written curated data
        """
        # Store available_types for use in fetch_raw/transform if needed
        self.available_types = available_types
        # Resolve contract with actual partition values
        if self.contract is None:
            # Some fields may be null in template but provided in partitions
            # (e.g., frequency is partition key but also part of contract identity)
            from qx.common.types import Frequency

            # Start with template
            actual_dt = self.output_dt_template

            # If frequency is null in template but provided in partitions, resolve it
            if actual_dt.frequency is None and "frequency" in partitions:
                freq_str = partitions["frequency"]
                frequency = Frequency(freq_str) if freq_str else None

                # Create resolved dataset type
                from qx.common.types import DatasetType

                actual_dt = DatasetType(
                    domain=actual_dt.domain,
                    asset_class=actual_dt.asset_class,
                    subdomain=actual_dt.subdomain,
                    subtype=actual_dt.subtype,
                    region=actual_dt.region,
                    frequency=frequency,
                    dims=actual_dt.dims,
                )

            # Find contract for resolved dataset type
            self.contract = self.registry.find(actual_dt)
            if self.contract is None:
                raise ValueError(f"No contract found for output type: {actual_dt}")

        # Pass parameters and partitions to transform/fetch
        fetch_kwargs = {**self.params, **kwargs}
        transform_kwargs = {**self.params, **kwargs, "partitions": partitions}

        curated = self.transform_to_curated(
            self.fetch_raw(**fetch_kwargs), **transform_kwargs
        )
        curated = curated.copy()
        curated["schema_version"] = self.contract.schema_version
        curated["ingest_ts"] = pd.Timestamp.utcnow()

        # Check if we need to partition by additional columns (symbol, year, etc.)
        partition_keys = self.contract.partition_keys
        missing_keys = [
            k for k in partition_keys if k not in partitions and k in curated.columns
        ]

        if missing_keys:
            # Need to split data by missing partition keys and write separately
            output_paths = []
            partition_groups = self._group_by_partitions(curated, missing_keys)

            for partition_values, group_df in partition_groups:
                # Merge partition values with base partitions
                partition_copy = self._merge_partition_values(
                    partitions, missing_keys, partition_values
                )

                # Write this partition
                rel_dir = self.resolver.curated_dir(self.contract, partition_copy)
                filename = f"part-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"
                path = self.adapter.write_batch(
                    group_df, rel_dir, filename, mode=self.write_mode
                )
                output_paths.append(path)

            return output_paths if len(output_paths) > 1 else output_paths[0]
        else:
            # All partition keys provided, write directly
            rel_dir = self.resolver.curated_dir(self.contract, partitions)
            filename = f"part-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"
            return self.adapter.write_batch(
                curated, rel_dir, filename, mode=self.write_mode
            )

    def _group_by_partitions(self, df: pd.DataFrame, partition_keys: List[str]):
        """
        Group DataFrame by partition keys.

        Args:
            df: DataFrame to group
            partition_keys: List of column names to group by

        Returns:
            Iterator of (partition_values, group_df) tuples
        """
        if len(partition_keys) == 1:
            for key, group in df.groupby(partition_keys[0]):
                yield (key,), group
        else:
            for keys, group in df.groupby(partition_keys):
                yield keys if isinstance(keys, tuple) else (keys,), group

    def _merge_partition_values(
        self, base_partitions: dict, keys: List[str], values: tuple
    ) -> dict:
        """
        Merge partition values into base partition dict.

        Args:
            base_partitions: Base partition dictionary
            keys: Partition key names
            values: Partition values (tuple)

        Returns:
            Merged partition dictionary
        """
        result = base_partitions.copy()
        for key, value in zip(keys, values):
            result[key] = value
        return result
