import abc
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml

from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.enum_validator import (
    EnumValidationError,
    validate_dataset_type_config,
    validate_frequency_parameter,
)
from qx.common.types import (
    AssetClass,
    DatasetType,
    Domain,
    Frequency,
    Region,
    Subdomain,
)
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter


def dt_from_cfg(d: Dict) -> DatasetType:
    """
    Convert YAML config dict to DatasetType with strict enum validation.

    Raises:
        EnumValidationError: If any enum value is invalid
    """
    try:
        validated = validate_dataset_type_config(d)
        kwargs = {}
        if validated["domain"]:
            kwargs["domain"] = Domain(validated["domain"])
        if validated["asset_class"]:
            kwargs["asset_class"] = AssetClass(validated["asset_class"])
        if validated["subdomain"]:
            kwargs["subdomain"] = Subdomain(validated["subdomain"])
        if validated["region"]:
            kwargs["region"] = Region(validated["region"])
        if validated["frequency"]:
            kwargs["frequency"] = Frequency(validated["frequency"])
        return DatasetType(**kwargs)
    except EnumValidationError as e:
        raise ValueError(
            f"Invalid dataset type configuration in builder.yaml:\n{str(e)}\n"
            f"Check your io.output.type section in builder.yaml"
        )


class DataBuilderBase(abc.ABC):
    """
    Base class for data builders (raw → curated).

    YAML-based initialization only - reads configuration from builder.yaml.
    Use with run_builder() factory for DAG orchestration.
    """

    def __init__(
        self,
        package_dir: str,
        registry: DatasetRegistry,
        adapter: TableFormatAdapter,
        resolver: PathResolver,
        overrides: Optional[Dict] = None,
    ):
        """
        Initialize builder from YAML configuration.

        Args:
            package_dir: Path to builder package containing builder.yaml
            registry: Dataset registry for resolving contracts
            adapter: Table format adapter for writing curated data
            resolver: Path resolver for output paths
            overrides: Parameter overrides (e.g., {"symbols": ["AAPL"], "start_date": "2020-01-01"})
        """
        # Load builder.yaml from package_dir
        self.package_dir = Path(package_dir)
        yaml_path = self.package_dir / "builder.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"builder.yaml not found in {package_dir}")

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.info = cfg["builder"]
        self.output_dt_template = dt_from_cfg(cfg["io"]["output"]["type"])
        self.params = self._resolve_params(cfg.get("parameters", {}), overrides or {})
        self.partition_spec = cfg.get("partitions", {})

        # Resolve relative paths in parameters (relative to package directory)
        self._resolve_relative_paths()

        # Store components
        self.registry = registry
        self.adapter = adapter

        # Set write mode if specified in parameters
        write_mode = self.params.get("write_mode", "append")
        if write_mode not in ["append", "overwrite"]:
            raise ValueError(
                f"Invalid write_mode '{write_mode}'. Must be 'append' or 'overwrite'"
            )
        self.write_mode = write_mode

        # Apply mode override if specified in builder.yaml
        # This allows packages to force a specific mode (e.g., reference data always uses prod)
        package_mode = self.info.get("mode")
        if (
            package_mode
            and resolver is not None
            and hasattr(resolver, "mode")
            and resolver.mode != package_mode
        ):
            # Create new resolver with overridden mode
            from qx.storage.pathing import PathResolver

            self.resolver = PathResolver(mode=package_mode)
        else:
            self.resolver = resolver

        # Contract will be resolved in build() with actual partition values
        self.contract = None

    def _resolve_params(self, spec: Dict, overrides: Dict) -> Dict:
        """Resolve parameters from spec + overrides (same logic as BaseModel)."""
        out = {}
        for k, s in spec.items():
            v = overrides.get(k, s.get("default"))
            t = s.get("type")

            # Skip type conversion if value is None
            if v is None:
                out[k] = v
                continue

            if t == "int":
                v = int(v)
            elif t == "float":
                v = float(v)
            elif t == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            elif t == "enum":
                allowed = s.get("allowed", [])
                assert v in allowed, f"Param {k} must be one of {allowed}"
            # string type: keep as-is
            out[k] = v
        return out

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
        for param_name, param_value in self.params.items():
            # Check if parameter looks like a file/directory path
            if (
                param_name.endswith("_path")
                or param_name.endswith("_root")
                or param_name.endswith("_file")
                or param_name.endswith("_dir")
            ):

                if isinstance(param_value, str) and param_value:
                    path = Path(param_value)
                    # Only resolve if it's a relative path
                    if not path.is_absolute():
                        resolved_path = (self.package_dir / path).resolve()
                        self.params[param_name] = str(resolved_path)

    @abc.abstractmethod
    def fetch_raw(self, **kwargs) -> pd.DataFrame: ...

    @abc.abstractmethod
    def transform_to_curated(self, raw_df: pd.DataFrame, **kwargs) -> pd.DataFrame: ...

    def build(self, partitions: dict, **kwargs) -> Union[str, List[str]]:
        """
        Execute the build pipeline: fetch → transform → write.

        Args:
            partitions: Partition values (e.g., {"exchange": "US", "frequency": "daily"})
            **kwargs: Additional arguments passed to fetch_raw and transform_to_curated

        Returns:
            Path(s) to written curated data
        """
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
                    region=actual_dt.region,
                    frequency=frequency,
                    dims=actual_dt.dims,
                )

            # Find contract for resolved dataset type
            self.contract = self.registry.find(actual_dt)
            if self.contract is None:
                raise ValueError(f"No contract found for output type: {actual_dt}")

        # Merge parameters into kwargs for backward compatibility
        merged_kwargs = {**self.params, **kwargs, "partitions": partitions}

        curated = self.transform_to_curated(
            self.fetch_raw(**merged_kwargs), **merged_kwargs
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

            # Group by all missing partition keys
            if len(missing_keys) == 1:
                groups = curated.groupby(missing_keys[0])
            else:
                groups = curated.groupby(missing_keys)

            for group_key, group_df in groups:
                # Create partition dict with group values
                partition_copy = partitions.copy()
                if len(missing_keys) == 1:
                    partition_copy[missing_keys[0]] = group_key
                else:
                    # group_key is a tuple when grouping by multiple keys
                    group_key_tuple = (
                        group_key if isinstance(group_key, tuple) else (group_key,)
                    )
                    for i, key in enumerate(missing_keys):
                        partition_copy[key] = group_key_tuple[i]

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
