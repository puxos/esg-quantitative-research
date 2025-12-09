import abc
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.types import AssetClass, DatasetType, Domain, Frequency, Region
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter


def _enum(cls, val):
    """Helper to convert string/None to enum value."""
    if val is None:
        return None
    for e in cls:
        if e.value == val or e.name == val:
            return e
    raise ValueError(f"Unknown {cls.__name__}: {val}")


def dt_from_cfg(d: Dict) -> DatasetType:
    """Convert YAML config dict to DatasetType."""
    return DatasetType(
        _enum(Domain, d["domain"]),
        _enum(AssetClass, d.get("asset_class")),
        d.get("subdomain"),
        _enum(Region, d.get("region")),
        _enum(Frequency, d.get("frequency")),
    )


class DataBuilderBase(abc.ABC):
    """
    Base class for data builders (raw → curated).

    Supports both legacy constructor (contract, adapter, resolver) and
    new YAML-based constructor (package_dir, registry, adapter, resolver, overrides).
    """

    def __init__(
        self,
        contract: Optional[DatasetContract] = None,
        adapter: Optional[TableFormatAdapter] = None,
        resolver: Optional[PathResolver] = None,
        package_dir: Optional[str] = None,
        registry: Optional[DatasetRegistry] = None,
        overrides: Optional[Dict] = None,
    ):
        """
        Initialize builder with either legacy args or YAML-based config.

        Legacy mode (backward compatible):
            contract, adapter, resolver

        YAML mode (new):
            package_dir, registry, adapter, resolver, overrides
        """
        # YAML mode: load builder.yaml from package_dir
        if package_dir is not None:
            self.package_dir = Path(package_dir)
            yaml_path = self.package_dir / "builder.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"builder.yaml not found in {package_dir}")

            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)

            self.info = cfg["builder"]
            self.output_dt_template = dt_from_cfg(cfg["io"]["output"]["type"])
            self.params = self._resolve_params(
                cfg.get("parameters", {}), overrides or {}
            )
            self.partition_spec = cfg.get("partitions", {})

            # Resolve relative paths in parameters (relative to package directory)
            self._resolve_relative_paths()

            # Store registry for later contract resolution (in build())
            if registry is None:
                raise ValueError("registry required for YAML-based builder")
            self.registry = registry
            self.adapter = adapter
            self.resolver = resolver

            # Contract will be resolved in build() with actual partition values
            self.contract = None

        # Legacy mode: direct instantiation
        else:
            if contract is None or adapter is None or resolver is None:
                raise ValueError(
                    "Must provide either (package_dir, registry, adapter, resolver) "
                    "or (contract, adapter, resolver)"
                )
            self.contract = contract
            self.adapter = adapter
            self.resolver = resolver
            self.output_dt_template = None
            self.registry = None
            self.params = {}
            self.info = {}
            self.output_dt = None
            self.partition_spec = {}
            self.registry = None

    def _resolve_params(self, spec: Dict, overrides: Dict) -> Dict:
        """Resolve parameters from spec + overrides (same logic as BaseModel)."""
        out = {}
        for k, s in spec.items():
            v = overrides.get(k, s.get("default"))
            t = s.get("type")
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

    def build(self, partitions: dict, **kwargs) -> str:
        """
        Execute the build pipeline: fetch → transform → write.

        Returns:
            Path to written curated data
        """
        # YAML mode: resolve contract with actual partition values (frequency)
        if self.contract is None and self.registry is not None:
            # Merge partition values into output_dt_template
            from qx.common.types import Frequency

            # Get frequency from partitions
            freq_str = partitions.get("frequency")

            # Convert to enum type
            frequency = Frequency(freq_str) if freq_str else None

            # Create actual output_dt with resolved values
            from qx.common.types import DatasetType

            actual_dt = DatasetType(
                domain=self.output_dt_template.domain,
                asset_class=self.output_dt_template.asset_class,
                subdomain=self.output_dt_template.subdomain,
                region=self.output_dt_template.region,  # Use region from template
                frequency=frequency,
                dims=self.output_dt_template.dims,
            )

            # Find contract for actual dataset type
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
                    for i, key in enumerate(missing_keys):
                        partition_copy[key] = group_key[i]

                # Write this partition
                rel_dir = self.resolver.curated_dir(self.contract, partition_copy)
                filename = f"part-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"
                path = self.adapter.write_batch(group_df, rel_dir, filename)
                output_paths.append(path)

            return output_paths if len(output_paths) > 1 else output_paths[0]
        else:
            # All partition keys provided, write directly
            rel_dir = self.resolver.curated_dir(self.contract, partitions)
            filename = f"part-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"
            return self.adapter.write_batch(curated, rel_dir, filename)
