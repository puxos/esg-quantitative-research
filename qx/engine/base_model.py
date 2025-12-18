import abc
import os
from typing import Dict, List, Optional

import pandas as pd
import yaml

from qx.common.config_utils import resolve_parameters
from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.types import DatasetType, dataset_type_from_config
from qx.engine.processed_writer import ProcessedWriterBase


class BaseModel(abc.ABC):
    def __init__(
        self,
        package_dir: str,
        loader: "TypedCuratedLoader",
        writer: ProcessedWriterBase,
        overrides: Optional[Dict] = None,
    ):
        """
        Initialize model from package directory.

        Args:
            package_dir: Path to model package containing model.yaml
            loader: High-level typed curated data loader (contains registry, backend, resolver)
            writer: High-level processed data writer
            overrides: Parameter overrides
        """
        with open(os.path.join(package_dir, "model.yaml"), "r") as f:
            cfg = yaml.safe_load(f)

        # Store high-level abstractions
        self.loader = loader
        self.writer = writer
        self.registry = loader.registry
        self.info = cfg["model"]
        self.inputs_cfg = cfg["io"]["inputs"]
        self.output_dt = dataset_type_from_config(cfg["io"]["output"]["type"])
        self.params = resolve_parameters(cfg.get("parameters", {}), overrides)
        self.constraints = cfg.get("constraints", {})

    def _resolve_inputs(
        self,
        available_types: List[DatasetType],
        partitions_by_input: Dict[str, Dict[str, str]] = None,
    ) -> Dict[str, DatasetContract]:
        """
        Resolve input contracts from available dataset types.

        Args:
            available_types: Types emitted by loaders (may have None for frequency/region)
            partitions_by_input: Partition values to refine types for registry lookup

        Returns:
            Dict mapping input name to resolved contract
        """
        resolved = {}
        partitions_by_input = partitions_by_input or {}

        for inp in self.inputs_cfg:
            name = inp["name"]
            pattern = dataset_type_from_config(inp["type"])
            match = [
                dt
                for dt in available_types
                if (
                    # Check domain (None = match any)
                    (pattern.domain is None or dt.domain == pattern.domain)
                    # Check subdomain (None = match any)
                    and (pattern.subdomain is None or dt.subdomain == pattern.subdomain)
                    # Check asset_class (None = match any)
                    and (
                        pattern.asset_class is None
                        or dt.asset_class == pattern.asset_class
                    )
                    # Check subtype (None = match any)
                    and (pattern.subtype is None or dt.subtype == pattern.subtype)
                    # Check region (None = match any)
                    and (pattern.region is None or dt.region == pattern.region)
                    # Check frequency (None = match any)
                    and (pattern.frequency is None or dt.frequency == pattern.frequency)
                )
            ]
            if inp.get("required", True) and not match:
                raise ValueError(f"Missing required input '{name}' type {pattern}")
            if match:
                matched_type = match[0]

                # Refine matched type with partition values for registry lookup
                partitions = partitions_by_input.get(name, {})
                if partitions:
                    # Apply partition values to create specific type for registry
                    from qx.common.types import Frequency, Region

                    # Convert partition string values to enums
                    # Note: "exchange" is a partition key but NOT a DatasetType field
                    # Only apply "region" from partitions to the DatasetType
                    region = None
                    if "region" in partitions:
                        region_val = partitions["region"]
                        if isinstance(region_val, str):
                            region = Region[region_val.upper().replace("-", "_")]
                        else:
                            region = region_val
                    # "exchange" is NOT mapped to region - it's only a partition key

                    frequency = None
                    if "frequency" in partitions:
                        freq_val = partitions["frequency"]
                        if isinstance(freq_val, str):
                            frequency = Frequency[freq_val.upper().replace("-", "_")]
                        else:
                            frequency = freq_val

                    refined_type = DatasetType(
                        domain=matched_type.domain,
                        asset_class=matched_type.asset_class,
                        subdomain=matched_type.subdomain,
                        subtype=matched_type.subtype,
                        region=region if region is not None else matched_type.region,
                        frequency=(
                            frequency
                            if frequency is not None
                            else matched_type.frequency
                        ),
                    )
                    matched_type = refined_type

                # Use pattern matching to find contract (handles None values)
                resolved[name] = self.registry.find_matching(matched_type)
        return resolved

    @abc.abstractmethod
    def run_impl(
        self, inputs: Dict[str, pd.DataFrame], params: Dict[str, any], **kwargs
    ) -> pd.DataFrame: ...

    def run(
        self,
        available_types: List[DatasetType],
        partitions_by_input: Dict[str, Dict[str, str]],
        **kwargs,
    ) -> pd.DataFrame:
        contracts_by_name = self._resolve_inputs(available_types, partitions_by_input)
        inputs_df = {
            name: self.loader.load(c.dataset_type, partitions_by_input.get(name, {}))
            for name, c in contracts_by_name.items()
        }
        outputs = self.run_impl(inputs_df, params=self.params, **kwargs).assign(
            model=self.info["id"],
            model_version=self.info["version"],
            featureset_id=self.info.get("featureset_id", ""),
            run_id=kwargs.get(
                "run_id", pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
            ),
            run_ts=pd.Timestamp.utcnow(),
        )
        # Pass full dataset_type (includes subtype) to writer
        # Convert subdomain enum to string value for path construction
        output_type_str = (
            self.output_dt.subdomain.value
            if hasattr(self.output_dt.subdomain, "value")
            else str(self.output_dt.subdomain)
        )
        self.writer.write(
            outputs, output_type_str, self.info["id"], dataset_type=self.output_dt
        )
        return outputs
