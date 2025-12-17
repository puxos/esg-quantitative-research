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
        self, available_types: List[DatasetType]
    ) -> Dict[str, DatasetContract]:
        resolved = {}
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
                resolved[name] = self.registry.find(match[0])
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
        contracts_by_name = self._resolve_inputs(available_types)
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
        self.writer.write(outputs, self.output_dt.subdomain, self.info["id"])
        return outputs
