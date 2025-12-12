import abc
import os
from typing import Dict, List, Optional

import pandas as pd
import yaml

from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.types import AssetClass, DatasetType, Domain, Subdomain, Frequency, Region
from qx.engine.processed_writer import ProcessedWriterBase


def _enum(cls, val):
    if val is None:
        return None
    for e in cls:
        if e.value == val or e.name == val:
            return e
    raise ValueError(f"Unknown {cls.__name__}: {val}")


def dt_from_cfg(d: Dict) -> DatasetType:
    return DatasetType(
        _enum(Domain, d["domain"]),
        _enum(AssetClass, d.get("asset_class")),
        _enum(Subdomain, d.get("subdomain")),
        d.get("subtype"),  # Custom string, no enum conversion
        _enum(Region, d.get("region")),
        _enum(Frequency, d.get("frequency")),
    )


class BaseModel(abc.ABC):
    def __init__(
        self,
        package_dir: str,
        registry: DatasetRegistry,
        loader,
        writer: ProcessedWriterBase,
        overrides: Optional[Dict] = None,
    ):
        with open(os.path.join(package_dir, "model.yaml"), "r") as f:
            cfg = yaml.safe_load(f)
        self.registry, self.loader, self.writer = registry, loader, writer
        self.info = cfg["model"]
        self.inputs_cfg = cfg["io"]["inputs"]
        self.output_dt = dt_from_cfg(cfg["io"]["output"]["type"])
        self.params = self._resolve_params(cfg.get("parameters", {}), overrides or {})
        self.constraints = cfg.get("constraints", {})

    def _resolve_params(self, spec: Dict, overrides: Dict) -> Dict:
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
            out[k] = v
        return out

    def _resolve_inputs(
        self, available_types: List[DatasetType]
    ) -> Dict[str, DatasetContract]:
        resolved = {}
        for inp in self.inputs_cfg:
            name = inp["name"]
            pattern = dt_from_cfg(inp["type"])
            match = [
                dt
                for dt in available_types
                if (
                    dt.domain == pattern.domain
                    and dt.subdomain == pattern.subdomain
                    and (
                        pattern.asset_class is None
                        or dt.asset_class == pattern.asset_class
                    )
                    and (pattern.region is None or dt.region == pattern.region)
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
