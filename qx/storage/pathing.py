from dataclasses import dataclass
from typing import Dict
from qx.common.contracts import DatasetContract

@dataclass
class PathResolver:
    def curated_dir(self, c: DatasetContract, partitions: Dict[str, str]) -> str:
        return c.path_template.format(schema_version=c.schema_version, **partitions)
    def processed_dir(self, output_type: str, model: str, run_date: str, c: DatasetContract) -> str:
        return c.path_template.format(output_type=output_type, model=model, run_date=run_date)
