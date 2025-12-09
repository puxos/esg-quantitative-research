from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from qx.common.types import DatasetType


@dataclass
class DatasetContract:
    dataset_type: DatasetType
    schema_version: str
    required_columns: Tuple[str, ...]
    partition_keys: Tuple[str, ...]
    path_template: str
    # New: schema metadata (populated by SchemaLoader)
    column_definitions: List = field(default_factory=list)  # List[ColumnDefinition]
    filter_definitions: Dict = field(
        default_factory=dict
    )  # Dict[str, FilterDefinition]


class DatasetRegistry:
    def __init__(self):
        self._c: Dict[DatasetType, DatasetContract] = {}

    def register(self, c: DatasetContract):
        self._c[c.dataset_type] = c

    def find(self, dt: DatasetType) -> DatasetContract:
        if dt not in self._c:
            raise KeyError(f"No contract for {dt}")
        return self._c[dt]
