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

    def find_matching(self, pattern: DatasetType) -> DatasetContract:
        """
        Find contract matching a pattern (None = match any).

        Args:
            pattern: DatasetType with None values as wildcards

        Returns:
            First matching contract

        Raises:
            KeyError: If no matching contract found
        """
        for dt, contract in self._c.items():
            if (
                (pattern.domain is None or dt.domain == pattern.domain)
                and (pattern.subdomain is None or dt.subdomain == pattern.subdomain)
                and (
                    pattern.asset_class is None or dt.asset_class == pattern.asset_class
                )
                and (pattern.subtype is None or dt.subtype == pattern.subtype)
                and (pattern.region is None or dt.region == pattern.region)
                and (pattern.frequency is None or dt.frequency == pattern.frequency)
            ):
                print(f"[REGISTRY] ✓ Found contract for pattern {pattern}")
                return contract
        raise KeyError(f"No contract matching pattern {pattern}")
