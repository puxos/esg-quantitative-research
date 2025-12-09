from typing import Any, Dict, List, Optional

import pandas as pd

from qx.common.contracts import DatasetContract, DatasetRegistry
from qx.common.types import DatasetType
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.pathing import PathResolver


class TypedCuratedLoader:
    """
    Loads curated datasets by type with optional filtering.

    Supports:
    - Column selection
    - Named filters from schema (e.g., date_range, ticker_list)
    - Ad-hoc PyArrow filters
    - Filter pushdown to Parquet reader for performance
    """

    def __init__(
        self,
        registry: DatasetRegistry,
        backend: LocalParquetBackend,
        resolver: PathResolver,
    ):
        self.registry, self.backend, self.resolver = registry, backend, resolver

    def load(
        self,
        dt: DatasetType,
        partitions: Dict[str, str],
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Load curated dataset with optional filtering.

        Args:
            dt: Dataset type to load
            partitions: Partition key-value pairs (e.g., {"universe": "sp500", "mode": "daily"})
            columns: Optional list of columns to load (None = all columns)
            filters: Optional named filters from schema
                     Format: {"filter_name": filter_value, ...}
                     Examples:
                       {"date_range": ("2020-01-01", "2024-12-31")}
                       {"ticker_list": ["AAPL", "MSFT", "GOOGL"]}
                       {"price_range": (100, 500)}

        Returns:
            DataFrame with loaded (and filtered) data

        Example:
            loader = TypedCuratedLoader(registry, backend, resolver)

            # Load with date range filter
            df = loader.load(
                dt=DatasetType(domain=Domain.MEMBERSHIP, subdomain="daily"),
                partitions={"universe": "sp500", "mode": "daily"},
                filters={"date_range": ("2020-01-01", "2024-12-31")}
            )

            # Load with multiple filters
            df = loader.load(
                dt=DatasetType(domain=Domain.MARKET_DATA, asset_class=AssetClass.EQUITY, subdomain="ohlcv"),
                partitions={"exchange": "US", "frequency": "Frequency.DAILY.value"},
                filters={
                    "date_range": ("2020-01-01", "2024-12-31"),
                    "symbol_list": ["AAPL", "MSFT"]
                }
            )
        """
        contract: DatasetContract = self.registry.find(dt)

        # Build PyArrow filters from named filters
        arrow_filters = None
        if filters:
            arrow_filters = self._build_arrow_filters(contract, filters)

        # Load data files
        rel_dir = self.resolver.curated_dir(contract, partitions)
        files = self.backend.list_files(rel_dir, suffix=".parquet")

        if not files:
            return pd.DataFrame(columns=columns or [])

        # Read files with filter pushdown
        dfs = [
            self.backend.read_parquet(f, columns=columns, filters=arrow_filters)
            for f in files
        ]
        return pd.concat(dfs, ignore_index=True)

    def _build_arrow_filters(
        self, contract: DatasetContract, filters: Dict[str, Any]
    ) -> Optional[List[tuple]]:
        """
        Convert named filters to PyArrow filter expressions.

        Args:
            contract: Dataset contract with filter definitions
            filters: Named filters dictionary

        Returns:
            List of PyArrow filter tuples

        Raises:
            ValueError: If filter name not found in schema or invalid filter value
        """
        arrow_filters = []

        for filter_name, filter_value in filters.items():
            # Look up filter definition in schema
            if filter_name not in contract.filter_definitions:
                raise ValueError(
                    f"Filter '{filter_name}' not defined in schema. "
                    f"Available filters: {list(contract.filter_definitions.keys())}"
                )

            filter_def = contract.filter_definitions[filter_name]

            # Convert based on filter type
            if filter_def.type == "range":
                # Range filter: (min, max) or [min, max]
                if (
                    not isinstance(filter_value, (tuple, list))
                    or len(filter_value) != 2
                ):
                    raise ValueError(
                        f"Filter '{filter_name}' expects range (min, max), got: {filter_value}"
                    )

                min_val, max_val = filter_value
                col = filter_def.columns[0]  # Range filters have single column

                if min_val is not None:
                    arrow_filters.append(
                        (col, ">=", self._convert_value_for_range(min_val))
                    )
                if max_val is not None:
                    arrow_filters.append(
                        (col, "<=", self._convert_value_for_range(max_val))
                    )

            elif filter_def.type == "in":
                # IN filter: list of values
                if not isinstance(filter_value, (list, tuple)):
                    filter_value = [filter_value]  # Wrap single value

                col = filter_def.columns[0]
                # For "in" filters, don't convert values - pass as-is
                arrow_filters.append((col, "in", filter_value))

            elif filter_def.type == "custom":
                # Custom filters (like date_overlap) need special handling
                # For now, skip pushdown and filter in pandas later
                # TODO: Implement custom filter logic
                pass

        return arrow_filters if arrow_filters else None

    def _convert_value_for_range(self, value: Any) -> Any:
        """
        Convert filter values for range filters to appropriate types for PyArrow.
        Only applies to range filters.

        Args:
            value: Filter value (str, int, float, date, etc.)

        Returns:
            Value in PyArrow-compatible format
        """
        # Convert string dates to datetime.date objects for PyArrow date32 columns
        if isinstance(value, str):
            try:
                # Try to parse as date (YYYY-MM-DD format)
                import datetime

                ts = pd.Timestamp(value)
                return ts.date()  # Convert to datetime.date
            except:
                # Not a date, return as-is
                return value

        return value

    def _convert_value(self, value: Any) -> Any:
        """
        Convert filter values to appropriate types for PyArrow.
        Deprecated - use _convert_value_for_range for range filters.

        Args:
            value: Filter value (str, int, float, date, etc.)

        Returns:
            Value in PyArrow-compatible format
        """
        # For "in" filters, pass through as-is
        return value
