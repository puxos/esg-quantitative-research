"""
Base Loader Class

Loaders read curated datasets and transform them into lightweight outputs
(lists, dicts, DataFrames) that can be consumed by downstream tasks.

Unlike Builders and Models, Loaders:
- Do NOT persist their outputs (memory only)
- Only read curated data (never write)
- Produce simple Python objects, not typed datasets
- Are used only in pipelines (no standalone execution)

Loaders bridge the gap between "curated data exists" and "use it as parameters".
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from qx.common.contracts import DatasetRegistry
from qx.foundation.typed_curated_loader import TypedCuratedLoader
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.pathing import PathResolver


class BaseLoader:
    """
    Base class for all loader packages.

    Loaders are lightweight data transformers that:
    1. Read curated datasets via direct file access
    2. Apply filters, selections, aggregations
    3. Return Python objects (List, Dict, DataFrame)
    4. Do NOT persist outputs (memory only)

    Each loader package contains:
    - loader.yaml: Configuration (parameters, input contracts)
    - loader.py: Implementation (extends BaseLoader)

    Example loader package structure:
        qx_loaders/continuous_universe/
        ├── loader.yaml
        └── loader.py

    Example loader.yaml:
        loader:
          id: continuous_universe
          version: 1.0.0
          description: "Select symbols continuously in universe during period"

        inputs:
          - name: membership_data
            type:
              domain: membership
              subdomain: sp500_membership_intervals

        parameters:
          start_date:
            type: str
            required: true
            description: "Period start date (YYYY-MM-DD)"
          end_date:
            type: str
            required: true
            description: "Period end date (YYYY-MM-DD)"
          universe:
            type: str
            default: "sp500"
            description: "Universe identifier"

        output:
          type: list
          description: "List of ticker symbols"
    """

    def __init__(
        self,
        package_dir: str,
        loader: TypedCuratedLoader,
        overrides: Optional[Dict] = None,
    ):
        """
        Initialize loader from package directory.

        Args:
            package_dir: Path to loader package (e.g., "qx_loaders/continuous_universe")
            loader: High-level typed curated data loader (contains registry, backend, resolver)
            overrides: Parameter overrides (e.g., {"start_date": "2014-01-01"})
        """
        self.package_dir = Path(package_dir)
        self.loader = loader
        self.registry = loader.registry
        self.backend = loader.backend

        # Apply mode override if specified in loader.yaml
        # This allows packages to force a specific mode (e.g., reference data always uses prod)
        yaml_path = self.package_dir / "loader.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"loader.yaml not found in {package_dir}")

        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.info = self.config.get("loader", {})
        package_mode = self.info.get("mode")
        resolver_source = loader.resolver
        if (
            package_mode
            and hasattr(resolver_source, "mode")
            and resolver_source.mode != package_mode
        ):
            # Create new resolver with overridden mode
            from qx.storage.pathing import PathResolver

            self.resolver = PathResolver(mode=package_mode)
        else:
            self.resolver = resolver_source

        # Load loader.yaml configuration
        self.loader_id = self.info.get("id", "unknown")
        self.version = self.info.get("version", "unknown")
        self.description = self.info.get("description", "")

        # Extract parameter definitions
        self.param_defs = self.config.get("parameters", {})

        # Merge default parameters with overrides
        self.params = {}
        for param_name, param_def in self.param_defs.items():
            # Use override if provided, else default, else None
            if overrides and param_name in overrides:
                self.params[param_name] = overrides[param_name]
            elif "default" in param_def:
                self.params[param_name] = param_def["default"]
            elif param_def.get("required", False):
                raise ValueError(
                    f"Required parameter '{param_name}' not provided for loader '{self.loader_id}'"
                )
            else:
                self.params[param_name] = None

        # Validate parameters (optional, can be overridden)
        self._validate_parameters()

    def _validate_parameters(self):
        """
        Validate parameters against definitions.

        Override this method for custom validation logic.
        Default implementation checks types and required fields.
        """
        for param_name, param_def in self.param_defs.items():
            value = self.params.get(param_name)

            # Check required
            if param_def.get("required", False) and value is None:
                raise ValueError(
                    f"Required parameter '{param_name}' is None for loader '{self.loader_id}'"
                )

            # Type checking (basic)
            if value is not None and "type" in param_def:
                expected_type = param_def["type"]
                if expected_type == "str" and not isinstance(value, str):
                    raise TypeError(
                        f"Parameter '{param_name}' expected str, got {type(value).__name__}"
                    )
                elif expected_type == "int" and not isinstance(value, int):
                    raise TypeError(
                        f"Parameter '{param_name}' expected int, got {type(value).__name__}"
                    )
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    raise TypeError(
                        f"Parameter '{param_name}' expected float, got {type(value).__name__}"
                    )
                elif expected_type == "bool" and not isinstance(value, bool):
                    raise TypeError(
                        f"Parameter '{param_name}' expected bool, got {type(value).__name__}"
                    )
                elif expected_type == "list" and not isinstance(value, list):
                    raise TypeError(
                        f"Parameter '{param_name}' expected list, got {type(value).__name__}"
                    )

    def _validate_output(self, output: Any) -> None:
        """
        Validate output against declared contract in loader.yaml.

        Raises:
            ValueError: If output doesn't match contract
            TypeError: If output type is incorrect
        """
        output_spec = self.config.get("output", {})
        if not output_spec:
            return  # No validation if no output spec

        expected_type = output_spec.get("type")
        validation_rules = output_spec.get("validation", {})

        # Type validation
        if expected_type == "list":
            if not isinstance(output, list):
                raise TypeError(
                    f"Loader '{self.loader_id}' output contract expects list, "
                    f"got {type(output).__name__}"
                )

            # List-specific validations
            if not validation_rules.get("allow_empty", True) and len(output) == 0:
                raise ValueError(
                    f"Loader '{self.loader_id}' returned empty list, "
                    f"but allow_empty=false in contract"
                )

            if (
                "min_length" in validation_rules
                and len(output) < validation_rules["min_length"]
            ):
                raise ValueError(
                    f"Loader '{self.loader_id}' returned {len(output)} items, "
                    f"but contract requires min_length={validation_rules['min_length']}"
                )

            if (
                "max_length" in validation_rules
                and len(output) > validation_rules["max_length"]
            ):
                raise ValueError(
                    f"Loader '{self.loader_id}' returned {len(output)} items, "
                    f"exceeds max_length={validation_rules['max_length']}"
                )

            # Item type validation
            item_type = output_spec.get("item_type")
            if item_type and output:
                type_map = {"str": str, "int": int, "float": float, "bool": bool}
                expected_item_type = type_map.get(item_type)
                if expected_item_type:
                    for i, item in enumerate(output[:10]):  # Check first 10 items
                        if not isinstance(item, expected_item_type):
                            raise TypeError(
                                f"Loader '{self.loader_id}' list item at index {i} "
                                f"expected {item_type}, got {type(item).__name__}"
                            )

            # Item pattern validation (regex)
            if "item_pattern" in validation_rules and output:
                import re

                pattern = re.compile(validation_rules["item_pattern"])
                for i, item in enumerate(output[:100]):  # Check first 100 items
                    if isinstance(item, str) and not pattern.match(item):
                        raise ValueError(
                            f"Loader '{self.loader_id}' list item at index {i} '{item}' "
                            f"doesn't match pattern: {validation_rules['item_pattern']}"
                        )

        elif expected_type == "dataframe":
            import pandas as pd

            if not isinstance(output, pd.DataFrame):
                raise TypeError(
                    f"Loader '{self.loader_id}' output contract expects DataFrame, "
                    f"got {type(output).__name__}"
                )

            # DataFrame-specific validations
            if not validation_rules.get("allow_empty", True) and len(output) == 0:
                raise ValueError(
                    f"Loader '{self.loader_id}' returned empty DataFrame, "
                    f"but allow_empty=false in contract"
                )

            # Required columns validation
            if "required_columns" in validation_rules:
                missing = set(validation_rules["required_columns"]) - set(
                    output.columns
                )
                if missing:
                    raise ValueError(
                        f"Loader '{self.loader_id}' DataFrame missing required columns: {missing}"
                    )

            # Row count validations
            if (
                "min_rows" in validation_rules
                and len(output) < validation_rules["min_rows"]
            ):
                raise ValueError(
                    f"Loader '{self.loader_id}' returned {len(output)} rows, "
                    f"but contract requires min_rows={validation_rules['min_rows']}"
                )

            if (
                "max_rows" in validation_rules
                and len(output) > validation_rules["max_rows"]
            ):
                raise ValueError(
                    f"Loader '{self.loader_id}' returned {len(output)} rows, "
                    f"exceeds max_rows={validation_rules['max_rows']}"
                )

        elif expected_type == "dict":
            if not isinstance(output, dict):
                raise TypeError(
                    f"Loader '{self.loader_id}' output contract expects dict, "
                    f"got {type(output).__name__}"
                )

            # Dict-specific validations
            if "required_keys" in validation_rules:
                missing = set(validation_rules["required_keys"]) - set(output.keys())
                if missing:
                    raise ValueError(
                        f"Loader '{self.loader_id}' dict missing required keys: {missing}"
                    )

        elif expected_type == "series":
            import pandas as pd

            if not isinstance(output, pd.Series):
                raise TypeError(
                    f"Loader '{self.loader_id}' output contract expects Series, "
                    f"got {type(output).__name__}"
                )

    def load(self, available_types: Optional[List] = None) -> Any:
        """
        Execute loader logic and return output.

        This is the main entry point that calls load_impl() and validates output.

        Args:
            available_types: Optional list of available DatasetTypes for loaders
                with curated data inputs (auto-injected by DAG)

        Returns:
            Python object (List, Dict, DataFrame, etc.)

        Raises:
            NotImplementedError: Must be implemented by subclass
            ValueError: If output doesn't match declared contract
            TypeError: If output type is incorrect
        """
        # Store available_types for use in load_impl if needed
        self.available_types = available_types

        # Execute loader implementation
        output = self.load_impl()

        # Validate output against contract
        self._validate_output(output)

        return output

    def load_impl(self) -> Any:
        """
        Implement loader logic here.

        This method should:
        1. Read curated data using self.curated_loader (type-safe, contract-based)
        2. Apply filters/transformations using self.params
        3. Return lightweight output (List, Dict, DataFrame)

        IMPORTANT: Use self.curated_loader.load() instead of direct file path access.
        This ensures type safety and consistency with the contract system.

        Returns:
            Python object (List, Dict, DataFrame, etc.)

        Example:
            def load_impl(self):
                # ✅ GOOD: Type-safe loading via contract
                df = self.curated_loader.load(
                    dataset_type=DatasetType(
                        domain=Domain.INSTRUMENT_REFERENCE,
                        subdomain=Subdomain.INDEX_CONSTITUENTS
                    ),
                    partitions={"universe": self.params["universe"], "mode": "intervals"}
                )

                # ❌ BAD: Direct file path access (avoid this)
                # base_path = Path("data/curated/membership/intervals/schema_v1")
                # partition_path = base_path / f"universe={universe}"
                # files = list(partition_path.glob("*.parquet"))

                # Filter continuous members
                start_date = pd.Timestamp(self.params["start_date"])
                end_date = pd.Timestamp(self.params["end_date"])

                continuous = df[
                    (df['start_date'] <= start_date) &
                    (df['end_date'] >= end_date)
                ]

                # Return list of symbols
                return continuous['symbol'].unique().tolist()
        """
        raise NotImplementedError(
            f"Loader '{self.loader_id}' must implement load_impl() method"
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.loader_id} version={self.version}>"
