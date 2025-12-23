"""
Schema Loader

Loads dataset contracts from YAML schema definitions.
Unified format supporting builders (output.dataset) and models (output.type).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from qx.common.contracts import DatasetContract
from qx.common.types import (
    AssetClass,
    DatasetType,
    Domain,
    Frequency,
    Region,
    Subdomain,
)


@dataclass
class ColumnDefinition:
    """Column definition with metadata."""

    name: str
    type: str
    description: str = ""
    required: bool = True
    filterable: bool = False
    filter_type: Optional[str] = None


@dataclass
class FilterDefinition:
    """Predefined filter for common query patterns."""

    name: str
    columns: List[str]
    type: str
    description: str = ""
    example: Optional[Any] = None


class SchemaLoader:
    """
    Loads DatasetContract objects from YAML schema definitions.

    Unified format for builders and models:

    Builders use output.dataset:
        output:
          dataset:
            domain: market-data
            subdomain: bars
            frequency: null
          schema:
            schema_version: schema_v1
            columns: [...]
          partition_keys: [...]
          path_template: "..."

    Models use output.type:
        output:
          type:
            domain: derived-metrics
            subdomain: factor-returns
          schema:
            schema_version: schema_v1
            columns: [...]
          partition_keys: [...]
          path_template: "..."

    Multi-mode builders use output.schemas:
        output:
          dataset: {...}
          schemas:
            daily: {...}
            intervals: {...}
          partition_keys: [...]
    """

    # Domain mapping
    DOMAIN_MAP = {
        "market-data": Domain.MARKET_DATA,
        "reference-rates": Domain.REFERENCE_RATES,
        "fundamentals": Domain.FUNDAMENTALS,
        "corporate-actions": Domain.CORPORATE_ACTIONS,
        "esg": Domain.ESG,
        "derived-metrics": Domain.DERIVED_METRICS,
        "portfolio": Domain.PORTFOLIO,
        "instrument-reference": Domain.INSTRUMENT_REFERENCE,
        "metadata": Domain.METADATA,
        "data-products": Domain.DATA_PRODUCTS,
    }

    # Asset class mapping
    ASSET_CLASS_MAP = {
        "equity": AssetClass.EQUITY,
        "fixed-income": AssetClass.FIXED_INCOME,
        "fx": AssetClass.FX,
        "commodity": AssetClass.COMMODITY,
        "derivative": AssetClass.DERIVATIVE,
        "crypto": AssetClass.CRYPTO,
        "multi-asset": AssetClass.MULTI_ASSET,
        "null": None,
        None: None,
    }

    # Frequency mapping
    FREQUENCY_MAP = {
        "daily": Frequency.DAILY,
        "weekly": Frequency.WEEKLY,
        "monthly": Frequency.MONTHLY,
        "quarterly": Frequency.QUARTERLY,
        "yearly": Frequency.YEARLY,
        "null": None,
        None: None,
    }

    # Region mapping
    REGION_MAP = {
        "US": Region.US,
        "HK": Region.HK,
        "GLOBAL": Region.GLOBAL,
        "null": None,
        None: None,
    }

    # Subdomain mapping (dynamic from enum)
    SUBDOMAIN_MAP = {member.value: member for member in Subdomain}
    SUBDOMAIN_MAP["null"] = None
    SUBDOMAIN_MAP[None] = None

    def __init__(self):
        pass

    def load_contract(self, schema_path: str | Path, **params) -> DatasetContract:
        """
        Load a dataset contract from YAML schema file.

        Args:
            schema_path: Path to YAML schema file
            **params: Parameters to substitute (e.g., exchange=Exchange.US, frequency=Frequency.DAILY)

        Returns:
            DatasetContract instance

        Raises:
            FileNotFoundError: If schema file doesn't exist
            ValueError: If schema is invalid or required parameters missing
        """
        schema_path = Path(schema_path)

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)

        return self._build_contract(schema, params)

    def _build_contract(
        self, schema: Dict[str, Any], params: Dict[str, Any]
    ) -> DatasetContract:
        """
        Build DatasetContract from YAML schema.

        Unified format with output.dataset/type + output.schema/schemas.

        Args:
            schema: Parsed YAML dictionary
            params: Runtime parameters (e.g., mode for multi-schema builders)

        Returns:
            DatasetContract instance
        """
        # Support both "dataset" (builders) and "type" (models)
        if "dataset" in schema["output"]:
            dataset_section = schema["output"]["dataset"]
        elif "type" in schema["output"]:
            dataset_section = schema["output"]["type"]
        else:
            raise ValueError(
                "output section must have 'dataset' (builders) or 'type' (models)"
            )

        # Support both "schema" (single) and "schemas" (multi-mode)
        if "schema" in schema["output"]:
            contract_section = schema["output"]["schema"]
        elif "schemas" in schema["output"]:
            mode = params.get("mode")
            if not mode:
                raise ValueError("'mode' parameter required for multi-schema output")
            if mode not in schema["output"]["schemas"]:
                available = list(schema["output"]["schemas"].keys())
                raise ValueError(f"Unknown mode '{mode}'. Available: {available}")
            contract_section = schema["output"]["schemas"][mode]
        else:
            raise ValueError("output section must have 'schema' or 'schemas'")

        # Parse dataset type
        domain = self._parse_domain(dataset_section.get("domain"))
        asset_class = self._parse_asset_class(dataset_section.get("asset_class"))
        subdomain = self._parse_subdomain(dataset_section.get("subdomain"))
        subtype = dataset_section.get("subtype")
        region = self._parse_region(dataset_section.get("region"))
        frequency = self._parse_frequency(dataset_section.get("frequency"))

        # Override with runtime parameters (for parameterized contracts like tiingo_ohlcv)
        # Only override if parameter is provided and YAML value is None
        if "frequency" in params and frequency is None:
            frequency = params["frequency"]
        if "region" in params and region is None:
            region = params["region"]
        if "asset_class" in params and asset_class is None:
            asset_class = params["asset_class"]

        dataset_type = DatasetType(
            domain=domain,
            asset_class=asset_class,
            subdomain=subdomain,
            subtype=subtype,
            region=region,
            frequency=frequency,
        )

        # Parse schema
        schema_version = contract_section.get("schema_version", "schema_v1")
        columns = contract_section.get("columns", [])

        # Extract column definitions
        column_defs = []
        required_columns = []

        for col in columns:
            if isinstance(col, dict):
                column_defs.append(
                    ColumnDefinition(
                        name=col["name"],
                        type=col.get("type", "string"),
                        description=col.get("description", ""),
                        required=col.get("required", True),
                        filterable=col.get("filterable", False),
                        filter_type=col.get("filter_type"),
                    )
                )
                if col.get("required", True):
                    required_columns.append(col["name"])
            else:
                # Simple string format
                column_defs.append(
                    ColumnDefinition(name=col, type="string", required=True)
                )
                required_columns.append(col)

        # Parse filters
        filter_defs = {}
        if "filters" in contract_section:
            for name, spec in contract_section["filters"].items():
                filter_defs[name] = FilterDefinition(
                    name=name,
                    columns=spec.get("columns", []),
                    type=spec.get("type", "range"),
                    description=spec.get("description", ""),
                    example=spec.get("example"),
                )

        # Get partition keys and path template
        # Builders: At top-level output
        # Models: Inside contract section (schema)
        # Try contract section first (models), fall back to top-level (builders)
        partition_keys = tuple(
            contract_section.get("partition_keys")
            or schema["output"].get("partition_keys", [])
        )
        path_template = contract_section.get("path_template") or schema["output"].get(
            "path_template", ""
        )

        # Create contract
        contract = DatasetContract(
            dataset_type=dataset_type,
            schema_version=schema_version,
            required_columns=tuple(required_columns),
            partition_keys=partition_keys,
            path_template=path_template,
        )

        # Attach metadata
        contract.column_definitions = column_defs
        contract.filter_definitions = filter_defs

        return contract

    def _parse_domain(self, value: Optional[str]) -> Domain:
        """Parse domain from string."""
        if not value:
            raise ValueError("Domain is required")
        if value not in self.DOMAIN_MAP:
            raise ValueError(f"Invalid domain: {value}")
        return self.DOMAIN_MAP[value]

    def _parse_asset_class(self, value: Optional[str]) -> Optional[AssetClass]:
        """Parse asset class from string."""
        return self.ASSET_CLASS_MAP.get(value)

    def _parse_subdomain(self, value: Optional[str]) -> Subdomain:
        """Parse subdomain from string."""
        if not value:
            raise ValueError("Subdomain is required")
        if value not in self.SUBDOMAIN_MAP:
            valid = [k for k in self.SUBDOMAIN_MAP.keys() if k]
            raise ValueError(f"Invalid subdomain: '{value}'. Valid: {valid[:10]}...")
        return self.SUBDOMAIN_MAP[value]

    def _parse_region(self, value: Optional[str]) -> Optional[Region]:
        """Parse region from string."""
        return self.REGION_MAP.get(value)

    def _parse_frequency(self, value: Optional[str]) -> Optional[Frequency]:
        """Parse frequency from string."""
        return self.FREQUENCY_MAP.get(value)

    def load_schema_metadata(self, schema_path: str | Path) -> Dict[str, Any]:
        """
        Load only the metadata section from schema.

        Useful for documentation generation, dependency analysis, etc.

        Args:
            schema_path: Path to YAML schema file

        Returns:
            Metadata dictionary with keys: source, description, dependencies, notes
        """
        schema_path = Path(schema_path)

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)

        return schema.get("metadata", {})


# Global loader instance
_loader = SchemaLoader()


def load_contract(schema_path: str | Path, **params) -> DatasetContract:
    """
    Convenience function to load a contract from YAML.

    Args:
        schema_path: Path to YAML schema file
        **params: Parameters to substitute

    Returns:
        DatasetContract instance

    Example:
        contract = load_contract('schemas/tiingo_ohlcv.yaml', exchange=Exchange.US, frequency=Frequency.DAILY)
    """
    return _loader.load_contract(schema_path, **params)


def load_schema_metadata(schema_path: str | Path) -> Dict[str, Any]:
    """
    Convenience function to load schema metadata.

    Args:
        schema_path: Path to YAML schema file

    Returns:
        Metadata dictionary
    """
    return _loader.load_schema_metadata(schema_path)
