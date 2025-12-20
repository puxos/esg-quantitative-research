"""
Schema Loader

Loads dataset contracts from YAML schema definitions.
Provides type-safe contract generation from declarative schemas.
Supports unified builder.yaml format with output.dataset structure.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from qx.common.contracts import DatasetContract
from qx.common.enum_validator import validate_dataset_type_config
from qx.common.types import (
    AssetClass,
    DatasetType,
    Domain,
    Exchange,
    Frequency,
    Region,
    Subdomain,
    dataset_type_from_config,
)


def load_contract_from_builder_yaml(yaml_path: Path) -> DatasetContract:
    """
    Load dataset contract from unified builder.yaml structure.

    New unified format with output.dataset, output.schema, and output.partition_keys.

    Args:
        yaml_path: Path to builder.yaml file

    Returns:
        DatasetContract instance

    Example builder.yaml structure:
        output:
          dataset:
            domain: reference-rates
            subdomain: yield-curves
            frequency: null
          schema:
            version: schema_v1
            required_columns:
              - name: date
                dtype: datetime64[ns]
          partition_keys:
            - rate_type
            - frequency
          path_template: "..."
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    output_cfg = cfg.get("output", {})

    # Load dataset type from output.dataset
    dataset_cfg = output_cfg.get("dataset", {})
    validate_dataset_type_config(dataset_cfg)
    dataset_type = dataset_type_from_config(dataset_cfg)

    # Load schema from output.schema
    schema_cfg = output_cfg.get("schema", {})
    schema_version = schema_cfg.get("version", "schema_v1")

    required_columns = []
    for col in schema_cfg.get("required_columns", []):
        required_columns.append(
            {
                "name": col["name"],
                "dtype": col["dtype"],
            }
        )

    # Load partition keys
    partition_keys = output_cfg.get("partition_keys", [])

    # Load path template (optional)
    path_template = output_cfg.get("path_template")
    if not path_template:
        # Auto-generate default template
        path_template = (
            "data/curated/{domain}/{subdomain}/{schema_version}/"
            + "/".join(f"{key}={{{key}}}" for key in partition_keys)
        )

    return DatasetContract(
        dataset_type=dataset_type,
        schema_version=schema_version,
        required_columns=required_columns,
        partition_keys=partition_keys,
        path_template=path_template,
    )


def load_contracts_from_builder_yaml(yaml_path: Path) -> list[DatasetContract]:
    """
    Load multiple dataset contracts from unified builder.yaml with multiple schemas.

    For builders that produce different schemas based on mode (e.g., daily vs intervals),
    the builder.yaml can define multiple schemas under output.schemas.

    Args:
        yaml_path: Path to builder.yaml file

    Returns:
        List of DatasetContract instances (one per schema)

    Example builder.yaml structure:
        output:
          dataset:
            domain: instrument-reference
            subdomain: index-constituents
          schemas:
            daily:
              version: schema_v1
              required_columns:
                - name: date
                  dtype: datetime64[ns]
                - name: ticker
                  dtype: object
            intervals:
              version: schema_v1
              required_columns:
                - name: ticker
                  dtype: object
                - name: start_date
                  dtype: datetime64[ns]
                - name: end_date
                  dtype: datetime64[ns]
          partition_keys:
            - universe
            - mode
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    output_cfg = cfg.get("output", {})

    # Load dataset type from output.dataset
    dataset_cfg = output_cfg.get("dataset", {})
    validate_dataset_type_config(dataset_cfg)
    base_dataset_type = dataset_type_from_config(dataset_cfg)

    # Load partition keys (shared)
    partition_keys = output_cfg.get("partition_keys", [])

    # Load path template (shared, optional)
    path_template = output_cfg.get("path_template")
    if not path_template:
        # Auto-generate default template
        path_template = (
            "data/curated/{domain}/{subdomain}/{schema_version}/"
            + "/".join(f"{key}={{{key}}}" for key in partition_keys)
        )

    # Load multiple schemas
    schemas_cfg = output_cfg.get("schemas", {})
    contracts = []

    for schema_name, schema_cfg in schemas_cfg.items():
        schema_version = schema_cfg.get("version", "schema_v1")

        required_columns = []
        for col in schema_cfg.get("required_columns", []):
            required_columns.append(
                {
                    "name": col["name"],
                    "dtype": col["dtype"],
                }
            )

        # Create contract for this schema
        contract = DatasetContract(
            dataset_type=base_dataset_type,
            schema_version=schema_version,
            required_columns=required_columns,
            partition_keys=partition_keys,
            path_template=path_template,
        )
        contracts.append(contract)

    return contracts


@dataclass
class ColumnDefinition:
    """Column definition with filter capabilities."""

    name: str
    type: str
    description: str = ""
    required: bool = True
    filterable: bool = False
    filter_type: Optional[str] = None  # range | in | exact | custom


@dataclass
class FilterDefinition:
    """Named filter definition for common query patterns."""

    name: str
    columns: List[str]
    type: str  # range | in | exact | custom
    description: str = ""
    example: Optional[Any] = None


class SchemaLoader:
    """
    Loads DatasetContract objects from YAML schema definitions.

    YAML schema format:
        dataset:
          domain: market-data | reference-rates | fundamentals | corporate-actions | esg | derived-metrics | portfolio | instrument-reference | metadata | membership | data-products
          asset_class: equity | bond | commodity | null
          subdomain: string (e.g., "bars", "benchmark-rates", "esg-scores")
          subtype: string (optional, custom subtype, not enum-restricted)

        contract:
          schema_version: string (e.g., "schema_v1")

          columns:
            - name: column_name
              type: date | string | int | float | bool
              description: human-readable description
              required: true | false (default: true)

          partition_keys:
            - partition_key_name

          path_template: string with {placeholders}

        parameters:  # Optional: for parameterized contracts
          param_name:
            type: enum | string | exchange | frequency
            values: [list, of, allowed, values]
            required: true | false
            default: value

        metadata:  # Optional
          source: string
          description: string
          dependencies: [list, of, builder, names]
          notes: string

    Example:
        loader = SchemaLoader()
        contract = loader.load_contract('path/to/schema.yaml', exchange=Exchange.US, frequency=Frequency.DAILY)
    """

    # Type mappings
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

    FREQUENCY_MAP = {
        "daily": Frequency.DAILY,
        "weekly": Frequency.WEEKLY,
        "monthly": Frequency.MONTHLY,
        "quarterly": Frequency.QUARTERLY,
        "yearly": Frequency.YEARLY,
        "null": None,
        None: None,
    }

    REGION_MAP = {
        "US": Region.US,
        "HK": Region.HK,
        "GLOBAL": Region.GLOBAL,
        "null": None,
        None: None,
    }

    EXCHANGE_MAP = {
        "NYSE": Exchange.NYSE,
        "NASDAQ": Exchange.NASDAQ,
        "AMEX": Exchange.AMEX,
        "HKEX": Exchange.HKEX,
        "US": Exchange.US,
        "HK": Exchange.HK,
        "null": None,
        None: None,
    }

    # Build SUBDOMAIN_MAP dynamically from Subdomain enum
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
        Build DatasetContract from parsed YAML schema.

        Expects unified builder.yaml format with output.dataset and output.schema sections.

        Args:
            schema: Parsed YAML schema dictionary
            params: Parameter values

        Returns:
            DatasetContract instance
        """
        # Validate required parameters
        if "parameters" in schema:
            self._validate_parameters(schema["parameters"], params)

        # Load from unified format: output.dataset + output.schema
        dataset_section = schema["output"]["dataset"]
        contract_section = schema["output"]["schema"]

        # Parse dataset type
        domain = self._parse_domain(dataset_section.get("domain"))
        asset_class = self._parse_asset_class(dataset_section.get("asset_class"))
        subdomain = self._parse_subdomain(dataset_section.get("subdomain"))
        subtype = dataset_section.get("subtype")  # Custom string, no enum validation

        # Handle parameterized region/frequency (for contract-level identity)
        region = self._parse_region(dataset_section.get("region"), params.get("region"))
        frequency = self._parse_frequency(
            dataset_section.get("frequency"), params.get("frequency")
        )

        dataset_type = DatasetType(
            domain=domain,
            asset_class=asset_class,
            subdomain=subdomain,
            subtype=subtype,
            region=region,
            frequency=frequency,
        )

        # Parse schema section
        schema_version = contract_section.get("schema_version", "schema_v1")

        # Parse columns (extract names and metadata)
        columns = contract_section.get("columns", [])
        column_defs = []
        if columns and isinstance(columns[0], dict):
            # Detailed format: [{name: ..., type: ..., description: ..., filterable: ...}]
            required_columns = tuple(
                col["name"]
                for col in columns
                if col.get("required", True)  # Default to required
            )
            # Parse full column definitions
            for col in columns:
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
        else:
            # Simple format: [name1, name2, ...]
            required_columns = tuple(columns)
            for col_name in columns:
                column_defs.append(
                    ColumnDefinition(name=col_name, type="string", required=True)
                )

        # Parse filters section
        filter_defs = {}
        if "filters" in contract_section:
            for filter_name, filter_spec in contract_section["filters"].items():
                filter_defs[filter_name] = FilterDefinition(
                    name=filter_name,
                    columns=filter_spec.get("columns", []),
                    type=filter_spec.get("type", "range"),
                    description=filter_spec.get("description", ""),
                    example=filter_spec.get("example"),
                )

        # Get partition_keys and path_template from output level
        partition_keys = tuple(schema["output"].get("partition_keys", []))
        path_template = schema["output"].get("path_template", "")

        # Substitute parameters in path_template if needed
        path_template = self._substitute_template_params(path_template, params)

        contract = DatasetContract(
            dataset_type=dataset_type,
            schema_version=schema_version,
            required_columns=required_columns,
            partition_keys=partition_keys,
            path_template=path_template,
        )

        # Attach metadata to contract
        contract.column_definitions = column_defs
        contract.filter_definitions = filter_defs

        return contract

    def _validate_parameters(
        self, param_defs: Dict[str, Any], provided_params: Dict[str, Any]
    ):
        """Validate that required parameters are provided."""
        for param_name, param_def in param_defs.items():
            if param_def.get("required", False):
                if param_name not in provided_params:
                    raise ValueError(
                        f"Required parameter '{param_name}' not provided. "
                        f"Expected type: {param_def.get('type')}"
                    )

            # Validate enum values
            if param_name in provided_params:
                param_value = provided_params[param_name]
                if param_def.get("type") == "enum":
                    allowed = param_def.get("values", [])
                    # Handle enum types (Region, Frequency, etc.)
                    if hasattr(param_value, "value"):
                        param_value = param_value.value
                    if param_value not in allowed:
                        raise ValueError(
                            f"Invalid value for parameter '{param_name}': {param_value}. "
                            f"Allowed values: {allowed}"
                        )

    def _parse_domain(self, value: Optional[str]) -> Domain:
        """Parse domain from string."""
        if value is None:
            raise ValueError("Domain is required")
        if value not in self.DOMAIN_MAP:
            raise ValueError(f"Invalid domain: {value}")
        return self.DOMAIN_MAP[value]

    def _parse_asset_class(self, value: Optional[str]) -> Optional[AssetClass]:
        """Parse asset class from string."""
        if value in self.ASSET_CLASS_MAP:
            return self.ASSET_CLASS_MAP[value]
        return None

    def _parse_subdomain(self, value: Optional[str]) -> Optional[Subdomain]:
        """Parse subdomain from string."""
        if value is None:
            raise ValueError("Subdomain is required")
        if value not in self.SUBDOMAIN_MAP:
            raise ValueError(
                f"Invalid subdomain: '{value}'. "
                f"Valid values: {list(self.SUBDOMAIN_MAP.keys())}"
            )
        return self.SUBDOMAIN_MAP[value]

    def _parse_region(
        self, schema_value: Optional[str], param_value: Optional[Any]
    ) -> Optional[Region]:
        """Parse region from schema or parameter (contract-level)."""
        # Parameter takes precedence
        if param_value is not None:
            if isinstance(param_value, Region):
                return param_value
            if param_value in self.REGION_MAP:
                return self.REGION_MAP[param_value]

        # Fallback to schema value
        if schema_value in self.REGION_MAP:
            return self.REGION_MAP[schema_value]

        return None

    def _parse_exchange(
        self, schema_value: Optional[str], param_value: Optional[Any]
    ) -> Optional[Exchange]:
        """Parse exchange from schema or parameter (partition-level)."""
        # Parameter takes precedence
        if param_value is not None:
            if isinstance(param_value, Exchange):
                return param_value
            if param_value in self.EXCHANGE_MAP:
                return self.EXCHANGE_MAP[param_value]

        # Fallback to schema value
        if schema_value in self.EXCHANGE_MAP:
            return self.EXCHANGE_MAP[schema_value]

        return None

    def _parse_frequency(
        self, schema_value: Optional[str], param_value: Optional[Any]
    ) -> Optional[Frequency]:
        """Parse frequency from schema or parameter."""
        # Parameter takes precedence
        if param_value is not None:
            if isinstance(param_value, Frequency):
                return param_value
            if param_value in self.FREQUENCY_MAP:
                return self.FREQUENCY_MAP[param_value]

        # Fallback to schema value
        if schema_value in self.FREQUENCY_MAP:
            return self.FREQUENCY_MAP[schema_value]

        return None

    def _substitute_template_params(self, template: str, params: Dict[str, Any]) -> str:
        """
        Substitute parameters in path template.
        Keeps standard placeholders like {schema_version}, {exchange}, etc.
        """
        # No substitution needed - path templates use runtime partition values
        return template

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
