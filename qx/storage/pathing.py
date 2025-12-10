import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

from qx.common.contracts import DatasetContract


@dataclass
class PathResolver:
    """
    Resolves storage paths for curated and processed data.

    Supports dev/prod modes for environment isolation:
    - dev: data/dev/curated, data/dev/processed (test data)
    - prod: data/curated, data/processed (production research data)

    Mode is determined by:
    1. Explicit mode parameter (highest priority)
    2. QX_MODE environment variable
    3. storage.mode in config/storage.yaml (default)

    Usage:
        # Auto-detect mode from env/config
        resolver = PathResolver()

        # Explicit mode
        resolver = PathResolver(mode="prod")

        # Convenience constructors
        resolver = PathResolver.for_dev()
        resolver = PathResolver.for_prod()
    """

    mode: Optional[str] = None
    _config_cache: Optional[Dict] = None

    @classmethod
    def for_dev(cls) -> "PathResolver":
        """Create a PathResolver explicitly set to dev mode."""
        return cls(mode="dev")

    @classmethod
    def for_prod(cls) -> "PathResolver":
        """Create a PathResolver explicitly set to prod mode."""
        return cls(mode="prod")

    @property
    def is_dev(self) -> bool:
        """Check if resolver is in dev mode."""
        return self.mode == "dev"

    @property
    def is_prod(self) -> bool:
        """Check if resolver is in prod mode."""
        return self.mode == "prod"

    def __post_init__(self):
        """Initialize mode from environment or config if not explicitly set."""
        if self.mode is None:
            # Try environment variable first
            self.mode = os.environ.get("QX_MODE")

            # Fall back to config file
            if self.mode is None:
                config = self._load_config()
                self.mode = config.get("storage", {}).get("mode", "dev")

        # Validate mode
        if self.mode not in ("dev", "prod"):
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'dev' or 'prod'.")

    def _load_config(self) -> Dict:
        """Load storage configuration from config/storage.yaml."""
        if self._config_cache is not None:
            return self._config_cache

        config_path = Path("config/storage.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                self._config_cache = yaml.safe_load(f) or {}
        else:
            self._config_cache = {}

        return self._config_cache

    def _get_path_prefix(self, data_type: str) -> str:
        """
        Get path prefix based on mode and data type.

        Args:
            data_type: Either 'curated' or 'processed'

        Returns:
            Path prefix (e.g., 'data/dev/curated' or 'data/curated')
        """
        config = self._load_config()
        paths = config.get("paths", {})

        if self.mode == "dev":
            return paths.get("dev", {}).get(data_type, f"data/dev/{data_type}")
        else:  # prod
            return paths.get("prod", {}).get(data_type, f"data/{data_type}")

    def curated_dir(self, c: DatasetContract, partitions: Dict[str, str]) -> str:
        """
        Resolve curated data directory path with mode-aware prefix.

        Args:
            c: Dataset contract with path template
            partitions: Partition values to format into path

        Returns:
            Full path with mode prefix (e.g., 'data/dev/curated/...' or 'data/curated/...')
        """
        # Get the base path from contract (e.g., 'data/curated/market-data/ohlcv/...')
        # Path template may need domain, subdomain from dataset_type
        template_vars = {
            "schema_version": c.schema_version,
            "domain": c.dataset_type.domain.value if c.dataset_type.domain else "",
            "subdomain": (
                c.dataset_type.subdomain.value if c.dataset_type.subdomain else ""
            ),
            **partitions,
        }
        base_path = c.path_template.format(**template_vars)

        # Replace 'data/curated' with mode-specific prefix
        if base_path.startswith("data/curated/"):
            prefix = self._get_path_prefix("curated")
            return base_path.replace("data/curated", prefix, 1)

        return base_path

    def processed_dir(
        self, output_type: str, model: str, run_date: str, c: DatasetContract
    ) -> str:
        """
        Resolve processed data directory path with mode-aware prefix.

        Args:
            output_type: Type of processed output
            model: Model identifier
            run_date: Run date string
            c: Dataset contract with path template

        Returns:
            Full path with mode prefix (e.g., 'data/dev/processed/...' or 'data/processed/...')
        """
        # Get the base path from contract
        base_path = c.path_template.format(
            output_type=output_type, model=model, run_date=run_date
        )

        # Replace 'data/processed' with mode-specific prefix
        if base_path.startswith("data/processed/"):
            prefix = self._get_path_prefix("processed")
            return base_path.replace("data/processed", prefix, 1)

        return base_path
