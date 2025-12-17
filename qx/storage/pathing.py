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

    Supports separate read/write environments for isolation:
    - dev: data/dev/curated, data/dev/processed (test data)
    - prod: data/curated, data/processed (production research data)

    Environments are determined by (highest priority first):
    1. Explicit env_read/env_write parameters
    2. QX_ENV_READ/QX_ENV_WRITE environment variables
    3. storage.env_read/env_write in config/config.yaml
    4. Backward compat: QX_MODE env var or storage.env (applies to both read/write)

    Usage:
        # Auto-detect from env/config
        resolver = PathResolver()

        # Separate read/write environments (recommended for loaders/models)
        resolver = PathResolver(env_read="prod", env_write="dev")

        # Single environment (backward compatible)
        resolver = PathResolver(env="prod")

        # Convenience constructors
        resolver = PathResolver.for_dev()  # Both read and write
        resolver = PathResolver.for_prod()
    """

    env: Optional[str] = None  # Backward compat: single environment for both read/write
    env_read: Optional[str] = None  # Environment for reading curated data
    env_write: Optional[str] = None  # Environment for writing processed data
    _config_cache: Optional[Dict] = None

    @classmethod
    def for_dev(cls) -> "PathResolver":
        """Create a PathResolver explicitly set to dev environment (both read and write)."""
        return cls(env_read="dev", env_write="dev")

    @classmethod
    def for_prod(cls) -> "PathResolver":
        """Create a PathResolver explicitly set to prod environment (both read and write)."""
        return cls(env_read="prod", env_write="prod")

    @property
    def is_dev(self) -> bool:
        """Check if resolver is in dev environment (both read and write)."""
        return self.env_read == "dev" and self.env_write == "dev"

    @property
    def is_prod(self) -> bool:
        """Check if resolver is in prod environment (both read and write)."""
        return self.env_read == "prod" and self.env_write == "prod"

    @property
    def effective_env(self) -> str:
        """
        Get effective environment for compatibility.

        Returns:
            - "prod" if both read and write are prod
            - "dev" if both read and write are dev
            - "mixed" if read and write environments differ
        """
        if self.env_read == self.env_write:
            return self.env_read
        return "mixed"

    def __post_init__(self):
        """Initialize read/write environments from environment variables or config if not explicitly set."""
        config = self._load_config()
        storage_config = config.get("storage", {})

        # Handle backward compatible single env parameter
        if self.env is not None:
            if self.env_read is None:
                self.env_read = self.env
            if self.env_write is None:
                self.env_write = self.env

        # Initialize env_read
        if self.env_read is None:
            # Try QX_ENV_READ env var
            self.env_read = os.environ.get("QX_ENV_READ")

            # Try storage.env_read from config
            if self.env_read is None:
                self.env_read = storage_config.get("env_read")

            # Fall back to single env (backward compat)
            if self.env_read is None:
                self.env_read = os.environ.get("QX_MODE") or storage_config.get(
                    "env", "prod"
                )

        # Initialize env_write
        if self.env_write is None:
            # Try QX_ENV_WRITE env var
            self.env_write = os.environ.get("QX_ENV_WRITE")

            # Try storage.env_write from config
            if self.env_write is None:
                self.env_write = storage_config.get("env_write")

            # Fall back to single env (backward compat)
            if self.env_write is None:
                self.env_write = os.environ.get("QX_MODE") or storage_config.get(
                    "env", "prod"
                )

        # Validate environments
        for env_name, env_value in [
            ("env_read", self.env_read),
            ("env_write", self.env_write),
        ]:
            if env_value not in ("dev", "prod"):
                raise ValueError(
                    f"Invalid {env_name} '{env_value}'. Must be 'dev' or 'prod'."
                )

    def _load_config(self) -> Dict:
        """Load storage configuration from config/config.yaml."""
        if self._config_cache is not None:
            return self._config_cache

        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                self._config_cache = yaml.safe_load(f) or {}
        else:
            self._config_cache = {}

        return self._config_cache

    def _get_path_prefix(self, data_type: str, env: Optional[str] = None) -> str:
        """
        Get path prefix based on environment and data type.

        Args:
            data_type: Either 'curated' or 'processed'
            env: Override environment to use (if None, uses env_read for curated, env_write for processed)

        Returns:
            Path prefix (e.g., 'data/dev/curated' or 'data/curated')
        """
        # Determine which environment to use
        if env is None:
            env = self.env_read if data_type == "curated" else self.env_write

        config = self._load_config()
        paths = config.get("paths", {})

        if env == "dev":
            return paths.get("dev", {}).get(data_type, f"data/dev/{data_type}")
        else:  # prod
            return paths.get("prod", {}).get(data_type, f"data/{data_type}")

    def curated_dir(self, c: DatasetContract, partitions: Dict[str, str]) -> str:
        """
        Resolve curated data directory path with environment-aware prefix.
        Uses env_read for reading curated data.

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
            "subtype": c.dataset_type.subtype if c.dataset_type.subtype else "",
            **partitions,
        }
        base_path = c.path_template.format(**template_vars)

        # Replace 'data/curated' with environment-specific prefix (using env_read)
        if base_path.startswith("data/curated/"):
            prefix = self._get_path_prefix("curated", env=self.env_read)
            return base_path.replace("data/curated", prefix, 1)

        return base_path

    def processed_dir(
        self, output_type: str, model: str, run_date: str, c: DatasetContract
    ) -> str:
        """
        Resolve processed data directory path with environment-aware prefix.
        Uses env_write for writing processed data.

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

        # Replace 'data/processed' with environment-specific prefix (using env_write)
        if base_path.startswith("data/processed/"):
            prefix = self._get_path_prefix("processed", env=self.env_write)
            return base_path.replace("data/processed", prefix, 1)

        return base_path
