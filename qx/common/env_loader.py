"""
Environment variable loader for Qx framework.

Provides automatic .env file loading for API keys and configuration.
New packages just need to add their variables to .env file and the system
loads them automatically at startup.
"""

import os
from pathlib import Path
from typing import Optional

_ENV_LOADED = False


def load_env_file(env_path: Optional[str] = None, force_reload: bool = False) -> None:
    """
    Load environment variables from .env file if it exists.

    This function is idempotent - it only loads the .env file once unless
    force_reload=True. Variables already set in the environment are not overwritten.

    The .env file should be located at the root of the project (same directory
    as pytest.ini and requirements.txt).

    Format:
    -------
    .env file should contain key=value pairs, one per line:

        # API Keys
        TIINGO_API_KEY=your_api_key_here
        FRED_API_KEY=your_fred_key_here

        # Other configuration
        LOG_LEVEL=INFO

    Usage:
    ------
    The .env file is automatically loaded when:
    1. Any builder is initialized (via base_builder.py)
    2. Any test script runs (via conftest.py or explicit call)
    3. Any DAG orchestration starts (via factories.py)

    Manual loading:
        from qx.common.env_loader import load_env_file
        load_env_file()

    Args:
        env_path: Path to .env file. If None, searches for .env in:
                  1. Current working directory
                  2. Project root (parent of qx/ directory)
        force_reload: If True, reload .env even if already loaded (default: False)

    Notes:
    ------
    - Comments (lines starting with #) are ignored
    - Empty lines are ignored
    - Existing environment variables are NOT overwritten
    - Keys are stripped of whitespace
    - Values are stripped of whitespace but quotes are preserved if needed
    """
    global _ENV_LOADED

    if _ENV_LOADED and not force_reload:
        return

    # Determine .env file path
    if env_path is None:
        # Try current working directory first
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            env_path = str(cwd_env)
        else:
            # Try project root (parent of qx/ directory)
            qx_dir = Path(__file__).parent.parent
            project_root = qx_dir.parent
            root_env = project_root / ".env"
            if root_env.exists():
                env_path = str(root_env)
            else:
                # No .env file found, that's okay
                _ENV_LOADED = True
                return

    if not os.path.exists(env_path):
        _ENV_LOADED = True
        return

    # Load .env file
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse key=value
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Set environment variable if key is valid and not already set
                if key and value and key not in os.environ:
                    os.environ[key] = value

    _ENV_LOADED = True


def get_env_var(
    key: str,
    default: Optional[str] = None,
    required: bool = False,
    param_name: Optional[str] = None,
    param_value: Optional[str] = None,
) -> Optional[str]:
    """
    Get environment variable with fallback to parameter value.

    This is the standard pattern for API keys in Qx builders:
    1. Check parameter overrides (e.g., {"fred_api_key": "xxx"})
    2. Check environment variables (from .env or system)
    3. Use default value if provided
    4. Raise error if required=True and not found

    Args:
        key: Environment variable name (e.g., "FRED_API_KEY")
        default: Default value if not found (default: None)
        required: If True, raise ValueError if not found (default: False)
        param_name: Parameter name to check first (e.g., "fred_api_key")
        param_value: Parameter value from builder overrides

    Returns:
        str: Environment variable value, parameter value, or default

    Raises:
        ValueError: If required=True and variable not found

    Example:
    --------
    In a builder __init__:

        from qx.common.env_loader import load_env_file, get_env_var

        # Ensure .env is loaded
        load_env_file()

        # Get API key (checks param, then env, then raises if missing)
        api_key = get_env_var(
            key="FRED_API_KEY",
            param_name="fred_api_key",
            param_value=self.params.get("fred_api_key"),
            required=True
        )
    """
    # Ensure .env is loaded
    load_env_file()

    # Check parameter value first (overrides take precedence)
    if param_value:
        return param_value

    # Check environment variable
    value = os.environ.get(key)
    if value:
        return value

    # Use default if provided
    if default is not None:
        return default

    # Raise error if required
    if required:
        param_hint = f" or '{param_name}' parameter" if param_name else ""
        raise ValueError(
            f"{key} is required. "
            f"Set '{key}' environment variable in .env file{param_hint}"
        )

    return None
