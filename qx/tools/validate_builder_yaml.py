"""
Validate Builder YAML Configuration

This script validates builder.yaml files to catch typos and invalid enum values early.

Usage:
    python -m qx.tools.validate_builder_yaml qx_builders/tiingo_ohlcv/builder.yaml
    python -m qx.tools.validate_builder_yaml qx_builders/*/builder.yaml
"""

import sys
from pathlib import Path

import yaml

from qx.common.enum_validator import (
    EnumValidationError,
    print_valid_enum_values,
    validate_dataset_type_config,
    validate_frequency_parameter,
)


def validate_builder_yaml(yaml_path: Path) -> bool:
    """
    Validate a builder.yaml file.

    Args:
        yaml_path: Path to builder.yaml file

    Returns:
        True if valid, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Validating: {yaml_path}")
    print(f"{'='*80}")

    try:
        # Load YAML
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate builder section
        if "builder" not in config:
            print("❌ Missing 'builder' section")
            return False

        builder = config["builder"]
        print(f"✅ Builder: {builder.get('id')} v{builder.get('version')}")

        # Validate output section (top-level)
        if "output" not in config:
            print("❌ Missing 'output' section")
            return False

        output = config["output"]

        # Validate output type
        if "type" not in output:
            print("❌ Missing 'output.type' section")
            return False

        output_type = output["type"]
        print(f"\n📊 Output Type:")
        print(f"   domain: {output_type.get('domain')}")
        print(f"   asset_class: {output_type.get('asset_class')}")
        print(f"   subdomain: {output_type.get('subdomain')}")
        print(f"   region: {output_type.get('region')}")
        print(f"   frequency: {output_type.get('frequency')}")

        # Validate with strict enum checking
        try:
            validated = validate_dataset_type_config(output_type)
            print("\n✅ All enum values are valid!")

            print("\n📋 Validated Enums:")
            for key, value in validated.items():
                if value is not None:
                    print(
                        f"   {key}: {value.value} ({value.__class__.__name__}.{value.name})"
                    )
                else:
                    print(f"   {key}: null")

        except EnumValidationError as e:
            print(f"\n❌ Enum Validation Error:")
            print(f"   {str(e)}")
            return False

        # Validate parameters with frequency
        if "parameters" in config:
            params = config["parameters"]
            if "frequency" in params:
                freq_param = params["frequency"]
                default_freq = freq_param.get("default")
                allowed_freq = freq_param.get("allowed", [])

                print(f"\n📅 Frequency Parameter:")
                print(f"   default: {default_freq}")
                print(f"   allowed: {allowed_freq}")

                # Validate default frequency
                if default_freq:
                    try:
                        validate_frequency_parameter(default_freq)
                        print(f"   ✅ Default frequency '{default_freq}' is valid")
                    except EnumValidationError as e:
                        print(f"   ❌ Invalid default frequency: {str(e)}")
                        return False

                # Validate allowed frequencies
                for freq in allowed_freq:
                    try:
                        validate_frequency_parameter(freq)
                    except EnumValidationError as e:
                        print(f"   ❌ Invalid allowed frequency '{freq}': {str(e)}")
                        return False

                if allowed_freq:
                    print(f"   ✅ All allowed frequencies are valid")

        print(f"\n{'='*80}")
        print("✅ VALIDATION PASSED")
        print(f"{'='*80}")
        return True

    except FileNotFoundError:
        print(f"❌ File not found: {yaml_path}")
        return False
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for CLI."""
    if len(sys.argv) < 2:
        print("Usage: python -m qx.tools.validate_builder_yaml <builder.yaml>")
        print("\nOr to see valid enum values:")
        print("       python -m qx.tools.validate_builder_yaml --show-enums")
        sys.exit(1)

    if sys.argv[1] == "--show-enums":
        print_valid_enum_values()
        sys.exit(0)

    # Validate each YAML file provided
    yaml_files = sys.argv[1:]
    all_valid = True

    for yaml_file in yaml_files:
        path = Path(yaml_file)
        if not validate_builder_yaml(path):
            all_valid = False

    if not all_valid:
        print("\n" + "=" * 80)
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 80)
        sys.exit(1)
    else:
        if len(yaml_files) > 1:
            print("\n" + "=" * 80)
            print(f"✅ ALL {len(yaml_files)} FILES VALIDATED SUCCESSFULLY")
            print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
