"""
Rebuild US Treasury Rate Data for 2014-2024

This script rebuilds treasury rate data with the correct date range
to support ESG factor research from 2014 onwards.
"""

from pathlib import Path

from qx.common.contracts import DatasetRegistry
from qx.common.predefined import seed_registry
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.curated_writer import CuratedWriter
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter
from qx_builders.us_treasury_rate.builder import USTreasuryRateBuilder

print("=" * 80)
print("Rebuild US Treasury Rate Data (2014-2024)")
print("=" * 80)

# Setup storage infrastructure
registry = DatasetRegistry()
seed_registry(registry)

backend = LocalParquetBackend(base_uri="file://.")
adapter = TableFormatAdapter(backend)
resolver = PathResolver()
writer = CuratedWriter(
    backend=backend, adapter=adapter, resolver=resolver, registry=registry
)

# Create builder with correct date range
package_dir = str(Path(__file__).parent.parent / "qx_builders" / "us_treasury_rate")

builder = USTreasuryRateBuilder(
    package_dir=package_dir,
    writer=writer,
    overrides={
        "start_date": "2014-01-01",  # ← Match ESG research period
        "end_date": "2024-12-31",
        "rate_types": ["3month"],  # Only 3-month for risk-free rate
        "frequency": "daily",
        "write_mode": "overwrite",
    },
)

print(f"\nBuilder: {builder.info['id']} v{builder.info['version']}")
print(f"Date range: 2014-01-01 to 2024-12-31")
print(f"Rate types: 3month (T-bill)")
print(f"Frequency: daily")
print()

# Build monthly data as well (for direct monthly use)
print("=" * 80)
print("Building Treasury Rates...")
print("=" * 80)

# Build daily data
print("\n📊 Building daily treasury rates...")
result_daily = builder.build(partitions={"region": "US", "frequency": "daily"})

if result_daily["status"] == "success":
    print(f"✅ Daily data built successfully")
    print(f"   Rows: {result_daily.get('rows', 'unknown')}")
    print(f"   Path: {result_daily.get('output_path', 'unknown')}")
else:
    print(f"❌ Daily build failed: {result_daily}")

print("\n" + "=" * 80)
print("✅ Treasury Rate Rebuild Complete!")
print("=" * 80)
print("\nNext steps:")
print("1. Re-run the ESG research pipeline")
print("2. Factor calculation should now start from 2014")
print("3. Expected factor observations: ~120 (10 years × 12 months)")
