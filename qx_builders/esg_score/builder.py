"""
ESG Score Builder

Builds ESG (Environmental, Social, Governance) scores from local Excel/CSV files.
Uses GVKEY-ticker mapping to link company identifiers to ticker symbols.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from qx.common.ticker_mapper import TickerMapper
from qx.common.types import AssetClass, DatasetType, Domain, Frequency, Subdomain
from qx.foundation.base_builder import DataBuilderBase
from qx.storage.curated_writer import CuratedWriter
from qx_builders.gvkey_mapping import load_gvkey_mapping_from_curated


class ESGScoreBuilder(DataBuilderBase):
    """
    SOURCE BUILDER: ESG scores from local Excel/CSV files.
    
    External Source: Local data files (data_matlab_ESG_withSIC.xlsx)
    Authentication: None (local files)

    ESG Data Structure:
    - Annual ESG scores (YearESG) published yearly
    - Monthly observations (YearMonth in YYYYMM format)
    - GVKEY identifiers mapped to ticker symbols
    - Four scores: ESG composite + Environmental, Social, Governance pillars

    Unlike price or treasury data, ESG scores:
    - Come from local files (no API fetching)
    - Require GVKEY-to-ticker mapping
    - Have annual publication with monthly granularity
    - Include industry classifications (SIC codes)
    """

    def __init__(self, package_dir: str, writer: CuratedWriter, overrides=None):
        """
        Initialize ESG score builder from YAML configuration.

        Args:
            package_dir: Path to builder package containing builder.yaml
            writer: High-level curated data writer
            overrides: Parameter overrides

        Example:
            builder = ESGScoreBuilder(
                package_dir="qx_builders/esg_score",
                writer=writer,
                overrides={"start_year": 2010, "end_year": 2020}
            )
        """
        super().__init__(
            package_dir=package_dir,
            writer=writer,
            overrides=overrides,
        )

        # Get parameters from YAML config
        self.esg_source_path = self.params.get(
            "esg_source_path", "raw/data_matlab_ESG_withSIC.xlsx"
        )

        # Create ticker mapper if enabled
        use_mapper = self.params.get("use_ticker_mapper", True)
        self.ticker_mapper = TickerMapper() if use_mapper else None

    def fetch_raw(self, **kwargs) -> pd.DataFrame:
        """
        Fetch raw ESG data from local Excel/CSV file.

        Returns:
            Raw DataFrame with columns: gvkey, YearESG, YearMonth, ESG Score, pillar scores, etc.
        """
        source_path = kwargs.get("esg_source_path", self.esg_source_path)
        raw_path = Path(source_path)

        # If path is relative, resolve it relative to package directory
        if not raw_path.is_absolute():
            raw_path = self.package_dir / raw_path

        if not raw_path.exists():
            raise FileNotFoundError(
                f"ESG data file not found: {raw_path}\n"
                f"Please ensure data file is in the builder package's raw/ directory"
            )

        print(f"📂 Loading ESG data from {raw_path}")

        try:
            # Read Excel or CSV
            if raw_path.suffix == ".xlsx":
                df = pd.read_excel(raw_path)
            elif raw_path.suffix == ".csv":
                df = pd.read_csv(raw_path)
            else:
                raise ValueError(f"Unsupported file format: {raw_path.suffix}")

            # Validate required columns
            required_cols = ["gvkey", "YearESG", "YearMonth", "ESG Score"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"ESG data missing required columns: {missing_cols}\n"
                    f"Available columns: {df.columns.tolist()}"
                )

            print(f"✅ Loaded {len(df):,} raw ESG records")
            print(f"   Companies: {df['gvkey'].nunique():,}")
            print(f"   Columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            raise RuntimeError(f"Error reading ESG data file: {e}")

    def transform_to_curated(
        self, raw_df: pd.DataFrame, gvkey_mapping: pd.DataFrame, exchange: str, **kwargs
    ) -> pd.DataFrame:
        """
        Transform raw ESG data to curated format.

        Steps:
        1. Deduplicate to one record per (gvkey, YearESG) - ESG scores are constant within year
        2. Map GVKEY to ticker using gvkey_mapping
        3. Resolve ticker transitions (FB → META, etc.) using TickerMapper
        4. Filter and validate scores

        Args:
            raw_df: Raw ESG data with GVKEY identifiers
            gvkey_mapping: GVKEY-to-ticker mapping (from gvkey_mapping builder)
            exchange: Exchange filter ("US" or "HK")
            **kwargs: Additional keyword arguments

        Returns:
            Curated DataFrame with ticker symbols and annual ESG scores
        """
        if raw_df.empty:
            return raw_df

        # Make a copy
        df = raw_df.copy()

        # Ensure gvkey is integer
        df["gvkey"] = df["gvkey"].astype(int)

        # CRITICAL: Deduplicate to one record per (gvkey, YearESG)
        # ESG scores are constant within each publication year (variance = 0.0)
        # Keep first record for each company-year combination
        print(f"📊 Deduplicating ESG records...")
        print(f"   Before: {len(df):,} monthly records")
        df = df.drop_duplicates(subset=["gvkey", "YearESG"], keep="first")
        print(f"   After: {len(df):,} annual records")
        print(
            f"   Reduction: {len(raw_df) - len(df):,} duplicate monthly records removed"
        )

        # Map GVKEY to ticker
        gvkey_to_ticker = gvkey_mapping.set_index("gvkey")["ticker"].to_dict()
        df["ticker"] = df["gvkey"].map(gvkey_to_ticker)

        # Log mapping statistics
        total_gvkeys = df["gvkey"].nunique()
        mapped_gvkeys = df["ticker"].notna().sum()
        unmapped = total_gvkeys - df[df["ticker"].notna()]["gvkey"].nunique()

        print(f"📊 GVKEY mapping results:")
        print(f"   Total unique GVKEYs in ESG data: {total_gvkeys:,}")
        print(
            f"   Successfully mapped: {df[df['ticker'].notna()]['gvkey'].nunique():,}"
        )
        print(f"   Unmapped (no ticker found): {unmapped}")

        # Remove records without ticker mapping
        df = df[df["ticker"].notna()].copy()

        if df.empty:
            print("⚠️  No records after ticker mapping!")
            return pd.DataFrame()

        # Resolve ticker symbols (handle corporate actions: FB→META, CBS→PARA, etc.)
        print(f"📝 Resolving ticker symbols (corporate actions)...")
        original_count = len(df)

        # Apply ticker resolution
        df["ticker"] = df["ticker"].apply(
            lambda x: self.ticker_mapper.resolve(x) if pd.notna(x) else None
        )

        # Drop rows where ticker resolved to None (delisted/acquired)
        delisted_count = df["ticker"].isna().sum()
        if delisted_count > 0:
            print(
                f"⚠️  {delisted_count:,} ESG records for delisted tickers (will be dropped)"
            )
            df = df.dropna(subset=["ticker"])

        if df.empty:
            print("⚠️  No records after ticker resolution!")
            return pd.DataFrame()

        resolved_count = original_count - len(df)
        if resolved_count > 0:
            print(f"✅ Resolved {resolved_count:,} ticker transitions/delisted tickers")

        # Normalize column names (use long names for clarity)
        df = df.rename(
            columns={
                "YearESG": "esg_year",
                "ESG Score": "esg_score",
                "Environmental Pillar Score": "environmental_pillar_score",
                "Social Pillar Score": "social_pillar_score",
                "Governance Pillar Score": "governance_pillar_score",
            }
        )

        # Add 'year' column: calendar year when data becomes available for trading
        # ESG year 2013 published in 2014 → available for 2014 trading
        df["year"] = df["esg_year"] + 1

        # Select required columns (annual schema - no date/year/month)
        required_cols = [
            "ticker",
            "gvkey",
            "esg_year",
            "year",  # Calendar year when data is available
            "esg_score",
            "environmental_pillar_score",
            "social_pillar_score",
            "governance_pillar_score",
        ]

        # Keep only columns that exist
        final_cols = [col for col in required_cols if col in df.columns]
        curated = df[final_cols].copy()

        # Convert scores to numeric
        score_cols = [
            "esg_score",
            "environmental_pillar_score",
            "social_pillar_score",
            "governance_pillar_score",
        ]
        for col in score_cols:
            if col in curated.columns:
                curated[col] = pd.to_numeric(curated[col], errors="coerce")

        # Remove rows with missing scores
        curated = curated.dropna(subset=["esg_score"])

        print(f"✅ Curated: {len(curated):,} annual ESG observations")
        print(f"   Unique tickers: {curated['ticker'].nunique():,}")
        print(
            f"   ESG year range: {curated['esg_year'].min()} to {curated['esg_year'].max()}"
        )
        print(f"\nESG Score Statistics:")
        print(curated["esg_score"].describe())

        return curated

    def build(self, partitions: dict = None, **kwargs):
        """
        Build ESG scores with auto-partitioning by ESG publication year.

        Args:
            partitions: Must contain {"exchange": "US"|"HK"}
            **kwargs: Additional parameters
                start_year (int): Start calendar year (inclusive). ESG year will be start_year-1.
                end_year (int): End calendar year (inclusive). ESG year will be end_year-1.
                                None = auto-detect

        Returns:
            Dict with status, output_path, and metadata

        Note:
            ESG data has 1-year lag. For calendar period 2014-2024:
            - Loads ESG years 2013-2023 (both start and end adjusted by -1)
            - ESG year 2013 published in 2014, available for 2014 trading
            - ESG year 2023 published in 2024, available for 2024 trading

        Example:
            task = Task(
                id="BuildESG",
                run=run_builder(
                    "qx_builders/esg_score",
                    partitions={"exchange": "US"},
                    overrides={"start_year": 2014, "end_year": 2024}  # Calendar years
                    # → Will build ESG years 2013-2023
                )
            )
        """
        # Extract exchange from partitions or kwargs
        if partitions is None:
            partitions = {}
        exchange = partitions.get("exchange") or kwargs.get("exchange", "US")

        # Get year range (these are CALENDAR years from user)
        start_year = kwargs.get("start_year") or self.params.get("start_year")
        end_year = kwargs.get("end_year") or self.params.get("end_year")

        print(f"\n📈 Building ESG scores for exchange={exchange}")

        # Fetch and transform once
        raw_df = self.fetch_raw(**kwargs)

        # Load GVKEY mapping
        print("📂 Loading GVKEY-ticker mapping...")
        gvkey_mapping = load_gvkey_mapping_from_curated(exchange=exchange)
        print(f"✅ Loaded {len(gvkey_mapping):,} mappings")

        curated = self.transform_to_curated(raw_df, gvkey_mapping, exchange, **kwargs)

        if curated.empty:
            print("⚠️  No ESG data to build")
            return []

        # Auto-detect ESG year range if not specified
        available_esg_years = sorted(curated["esg_year"].unique())

        # Convert calendar years to ESG years (with 1-year lag)
        # Calendar 2014 → ESG year 2013 (published in 2014)
        if start_year is None:
            start_esg_year = int(available_esg_years[0])
            print(f"📅 Auto-detected start ESG year: {start_esg_year}")
        else:
            start_esg_year = start_year - 1  # Apply 1-year lag
            print(f"📅 Calendar year {start_year} → ESG year {start_esg_year}")

        if end_year is None:
            end_esg_year = int(available_esg_years[-1])
            print(f"📅 Auto-detected end ESG year: {end_esg_year}")
        else:
            end_esg_year = end_year - 1  # Apply 1-year lag
            print(f"📅 Calendar year {end_year} → ESG year {end_esg_year}")

        print(f"📅 Building ESG years: {start_esg_year} to {end_esg_year}")
        print(
            f"💾 Available ESG years in data: {available_esg_years[0]}-{available_esg_years[-1]}"
        )

        # Build each ESG year using writer (handles partitioning automatically)
        output_paths = []
        for esg_year in range(start_esg_year, end_esg_year + 1):
            year_data = curated[curated["esg_year"] == esg_year].copy()

            if year_data.empty:
                print(f"⚠️  ESG year {esg_year}: No data (skipped)")
                continue

            # Add metadata
            year_data["schema_version"] = self.output_dt_template.frequency or "v1"
            year_data["ingest_ts"] = pd.Timestamp.utcnow()

            # Write partition using writer abstraction
            partitions_dict = {"exchange": exchange, "esg_year": str(esg_year)}
            output_path = self.writer.write(
                data=year_data,
                dataset_type=self.output_dt_template,
                partitions=partitions_dict
            )
            output_paths.append(output_path)

            print(
                f"💾 ESG year {esg_year}: {output_path} ({len(year_data)} rows, {year_data['ticker'].nunique()} unique tickers)"
            )

        print(f"\n✅ Built {len(output_paths)} ESG year partitions")
        
        # Return manifest dict
        return {
            "status": "success",
            "builder": self.info["id"],
            "version": self.info["version"],
            "output_path": output_paths[0] if output_paths else None,
            "rows": sum(len(curated[curated["esg_year"] == y]) for y in range(start_esg_year, end_esg_year + 1)),
            "layer": "curated"
        }
