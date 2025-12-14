"""
ESG Panel Loader

Load ESG scores for a list of symbols with optional continuity filtering.

This loader bridges universe selection (historic_members, continuous_universe)
with ESG research by:
1. Loading ESG data for specified symbols and ESG year range
2. Optionally filtering to symbols with continuous coverage (no gaps)

Use Cases:
    - ESG factor research: Load continuous ESG data for universe
    - Portfolio construction: Get ESG scores for selected stocks
    - Coverage analysis: Identify symbols with complete ESG histories

Continuity Logic:
    ESG scores are published annually (one record per company per year).
    When continuity filtering is enabled, the loader checks for gaps in annual ESG data.
    A symbol is "continuous" if it has ESG data for EVERY year in the range.

Example usage:
    # Load all available ESG data for universe members
    Task(
        id="LoadESG",
        run=run_loader(
            package_path="qx_loaders/esg_panel",
            registry=registry,
            backend=backend,
            resolver=resolver,
            overrides={
                "start_date": "2014-01-01",
                "end_date": "2024-12-31",
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "continuous": False,
                "exchange": "US"
            }
        ),
        deps=[]
    )

    # Load only continuous ESG data (no gaps)
    Task(
        id="LoadContinuousESG",
        run=run_loader(
            package_path="qx_loaders/esg_panel",
            overrides={
                "symbols": ctx["GetHistoricMembers"]["output"],  # From upstream loader
                "continuous": True
            }
        ),
        deps=["GetHistoricMembers"]
    )
"""

from typing import List

import pandas as pd

from qx.common.types import AssetClass, DatasetType, Domain, Frequency, Subdomain
from qx.foundation.base_loader import BaseLoader


class ESGPanelLoader(BaseLoader):
    """
    Load ESG panel data for specified symbols with optional continuity filtering.

    Reads curated ESG scores and returns a DataFrame with ESG data for the
    specified symbols and date range. Optionally filters to symbols with
    continuous monthly coverage (no gaps).

    Parameters (from loader.yaml):
        start_date: Period start (YYYY-MM-DD) - used to determine ESG year range
        end_date: Period end (YYYY-MM-DD) - used to determine ESG year range
        symbols: List of ticker symbols (can be from upstream loader)
        continuity: Continuity mode - "any", "available", or "complete" (default: "any")
        exchange: Exchange filter (default: "US")

    Returns:
        pd.DataFrame: Annual ESG scores with columns:
            - ticker, gvkey, esg_year
            - esg_score, environmental_pillar_score, social_pillar_score, governance_pillar_score

    Continuity Filtering:
        ESG data is annual (one record per company per ESG publication year).
        When continuity != "any":
        1. Load all ESG data for symbols in ESG year range
        2. Check each symbol for annual gaps
        3. Exclude symbols with any missing ESG years
        4. Return only symbols with complete coverage
    """

    def load_impl(self) -> pd.DataFrame:
        """
        Load ESG panel data with optional continuity filtering.

        Returns:
            DataFrame with ESG scores for specified symbols
        """
        # Get parameters
        start_date = pd.Timestamp(self.params["start_date"])
        end_date = pd.Timestamp(self.params["end_date"])
        symbols = self.params["symbols"]
        continuous = self.params.get("continuous", False)
        exchange = self.params.get("exchange", "US")

        print(f"📊 Loading ESG panel data")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print(f"   Symbols: {len(symbols)} tickers")
        print(f"   Continuity mode: {self.params.get('continuity', 'any')}")
        print(f"   Exchange: {exchange}")

        # Handle empty symbol list
        if not symbols:
            print("⚠️  Empty symbol list, returning empty DataFrame")
            return pd.DataFrame()

        # Load ESG scores from curated data via typed loader (contract-based)
        # Note: ESG data is now YEARLY (one record per company per ESG publication year)
        esg_type = DatasetType(
            domain=Domain.ESG,
            asset_class=AssetClass.EQUITY,
            subdomain=Subdomain.ESG_SCORES,
            region=None,
            frequency=Frequency.YEARLY,
        )

        # ESG years use 1-year lag: calendar 2014 → esg_year=2013
        # For period 2014-2024, load esg_years 2013-2023
        esg_years = range(start_date.year - 1, end_date.year)

        # Use PyArrow filter pushdown for ticker filtering
        arrow_filters = [("ticker", "in", symbols)]

        dfs = []
        for esg_year in esg_years:
            try:
                # Use typed loading with filters
                df_year = self.curated_loader.load(
                    dataset_type=esg_type,
                    partitions={"exchange": exchange, "esg_year": str(esg_year)},
                    filters=arrow_filters,
                )
                if not df_year.empty:
                    dfs.append(df_year)
            except FileNotFoundError:
                # No data for this year, skip
                continue
            except Exception as e:
                print(f"⚠️  Error loading ESG data for year {esg_year}: {e}")
                continue

        # Combine all ESG years
        if not dfs:
            print("   ⚠️  No ESG data found for specified symbols and period")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Deduplicate in case multiple builder runs created duplicate files
        if not df.empty:
            if "ingest_ts" in df.columns:
                df = df.sort_values("ingest_ts", ascending=False)
            df = df.drop_duplicates(
                subset=["ticker", "gvkey", "esg_year"], keep="first"
            )

        if df.empty:
            print("⚠️  No ESG data found for specified symbols and period")
            return df

        # Filter by calendar year (user's trading period)
        # The 'year' column represents when data is available for trading
        # Calendar 2014-2024 → year 2014-2024 (automatically includes esg_year 2013-2023)
        start_calendar_year = start_date.year
        end_calendar_year = end_date.year
        df = df[
            (df["year"] >= start_calendar_year) & (df["year"] <= end_calendar_year)
        ].copy()

        print(f"✅ Loaded {len(df):,} annual ESG records")
        print(f"   Unique symbols: {df['ticker'].nunique()}")
        print(f"   ESG year range: {df['esg_year'].min()} to {df['esg_year'].max()}")

        # Apply continuity filter based on mode
        continuity = self.params.get("continuity", "any")
        if continuity != "any":
            print(f"🔍 Applying continuity filter (mode: {continuity})...")
            df = self._filter_continuous_symbols(df, start_date, end_date, continuity)

            if df.empty:
                print(f"⚠️  No symbols pass {continuity} continuity filter")
            else:
                print(f"✅ After {continuity} filter: {df['ticker'].nunique()} symbols")

        return df

    def _filter_continuous_symbols(
        self,
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        mode: str,
    ) -> pd.DataFrame:
        """
        Filter to symbols with continuous annual ESG data based on mode.

        ESG scores are published annually (one record per company per ESG year).
        This method checks for gaps in annual ESG publication years.

        Args:
            df: Annual ESG DataFrame
            start_date: Period start (used to determine ESG year range)
            end_date: Period end (used to determine ESG year range)
            mode: Continuity mode - "available" or "complete"

        Returns:
            DataFrame with only continuous symbols
        """
        if df.empty:
            return df

        print(
            f"   Checking annual ESG continuity for {df['ticker'].nunique()} symbols..."
        )

        continuous_symbols = []
        discontinuous_symbols = []

        if mode == "available":
            # Legacy method: Check each symbol from its own first to last ESG year
            for ticker in df["ticker"].unique():
                ticker_data = df[df["ticker"] == ticker]

                # Get the ticker's own ESG year range
                ticker_first_year = ticker_data["esg_year"].min()
                ticker_last_year = ticker_data["esg_year"].max()

                # Get expected years for THIS ticker's range
                expected_years = set(range(ticker_first_year, ticker_last_year + 1))

                # Get actual years
                actual_years = set(
                    ticker_data["esg_year"].dropna().astype(int).unique()
                )

                # Check for gaps in THIS ticker's range
                missing_years = expected_years - actual_years

                if len(missing_years) == 0:
                    continuous_symbols.append(ticker)
                else:
                    discontinuous_symbols.append(
                        {
                            "ticker": ticker,
                            "first_esg_year": ticker_first_year,
                            "last_esg_year": ticker_last_year,
                            "expected_years": len(expected_years),
                            "missing": len(missing_years),
                            "gap_pct": (
                                (len(missing_years) / len(expected_years)) * 100
                                if expected_years
                                else 0
                            ),
                        }
                    )

            print(f"   ✅ Continuous (in own range): {len(continuous_symbols)} symbols")
            print(f"   ⚠️  Gaps found: {len(discontinuous_symbols)} symbols")

        elif mode == "complete":
            # Strict method: All symbols must cover the full fixed period
            # ESG data uses 1-year lag convention (e.g., calendar 2014 data has esg_year=2013)
            # For date range 2014-2024, we expect esg_years: 2013, 2014, ..., 2023
            start_esg_year = start_date.year - 1  # Lag: 2014 → 2013
            end_esg_year = end_date.year - 1  # Lag: 2024 → 2023
            expected_years = set(range(start_esg_year, end_esg_year + 1))
            expected_count = len(expected_years)

            for ticker in df["ticker"].unique():
                ticker_data = df[df["ticker"] == ticker]

                # Get actual ESG years for this symbol
                actual_years = set(
                    ticker_data["esg_year"].dropna().astype(int).unique()
                )

                # Check if all expected years are present
                missing_years = expected_years - actual_years

                if len(missing_years) == 0:
                    continuous_symbols.append(ticker)
                else:
                    discontinuous_symbols.append(
                        {
                            "ticker": ticker,
                            "records": len(ticker_data),
                            "expected_years": expected_count,
                            "actual_years": len(actual_years),
                            "missing": len(missing_years),
                            "gap_pct": (len(missing_years) / expected_count) * 100,
                        }
                    )

            print(
                f"   ✅ Continuous (full period): {len(continuous_symbols)} symbols (all {expected_count} years)"
            )
            print(f"   ⚠️  Gaps found: {len(discontinuous_symbols)} symbols")

        # Show sample of excluded symbols
        if discontinuous_symbols and len(discontinuous_symbols) <= 10:
            print(f"   Sample excluded:")
            for info in sorted(
                discontinuous_symbols, key=lambda x: x["gap_pct"], reverse=True
            )[:10]:
                if mode == "available":
                    print(
                        f"      {info['ticker']:8} - {info['expected_years']} years expected, "
                        f"{info['missing']} missing ({info['gap_pct']:5.1f}%)"
                    )
                else:
                    print(
                        f"      {info['ticker']:8} - {info['actual_years']}/{info['expected_years']} years, "
                        f"{info['missing']} missing ({info['gap_pct']:5.1f}%)"
                    )

        # Filter to continuous symbols only
        return df[df["ticker"].isin(continuous_symbols)].copy()
