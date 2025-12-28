"""
ESG Constraints Loader

Generates ESG-based portfolio constraints by loading ESG scores and creating
factor exposure constraints.

Design: Dimension-specific (requires symbols for DotConstraint coefficients).
Unlike basic constraints (SumConstraint, BoundsConstraint), ESG constraints
use DotConstraint which needs symbol-aligned coefficient vectors.

BEST PRACTICE: Get symbols from model's expected returns, not DAG hardcoding:
    # In model:
    symbols = exp_ret_series.index.tolist()
    esg_dsl = esg_loader.load(overrides={'symbols': symbols})

Usage:
    loader = ESGConstraintLoader(
        package_dir="qx_loaders/esg_constraints",
        loader=typed_loader,
        overrides={"symbols": symbols, "min_esg_score": 60}
    )
    constraint_dsl = loader.load()
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd

from qx.common.types import DatasetType, Domain, Subdomain
from qx.engine.constraint_dsl import ConstraintDSL, DotConstraint
from qx.foundation.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class ESGConstraintLoader(BaseLoader):
    """
    Load ESG scores and generate portfolio constraints.

    Returns ConstraintDSL with DotConstraint enforcing minimum portfolio ESG score.
    If no ESG data is available, returns empty ConstraintDSL (no constraints).

    Design Note:
        This loader is DIMENSION-SPECIFIC (requires symbols) because DotConstraint
        needs symbol-aligned coefficient vectors. Get symbols from expected returns
        in the model, not from hardcoded DAG tasks.
    """

    def load_impl(self) -> ConstraintDSL:
        """
        Load ESG data and generate constraints.

        Returns:
            ConstraintDSL with ESG exposure constraint
        """
        symbols = self.params["symbols"]
        min_esg_score = self.params.get("min_esg_score")
        max_esg_score = self.params.get("max_esg_score")
        target_esg_score = self.params.get("target_esg_score")

        if not symbols:
            raise ValueError("symbols parameter is required")

        logger.info(f"Loading ESG constraints for {len(symbols)} symbols")

        # Load ESG scores from curated data
        try:
            esg_df = self._load_esg_scores(symbols)
        except Exception as e:
            logger.warning(
                f"Could not load ESG data: {e}. Returning empty constraints."
            )
            return ConstraintDSL(
                constraints=[],
                symbols=symbols,
                metadata={
                    "loader": "esg_constraints",
                    "warning": "No ESG data available",
                },
            )

        if esg_df.empty:
            logger.warning("No ESG scores found. Returning empty constraints.")
            return ConstraintDSL(
                constraints=[],
                symbols=symbols,
                metadata={
                    "loader": "esg_constraints",
                    "warning": "No ESG data for symbols",
                },
            )

        # Create ESG score vector aligned with symbols
        esg_scores = self._align_esg_scores(esg_df, symbols)

        # Build DotConstraint: esg_scores' * weights ∈ [min, max] or = target
        constraints = []

        if target_esg_score is not None:
            # Equality constraint
            constraint = DotConstraint(
                coefficients=esg_scores,
                target=target_esg_score,
                lower=None,
                upper=None,
                label="target_esg_score",
            )
            constraints.append(constraint)
            logger.info(f"Added target ESG score constraint: {target_esg_score:.2f}")

        else:
            # Range constraint
            constraint = DotConstraint(
                coefficients=esg_scores,
                target=None,
                lower=min_esg_score,
                upper=max_esg_score,
                label="esg_score_range",
            )
            constraints.append(constraint)

            if min_esg_score is not None:
                logger.info(f"Added minimum ESG score constraint: {min_esg_score:.2f}")
            if max_esg_score is not None:
                logger.info(f"Added maximum ESG score constraint: {max_esg_score:.2f}")

        return ConstraintDSL(
            constraints=constraints,
            symbols=symbols,
            metadata={
                "loader": "esg_constraints",
                "esg_scores_count": len([s for s in esg_scores if s is not None]),
                "avg_esg_score": (
                    sum(s for s in esg_scores if s is not None)
                    / len([s for s in esg_scores if s is not None])
                    if any(s is not None for s in esg_scores)
                    else None
                ),
            },
        )

    def _load_esg_scores(self, symbols: List[str]) -> pd.DataFrame:
        """
        Load ESG scores from curated data.

        Args:
            symbols: List of ticker symbols

        Returns:
            DataFrame with columns: [ticker, esg_score]
        """
        # Load ESG scores dataset
        esg_dt = DatasetType(domain=Domain.ESG, subdomain=Subdomain.ESG_SCORES)

        # Load most recent ESG scores
        # Note: This assumes annual ESG data is already loaded by esg_score builder
        df = self.loader.load(
            dt=esg_dt,
            partitions={"exchange": self.params.get("exchange", "US")},
            columns=["ticker", "esg_score"],
            filters=None,  # Get all available data, will filter to symbols
        )

        # Filter to requested symbols
        df = df[df["ticker"].isin(symbols)]

        # If multiple years, take most recent
        if "year" in df.columns:
            df = (
                df.sort_values("year", ascending=False)
                .groupby("ticker")
                .first()
                .reset_index()
            )

        logger.info(f"Loaded ESG scores for {len(df)} / {len(symbols)} symbols")

        return df

    def _align_esg_scores(
        self, esg_df: pd.DataFrame, symbols: List[str]
    ) -> List[float]:
        """
        Create ESG score vector aligned with symbols list.

        Args:
            esg_df: DataFrame with ticker and esg_score columns
            symbols: List of symbols in portfolio universe

        Returns:
            List of ESG scores, with None for missing symbols
        """
        # Create lookup dict
        esg_lookup = dict(zip(esg_df["ticker"], esg_df["esg_score"]))

        # Align with symbols
        aligned_scores = []
        for symbol in symbols:
            score = esg_lookup.get(symbol)
            if score is None:
                logger.warning(f"No ESG score for {symbol}, using 0.0 as default")
                score = 0.0  # Default to 0 for missing scores
            aligned_scores.append(float(score))

        return aligned_scores
