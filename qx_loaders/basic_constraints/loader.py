"""
Basic portfolio constraint loader.

Generates fundamental constraints for portfolio optimization:
- Budget constraint (weights sum to 1)
- Long-only constraint (no shorting)
- Position limits (max/min exposure per asset)
"""

import logging
from pathlib import Path

import pandas as pd

from qx.engine.constraint_dsl import BoundsConstraint, ConstraintDSL, SumConstraint
from qx.foundation.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class BasicConstraintLoader(BaseLoader):
    """
    Generate basic portfolio constraints.

    This loader emits fundamental constraints that apply to most portfolio
    optimization problems:
        - Budget: ∑w_i = 1 (fully invested)
        - Long-only: w_i ≥ 0 (no shorting)
        - Position limits: w_i ≤ max_position

    Parameters (from loader.yaml):
        budget: Target sum of weights (default: 1.0)
        long_only: Disallow shorting (default: true)
        position_max: Maximum weight per position (default: 0.5)
        position_min: Minimum weight per position (default: 0.0)

    Returns:
        ConstraintDSL with 2 constraints: SumConstraint + BoundsConstraint

    Design Philosophy:
        Constraints are UNIVERSE-AGNOSTIC. They don't need to know which symbols
        they apply to. Symbol ordering comes from the expected returns in the
        optimizer. This makes constraints reusable across different universes.
    """

    def load_impl(self) -> ConstraintDSL:
        """
        Generate basic constraints from parameters.

        Returns:
            ConstraintDSL object ready for compilation (symbols=None, inferred from optimizer)
        """
        # Extract constraint parameters (NO symbols needed!)
        budget = self.params.get("budget", 1.0)
        long_only = self.params.get("long_only", True)
        position_max = self.params.get("position_max", 0.5)
        position_min = self.params.get("position_min", 0.0)

        logger.info("Generating universe-agnostic basic constraints")
        logger.info(f"  Budget: {budget}")
        logger.info(f"  Long-only: {long_only}")
        logger.info(f"  Position limits: [{position_min}, {position_max}]")

        # Build constraints
        constraints = []

        # Budget constraint
        constraints.append(SumConstraint(target=budget))

        # Position bounds
        lower = position_min if long_only else -position_max
        constraints.append(BoundsConstraint(lower=lower, upper=position_max))

        dsl = ConstraintDSL(
            constraints=constraints,
            symbols=None,  # ✅ Universe-agnostic: inferred from optimizer
            metadata={
                "loader": "basic_constraints",
                "version": self.info["version"],
                "budget": budget,
                "long_only": long_only,
                "position_limits": [lower, position_max],
            },
        )

        logger.info(
            f"✅ Created ConstraintDSL: {len(constraints)} constraints (universe-agnostic)"
        )

        return dsl
