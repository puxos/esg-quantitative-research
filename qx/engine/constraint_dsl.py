"""
Constraint DSL (Domain-Specific Language) schema for portfolio optimization.

This module defines type-safe dataclasses for specifying portfolio constraints
in a declarative, composable way. Constraints are compiled into canonical
optimization problem matrices by the ConstraintCompiler.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np


@dataclass
class BoundsConstraint:
    """
    Box constraints on decision variables: l ≤ w ≤ u

    Examples:
        - Long-only: BoundsConstraint(lower=0.0, upper=1.0)
        - Position limits: BoundsConstraint(lower=0.0, upper=0.1)  # max 10% per position
        - Leverage cap: BoundsConstraint(lower=-0.5, upper=0.5)   # allow shorting

    Attributes:
        lower: Lower bound (scalar or per-asset array)
        upper: Upper bound (scalar or per-asset array)
    """

    type: Literal["bounds"] = "bounds"
    lower: float | np.ndarray = 0.0
    upper: float | np.ndarray = 1.0

    def __post_init__(self):
        """Ensure bounds are numeric."""
        if isinstance(self.lower, (list, tuple)):
            self.lower = np.asarray(self.lower, dtype=float)
        if isinstance(self.upper, (list, tuple)):
            self.upper = np.asarray(self.upper, dtype=float)


@dataclass
class SumConstraint:
    """
    Sum constraint: ∑ w_i = target

    Examples:
        - Budget constraint: SumConstraint(target=1.0)
        - Dollar-neutral: SumConstraint(target=0.0)
        - Partial investment: SumConstraint(target=0.8)

    Attributes:
        target: Target sum value
    """

    type: Literal["sum"] = "sum"
    target: float = 1.0


@dataclass
class DotConstraint:
    """
    Dot product constraint: coefficients' w ∈ [lower, upper] or = target

    Used for factor exposures, ESG constraints, beta neutrality, etc.

    Examples:
        - ESG minimum: DotConstraint(coefficients=esg_scores, lower=0.6)
        - Beta neutral: DotConstraint(coefficients=market_betas, target=0.0)
        - Dividend yield: DotConstraint(coefficients=div_yields, lower=0.02, upper=0.04)

    Attributes:
        coefficients: Coefficient vector (e.g., ESG scores, betas)
        lower: Lower bound on dot product (inequality)
        upper: Upper bound on dot product (inequality)
        target: Exact target value (equality constraint)

    Note: Exactly one of (lower, upper, target) or (lower and upper) must be specified.
    """

    type: Literal["dot"] = "dot"
    coefficients: np.ndarray = field(default_factory=lambda: np.array([]))
    lower: Optional[float] = None
    upper: Optional[float] = None
    target: Optional[float] = None

    def __post_init__(self):
        """Validate constraint specification and convert coefficients."""
        self.coefficients = np.asarray(self.coefficients, dtype=float)

        # Validate that we have a valid constraint specification
        has_target = self.target is not None
        has_bounds = self.lower is not None or self.upper is not None

        if has_target and has_bounds:
            raise ValueError(
                "DotConstraint cannot have both 'target' (equality) and bounds (inequality). "
                "Use either target=value OR lower/upper bounds."
            )

        if not has_target and not has_bounds:
            raise ValueError(
                "DotConstraint must specify either 'target' (equality) or "
                "'lower'/'upper' (inequality bounds)."
            )


@dataclass
class GroupBandConstraint:
    """
    Group-level active weight bands relative to benchmark.

    Used for sector neutrality, country constraints, etc.

    For each group g:
        lower ≤ (∑_{i∈g} w_i) - (∑_{i∈g} benchmark_i) ≤ upper

    Examples:
        - Sector neutral: GroupBandConstraint(
            group_labels=sector_ids,
            benchmark=benchmark_weights,
            lower=-0.02, upper=0.02
          )
        - Country bands: GroupBandConstraint(
            group_labels=country_codes,
            benchmark=None,  # relative to 0
            lower=-0.10, upper=0.10
          )

    Attributes:
        group_labels: Array of group identifiers (e.g., sector IDs, country codes)
        benchmark: Reference weights (None means 0 for all groups)
        lower: Lower band on active weight
        upper: Upper band on active weight
    """

    type: Literal["group_band"] = "group_band"
    group_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    benchmark: Optional[np.ndarray] = None
    lower: float = -0.05
    upper: float = 0.05

    def __post_init__(self):
        """Convert inputs to numpy arrays."""
        self.group_labels = np.asarray(self.group_labels)
        if self.benchmark is not None:
            self.benchmark = np.asarray(self.benchmark, dtype=float)


# Type alias for any constraint specification
ConstraintSpec = Union[
    BoundsConstraint, SumConstraint, DotConstraint, GroupBandConstraint
]


@dataclass
class ConstraintDSL:
    """
    Container for all constraint specifications.

    This is the output format for constraint loaders. Multiple ConstraintDSL
    objects can be merged by concatenating their constraint lists.

    Attributes:
        constraints: List of constraint specifications
        symbols: Asset universe (OPTIONAL - inferred from optimization problem if not provided)
        metadata: Provenance tracking (loader name, version, parameters)

    Design Philosophy:
        - Constraints are universe-agnostic (SumConstraint, BoundsConstraint don't need symbols)
        - Symbol ordering comes from expected returns/covariance matrix in optimizer
        - Loaders should NOT require symbols - constraints are reusable across universes

    Example:
        >>> # ✅ RECOMMENDED: No symbols (inferred from optimizer)
        >>> dsl = ConstraintDSL(
        ...     constraints=[
        ...         SumConstraint(target=1.0),
        ...         BoundsConstraint(lower=0.0, upper=0.5)
        ...     ],
        ...     metadata={'loader': 'basic_constraints', 'version': '1.0.0'}
        ... )
        >>>
        >>> # ⚠️ LEGACY: Explicit symbols (for backward compatibility)
        >>> dsl = ConstraintDSL(
        ...     constraints=[DotConstraint(coefficients=esg_scores, lower=0.6)],
        ...     symbols=['AAPL', 'MSFT', 'GOOGL'],  # Only needed if coefficients are pre-aligned
        ...     metadata={'loader': 'esg_constraints', 'version': '1.0.0'}
        ... )
    """

    constraints: list[ConstraintSpec]
    symbols: list[str] | None = None  # ✅ OPTIONAL: Inferred from optimizer if None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of constraint specifications."""
        return len(self.constraints)

    def validate_alignment(self, symbols: list[str] | None = None):
        """
        Validate that all array-based constraints are aligned to symbols.

        Args:
            symbols: Symbol list to validate against (uses self.symbols if not provided)

        Raises:
            ValueError: If any constraint has mismatched dimensions or symbols not provided
        """
        syms = symbols if symbols is not None else self.symbols
        if syms is None:
            # Cannot validate without symbols - defer to compile time
            return

        n = len(syms)

        for i, spec in enumerate(self.constraints):
            if isinstance(spec, BoundsConstraint):
                if isinstance(spec.lower, np.ndarray) and len(spec.lower) != n:
                    raise ValueError(
                        f"Constraint {i} (BoundsConstraint): lower has length {len(spec.lower)}, "
                        f"expected {n} (universe size)"
                    )
                if isinstance(spec.upper, np.ndarray) and len(spec.upper) != n:
                    raise ValueError(
                        f"Constraint {i} (BoundsConstraint): upper has length {len(spec.upper)}, "
                        f"expected {n} (universe size)"
                    )

            elif isinstance(spec, DotConstraint):
                if len(spec.coefficients) != n:
                    raise ValueError(
                        f"Constraint {i} (DotConstraint): coefficients has length {len(spec.coefficients)}, "
                        f"expected {n} (universe size)"
                    )

            elif isinstance(spec, GroupBandConstraint):
                if len(spec.group_labels) != n:
                    raise ValueError(
                        f"Constraint {i} (GroupBandConstraint): group_labels has length {len(spec.group_labels)}, "
                        f"expected {n} (universe size)"
                    )
                if spec.benchmark is not None and len(spec.benchmark) != n:
                    raise ValueError(
                        f"Constraint {i} (GroupBandConstraint): benchmark has length {len(spec.benchmark)}, "
                        f"expected {n} (universe size)"
                    )

    def summary(self) -> str:
        """Human-readable summary of constraint DSL."""
        constraint_counts = {}
        for spec in self.constraints:
            ctype = spec.type
            constraint_counts[ctype] = constraint_counts.get(ctype, 0) + 1

        counts_str = ", ".join(f"{k}={v}" for k, v in constraint_counts.items())

        symbols_str = (
            f"{len(self.symbols)}" if self.symbols is not None else "<inferred>"
        )

        return (
            f"ConstraintDSL:\n"
            f"  Symbols: {symbols_str}\n"
            f"  Constraints: {len(self.constraints)} ({counts_str})\n"
            f"  Metadata: {self.metadata.get('loader', 'unknown')}"
        )
