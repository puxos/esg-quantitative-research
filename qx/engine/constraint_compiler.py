"""
Constraint compiler: Transform DSL specifications into canonical QP/LP matrices.

This module provides the ConstraintCompiler class which translates high-level
constraint specifications (ConstraintDSL) into canonical optimization problem
matrices (G, h, A, b, l, u) for quadratic/linear programming solvers.
"""

import logging
from typing import Optional

import numpy as np

from qx.engine.constraint_dsl import (
    BoundsConstraint,
    ConstraintDSL,
    DotConstraint,
    GroupBandConstraint,
    SumConstraint,
)
from qx.engine.lp_problem import LPProblem
from qx.engine.qp_problem import QPProblem

logger = logging.getLogger(__name__)


class ConstraintCompiler:
    """
    Compile constraint DSL specifications into canonical optimization matrices.

    This compiler transforms declarative constraint specifications into the
    canonical form required by QP/LP solvers:
        - Inequality constraints: G w ≤ h
        - Equality constraints: A w = b
        - Box constraints: l ≤ w ≤ u

    Example:
        >>> compiler = ConstraintCompiler()
        >>> qp = compiler.compile_qp(
        ...     dsl=constraint_dsl,
        ...     mu=expected_returns,
        ...     Sigma=covariance_matrix,
        ...     risk_aversion=4.0
        ... )
        >>> # qp now contains canonical matrices ready for solver
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compile_qp(
        self,
        dsl: ConstraintDSL,
        mu: np.ndarray,
        Sigma: np.ndarray,
        risk_aversion: float = 1.0,
        symbols: list[str] | None = None,
    ) -> QPProblem:
        """
        Compile constraint DSL + objective parameters into canonical QP problem.

        Args:
            dsl: Constraint specifications
            mu: Expected returns vector (n,)
            Sigma: Covariance matrix (n,n)
            risk_aversion: Risk-return tradeoff parameter λ
            symbols: Symbol ordering for variables. If None, uses dsl.symbols.
                     At least one must be provided.

        Returns:
            QPProblem with canonical matrices ready for solver

        Raises:
            ValueError: If constraints are inconsistent or infeasible
        """
        # Resolve symbols: use parameter if provided, else DSL's symbols
        syms = symbols if symbols is not None else dsl.symbols
        if syms is None:
            raise ValueError(
                "symbols must be provided either via parameter or in ConstraintDSL. "
                "Pass symbols= to compile_qp() or set symbols in ConstraintDSL."
            )

        # Validate alignment (optionally checks against provided symbols)
        dsl.validate_alignment(symbols=syms)
        n = len(syms)

        # Validate objective dimensions
        mu = np.asarray(mu, dtype=float)
        Sigma = np.asarray(Sigma, dtype=float)

        if mu.shape != (n,):
            raise ValueError(
                f"Expected returns mu must have shape ({n},), got {mu.shape}"
            )
        if Sigma.shape != (n, n):
            raise ValueError(
                f"Covariance Sigma must have shape ({n},{n}), got {Sigma.shape}"
            )

        self.logger.info(f"Compiling {len(dsl)} constraint(s) for {n} assets")

        # Initialize canonical matrices
        G_rows, h_rows = [], []
        A_rows, b_rows = [], []
        l = -np.inf * np.ones(n)
        u = np.inf * np.ones(n)
        meta_rows = []

        # Compile each constraint specification
        for idx, spec in enumerate(dsl.constraints):
            if isinstance(spec, BoundsConstraint):
                self._compile_bounds(spec, l, u, meta_rows, n)

            elif isinstance(spec, SumConstraint):
                self._compile_sum(spec, A_rows, b_rows, meta_rows, n)

            elif isinstance(spec, DotConstraint):
                self._compile_dot(spec, G_rows, h_rows, A_rows, b_rows, meta_rows)

            elif isinstance(spec, GroupBandConstraint):
                self._compile_group_band(spec, G_rows, h_rows, meta_rows, n)

            else:
                raise ValueError(f"Unknown constraint type: {type(spec).__name__}")

        # Build objective matrices
        P = risk_aversion * Sigma
        q = -mu

        # Stack constraint rows into matrices
        G = np.vstack(G_rows) if G_rows else np.zeros((0, n))
        h = np.array(h_rows) if h_rows else np.zeros(0)
        A = np.vstack(A_rows) if A_rows else np.zeros((0, n))
        b = np.array(b_rows) if b_rows else np.zeros(0)

        self.logger.info(
            f"Compiled to: {A.shape[0]} equality, {G.shape[0]} inequality, "
            f"bounds=[{l.max():.3f}, {u.min():.3f}]"
        )

        # Create QP problem
        qp = QPProblem(
            P=P,
            q=q,
            G=G,
            h=h,
            A=A,
            b=b,
            l=l,
            u=u,
            symbols=syms,  # Use resolved symbols (from parameter or DSL)
            meta={
                "constraints": meta_rows,
                "dsl_metadata": dsl.metadata,
                "risk_aversion": risk_aversion,
            },
        )

        # Validate before returning
        qp.validate()

        return qp

    def compile_lp(
        self,
        dsl: ConstraintDSL,
        objective_coefficients: np.ndarray,
        symbols: list[str] | None = None,
    ) -> LPProblem:
        """
        Compile constraint DSL + linear objective into canonical LP problem.

        Args:
            dsl: Constraint specifications
            objective_coefficients: Linear objective c (minimize c' w)
            symbols: Symbol ordering for variables. If None, uses dsl.symbols.
                     At least one must be provided.

        Returns:
            LPProblem with canonical matrices ready for solver
        """
        # Resolve symbols: use parameter if provided, else DSL's symbols
        syms = symbols if symbols is not None else dsl.symbols
        if syms is None:
            raise ValueError(
                "symbols must be provided either via parameter or in ConstraintDSL. "
                "Pass symbols= to compile_lp() or set symbols in ConstraintDSL."
            )

        dsl.validate_alignment(symbols=syms)
        n = len(syms)

        c = np.asarray(objective_coefficients, dtype=float)
        if c.shape != (n,):
            raise ValueError(
                f"Objective coefficients must have shape ({n},), got {c.shape}"
            )

        # Initialize canonical matrices
        G_rows, h_rows = [], []
        A_rows, b_rows = [], []
        l = -np.inf * np.ones(n)
        u = np.inf * np.ones(n)
        meta_rows = []

        # Compile each constraint (same logic as QP, just no objective matrices)
        for spec in dsl.constraints:
            if isinstance(spec, BoundsConstraint):
                self._compile_bounds(spec, l, u, meta_rows, n)
            elif isinstance(spec, SumConstraint):
                self._compile_sum(spec, A_rows, b_rows, meta_rows, n)
            elif isinstance(spec, DotConstraint):
                self._compile_dot(spec, G_rows, h_rows, A_rows, b_rows, meta_rows)
            elif isinstance(spec, GroupBandConstraint):
                self._compile_group_band(spec, G_rows, h_rows, meta_rows, n)
            else:
                raise ValueError(f"Unknown constraint type: {type(spec).__name__}")

        # Stack rows
        G = np.vstack(G_rows) if G_rows else np.zeros((0, n))
        h = np.array(h_rows) if h_rows else np.zeros(0)
        A = np.vstack(A_rows) if A_rows else np.zeros((0, n))
        b = np.array(b_rows) if b_rows else np.zeros(0)

        lp = LPProblem(
            c=c,
            G=G,
            h=h,
            A=A,
            b=b,
            l=l,
            u=u,
            symbols=syms,  # Use resolved symbols (from parameter or DSL)
            meta={"constraints": meta_rows, "dsl_metadata": dsl.metadata},
        )

        lp.validate()
        return lp

    def _compile_bounds(
        self,
        spec: BoundsConstraint,
        l: np.ndarray,
        u: np.ndarray,
        meta: list,
        n: int,
    ):
        """Compile bounds constraint: l ≤ w ≤ u"""
        # Handle scalar or vector bounds
        if isinstance(spec.lower, (int, float)):
            lower = np.full(n, spec.lower)
        else:
            lower = spec.lower

        if isinstance(spec.upper, (int, float)):
            upper = np.full(n, spec.upper)
        else:
            upper = spec.upper

        # Intersect with existing bounds (tighten)
        np.maximum(l, lower, out=l)
        np.minimum(u, upper, out=u)

        meta.append({"type": "bounds", "spec": spec})
        self.logger.debug(f"Added bounds: [{lower.min():.3f}, {upper.max():.3f}]")

    def _compile_sum(
        self,
        spec: SumConstraint,
        A_rows: list,
        b_rows: list,
        meta: list,
        n: int,
    ):
        """Compile sum constraint: ∑ w_i = target"""
        A_rows.append(np.ones(n))
        b_rows.append(spec.target)
        meta.append({"type": "sum", "spec": spec})
        self.logger.debug(f"Added sum constraint: ∑w = {spec.target}")

    def _compile_dot(
        self,
        spec: DotConstraint,
        G_rows: list,
        h_rows: list,
        A_rows: list,
        b_rows: list,
        meta: list,
    ):
        """Compile dot product constraint."""
        coef = spec.coefficients

        if spec.target is not None:
            # Equality: coef' w = target
            A_rows.append(coef)
            b_rows.append(spec.target)
            meta.append({"type": "dot_eq", "target": spec.target, "spec": spec})
            self.logger.debug(f"Added dot equality: coef'w = {spec.target}")

        else:
            # Inequality: lower ≤ coef' w ≤ upper
            if spec.lower is not None:
                # coef' w ≥ lower  →  -coef' w ≤ -lower
                G_rows.append(-coef)
                h_rows.append(-spec.lower)
                meta.append({"type": "dot_ge", "lower": spec.lower, "spec": spec})
                self.logger.debug(f"Added dot lower bound: coef'w ≥ {spec.lower}")

            if spec.upper is not None:
                # coef' w ≤ upper
                G_rows.append(coef)
                h_rows.append(spec.upper)
                meta.append({"type": "dot_le", "upper": spec.upper, "spec": spec})
                self.logger.debug(f"Added dot upper bound: coef'w ≤ {spec.upper}")

    def _compile_group_band(
        self,
        spec: GroupBandConstraint,
        G_rows: list,
        h_rows: list,
        meta: list,
        n: int,
    ):
        """
        Compile group active weight bands.

        For each group g:
            lower ≤ (∑_{i∈g} w_i) - (∑_{i∈g} benchmark_i) ≤ upper

        Expands to 2 × (# groups) inequality constraints.
        """
        groups = np.unique(spec.group_labels)
        benchmark = spec.benchmark if spec.benchmark is not None else np.zeros(n)

        for group in groups:
            # Create indicator vector for this group
            mask = (spec.group_labels == group).astype(float)
            bench_weight = float(mask @ benchmark)

            # Upper band: mask' w ≤ upper + bench_weight
            G_rows.append(mask)
            h_rows.append(spec.upper + bench_weight)
            meta.append(
                {
                    "type": "group_upper",
                    "group": str(group),
                    "upper": spec.upper,
                    "spec": spec,
                }
            )

            # Lower band: -mask' w ≤ -lower - bench_weight
            G_rows.append(-mask)
            h_rows.append(-spec.lower - bench_weight)
            meta.append(
                {
                    "type": "group_lower",
                    "group": str(group),
                    "lower": spec.lower,
                    "spec": spec,
                }
            )

        self.logger.debug(
            f"Added group bands for {len(groups)} groups: "
            f"[{spec.lower:.3f}, {spec.upper:.3f}]"
        )
