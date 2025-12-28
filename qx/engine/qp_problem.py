"""
Canonical quadratic programming problem representation.

This module defines the standard form for mean-variance portfolio optimization
and other quadratic programming problems in the Qx framework.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class QPProblem:
    """
    Canonical quadratic programming problem.

    Optimization problem:
        minimize    ½ w' P w + q' w
        subject to  G w ≤ h           (inequality constraints)
                    A w = b           (equality constraints)
                    l ≤ w ≤ u         (box constraints)

    Attributes:
        P: (n,n) positive semidefinite objective Hessian matrix
        q: (n,) linear objective coefficient vector
        G: (m,n) inequality constraint matrix (G w ≤ h)
        h: (m,) inequality constraint RHS
        A: (k,n) equality constraint matrix (A w = b)
        b: (k,) equality constraint RHS
        l: (n,) lower bounds on decision variables
        u: (n,) upper bounds on decision variables
        symbols: asset universe (for alignment and output)
        meta: metadata for provenance tracking and debugging

    For mean-variance portfolio optimization:
        - P = risk_aversion × Σ (covariance matrix)
        - q = -μ (negative expected returns)
        - G, h: inequality constraints (ESG bounds, sector bands)
        - A, b: equality constraints (budget, factor neutrality)
        - l, u: position limits (long-only, leverage caps)
    """

    P: np.ndarray
    q: np.ndarray
    G: np.ndarray
    h: np.ndarray
    A: np.ndarray
    b: np.ndarray
    l: np.ndarray
    u: np.ndarray
    symbols: list[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        """Convert inputs to numpy arrays and validate dimensions."""
        self.P = np.asarray(self.P, dtype=float)
        self.q = np.asarray(self.q, dtype=float)
        self.G = np.asarray(self.G, dtype=float)
        self.h = np.asarray(self.h, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.l = np.asarray(self.l, dtype=float)
        self.u = np.asarray(self.u, dtype=float)

    def validate(self):
        """
        Validate problem dimensions and feasibility.

        Raises:
            ValueError: If dimensions are inconsistent or problem is obviously infeasible
        """
        n = len(self.symbols)

        # Check objective dimensions
        if self.P.shape != (n, n):
            raise ValueError(f"P must be ({n},{n}), got {self.P.shape}")
        if self.q.shape != (n,):
            raise ValueError(f"q must be ({n},), got {self.q.shape}")

        # Check P is symmetric (approximately)
        if not np.allclose(self.P, self.P.T, rtol=1e-10):
            raise ValueError("P must be symmetric")

        # Check P is positive semidefinite (all eigenvalues ≥ 0)
        eigenvalues = np.linalg.eigvalsh(self.P)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                f"P must be positive semidefinite, min eigenvalue: {eigenvalues.min()}"
            )

        # Check inequality constraints
        if self.G.size > 0:
            if self.G.shape[1] != n:
                raise ValueError(f"G must have {n} columns, got {self.G.shape[1]}")
            if self.h.shape != (self.G.shape[0],):
                raise ValueError(
                    f"h must have length {self.G.shape[0]}, got {self.h.shape}"
                )

        # Check equality constraints
        if self.A.size > 0:
            if self.A.shape[1] != n:
                raise ValueError(f"A must have {n} columns, got {self.A.shape[1]}")
            if self.b.shape != (self.A.shape[0],):
                raise ValueError(
                    f"b must have length {self.A.shape[0]}, got {self.b.shape}"
                )

        # Check bounds
        if self.l.shape != (n,):
            raise ValueError(f"l must have length {n}, got {self.l.shape}")
        if self.u.shape != (n,):
            raise ValueError(f"u must have length {n}, got {self.u.shape}")

        # Check bounds feasibility
        if np.any(self.l > self.u):
            infeasible_idx = np.where(self.l > self.u)[0]
            raise ValueError(
                f"Infeasible bounds: l > u at positions {infeasible_idx}. "
                f"Example: l[{infeasible_idx[0]}]={self.l[infeasible_idx[0]]:.4f} > "
                f"u[{infeasible_idx[0]}]={self.u[infeasible_idx[0]]:.4f}"
            )

    @property
    def n_vars(self) -> int:
        """Number of decision variables."""
        return len(self.symbols)

    @property
    def n_ineq(self) -> int:
        """Number of inequality constraints."""
        return self.G.shape[0] if self.G.size > 0 else 0

    @property
    def n_eq(self) -> int:
        """Number of equality constraints."""
        return self.A.shape[0] if self.A.size > 0 else 0

    def summary(self) -> str:
        """Return a human-readable summary of the problem."""
        return (
            f"QPProblem:\n"
            f"  Variables: {self.n_vars}\n"
            f"  Inequality constraints: {self.n_ineq}\n"
            f"  Equality constraints: {self.n_eq}\n"
            f"  Bounds: [{self.l.min():.3f}, {self.u.max():.3f}]\n"
            f"  Symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}"
        )
