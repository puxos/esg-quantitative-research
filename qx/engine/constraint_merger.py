"""
Utility for merging multiple ConstraintDSL objects.

This module provides functions to safely combine constraint specifications
from different loaders, validating alignment and preserving provenance.
"""

import logging

from qx.engine.constraint_dsl import ConstraintDSL

logger = logging.getLogger(__name__)


def merge_constraint_dsls(*dsls: ConstraintDSL) -> ConstraintDSL:
    """
    Merge multiple ConstraintDSL objects into a single specification.

    This function:
    1. Validates all DSLs have the same symbol universe
    2. Concatenates constraint lists
    3. Merges metadata with provenance tracking

    Args:
        *dsls: Variable number of ConstraintDSL objects to merge

    Returns:
        Single merged ConstraintDSL

    Raises:
        ValueError: If no DSLs provided or symbol universes don't match

    Example:
        >>> basic_dsl = BasicConstraintLoader().load()
        >>> esg_dsl = ESGConstraintLoader().load()
        >>> merged = merge_constraint_dsls(basic_dsl, esg_dsl)
        >>> # merged contains constraints from both loaders
    """
    if not dsls:
        raise ValueError("No ConstraintDSL objects to merge")

    if len(dsls) == 1:
        # Nothing to merge, return as-is
        return dsls[0]

    # Validate all have the same symbols
    symbols = dsls[0].symbols
    for i, dsl in enumerate(dsls[1:], start=1):
        if dsl.symbols != symbols:
            raise ValueError(
                f"Symbol universe mismatch in DSL merge:\n"
                f"  DSL[0]: {len(symbols)} symbols (first 5: {symbols[:5]})\n"
                f"  DSL[{i}]: {len(dsl.symbols)} symbols (first 5: {dsl.symbols[:5]})\n"
                f"All constraint loaders must operate on the same universe."
            )

    # Concatenate all constraints
    all_constraints = []
    for dsl in dsls:
        all_constraints.extend(dsl.constraints)

    # Merge metadata with provenance tracking
    merged_metadata = {
        "merged_from": [dsl.metadata for dsl in dsls],
        "n_sources": len(dsls),
        "total_constraints": len(all_constraints),
    }

    # Count constraints by type for summary
    constraint_counts = {}
    for dsl in dsls:
        for spec in dsl.constraints:
            ctype = spec.type
            constraint_counts[ctype] = constraint_counts.get(ctype, 0) + 1
    merged_metadata["constraint_type_counts"] = constraint_counts

    merged = ConstraintDSL(
        constraints=all_constraints,
        symbols=symbols,
        metadata=merged_metadata,
    )

    logger.info(
        f"Merged {len(dsls)} ConstraintDSL objects: "
        f"{len(all_constraints)} total constraints for {len(symbols)} symbols"
    )

    return merged


def validate_constraint_compatibility(*dsls: ConstraintDSL) -> bool:
    """
    Check if multiple ConstraintDSL objects can be safely merged.

    Args:
        *dsls: ConstraintDSL objects to validate

    Returns:
        True if all DSLs are compatible, False otherwise
    """
    if len(dsls) <= 1:
        return True

    symbols = dsls[0].symbols
    for dsl in dsls[1:]:
        if dsl.symbols != symbols:
            return False

    return True
