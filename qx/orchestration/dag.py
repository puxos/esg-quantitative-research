from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from qx.common.types import DatasetType


@dataclass
class Task:
    id: str
    run: Union[Callable[[], dict], Callable[[Dict], dict]]  # Support both signatures
    deps: List[str] = None
    outputs: List[DatasetType] = field(default_factory=list)  # What this task produces


class DAG:
    def __init__(self, tasks: List[Task]):
        self.tasks = {t.id: t for t in tasks}
        self.context = {}  # Store task outputs for downstream tasks
        self._extract_output_metadata()  # Extract outputs from callables

    def _extract_output_metadata(self):
        """Extract output types from task callables and populate task.outputs."""
        for task_id, task in self.tasks.items():
            if hasattr(task.run, "output_types"):
                task.outputs = task.run.output_types

    def get_available_types_for_task(self, task_id: str) -> List[DatasetType]:
        """
        Get all available dataset types for a task based on its dependencies.

        Args:
            task_id: Task identifier

        Returns:
            List of DatasetTypes produced by all upstream dependencies
        """
        task = self.tasks.get(task_id)
        if not task or not task.deps:
            return []

        available = []
        for dep_id in task.deps:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.outputs:
                available.extend(dep_task.outputs)

        return available

    def validate(self) -> Dict[str, any]:
        """
        Validate contract compatibility before execution.

        Returns:
            Dict with validation results:
            {
                "is_valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "task_outputs": Dict[str, List[DatasetType]]
            }
        """
        errors = []
        warnings = []
        task_outputs = {}

        print("=" * 80)
        print("DAG CONTRACT VALIDATION")
        print("=" * 80)

        # Collect outputs for each task
        for task_id, task in self.tasks.items():
            task_outputs[task_id] = task.outputs
            if task.outputs:
                print(f"\n✓ Task: {task_id}")
                for output_type in task.outputs:
                    print(
                        f"  Output: {output_type.domain.value}/{output_type.subdomain.value}",
                        end="",
                    )
                    if output_type.subtype:
                        print(f"/{output_type.subtype}", end="")
                    if output_type.frequency:
                        print(f" (frequency={output_type.frequency.value})", end="")
                    print()
            else:
                # Task without declared outputs (might be a loader or custom task)
                if task.deps:
                    warnings.append(
                        f"Task '{task_id}' has dependencies but no declared outputs. "
                        "This is OK for loaders/custom tasks, but models/builders should declare outputs."
                    )

        # Validate models have required inputs
        print("\n" + "=" * 80)
        print("VALIDATING MODEL INPUTS")
        print("=" * 80)

        for task_id, task in self.tasks.items():
            # Check if task callable has input requirements (models only)
            if not hasattr(task.run, "input_requirements"):
                continue

            available_types = self.get_available_types_for_task(task_id)

            print(f"\n📋 Task: {task_id}")
            if task.deps:
                print(f"  Dependencies: {', '.join(task.deps)}")
            else:
                print(f"  ⚠️  No dependencies declared (may fail at runtime)")

            # Show available types from dependencies
            if available_types:
                print(f"  Available dataset types ({len(available_types)}):")
                for dt in available_types:
                    self._print_dataset_type(dt, indent=4)
            else:
                print(f"  ⚠️  No available dataset types from dependencies")

            # Validate each input requirement
            print(f"  Input requirements:")
            for req in task.run.input_requirements:
                req_name = req["name"]
                req_type = req["type"]
                req_required = req["required"]

                # Perform contract matching (same logic as BaseModel._resolve_inputs)
                matching_types = self._find_matching_types(req_type, available_types)

                if matching_types:
                    # Input satisfied
                    print(f"    ✅ {req_name} (required={req_required})")
                    self._print_dataset_type(req_type, indent=7, prefix="Pattern: ")
                    if len(matching_types) == 1:
                        print(f"       Match: ", end="")
                        self._format_dataset_type_inline(matching_types[0])
                    else:
                        print(f"       Matches ({len(matching_types)}): ", end="")
                        for i, dt in enumerate(matching_types):
                            if i > 0:
                                print(", ", end="")
                            self._format_dataset_type_inline(dt)
                        print()
                elif req_required:
                    # Required input missing - ERROR
                    error_msg = (
                        f"Task '{task_id}' requires input '{req_name}' but no upstream task provides it. "
                        f"Required type: {self._format_type_compact(req_type)}"
                    )
                    errors.append(error_msg)
                    print(f"    ❌ {req_name} (required=True) - MISSING!")
                    self._print_dataset_type(req_type, indent=7, prefix="Required: ")
                    print(
                        f"       💡 Suggestion: Add a dependency that produces this type"
                    )
                else:
                    # Optional input missing - WARNING
                    warning_msg = (
                        f"Task '{task_id}' has optional input '{req_name}' not satisfied. "
                        f"Type: {self._format_type_compact(req_type)}"
                    )
                    warnings.append(warning_msg)
                    print(f"    ⚠️  {req_name} (required=False) - not available")
                    self._print_dataset_type(req_type, indent=7, prefix="Optional: ")

        is_valid = len(errors) == 0

        print("\n" + "=" * 80)
        if is_valid:
            print("✅ VALIDATION PASSED")
            if warnings:
                print(f"   {len(warnings)} warning(s)")
        else:
            print("❌ VALIDATION FAILED")
            print(f"   {len(errors)} error(s)")
        print("=" * 80)

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")

        if errors:
            print("\nErrors:")
            for error in errors:
                print(f"  ❌ {error}")

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "task_outputs": task_outputs,
        }

    def _find_matching_types(
        self, pattern: DatasetType, available: List[DatasetType]
    ) -> List[DatasetType]:
        """
        Find dataset types that match the pattern.

        Uses same matching logic as BaseModel._resolve_inputs():
        - None in pattern = match any value
        - Specific value in pattern = must match exactly

        Args:
            pattern: DatasetType pattern with optional None values
            available: List of available DatasetTypes

        Returns:
            List of matching DatasetTypes
        """
        matches = []
        for dt in available:
            if (
                # Check domain (None = match any)
                (pattern.domain is None or dt.domain == pattern.domain)
                # Check subdomain (None = match any)
                and (pattern.subdomain is None or dt.subdomain == pattern.subdomain)
                # Check asset_class (None = match any)
                and (
                    pattern.asset_class is None or dt.asset_class == pattern.asset_class
                )
                # Check subtype (None = match any)
                and (pattern.subtype is None or dt.subtype == pattern.subtype)
                # Check region (None = match any)
                and (pattern.region is None or dt.region == pattern.region)
                # Check frequency (None = match any)
                and (pattern.frequency is None or dt.frequency == pattern.frequency)
            ):
                matches.append(dt)
        return matches

    def _format_type_compact(self, dt: DatasetType) -> str:
        """Format DatasetType as compact string for error messages."""
        parts = [dt.domain.value if dt.domain else "*"]
        if dt.asset_class:
            parts.append(dt.asset_class.value)
        if dt.subdomain:
            parts.append(dt.subdomain.value)
        if dt.subtype:
            parts.append(dt.subtype)
        if dt.region:
            parts.append(dt.region.value)
        if dt.frequency:
            parts.append(dt.frequency.value)
        return "/".join(parts)

    def _print_dataset_type(self, dt: DatasetType, indent: int = 0, prefix: str = ""):
        """Print DatasetType with nice formatting."""
        indent_str = " " * indent
        print(f"{indent_str}{prefix}{dt.domain.value if dt.domain else '*'}/", end="")
        print(f"{dt.subdomain.value if dt.subdomain else '*'}", end="")
        if dt.subtype:
            print(f"/{dt.subtype}", end="")

        details = []
        if dt.asset_class:
            details.append(f"asset={dt.asset_class.value}")
        if dt.region:
            details.append(f"region={dt.region.value}")
        if dt.frequency:
            details.append(f"freq={dt.frequency.value}")

        if details:
            print(f" ({', '.join(details)})", end="")
        print()

    def _format_dataset_type_inline(self, dt: DatasetType):
        """Format DatasetType inline (no newline)."""
        print(f"{dt.domain.value if dt.domain else '*'}/", end="")
        print(f"{dt.subdomain.value if dt.subdomain else '*'}", end="")
        if dt.subtype:
            print(f"/{dt.subtype}", end="")

    def execute(self):
        completed = set()
        while len(completed) < len(self.tasks):
            progressed = False
            for tid, t in self.tasks.items():
                if tid in completed:
                    continue
                if not t.deps or all(d in completed for d in (t.deps or [])):
                    # Check if task is a model (has input_requirements metadata)
                    # Models have both output_types and input_requirements
                    # Builders only have output_types
                    if hasattr(t.run, "input_requirements"):
                        # Model task - auto-inject available_types from dependencies
                        available_types = self.get_available_types_for_task(tid)
                        print(
                            f"[AUTO-INJECT] Task {tid}: {len(available_types)} dataset type(s)"
                        )
                        manifest = t.run(available_types=available_types)
                    else:
                        # Non-model task (builder, loader, etc.)
                        try:
                            manifest = t.run(self.context)
                        except TypeError:
                            # Function doesn't accept context argument
                            manifest = t.run()

                    print(f"[OK] Task {tid}: {manifest}")

                    # Store manifest in context for downstream tasks
                    self.context[tid] = manifest
                    completed.add(tid)
                    progressed = True
            if not progressed:
                raise RuntimeError("DAG deadlock: dependencies not satisfied")
