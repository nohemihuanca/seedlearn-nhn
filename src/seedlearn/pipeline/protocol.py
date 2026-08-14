"""Pipeline stage protocol and base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class StageResult:
    """Result from a single pipeline stage.

    Args:
        stage_name: Name of the stage that produced this result.
        data: Stage-specific output data.
        skipped: Whether the stage was skipped.
        error: Error message if the stage failed.
        elapsed_ms: Wall-clock time for stage execution in milliseconds.
    """

    stage_name: str
    data: dict[str, Any]
    skipped: bool = False
    error: str | None = None
    elapsed_ms: float = 0.0


@runtime_checkable
class PipelineStage(Protocol):
    """Protocol that all pipeline stages must satisfy.

    Every stage can run, skip, and validate its required inputs.
    """

    @property
    def name(self) -> str:
        """Short identifier for this stage (e.g. 'morphology', 'classification')."""
        ...

    def validate_input(self, context: dict[str, Any]) -> list[str]:
        """Check that required inputs exist in context.

        Args:
            context: Accumulated pipeline state from prior stages.

        Returns:
            List of error messages. Empty list means valid.
        """
        ...

    def run(self, context: dict[str, Any]) -> StageResult:
        """Execute this stage.

        Args:
            context: Accumulated pipeline state from prior stages.

        Returns:
            StageResult with stage output data.
        """
        ...

    def skip(self, context: dict[str, Any]) -> StageResult:
        """Return a skipped result (no-op).

        Args:
            context: Accumulated pipeline state (unused).

        Returns:
            StageResult with skipped=True and empty data.
        """
        ...
