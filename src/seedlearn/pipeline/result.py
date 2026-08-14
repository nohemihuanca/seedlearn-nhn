"""Pipeline result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from seedlearn.pipeline.protocol import StageResult


@dataclass
class PipelineResult:
    """Accumulated result from all pipeline stages.

    Args:
        specimen_id: Identifier for the specimen being classified.
        image_paths: Input image paths for this specimen.
        stage_results: Ordered dict of stage name -> StageResult.
    """

    specimen_id: str
    image_paths: list[str]
    stage_results: dict[str, StageResult] = field(default_factory=dict)

    @property
    def total_elapsed_ms(self) -> float:
        """Total wall-clock time across all stages."""
        return sum(sr.elapsed_ms for sr in self.stage_results.values())

    def add_stage_result(self, stage_result: StageResult) -> None:
        """Add a stage result to the collection.

        Args:
            stage_result: Result from a completed stage.
        """
        self.stage_results[stage_result.stage_name] = stage_result

    def get_stage_data(self, stage_name: str) -> dict[str, Any]:
        """Get the data dict for a specific stage.

        Args:
            stage_name: Name of the stage.

        Returns:
            Stage data dict, or empty dict if stage not found.
        """
        if stage_name in self.stage_results:
            return self.stage_results[stage_name].data
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Returns:
            Nested dict with specimen info, per-stage results, and timing.
        """
        return {
            "specimen_id": self.specimen_id,
            "image_paths": self.image_paths,
            "stages": {
                name: {
                    "data": sr.data,
                    "skipped": sr.skipped,
                    "error": sr.error,
                    "elapsed_ms": sr.elapsed_ms,
                }
                for name, sr in self.stage_results.items()
            },
            "total_elapsed_ms": self.total_elapsed_ms,
        }
