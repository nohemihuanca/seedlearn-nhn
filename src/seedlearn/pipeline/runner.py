"""Pipeline orchestrator that sequences stages and accumulates results."""

from __future__ import annotations

import logging
import time
from typing import Any

from seedlearn.pipeline.config import PipelineConfig
from seedlearn.pipeline.protocol import PipelineStage, StageResult
from seedlearn.pipeline.result import PipelineResult

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrator that runs pipeline stages in order.

    The runner sequences stages, handles skip logic, validates inputs,
    accumulates context between stages, and produces a PipelineResult.

    Stage construction is NOT handled here -- callers inject stages via
    the ``stages`` attribute.

    Args:
        config: Pipeline configuration (used for skip_stages list).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self.stages: list[PipelineStage] = []

    def run(self, specimen_id: str, image_paths: list[str]) -> PipelineResult:
        """Execute all stages in order, accumulating results.

        Args:
            specimen_id: Identifier for the specimen being classified.
            image_paths: List of image file paths for this specimen.

        Returns:
            PipelineResult with per-stage results and timing.
        """
        context: dict[str, Any] = {
            "specimen_id": specimen_id,
            "image_paths": image_paths,
        }
        pipeline_result = PipelineResult(
            specimen_id=specimen_id, image_paths=image_paths
        )

        for stage in self.stages:
            stage_result = self._execute_stage(stage, context)
            pipeline_result.add_stage_result(stage_result)

            # Accumulate successful stage data into context for downstream stages
            if not stage_result.skipped and stage_result.error is None:
                context[stage.name] = stage_result.data

        return pipeline_result

    def _execute_stage(
        self, stage: PipelineStage, context: dict[str, Any]
    ) -> StageResult:
        """Run a single stage with skip/validation/error handling.

        Args:
            stage: The pipeline stage to execute.
            context: Current accumulated pipeline context.

        Returns:
            StageResult from running, skipping, or recording an error.
        """
        if stage.name in self._config.skip_stages:
            logger.info("Skipping stage: %s", stage.name)
            return stage.skip(context)

        errors = stage.validate_input(context)
        if errors:
            error_msg = "; ".join(errors)
            logger.warning(
                "Validation failed for stage %s: %s", stage.name, error_msg
            )
            return StageResult(
                stage_name=stage.name, data={}, error=error_msg, elapsed_ms=0.0
            )

        start = time.perf_counter()
        try:
            return stage.run(context)
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception("Stage %s raised an exception", stage.name)
            return StageResult(
                stage_name=stage.name,
                data={},
                error=f"Stage {stage.name} raised an exception",
                elapsed_ms=elapsed,
            )
