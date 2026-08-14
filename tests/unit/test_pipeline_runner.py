"""Tests for PipelineRunner orchestrator."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from seedlearn.pipeline.config import PipelineConfig
from seedlearn.pipeline.protocol import StageResult
from seedlearn.pipeline.result import PipelineResult
from seedlearn.pipeline.runner import PipelineRunner


def _make_mock_stage(
    name: str,
    run_data: dict[str, Any] | None = None,
    run_error: str | None = None,
    validation_errors: list[str] | None = None,
) -> MagicMock:
    """Create a mock stage satisfying the PipelineStage protocol.

    Args:
        name: Stage name returned by the ``name`` property.
        run_data: Data dict returned by ``run()``.
        run_error: Error string set on the ``run()`` StageResult.
        validation_errors: Errors returned by ``validate_input()``.

    Returns:
        Configured MagicMock with name, validate_input, run, and skip.
    """
    stage = MagicMock()
    stage.name = name
    stage.validate_input.return_value = validation_errors or []
    stage.run.return_value = StageResult(
        stage_name=name,
        data=run_data or {},
        error=run_error,
        elapsed_ms=10.0,
    )
    stage.skip.return_value = StageResult(
        stage_name=name, data={}, skipped=True, elapsed_ms=0.0
    )
    return stage


class TestPipelineRunner:
    """Tests for PipelineRunner orchestration logic."""

    def test_runs_all_stages_in_order(self) -> None:
        """All 5 mock stages get ``run()`` called in insertion order."""
        config = PipelineConfig()
        runner = PipelineRunner(config)
        stages = [_make_mock_stage(f"stage_{i}") for i in range(5)]
        runner.stages = stages

        result = runner.run("spec-001", ["img1.jpg"])

        for s in stages:
            s.run.assert_called_once()
            s.skip.assert_not_called()
        assert len(result.stage_results) == 5
        assert list(result.stage_results.keys()) == [f"stage_{i}" for i in range(5)]

    def test_skips_configured_stages(self) -> None:
        """A stage listed in skip_stages gets ``skip()`` instead of ``run()``."""
        config = PipelineConfig(skip_stages=["stage_1"])
        runner = PipelineRunner(config)
        stages = [_make_mock_stage("stage_0"), _make_mock_stage("stage_1")]
        runner.stages = stages

        result = runner.run("spec-002", ["img.jpg"])

        stages[0].run.assert_called_once()
        stages[0].skip.assert_not_called()
        stages[1].skip.assert_called_once()
        stages[1].run.assert_not_called()
        assert result.stage_results["stage_1"].skipped is True

    def test_validation_failure_records_error(self) -> None:
        """Stage with validation errors records error, pipeline continues."""
        config = PipelineConfig()
        runner = PipelineRunner(config)
        bad_stage = _make_mock_stage(
            "bad", validation_errors=["Missing key: image_paths"]
        )
        good_stage = _make_mock_stage("good", run_data={"ok": True})
        runner.stages = [bad_stage, good_stage]

        result = runner.run("spec-003", ["img.jpg"])

        bad_stage.run.assert_not_called()
        good_stage.run.assert_called_once()
        assert result.stage_results["bad"].error is not None
        assert "Missing key: image_paths" in result.stage_results["bad"].error
        assert result.stage_results["good"].error is None

    def test_context_accumulates(self) -> None:
        """Stage 2 receives Stage 1 output in context."""
        config = PipelineConfig()
        runner = PipelineRunner(config)
        s1 = _make_mock_stage("s1", run_data={"traits": {"leaf": "green"}})
        s2 = _make_mock_stage("s2")
        runner.stages = [s1, s2]

        runner.run("spec-004", ["img.jpg"])

        # Stage 2's run() should have been called with context containing s1 data
        ctx_passed = s2.run.call_args[0][0]
        assert "s1" in ctx_passed
        assert ctx_passed["s1"] == {"traits": {"leaf": "green"}}

    def test_pipeline_result_serializable(self) -> None:
        """``result.to_dict()`` can be round-tripped through JSON."""
        config = PipelineConfig()
        runner = PipelineRunner(config)
        runner.stages = [
            _make_mock_stage("morphology", run_data={"raw": "response text"}),
        ]

        result = runner.run("spec-005", ["a.jpg", "b.jpg"])
        serialized = json.dumps(result.to_dict())
        parsed = json.loads(serialized)

        assert parsed["specimen_id"] == "spec-005"
        assert "morphology" in parsed["stages"]

    def test_stage_error_continues(self) -> None:
        """If a stage returns an error, pipeline records it and continues."""
        config = PipelineConfig()
        runner = PipelineRunner(config)
        err_stage = _make_mock_stage("err", run_error="VLM timeout")
        ok_stage = _make_mock_stage("ok", run_data={"result": 42})
        runner.stages = [err_stage, ok_stage]

        result = runner.run("spec-006", ["img.jpg"])

        assert result.stage_results["err"].error == "VLM timeout"
        ok_stage.run.assert_called_once()
        assert result.stage_results["ok"].error is None

    def test_empty_pipeline(self) -> None:
        """Runner with no stages returns empty PipelineResult."""
        config = PipelineConfig()
        runner = PipelineRunner(config)
        runner.stages = []

        result = runner.run("spec-007", ["img.jpg"])

        assert isinstance(result, PipelineResult)
        assert result.specimen_id == "spec-007"
        assert result.stage_results == {}
        assert result.total_elapsed_ms == 0.0
