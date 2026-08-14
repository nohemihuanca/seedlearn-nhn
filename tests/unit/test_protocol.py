"""Tests for pipeline stage protocol."""

from __future__ import annotations

from seedlearn.pipeline.protocol import PipelineStage, StageResult


class TestStageResult:
    """Tests for the StageResult dataclass."""

    def test_create_success(self):
        result = StageResult(stage_name="test", data={"key": "value"})
        assert result.stage_name == "test"
        assert result.data == {"key": "value"}
        assert result.skipped is False
        assert result.error is None
        assert result.elapsed_ms >= 0.0

    def test_create_skipped(self):
        result = StageResult(stage_name="test", data={}, skipped=True)
        assert result.skipped is True

    def test_create_with_error(self):
        result = StageResult(stage_name="test", data={}, error="something broke")
        assert result.error == "something broke"


class TestPipelineStageProtocol:
    """Verify that a concrete class satisfying PipelineStage works."""

    def test_concrete_stage_satisfies_protocol(self):
        """A class with run/skip/validate_input/name satisfies PipelineStage."""

        class DummyStage:
            @property
            def name(self) -> str:
                return "dummy"

            def validate_input(self, context: dict) -> list[str]:
                return []

            def run(self, context: dict) -> StageResult:
                return StageResult(stage_name=self.name, data={"ok": True})

            def skip(self, context: dict) -> StageResult:
                return StageResult(stage_name=self.name, data={}, skipped=True)

        stage = DummyStage()
        assert isinstance(stage, PipelineStage)

    def test_missing_method_fails_protocol(self):
        """A class missing 'run' does NOT satisfy PipelineStage."""

        class BrokenStage:
            @property
            def name(self) -> str:
                return "broken"

        stage = BrokenStage()
        assert not isinstance(stage, PipelineStage)
