"""Tests for PipelineResult."""

import json

from seedlearn.pipeline.protocol import StageResult
from seedlearn.pipeline.result import PipelineResult


class TestPipelineResult:
    def test_create_empty(self) -> None:
        result = PipelineResult(specimen_id="test_specimen", image_paths=["/a.jpg"])
        assert result.specimen_id == "test_specimen"
        assert result.image_paths == ["/a.jpg"]
        assert result.stage_results == {}
        assert result.total_elapsed_ms == 0.0

    def test_add_stage_result(self) -> None:
        result = PipelineResult(specimen_id="test", image_paths=["/a.jpg"])
        stage = StageResult(
            stage_name="morphology", data={"traits": {}}, elapsed_ms=100.0
        )
        result.add_stage_result(stage)
        assert "morphology" in result.stage_results
        assert result.stage_results["morphology"].elapsed_ms == 100.0

    def test_total_elapsed(self) -> None:
        result = PipelineResult(specimen_id="test", image_paths=["/a.jpg"])
        result.add_stage_result(StageResult("s1", {}, elapsed_ms=100.0))
        result.add_stage_result(StageResult("s2", {}, elapsed_ms=200.0))
        assert result.total_elapsed_ms == 300.0

    def test_to_dict_roundtrip(self) -> None:
        result = PipelineResult(
            specimen_id="test", image_paths=["/a.jpg", "/b.jpg"]
        )
        result.add_stage_result(
            StageResult(
                "morphology",
                {"traits": {"leaf_shape": "elliptic"}},
                elapsed_ms=50.0,
            )
        )
        result.add_stage_result(
            StageResult(
                "classification",
                {"predictions": [{"family": "Fabaceae"}]},
                elapsed_ms=30.0,
            )
        )
        d = result.to_dict()
        assert d["specimen_id"] == "test"
        assert len(d["stages"]) == 2
        assert d["stages"]["morphology"]["data"]["traits"]["leaf_shape"] == "elliptic"
        assert d["total_elapsed_ms"] == 80.0
        # Verify JSON-serializable
        json.dumps(d)

    def test_get_stage_data(self) -> None:
        result = PipelineResult(specimen_id="test", image_paths=[])
        result.add_stage_result(StageResult("morphology", {"traits": {"a": 1}}))
        assert result.get_stage_data("morphology") == {"traits": {"a": 1}}
        assert result.get_stage_data("missing") == {}

    def test_skipped_stages(self) -> None:
        result = PipelineResult(specimen_id="test", image_paths=[])
        result.add_stage_result(StageResult("morphology", {}, skipped=True))
        result.add_stage_result(StageResult("classification", {"ok": True}))
        d = result.to_dict()
        assert d["stages"]["morphology"]["skipped"] is True
        assert d["stages"]["classification"]["skipped"] is False
