"""Tests for pipeline configuration."""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from seedlearn.pipeline.config import (
    PipelineConfig,
    VLMConfig,
    ClassifierConfig,
    TraitRetrievalConfig,
    EvidenceSynthesisConfig,
    ReasoningConfig,
    OutputConfig,
    load_config,
)


class TestConfigDefaults:
    """Verify every config dataclass has the expected defaults."""

    def test_vlm_config_defaults(self):
        cfg = VLMConfig()
        assert cfg.model == "Qwen/Qwen3-VL-32B-Instruct-FP8"
        assert cfg.endpoint == "http://localhost:8000/v1"
        assert cfg.prompt_style == "sys4"
        assert cfg.image_mode == "file"
        assert cfg.max_images == 10
        assert cfg.max_tokens == 8192
        assert cfg.temperature == 0.6

    def test_classifier_config_defaults(self):
        cfg = ClassifierConfig()
        assert cfg.rank == "family"
        assert cfg.k_shot == 10
        assert cfg.top_k == 5
        assert cfg.model_str == "hf-hub:imageomics/bioclip-2"
        assert cfg.feature_aggregation == "mean"

    def test_trait_retrieval_config_defaults(self):
        cfg = TraitRetrievalConfig()
        assert cfg.enabled is True
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.top_k == 20

    def test_evidence_synthesis_config_defaults(self):
        cfg = EvidenceSynthesisConfig()
        assert cfg.include_rag_passages is True
        assert cfg.max_rag_excerpts == 5

    def test_reasoning_config_defaults(self):
        cfg = ReasoningConfig()
        assert cfg.model == "Qwen/Qwen3-VL-32B-Instruct-FP8"
        assert cfg.max_tokens == 4096

    def test_output_config_defaults(self):
        cfg = OutputConfig()
        assert cfg.format == ["json", "html"]
        assert cfg.directory == "results/pipeline/"
        assert cfg.verbose is True

    def test_pipeline_config_defaults(self):
        cfg = PipelineConfig()
        assert cfg.skip_stages == []
        assert isinstance(cfg.vlm, VLMConfig)
        assert isinstance(cfg.classifier, ClassifierConfig)
        assert isinstance(cfg.trait_retrieval, TraitRetrievalConfig)
        assert isinstance(cfg.evidence_synthesis, EvidenceSynthesisConfig)
        assert isinstance(cfg.reasoning, ReasoningConfig)
        assert isinstance(cfg.output, OutputConfig)

    def test_pipeline_config_from_dict(self):
        data = {"vlm": {"model": "custom/model"}, "classifier": {"rank": "species"}}
        cfg = PipelineConfig.from_dict(data)
        assert cfg.vlm.model == "custom/model"
        assert cfg.classifier.rank == "species"
        # Other fields keep defaults
        assert cfg.vlm.prompt_style == "sys4"
        assert cfg.classifier.k_shot == 10

    def test_pipeline_config_from_empty_dict(self):
        cfg = PipelineConfig.from_dict({})
        assert cfg.vlm.model == "Qwen/Qwen3-VL-32B-Instruct-FP8"
        assert cfg.classifier.rank == "family"


def test_classifier_config_has_multirank_fields():
    from seedlearn.pipeline.config import ClassifierConfig

    cfg = ClassifierConfig()
    assert hasattr(cfg, "ranks")
    assert cfg.ranks == ["family", "genus", "species"]
    assert hasattr(cfg, "ood_percentile")
    assert cfg.ood_percentile == 95.0


class TestLoadConfig:
    """Tests for YAML loading and CLI override merging."""

    def test_load_from_yaml(self, tmp_path: Path):
        yaml_content = {
            "vlm": {"model": "test/model", "prompt_style": "sys2"},
            "classifier": {"rank": "genus"},
        }
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        cfg = load_config(yaml_path)
        assert cfg.vlm.model == "test/model"
        assert cfg.vlm.prompt_style == "sys2"
        assert cfg.classifier.rank == "genus"
        # Defaults preserved
        assert cfg.vlm.max_tokens == 8192

    def test_load_default_when_no_file(self):
        cfg = load_config(None)
        assert isinstance(cfg, PipelineConfig)
        assert cfg.vlm.model == "Qwen/Qwen3-VL-32B-Instruct-FP8"

    def test_cli_overrides(self, tmp_path: Path):
        yaml_path = tmp_path / "base.yaml"
        yaml_path.write_text(yaml.dump({"vlm": {"model": "base/model"}}))

        overrides = {"vlm.model": "cli/model", "classifier.rank": "species"}
        cfg = load_config(yaml_path, overrides=overrides)
        assert cfg.vlm.model == "cli/model"
        assert cfg.classifier.rank == "species"

    def test_cli_overrides_without_yaml(self):
        overrides = {"vlm.temperature": "0.8", "reasoning.max_tokens": "2048"}
        cfg = load_config(None, overrides=overrides)
        assert cfg.vlm.temperature == 0.8
        assert cfg.reasoning.max_tokens == 2048

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/path.yaml"))

    def test_cli_override_invalid_key_raises(self):
        with pytest.raises(KeyError):
            load_config(None, overrides={"vlm.nonexistent_field": "value"})

    def test_cli_override_top_level_key(self):
        overrides = {"skip_stages": '["classification"]'}
        cfg = load_config(None, overrides=overrides)
        assert cfg.skip_stages == ["classification"]
