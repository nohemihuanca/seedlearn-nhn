"""Tests for user-configurable prompt loading."""

from __future__ import annotations

import logging
from pathlib import Path

from seedlearn.pipeline.config import (
    PipelineConfig,
    PromptsConfig,
    load_config,
    load_prompt,
)


class TestLoadPrompt:
    """Tests for the load_prompt utility function."""

    def test_returns_fallback_when_path_is_none(self):
        result = load_prompt(None, fallback="default prompt")
        assert result == "default prompt"

    def test_loads_file_content(self, tmp_path: Path):
        prompt_file = tmp_path / "custom.txt"
        prompt_file.write_text("Custom prompt text here")
        result = load_prompt(str(prompt_file), fallback="default")
        assert result == "Custom prompt text here"

    def test_falls_back_on_missing_file(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = load_prompt("/nonexistent/prompt.txt", fallback="fallback")
        assert result == "fallback"
        assert "not found" in caplog.text

    def test_falls_back_on_empty_file(self, tmp_path: Path, caplog):
        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("   ")
        with caplog.at_level(logging.WARNING):
            result = load_prompt(str(prompt_file), fallback="fallback")
        assert result == "fallback"
        assert "empty" in caplog.text

    def test_strips_whitespace(self, tmp_path: Path):
        prompt_file = tmp_path / "padded.txt"
        prompt_file.write_text("\n  prompt content  \n\n")
        result = load_prompt(str(prompt_file), fallback="default")
        assert result == "prompt content"

    def test_preserves_internal_newlines(self, tmp_path: Path):
        prompt_file = tmp_path / "multi.txt"
        prompt_file.write_text("line one\nline two\nline three")
        result = load_prompt(str(prompt_file), fallback="default")
        assert "line one\nline two\nline three" == result

    def test_template_variables_in_file(self, tmp_path: Path):
        prompt_file = tmp_path / "template.txt"
        prompt_file.write_text("Seedling with {traits}")
        result = load_prompt(str(prompt_file), fallback="default")
        # Template variables should be preserved (substituted later)
        assert "{traits}" in result
        # Verify substitution works
        assert result.format(traits="simple leaves") == "Seedling with simple leaves"


class TestPromptsConfig:
    """Tests for the PromptsConfig dataclass."""

    def test_defaults_are_none(self):
        cfg = PromptsConfig()
        assert cfg.morphology is None
        assert cfg.rag_query is None
        assert cfg.reasoning is None

    def test_pipeline_config_includes_prompts(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.prompts, PromptsConfig)
        assert cfg.prompts.morphology is None

    def test_from_dict_with_prompts(self):
        data = {
            "prompts": {
                "morphology": "configs/prompts/stage1_morphology.txt",
                "reasoning": "configs/prompts/stage5_reasoning.txt",
            }
        }
        cfg = PipelineConfig.from_dict(data)
        assert cfg.prompts.morphology == "configs/prompts/stage1_morphology.txt"
        assert cfg.prompts.reasoning == "configs/prompts/stage5_reasoning.txt"
        assert cfg.prompts.rag_query is None

    def test_load_config_with_prompts_yaml(self, tmp_path: Path):
        import yaml

        yaml_content = {
            "prompts": {
                "morphology": "my/custom/prompt.txt",
            }
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        cfg = load_config(yaml_path)
        assert cfg.prompts.morphology == "my/custom/prompt.txt"

    def test_cli_override_prompts(self):
        overrides = {"prompts.morphology": "override/prompt.txt"}
        cfg = load_config(None, overrides=overrides)
        assert cfg.prompts.morphology == "override/prompt.txt"


class TestStagePromptIntegration:
    """Tests that stages use custom prompt files when provided."""

    def test_morphology_stage_with_custom_prompt(self, tmp_path: Path):
        from seedlearn.pipeline.config import VLMConfig
        from seedlearn.pipeline.stages.morphology import MorphologyStage

        prompt_file = tmp_path / "custom_morph.txt"
        prompt_file.write_text("Custom morphology prompt")

        stage = MorphologyStage(VLMConfig(), prompt_file=str(prompt_file))
        assert stage._prompt_file == str(prompt_file)

    def test_morphology_stage_without_prompt_file(self):
        from seedlearn.pipeline.config import VLMConfig
        from seedlearn.pipeline.stages.morphology import MorphologyStage

        stage = MorphologyStage(VLMConfig())
        assert stage._prompt_file is None

    def test_trait_retrieval_with_custom_template(self, tmp_path: Path):
        from seedlearn.pipeline.stages.trait_retrieval import _compose_query

        traits = {"leaf complexity": "compound", "stipules": "present"}

        # Default template
        default = _compose_query(traits)
        assert default.startswith("Tropical tree seedling with ")

        # Custom template
        custom = _compose_query(traits, template="Plant showing {traits}")
        assert custom.startswith("Plant showing ")
        assert "compound" in custom

    def test_trait_retrieval_custom_fallback(self):
        from seedlearn.pipeline.stages.trait_retrieval import _compose_query

        result = _compose_query({}, fallback="unknown plant")
        assert result == "unknown plant"

    def test_reasoning_stage_with_custom_prompt(self, tmp_path: Path):
        from seedlearn.pipeline.config import ReasoningConfig
        from seedlearn.pipeline.stages.reasoning import ReasoningStage

        prompt_file = tmp_path / "custom_reason.txt"
        prompt_file.write_text("Custom reasoning prompt")

        stage = ReasoningStage(ReasoningConfig(), prompt_file=str(prompt_file))
        assert stage._prompt_file == str(prompt_file)
