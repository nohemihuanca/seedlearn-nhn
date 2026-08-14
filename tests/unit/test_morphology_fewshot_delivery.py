"""Tests for C4 few-shot image + prompt delivery evidence (morphology stage)."""

from unittest.mock import MagicMock, patch

from seedlearn.pipeline.config import VLMConfig
from seedlearn.pipeline.stages.morphology import MorphologyStage
from seedlearn.pipeline.vlm_client import load_examples


def _mock_client():
    inst = MagicMock()
    inst.chat.return_value = MagicMock(
        content="7. Leaf margin: entire",
        raw_content="7. Leaf margin: entire",
        thinking=None,
        processing_time_ms=10.0,
    )
    return inst


def _run(stage, image_paths):
    stage.client = _mock_client()
    return stage.run({"image_paths": image_paths})


def test_delivery_recorded_with_examples():
    stage = MorphologyStage(config=VLMConfig(prompt_style="margin_rich"))
    stage._examples = [{"images": ["a.jpg", "b.jpg", "c.jpg"], "answer": "entire"}]
    d = _run(stage, ["/s1.jpg", "/s2.jpg"]).data["few_shot_delivery"]
    assert d["n_examples"] == 1
    assert d["n_exemplar_images"] == 3
    assert d["n_specimen_images"] == 2
    assert d["prompt_chars"] > 0
    assert d["images_over_budget"] is False  # 2 + 3 = 5 <= default max_images 10


def test_over_budget_is_flagged():
    # Tight budget so specimen + exemplar images overflow -> at risk of dropped views.
    stage = MorphologyStage(config=VLMConfig(prompt_style="margin_rich", max_images=4))
    stage._examples = [{"images": ["a.jpg", "b.jpg", "c.jpg"], "answer": "entire"}]
    d = _run(stage, ["/s1.jpg", "/s2.jpg"]).data["few_shot_delivery"]
    assert d["images_over_budget"] is True  # 2 + 3 = 5 > 4


def test_no_examples_records_zero():
    stage = MorphologyStage(config=VLMConfig())  # no examples_file
    d = _run(stage, ["/s1.jpg"]).data["few_shot_delivery"]
    assert d["n_examples"] == 0
    assert d["n_exemplar_images"] == 0
    assert d["images_over_budget"] is False


def test_real_c4_examples_file_loads_three_images():
    # Integration: the actual C4 exemplar file loaded through the real loader.
    examples = load_examples("configs/experiments/leaf_margin_examples.json")
    stage = MorphologyStage(config=VLMConfig(prompt_style="margin_rich"))
    stage._examples = examples
    d = _run(stage, ["/s1.jpg", "/s2.jpg", "/s3.jpg"]).data["few_shot_delivery"]
    n_imgs = sum(len(ex["images"]) for ex in examples)
    assert d["n_exemplar_images"] == n_imgs == 3
    assert d["images_over_budget"] is False  # 3 + 3 = 6 <= 10
