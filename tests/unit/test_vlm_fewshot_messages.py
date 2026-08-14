"""Tests for in-context few-shot image support in the Stage-1 message builder."""

import json
import logging
from pathlib import Path

import pytest

from seedlearn.pipeline.config import VLMConfig
from seedlearn.pipeline.stages.morphology import MorphologyStage
from seedlearn.pipeline.vlm_client import build_messages, load_examples


def test_no_examples_is_unchanged():
    # The no-examples path must stay identical: [system, user].
    msgs = build_messages("SYS", ["/a.jpg", "/b.jpg"], image_mode="file")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert len(msgs[1]["content"]) == 2
    assert all(c["type"] == "image_url" for c in msgs[1]["content"])


def test_examples_interleave_user_assistant_turns():
    examples = [
        {"images": ["/ex_entire.png"], "answer": "7. Leaf margin: entire"},
        {"images": ["/ex_toothed.png"], "answer": "7. Leaf margin: toothed"},
    ]
    msgs = build_messages("SYS", ["/spec1.jpg", "/spec2.jpg"], image_mode="file", examples=examples)
    # system, (user,assistant) x2, user
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "7. Leaf margin: entire"
    # exemplar image blocks use the same file:// scheme as specimen images.
    assert msgs[1]["content"][0]["image_url"]["url"].startswith("file://")
    assert msgs[-1]["content"][0]["image_url"]["url"].startswith("file://")
    # the real specimen query is last and carries both specimen views.
    assert len(msgs[-1]["content"]) == 2


def test_reference_turn_carries_intro_text_before_images():
    # A reference-illustration turn: intro text first, then the chart images.
    examples = [{
        "text": "REFERENCE ILLUSTRATIONS (not the specimen):",
        "images": ["/chart1.png", "/chart2.png"],
        "answer": "Understood. I will complete the form.",
    }]
    msgs = build_messages("SYS", ["/spec.jpg"], image_mode="file", examples=examples)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    ref_user = msgs[1]["content"]
    assert ref_user[0] == {"type": "text", "text": "REFERENCE ILLUSTRATIONS (not the specimen):"}
    assert all(c["type"] == "image_url" for c in ref_user[1:])
    assert len(ref_user) == 3  # text + 2 chart images


def test_real_leaf_margin_examples_manifest_loads_and_builds():
    # The committed reference manifest must load and produce a valid message array.
    manifest = Path(__file__).resolve().parents[2] / "configs/experiments/leaf_margin_examples.json"
    examples = load_examples(manifest)
    assert len(examples) == 1 and len(examples[0]["images"]) == 3
    assert examples[0]["text"] and "REFERENCE" in examples[0]["text"].upper()
    msgs = build_messages("SYS", ["/spec.jpg"], image_mode="file", examples=examples)
    # system, reference user turn, assistant ack, specimen user turn
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


def test_load_examples_list_and_dict_forms(tmp_path):
    list_form = tmp_path / "list.json"
    list_form.write_text(json.dumps([{"images": ["/a.png"], "answer": "entire"}]))
    dict_form = tmp_path / "dict.json"
    dict_form.write_text(json.dumps({"0": {"img_list": ["/a.png"], "target": "entire"}}))
    for path in (list_form, dict_form):
        loaded = load_examples(path)
        assert loaded == [{"images": ["/a.png"], "answer": "entire"}]


def test_load_examples_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"images": ["/a.png"]}]))  # no answer/target
    with pytest.raises(ValueError):
        load_examples(bad)


def test_stage_loads_examples_from_config(tmp_path):
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps([{"images": ["/e.png"], "answer": "7. Leaf margin: entire"}]))
    stage = MorphologyStage(VLMConfig(examples_file=str(ex)))
    assert stage._examples == [{"images": ["/e.png"], "answer": "7. Leaf margin: entire"}]
    # Default config loads no examples.
    assert MorphologyStage(VLMConfig())._examples is None


def test_image_budget_warns_when_exceeded(caplog, tmp_path):
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps([{"images": [f"/e{i}.png"], "answer": "x"} for i in range(3)]))
    stage = MorphologyStage(VLMConfig(examples_file=str(ex), max_images=10))
    with caplog.at_level(logging.WARNING):
        report = stage._delivery_report([f"/v{i}.jpg" for i in range(8)], "prompt")  # 8 + 3 = 11 > 10
    assert "image budget exceeded" in caplog.text
    assert report["images_over_budget"] is True
    assert report["n_exemplar_images"] == 3 and report["n_specimen_images"] == 8
