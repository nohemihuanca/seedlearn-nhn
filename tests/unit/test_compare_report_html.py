"""Tests for the comparison-report HTML (drill-down cells + prompt modal payload)."""

import importlib.util
import json
from pathlib import Path

from seedlearn.benchmarking.human.experiment_compare import (
    AxisMetric,
    Ceiling,
    ConditionMetrics,
    PromptInfo,
    SpecimenCell,
)
from seedlearn.benchmarking.human.value_map import MISSING

_SPEC = importlib.util.spec_from_file_location(
    "compare_trait_experiments",
    Path(__file__).resolve().parents[2] / "scripts" / "compare_trait_experiments.py",
)
cmp_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cmp_mod)


def _cell(sid, model_raw, model_canonical, dropped, roni_c, counted, agree):
    return SpecimenCell(
        specimen_id=sid, model_raw=model_raw, model_canonical=model_canonical,
        model_dropped=dropped, roni_views=["entero"], roni_canonical=roni_c,
        carmen_views=[], carmen_canonical=MISSING, counted=counted, agree=agree,
    )


def _condition(label, external, prompt):
    cells = {
        "model_vs_roni": [
            _cell("s_drop", "entire <to> toothed", MISSING, True, "entire", False, False),
            _cell("s_ok", "entire", "entire", False, "entire", True, True),
        ],
        "model_vs_carmen": [],
        "roni_vs_carmen": [_cell("s_ok", None, MISSING, False, "entire", True, True)],
    }
    return ConditionMetrics(
        label=label, model="Qwen", granularity="", external=external,
        n_model_specimens=2,
        vs_roni=AxisMetric(0.5, 0.4, 2), vs_carmen=AxisMetric(None, None, 0),
        stri_accuracy=None, stri_n=0, cells=cells, prompt=prompt,
    )


def _write(tmp_path, metrics):
    out = tmp_path / "report.html"
    cmp_mod.write_html(out, metrics, Ceiling(0.93, 0.79, 112), [])
    return out.read_text()


def test_cells_are_clickable_with_data_key(tmp_path):
    local = _condition("C0", False, PromptInfo("sys4", "Qwen", None, False, "PROMPT-TEXT", None))
    html = _write(tmp_path, [local])
    assert "data-key" in html
    assert "C0|model_vs_roni" in html
    assert "C0|model_vs_carmen" in html
    assert "ceiling|roni_vs_carmen" in html  # ceiling banner drill-down


def test_dropped_row_present_in_payload(tmp_path):
    local = _condition("C0", False, PromptInfo("sys4", "Qwen", None, False, "X", None))
    html = _write(tmp_path, [local])
    payload = json.loads(_extract(html, "cells"))
    rows = payload["C0|model_vs_roni"]
    dropped = [r for r in rows if r["md"]]
    assert dropped and dropped[0]["id"] == "s_drop" and dropped[0]["mc"] == MISSING


def test_local_prompt_text_and_external_marker(tmp_path):
    local = _condition("C0", False, PromptInfo("sys4", "Qwen", None, False, "LOCAL-PROMPT", None))
    ext = _condition("K1", True,
                     PromptInfo("all_traits", "gpt-5.4", None, True, None, "as-run cloud prompt not reconstructable"))
    html = _write(tmp_path, [local, ext])
    prompts = json.loads(_extract(html, "prompts"))
    assert prompts["C0"]["text"] == "LOCAL-PROMPT"
    assert prompts["K1"]["text"] is None
    assert "unreconstructable" in prompts["K1"]["unavailable_reason"] or \
           prompts["K1"]["unavailable_reason"]


def test_html_special_chars_escaped_in_cell_markup(tmp_path):
    # The raw value contains angle brackets; they must not appear raw in the JSON
    # script block in a way that closes the <script> tag.
    local = _condition("C0", False, PromptInfo("sys4", "Qwen", None, False, "X", None))
    html = _write(tmp_path, [local])
    # The embedded JSON must neutralize any </ sequence.
    assert "</script" not in _extract(html, "cells")


def _extract(html: str, script_id: str) -> str:
    """Pull the text content of an inline <script id=...> JSON block."""
    marker = f'id="{script_id}">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return html[start:end].replace("<\\/", "</")
