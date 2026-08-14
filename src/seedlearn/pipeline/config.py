"""Pipeline configuration dataclasses with YAML loading and CLI overrides.

Hierarchy: code defaults -> config.yaml -> CLI overrides.
"""

from __future__ import annotations

import json
import logging
import types
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VLMConfig:
    """Configuration for the VLM morphological-extraction stage.

    Args:
        model: HuggingFace model identifier.
        endpoint: OpenAI-compatible API endpoint.
        prompt_style: System prompt variant to use.
        image_mode: How images are passed to the VLM ("file" or "base64").
        max_images: Maximum number of images per request.
        max_tokens: Maximum tokens in the VLM response.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability mass.
        top_k: Top-k sampling cutoff.
        min_p: Minimum probability threshold (-1.0 disables).
        examples_file: Optional path to an in-context few-shot exemplar JSON.
            When set, exemplars are prepended to each Stage-1 request. ``None``
            (the default) means no few-shot — behavior is unchanged.
    """

    model: str = "Qwen/Qwen3-VL-32B-Instruct-FP8"
    endpoint: str = "http://localhost:8000/v1"
    prompt_style: str = "sys4"
    image_mode: str = "file"
    max_images: int = 10
    max_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = -1.0
    examples_file: str | None = None


@dataclass
class ClassifierConfig:
    """Configuration for the few-shot classification stage.

    Args:
        rank: Taxonomic rank for classification.
        ranks: All taxonomic ranks to evaluate (multi-rank pipeline).
        ood_percentile: Percentile for distance-based OOD threshold calibration.
        k_shot: Number of support examples per class.
        top_k: Number of top predictions to return.
        split_seed: Random seed for train/test splits.
        device: Compute device ("cuda" or "cpu").
        catalog: Path to a species catalog CSV (None = use default).
        model_str: Model identifier for the vision backbone.
        feature_aggregation: How to aggregate features ("mean" or "max").
    """

    rank: str = "family"
    ranks: list[str] = field(default_factory=lambda: ["family", "genus", "species"])
    ood_percentile: float = 95.0
    k_shot: int = 10
    top_k: int = 5
    split_seed: int = 42
    device: str = "cuda"
    catalog: str | None = None
    model_str: str = "hf-hub:imageomics/bioclip-2"
    feature_aggregation: str = "mean"


@dataclass
class TraitRetrievalConfig:
    """Configuration for RAG-based trait retrieval.

    Args:
        enabled: Whether trait retrieval is active.
        index_path: Path to the FAISS/vector index (None = build on the fly).
        descriptions_csv: Path to trait descriptions CSV (None = use default).
        embedding_model: Sentence-transformer model for embedding queries.
        top_k: Number of passages to retrieve.
        min_similarity: Minimum cosine similarity threshold.
        cross_reference: Whether to cross-reference multiple sources.
    """

    enabled: bool = True
    index_path: str | None = None
    descriptions_csv: str | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 20
    min_similarity: float = 0.3
    cross_reference: bool = True


@dataclass
class EvidenceSynthesisConfig:
    """Configuration for evidence synthesis.

    Args:
        include_raw_traits: Include raw VLM trait text in the synthesis prompt.
        include_rag_passages: Include retrieved literature passages.
        max_rag_excerpts: Maximum number of RAG excerpts to include.
        convergence_threshold: Minimum agreement score for convergence.
    """

    include_raw_traits: bool = False
    include_rag_passages: bool = True
    max_rag_excerpts: int = 5
    convergence_threshold: float = 0.3


@dataclass
class ReasoningConfig:
    """Configuration for the LLM reasoning stage.

    Args:
        model: HuggingFace model identifier for the reasoning LLM.
        endpoint: OpenAI-compatible API endpoint.
        max_tokens: Maximum tokens in the reasoning response.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability mass.
        top_k: Top-k sampling cutoff.
    """

    model: str = "Qwen/Qwen3-VL-32B-Instruct-FP8"
    endpoint: str = "http://localhost:8000/v1"
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


@dataclass
class PromptsConfig:
    """Configuration for user-customizable prompt files.

    Each field holds a path to a text file containing the prompt for that stage.
    When ``None``, the stage uses its hardcoded default prompt.  Relative paths
    are resolved from the project root (working directory).

    Args:
        morphology: Path to Stage 1 system prompt file.
        rag_query: Path to Stage 3 RAG query template file.
        reasoning: Path to Stage 5 system prompt file.
    """

    morphology: str | None = None
    rag_query: str | None = None
    reasoning: str | None = None


def load_prompt(path: str | Path | None, fallback: str) -> str:
    """Load a prompt from a text file with graceful fallback.

    Args:
        path: Path to a prompt text file.  If ``None`` or the file does not
            exist, *fallback* is returned instead (with a warning log for
            missing files).
        fallback: Hardcoded default prompt text.

    Returns:
        Prompt string (file contents or fallback).
    """
    if path is None:
        return fallback
    prompt_path = Path(path)
    if not prompt_path.exists():
        logger.warning("Prompt file not found: %s — using hardcoded default", path)
        return fallback
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.warning("Prompt file is empty: %s — using hardcoded default", path)
        return fallback
    logger.debug("Loaded prompt from %s (%d chars)", path, len(text))
    return text


@dataclass
class OutputConfig:
    """Configuration for pipeline output.

    Args:
        format: List of output formats to produce.
        directory: Directory to write results into.
        verbose: Whether to enable verbose logging.
    """

    format: list[str] = field(default_factory=lambda: ["json", "html"])
    directory: str = "results/pipeline/"
    verbose: bool = True


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

# Mapping from field name -> sub-config class for recursive construction.
_SUB_CONFIG_CLASSES: dict[str, type] = {
    "vlm": VLMConfig,
    "classifier": ClassifierConfig,
    "trait_retrieval": TraitRetrievalConfig,
    "evidence_synthesis": EvidenceSynthesisConfig,
    "reasoning": ReasoningConfig,
    "prompts": PromptsConfig,
    "output": OutputConfig,
}


@dataclass
class PipelineConfig:
    """Top-level configuration for the seedling classification pipeline.

    Args:
        skip_stages: List of stage names to skip during execution.
        vlm: VLM morphological-extraction config.
        classifier: Few-shot classification config.
        trait_retrieval: RAG trait-retrieval config.
        evidence_synthesis: Evidence synthesis config.
        reasoning: LLM reasoning config.
        prompts: User-customizable prompt file paths.
        output: Output config.
    """

    skip_stages: list[str] = field(default_factory=list)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    trait_retrieval: TraitRetrievalConfig = field(default_factory=TraitRetrievalConfig)
    evidence_synthesis: EvidenceSynthesisConfig = field(
        default_factory=EvidenceSynthesisConfig
    )
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Construct a PipelineConfig from a (possibly partial) dictionary.

        Sub-config keys present in *data* override the corresponding defaults;
        missing keys keep their defaults.  Within each sub-config, the same
        partial-override logic applies at the field level.

        Args:
            data: Dictionary with optional keys matching PipelineConfig fields.

        Returns:
            A fully populated PipelineConfig instance.
        """
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in _SUB_CONFIG_CLASSES and f.name in data:
                sub_cls = _SUB_CONFIG_CLASSES[f.name]
                kwargs[f.name] = sub_cls(**data[f.name])
            elif f.name in data:
                kwargs[f.name] = data[f.name]
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Type-coercion helpers
# ---------------------------------------------------------------------------


def _resolve_type_hint(dc_cls: type, field_name: str) -> type:
    """Resolve the runtime type for a dataclass field.

    Uses ``typing.get_type_hints`` to evaluate stringified annotations
    safely, then unwraps ``Optional`` / ``X | None`` unions.

    Args:
        dc_cls: The dataclass class containing the field.
        field_name: Name of the field.

    Returns:
        The resolved, unwrapped type (e.g. ``str``, ``int``, ``list``).

    Raises:
        KeyError: If *field_name* is not a field of *dc_cls*.
    """
    field_names = {f.name for f in fields(dc_cls)}
    if field_name not in field_names:
        raise KeyError(
            f"'{field_name}' is not a valid field of {dc_cls.__name__}. "
            f"Valid fields: {sorted(field_names)}"
        )

    hints = get_type_hints(dc_cls)
    type_hint = hints[field_name]

    # Unwrap Optional / union-with-None (e.g. ``str | None``).
    origin = getattr(type_hint, "__origin__", None)
    if origin is types.UnionType:
        args = [a for a in type_hint.__args__ if a is not type(None)]
        type_hint = args[0] if args else str

    return type_hint


def _cast_field_value(dc_cls: type, field_name: str, raw: str) -> Any:
    """Cast a raw string value to the type expected by a dataclass field.

    Args:
        dc_cls: The dataclass class containing the field.
        field_name: Name of the field to look up.
        raw: Raw string value (typically from CLI).

    Returns:
        Value cast to the field's declared type.

    Raises:
        KeyError: If *field_name* is not a field of *dc_cls*.
    """
    type_hint = _resolve_type_hint(dc_cls, field_name)

    if type_hint is bool:
        return raw.lower() in ("true", "1", "yes")
    if type_hint is int:
        return int(raw)
    if type_hint is float:
        return float(raw)
    if type_hint is list or getattr(type_hint, "__origin__", None) is list:
        return json.loads(raw)
    return raw


# ---------------------------------------------------------------------------
# YAML loader with CLI override merging
# ---------------------------------------------------------------------------


def _apply_overrides(data: dict[str, Any], overrides: dict[str, str]) -> dict[str, Any]:
    """Merge dot-notation CLI overrides into a nested dict.

    Args:
        data: Base config dictionary (mutated in place and returned).
        overrides: Mapping of dot-notation keys to raw string values.
            For sub-config fields use ``"section.field"`` (e.g. ``"vlm.model"``).
            For top-level fields use the field name directly.

    Returns:
        The mutated *data* dictionary.

    Raises:
        KeyError: If the override key references a non-existent field.
    """
    for key, raw_value in overrides.items():
        parts = key.split(".", maxsplit=1)
        if len(parts) == 2:
            section, field_name = parts
            if section not in _SUB_CONFIG_CLASSES:
                raise KeyError(
                    f"'{section}' is not a valid config section. "
                    f"Valid sections: {sorted(_SUB_CONFIG_CLASSES)}"
                )
            value = _cast_field_value(
                _SUB_CONFIG_CLASSES[section], field_name, raw_value
            )
            data.setdefault(section, {})[field_name] = value
        else:
            value = _cast_field_value(PipelineConfig, key, raw_value)
            data[key] = value
    return data


def load_config(
    path: str | Path | None,
    overrides: dict[str, str] | None = None,
) -> PipelineConfig:
    """Load pipeline configuration from YAML with optional CLI overrides.

    Resolution order: code defaults -> YAML file -> CLI overrides.

    Args:
        path: Path to a YAML config file.  If ``None``, only code defaults
            (and any *overrides*) are used.
        overrides: Dot-notation CLI overrides, e.g.
            ``{"vlm.model": "custom/model", "classifier.rank": "species"}``.

    Returns:
        A fully populated PipelineConfig.

    Raises:
        FileNotFoundError: If *path* is not None and does not exist.
    """
    data: dict[str, Any] = {}

    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as fh:
            loaded = yaml.safe_load(fh)
            if loaded is not None:
                data = loaded

    if overrides:
        _apply_overrides(data, overrides)

    return PipelineConfig.from_dict(data)
