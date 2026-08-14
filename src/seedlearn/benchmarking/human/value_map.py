"""Bilingual trait/value mapping for human-annotation grading.

The Vision-LLM emits morphological traits with **English** categorical values
(e.g. ``whorled``, ``entire``); the human annotators recorded the same traits
with **Spanish** values (e.g. ``verticilada``, ``entero``). To compare them, both
sides are collapsed to shared *canonical tokens* (English) before grading.

Each trait is described by a :class:`TraitSpec` carrying its location in the
model output, the Spanish column header it corresponds to, the canonical value
set, the English/Spanish aliases that map onto those values, and whether the
trait is *gradable* (a clean single-value categorical) or free-text/numeric and
therefore reported descriptively rather than scored.

A value that is blank, ``N/A``, ``no claro``, ``not observed`` and similar — in
either language — collapses to :data:`MISSING` and is treated as not-comparable.
A non-missing value that matches no alias also collapses to :data:`MISSING`
(conservative: an unmapped value never produces a false match); use
:func:`unmapped_values` to surface such coverage gaps.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

#: Canonical token for a trait value that is absent / not observable / unmapped.
MISSING = "__missing__"

#: Normalized raw values (either language) that mean "not recorded / not visible".
_MISSING_SENTINELS: frozenset[str] = frozenset(
    {
        "",
        "n a",  # "n/a" and "n.a." both normalize here after punctuation removal
        "na",
        "none",
        "ninguno",
        "ninguna",
        "no aplica",
        "not applicable",
        "not visible",
        "no visible",
        "no visibles",
        "not observed",
        "no observado",
        "unclear",
        "no claro",
        "no",
        "unknown",
        "desconocido",
        "cannot determine",
        "no se puede determinar",
    }
)


def normalize(value: object) -> str:
    """Lowercase, strip accents, drop punctuation, and collapse whitespace.

    Punctuation (commas, semicolons, periods) is converted to whitespace so that
    annotator variants like ``"dentado, serrado"`` or ``"oblanceolada."`` match the
    same way as their clean forms. Bracket characters are treated the same way so
    that prompt-template placeholder residue (e.g. a model that echoes the
    ``[ ]`` field markers, emitting ``"[entire"`` or ``"[entire]"``) normalizes
    back to its bare token instead of silently failing to map.
    """
    if value is None:
        return ""
    text = str(value)
    # Strip combining accent marks (eliptica vs elíptica, lenoso vs leñoso).
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    for punct in (",", ";", ".", "/", "[", "]", "(", ")", "{", "}"):
        stripped = stripped.replace(punct, " ")
    return " ".join(stripped.lower().split())


@dataclass(frozen=True)
class TraitSpec:
    """Mapping between one model trait field and its Spanish annotation column.

    Attributes:
        key: Internal trait identifier (stable, snake_case).
        model_section: Section name in the model traits dict, or ``None`` for
            human-only traits with no model counterpart.
        model_field: Field name within ``model_section``.
        spanish_header_prefix: Normalized prefix of the human column header used
            to locate this trait's column.
        canonical_values: Allowed canonical tokens for this trait.
        aliases: Normalized raw value (English or Spanish) -> canonical token.
        gradable: Whether the trait is scored (clean single-value categorical).
    """

    key: str
    model_section: str | None
    model_field: str | None
    spanish_header_prefix: str
    canonical_values: tuple[str, ...]
    aliases: Mapping[str, str] = field(default_factory=dict)
    gradable: bool = True


def _present_absent() -> dict[str, str]:
    return {
        "present": "present",
        "absent": "absent",
        "presente": "present",
        "presentes": "present",
        "ausente": "absent",
        "ausentes": "absent",
    }


# Ordered table of every trait we map. Header prefixes are already normalized
# (accent-stripped, lowercased) to match :func:`normalize` output.
TRAIT_SPECS: tuple[TraitSpec, ...] = (
    TraitSpec(
        key="leaf_relative_position",
        model_section="leaf_arrangement",
        model_field="relative_position",
        spanish_header_prefix="posicion relativa de las hojas",
        canonical_values=("alternate", "opposite", "whorled"),
        aliases={
            "alternate": "alternate",
            "opposite": "opposite",
            "whorled": "whorled",
            "alterna": "alternate",
            "opuesta": "opposite",
            "verticilada": "whorled",
        },
    ),
    TraitSpec(
        key="leaf_spacing",
        model_section="leaf_arrangement",
        model_field="spacing",
        spanish_header_prefix="espaciamiento de hojas",
        canonical_values=("clustered", "distal", "evenly_spaced"),
        aliases={
            "clustered": "clustered",
            "distal": "distal",
            "evenly_spaced": "evenly_spaced",
            "evenly spaced": "evenly_spaced",
            "agrupada": "clustered",
            "agrupadas": "clustered",
            "distales": "distal",
        },
    ),
    TraitSpec(
        key="leaf_complexity_type",
        model_section="leaf_complexity",
        model_field="type",
        spanish_header_prefix="complejidad de la hoja",
        canonical_values=("simple", "compound"),
        aliases={
            "simple": "simple",
            "simples": "simple",
            "compound": "compound",
            "compuesta": "compound",
            "compuestas": "compound",
            "compuesto": "compound",
            "compusta": "compound",  # common misspelling
        },
    ),
    TraitSpec(
        key="compound_leaf_type",
        model_section="leaf_complexity",
        model_field="compound_type",
        spanish_header_prefix="tipo de hoja compuesta",
        canonical_values=(
            "odd-pinnate",
            "even-pinnate",
            "bipinnate",
            "trifoliate",
            "palmate",
        ),
        aliases={
            "odd-pinnate": "odd-pinnate",
            "even-pinnate": "even-pinnate",
            "bipinnate": "bipinnate",
            "trifoliate": "trifoliate",
            "palmate": "palmate",
            "imparipinnada": "odd-pinnate",
            "imparipinada": "odd-pinnate",  # one-n misspelling
            "paripinnada": "even-pinnate",
            "paripinada": "even-pinnate",
            "bipinnada": "bipinnate",
            "trifoliada": "trifoliate",
            "palmada": "palmate",
        },
    ),
    TraitSpec(
        key="num_leaflets",
        model_section="leaf_complexity",
        model_field="num_leaflets",
        spanish_header_prefix="numero de foliolos",
        canonical_values=(),
        gradable=False,
    ),
    TraitSpec(
        key="leaflet_arrangement",
        model_section="leaf_complexity",
        model_field="leaflet_arrangement",
        spanish_header_prefix="arreglo de foliolos",
        canonical_values=("opposite", "alternate", "subopposite"),
        aliases={
            "opposite": "opposite",
            "alternate": "alternate",
            "subopposite": "subopposite",
            "opuesto": "opposite",
            "opuestos": "opposite",
            "alterno": "alternate",
            "alternos": "alternate",
            "alterna": "alternate",
            "subopuesto": "subopposite",
            "subopuestos": "subopposite",
        },
    ),
    TraitSpec(
        key="leaf_margin",
        model_section="leaf_morphology",
        model_field="margin",
        spanish_header_prefix="margen de la hoja",
        canonical_values=("entire", "toothed", "lobed"),
        aliases={
            "entire": "entire",
            "toothed": "toothed",
            "lobed": "lobed",
            "serrate": "toothed",
            "dentate": "toothed",
            "crenate": "toothed",
            "serrulate": "toothed",
            "denticulate": "toothed",
            "entero": "entire",
            "dentado": "toothed",
            "serrado": "toothed",
            "crenado": "toothed",
            "dentado serrado": "toothed",
            "dentado crenado": "toothed",
            "lobulado": "lobed",
            "lobada": "lobed",
            "lobulada": "lobed",
        },
    ),
    TraitSpec(
        key="leaf_shape",
        model_section="leaf_morphology",
        model_field="shape",
        spanish_header_prefix="forma de la hoja",
        canonical_values=(
            "elliptic",
            "obovate",
            "ovate",
            "lanceolate",
            "oblanceolate",
            "oblong",
            "orbicular",
            "other",
        ),
        aliases={
            "elliptic": "elliptic",
            "obovate": "obovate",
            "ovate": "ovate",
            "lanceolate": "lanceolate",
            "oblanceolate": "oblanceolate",
            "oblong": "oblong",
            "orbicular": "orbicular",
            "other": "other",
            "eliptica": "elliptic",
            "obovada": "obovate",
            "ovada": "ovate",
            "lanceolada": "lanceolate",
            "oblanceolada": "oblanceolate",
            "oblonga": "oblong",
            "otra": "other",
            "linear": "other",
        },
    ),
    TraitSpec(
        key="leaf_apex",
        model_section="leaf_morphology",
        model_field="apex",
        spanish_header_prefix="apice de la hoja",
        canonical_values=(
            "acute",
            "obtuse",
            "acuminate",
            "rounded",
            "emarginate",
            "mucronate",
        ),
        aliases={
            "acute": "acute",
            "obtuse": "obtuse",
            "acuminate": "acuminate",
            "rounded": "rounded",
            "emarginate": "emarginate",
            "mucronate": "mucronate",
            "agudo": "acute",
            "obtuso": "obtuse",
            "acuminado": "acuminate",
            "redondeado": "rounded",
            "emarginado": "emarginate",
            "mucronado": "mucronate",
        },
    ),
    TraitSpec(
        key="leaf_base",
        model_section="leaf_morphology",
        model_field="base",
        spanish_header_prefix="base de la hoja",
        canonical_values=(
            "cuneate",
            "rounded",
            "cordate",
            "attenuate",
            "oblique",
            "truncate",
        ),
        aliases={
            "cuneate": "cuneate",
            "rounded": "rounded",
            "cordate": "cordate",
            "attenuate": "attenuate",
            "oblique": "oblique",
            "truncate": "truncate",
            "cuneada": "cuneate",
            "redondeada": "rounded",
            "cordada": "cordate",
            "atenuada": "attenuate",
            "oblicua": "oblique",
            "truncada": "truncate",
        },
    ),
    TraitSpec(
        key="venation",
        model_section="leaf_morphology",
        model_field="venation",
        spanish_header_prefix="tipo de venacion",
        canonical_values=("pinnate", "palmate", "parallel", "arcuate"),
        aliases={
            "pinnate": "pinnate",
            "palmate": "palmate",
            "parallel": "parallel",
            "arcuate": "arcuate",
            "pinnada": "pinnate",
            "palmada": "palmate",
            "paralela": "parallel",
            "arqueada": "arcuate",
        },
    ),
    TraitSpec(
        key="secondary_veins",
        model_section="leaf_morphology",
        model_field="secondary_veins",
        spanish_header_prefix="venas secundarias",
        canonical_values=(),
        gradable=False,
    ),
    TraitSpec(
        key="leaf_surface",
        model_section="leaf_morphology",
        model_field="surface_features",
        spanish_header_prefix="caracteristicas de superficie foliar",
        # Model frequently emits compound descriptions ("dull, slightly rugose");
        # reported descriptively rather than scored.
        canonical_values=(),
        gradable=False,
    ),
    TraitSpec(
        key="leaf_trichomes",
        model_section="leaf_morphology",
        model_field="trichomes",
        spanish_header_prefix="tricomas en superficie foliar",
        canonical_values=("present", "absent"),
        aliases=_present_absent(),
    ),
    TraitSpec(
        key="petiole_length",
        model_section="leaf_morphology",
        model_field="petiole_length",
        spanish_header_prefix="longitud del peciolo",
        canonical_values=("short", "medium", "long", "sessile"),
        aliases={
            "short": "short",
            "medium": "medium",
            "long": "long",
            "sessile": "sessile",
            "corto": "short",
            "medio": "medium",
            "largo": "long",
            "sesil": "sessile",
            "ausente": "sessile",  # no petiole -> sessile
        },
    ),
    TraitSpec(
        key="petiole_features",
        model_section="leaf_morphology",
        model_field="petiole_features",
        spanish_header_prefix="caracteristicas del peciolo",
        canonical_values=("winged", "grooved", "terete", "pulvinate"),
        aliases={
            "winged": "winged",
            "grooved": "grooved",
            "terete": "terete",
            "pulvinate": "pulvinate",
            "alado": "winged",
            "acanalado": "grooved",
            "pulvinado": "pulvinate",
            "terete piloso": "terete",
            "terete-piloso": "terete",
            "semiterete": "terete",
        },
    ),
    TraitSpec(
        key="stem_type",
        model_section="stem_traits",
        model_field="type",
        spanish_header_prefix="tipo de tallo",
        canonical_values=("woody", "herbaceous"),
        aliases={
            "woody": "woody",
            "herbaceous": "herbaceous",
            "lenoso": "woody",
            "herbaceo": "herbaceous",
        },
    ),
    TraitSpec(
        key="stem_trichomes",
        model_section="stem_traits",
        model_field="trichomes",
        spanish_header_prefix="tricomas en tallo",
        canonical_values=("present", "absent"),
        aliases=_present_absent(),
    ),
    TraitSpec(
        key="stem_color",
        model_section="stem_traits",
        model_field="color",
        spanish_header_prefix="color del tallo",
        canonical_values=(),
        gradable=False,
    ),
    TraitSpec(
        key="stem_texture",
        model_section="stem_traits",
        model_field="texture",
        spanish_header_prefix="textura del tallo",
        canonical_values=("smooth", "ridged", "lenticellate"),
        aliases={
            "smooth": "smooth",
            "ridged": "ridged",
            "lenticellate": "lenticellate",
            "lisa": "smooth",
            "estriada": "ridged",
            "lenticelada": "lenticellate",
            "lenticelado": "lenticellate",
        },
    ),
    TraitSpec(
        key="stipules",
        model_section="special_features",
        model_field="stipules",
        spanish_header_prefix="estipulas",
        canonical_values=("present", "absent"),
        aliases=_present_absent(),
    ),
    TraitSpec(
        key="latex",
        model_section="special_features",
        model_field="latex",
        spanish_header_prefix="latex",
        canonical_values=("present", "absent"),
        # "no observado" / "not observed" fall through to MISSING via sentinels.
        aliases={
            "present": "present",
            "absent": "absent",
            "presente": "present",
            "ausente": "absent",
        },
    ),
    TraitSpec(
        key="pulvinus",
        model_section="special_features",
        model_field="pulvinus",
        spanish_header_prefix="pulvino",
        canonical_values=("present", "absent"),
        # Annotators record presence together with a position note; any such note
        # means the pulvinus is present.
        aliases={
            **_present_absent(),
            "basal": "present",
            "apical": "present",
            "ambos": "present",
            "basal y apical": "present",
            "apical y basal": "present",
            "terminal y basal": "present",
            "presente basal": "present",
            "presente apical": "present",
            "presente ambos": "present",
            "presente apical y basal": "present",
            "presente terminal y basal": "present",
        },
    ),
    TraitSpec(
        key="tendrils",
        model_section="special_features",
        model_field="tendrils",
        spanish_header_prefix="zarcillos",
        canonical_values=("present", "absent"),
        aliases=_present_absent(),
    ),
    TraitSpec(
        # Human-only: the model has no structured damage field (only free notes).
        key="damage",
        model_section=None,
        model_field=None,
        spanish_header_prefix="dano visible",
        canonical_values=("present", "absent"),
        aliases=_present_absent(),
    ),
)


def gradable_specs() -> tuple[TraitSpec, ...]:
    """Return only the specs that are scored (clean single-value categoricals)."""
    return tuple(s for s in TRAIT_SPECS if s.gradable)


def spec_for_spanish_header(header: str) -> TraitSpec | None:
    """Find the :class:`TraitSpec` whose Spanish column matches ``header``.

    Matching is prefix-based on the normalized header, so the long parenthetical
    legend text in the spreadsheet headers does not need to be reproduced exactly.
    """
    norm = normalize(header)
    if not norm:
        return None
    for spec in TRAIT_SPECS:
        if norm.startswith(spec.spanish_header_prefix):
            return spec
    return None


_NEGATION_TOKENS = frozenset({"not", "no", "non", "sin", "without"})


def _canonical_from_tokens(spec: TraitSpec, norm: str) -> str | None:
    """Resolve a compound descriptor by agreement of its recognized tokens.

    Some models emit multi-word margin/shape descriptors (e.g.
    ``"toothed, serrate"`` or ``"toothed – crenate"``) that are not a single
    alias but whose individual words all point at one canonical token. This
    resolves such a value **only** when every recognized token agrees on a single
    canonical and no negation word is present — otherwise it returns ``None`` so
    the caller falls back to :data:`MISSING`. This preserves the "never a false
    match" guarantee: ambiguous (``"entire to toothed"``) and negated
    (``"not toothed"``) values stay unresolved.

    Args:
        spec: The trait spec whose alias/canonical table to consult.
        norm: A normalized value (from :func:`normalize`).

    Returns:
        The single agreed canonical token, or ``None`` when unresolved.
    """
    tokens = norm.replace("-", " ").replace("–", " ").replace("—", " ").split()
    if any(tok in _NEGATION_TOKENS for tok in tokens):
        return None
    found: set[str] = set()
    for tok in tokens:
        if tok in spec.aliases:
            found.add(spec.aliases[tok])
        elif tok in spec.canonical_values:
            found.add(tok)
    return next(iter(found)) if len(found) == 1 else None


def to_canonical(spec: TraitSpec, raw: object) -> str:
    """Map a raw value (English model or Spanish human) to a canonical token.

    Returns :data:`MISSING` for blank/not-observed sentinels and for any
    non-missing value that matches no alias (conservative — never a false match).
    A compound descriptor whose recognized words unanimously point at one
    canonical (e.g. ``"toothed, serrate"`` -> ``"toothed"``) is resolved via
    :func:`_canonical_from_tokens`; ambiguous or negated compounds stay
    :data:`MISSING`.
    """
    norm = normalize(raw)
    if norm in _MISSING_SENTINELS:
        return MISSING
    if norm in spec.aliases:
        return spec.aliases[norm]
    if norm in spec.canonical_values:
        return norm
    resolved = _canonical_from_tokens(spec, norm)
    return resolved if resolved is not None else MISSING


def model_value(traits: Mapping[str, object], spec: TraitSpec) -> str | None:
    """Extract the raw model value for ``spec`` from a Stage-1 traits dict.

    Returns ``None`` when the trait has no model counterpart or the path is
    absent from the model output.
    """
    if spec.model_section is None or spec.model_field is None:
        return None
    section = traits.get(spec.model_section)
    if not isinstance(section, Mapping):
        return None
    value = section.get(spec.model_field)
    return None if value is None else str(value)


def unmapped_values(spec: TraitSpec, raws: Iterable[object]) -> list[str]:
    """Return distinct non-missing raw values that map to :data:`MISSING`.

    Used to surface coverage gaps (a value present in the data that the alias
    table does not recognize) so the map can be extended.
    """
    seen: dict[str, None] = {}
    for raw in raws:
        norm = normalize(raw)
        if norm in _MISSING_SENTINELS:
            continue
        if to_canonical(spec, raw) is MISSING:
            seen.setdefault(str(raw), None)
    return list(seen)
