"""Load human trait annotations and join them to specimens and true taxonomy.

The two annotators (Roni, Carmen) recorded morphological traits **per view** in
Spanish spreadsheets, using blinded ``anonymous_id`` labels. The private curator
key maps each ``anonymous_id`` to a real ``specimen_id`` and true family / genus /
species; the image key maps each view to its original image (provenance).

This module reads the spreadsheets into typed per-view :class:`AnnotationRecord`
objects and joins them to specimens via the curator key. Spreadsheet I/O is kept
thin (``openpyxl`` primary, stdlib fallback) so the parse/join logic is pure and
unit-testable without a real ``.xlsx`` file.
"""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from seedlearn.benchmarking.human.value_map import normalize, spec_for_spanish_header

# Header tokens (normalized) for the non-trait identity columns.
_ID_COL = "identificador_unico"
_VIEW_COL = "vista"


@dataclass
class AnnotationRecord:
    """One annotator's trait observations for a single specimen view."""

    annotator: str
    anonymous_id: str
    view_id: str
    traits: dict[str, str] = field(default_factory=dict)
    id_family: str | None = None
    id_genus: str | None = None
    id_species: str | None = None
    # Filled by :func:`join_specimens`.
    specimen_id: str | None = None
    true_family: str | None = None
    true_genus: str | None = None
    true_species: str | None = None


@dataclass(frozen=True)
class CuratorEntry:
    """A row of the curator taxonomic key (anonymous_id -> real specimen)."""

    anonymous_id: str
    specimen_id: str
    family: str
    genus: str
    species: str  # epithet


# --------------------------------------------------------------------------- #
# Spreadsheet I/O (thin; openpyxl primary, stdlib fallback)
# --------------------------------------------------------------------------- #

_SSML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _col_index(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _read_xlsx_stdlib(path: str | Path, sheet: int = 0) -> list[list[str]]:
    """Read a worksheet to a list of string rows using only the stdlib."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{_SSML}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{_SSML}t")))
        sheets = sorted(n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml", n))
        root = ET.fromstring(zf.read(sheets[sheet]))
        rows: list[list[str]] = []
        for row in root.iter(f"{_SSML}row"):
            cells: dict[int, str] = {}
            for c in row.findall(f"{_SSML}c"):
                v = c.find(f"{_SSML}v")
                if v is not None and v.text is not None:
                    cells[_col_index(c.get("r"))] = (
                        shared[int(v.text)] if c.get("t") == "s" else v.text
                    )
                else:
                    is_ = c.find(f"{_SSML}is")
                    if is_ is not None:
                        cells[_col_index(c.get("r"))] = "".join(
                            t.text or "" for t in is_.iter(f"{_SSML}t")
                        )
            width = max(cells) + 1 if cells else 0
            rows.append([cells.get(i, "") for i in range(width)])
        return rows


def read_xlsx(path: str | Path, sheet: int = 0) -> list[list[str]]:
    """Read a worksheet to string rows; prefer ``openpyxl``, fall back to stdlib."""
    try:
        import openpyxl  # noqa: PLC0415 - optional, idiomatic when installed
    except ImportError:
        return _read_xlsx_stdlib(path, sheet)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[sheet]
    rows = [
        ["" if v is None else str(v) for v in row]
        for row in ws.iter_rows(values_only=True)
    ]
    wb.close()
    return rows


# --------------------------------------------------------------------------- #
# Pure parsing / joining
# --------------------------------------------------------------------------- #


def _header_index(header: list[str]) -> dict[str, int]:
    return {normalize(h): i for i, h in enumerate(header)}


def parse_annotation_rows(
    header: list[str], rows: list[list[str]], annotator: str
) -> list[AnnotationRecord]:
    """Parse spreadsheet rows into per-view records (pure; no I/O).

    Rows with a blank ``anonymous_id`` or ``view_id`` are dropped (handles the
    trailing/blank rows present in the source sheets). Trait columns are mapped to
    internal trait keys via :func:`spec_for_spanish_header`; unrecognized columns
    (identity/quality metadata) are ignored.
    """
    idx = _header_index(header)
    id_i = idx.get(_ID_COL)
    view_i = idx.get(_VIEW_COL)
    if id_i is None or view_i is None:
        raise ValueError(
            f"{annotator}: sheet missing required '{_ID_COL}'/'{_VIEW_COL}' columns"
        )

    # Identity-prediction columns (present/blank depending on annotator).
    fam_i = next((i for h, i in idx.items() if h.startswith("familia")), None)
    gen_i = next((i for h, i in idx.items() if h.startswith("genero")), None)
    sp_i = next((i for h, i in idx.items() if h.startswith("especie")), None)

    # Pre-resolve trait columns to (col_index, trait_key).
    trait_cols: list[tuple[int, str]] = []
    for col, raw_header in enumerate(header):
        spec = spec_for_spanish_header(raw_header)
        if spec is not None:
            trait_cols.append((col, spec.key))

    def cell(row: list[str], i: int | None) -> str:
        if i is None or i >= len(row):
            return ""
        return str(row[i]).strip()

    records: list[AnnotationRecord] = []
    for row in rows:
        anon = cell(row, id_i)
        view = cell(row, view_i)
        if not anon or not view:
            continue
        traits = {key: cell(row, col) for col, key in trait_cols if cell(row, col)}
        records.append(
            AnnotationRecord(
                annotator=annotator,
                anonymous_id=anon,
                view_id=view,
                traits=traits,
                id_family=cell(row, fam_i) or None,
                id_genus=cell(row, gen_i) or None,
                id_species=cell(row, sp_i) or None,
            )
        )
    return records


def parse_curator_rows(rows: list[dict[str, str]]) -> dict[str, CuratorEntry]:
    """Build an ``anonymous_id -> CuratorEntry`` map from curator-key dict rows."""
    out: dict[str, CuratorEntry] = {}
    for row in rows:
        anon = (row.get("anonymous_id") or "").strip()
        if not anon:
            continue
        out[anon] = CuratorEntry(
            anonymous_id=anon,
            specimen_id=(row.get("individual_code") or "").strip(),
            family=(row.get("family") or "").strip(),
            genus=(row.get("genus") or "").strip(),
            species=(row.get("species") or "").strip(),
        )
    return out


def join_specimens(
    records: list[AnnotationRecord], curator: dict[str, CuratorEntry]
) -> tuple[list[AnnotationRecord], list[str]]:
    """Attach specimen_id + true taxonomy to records via the curator key.

    Returns ``(joined_records, unmatched_anonymous_ids)``. Records whose
    ``anonymous_id`` is absent from the curator key are returned unchanged (their
    ``specimen_id`` stays ``None``) and their id is collected for reporting.
    """
    unmatched: list[str] = []
    seen_unmatched: set[str] = set()
    for rec in records:
        entry = curator.get(rec.anonymous_id)
        if entry is None:
            if rec.anonymous_id not in seen_unmatched:
                seen_unmatched.add(rec.anonymous_id)
                unmatched.append(rec.anonymous_id)
            continue
        rec.specimen_id = entry.specimen_id
        rec.true_family = entry.family
        rec.true_genus = entry.genus
        rec.true_species = entry.species
    return records, unmatched


# --------------------------------------------------------------------------- #
# Convenience loaders (I/O + parse + join)
# --------------------------------------------------------------------------- #


def load_curator_key(path: str | Path) -> dict[str, CuratorEntry]:
    """Load the curator taxonomic key CSV into an ``anonymous_id`` map."""
    with open(path, newline="") as fh:
        return parse_curator_rows(list(csv.DictReader(fh)))


def load_annotations(
    path: str | Path,
    annotator: str,
    curator: dict[str, CuratorEntry] | None = None,
) -> tuple[list[AnnotationRecord], list[str]]:
    """Load one annotator's spreadsheet into joined per-view records.

    Returns ``(records, unmatched_ids)``. When ``curator`` is omitted the records
    are returned unjoined with an empty unmatched list.
    """
    grid = read_xlsx(path)
    if not grid:
        return [], []
    header, body = grid[0], grid[1:]
    records = parse_annotation_rows(header, body, annotator)
    if curator is None:
        return records, []
    return join_specimens(records, curator)
