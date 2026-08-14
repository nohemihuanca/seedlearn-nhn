"""Downscale specimen view images into size-controlled base64 data URIs.

Keeps the grading report a single self-contained HTML: each annotated view is
downscaled to a fixed longest edge and JPEG-encoded, then base64-inlined as a
``data:`` URI so no external image files are needed. A soft total-byte budget bounds
the report size and is logged (not enforced) when exceeded, and unreadable images are
skipped with a warning rather than aborting the report.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Defaults chosen so ~570 views fit in a few MB while staying legible. The report embeds
# larger (see report.THUMB_MAX_EDGE) for a crisp click-to-zoom lightbox, so the soft budget
# is generous; it only logs when exceeded (never enforced).
DEFAULT_MAX_EDGE = 160
DEFAULT_QUALITY = 70
DEFAULT_BUDGET_BYTES = 30_000_000


def thumbnail_data_uri(
    path: str | Path,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = DEFAULT_QUALITY,
) -> str | None:
    """Return a base64 JPEG ``data:`` URI for ``path``, or ``None`` if unreadable.

    The image is converted to RGB and downscaled so its longest edge is at most
    ``max_edge`` (aspect preserved; never upscaled).
    """
    try:
        from PIL import Image  # noqa: PLC0415 - optional-at-runtime, always installed here

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_edge, max_edge))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
    except (OSError, ValueError) as exc:  # unreadable / not an image
        logger.warning("thumbnail skipped (%s): %s", exc, path)
        return None
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def load_specimen_image_paths(results_dir: str | Path) -> dict[str, list[str]]:
    """Map ``specimen_id -> [image_path]`` from the model-run JSONs' ``image_paths``."""
    out: dict[str, list[str]] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name == "run_metadata.json":
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        specimen_id = data.get("specimen_id") or path.stem
        paths = data.get("image_paths") or []
        if isinstance(paths, list):
            out[specimen_id] = [str(p) for p in paths]
    return out


def specimen_thumbnails(
    paths_by_specimen: dict[str, list[str]],
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = DEFAULT_QUALITY,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
) -> tuple[dict[str, list[str]], int]:
    """Build ``specimen_id -> [data_uri]`` thumbnails and the total embedded bytes.

    Unreadable images are skipped. When the running total exceeds ``budget_bytes`` the
    overage is logged, but every readable thumbnail is still returned (soft budget).
    """
    out: dict[str, list[str]] = {}
    total = 0
    for sid, paths in paths_by_specimen.items():
        uris: list[str] = []
        for p in paths:
            uri = thumbnail_data_uri(p, max_edge, quality)
            if uri is None:
                continue
            uris.append(uri)
            total += len(uri)
        out[sid] = uris
    if total > budget_bytes:
        logger.warning(
            "embedded thumbnails total %d bytes exceed soft budget %d (lower max_edge/quality)",
            total, budget_bytes,
        )
    return out, total
