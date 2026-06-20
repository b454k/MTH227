"""Shared helpers for the AI exposure evidence pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATISTICS_SOURCE_IDS = {"nber_w31222", "anthropic_economic_index"}


def created_at() -> str:
    """Return a stable UTC timestamp for generated rows."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when needed."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def clean_text(value: Any) -> str:
    """Normalize whitespace while preserving ordinary punctuation."""
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def one_line(value: Any) -> str:
    """Return compact single-line text."""
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def normalize_text(value: Any) -> str:
    """Return a lowercase alphanumeric-ish string for matching and deduping."""
    text = one_line(value).lower()
    text = re.sub(r"[^a-z0-9.%$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: Any, max_length: int = 80) -> str:
    """Create a filesystem-safe slug."""
    slug = normalize_text(value)
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    if not slug:
        slug = "source"
    return slug[:max_length].strip("_")


def short_hash(value: Any, length: int = 10) -> str:
    """Return a short sha256 hash for stable IDs."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def stable_doc_id(*parts: Any, prefix: str = "doc") -> str:
    """Build a readable ID with a short collision-resistant suffix."""
    readable = slugify("_".join(one_line(part) for part in parts if one_line(part)), 80)
    return f"{prefix}_{readable}_{short_hash(parts)}"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL rows, skipping blank lines."""
    resolved = resolve_project_path(path)
    rows: list[dict[str, Any]] = []
    if not resolved.exists():
        return rows
    with resolved.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {resolved} line {line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write dictionaries as JSONL and return the number of rows written."""
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with resolved.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    """Append one JSONL row."""
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    """Compute a file hash."""
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def detect_source_policy(url: str, title: str = "") -> dict[str, Any]:
    """Return source-policy metadata for one URL/title."""
    haystack = f"{url} {title}".lower()

    if "w31222" in haystack or "working_papers/w31222" in haystack:
        return {
            "source_id": "nber_w31222",
            "source_name": "NBER Working Paper 31222",
            "statistics_allowed": True,
            "tables_allowed": True,
            "images_allowed": False,
            "chunks_allowed": True,
            "usage": "statistics_methodology",
        }

    anthropic_markers = (
        "anthropic/economicindex",
        "anthropic economic index",
        "economic-index",
        "which economic tasks are performed with ai",
        "arxiv.org/pdf/2503.04761",
        "arxiv.org/abs/2503.04761",
    )
    if any(marker in haystack for marker in anthropic_markers):
        return {
            "source_id": "anthropic_economic_index",
            "source_name": "Anthropic Economic Index",
            "statistics_allowed": True,
            "tables_allowed": True,
            "images_allowed": False,
            "chunks_allowed": True,
            "usage": "statistics_methodology",
        }

    return {
        "source_id": "",
        "source_name": "",
        "statistics_allowed": False,
        "tables_allowed": False,
        "images_allowed": False,
        "chunks_allowed": True,
        "usage": "inference_methodology_caveat_only",
    }


def normalize_soc_code(value: Any) -> str:
    """Normalize likely SOC/O*NET-SOC strings to the O*NET 00-0000.00 shape."""
    text = one_line(value)
    if not text:
        return ""
    text = text.upper().replace("O*", "").replace("NET", "")
    match = re.search(r"(\d{2})[- ]?(\d{4})(?:[.\- ]?(\d{2}))?", text)
    if not match:
        return ""
    suffix = match.group(3) or "00"
    return f"{match.group(1)}-{match.group(2)}.{suffix}"


def parse_float(value: Any) -> float | None:
    """Parse a numeric value if possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = one_line(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def chroma_scalar(value: Any) -> str | int | float | bool:
    """Convert metadata values to Chroma-compatible scalars."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return ""
        return value
    return one_line(value)


def compact_dict_text(values: dict[str, Any], keys: list[str]) -> str:
    """Format selected non-empty key/value pairs for evidence text."""
    parts = []
    for key in keys:
        value = one_line(values.get(key))
        if value:
            parts.append(f"{key}: {value}")
    return "; ".join(parts)

