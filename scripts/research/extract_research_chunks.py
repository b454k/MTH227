#!/usr/bin/env python3
"""Extract policy-limited inference chunks from downloaded research sources."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from career_rag.ai_exposure_utils import (
    clean_text,
    created_at,
    one_line,
    read_jsonl,
    resolve_project_path,
    stable_doc_id,
    write_jsonl,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "research_sources" / "source_manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "research_inference_chunks.jsonl"

MIN_WORDS = 120
TARGET_WORDS = 700
MAX_WORDS = 900
OVERLAP_WORDS = 100

METHOD_CAVEAT_TERMS = (
    "method",
    "methodology",
    "measure",
    "definition",
    "limitation",
    "caveat",
    "uncertain",
    "uncertainty",
    "exposure",
    "automation",
    "augmentation",
    "task",
    "occupation",
    "generative ai",
    "large language model",
    "llm",
)

GENERIC_SKIP_TERMS = (
    "copyright",
    "all rights reserved",
    "subscribe",
    "newsletter",
    "terms of use",
    "privacy policy",
    "acknowledgement",
)


def extract_pdf_pages(path: Path) -> list[dict[str, Any]]:
    """Extract text page by page from a PDF."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF extraction (`pip install PyMuPDF`).") from exc

    doc = fitz.open(path)
    try:
        raw_pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()

    repeated = repeated_pdf_furniture(raw_pages)
    pages: list[dict[str, Any]] = []
    for page_number, raw_text in enumerate(raw_pages, start=1):
        pages.append(
            {
                "page": page_number,
                "text": clean_page_text(raw_text, repeated),
            }
        )
    return pages


def repeated_pdf_furniture(page_texts: list[str]) -> set[str]:
    """Detect repeated header/footer lines from page edges."""
    counts: dict[str, int] = {}
    for text in page_texts:
        lines = [one_line(line) for line in str(text).splitlines() if one_line(line)]
        for line in set(lines[:4] + lines[-4:]):
            if len(line) <= 160:
                counts[line] = counts.get(line, 0) + 1
    threshold = max(3, len(page_texts) // 4)
    return {line for line, count in counts.items() if count >= threshold}


def clean_page_text(text: str, repeated_lines: set[str]) -> str:
    """Remove page furniture and normalize PDF text."""
    lines: list[str] = []
    for line in str(text).splitlines():
        normalized = one_line(line)
        if normalized in repeated_lines:
            continue
        if re.fullmatch(r"[-\s]*\d{1,4}[-\s]*", normalized):
            continue
        lines.append(line)
    return clean_text("\n".join(lines))


def extract_html_text(path: Path) -> str:
    """Extract readable text from a saved HTML file."""
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return clean_text(soup.get_text("\n", strip=True))


def extract_text_pages(path: Path) -> list[dict[str, Any]]:
    """Extract pages from PDF/HTML/TXT sources."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(path)
    if suffix in {".html", ".htm"}:
        return [{"page": None, "text": extract_html_text(path)}]
    if suffix in {".txt", ".md"}:
        return [{"page": None, "text": clean_text(path.read_text(encoding="utf-8", errors="replace"))}]
    return []


def paragraph_units(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split extracted pages into paragraph units with page numbers."""
    units: list[dict[str, Any]] = []
    for page in pages:
        for paragraph in re.split(r"\n\s*\n", page.get("text") or ""):
            paragraph = one_line(paragraph)
            if paragraph:
                units.append({"page": page.get("page"), "text": paragraph})
    return units


def word_count(text: str) -> int:
    """Approximate token count with words."""
    return len(re.findall(r"\b\w+\b", text))


def split_oversized_unit(unit: dict[str, Any]) -> list[dict[str, Any]]:
    """Split unusually long paragraphs by sentence/word boundaries."""
    text = unit["text"]
    if word_count(text) <= MAX_WORDS:
        return [unit]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[dict[str, Any]] = []
    current: list[str] = []
    for sentence in sentences:
        if word_count(" ".join(current + [sentence])) <= MAX_WORDS:
            current.append(sentence)
        else:
            if current:
                pieces.append({"page": unit.get("page"), "text": " ".join(current).strip()})
            current = [sentence]
    if current:
        pieces.append({"page": unit.get("page"), "text": " ".join(current).strip()})
    return [piece for piece in pieces if piece["text"]]


def chunk_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build 500-900 word chunks with 100-word overlap."""
    expanded: list[dict[str, Any]] = []
    for unit in units:
        expanded.extend(split_oversized_unit(unit))

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_words = 0

    for unit in expanded:
        unit_words = word_count(unit["text"])
        if current and current_words + unit_words > MAX_WORDS:
            maybe_add_chunk(chunks, current)
            current = overlap_units(current)
            current_words = sum(word_count(item["text"]) for item in current)
        current.append(unit)
        current_words += unit_words
        if current_words >= TARGET_WORDS:
            maybe_add_chunk(chunks, current)
            current = overlap_units(current)
            current_words = sum(word_count(item["text"]) for item in current)

    maybe_add_chunk(chunks, current)
    return chunks


def overlap_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the tail units that approximate the overlap window."""
    selected: list[dict[str, Any]] = []
    total = 0
    for unit in reversed(units):
        selected.insert(0, unit)
        total += word_count(unit["text"])
        if total >= OVERLAP_WORDS:
            break
    return selected


def maybe_add_chunk(chunks: list[dict[str, Any]], units: list[dict[str, Any]]) -> None:
    """Append a chunk if it has enough useful text."""
    if not units:
        return
    text = "\n\n".join(unit["text"] for unit in units).strip()
    if word_count(text) < MIN_WORDS:
        return
    pages = [unit.get("page") for unit in units if unit.get("page") is not None]
    chunks.append({"text": text, "page": min(pages) if pages else None})


def useful_inference_chunk(text: str) -> bool:
    """Filter references, boilerplate, and generic paragraphs where possible."""
    normalized = one_line(text).lower()
    if not normalized:
        return False
    if normalized.startswith("references") or normalized.startswith("bibliography"):
        return any(term in normalized for term in ("method", "definition", "measure"))
    if any(term in normalized[:500] for term in GENERIC_SKIP_TERMS):
        return False
    return any(term in normalized for term in METHOD_CAVEAT_TERMS)


def should_process_source(row: dict[str, Any]) -> bool:
    """Return True for non-statistics sources with chunking enabled."""
    if row.get("statistics_allowed") is True:
        return False
    if row.get("source_id") == "nber_w31222":
        return False
    if row.get("chunks_allowed") is False:
        return False
    local_path = one_line(row.get("local_path"))
    if not local_path:
        return False
    return one_line(row.get("download_status")).startswith("downloaded")


def row_source_file(row: dict[str, Any]) -> Path:
    """Resolve a manifest local path."""
    return resolve_project_path(one_line(row.get("local_path")))


def build_chunk_rows(source: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw chunks into requested JSONL schema."""
    rows: list[dict[str, Any]] = []
    source_id = one_line(source.get("source_id"))
    for index, chunk in enumerate(chunks, start=1):
        text = chunk["text"]
        if not useful_inference_chunk(text):
            continue
        rows.append(
            {
                "doc_id": stable_doc_id(source_id, index, text[:80], prefix="research_inference"),
                "doc_type": "research_inference_chunk",
                "source_id": source_id,
                "source_name": one_line(source.get("source_name")) or one_line(source.get("title")),
                "source_url": one_line(source.get("source_url")),
                "source_file": one_line(source.get("local_path")),
                "page": chunk.get("page"),
                "chunk_text": text,
                "allowed_usage": "methodology_caveat_inference_only",
                "statistics_allowed": False,
                "created_at": created_at(),
            }
        )
    return rows


def extract_source(source: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Extract chunks for one source, returning rows and an error message."""
    path = row_source_file(source)
    if not path.exists():
        return [], f"Local source file not found: {path}"
    try:
        pages = extract_text_pages(path)
        chunks = chunk_units(paragraph_units(pages))
        return build_chunk_rows(source, chunks), ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Extract research inference chunks from source manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    """Run extraction."""
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest)
    output_path = resolve_project_path(args.output)
    sources = read_jsonl(manifest_path)
    if not sources:
        raise FileNotFoundError(f"No source manifest rows found at {manifest_path}")

    all_rows: list[dict[str, Any]] = []
    processed_sources = 0
    skipped_sources = 0
    errors: list[tuple[str, str]] = []

    for source in sources:
        if not should_process_source(source):
            skipped_sources += 1
            continue
        rows, error = extract_source(source)
        if error:
            errors.append((one_line(source.get("source_id")), error))
        if rows:
            processed_sources += 1
            all_rows.extend(rows)

    write_jsonl(all_rows, output_path)
    print("Research inference chunk extraction complete")
    print(f"Manifest rows: {len(sources)}")
    print(f"Sources processed: {processed_sources}")
    print(f"Sources skipped by policy/status: {skipped_sources}")
    print(f"Sources with errors: {len(errors)}")
    print(f"Chunks written: {len(all_rows)}")
    print(f"Output: {output_path}")
    for source_id, error in errors[:10]:
        print(f"Warning: {source_id}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
