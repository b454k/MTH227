"""Extract clean text chunks from saved research PDFs and web snapshots.

The script reads ``research_sources_enriched.csv``, extracts text from local
PDF/TXT/HTML files, chunks it with page metadata, and writes
``research_chunks.jsonl`` plus a chunk summary CSV for later claim extraction.
"""

from pathlib import Path
import json
import re

import fitz  # PyMuPDF
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = REPO_ROOT / "data" / "research"
SOURCES_CSV = RESEARCH_DIR / "research_sources_enriched.csv"
CHUNKS_JSONL = RESEARCH_DIR / "research_chunks.jsonl"
SUMMARY_CSV = RESEARCH_DIR / "research_chunk_summary.csv"

PROCESSABLE_STATUSES = {
    "saved_pdf",
    "saved_pdf_from_page",
    "saved_webpage_text",
}

MIN_CHARS = 300
TARGET_CHARS = 2000
MAX_CHARS = 2500
OVERLAP_CHARS = 250
MAX_PAGES_PER_PDF_CHUNK = 3

CHUNK_FIELDS = [
    "chunk_id",
    "source_id",
    "chunk_index",
    "source_type",
    "title",
    "authors",
    "year",
    "doi",
    "url",
    "final_url",
    "access_date",
    "local_path",
    "status",
    "page_start",
    "page_end",
    "section_title",
    "text",
    "char_count",
    "word_count",
]

SUMMARY_FIELDS = [
    "source_id",
    "title",
    "status",
    "local_path",
    "processed",
    "num_chunks",
    "num_chars_extracted",
    "num_words_extracted",
    "error_message",
]


def get_value(row, column):
    """Return a metadata value as a clean string, even when the column is absent."""
    value = row.get(column, "")
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def resolve_local_path(local_path):
    """Resolve paths saved in the CSV, which are usually relative Windows paths."""
    if not local_path:
        return None

    path = Path(local_path)
    if path.is_absolute():
        return path

    return REPO_ROOT / path


def normalize_line(line):
    return re.sub(r"\s+", " ", line).strip()


def is_page_number_line(line):
    normalized = normalize_line(line).lower()
    if re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", normalized):
        return True
    if re.fullmatch(r"(page\s*)?\d{1,4}\s*(of|/)\s*\d{1,4}", normalized):
        return True
    return False


def clean_text_block(text):
    """
    Normalize text without stripping useful punctuation or citation context.

    The goal is to repair extraction whitespace, not to rewrite the document.
    """
    text = "" if text is None else str(text)
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def find_repeated_pdf_furniture(page_texts):
    """
    Find repeated header/footer lines from page edges.

    This intentionally only inspects the first and last few non-empty lines on
    each page, so repeated phrases in the body text are not removed.
    """
    counts = {}

    for text in page_texts:
        lines = [normalize_line(line) for line in str(text).splitlines()]
        lines = [line for line in lines if line]
        edge_lines = lines[:4] + lines[-4:]

        for line in set(edge_lines):
            if len(line) > 160:
                continue
            counts[line] = counts.get(line, 0) + 1

    threshold = max(3, len(page_texts) // 4)
    return {line for line, count in counts.items() if count >= threshold}


def clean_pdf_page_text(page_text, repeated_lines):
    lines = []

    for line in str(page_text).splitlines():
        normalized = normalize_line(line)
        if not normalized:
            lines.append("")
            continue
        if normalized in repeated_lines:
            continue
        if is_page_number_line(normalized):
            continue
        lines.append(line)

    return clean_text_block("\n".join(lines))


def extract_pdf_pages(path):
    """Extract text page by page with PyMuPDF. No OCR is attempted."""
    doc = fitz.open(path)
    try:
        raw_pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()

    repeated_lines = find_repeated_pdf_furniture(raw_pages)
    pages = []

    for index, raw_text in enumerate(raw_pages, start=1):
        cleaned_text = clean_pdf_page_text(raw_text, repeated_lines)
        pages.append(
            {
                "page_number": index,
                "text": cleaned_text,
            }
        )

    return pages


def read_text_snapshot(path):
    return path.read_text(encoding="utf-8", errors="replace")


def count_words(text):
    return len(re.findall(r"\b\w+\b", text))


def split_long_text(text, max_chars=MAX_CHARS):
    """
    Split very long paragraphs at sentence or word boundaries.

    Most chunks are paragraph-based; this is only for unusually long extracted
    blocks that would otherwise exceed the chunk size.
    """
    text = clean_text_block(text)
    if len(text) <= max_chars:
        return [text] if text else []

    pieces = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(split_by_words(sentence, max_chars))
            continue

        candidate = sentence if not current else current + " " + sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current.strip())
            current = sentence

    if current:
        pieces.append(current.strip())

    return pieces


def split_by_words(text, max_chars):
    chunks = []
    current_words = []
    current_length = 0

    for word in text.split():
        extra = len(word) + (1 if current_words else 0)
        if current_words and current_length + extra > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length += extra

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def split_into_paragraphs(text):
    text = clean_text_block(text)
    paragraphs = []

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraphs.extend(split_long_text(paragraph))

    return paragraphs


def make_overlap_units(units, overlap_chars):
    """Carry a small tail of previous text into the next chunk."""
    if not units or overlap_chars <= 0:
        return []

    selected = []
    total_chars = 0

    for unit in reversed(units):
        selected.insert(0, unit)
        total_chars += len(unit["text"]) + 2
        if total_chars >= overlap_chars:
            break

    overlap_text = join_units(selected)
    if len(overlap_text) > overlap_chars:
        overlap_text = overlap_text[-overlap_chars:]
        first_space = overlap_text.find(" ")
        if first_space != -1:
            overlap_text = overlap_text[first_space + 1 :]
        overlap_text = overlap_text.strip()

    page_start, page_end = page_range_for_units(selected)
    return [
        {
            "text": overlap_text,
            "page_start": page_start,
            "page_end": page_end,
        }
    ] if overlap_text else []


def join_units(units):
    return "\n\n".join(unit["text"] for unit in units if unit.get("text", "").strip()).strip()


def page_range_for_units(units):
    starts = [unit.get("page_start") for unit in units if unit.get("page_start") is not None]
    ends = [unit.get("page_end") for unit in units if unit.get("page_end") is not None]

    if not starts or not ends:
        return None, None

    return min(starts), max(ends)


def should_flush_chunk(current_units, next_unit, max_pages_per_chunk):
    if not current_units:
        return False

    candidate_text = join_units(current_units + [next_unit])

    if len(candidate_text) > MAX_CHARS:
        return True

    if max_pages_per_chunk is not None:
        page_start, _ = page_range_for_units(current_units)
        next_page_end = next_unit.get("page_end")
        if page_start is not None and next_page_end is not None:
            page_span = next_page_end - page_start + 1
            if page_span > max_pages_per_chunk:
                return True

    return False


def overlap_fits_next_chunk(overlap_units, next_unit, max_pages_per_chunk):
    if not overlap_units:
        return True

    candidate_text = join_units(overlap_units + [next_unit])
    if len(candidate_text) > MAX_CHARS:
        return False

    if max_pages_per_chunk is not None:
        page_start, _ = page_range_for_units(overlap_units)
        next_page_end = next_unit.get("page_end")
        if page_start is not None and next_page_end is not None:
            page_span = next_page_end - page_start + 1
            if page_span > max_pages_per_chunk:
                return False

    return True


def build_chunks_from_units(units, max_pages_per_chunk=None):
    chunks = []
    current_units = []
    current_has_new_content = False

    for unit in units:
        text = unit.get("text", "").strip()
        if not text:
            continue

        if should_flush_chunk(current_units, unit, max_pages_per_chunk):
            chunk_text = join_units(current_units)
            if current_has_new_content and len(chunk_text) >= MIN_CHARS:
                page_start, page_end = page_range_for_units(current_units)
                chunks.append(
                    {
                        "text": chunk_text,
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                )
                overlap_units = make_overlap_units(current_units, OVERLAP_CHARS)
                if overlap_fits_next_chunk(overlap_units, unit, max_pages_per_chunk):
                    current_units = overlap_units
                else:
                    current_units = []
            else:
                current_units = []
            current_has_new_content = False

        current_units.append(unit)
        current_has_new_content = True

        current_text = join_units(current_units)
        if len(current_text) >= TARGET_CHARS:
            page_start, page_end = page_range_for_units(current_units)
            chunks.append(
                {
                    "text": current_text,
                    "page_start": page_start,
                    "page_end": page_end,
                }
            )
            current_units = make_overlap_units(current_units, OVERLAP_CHARS)
            current_has_new_content = False

    final_text = join_units(current_units)
    if current_has_new_content and len(final_text) >= MIN_CHARS:
        page_start, page_end = page_range_for_units(current_units)
        chunks.append(
            {
                "text": final_text,
                "page_start": page_start,
                "page_end": page_end,
            }
        )

    return chunks


def make_pdf_units(pages):
    units = []

    for page in pages:
        page_number = page["page_number"]
        for paragraph in split_into_paragraphs(page["text"]):
            units.append(
                {
                    "text": paragraph,
                    "page_start": page_number,
                    "page_end": page_number,
                }
            )

    return units


def make_web_units(text):
    return [
        {
            "text": paragraph,
            "page_start": None,
            "page_end": None,
        }
        for paragraph in split_into_paragraphs(text)
    ]


def metadata_for_chunk(row):
    return {
        "source_id": get_value(row, "source_id"),
        "source_type": get_value(row, "source_type"),
        "title": get_value(row, "title"),
        "authors": get_value(row, "authors"),
        "year": get_value(row, "year"),
        "doi": get_value(row, "doi"),
        "url": get_value(row, "url"),
        "final_url": get_value(row, "final_url"),
        "access_date": get_value(row, "access_date"),
        "local_path": get_value(row, "local_path"),
        "status": get_value(row, "status"),
    }


def make_chunk_rows(row, raw_chunks):
    metadata = metadata_for_chunk(row)
    source_id = metadata["source_id"]
    rows = []

    for index, chunk in enumerate(raw_chunks, start=1):
        text = chunk["text"].strip()
        chunk_row = {
            **metadata,
            "chunk_id": f"{source_id}_chunk_{index:04d}",
            "chunk_index": index,
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "section_title": "",
            "text": text,
            "char_count": len(text),
            "word_count": count_words(text),
        }
        rows.append({field: chunk_row.get(field, "") for field in CHUNK_FIELDS})

    return rows


def process_source(row):
    status = get_value(row, "status")
    source_id = get_value(row, "source_id")
    title = get_value(row, "title")
    local_path_text = get_value(row, "local_path")

    summary = {
        "source_id": source_id,
        "title": title,
        "status": status,
        "local_path": local_path_text,
        "processed": False,
        "num_chunks": 0,
        "num_chars_extracted": 0,
        "num_words_extracted": 0,
        "error_message": "",
    }

    if status not in PROCESSABLE_STATUSES:
        summary["error_message"] = f"Skipped because status is '{status}'."
        return [], summary

    path = resolve_local_path(local_path_text)
    if path is None:
        summary["error_message"] = "Missing local_path."
        return [], summary
    if not path.exists():
        summary["error_message"] = f"Local file not found: {path}"
        return [], summary

    try:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            pages = extract_pdf_pages(path)
            extracted_text = "\n\n".join(page["text"] for page in pages if page["text"].strip())
            raw_chunks = build_chunks_from_units(
                make_pdf_units(pages),
                max_pages_per_chunk=MAX_PAGES_PER_PDF_CHUNK,
            )
        elif suffix == ".txt":
            extracted_text = clean_text_block(read_text_snapshot(path))
            raw_chunks = build_chunks_from_units(make_web_units(extracted_text))
        else:
            summary["error_message"] = f"Unsupported local file type: {suffix or '(none)'}"
            return [], summary

        summary["num_chars_extracted"] = len(extracted_text)
        summary["num_words_extracted"] = count_words(extracted_text)

        if not extracted_text.strip():
            summary["error_message"] = "No text extracted."
            return [], summary

        chunk_rows = make_chunk_rows(row, raw_chunks)
        summary["num_chunks"] = len(chunk_rows)

        if not chunk_rows:
            summary["error_message"] = "No chunks created after filtering short text."
            return [], summary

        summary["processed"] = True
        return chunk_rows, summary

    except Exception as exc:
        summary["error_message"] = f"{type(exc).__name__}: {exc}"
        return [], summary


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_output(path=CHUNKS_JSONL, preview_count=3):
    print("\nValidation")

    if not path.exists():
        print(f"Missing output file: {path}")
        return False

    required_fields = ["chunk_id", "source_id", "text"]
    errors = []
    previews = []
    total_rows = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: invalid JSON ({exc})")
                continue

            total_rows += 1

            for field in required_fields:
                if not str(row.get(field, "")).strip():
                    errors.append(f"Line {line_number}: missing or empty '{field}'")

            if not str(row.get("text", "")).strip():
                errors.append(f"Line {line_number}: empty text chunk")

            if len(previews) < preview_count:
                previews.append(row)

    if errors:
        print(f"Validation failed with {len(errors)} issue(s).")
        for error in errors[:10]:
            print(f"- {error}")
        if len(errors) > 10:
            print(f"- ... {len(errors) - 10} more")
        return False

    print(f"Validated {total_rows} chunk row(s).")
    print(f"Showing first {min(preview_count, len(previews))} chunk(s):")

    for row in previews:
        title = row.get("title") or "(untitled)"
        page_start = row.get("page_start")
        page_end = row.get("page_end")

        if page_start is None and page_end is None:
            page_range = "web"
        elif page_start == page_end:
            page_range = f"page {page_start}"
        else:
            page_range = f"pages {page_start}-{page_end}"

        excerpt = re.sub(r"\s+", " ", row.get("text", "")).strip()[:300]
        print(f"\n{row.get('chunk_id')} | {title} | {page_range}")
        print(excerpt)

    return True


def main():
    if not SOURCES_CSV.exists():
        raise FileNotFoundError(f"Metadata file not found: {SOURCES_CSV}")

    sources = pd.read_csv(SOURCES_CSV, dtype=str, keep_default_na=False)

    all_chunks = []
    summary_rows = []

    for _, row in sources.iterrows():
        chunk_rows, summary = process_source(row)
        all_chunks.extend(chunk_rows)
        summary_rows.append(summary)

    write_jsonl(all_chunks, CHUNKS_JSONL)
    pd.DataFrame(summary_rows, columns=SUMMARY_FIELDS).to_csv(SUMMARY_CSV, index=False)

    total_sources = len(sources)
    processed_sources = sum(1 for row in summary_rows if row["processed"])
    skipped_failed_sources = sum(
        1 for row in summary_rows if row["status"] not in PROCESSABLE_STATUSES
    )
    extraction_errors = sum(
        1
        for row in summary_rows
        if row["status"] in PROCESSABLE_STATUSES and not row["processed"]
    )

    print("\nResearch chunk extraction complete")
    print(f"Total sources in metadata file: {total_sources}")
    print(f"Successfully processed sources: {processed_sources}")
    print(f"Skipped failed sources: {skipped_failed_sources}")
    print(f"Sources with extraction errors: {extraction_errors}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Output path: {CHUNKS_JSONL}")
    print(f"Summary path: {SUMMARY_CSV}")

    validate_output(CHUNKS_JSONL)


if __name__ == "__main__":
    main()
