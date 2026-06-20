"""Enrich collected research source metadata before chunk extraction.

This script reads the collected source inventory, inspects downloaded PDFs when
available, guesses missing title/author/year metadata, and writes the enriched
CSV consumed by ``extract_research_chunks.py``.
"""

from pathlib import Path
import re

import pandas as pd
import fitz  # PyMuPDF


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "data" / "research"
SOURCES_CSV = BASE_DIR / "research_sources.csv"
OUTPUT_CSV = BASE_DIR / "research_sources_enriched.csv"


def is_missing(value) -> bool:
    """Correctly handles NaN, None, and empty strings."""
    if value is None:
        return True
    if pd.isna(value):
        return True
    if str(value).strip() == "":
        return True
    return False


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_first_pages(path: Path, max_pages: int = 3) -> str:
    doc = fitz.open(path)
    texts = []

    for i in range(min(max_pages, len(doc))):
        texts.append(doc[i].get_text("text"))

    return "\n".join(texts)


def extract_pdf_builtin_metadata(path: Path) -> dict:
    try:
        doc = fitz.open(path)
        meta = doc.metadata or {}

        return {
            "title": clean_text(meta.get("title", "")),
            "authors": clean_text(meta.get("author", "")),
            "subject": clean_text(meta.get("subject", "")),
        }

    except Exception:
        return {
            "title": "",
            "authors": "",
            "subject": "",
        }


def guess_title_from_text(text: str) -> str:
    lines = [clean_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    bad_patterns = [
        "working paper",
        "nber working paper",
        "abstract",
        "downloaded from",
        "copyright",
        "issn",
        "isbn",
        "journal",
        "vol.",
        "doi:",
        "http",
        "www.",
    ]

    candidates = []

    for line in lines[:80]:
        lower = line.lower()

        if any(p in lower for p in bad_patterns):
            continue

        if len(line) < 10 or len(line) > 200:
            continue

        if "@" in line:
            continue

        # Avoid lines that are mostly numbers or punctuation
        letters = sum(ch.isalpha() for ch in line)
        if letters < len(line) * 0.5:
            continue

        candidates.append(line)

    if candidates:
        return candidates[0]

    return ""


def guess_authors_from_text(text: str, title: str) -> str:
    lines = [clean_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    if not title:
        return ""

    title_index = None

    for i, line in enumerate(lines[:100]):
        if title.lower() in line.lower() or line.lower() in title.lower():
            title_index = i
            break

    if title_index is None:
        return ""

    possible_lines = lines[title_index + 1:title_index + 8]
    author_candidates = []

    bad_words = [
        "abstract",
        "working paper",
        "university",
        "department",
        "school",
        "institute",
        "http",
        "www",
        "email",
        "copyright",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]

    for line in possible_lines:
        lower = line.lower()

        if any(word in lower for word in bad_words):
            continue

        if len(line) < 3 or len(line) > 180:
            continue

        # Author lines often contain commas, "and", initials, or multiple capitalized names
        capitalized_words = re.findall(r"\b[A-Z][a-zA-Z\-]+\b", line)

        if len(capitalized_words) >= 2:
            author_candidates.append(line)

    if author_candidates:
        return "; ".join(author_candidates[:2])

    return ""


def guess_year(text: str, url: str = "") -> str:
    combined = text[:5000] + " " + str(url)

    years = re.findall(r"\b(20[0-2][0-9]|19[8-9][0-9])\b", combined)

    if not years:
        return ""

    # Usually the paper year appears near the first pages.
    # Use the most recent year found.
    return str(max(int(y) for y in years))


def guess_doi(text: str) -> str:
    doi_pattern = r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b"
    match = re.search(doi_pattern, text)

    if match:
        return match.group(0).rstrip(".")

    return ""


def guess_title_from_url(url: str) -> str:
    url = str(url)
    last_part = url.rstrip("/").split("/")[-1]
    last_part = re.sub(r"\.pdf$", "", last_part, flags=re.I)
    last_part = re.sub(r"[_\-]+", " ", last_part)
    last_part = clean_text(last_part)

    if len(last_part) >= 8:
        return last_part.title()

    return ""


def process_row(row: pd.Series) -> dict:
    row = row.to_dict()

    local_path = row.get("local_path", "")

    if is_missing(local_path):
        return row

    path = Path(str(local_path))

    if not path.exists():
        row["metadata_error"] = f"local_path not found: {path}"
        return row

    text = ""

    try:
        if path.suffix.lower() == ".pdf":
            pdf_meta = extract_pdf_builtin_metadata(path)

            text = extract_pdf_first_pages(path, max_pages=3)

            if is_missing(row.get("title")) and not is_missing(pdf_meta.get("title")):
                row["title"] = pdf_meta["title"]

            if is_missing(row.get("authors")) and not is_missing(pdf_meta.get("authors")):
                row["authors"] = pdf_meta["authors"]

        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")[:15000]

        if is_missing(row.get("title")):
            guessed_title = guess_title_from_text(text)

            if not guessed_title:
                guessed_title = guess_title_from_url(row.get("url", ""))

            row["title"] = guessed_title

        if is_missing(row.get("authors")):
            row["authors"] = guess_authors_from_text(text, row.get("title", ""))

        if is_missing(row.get("year")):
            row["year"] = guess_year(text, row.get("url", ""))

        if is_missing(row.get("doi")):
            row["doi"] = guess_doi(text)

        row["metadata_error"] = ""

    except Exception as e:
        row["metadata_error"] = f"{type(e).__name__}: {e}"

    return row


def main():
    df = pd.read_csv(SOURCES_CSV)

    records = []

    for _, row in df.iterrows():
        records.append(process_row(row))

    enriched = pd.DataFrame(records)

    if "metadata_review_status" not in enriched.columns:
        enriched["metadata_review_status"] = "auto_extracted"

    if "citation_label" not in enriched.columns:
        enriched["citation_label"] = ""

    enriched.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved enriched metadata to: {OUTPUT_CSV}")
    print()

    print("Missing values after enrichment:")
    print(enriched[["title", "authors", "year", "doi"]].isna().sum())
    print()

    print(enriched[[
        "source_id",
        "title",
        "authors",
        "year",
        "doi",
        "status",
        "metadata_error",
    ]].head(20))


if __name__ == "__main__":
    main()
