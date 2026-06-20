#!/usr/bin/env python3
"""Extract structured AI-exposure evidence from NBER Working Paper 31222."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from career_rag.ai_exposure_utils import (
    clean_text,
    created_at,
    one_line,
    parse_float,
    read_jsonl,
    resolve_project_path,
    stable_doc_id,
    write_jsonl,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "research_sources" / "source_manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "nber_w31222_ai_exposure.jsonl"
DEFAULT_METHOD_OUTPUT = PROJECT_ROOT / "data" / "processed" / "nber_w31222_method_chunks.jsonl"
DEFAULT_ERRORS = PROJECT_ROOT / "data" / "processed" / "nber_w31222_extraction_errors.jsonl"
DEFAULT_MODEL = "gpt-4o-mini"

EXPOSURE_TERMS = (
    "exposure",
    "direct",
    "indirect",
    "no exposure",
    "core",
    "supplemental",
    "task-level",
    "occupation-level",
    "firm-level",
    "firm exposure",
    "o*net",
    "onet",
    "generative ai",
)

STATISTIC_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?\s*(?:%|percent|percentage points|pp)?", re.I)

LLM_SYSTEM_PROMPT = (
    "You are extracting structured AI exposure evidence from NBER Working Paper 31222. "
    "Extract only claims directly supported by the text. Do not infer missing values. "
    "Return JSON list only. Each row must include metric_name, metric_value if present, "
    "scope, evidence_text, supporting_excerpt, page, and confidence. If no extractable "
    "evidence exists, return []."
)


def load_dotenv_if_available() -> None:
    """Load .env when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def find_nber_source(manifest_path: Path) -> dict[str, Any]:
    """Find the NBER W31222 manifest row."""
    rows = read_jsonl(manifest_path)
    for row in rows:
        if row.get("source_id") == "nber_w31222":
            return row
    for row in rows:
        haystack = f"{row.get('source_url', '')} {row.get('local_path', '')}".lower()
        if "w31222" in haystack:
            row = dict(row)
            row["source_id"] = "nber_w31222"
            row["source_name"] = "NBER Working Paper 31222"
            return row
    raise FileNotFoundError(f"No nber_w31222 row found in {manifest_path}")


def extract_pdf_pages(path: Path) -> list[dict[str, Any]]:
    """Extract text by page with PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for NBER extraction (`pip install PyMuPDF`).") from exc

    doc = fitz.open(path)
    try:
        return [
            {"page": index + 1, "text": clean_text(page.get_text("text"))}
            for index, page in enumerate(doc)
        ]
    finally:
        doc.close()


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units."""
    text = one_line(text)
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraph-like units."""
    return [one_line(paragraph) for paragraph in re.split(r"\n\s*\n", clean_text(text)) if one_line(paragraph)]


def is_relevant_method_text(text: str) -> bool:
    """Return True for text likely to describe exposure methods."""
    lowered = one_line(text).lower()
    return any(term in lowered for term in EXPOSURE_TERMS) and len(lowered) > 80


def make_method_chunks(source: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create method chunks from relevant page paragraphs."""
    chunks: list[dict[str, Any]] = []
    for page in pages:
        paragraphs = [p for p in split_paragraphs(page["text"]) if is_relevant_method_text(p)]
        current: list[str] = []
        current_words = 0
        for paragraph in paragraphs:
            words = len(paragraph.split())
            if current and current_words + words > 850:
                chunks.append(build_method_chunk(source, page["page"], "\n\n".join(current)))
                current = []
                current_words = 0
            current.append(paragraph)
            current_words += words
        if current:
            chunks.append(build_method_chunk(source, page["page"], "\n\n".join(current)))
    return dedupe_rows(chunks, key_fields=["source_page", "chunk_text"])


def build_method_chunk(source: dict[str, Any], page: int | None, text: str) -> dict[str, Any]:
    """Build one NBER method chunk row."""
    return {
        "doc_id": stable_doc_id("nber_w31222_method", page, text[:80], prefix="nber_method"),
        "doc_type": "ai_methodology",
        "source_id": "nber_w31222",
        "source_name": "NBER Working Paper 31222",
        "source_url": one_line(source.get("source_url")),
        "source_file": one_line(source.get("local_path")),
        "source_page": page,
        "chunk_text": text,
        "allowed_usage": "nber_methodology_and_statistics_only",
        "statistics_allowed": True,
        "created_at": created_at(),
    }


def infer_doc_type(text: str) -> str:
    """Classify an evidence row without changing its scope."""
    lowered = text.lower()
    if "firm" in lowered:
        return "ai_firm_exposure"
    if "core" in lowered or "supplemental" in lowered:
        return "ai_core_supplemental_exposure"
    if "occupation" in lowered or "occupational" in lowered:
        return "ai_occupation_impact"
    if "task" in lowered or "o*net" in lowered or "onet" in lowered:
        return "ai_task_impact"
    return "ai_methodology"


def infer_impact_type(text: str, has_metric: bool) -> str:
    """Map text to the requested impact_type labels."""
    lowered = text.lower()
    if "direct" in lowered and "indirect" not in lowered:
        return "direct_exposure"
    if "indirect" in lowered:
        return "indirect_exposure"
    if "no exposure" in lowered or "not exposed" in lowered:
        return "no_exposure"
    if "core" in lowered:
        return "core_task_exposure"
    if "supplemental" in lowered:
        return "supplemental_task_exposure"
    if "firm" in lowered:
        return "firm_exposure"
    if has_metric or "exposure" in lowered:
        return "exposure"
    return "methodology"


def metric_unit(text: str) -> str | None:
    """Infer metric unit from nearby text."""
    lowered = text.lower()
    if "%" in text or "percent" in lowered:
        return "percent"
    if "percentage point" in lowered or re.search(r"\bpp\b", lowered):
        return "percentage_points"
    return None


def make_schema_row(
    source: dict[str, Any],
    page: int | None,
    text: str,
    extraction_method: str,
    metric_name: str | None = None,
    metric_value: float | None = None,
    confidence: str = "medium",
) -> dict[str, Any]:
    """Build one structured NBER row in the requested schema."""
    has_metric = metric_value is not None
    doc_type = infer_doc_type(text)
    impact_type = infer_impact_type(text, has_metric)
    if doc_type == "ai_methodology":
        impact_type = "methodology"
    return {
        "doc_id": stable_doc_id("nber_w31222", page, doc_type, impact_type, text[:120], prefix="nber"),
        "doc_type": doc_type,
        "source_id": "nber_w31222",
        "source_name": "NBER Working Paper 31222",
        "source_url": one_line(source.get("source_url")),
        "source_file": one_line(source.get("local_path")),
        "source_page": page,
        "soc_code": None,
        "occupation_title": None,
        "onet_task_id": None,
        "task_text": None,
        "field_or_industry": None,
        "impact_type": impact_type,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_unit": metric_unit(text),
        "comparison_group": None,
        "time_period": None,
        "evidence_text": text,
        "interpretation": interpretation_for_row(doc_type, impact_type, has_metric),
        "supporting_excerpt": text[:1200],
        "confidence": confidence,
        "extraction_method": extraction_method,
        "statistics_allowed": True,
        "created_at": created_at(),
    }


def interpretation_for_row(doc_type: str, impact_type: str, has_metric: bool) -> str:
    """Write cautious interpretation text."""
    if impact_type == "methodology":
        return "Use this as methodology evidence; it is not an empirical task statistic."
    if doc_type == "ai_firm_exposure":
        return "This is firm-level exposure evidence and should not be presented as task-level exposure."
    if has_metric:
        return "This is a reported NBER W31222 exposure metric; keep its original scope."
    return "This supports the exposure method, but does not provide a specific numeric value."


def rule_based_rows(source: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract obvious methodology and numeric exposure sentences."""
    rows: list[dict[str, Any]] = []
    for page in pages:
        page_number = page["page"]
        for paragraph in split_paragraphs(page["text"]):
            if not is_relevant_method_text(paragraph):
                continue

            lowered = paragraph.lower()
            definition_like = any(
                phrase in lowered
                for phrase in (
                    "we define",
                    "defined as",
                    "classified as",
                    "direct exposure",
                    "indirect exposure",
                    "no exposure",
                    "core tasks",
                    "supplemental tasks",
                    "task-level exposure",
                    "occupation-level exposure",
                    "firm-level exposure",
                )
            )
            if definition_like:
                rows.append(
                    make_schema_row(
                        source,
                        page_number,
                        paragraph[:1600],
                        extraction_method="pdf_text",
                        confidence="medium",
                    )
                )

        for sentence in split_sentences(page["text"]):
            lowered = sentence.lower()
            if "exposure" not in lowered and "exposed" not in lowered:
                continue
            if not STATISTIC_RE.search(sentence):
                continue
            value = parse_float(sentence)
            rows.append(
                make_schema_row(
                    source,
                    page_number,
                    sentence,
                    extraction_method="pdf_text",
                    metric_name="reported_exposure_statistic",
                    metric_value=value,
                    confidence="low" if value is None else "medium",
                )
            )
    return dedupe_rows(rows, key_fields=["doc_type", "impact_type", "source_page", "evidence_text"])


def dedupe_rows(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    """Deduplicate rows by normalized selected fields."""
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(one_line(row.get(field)).lower() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def candidate_llm_chunks(method_chunks: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Pick chunks that are worth sending to the optional LLM extractor."""
    candidates = [
        chunk
        for chunk in method_chunks
        if any(term in one_line(chunk.get("chunk_text")).lower() for term in ("exposure", "direct", "indirect", "core", "supplemental", "firm"))
    ]
    if limit is not None:
        return candidates[:limit]
    return candidates


def make_openai_client() -> Any:
    """Create an OpenAI client if credentials are available."""
    load_dotenv_if_available()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available; skipping LLM extraction.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed; skipping LLM extraction.") from exc
    return OpenAI()


def call_llm(client: Any, model: str, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract rows from one text chunk with an LLM."""
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Page: {chunk.get('source_page')}\n\n"
                f"Text:\n{chunk.get('chunk_text')}\n\n"
                "Return JSON list only."
            ),
        },
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1600,
        "response_format": {"type": "json_object"},
    }
    try:
        response = client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("response_format", None)
        response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or "[]"
    return parse_llm_rows(content)


def parse_llm_rows(content: str) -> list[dict[str, Any]]:
    """Parse LLM JSON that may be a list or an object wrapping a list."""
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        parsed = json.loads(text[start : end + 1])
    if isinstance(parsed, dict):
        for key in ("rows", "claims", "evidence"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    return parsed if isinstance(parsed, list) else []


def llm_row_to_schema(source: dict[str, Any], raw: dict[str, Any], fallback_page: Any) -> dict[str, Any] | None:
    """Convert an LLM extraction row to the strict schema."""
    if not isinstance(raw, dict):
        return None
    evidence = one_line(raw.get("evidence_text") or raw.get("claim") or raw.get("scope"))
    excerpt = one_line(raw.get("supporting_excerpt") or raw.get("quote") or evidence)
    if not evidence or not excerpt:
        return None
    text = evidence if excerpt in evidence else f"{evidence} Supporting excerpt: {excerpt}"
    page = raw.get("page") or fallback_page
    value = parse_float(raw.get("metric_value"))
    row = make_schema_row(
        source,
        int(page) if str(page).isdigit() else None,
        text,
        extraction_method="llm_from_text",
        metric_name=one_line(raw.get("metric_name")) or None,
        metric_value=value,
        confidence=one_line(raw.get("confidence")).lower() if one_line(raw.get("confidence")).lower() in {"high", "medium", "low"} else "low",
    )
    row["supporting_excerpt"] = excerpt
    return row


def llm_rows(source: dict[str, Any], method_chunks: list[dict[str, Any]], model: str, limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run optional LLM extraction and return rows/errors."""
    errors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        client = make_openai_client()
    except Exception as exc:
        errors.append(error_row("llm_setup", str(exc)))
        return rows, errors

    chunks = candidate_llm_chunks(method_chunks, limit=limit)
    for index, chunk in enumerate(chunks, start=1):
        try:
            raw_rows = call_llm(client, model, chunk)
            for raw in raw_rows:
                row = llm_row_to_schema(source, raw, chunk.get("source_page"))
                if row:
                    rows.append(row)
        except Exception as exc:
            errors.append(error_row(f"llm_chunk_{index}", f"{type(exc).__name__}: {exc}", chunk.get("source_page")))
        time.sleep(0.2)
    return dedupe_rows(rows, key_fields=["doc_type", "impact_type", "source_page", "evidence_text"]), errors


def error_row(stage: str, message: str, page: Any = None) -> dict[str, Any]:
    """Build an extraction error row."""
    return {
        "source_id": "nber_w31222",
        "stage": stage,
        "source_page": page,
        "error_message": message,
        "created_at": created_at(),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Extract NBER W31222 AI exposure evidence.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--method-output", default=str(DEFAULT_METHOD_OUTPUT))
    parser.add_argument("--errors", default=str(DEFAULT_ERRORS))
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=12,
        help="Maximum method chunks to send to the optional LLM extractor.",
    )
    return parser.parse_args()


def main() -> int:
    """Run extraction."""
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest)
    output_path = resolve_project_path(args.output)
    method_output_path = resolve_project_path(args.method_output)
    errors_path = resolve_project_path(args.errors)

    errors: list[dict[str, Any]] = []
    source = find_nber_source(manifest_path)
    pdf_path = resolve_project_path(one_line(source.get("local_path")))
    if not pdf_path.exists():
        raise FileNotFoundError(f"NBER W31222 PDF not found: {pdf_path}")

    pages = extract_pdf_pages(pdf_path)
    method_chunks = make_method_chunks(source, pages)
    rows = rule_based_rows(source, pages)

    if args.use_llm:
        extracted, llm_errors = llm_rows(
            source,
            method_chunks,
            model=args.llm_model,
            limit=args.llm_limit,
        )
        rows.extend(extracted)
        errors.extend(llm_errors)
    else:
        errors.append(error_row("llm_skipped", "LLM extraction was not requested."))

    rows = dedupe_rows(rows, key_fields=["doc_type", "impact_type", "source_page", "metric_name", "metric_value", "evidence_text"])
    write_jsonl(rows, output_path)
    write_jsonl(method_chunks, method_output_path)
    write_jsonl(errors, errors_path)

    print("NBER W31222 extraction complete")
    print(f"PDF: {pdf_path}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Structured evidence rows: {len(rows)}")
    print(f"Method chunks: {len(method_chunks)}")
    print(f"Errors/warnings: {len(errors)}")
    print(f"Evidence output: {output_path}")
    print(f"Method output: {method_output_path}")
    print(f"Errors output: {errors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
