#!/usr/bin/env python3
"""Extract AI/job-impact evidence claims from research chunks.

Flow:
research_chunks.jsonl -> LLM extraction -> quote validation -> deduplication
-> ai_impact_claims.jsonl

The raw output is intentionally one JSONL row per chunk. That makes resume
behavior cheap and reliable while preserving every claim returned by the LLM.
"""

from pathlib import Path
import argparse
import json
import os
import re
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "data" / "research"

DEFAULT_INPUT = RESEARCH_DIR / "research_chunks.jsonl"
DEFAULT_RAW_OUTPUT = RESEARCH_DIR / "ai_impact_claims_raw.jsonl"
DEFAULT_OUTPUT = RESEARCH_DIR / "ai_impact_claims.jsonl"
DEFAULT_SUMMARY = RESEARCH_DIR / "claim_extraction_summary.csv"
DEFAULT_MODEL = "gpt-4o-mini"

REQUIRED_CLEAN_FIELDS = [
    "claim_id",
    "source_id",
    "chunk_id",
    "claim_text",
    "evidence_quote",
]

CLAIM_OUTPUT_FIELDS = [
    "claim_id",
    "source_id",
    "chunk_id",
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
    "page_start",
    "page_end",
    "section_title",
    "claim_text",
    "evidence_quote",
    "impact_type",
    "impact_direction",
    "affected_entity_type",
    "affected_entity_text",
    "occupation_or_skill_mentions",
    "evidence_strength",
    "confidence",
    "quote_validation",
    "review_status",
]

SUMMARY_FIELDS = [
    "chunk_id",
    "source_id",
    "title",
    "processed",
    "num_claims_raw",
    "num_claims_validated",
    "num_claims_failed_quote_validation",
    "error_message",
]

IMPACT_TYPES = {
    "automation_risk",
    "augmentation_potential",
    "llm_exposure",
    "task_displacement",
    "task_transformation",
    "skill_change",
    "employment_effect",
    "wage_effect",
    "productivity_effect",
    "job_creation",
    "job_loss",
    "inequality_or_polarization",
    "worker_vulnerability",
    "worker_advantage",
    "education_or_training_need",
    "human_advantage",
    "ethical_concern",
    "bias_concern",
    "surveillance_concern",
    "safety_concern",
    "uncertainty_or_limitation",
    "other",
}

IMPACT_DIRECTIONS = {"positive", "negative", "mixed", "uncertain"}

AFFECTED_ENTITY_TYPES = {
    "occupation",
    "task",
    "skill",
    "industry",
    "worker_group",
    "education",
    "general_labor_market",
    "other",
}

STRENGTH_VALUES = {"high", "medium", "low"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
VALID_QUOTE_STATUSES = {"exact_match", "normalized_match"}

SYSTEM_PROMPT = (
    "You are an evidence extraction assistant for a career guidance RAG system. "
    "Extract only claims directly supported by the provided passage. Do not add "
    "outside knowledge. Every claim must include a short exact quote copied from "
    "the passage."
)

USER_PROMPT_TEMPLATE = """Extract all important claims from the passage related to AI impact on jobs, automation, augmentation, skills, wages, employment, productivity, inequality, worker concerns, ethics, bias, surveillance, education/training needs, or uncertainty.

Rules:

1. Extract only claims supported by the passage.
2. Do not use outside knowledge.
3. Each claim must be atomic: one idea per claim.
4. Each claim must include a short exact quote from the passage.
5. The evidence_quote must be copied exactly from the passage.
6. If there are no relevant claims, return an empty claims list.
7. Prefer claims that could help explain AI impact for occupations, tasks, or skills.
8. Return valid JSON only.

Return schema:
{{
  "claims": [
    {{
      "claim_text": "...",
      "evidence_quote": "...",
      "impact_type": "automation_risk | augmentation_potential | llm_exposure | task_displacement | task_transformation | skill_change | employment_effect | wage_effect | productivity_effect | job_creation | job_loss | inequality_or_polarization | worker_vulnerability | worker_advantage | education_or_training_need | human_advantage | ethical_concern | bias_concern | surveillance_concern | safety_concern | uncertainty_or_limitation | other",
      "impact_direction": "positive | negative | mixed | uncertain",
      "affected_entity_type": "occupation | task | skill | industry | worker_group | education | general_labor_market | other",
      "affected_entity_text": "...",
      "occupation_or_skill_mentions": ["..."],
      "evidence_strength": "high | medium | low",
      "confidence": "high | medium | low"
    }}
  ]
}}

Passage metadata:
Title: {title}
Authors: {authors}
Year: {year}
Source ID: {source_id}
Chunk ID: {chunk_id}
Page range: {page_range}

Passage:
{chunk_text}
"""


def resolve_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_dotenv_if_available():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def make_openai_client():
    load_dotenv_if_available()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Set it in the environment or in the project .env file."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Python package is not installed. Install requirements in the active environment."
        ) from exc

    return OpenAI()


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSONL row in {path} line {line_number}: {exc}")

    return rows


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_claim_text(text):
    return normalize_whitespace(text).lower()


def clean_string(value):
    if value is None:
        return ""
    return str(value).strip()


def clean_choice(value, allowed_values, default_value):
    value = clean_string(value).lower()
    if value in allowed_values:
        return value
    return default_value


def clean_mentions(value):
    if isinstance(value, list):
        return [clean_string(item) for item in value if clean_string(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def validate_quote(evidence_quote, chunk_text):
    quote = clean_string(evidence_quote)
    text = str(chunk_text or "")

    if not quote:
        return "failed"
    if quote in text:
        return "exact_match"

    normalized_quote = normalize_whitespace(quote)
    normalized_text = normalize_whitespace(text)
    if normalized_quote and normalized_quote in normalized_text:
        return "normalized_match"

    return "failed"


def chunk_metadata(chunk):
    return {
        "source_id": clean_string(chunk.get("source_id")),
        "chunk_id": clean_string(chunk.get("chunk_id")),
        "chunk_index": chunk.get("chunk_index"),
        "source_type": clean_string(chunk.get("source_type")),
        "title": clean_string(chunk.get("title")),
        "authors": clean_string(chunk.get("authors")),
        "year": clean_string(chunk.get("year")),
        "doi": clean_string(chunk.get("doi")),
        "url": clean_string(chunk.get("url")),
        "final_url": clean_string(chunk.get("final_url")),
        "access_date": clean_string(chunk.get("access_date")),
        "local_path": clean_string(chunk.get("local_path")),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "section_title": clean_string(chunk.get("section_title")),
    }


def page_range_text(chunk):
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")

    if page_start is None and page_end is None:
        return "web"
    if page_start == "" and page_end == "":
        return "web"
    return f"{page_start}-{page_end}"


def build_user_prompt(chunk):
    return USER_PROMPT_TEMPLATE.format(
        title=clean_string(chunk.get("title")) or "(untitled)",
        authors=clean_string(chunk.get("authors")) or "(unknown)",
        year=clean_string(chunk.get("year")) or "(unknown)",
        source_id=clean_string(chunk.get("source_id")),
        chunk_id=clean_string(chunk.get("chunk_id")),
        page_range=page_range_text(chunk),
        chunk_text=clean_string(chunk.get("text")),
    )


def call_llm_for_chunk(client, model, chunk):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(chunk)},
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
    }

    last_error = None
    for attempt in range(1, 4):
        try:
            try:
                response = client.chat.completions.create(**kwargs)
            except TypeError:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("response_format", None)
                response = client.chat.completions.create(**fallback_kwargs)

            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("OpenAI returned an empty response.")
            return content.strip()

        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"OpenAI API call failed after 3 attempts: {last_error}")


def strip_json_fences(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_llm_json(content):
    cleaned = strip_json_fences(content)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def sanitize_claim(raw_claim, chunk_text, raw_claim_index):
    if not isinstance(raw_claim, dict):
        return {
            "raw_claim_index": raw_claim_index,
            "claim_text": "",
            "evidence_quote": "",
            "impact_type": "other",
            "impact_direction": "uncertain",
            "affected_entity_type": "other",
            "affected_entity_text": "",
            "occupation_or_skill_mentions": [],
            "evidence_strength": "low",
            "confidence": "low",
            "quote_validation": "failed",
            "raw_claim": raw_claim,
        }

    claim = {
        "raw_claim_index": raw_claim_index,
        "claim_text": clean_string(raw_claim.get("claim_text")),
        "evidence_quote": clean_string(raw_claim.get("evidence_quote")),
        "impact_type": clean_choice(raw_claim.get("impact_type"), IMPACT_TYPES, "other"),
        "impact_direction": clean_choice(
            raw_claim.get("impact_direction"), IMPACT_DIRECTIONS, "uncertain"
        ),
        "affected_entity_type": clean_choice(
            raw_claim.get("affected_entity_type"), AFFECTED_ENTITY_TYPES, "other"
        ),
        "affected_entity_text": clean_string(raw_claim.get("affected_entity_text")),
        "occupation_or_skill_mentions": clean_mentions(
            raw_claim.get("occupation_or_skill_mentions")
        ),
        "evidence_strength": clean_choice(
            raw_claim.get("evidence_strength"), STRENGTH_VALUES, "low"
        ),
        "confidence": clean_choice(raw_claim.get("confidence"), CONFIDENCE_VALUES, "low"),
    }
    claim["quote_validation"] = validate_quote(claim["evidence_quote"], chunk_text)
    return claim


def build_raw_row(chunk, model, raw_response_text="", parsed_response=None, error_message=""):
    text = str(chunk.get("text") or "")
    parsed_response = parsed_response or {}
    raw_claims = parsed_response.get("claims", [])

    if raw_claims is None:
        raw_claims = []
    if not isinstance(raw_claims, list):
        error_message = error_message or "Parsed JSON field 'claims' was not a list."
        raw_claims = []

    claims = [
        sanitize_claim(raw_claim, text, raw_claim_index=index)
        for index, raw_claim in enumerate(raw_claims, start=1)
    ]

    return {
        **chunk_metadata(chunk),
        "model": model,
        "processed": not bool(error_message),
        "num_claims_raw": len(claims),
        "num_claims_validated": sum(
            1 for claim in claims if claim["quote_validation"] in VALID_QUOTE_STATUSES
        ),
        "num_claims_failed_quote_validation": sum(
            1 for claim in claims if claim["quote_validation"] == "failed"
        ),
        "error_message": error_message,
        "raw_response_text": raw_response_text,
        "claims": claims,
    }


def process_chunk(client, model, chunk):
    raw_response_text = ""

    try:
        raw_response_text = call_llm_for_chunk(client, model, chunk)
    except Exception as exc:
        return build_raw_row(
            chunk,
            model,
            raw_response_text=raw_response_text,
            error_message=str(exc),
        )

    try:
        parsed_response = parse_llm_json(raw_response_text)
    except Exception as exc:
        return build_raw_row(
            chunk,
            model,
            raw_response_text=raw_response_text,
            error_message=f"JSON parse error: {exc}",
        )

    if not isinstance(parsed_response, dict):
        return build_raw_row(
            chunk,
            model,
            raw_response_text=raw_response_text,
            error_message="JSON parse error: top-level response was not an object.",
        )

    return build_raw_row(
        chunk,
        model,
        raw_response_text=raw_response_text,
        parsed_response=parsed_response,
    )


def latest_raw_rows_by_chunk(raw_rows):
    latest = {}
    for row in raw_rows:
        chunk_id = clean_string(row.get("chunk_id"))
        if chunk_id:
            latest[chunk_id] = row
    return latest


def already_processed_chunk_ids(raw_rows):
    return set(latest_raw_rows_by_chunk(raw_rows))


def claim_score(claim):
    value_scores = {"high": 3, "medium": 2, "low": 1}
    quote_score = 2 if claim.get("quote_validation") == "exact_match" else 1
    return (
        value_scores.get(claim.get("confidence"), 0),
        value_scores.get(claim.get("evidence_strength"), 0),
        quote_score,
        len(clean_string(claim.get("evidence_quote"))),
    )


def claims_are_near_duplicates(normalized_a, normalized_b):
    if normalized_a == normalized_b:
        return True
    if not normalized_a or not normalized_b:
        return False

    shorter, longer = sorted([normalized_a, normalized_b], key=len)
    if len(shorter) >= 40 and shorter in longer and len(shorter) / len(longer) >= 0.9:
        return True

    tokens_a = set(normalized_a.split())
    tokens_b = set(normalized_b.split())
    if len(tokens_a) < 6 or len(tokens_b) < 6:
        return False

    overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
    return overlap >= 0.92


def deduplicate_claims(claim_rows):
    kept = []

    for row in claim_rows:
        source_id = row["source_id"]
        normalized_text = normalize_claim_text(row["claim_text"])

        duplicate_index = None
        for index, kept_row in enumerate(kept):
            if kept_row["source_id"] != source_id:
                continue
            kept_normalized = normalize_claim_text(kept_row["claim_text"])
            if claims_are_near_duplicates(normalized_text, kept_normalized):
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(row)
            continue

        if claim_score(row) > claim_score(kept[duplicate_index]):
            kept[duplicate_index] = row

    for index, row in enumerate(kept, start=1):
        row["claim_id"] = f"claim_{index:06d}"

    return kept


def build_clean_claim_rows(raw_rows, input_chunks):
    latest = latest_raw_rows_by_chunk(raw_rows)
    ordered_chunk_ids = [clean_string(chunk.get("chunk_id")) for chunk in input_chunks]
    raw_rows_in_order = []
    seen = set()

    for chunk_id in ordered_chunk_ids:
        if chunk_id and chunk_id in latest:
            raw_rows_in_order.append(latest[chunk_id])
            seen.add(chunk_id)

    for row in raw_rows:
        chunk_id = clean_string(row.get("chunk_id"))
        if chunk_id and chunk_id not in seen and latest.get(chunk_id) is row:
            raw_rows_in_order.append(row)
            seen.add(chunk_id)

    clean_rows = []
    for raw_row in raw_rows_in_order:
        metadata = {
            field: raw_row.get(field)
            for field in CLAIM_OUTPUT_FIELDS
            if field
            not in {
                "claim_id",
                "claim_text",
                "evidence_quote",
                "impact_type",
                "impact_direction",
                "affected_entity_type",
                "affected_entity_text",
                "occupation_or_skill_mentions",
                "evidence_strength",
                "confidence",
                "quote_validation",
                "review_status",
            }
        }

        for claim in raw_row.get("claims", []):
            if claim.get("quote_validation") not in VALID_QUOTE_STATUSES:
                continue
            if not clean_string(claim.get("claim_text")):
                continue
            if not clean_string(claim.get("evidence_quote")):
                continue

            row = {
                **metadata,
                "claim_id": "",
                "claim_text": clean_string(claim.get("claim_text")),
                "evidence_quote": clean_string(claim.get("evidence_quote")),
                "impact_type": clean_choice(claim.get("impact_type"), IMPACT_TYPES, "other"),
                "impact_direction": clean_choice(
                    claim.get("impact_direction"), IMPACT_DIRECTIONS, "uncertain"
                ),
                "affected_entity_type": clean_choice(
                    claim.get("affected_entity_type"), AFFECTED_ENTITY_TYPES, "other"
                ),
                "affected_entity_text": clean_string(claim.get("affected_entity_text")),
                "occupation_or_skill_mentions": clean_mentions(
                    claim.get("occupation_or_skill_mentions")
                ),
                "evidence_strength": clean_choice(
                    claim.get("evidence_strength"), STRENGTH_VALUES, "low"
                ),
                "confidence": clean_choice(claim.get("confidence"), CONFIDENCE_VALUES, "low"),
                "quote_validation": claim.get("quote_validation"),
                "review_status": "pending",
            }
            clean_rows.append({field: row.get(field, "") for field in CLAIM_OUTPUT_FIELDS})

    return deduplicate_claims(clean_rows)


def write_summary(raw_rows, input_chunks, path):
    latest = latest_raw_rows_by_chunk(raw_rows)
    summary_rows = []

    for chunk in input_chunks:
        chunk_id = clean_string(chunk.get("chunk_id"))
        raw_row = latest.get(chunk_id)

        if raw_row is None:
            summary_rows.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": clean_string(chunk.get("source_id")),
                    "title": clean_string(chunk.get("title")),
                    "processed": False,
                    "num_claims_raw": 0,
                    "num_claims_validated": 0,
                    "num_claims_failed_quote_validation": 0,
                    "error_message": "Not processed in this run.",
                }
            )
            continue

        summary_rows.append(
            {
                "chunk_id": chunk_id,
                "source_id": clean_string(chunk.get("source_id")),
                "title": clean_string(chunk.get("title")),
                "processed": bool(raw_row.get("processed")),
                "num_claims_raw": int(raw_row.get("num_claims_raw") or 0),
                "num_claims_validated": int(raw_row.get("num_claims_validated") or 0),
                "num_claims_failed_quote_validation": int(
                    raw_row.get("num_claims_failed_quote_validation") or 0
                ),
                "error_message": clean_string(raw_row.get("error_message")),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows, columns=SUMMARY_FIELDS).to_csv(path, index=False)
    return summary_rows


def count_raw_claims(raw_rows):
    latest = latest_raw_rows_by_chunk(raw_rows)
    rows = list(latest.values())
    total = sum(int(row.get("num_claims_raw") or 0) for row in rows)
    valid = sum(int(row.get("num_claims_validated") or 0) for row in rows)
    failed = sum(int(row.get("num_claims_failed_quote_validation") or 0) for row in rows)
    return total, valid, failed


def validate_clean_output(path=DEFAULT_OUTPUT, preview_count=5):
    print("\nValidation")

    if not path.exists():
        print(f"Missing output file: {path}")
        return False

    rows = read_jsonl(path)
    errors = []

    for index, row in enumerate(rows, start=1):
        for field in REQUIRED_CLEAN_FIELDS:
            if not clean_string(row.get(field)):
                errors.append(f"Line {index}: missing or empty '{field}'")

        if row.get("quote_validation") not in VALID_QUOTE_STATUSES:
            errors.append(
                f"Line {index}: quote_validation is '{row.get('quote_validation')}'"
            )

    if errors:
        print(f"Validation failed with {len(errors)} issue(s).")
        for error in errors[:10]:
            print(f"- {error}")
        if len(errors) > 10:
            print(f"- ... {len(errors) - 10} more")
        return False

    print(f"Validated {len(rows)} clean claim row(s).")
    print(f"Showing first {min(preview_count, len(rows))} clean claim(s):")

    for row in rows[:preview_count]:
        title = row.get("title") or "(untitled)"
        page_start = row.get("page_start")
        page_end = row.get("page_end")
        if page_start is None and page_end is None:
            page_range = "web"
        elif page_start == page_end:
            page_range = f"page {page_start}"
        else:
            page_range = f"pages {page_start}-{page_end}"

        print(f"\n{row.get('claim_id')} | {title} | {page_range}")
        print(f"Claim: {row.get('claim_text')}")
        print(f"Quote: {row.get('evidence_quote')}")

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract AI/job-impact claims from research chunks with quote validation."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = resolve_path(args.input)
    raw_output_path = resolve_path(args.raw_output)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary)

    if not input_path.exists():
        raise FileNotFoundError(f"Input chunks file not found: {input_path}")

    chunks = read_jsonl(input_path)
    if args.limit is not None:
        chunks = chunks[: args.limit]

    existing_raw_rows = read_jsonl(raw_output_path)
    processed_ids = already_processed_chunk_ids(existing_raw_rows) if args.resume else set()

    chunks_to_process = [
        chunk
        for chunk in chunks
        if clean_string(chunk.get("chunk_id")) not in processed_ids
    ]
    skipped_existing = len(chunks) - len(chunks_to_process)

    client = None
    if chunks_to_process:
        client = make_openai_client()

    processed_this_run = 0

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = clean_string(chunk.get("chunk_id"))

        if args.resume and chunk_id in processed_ids:
            if index % 25 == 0:
                print(
                    f"Progress: {index}/{len(chunks)} chunks considered "
                    f"({processed_this_run} processed this run, {skipped_existing} skipped)."
                )
            continue

        raw_row = process_chunk(client, args.model, chunk)
        append_jsonl(raw_row, raw_output_path)
        processed_this_run += 1

        if processed_this_run % 25 == 0 or index == len(chunks):
            print(
                f"Progress: {index}/{len(chunks)} chunks considered "
                f"({processed_this_run} processed this run, {skipped_existing} skipped)."
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    raw_rows = read_jsonl(raw_output_path)
    clean_claims = build_clean_claim_rows(raw_rows, chunks)
    write_jsonl(clean_claims, output_path)
    write_summary(raw_rows, chunks, summary_path)

    total_raw_claims, valid_claims, failed_claims = count_raw_claims(raw_rows)

    print("\nAI/job-impact claim extraction complete")
    print(f"Total chunks read: {len(chunks)}")
    print(f"Chunks processed in this run: {processed_this_run}")
    print(f"Chunks skipped because already processed: {skipped_existing}")
    print(f"Total raw claims: {total_raw_claims}")
    print(f"Claims passing quote validation: {valid_claims}")
    print(f"Claims failing quote validation: {failed_claims}")
    print(f"Claims after deduplication: {len(clean_claims)}")
    print(f"Raw output path: {raw_output_path}")
    print(f"Clean output path: {output_path}")
    print(f"Summary path: {summary_path}")

    validate_clean_output(output_path)


if __name__ == "__main__":
    main()
