#!/usr/bin/env python3
"""Post-process AI/job-impact claims without making API calls.

Inputs:
- research_chunks.jsonl
- ai_impact_claims_raw.jsonl

Outputs:
- ai_impact_claims_postprocessed.jsonl
- ai_impact_claims_failed_quotes.jsonl
- ai_impact_claims_postprocess_summary.csv
- label_cleaning_warnings.csv

Optionally, --overwrite-clean writes the postprocessed clean claims back to
data/research/ai_impact_claims.jsonl.
"""

from pathlib import Path
from difflib import SequenceMatcher
import argparse
import json
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "data" / "research"

DEFAULT_CHUNKS = RESEARCH_DIR / "research_chunks.jsonl"
DEFAULT_RAW_CLAIMS = RESEARCH_DIR / "ai_impact_claims_raw.jsonl"
DEFAULT_OUTPUT = RESEARCH_DIR / "ai_impact_claims_postprocessed.jsonl"
DEFAULT_FAILED_OUTPUT = RESEARCH_DIR / "ai_impact_claims_failed_quotes.jsonl"
DEFAULT_SUMMARY = RESEARCH_DIR / "ai_impact_claims_postprocess_summary.csv"
DEFAULT_WARNINGS = RESEARCH_DIR / "label_cleaning_warnings.csv"
DEFAULT_CLEAN_OUTPUT = RESEARCH_DIR / "ai_impact_claims.jsonl"

FINAL_FIELDS = [
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
    "impact_type_clean",
    "impact_type_clean_reason",
    "impact_direction",
    "impact_direction_clean",
    "impact_direction_clean_reason",
    "affected_entity_type",
    "affected_entity_text",
    "occupation_or_skill_mentions",
    "evidence_strength",
    "confidence",
    "quote_validation",
    "quote_validation_score",
    "quote_validation_method_detail",
    "ai_relevance",
    "generator_use_scope",
    "review_status",
    "dedup_status",
    "duplicate_group_id",
]

FAILED_FIELDS = [
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
    "raw_claim_index",
    "claim_text",
    "evidence_quote",
    "impact_type",
    "impact_type_clean",
    "impact_type_clean_reason",
    "impact_direction",
    "impact_direction_clean",
    "impact_direction_clean_reason",
    "affected_entity_type",
    "affected_entity_text",
    "occupation_or_skill_mentions",
    "evidence_strength",
    "confidence",
    "original_quote_validation",
    "quote_validation",
    "quote_validation_score",
    "quote_validation_method_detail",
    "ai_relevance",
    "generator_use_scope",
]

SUMMARY_FIELDS = [
    "source_id",
    "title",
    "raw_claims",
    "validated_claims_before_dedup",
    "failed_quote_claims",
    "kept_after_dedup",
    "duplicates_removed",
    "direct_ai_claims",
    "automation_or_technology_claims",
    "indirect_labor_context_claims",
    "non_ai_context_claims",
    "job_specific_claims",
    "task_or_skill_linkable_claims",
    "background_context_claims",
    "excluded_claims",
]

WARNING_FIELDS = [
    "claim_id",
    "source_id",
    "chunk_id",
    "claim_text",
    "evidence_quote",
    "impact_type",
    "impact_type_clean",
    "impact_direction",
    "impact_direction_clean",
    "warning_type",
]

VALID_IMPACT_TYPES = {
    "automation_exposure",
    "automation_risk",
    "augmentation_potential",
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
    "labor_market_context",
    "other",
}

VALID_DIRECTIONS = {"positive", "negative", "mixed", "uncertain"}
VALID_RELEVANCE = {
    "direct_ai",
    "automation_or_technology",
    "indirect_labor_context",
    "non_ai_context",
}
VALID_SCOPES = {
    "job_specific",
    "task_or_skill_linkable",
    "background_context",
    "exclude",
}
VALID_QUOTE_STATUSES = {
    "exact_match",
    "normalized_match",
    "relaxed_punctuation_match",
    "fuzzy_match",
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}

STRONG_ORIGINAL_IMPACT_TYPES = {
    "worker_vulnerability",
    "worker_advantage",
    "job_loss",
    "job_creation",
    "inequality_or_polarization",
    "ethical_concern",
    "bias_concern",
    "surveillance_concern",
    "safety_concern",
    "uncertainty_or_limitation",
    "education_or_training_need",
    "skill_change",
    "wage_effect",
    "productivity_effect",
}


def resolve_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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


def clean_string(value):
    if value is None:
        return ""
    return str(value).strip()


def clean_choice(value, valid_values, default):
    value = clean_string(value).lower()
    if value in valid_values:
        return value
    return default


def clean_mentions(value):
    if isinstance(value, list):
        return [clean_string(item) for item in value if clean_string(item)]
    if isinstance(value, str) and value.strip():
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [clean_string(item) for item in parsed if clean_string(item)]
            except json.JSONDecodeError:
                pass
        return [stripped]
    return []


def text_for_rules(row):
    pieces = [
        row.get("claim_text", ""),
        row.get("evidence_quote", ""),
        row.get("affected_entity_text", ""),
    ]
    pieces.extend(clean_mentions(row.get("occupation_or_skill_mentions")))
    return " ".join(clean_string(piece) for piece in pieces if clean_string(piece))


def text_for_label_rules(row, include_evidence=True):
    pieces = [row.get("claim_text", ""), row.get("affected_entity_text", "")]
    if include_evidence:
        pieces.append(row.get("evidence_quote", ""))
    pieces.extend(clean_mentions(row.get("occupation_or_skill_mentions")))
    return " ".join(clean_string(piece) for piece in pieces if clean_string(piece))


def normalized_lower(text):
    return normalize_basic(text).lower()


def normalize_basic(text):
    """Unicode, ligature, quote, dash, hyphen-line, and whitespace normalization."""
    text = clean_string(text)
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00ef\u00ac\u0081": "fi",
        "\u00ef\u00ac\u0082": "fl",
        "\u00ef\u00ac\u0083": "ffi",
        "\u00ef\u00ac\u0084": "ffl",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u2122": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00e2\u20ac\u201c": "-",
        "\u00e2\u20ac\u201d": "-",
        "\u2026": "...",
        "\u00e2\u20ac\u00a6": "...",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_relaxed(text, keep_gap=False):
    text = normalize_basic(text).lower()
    if keep_gap:
        text = re.sub(r"\.\.\.|…", " <gap> ", text)
    else:
        text = re.sub(r"\.\.\.|…", " ", text)
    text = re.sub(r"[,.;:(){}\[\]\"']", " ", text)
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^\w\s<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_relaxed_quote(relaxed_quote, relaxed_chunk):
    if not relaxed_quote:
        return False
    if "<gap>" not in relaxed_quote:
        return relaxed_quote in relaxed_chunk

    parts = [part.strip() for part in relaxed_quote.split("<gap>") if part.strip()]
    if not parts:
        return False

    position = 0
    for part in parts:
        found = relaxed_chunk.find(part, position)
        if found == -1:
            return False
        position = found + len(part)
    return True


def fuzzy_quote_match(quote, chunk_text):
    normalized_quote = normalize_for_relaxed(quote)
    normalized_chunk = normalize_for_relaxed(chunk_text)

    if len(normalized_quote) < 40 or not normalized_chunk:
        return 0.0, "quote shorter than fuzzy threshold"

    quote_length = len(normalized_quote)
    if quote_length > len(normalized_chunk):
        return 0.0, "quote longer than chunk after normalization"

    seed_match = SequenceMatcher(None, normalized_quote, normalized_chunk).find_longest_match(
        0,
        quote_length,
        0,
        len(normalized_chunk),
    )

    step = max(20, quote_length // 4)
    starts = set()
    rough_start = max(0, seed_match.b - quote_length)
    rough_end = min(len(normalized_chunk), seed_match.b + quote_length)
    for start in range(rough_start, rough_end + 1, step):
        starts.add(start)
    for start in range(0, max(1, len(normalized_chunk) - quote_length + 1), step):
        starts.add(start)

    window_lengths = {
        max(20, int(quote_length * 0.75)),
        quote_length,
        min(len(normalized_chunk), int(quote_length * 1.25)),
        min(len(normalized_chunk), quote_length + 80),
    }

    best_score = 0.0
    best_detail = "no candidate window"

    for start in sorted(starts):
        for window_length in window_lengths:
            end = min(len(normalized_chunk), start + window_length)
            if end <= start:
                continue
            window = normalized_chunk[start:end]
            score = SequenceMatcher(None, normalized_quote, window).ratio()
            if score > best_score:
                best_score = score
                best_detail = f"best_ratio={score:.3f}; window_start={start}; window_end={end}"

    return best_score, best_detail


def validate_quote(evidence_quote, chunk_text):
    quote = clean_string(evidence_quote)
    chunk_text = clean_string(chunk_text)

    if not quote:
        return "failed", 0.0, "empty quote"
    if not chunk_text:
        return "failed", 0.0, "missing chunk text"

    if quote in chunk_text:
        return "exact_match", 1.0, "raw quote is an exact substring"

    normalized_quote = normalize_basic(quote)
    normalized_chunk = normalize_basic(chunk_text)
    if normalized_quote and normalized_quote in normalized_chunk:
        return "normalized_match", 1.0, "NFKC, quote, dash, ligature, and whitespace normalization"

    relaxed_quote = normalize_for_relaxed(quote, keep_gap=True)
    relaxed_chunk = normalize_for_relaxed(chunk_text)
    if contains_relaxed_quote(relaxed_quote, relaxed_chunk):
        return "relaxed_punctuation_match", 1.0, "punctuation-insensitive match with flexible ellipsis gaps"

    score, detail = fuzzy_quote_match(quote, chunk_text)
    if score >= 0.88:
        return "fuzzy_match", score, detail

    return "failed", score, detail


def load_chunks(path):
    chunks = read_jsonl(path)
    chunk_by_id = {}
    for chunk in chunks:
        chunk_id = clean_string(chunk.get("chunk_id"))
        if chunk_id:
            chunk_by_id[chunk_id] = chunk
    return chunk_by_id


def metadata_from_raw_row(raw_row):
    return {
        "source_id": clean_string(raw_row.get("source_id")),
        "chunk_id": clean_string(raw_row.get("chunk_id")),
        "chunk_index": raw_row.get("chunk_index"),
        "source_type": clean_string(raw_row.get("source_type")),
        "title": clean_string(raw_row.get("title")),
        "authors": clean_string(raw_row.get("authors")),
        "year": clean_string(raw_row.get("year")),
        "doi": clean_string(raw_row.get("doi")),
        "url": clean_string(raw_row.get("url")),
        "final_url": clean_string(raw_row.get("final_url")),
        "access_date": clean_string(raw_row.get("access_date")),
        "local_path": clean_string(raw_row.get("local_path")),
        "page_start": raw_row.get("page_start"),
        "page_end": raw_row.get("page_end"),
        "section_title": clean_string(raw_row.get("section_title")),
    }


def flatten_raw_claims(raw_rows):
    flat_rows = []
    for raw_row in raw_rows:
        metadata = metadata_from_raw_row(raw_row)
        claims = raw_row.get("claims", [])
        if not isinstance(claims, list):
            continue

        for position, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                continue

            row = {
                **metadata,
                "raw_claim_index": claim.get("raw_claim_index") or position,
                "claim_text": clean_string(claim.get("claim_text")),
                "evidence_quote": clean_string(claim.get("evidence_quote")),
                "impact_type": clean_choice(claim.get("impact_type"), set(), clean_string(claim.get("impact_type"))),
                "impact_direction": clean_choice(
                    claim.get("impact_direction"),
                    VALID_DIRECTIONS,
                    "uncertain",
                ),
                "affected_entity_type": clean_choice(
                    claim.get("affected_entity_type"),
                    {
                        "occupation",
                        "task",
                        "skill",
                        "industry",
                        "worker_group",
                        "education",
                        "general_labor_market",
                        "other",
                    },
                    "other",
                ),
                "affected_entity_text": clean_string(claim.get("affected_entity_text")),
                "occupation_or_skill_mentions": clean_mentions(
                    claim.get("occupation_or_skill_mentions")
                ),
                "evidence_strength": clean_choice(
                    claim.get("evidence_strength"),
                    {"high", "medium", "low"},
                    "low",
                ),
                "confidence": clean_choice(claim.get("confidence"), {"high", "medium", "low"}, "low"),
                "original_quote_validation": clean_string(claim.get("quote_validation")),
            }
            flat_rows.append(row)
    return flat_rows


def classify_ai_relevance(row):
    text = normalize_basic(text_for_rules(row)).lower()

    direct_patterns = [
        r"\bai\b",
        r"artificial intelligence",
        r"generative ai",
        r"\bgen ai\b",
        r"\bllm\b",
        r"\bllms\b",
        r"large language model",
        r"chatgpt",
        r"machine learning",
        r"algorithmic",
    ]
    automation_patterns = [
        r"automation",
        r"automated",
        r"automate",
        r"technology adoption",
        r"technological change",
        r"digitalization",
        r"digitalisation",
        r"robotics",
        r"software",
        r"computerization",
        r"computerisation",
    ]
    labor_patterns = [
        r"\bjob\b",
        r"\bjobs\b",
        r"occupation",
        r"occupational",
        r"worker",
        r"workers",
        r"employment",
        r"\blabor\b",
        r"\blabour\b",
        r"wage",
        r"skill",
        r"skills",
        r"training",
        r"productivity",
        r"transition",
        r"redeployment",
        r"displacement",
    ]

    if any(re.search(pattern, text) for pattern in direct_patterns):
        return "direct_ai"
    if any(re.search(pattern, text) for pattern in automation_patterns):
        return "automation_or_technology"
    if any(re.search(pattern, text) for pattern in labor_patterns):
        return "indirect_labor_context"
    return "non_ai_context"


def has_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def term_regex(term):
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    prefix = r"\b" if term[:1].isalnum() else ""
    suffix = r"\b" if term[-1:].isalnum() else ""
    return prefix + escaped + suffix


def matched_terms(text, terms):
    matches = []
    for term in terms:
        if re.search(term_regex(term), text):
            matches.append(term)
    return matches


def has_term(text, terms):
    return bool(matched_terms(text, terms))


def first_terms_reason(matches, max_terms=3):
    quoted = [f"'{match}'" for match in matches[:max_terms]]
    if not quoted:
        return ""
    return ", ".join(quoted)


NEGATIVE_DIRECTION_TERMS = [
    "limit",
    "limited",
    "slow",
    "slower",
    "decline",
    "declining",
    "decrease",
    "decreases",
    "fall",
    "falling",
    "loss",
    "losses",
    "job loss",
    "displacement",
    "displaced",
    "dislocation",
    "unable",
    "vulnerability",
    "vulnerable",
    "risk",
    "harm",
    "barrier",
    "hinder",
    "shortage",
    "concern",
    "fears",
    "threat",
    "lower",
    "reduced",
    "negative",
    "replacement",
    "replace",
]

POSITIVE_DIRECTION_TERMS = [
    "gain",
    "gains",
    "increase",
    "increases",
    "increasing",
    "rise",
    "rises",
    "rising",
    "growth",
    "grow",
    "improve",
    "improvement",
    "create",
    "creates",
    "creation",
    "opportunity",
    "opportunities",
    "advantage",
    "benefit",
    "benefits",
    "boost",
    "add",
    "higher demand",
    "successful",
    "effectively",
]

UNCERTAIN_DIRECTION_TERMS = [
    "could",
    "may",
    "might",
    "potential",
    "scenario",
    "exposure",
    "transition",
    "transitions",
    "estimate",
    "uncertain",
    "suggesting",
]

OCCUPATIONAL_TRANSITION_TERMS = [
    "occupational transition",
    "occupational transitions",
    "occupational shift",
    "occupational shifts",
    "workers transition",
    "worker transition",
    "declining occupations",
    "redeployment",
    "displacement",
]

AUTOMATION_EXPOSURE_PATTERNS = [
    r"\bcould\s+be\s+automated\b",
    r"\bmay\s+be\s+automated\b",
    r"\bmight\s+be\s+automated\b",
    r"\bhours\s+worked\s+could\s+be\s+automated\b",
    r"\bhours\s+automated\b",
    r"\btasks?\s+(?:could\s+be\s+)?automated\b",
    r"\bautomation\s+potential\b",
    r"\bautomatable\b",
    r"\bexposure\s+to\s+automation\b",
    r"\bexposure\s+to\s+ai\b",
    r"\bai\s+exposure\b",
]


def has_regex(text, patterns):
    return any(re.search(pattern, text) for pattern in patterns)


def has_occupational_transition(text):
    return has_term(text, OCCUPATIONAL_TRANSITION_TERMS)


def has_automation_exposure(text):
    return has_regex(text, AUTOMATION_EXPOSURE_PATTERNS)


def has_automation_risk(text):
    risk_terms = [
        "risk",
        "displacement",
        "displaced",
        "decline",
        "declining",
        "loss",
        "losses",
        "replacement",
        "replace",
        "job loss",
        "employment decline",
        "harm",
        "vulnerability",
        "vulnerable",
    ]
    automation_terms = [
        "automation",
        "automated",
        "automate",
        "automatable",
        "ai exposure",
        "exposure to ai",
        "exposure to automation",
    ]
    return has_term(text, automation_terms) and has_term(text, risk_terms)


def has_productivity_main_claim(claim_text):
    productivity_patterns = [
        r"\bproductivity\b",
        r"\bproductivity\s+growth\b",
        r"\btotal\s+factor\s+productivity\b",
        r"\boutput\b",
        r"\befficiency\b",
        r"\befficient\b",
    ]
    if not has_regex(claim_text, productivity_patterns):
        return False
    job_creation_patterns = [
        r"\bnew\s+job\s+roles?\b",
        r"\bjob\s+roles?\b",
        r"\bnew\s+jobs?\b",
        r"\bjobs?\s+(?:created|creation|added)\b",
        r"\bcreation\s+of\s+new\s+jobs?\b",
    ]
    return not has_regex(claim_text, job_creation_patterns)


def clean_impact_type_with_reason(row):
    claim_text = normalized_lower(text_for_label_rules(row, include_evidence=False))
    evidence_text = normalized_lower(row.get("evidence_quote", ""))
    text = normalized_lower(text_for_label_rules(row, include_evidence=True))
    original = clean_string(row.get("impact_type")).lower()
    ai_relevance = row.get("ai_relevance")
    affected_type = clean_string(row.get("affected_entity_type")).lower()

    if original == "llm_exposure":
        return "automation_exposure", "mapped legacy label llm_exposure to automation_exposure"

    concern_rules = [
        ("bias_concern", ["bias", "discrimination"]),
        ("surveillance_concern", ["surveillance", "monitoring"]),
        ("safety_concern", ["safety", "unsafe", "risk to safety"]),
        ("ethical_concern", ["ethic", "privacy", "fairness", "accountability"]),
    ]
    for label, terms in concern_rules:
        matches = matched_terms(claim_text, terms)
        if matches:
            return label, f"explicit concern pattern in claim: {first_terms_reason(matches)}"

    if has_occupational_transition(claim_text):
        vulnerability_matches = matched_terms(
            claim_text,
            ["unable", "vulnerability", "vulnerable", "left behind"],
        )
        if not vulnerability_matches and has_term(claim_text, ["worker", "workers"]):
            vulnerability_matches = matched_terms(claim_text, ["slow", "slower"])
        if vulnerability_matches:
            return (
                "worker_vulnerability",
                f"occupational transition wording with vulnerability terms {first_terms_reason(vulnerability_matches)}",
            )
        if has_term(claim_text, ["declining occupations"]) and has_term(
            claim_text,
            ["rising", "rising ones"],
        ):
            return (
                "task_transformation",
                "workers transition from declining occupations to rising ones",
            )
        if has_term(claim_text, ["occupational transition", "occupational transitions"]):
            return "employment_effect", "occupational transitions treated as employment_effect"
        if original in {"task_displacement", "task_transformation", "employment_effect"}:
            return (
                original if original != "task_displacement" else "task_transformation",
                "kept transition-related original label conservatively"
                if original != "task_displacement"
                else "transition wording is closer to task_transformation than displacement",
            )

    if has_automation_exposure(claim_text):
        if has_automation_risk(claim_text):
            if original in {"job_loss", "worker_vulnerability", "employment_effect"}:
                return original, f"kept original {original} despite automation risk wording"
            return "automation_risk", "explicit automation risk pattern in claim"
        return "automation_exposure", "claim describes measurable automation exposure or potential"

    if has_automation_risk(claim_text):
        if original in {"job_loss", "worker_vulnerability", "employment_effect"}:
            return original, f"kept original {original} despite automation risk wording"
        return "automation_risk", "explicit automation risk pattern in claim"

    training_matches = matched_terms(
        claim_text,
        ["training", "retraining", "reskilling", "education", "upskilling", "skills upgrade"],
    )
    if training_matches:
        if original in STRONG_ORIGINAL_IMPACT_TYPES:
            return original, f"kept original strong label {original}"
        return "education_or_training_need", f"training or education terms {first_terms_reason(training_matches)}"

    inequality_matches = matched_terms(
        claim_text,
        ["inequality", "polarization", "polarisation", "polarized", "polarised", "unequal"],
    )
    if inequality_matches:
        return "inequality_or_polarization", f"inequality terms {first_terms_reason(inequality_matches)}"

    worker_vulnerability_matches = matched_terms(
        claim_text,
        ["vulnerability", "vulnerable", "challenge", "barrier", "risk", "unable", "left behind"],
    )
    if worker_vulnerability_matches and (
        affected_type == "worker_group" or original == "worker_vulnerability"
    ):
        return (
            "worker_vulnerability",
            f"worker vulnerability terms {first_terms_reason(worker_vulnerability_matches)}",
        )

    job_loss_patterns = [
        r"\bjobs?\s+(?:loss|losses|lost|decline|declines|declined|fall|falling)\b",
        r"\bemployment\s+(?:decline|declines|declined|fall|falling|loss|losses)\b",
        r"\bdemand\s+for\s+(?:workers|occupations|jobs|professions).*\bdecline\b",
        r"\boccupations?.*\bdecline\b",
        r"\bworkers?.*\bdisplaced\b",
    ]
    if has_regex(claim_text, job_loss_patterns):
        if original == "wage_effect" and has_term(claim_text, ["wage", "wages", "earnings", "pay", "salary"]):
            return "wage_effect", "kept wage_effect because claim is about wages"
        return "job_loss", "explicit job or employment loss pattern in claim"

    job_creation_patterns = [
        r"\bnew\s+job\s+roles?\b",
        r"\bnew\s+jobs?\b",
        r"\bjobs?\s+(?:created|creation|added|gained)\b",
        r"\bcreate\s+(?:new\s+)?jobs?\b",
        r"\bdemand\s+for\s+(?:workers|occupations|jobs|professions).*\b(?:rise|rises|rising|increase|increases|grow|growth)\b",
    ]
    if has_regex(claim_text, job_creation_patterns):
        return "job_creation", "explicit job creation or rising demand pattern in claim"

    wage_matches = matched_terms(claim_text, ["wage", "wages", "earnings", "pay", "salary"])
    if wage_matches:
        return "wage_effect", f"wage terms {first_terms_reason(wage_matches)}"

    skill_specific_text = (
        re.search(r"\bskills?\b", claim_text)
        or has_term(
            claim_text,
            [
                "skill demand",
                "skill requirements",
                "skill needs",
                "skill change",
                "skill upgrade",
                "human skill",
                "technical skill",
            ],
        )
    )
    if affected_type == "skill" or skill_specific_text:
        return "skill_change", "claim is primarily about skills"

    if affected_type == "task" and has_term(claim_text, ["displace", "substitute", "replace"]):
        return "task_displacement", "task-level displacement pattern"
    if affected_type == "task" and has_term(claim_text, ["transform", "change", "redesign", "reweight"]):
        return "task_transformation", "task-level transformation pattern"

    if has_productivity_main_claim(claim_text):
        return "productivity_effect", "claim is directly about productivity, output, or efficiency"

    if original in STRONG_ORIGINAL_IMPACT_TYPES:
        return original, f"kept original strong label {original}"
    if original in VALID_IMPACT_TYPES:
        return original, f"kept original valid label {original}"

    if has_productivity_main_claim(evidence_text):
        return (
            "productivity_effect",
            "used evidence productivity pattern because original label was invalid",
        )
    if ai_relevance == "indirect_labor_context":
        return "labor_market_context", "fallback for indirect labor context with invalid original label"
    return "other", "fallback because original label was invalid"


def clean_impact_type(row):
    return clean_impact_type_with_reason(row)[0]


def clean_impact_direction_with_reason(row):
    claim_text = normalized_lower(text_for_label_rules(row, include_evidence=False))
    text = normalized_lower(text_for_label_rules(row, include_evidence=True))
    original = clean_string(row.get("impact_direction")).lower()
    impact_type_clean = row.get("impact_type_clean")

    claim_negative_matches = matched_terms(claim_text, NEGATIVE_DIRECTION_TERMS)
    text_negative_matches = matched_terms(text, NEGATIVE_DIRECTION_TERMS)
    claim_positive_matches = matched_terms(claim_text, POSITIVE_DIRECTION_TERMS)
    text_positive_matches = matched_terms(text, POSITIVE_DIRECTION_TERMS)
    uncertain_matches = matched_terms(text, UNCERTAIN_DIRECTION_TERMS)

    if has_occupational_transition(claim_text):
        if matched_terms(claim_text, ["slow", "slower", "unable", "displacement", "displaced"]):
            matches = matched_terms(
                claim_text,
                ["slow", "slower", "unable", "displacement", "displaced"],
            )
            return "negative", f"transition claim contains negative terms {first_terms_reason(matches)}"
        if has_term(claim_text, ["declining occupations"]) and has_term(
            claim_text,
            ["rising", "rising ones"],
        ):
            return "mixed", "transition from declining occupations to rising ones is mixed"
        if has_term(claim_text, ["occupational transition", "occupational transitions"]):
            return "uncertain", "occupational transitions are not automatically positive"

    if impact_type_clean == "automation_exposure" and not text_negative_matches:
        return "uncertain", "automation exposure or potential without explicit harm is uncertain"

    if original == "uncertain" and uncertain_matches:
        if claim_negative_matches and claim_positive_matches:
            return "mixed", "uncertain original with both negative and positive terms in claim"
        return "uncertain", f"kept uncertain because text contains {first_terms_reason(uncertain_matches)}"

    if original == "negative":
        if claim_negative_matches or text_negative_matches:
            matches = claim_negative_matches or text_negative_matches
            if claim_positive_matches and not matched_terms(
                claim_text,
                ["limit", "limited", "slow", "slower", "hinder", "reduced"],
            ):
                return (
                    "mixed",
                    f"negative original with both negative and positive terms; negative terms {first_terms_reason(matches)}",
                )
            return "negative", f"kept negative because text contains {first_terms_reason(matches)}"
        if has_automation_exposure(claim_text):
            return "uncertain", "automation exposure phrasing is potential rather than harm"
        return "negative", "kept original negative direction conservatively"

    if claim_negative_matches or text_negative_matches:
        matches = claim_negative_matches or text_negative_matches
        if claim_positive_matches:
            return (
                "mixed",
                f"text contains both negative and positive terms; negative terms {first_terms_reason(matches)}",
            )
        return "negative", f"text contains negative terms {first_terms_reason(matches)}"

    if claim_positive_matches or text_positive_matches:
        matches = claim_positive_matches or text_positive_matches
        return "positive", f"text contains positive terms {first_terms_reason(matches)}"

    if uncertain_matches:
        return "uncertain", f"text contains uncertainty terms {first_terms_reason(uncertain_matches)}"

    if original in VALID_DIRECTIONS:
        return original, f"kept original direction {original}"
    return "uncertain", "fallback because original direction was invalid"


def clean_impact_direction(row):
    return clean_impact_direction_with_reason(row)[0]


def has_concrete_affected_entity(row):
    entity = normalize_basic(row.get("affected_entity_text")).lower()
    if not entity:
        return False
    generic = {
        "occupation",
        "occupations",
        "job",
        "jobs",
        "workers",
        "worker",
        "workforce",
        "labor market",
        "labour market",
        "general labor market",
        "businesses",
    }
    return entity not in generic


def assign_generator_scope(row):
    if row.get("ai_relevance") == "non_ai_context":
        return "exclude"

    affected_type = clean_string(row.get("affected_entity_type")).lower()
    if affected_type == "occupation" and has_concrete_affected_entity(row):
        return "job_specific"
    if affected_type in {"task", "skill"}:
        return "task_or_skill_linkable"
    if affected_type in {"general_labor_market", "worker_group", "industry", "education"}:
        return "background_context"
    return "background_context"


def postprocess_flat_claims(flat_rows, chunk_by_id):
    validated_rows = []
    failed_rows = []

    for row in flat_rows:
        chunk_id = clean_string(row.get("chunk_id"))
        chunk = chunk_by_id.get(chunk_id, {})
        chunk_text = clean_string(chunk.get("text"))
        quote_status, quote_score, quote_detail = validate_quote(
            row.get("evidence_quote"),
            chunk_text,
        )

        row["quote_validation"] = quote_status
        row["quote_validation_score"] = round(float(quote_score), 4)
        row["quote_validation_method_detail"] = quote_detail
        row["ai_relevance"] = classify_ai_relevance(row)
        impact_type_clean, impact_type_reason = clean_impact_type_with_reason(row)
        row["impact_type_clean"] = impact_type_clean
        row["impact_type_clean_reason"] = impact_type_reason
        impact_direction_clean, impact_direction_reason = clean_impact_direction_with_reason(row)
        row["impact_direction_clean"] = impact_direction_clean
        row["impact_direction_clean_reason"] = impact_direction_reason
        row["generator_use_scope"] = assign_generator_scope(row)

        if quote_status == "failed":
            row["generator_use_scope"] = "exclude"
            failed_rows.append({field: row.get(field, "") for field in FAILED_FIELDS})
        else:
            validated_rows.append(row)

    return validated_rows, failed_rows


def normalize_for_dedup(text, remove_stop_words=True):
    text = normalize_basic(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = re.sub(r"\s+", " ", text).strip().split()
    if remove_stop_words:
        words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)


def quote_for_dedup(text):
    return normalize_for_dedup(text, remove_stop_words=False)


def are_duplicate_claims(row_a, row_b):
    claim_a = normalize_for_dedup(row_a.get("claim_text"))
    claim_b = normalize_for_dedup(row_b.get("claim_text"))
    if claim_a and claim_a == claim_b:
        return True

    quote_a = quote_for_dedup(row_a.get("evidence_quote"))
    quote_b = quote_for_dedup(row_b.get("evidence_quote"))
    if quote_a and quote_a == quote_b:
        return True

    if not claim_a or not claim_b:
        return False
    return SequenceMatcher(None, claim_a, claim_b).ratio() >= 0.90


def row_rank(row):
    quote_rank = {
        "exact_match": 4,
        "normalized_match": 3,
        "relaxed_punctuation_match": 2,
        "fuzzy_match": 1,
    }
    value_rank = {"high": 3, "medium": 2, "low": 1}
    entity_rank = {
        "occupation": 7,
        "task": 6,
        "skill": 5,
        "industry": 4,
        "worker_group": 3,
        "general_labor_market": 2,
        "education": 2,
        "other": 1,
    }
    return (
        quote_rank.get(row.get("quote_validation"), 0),
        value_rank.get(row.get("confidence"), 0),
        value_rank.get(row.get("evidence_strength"), 0),
        entity_rank.get(row.get("affected_entity_type"), 0),
        len(clean_string(row.get("evidence_quote"))),
    )


def deduplicate_claims(validated_rows):
    groups = []
    source_to_group_indexes = {}

    for row in validated_rows:
        source_id = clean_string(row.get("source_id"))
        candidate_indexes = source_to_group_indexes.setdefault(source_id, [])
        duplicate_group_index = None

        for group_index in candidate_indexes:
            group = groups[group_index]
            if any(are_duplicate_claims(row, group_row) for group_row in group["rows"]):
                duplicate_group_index = group_index
                break

        if duplicate_group_index is None:
            group_id = f"duplicate_group_{len(groups) + 1:06d}"
            groups.append({"duplicate_group_id": group_id, "rows": [row]})
            candidate_indexes.append(len(groups) - 1)
        else:
            groups[duplicate_group_index]["rows"].append(row)

    kept_rows = []
    duplicate_counts_by_source = {}

    for group in groups:
        best_row = group["rows"][0]
        for row in group["rows"][1:]:
            if row_rank(row) > row_rank(best_row):
                best_row = row

        source_id = clean_string(best_row.get("source_id"))
        duplicate_counts_by_source[source_id] = duplicate_counts_by_source.get(source_id, 0) + (
            len(group["rows"]) - 1
        )

        best_row["dedup_status"] = "kept"
        best_row["duplicate_group_id"] = group["duplicate_group_id"]
        best_row["review_status"] = "pending"
        kept_rows.append(best_row)

    for index, row in enumerate(kept_rows, start=1):
        row["claim_id"] = f"claim_{index:06d}"

    final_rows = [{field: row.get(field, "") for field in FINAL_FIELDS} for row in kept_rows]
    return final_rows, duplicate_counts_by_source


def count_values(rows, field):
    counts = {}
    for row in rows:
        value = clean_string(row.get(field)) or "(empty)"
        counts[value] = counts.get(value, 0) + 1
    return counts


def print_counts(title, counts):
    print(title)
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


def build_source_summary(raw_chunk_rows, flat_rows, validated_rows, failed_rows, final_rows, duplicate_counts_by_source):
    source_info = {}
    for raw_row in raw_chunk_rows:
        source_id = clean_string(raw_row.get("source_id"))
        if source_id and source_id not in source_info:
            source_info[source_id] = clean_string(raw_row.get("title"))

    for row in flat_rows:
        source_id = clean_string(row.get("source_id"))
        if source_id and source_id not in source_info:
            source_info[source_id] = clean_string(row.get("title"))

    raw_counts = count_by_source(flat_rows)
    validated_counts = count_by_source(validated_rows)
    failed_counts = count_by_source(failed_rows)
    kept_counts = count_by_source(final_rows)

    relevance_by_source = count_field_by_source(final_rows, "ai_relevance")
    scope_by_source = count_field_by_source(final_rows, "generator_use_scope")

    summary_rows = []
    for source_id in sorted(source_info):
        relevance_counts = relevance_by_source.get(source_id, {})
        scope_counts = scope_by_source.get(source_id, {})
        summary_rows.append(
            {
                "source_id": source_id,
                "title": source_info[source_id],
                "raw_claims": raw_counts.get(source_id, 0),
                "validated_claims_before_dedup": validated_counts.get(source_id, 0),
                "failed_quote_claims": failed_counts.get(source_id, 0),
                "kept_after_dedup": kept_counts.get(source_id, 0),
                "duplicates_removed": duplicate_counts_by_source.get(source_id, 0),
                "direct_ai_claims": relevance_counts.get("direct_ai", 0),
                "automation_or_technology_claims": relevance_counts.get(
                    "automation_or_technology", 0
                ),
                "indirect_labor_context_claims": relevance_counts.get(
                    "indirect_labor_context", 0
                ),
                "non_ai_context_claims": relevance_counts.get("non_ai_context", 0),
                "job_specific_claims": scope_counts.get("job_specific", 0),
                "task_or_skill_linkable_claims": scope_counts.get(
                    "task_or_skill_linkable", 0
                ),
                "background_context_claims": scope_counts.get("background_context", 0),
                "excluded_claims": scope_counts.get("exclude", 0),
            }
        )

    return summary_rows


def count_by_source(rows):
    counts = {}
    for row in rows:
        source_id = clean_string(row.get("source_id"))
        counts[source_id] = counts.get(source_id, 0) + 1
    return counts


def count_field_by_source(rows, field):
    counts = {}
    for row in rows:
        source_id = clean_string(row.get("source_id"))
        value = clean_string(row.get(field)) or "(empty)"
        if source_id not in counts:
            counts[source_id] = {}
        counts[source_id][value] = counts[source_id].get(value, 0) + 1
    return counts


def recovered_count(validated_rows):
    count = 0
    for row in validated_rows:
        original = clean_string(row.get("original_quote_validation"))
        if original == "failed" and row.get("quote_validation") in VALID_QUOTE_STATUSES:
            count += 1
    return count


def warning_row(row, warning_type):
    values = {field: row.get(field, "") for field in WARNING_FIELDS}
    values["warning_type"] = warning_type
    return values


def build_label_cleaning_warnings(rows):
    warnings = []
    for row in rows:
        claim_text = normalized_lower(row.get("claim_text", ""))
        original_type = clean_string(row.get("impact_type")).lower()
        clean_type = clean_string(row.get("impact_type_clean")).lower()
        original_direction = clean_string(row.get("impact_direction")).lower()
        clean_direction = clean_string(row.get("impact_direction_clean")).lower()

        if original_direction == "negative" and clean_direction == "positive":
            warnings.append(warning_row(row, "negative_direction_cleaned_to_positive"))
        if original_type == "worker_vulnerability" and clean_type == "productivity_effect":
            warnings.append(warning_row(row, "worker_vulnerability_cleaned_to_productivity"))
        if original_type == "job_loss" and clean_type == "job_creation":
            warnings.append(warning_row(row, "job_loss_cleaned_to_job_creation"))
        if original_type == "job_creation" and clean_type == "job_loss":
            warnings.append(warning_row(row, "job_creation_cleaned_to_job_loss"))
        if matched_terms(claim_text, ["decline", "declining", "loss", "losses"]) and clean_direction == "positive":
            warnings.append(warning_row(row, "decline_or_loss_claim_cleaned_positive"))
        if matched_terms(claim_text, ["slow", "slower", "limit", "limited"]) and clean_direction == "positive":
            warnings.append(warning_row(row, "slow_or_limit_claim_cleaned_positive"))
        if matched_terms(claim_text, ["unable", "vulnerability", "vulnerable"]) and clean_type == "productivity_effect":
            warnings.append(warning_row(row, "vulnerability_claim_cleaned_to_productivity"))
        if has_occupational_transition(claim_text) and clean_direction == "positive":
            warnings.append(warning_row(row, "occupational_transition_cleaned_positive"))

    return warnings


def write_label_cleaning_warnings(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=WARNING_FIELDS).to_csv(path, index=False)


def run_label_cleaning_regression_tests():
    cases = [
        {
            "claim": "Slow adoption and slow redeployment would limit productivity growth.",
            "original_type": "productivity_effect",
            "original_direction": "negative",
            "expected_type": "productivity_effect",
            "expected_direction": "negative",
        },
        {
            "claim": "Slow worker redeployment would leave millions unable to participate productively.",
            "original_type": "worker_vulnerability",
            "original_direction": "negative",
            "expected_type": "worker_vulnerability",
            "expected_direction": "negative",
        },
        {
            "claim": "By 2030, up to 30 percent of current hours worked could be automated.",
            "original_type": "automation_risk",
            "original_direction": "negative",
            "expected_type": "automation_exposure",
            "expected_direction": "uncertain",
        },
        {
            "claim": "Demand for occupations such as office workers, production workers, and customer service representatives would decline.",
            "original_type": "job_loss",
            "original_direction": "negative",
            "expected_type": "job_loss",
            "expected_direction": "negative",
        },
        {
            "claim": "A faster technology adoption scenario could require some 12 million occupational transitions.",
            "original_type": "employment_effect",
            "original_direction": "negative",
            "expected_type": "employment_effect",
            "expected_direction": "uncertain",
        },
        {
            "claim": "The need for workers to transition from declining occupations to rising ones will be underscored.",
            "original_type": "task_displacement",
            "original_direction": "negative",
            "expected_type": "task_transformation",
            "expected_direction": "mixed",
        },
    ]

    failures = []
    for index, case in enumerate(cases, start=1):
        row = {
            "claim_text": case["claim"],
            "evidence_quote": case["claim"],
            "impact_type": case["original_type"],
            "impact_direction": case["original_direction"],
            "affected_entity_type": "general_labor_market",
            "affected_entity_text": "",
            "occupation_or_skill_mentions": [],
        }
        row["ai_relevance"] = classify_ai_relevance(row)
        type_clean, type_reason = clean_impact_type_with_reason(row)
        row["impact_type_clean"] = type_clean
        row["impact_type_clean_reason"] = type_reason
        direction_clean, direction_reason = clean_impact_direction_with_reason(row)
        row["impact_direction_clean"] = direction_clean
        row["impact_direction_clean_reason"] = direction_reason

        if type_clean != case["expected_type"]:
            failures.append(
                f"case {index}: expected type {case['expected_type']}, got {type_clean}"
            )
        if direction_clean != case["expected_direction"]:
            failures.append(
                f"case {index}: expected direction {case['expected_direction']}, got {direction_clean}"
            )

    if failures:
        print("Label-cleaning regression tests failed:")
        for failure in failures:
            print(f"- {failure}")
        return False

    print(f"Label-cleaning regression tests passed ({len(cases)} cases).")
    return True


def page_range(row):
    page_start = row.get("page_start")
    page_end = row.get("page_end")
    if page_start is None and page_end is None:
        return "web"
    if page_start == "" and page_end == "":
        return "web"
    if page_start == page_end:
        return f"page {page_start}"
    return f"pages {page_start}-{page_end}"


def validate_output(path):
    print("\nValidation")
    if not path.exists():
        print(f"Missing output file: {path}")
        return False

    rows = read_jsonl(path)
    errors = []
    seen_claim_ids = set()

    required_fields = [
        "claim_id",
        "source_id",
        "chunk_id",
        "claim_text",
        "evidence_quote",
        "ai_relevance",
        "impact_type_clean",
        "impact_type_clean_reason",
        "impact_direction_clean",
        "impact_direction_clean_reason",
        "generator_use_scope",
    ]

    for index, row in enumerate(rows, start=1):
        for field in required_fields:
            if not clean_string(row.get(field)):
                errors.append(f"Line {index}: missing or empty '{field}'")

        if row.get("quote_validation") == "failed":
            errors.append(f"Line {index}: quote_validation is failed")
        if row.get("quote_validation") not in VALID_QUOTE_STATUSES:
            errors.append(f"Line {index}: unexpected quote_validation '{row.get('quote_validation')}'")
        if row.get("ai_relevance") not in VALID_RELEVANCE:
            errors.append(f"Line {index}: unexpected ai_relevance '{row.get('ai_relevance')}'")
        if row.get("impact_type_clean") not in VALID_IMPACT_TYPES:
            errors.append(
                f"Line {index}: unexpected impact_type_clean '{row.get('impact_type_clean')}'"
            )
        if row.get("impact_direction_clean") not in VALID_DIRECTIONS:
            errors.append(
                f"Line {index}: unexpected impact_direction_clean '{row.get('impact_direction_clean')}'"
            )
        if row.get("generator_use_scope") not in VALID_SCOPES:
            errors.append(
                f"Line {index}: unexpected generator_use_scope '{row.get('generator_use_scope')}'"
            )

        claim_id = clean_string(row.get("claim_id"))
        if claim_id in seen_claim_ids:
            errors.append(f"Line {index}: duplicate claim_id '{claim_id}'")
        seen_claim_ids.add(claim_id)

    if errors:
        print(f"Validation failed with {len(errors)} issue(s).")
        for error in errors[:15]:
            print(f"- {error}")
        if len(errors) > 15:
            print(f"- ... {len(errors) - 15} more")
        return False

    print(f"Validated {len(rows)} final clean claim row(s).")
    print(f"Showing first {min(10, len(rows))} final claim(s):")

    for row in rows[:10]:
        print(f"\n{row.get('claim_id')} | {row.get('title') or '(untitled)'} | {page_range(row)}")
        print(f"impact_type_clean: {row.get('impact_type_clean')}")
        print(f"ai_relevance: {row.get('ai_relevance')}")
        print(f"generator_use_scope: {row.get('generator_use_scope')}")
        print(f"Claim: {row.get('claim_text')}")
        print(f"Quote: {row.get('evidence_quote')}")

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process AI/job-impact claims without OpenAI API calls."
    )
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--raw-claims", default=str(DEFAULT_RAW_CLAIMS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--failed-output", default=str(DEFAULT_FAILED_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--warnings-output", default=str(DEFAULT_WARNINGS))
    parser.add_argument("--overwrite-clean", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not run_label_cleaning_regression_tests():
        raise SystemExit(1)

    chunks_path = resolve_path(args.chunks)
    raw_claims_path = resolve_path(args.raw_claims)
    output_path = resolve_path(args.output)
    failed_output_path = resolve_path(args.failed_output)
    summary_path = resolve_path(args.summary)
    warnings_path = resolve_path(args.warnings_output)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    if not raw_claims_path.exists():
        raise FileNotFoundError(f"Raw claims file not found: {raw_claims_path}")

    chunk_by_id = load_chunks(chunks_path)
    raw_rows = read_jsonl(raw_claims_path)
    flat_rows = flatten_raw_claims(raw_rows)

    validated_rows, failed_rows = postprocess_flat_claims(flat_rows, chunk_by_id)
    final_rows, duplicate_counts_by_source = deduplicate_claims(validated_rows)
    warning_rows = build_label_cleaning_warnings(final_rows)
    summary_rows = build_source_summary(
        raw_rows,
        flat_rows,
        validated_rows,
        failed_rows,
        final_rows,
        duplicate_counts_by_source,
    )

    write_jsonl(final_rows, output_path)
    write_jsonl(failed_rows, failed_output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows, columns=SUMMARY_FIELDS).to_csv(summary_path, index=False)
    write_label_cleaning_warnings(warning_rows, warnings_path)

    if args.overwrite_clean:
        write_jsonl(final_rows, DEFAULT_CLEAN_OUTPUT)

    print("\nAI impact claim post-processing complete")
    print(f"Total raw claims: {len(flat_rows)}")
    print(f"Validated before dedup: {len(validated_rows)}")
    print(f"Failed quote validation: {len(failed_rows)}")
    print(f"Recovered by improved validation: {recovered_count(validated_rows)}")
    print(f"Duplicates removed: {len(validated_rows) - len(final_rows)}")
    print(f"Final clean claims: {len(final_rows)}")
    print_counts("Counts by quote_validation:", count_values(validated_rows + failed_rows, "quote_validation"))
    print_counts("Counts by ai_relevance:", count_values(final_rows, "ai_relevance"))
    print_counts("Counts by impact_type_clean:", count_values(final_rows, "impact_type_clean"))
    print_counts("Counts by generator_use_scope:", count_values(final_rows, "generator_use_scope"))
    print(f"Label cleaning warnings: {len(warning_rows)}")
    if warning_rows:
        print_counts("Warnings by type:", count_values(warning_rows, "warning_type"))
    print(f"Postprocessed output path: {output_path}")
    print(f"Failed quotes output path: {failed_output_path}")
    print(f"Summary path: {summary_path}")
    print(f"Label cleaning warnings path: {warnings_path}")
    if args.overwrite_clean:
        print(f"Overwritten clean output path: {DEFAULT_CLEAN_OUTPUT}")

    validate_output(output_path)


if __name__ == "__main__":
    main()
