#!/usr/bin/env python3
"""Merge and deduplicate structured AI-impact evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

try:
    from career_rag.config import (
        EMBEDDING_MODEL_NAME,
        quiet_huggingface_model_load,
        require_hf_token,
    )
    from career_rag.ai_exposure_utils import (
        PROJECT_ROOT,
        normalize_text,
        one_line,
        read_jsonl,
        resolve_project_path,
        write_jsonl,
    )
except ImportError:  # Allows: py career_rag/merge_ai_impact_evidence.py
    from config import (  # type: ignore
        EMBEDDING_MODEL_NAME,
        quiet_huggingface_model_load,
        require_hf_token,
    )
    from ai_exposure_utils import (  # type: ignore
        PROJECT_ROOT,
        normalize_text,
        one_line,
        read_jsonl,
        resolve_project_path,
        write_jsonl,
    )


DEFAULT_ANTHROPIC = PROJECT_ROOT / "data" / "processed" / "anthropic_ai_impact.jsonl"
DEFAULT_NBER = PROJECT_ROOT / "data" / "processed" / "nber_w31222_ai_exposure.jsonl"
DEFAULT_MERGED = PROJECT_ROOT / "data" / "processed" / "ai_impact_evidence.jsonl"
DEFAULT_DEDUPED = PROJECT_ROOT / "data" / "processed" / "ai_impact_evidence_deduped.jsonl"

SOURCE_PRIORITY = {"anthropic_economic_index": 3, "nber_w31222": 2}
CONFIDENCE_PRIORITY = {"high": 3, "medium": 2, "low": 1}
TASK_DOC_TYPES = {"ai_task_impact", "ai_task_penetration"}
OCC_DOC_TYPES = {"ai_occupation_impact", "ai_job_exposure"}


def exact_fingerprint(row: dict[str, Any]) -> tuple[str, ...]:
    """Return the requested dedup fingerprint."""
    return (
        one_line(row.get("source_id")).lower(),
        one_line(row.get("source_file") or row.get("source_release")).lower(),
        one_line(row.get("soc_code")).lower(),
        one_line(row.get("occupation_title")).lower(),
        normalize_text(row.get("task_text")),
        one_line(row.get("impact_type")).lower(),
        one_line(row.get("metric_name")).lower(),
        one_line(row.get("metric_value")).lower(),
        normalize_text(row.get("evidence_text")),
    )


def loose_group_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Group rows for near-duplicate checks."""
    return (
        one_line(row.get("source_id")).lower(),
        one_line(row.get("soc_code")).lower(),
        one_line(row.get("occupation_title")).lower(),
        normalize_text(row.get("task_text"))[:100],
        one_line(row.get("impact_type")).lower(),
        one_line(row.get("metric_name")).lower(),
        one_line(row.get("metric_value")).lower(),
    )


def row_priority(row: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, int]:
    """Priority for deciding which duplicate row to keep."""
    doc_type = one_line(row.get("doc_type"))
    source_id = one_line(row.get("source_id"))
    is_task = 1 if doc_type in TASK_DOC_TYPES or row.get("task_text") else 0
    is_occupation = 1 if doc_type in OCC_DOC_TYPES or row.get("occupation_title") else 0
    source_score = SOURCE_PRIORITY.get(source_id, 0)
    has_metric = 1 if row.get("metric_value") is not None else 0
    has_soc = 1 if one_line(row.get("soc_code")) else 0
    has_task = 1 if one_line(row.get("task_text")) else 0
    confidence = CONFIDENCE_PRIORITY.get(one_line(row.get("confidence")).lower(), 0)
    shorter = -len(one_line(row.get("evidence_text")))
    return (is_task, source_score, has_metric, has_soc, has_task, confidence, is_occupation, shorter)


def choose_better(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Keep the higher-priority row."""
    return first if row_priority(first) >= row_priority(second) else second


def exact_dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove exact fingerprint duplicates."""
    best_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    removed = 0
    for row in rows:
        key = exact_fingerprint(row)
        if key in best_by_key:
            best_by_key[key] = choose_better(best_by_key[key], row)
            removed += 1
        else:
            best_by_key[key] = row
    return list(best_by_key.values()), removed


def token_similarity(left: str, right: str) -> float:
    """Small fallback similarity for near duplicates."""
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def try_embedding_near_duplicate(left: str, right: str, model: Any | None) -> bool:
    """Return True when sentence-transformers says two texts are near duplicates."""
    if model is None:
        return False
    try:
        embeddings = model.encode([left, right], normalize_embeddings=True)
        similarity = float(embeddings[0] @ embeddings[1])
        return similarity >= 0.94
    except Exception:
        return False


def load_near_duplicate_model() -> Any | None:
    """Load an optional sentence-transformer model for near-deduping."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    require_hf_token()
    try:
        with quiet_huggingface_model_load():
            return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        return None


def near_dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove near duplicates within small comparable groups."""
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[loose_group_key(row)].append(row)

    model = load_near_duplicate_model()
    deduped: list[dict[str, Any]] = []
    removed = 0

    for group_rows in grouped.values():
        kept: list[dict[str, Any]] = []
        for row in group_rows:
            row_text = one_line(row.get("evidence_text"))
            duplicate_index: int | None = None
            for index, kept_row in enumerate(kept):
                kept_text = one_line(kept_row.get("evidence_text"))
                if normalize_text(row_text) == normalize_text(kept_text):
                    duplicate_index = index
                    break
                if token_similarity(row_text, kept_text) >= 0.94:
                    duplicate_index = index
                    break
                if try_embedding_near_duplicate(row_text, kept_text, model):
                    duplicate_index = index
                    break
            if duplicate_index is None:
                kept.append(row)
                continue
            kept[duplicate_index] = choose_better(kept[duplicate_index], row)
            removed += 1
        deduped.extend(kept)

    return deduped, removed


def allowed_statistics_row(row: dict[str, Any]) -> bool:
    """Only NBER W31222 and Anthropic Economic Index can contribute statistics."""
    return one_line(row.get("source_id")) in {"nber_w31222", "anthropic_economic_index"} and row.get("statistics_allowed") is True


def print_counts(rows: list[dict[str, Any]], label: str) -> None:
    """Print requested count summaries."""
    print(f"\n{label}")
    print(f"Rows: {len(rows):,}")
    print("Rows by source:")
    for key, value in Counter(one_line(row.get("source_id")) or "missing" for row in rows).most_common():
        print(f"  {key}: {value:,}")
    print("Rows by doc_type:")
    for key, value in Counter(one_line(row.get("doc_type")) or "missing" for row in rows).most_common():
        print(f"  {key}: {value:,}")
    print("Rows by impact_type:")
    for key, value in Counter(one_line(row.get("impact_type")) or "missing" for row in rows).most_common():
        print(f"  {key}: {value:,}")
    print(f"Rows with SOC: {sum(1 for row in rows if one_line(row.get('soc_code'))):,}")
    print(f"Rows with task_text: {sum(1 for row in rows if one_line(row.get('task_text'))):,}")
    print(f"Rows with metric_value: {sum(1 for row in rows if row.get('metric_value') is not None):,}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Merge and deduplicate AI-impact evidence.")
    parser.add_argument("--anthropic", default=str(DEFAULT_ANTHROPIC))
    parser.add_argument("--nber", default=str(DEFAULT_NBER))
    parser.add_argument("--output", default=str(DEFAULT_MERGED))
    parser.add_argument("--deduped-output", default=str(DEFAULT_DEDUPED))
    return parser.parse_args()


def main() -> int:
    """Run merge and deduplication."""
    args = parse_args()
    anthropic_path = resolve_project_path(args.anthropic)
    nber_path = resolve_project_path(args.nber)
    output_path = resolve_project_path(args.output)
    deduped_path = resolve_project_path(args.deduped_output)

    anthropic_rows = [row for row in read_jsonl(anthropic_path) if allowed_statistics_row(row)]
    nber_rows = [row for row in read_jsonl(nber_path) if allowed_statistics_row(row)]
    merged = anthropic_rows + nber_rows
    write_jsonl(merged, output_path)

    exact_rows, exact_removed = exact_dedupe(merged)
    deduped, near_removed = near_dedupe(exact_rows)
    deduped.sort(key=row_priority, reverse=True)
    write_jsonl(deduped, deduped_path)

    print("AI impact evidence merge complete")
    print(f"Anthropic input rows kept: {len(anthropic_rows):,}")
    print(f"NBER input rows kept: {len(nber_rows):,}")
    print(f"Merged output rows: {len(merged):,}")
    print(f"Deduped output rows: {len(deduped):,}")
    print(f"Duplicates removed: {exact_removed + near_removed:,}")
    print(f"  Exact duplicates removed: {exact_removed:,}")
    print(f"  Near duplicates removed: {near_removed:,}")
    print(f"Merged output: {output_path}")
    print(f"Deduped output: {deduped_path}")
    print_counts(merged, "Merged counts")
    print_counts(deduped, "Deduped counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
