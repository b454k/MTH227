#!/usr/bin/env python3
"""Prioritized retriever for structured AI-impact evidence."""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Any

try:
    from career_rag.config import (
        EMBEDDING_MODEL_NAME,
        quiet_huggingface_model_load,
        require_hf_token,
        validate_collection_embedding_model,
    )
    from career_rag.ai_exposure_utils import (
        PROJECT_ROOT,
        normalize_soc_code,
        normalize_text,
        one_line,
    )
except ImportError:  # Allows: py career_rag/ai_impact_retriever.py
    from config import (  # type: ignore
        EMBEDDING_MODEL_NAME,
        quiet_huggingface_model_load,
        require_hf_token,
        validate_collection_embedding_model,
    )
    from ai_exposure_utils import (  # type: ignore
        PROJECT_ROOT,
        normalize_soc_code,
        normalize_text,
        one_line,
    )


DEFAULT_AI_PERSIST_DIR = PROJECT_ROOT / "chroma_ai_impact"
DEFAULT_AI_COLLECTION = "ai_impact_evidence"
DEFAULT_RESEARCH_PERSIST_DIR = PROJECT_ROOT / "chroma_research"
DEFAULT_RESEARCH_COLLECTION = "research_inference"
DEFAULT_MODEL = EMBEDDING_MODEL_NAME

TASK_DOC_TYPES = {"ai_task_impact", "ai_task_penetration"}
METHODOLOGY_DOC_TYPES = {"ai_methodology", "ai_core_supplemental_exposure"}
CONFIDENCE_SCORE = {"high": 0.5, "medium": 0.25, "low": 0.0}


def format_distance_to_score(distance: float) -> float:
    """Convert Chroma distance to a simple similarity score."""
    return 1.0 / (1.0 + distance)


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    require_hf_token()
    with quiet_huggingface_model_load():
        return SentenceTransformer(DEFAULT_MODEL)


@lru_cache(maxsize=4)
def load_collection(persist_dir_text: str, collection_name: str) -> Any:
    """Load a Chroma collection."""
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir_text)
    collection = client.get_collection(collection_name)
    validate_collection_embedding_model(collection, DEFAULT_MODEL)
    return collection


def embed_query(query: str) -> list[float]:
    """Embed one query."""
    model = load_model()
    embedding = model.encode([query], normalize_embeddings=True)[0]
    return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


def first_list(results: dict[str, Any], key: str) -> list[Any]:
    """Return Chroma's first result list."""
    values = results.get(key) or [[]]
    return values[0] if values else []


def query_collection(
    query: str,
    persist_dir: Any,
    collection_name: str,
    candidate_count: int,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run a raw Chroma query."""
    collection = load_collection(str(persist_dir), collection_name)
    count = collection.count()
    if count <= 0:
        return []
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [embed_query(query)],
        "n_results": min(candidate_count, count),
        "include": ["documents", "distances", "metadatas"],
    }
    if where:
        query_kwargs["where"] = where
    results = collection.query(**query_kwargs)
    rows: list[dict[str, Any]] = []
    ids = first_list(results, "ids")
    documents = first_list(results, "documents")
    distances = first_list(results, "distances")
    metadatas = first_list(results, "metadatas")
    for doc_id, document, distance, metadata in zip(ids, documents, distances, metadatas):
        distance_float = float(distance)
        rows.append(
            {
                "id": doc_id,
                "doc_id": doc_id,
                "text": document or "",
                "metadata": metadata or {},
                "distance": distance_float,
                "score": format_distance_to_score(distance_float),
                "collection": collection_name,
            }
        )
    return rows


def merge_candidate_sets(candidate_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge candidate lists without duplicate IDs."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidates in candidate_sets:
        for result in candidates:
            doc_id = one_line(result.get("doc_id") or result.get("id"))
            if doc_id in seen:
                continue
            seen.add(doc_id)
            merged.append(result)
    return merged


def exact_title_match(expected: str | None, actual: Any) -> bool:
    """Return True for exact occupation title match."""
    if not expected:
        return False
    return normalize_text(expected) == normalize_text(actual)


def task_similarity(query_task: str, result_task: str) -> float:
    """Compute a small lexical task similarity."""
    query_norm = normalize_text(query_task)
    result_norm = normalize_text(result_task)
    if not query_norm or not result_norm:
        return 0.0
    if query_norm == result_norm:
        return 1.0
    query_tokens = set(query_norm.split())
    result_tokens = set(result_norm.split())
    return len(query_tokens & result_tokens) / max(len(query_tokens | result_tokens), 1)


def best_task_similarity(task_texts: list[str] | None, result_task: Any) -> float:
    """Return best task similarity against provided O*NET task texts."""
    if not task_texts:
        return 0.0
    result_task_text = one_line(result_task)
    return max((task_similarity(task, result_task_text) for task in task_texts), default=0.0)


def rerank_result(
    result: dict[str, Any],
    query: str,
    soc_code: str | None,
    occupation_title: str | None,
    task_texts: list[str] | None,
) -> tuple[float, list[str]]:
    """Apply requested ranking priorities."""
    metadata = result.get("metadata") or {}
    reasons: list[str] = []
    score = float(result.get("score") or 0.0)

    expected_soc = normalize_soc_code(soc_code) if soc_code else ""
    actual_soc = normalize_soc_code(metadata.get("soc_code")) or one_line(metadata.get("soc_code"))
    if expected_soc and actual_soc == expected_soc:
        score += 3.0
        reasons.append("exact_soc")

    if exact_title_match(occupation_title, metadata.get("occupation_title")):
        score += 2.0
        reasons.append("exact_occupation_title")

    task_match = best_task_similarity(task_texts, metadata.get("task_text"))
    if task_match >= 1.0:
        score += 2.5
        reasons.append("exact_task_text")
    elif task_match >= 0.72:
        score += 1.5
        reasons.append(f"near_task_text:{task_match:.2f}")

    doc_type = one_line(metadata.get("doc_type"))
    if doc_type in TASK_DOC_TYPES or one_line(metadata.get("task_text")):
        score += 1.2
        reasons.append("task_level")
    elif doc_type in METHODOLOGY_DOC_TYPES:
        score += 0.3
        reasons.append("methodology")

    source_id = one_line(metadata.get("source_id"))
    if source_id == "anthropic_economic_index" and doc_type in TASK_DOC_TYPES:
        score += 1.0
        reasons.append("anthropic_task_metric")
    elif source_id == "nber_w31222" and doc_type in METHODOLOGY_DOC_TYPES:
        score += 0.6
        reasons.append("nber_methodology")

    if one_line(metadata.get("metric_value")):
        score += 0.8
        reasons.append("numeric_metric")

    impact_type = one_line(metadata.get("impact_type")).lower()
    metric_name = one_line(metadata.get("metric_name")).lower()
    query_lower = query.lower()
    if "automation" in query_lower and impact_type == "automation":
        score += 0.8
        reasons.append("query_prefers_automation")
    elif "augmentation" in query_lower and impact_type == "augmentation":
        score += 0.8
        reasons.append("query_prefers_augmentation")
    elif "penetration" in query_lower and impact_type == "task_penetration":
        score += 0.8
        reasons.append("query_prefers_penetration")
    elif "exposure" in query_lower and impact_type in {"task_penetration", "exposure", "job_exposure"}:
        score += 0.5
        reasons.append("query_prefers_exposure")
    elif impact_type == "task_penetration":
        score += 0.35
        reasons.append("default_prefers_task_penetration")
    elif impact_type in {"exposure", "job_exposure"} or "exposure" in metric_name:
        score += 0.2
        reasons.append("default_prefers_exposure")

    confidence = one_line(metadata.get("confidence")).lower()
    score += CONFIDENCE_SCORE.get(confidence, 0.0)
    if confidence:
        reasons.append(f"confidence:{confidence}")

    return score, reasons


def claim_fingerprint(result: dict[str, Any]) -> tuple[str, ...]:
    """Deduplicate normalized metric/evidence claims."""
    metadata = result.get("metadata") or {}
    return (
        one_line(metadata.get("source_id")).lower(),
        one_line(metadata.get("source_file") or metadata.get("source_release")).lower(),
        one_line(metadata.get("soc_code")).lower(),
        normalize_text(metadata.get("occupation_title")),
        normalize_text(metadata.get("task_text")),
        one_line(metadata.get("impact_type")).lower(),
        one_line(metadata.get("metric_name")).lower(),
        one_line(metadata.get("metric_value")).lower(),
        normalize_text(result.get("text")),
    )


def source_metric_key(result: dict[str, Any]) -> tuple[str, ...]:
    """Avoid returning the same metric/evidence multiple times from one source."""
    metadata = result.get("metadata") or {}
    return (
        one_line(metadata.get("source_id")).lower(),
        one_line(metadata.get("metric_name")).lower(),
        one_line(metadata.get("metric_value")).lower(),
        one_line(metadata.get("impact_type")).lower(),
        normalize_text(metadata.get("task_text")),
        normalize_text(metadata.get("occupation_title")),
    )


def task_diversity_key(result: dict[str, Any]) -> tuple[str, ...] | None:
    """Return a key for preventing repeated same-task crowding."""
    metadata = result.get("metadata") or {}
    task = normalize_text(metadata.get("task_text"))
    if not task:
        return None
    return (
        one_line(metadata.get("source_id")).lower(),
        one_line(metadata.get("soc_code")).lower(),
        normalize_text(metadata.get("occupation_title")),
        task,
    )


def result_text_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Lexical near-duplicate similarity."""
    left_tokens = set(normalize_text(left.get("text")).split())
    right_tokens = set(normalize_text(right.get("text")).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def dedupe_ranked(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Deduplicate ranked results."""
    kept: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_claims: set[tuple[str, ...]] = set()
    seen_source_metrics: set[tuple[str, ...]] = set()
    seen_tasks: set[tuple[str, ...]] = set()

    for result in results:
        doc_id = one_line(result.get("doc_id") or result.get("id"))
        if doc_id in seen_ids:
            continue
        fingerprint = claim_fingerprint(result)
        if fingerprint in seen_claims:
            continue
        source_metric = source_metric_key(result)
        if source_metric in seen_source_metrics:
            continue
        task_key = task_diversity_key(result)
        if task_key and task_key in seen_tasks:
            continue
        if any(result_text_similarity(result, kept_result) >= 0.96 for kept_result in kept):
            continue
        seen_ids.add(doc_id)
        seen_claims.add(fingerprint)
        seen_source_metrics.add(source_metric)
        if task_key:
            seen_tasks.add(task_key)
        kept.append(result)
        if len(kept) >= top_k:
            break
    return kept


def retrieve_ai_impact(
    query: str,
    soc_code: str | None = None,
    occupation_title: str | None = None,
    task_texts: list[str] | None = None,
    top_k: int = 8,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve and rerank structured AI-impact evidence."""
    query = one_line(query)
    if not query:
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    candidate_count = max(top_k * 8, 40)
    candidate_sets = [
        query_collection(query, DEFAULT_AI_PERSIST_DIR, DEFAULT_AI_COLLECTION, candidate_count)
    ]
    expected_soc = normalize_soc_code(soc_code) if soc_code else ""
    if expected_soc:
        try:
            candidate_sets.append(
                query_collection(
                    query,
                    DEFAULT_AI_PERSIST_DIR,
                    DEFAULT_AI_COLLECTION,
                    max(top_k * 4, 20),
                    where={"soc_code": expected_soc},
                )
            )
        except Exception:
            pass
    if occupation_title:
        try:
            candidate_sets.append(
                query_collection(
                    query,
                    DEFAULT_AI_PERSIST_DIR,
                    DEFAULT_AI_COLLECTION,
                    max(top_k * 3, 12),
                    where={"occupation_title": occupation_title},
                )
            )
        except Exception:
            pass
    candidates = merge_candidate_sets(candidate_sets)
    ranked: list[dict[str, Any]] = []
    for result in candidates:
        reranked_score, reasons = rerank_result(
            result,
            query,
            soc_code,
            occupation_title,
            task_texts,
        )
        result["reranked_score"] = reranked_score
        if debug:
            result["ranking_reason"] = reasons
        ranked.append(result)
    if expected_soc:
        exact_soc_ranked = [
            result
            for result in ranked
            if (normalize_soc_code((result.get("metadata") or {}).get("soc_code")) or one_line((result.get("metadata") or {}).get("soc_code"))) == expected_soc
        ]
        if len(exact_soc_ranked) >= top_k:
            ranked = exact_soc_ranked
    ranked.sort(key=lambda row: float(row.get("reranked_score") or 0.0), reverse=True)
    return dedupe_ranked(ranked, top_k)


def retrieve_research_inference(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Retrieve methodology/caveat chunks from the research_inference collection."""
    query = one_line(query)
    if not query:
        raise ValueError("query must not be empty.")
    candidates = query_collection(query, DEFAULT_RESEARCH_PERSIST_DIR, DEFAULT_RESEARCH_COLLECTION, max(top_k * 4, 20))
    candidates.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in candidates:
        fingerprint = normalize_text(result.get("text"))[:200]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(result)
        if len(kept) >= top_k:
            break
    return kept


def print_result(rank: int, result: dict[str, Any], debug: bool) -> None:
    """Print one CLI retrieval result."""
    metadata = result.get("metadata") or {}
    print(f"{rank}. score={float(result.get('reranked_score') or result.get('score') or 0.0):.4f}")
    print(f"   source={metadata.get('source_name') or metadata.get('source_id')}")
    print(f"   doc_type={metadata.get('doc_type')} impact_type={metadata.get('impact_type')}")
    print(f"   occupation={metadata.get('occupation_title')} soc={metadata.get('soc_code')}")
    print(f"   task={metadata.get('task_text')}")
    print(f"   metric={metadata.get('metric_name')} = {metadata.get('metric_value')} {metadata.get('metric_unit')}")
    print(f"   confidence={metadata.get('confidence')}")
    if debug:
        print(f"   reason={', '.join(result.get('ranking_reason') or [])}")
    preview = one_line(result.get("text"))[:300]
    print(f"   text={preview}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Retrieve prioritized AI impact evidence.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--soc-code")
    parser.add_argument("--occupation-title")
    parser.add_argument("--task-text", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run CLI retrieval."""
    args = parse_args()
    query = " ".join(args.query)
    results = retrieve_ai_impact(
        query,
        soc_code=args.soc_code,
        occupation_title=args.occupation_title,
        task_texts=args.task_text,
        top_k=args.top_k,
        debug=args.debug,
    )
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    for rank, result in enumerate(results, start=1):
        print_result(rank, result, args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
