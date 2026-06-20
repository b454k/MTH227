#!/usr/bin/env python3
"""Research-claim retriever for AI-impact evidence.

This module reads the existing ChromaDB collection built from
``data/research/ai_impact_claims.jsonl``. It does not create, delete, or
re-embed research documents.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from career_rag.config import (
        BGE_QUERY_PREFIX,
        EMBEDDING_MODEL_NAME,
        quiet_huggingface_model_load,
        require_hf_token,
        validate_collection_embedding_model,
    )
except ImportError:  # Allows: py career_rag/research_retriever.py
    from config import (  # type: ignore
        BGE_QUERY_PREFIX,
        EMBEDDING_MODEL_NAME,
        quiet_huggingface_model_load,
        require_hf_token,
        validate_collection_embedding_model,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_research"
DEFAULT_COLLECTION = "research_ai_impact_claims"
DEFAULT_MODEL = EMBEDDING_MODEL_NAME

RETURN_FIELDS = (
    "claim_id",
    "claim_text",
    "evidence_quote",
    "title",
    "authors",
    "year",
    "page_start",
    "page_end",
    "url",
    "final_url",
    "impact_type_clean",
    "impact_direction_clean",
    "affected_entity_type",
    "affected_entity_text",
    "occupation_or_skill_mentions",
    "ai_relevance",
    "generator_use_scope",
    "quote_validation",
)

PREFERRED_AI_RELEVANCE = {"direct_ai", "automation_or_technology"}


def retrieve_research_claims(
    query: str,
    top_k: int = 8,
    ai_relevance: list[str] | None = None,
    generator_use_scope: list[str] | None = None,
    impact_types: list[str] | None = None,
    include_excluded: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve AI-impact research claims from the existing Chroma collection."""
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    collection = _load_collection()
    model = _load_model()
    collection_count = collection.count()
    if collection_count <= 0:
        return []

    query_text = f"{BGE_QUERY_PREFIX}{query}"
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
    )[0]
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()

    candidate_count = min(collection_count, max(top_k * 6, top_k + 20))
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "distances", "metadatas"],
    )

    claims: list[dict[str, Any]] = []
    ids = _first_result_list(results, "ids")
    distances = _first_result_list(results, "distances")
    metadatas = _first_result_list(results, "metadatas")

    for doc_id, distance, metadata in zip(ids, distances, metadatas):
        metadata = metadata or {}
        claim = _format_claim(doc_id, float(distance), metadata)
        if not _claim_matches_filters(
            claim,
            ai_relevance=ai_relevance,
            generator_use_scope=generator_use_scope,
            impact_types=impact_types,
            include_excluded=include_excluded,
        ):
            continue
        claims.append(claim)

    claims.sort(key=_claim_rank_key, reverse=True)
    return claims[:top_k]


def format_research_source(claim: dict[str, Any]) -> str:
    """Format a compact title/year/page citation for a research claim."""
    title = _clean_text(claim.get("title"))
    year = _clean_text(claim.get("year"))
    source_id = _clean_text(
        claim.get("source_id")
        or claim.get("claim_id")
        or claim.get("id")
        or claim.get("url")
    )
    label = title or source_id or "research source"
    page_text = format_pages(claim)

    parts = [label]
    if year:
        parts.append(year)
    if page_text:
        parts.append(page_text)
    return ", ".join(parts)


def format_pages(claim: dict[str, Any]) -> str:
    """Format page metadata as p. N, pp. N-M, or an empty string."""
    page_start = _clean_page(claim.get("page_start"))
    page_end = _clean_page(claim.get("page_end"))

    if not page_start and not page_end:
        return ""
    if not page_start:
        return f"p. {page_end}"
    if not page_end or page_start == page_end:
        return f"p. {page_start}"
    return f"pp. {page_start}-{page_end}"


def _format_claim(
    doc_id: str,
    distance: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Convert Chroma metadata into a stable research-claim result."""
    claim = {field: metadata.get(field, "") for field in RETURN_FIELDS}
    claim_id = _clean_text(claim.get("claim_id")) or str(doc_id)
    claim["claim_id"] = claim_id
    claim["id"] = str(doc_id)
    claim["source_id"] = metadata.get("source_id", "")
    claim["collection"] = DEFAULT_COLLECTION
    claim["distance"] = distance
    claim["score"] = 1.0 / (1.0 + distance)
    return claim


def _claim_matches_filters(
    claim: dict[str, Any],
    ai_relevance: list[str] | None,
    generator_use_scope: list[str] | None,
    impact_types: list[str] | None,
    include_excluded: bool,
) -> bool:
    """Apply default safety filters and optional strict caller filters."""
    quote_validation = _clean_text(claim.get("quote_validation")).lower()
    if quote_validation == "failed":
        return False

    scope = _clean_text(claim.get("generator_use_scope"))
    if not include_excluded and scope == "exclude":
        return False

    if ai_relevance is not None:
        allowed = {_clean_text(value) for value in ai_relevance if _clean_text(value)}
        if allowed and _clean_text(claim.get("ai_relevance")) not in allowed:
            return False

    if generator_use_scope is not None:
        allowed = {
            _clean_text(value)
            for value in generator_use_scope
            if _clean_text(value)
        }
        if allowed and scope not in allowed:
            return False

    if impact_types is not None:
        allowed = {_clean_text(value) for value in impact_types if _clean_text(value)}
        if allowed and _clean_text(claim.get("impact_type_clean")) not in allowed:
            return False

    return True


def _claim_rank_key(claim: dict[str, Any]) -> tuple[float, float]:
    """Rank by Chroma score with a modest preference for direct AI evidence."""
    score = float(claim.get("score") or 0.0)
    relevance = _clean_text(claim.get("ai_relevance"))
    preference = 0.03 if relevance in PREFERRED_AI_RELEVANCE else 0.0
    return (score + preference, score)


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the sentence-transformer model used for research embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Could not import sentence_transformers. Install project dependencies "
            "before using the research retriever."
        ) from exc

    require_hf_token()
    with quiet_huggingface_model_load():
        return SentenceTransformer(DEFAULT_MODEL)


@lru_cache(maxsize=1)
def _load_collection() -> Any:
    """Load the existing research Chroma collection."""
    if not DEFAULT_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Research Chroma persist directory not found: {DEFAULT_PERSIST_DIR}"
        )

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Could not import chromadb. Install project dependencies before using "
            "the research retriever."
        ) from exc

    client = chromadb.PersistentClient(path=str(DEFAULT_PERSIST_DIR))
    collection = client.get_collection(name=DEFAULT_COLLECTION)
    validate_collection_embedding_model(collection, DEFAULT_MODEL)
    return collection


def _first_result_list(results: dict[str, Any], key: str) -> list[Any]:
    """Return Chroma's first query result list for one result key."""
    values = results.get(key) or [[]]
    if not values:
        return []
    return values[0] or []


def _clean_text(value: Any) -> str:
    """Return a stripped string for metadata display and comparison."""
    if value is None:
        return ""
    return str(value).strip()


def _clean_page(value: Any) -> str:
    """Format numeric page values without unnecessary .0 suffixes."""
    text = _clean_text(value)
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text
