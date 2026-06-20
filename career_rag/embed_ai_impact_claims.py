#!/usr/bin/env python3
"""Embed clean AI/job-impact claims into a ChromaDB research collection."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from career_rag.config import (
        BGE_QUERY_PREFIX,
        EMBEDDING_MODEL_NAME,
        EXPECTED_EMBEDDING_DIMENSION,
        embedding_model_mismatch_message,
        require_configured_embedding_model,
        require_hf_token,
    )
except ImportError:  # Allows: py career_rag/embed_ai_impact_claims.py
    from config import (  # type: ignore
        BGE_QUERY_PREFIX,
        EMBEDDING_MODEL_NAME,
        EXPECTED_EMBEDDING_DIMENSION,
        embedding_model_mismatch_message,
        require_configured_embedding_model,
        require_hf_token,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "research" / "ai_impact_claims.jsonl"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_research"
DEFAULT_COLLECTION = "research_ai_impact_claims"
DEFAULT_MODEL = EMBEDDING_MODEL_NAME
DEFAULT_BATCH_SIZE = 64

METADATA_FIELDS = [
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
    "impact_direction",
    "impact_direction_clean",
    "affected_entity_type",
    "affected_entity_text",
    "occupation_or_skill_mentions",
    "evidence_strength",
    "confidence",
    "quote_validation",
    "quote_validation_score",
    "ai_relevance",
    "generator_use_scope",
    "review_status",
    "dedup_status",
    "duplicate_group_id",
    "impact_type_clean_reason",
    "impact_direction_clean_reason",
]

TEST_QUERIES = [
    "AI impact on office support workers",
    "generative AI automation exposure for work activities",
    "AI skills demand technological and social emotional skills",
    "AI impact on legal and business professions",
    "workers vulnerable to automation and occupational transitions",
]


def resolve_path(path_text: str) -> Path:
    """Resolve CLI paths relative to the project root."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def clean_string(value: Any) -> str:
    """Return a stripped string for text fields."""
    if value is None:
        return ""
    return str(value).strip()


def console_text(value: Any) -> str:
    """Return text that can be printed on the active console encoding."""
    text = clean_string(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def parse_int(value: Any) -> int | str:
    """Return an int when possible, otherwise an empty string."""
    if value in (None, ""):
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def parse_float(value: Any) -> float | str:
    """Return a float when possible, otherwise an empty string."""
    if value in (None, ""):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def json_string(value: Any) -> str:
    """Store list-like values as JSON strings for Chroma metadata."""
    if value in (None, ""):
        return "[]"
    if isinstance(value, list):
        return json.dumps([clean_string(item) for item in value if clean_string(item)], ensure_ascii=False)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return json.dumps([stripped], ensure_ascii=False)
            if isinstance(parsed, list):
                return json.dumps([clean_string(item) for item in parsed if clean_string(item)], ensure_ascii=False)
        return json.dumps([stripped], ensure_ascii=False) if stripped else "[]"
    return json.dumps([clean_string(value)], ensure_ascii=False)


def mentions_text(value: Any) -> str:
    """Format occupation and skill mentions for embedding text."""
    if isinstance(value, list):
        mentions = [clean_string(item) for item in value if clean_string(item)]
    elif isinstance(value, str) and value.strip():
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                mentions = [stripped]
            else:
                mentions = [clean_string(item) for item in parsed if clean_string(item)] if isinstance(parsed, list) else [stripped]
        else:
            mentions = [stripped]
    else:
        mentions = []
    return ", ".join(mentions) if mentions else "None."


def page_range(claim: dict[str, Any]) -> str:
    """Return a compact source page range for embedding and display."""
    page_start = claim.get("page_start")
    page_end = claim.get("page_end")
    if page_start in (None, "") and page_end in (None, ""):
        return "web"
    if page_start == page_end or page_end in (None, ""):
        return f"page {page_start}"
    if page_start in (None, ""):
        return f"page {page_end}"
    return f"pages {page_start}-{page_end}"


def build_embedding_text(claim: dict[str, Any]) -> str:
    """Build the semantic text stored and embedded for one clean claim."""
    title = clean_string(claim.get("title")) or "Unknown source"
    year = clean_string(claim.get("year")) or "n.d."
    return "\n".join(
        [
            f"AI impact claim: {clean_string(claim.get('claim_text'))}",
            f"Impact type: {clean_string(claim.get('impact_type_clean'))}.",
            f"Direction: {clean_string(claim.get('impact_direction_clean'))}.",
            f"Affected entity type: {clean_string(claim.get('affected_entity_type'))}.",
            f"Affected entity: {clean_string(claim.get('affected_entity_text'))}.",
            f"Mentioned occupations or skills: {mentions_text(claim.get('occupation_or_skill_mentions'))}",
            f"AI relevance: {clean_string(claim.get('ai_relevance'))}.",
            f"Generator use scope: {clean_string(claim.get('generator_use_scope'))}.",
            f"Evidence quote: \"{clean_string(claim.get('evidence_quote'))}\"",
            f"Source: {title}, {year}, {page_range(claim)}.",
        ]
    )


def build_metadata(claim: dict[str, Any]) -> dict[str, str | int | float]:
    """Normalize claim metadata into Chroma-compatible scalar values."""
    metadata: dict[str, str | int | float] = {}
    for field in METADATA_FIELDS:
        value = claim.get(field)
        if field == "occupation_or_skill_mentions":
            metadata[field] = json_string(value)
        elif field in {"chunk_index", "page_start", "page_end"}:
            metadata[field] = parse_int(value)
        elif field == "quote_validation_score":
            metadata[field] = parse_float(value)
        else:
            metadata[field] = clean_string(value)
    return metadata


def should_embed_claim(claim: dict[str, Any]) -> bool:
    """Apply the requested filtering rules for embedding."""
    quote_validation = clean_string(claim.get("quote_validation")).lower()
    dedup_status = clean_string(claim.get("dedup_status")).lower()
    review_status = clean_string(claim.get("review_status")).lower()
    if quote_validation == "failed":
        return False
    if dedup_status != "kept":
        return False
    return review_status in {"pending", "approved"}


def load_claims(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all claims and return the subset eligible for embedding."""
    if not path.exists():
        raise FileNotFoundError(f"Clean claims file not found: {path}")

    all_claims: list[dict[str, Any]] = []
    embed_claims: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                claim = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            all_claims.append(claim)
            if should_embed_claim(claim):
                embed_claims.append(claim)

    return all_claims, embed_claims


def prepare_documents(claims: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Convert claims into Chroma IDs, documents, and metadatas."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, claim in enumerate(claims, start=1):
        claim_id = clean_string(claim.get("claim_id")) or f"claim_{index:06d}"
        if claim_id in seen_ids:
            raise ValueError(f"Duplicate claim_id in clean claims file: {claim_id}")
        seen_ids.add(claim_id)

        ids.append(claim_id)
        documents.append(build_embedding_text(claim))
        metadatas.append(build_metadata(claim))

    return ids, documents, metadatas


def load_model(model_name: str) -> SentenceTransformer:
    """Load the requested sentence-transformer model."""
    model_name = require_configured_embedding_model(model_name)
    require_hf_token()
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dimension = get_embedding_dimension(model)
    print(f"Model loaded. Embedding dimension: {dimension}")
    if dimension != EXPECTED_EMBEDDING_DIMENSION:
        print(
            f"Warning: expected embedding dimension {EXPECTED_EMBEDDING_DIMENSION}, got {dimension}.",
            file=sys.stderr,
        )
    return model


def get_embedding_dimension(model: SentenceTransformer) -> int:
    """Return embedding dimension across sentence-transformers versions."""
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    """Create embeddings for all claim texts."""
    print(f"Embedding {len(texts):,} claim document(s) with batch size {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return embeddings


def source_file_for_metadata(input_path: Path) -> str:
    """Return a stable source-file value for collection metadata."""
    try:
        return str(input_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(input_path)


def collection_metadata(model_name: str, embedding_dimension: int, input_path: Path) -> dict[str, Any]:
    """Build collection-level metadata describing the embedding setup."""
    return {
        "embedding_model": model_name,
        "embedding_dimension": embedding_dimension,
        "source_file": source_file_for_metadata(input_path),
    }


def validate_reset_target(persist_dir: Path) -> Path:
    """Resolve and validate that only the research Chroma directory is reset."""
    resolved = persist_dir.resolve()
    expected = DEFAULT_PERSIST_DIR.resolve()
    if resolved != expected:
        raise ValueError(
            f"--reset-persist-dir only supports the default research path: {expected}"
        )
    if resolved == (PROJECT_ROOT / "data" / "chroma_onet").resolve():
        raise ValueError("Refusing to reset O*NET Chroma directory.")
    if resolved == (PROJECT_ROOT / "data" / "research").resolve():
        raise ValueError("Refusing to reset data/research.")
    return resolved


def reset_persist_dir_if_requested(persist_dir: Path, reset_persist_dir: bool) -> None:
    """Delete chroma_research when explicitly requested."""
    if not reset_persist_dir:
        return
    resolved = validate_reset_target(persist_dir)
    if resolved.exists():
        print(f"Resetting research Chroma persist directory: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def create_or_replace_collection(
    persist_dir: Path,
    collection_name: str,
    recreate: bool,
    metadata: dict[str, Any],
) -> chromadb.Collection:
    """Create or replace the Chroma collection."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        existing = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name, metadata=metadata)
        print(f"Created collection '{collection_name}'.")
        return collection

    existing_metadata = existing.metadata or {}
    existing_model = clean_string(existing_metadata.get("embedding_model"))
    expected_model = clean_string(metadata.get("embedding_model"))
    if existing_model and existing_model != expected_model:
        if not recreate:
            raise RuntimeError(
                embedding_model_mismatch_message(
                    collection_name,
                    existing_model,
                    expected_model,
                )
            )

    if recreate:
        print(
            f"Warning: old research collection '{collection_name}' exists and will be recreated."
        )
        client.delete_collection(name=collection_name)
        collection = client.create_collection(name=collection_name, metadata=metadata)
        print(f"Recreated collection '{collection_name}'.")
        return collection

    print(f"Using existing collection '{collection_name}'.")
    return client.get_collection(name=collection_name)


def add_documents_in_batches(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
    batch_size: int,
) -> int:
    """Add embedded claim documents to Chroma in batches."""
    inserted = 0
    total_batches = (len(ids) + batch_size - 1) // batch_size
    print(f"Adding documents to collection '{collection.name}'...")

    with tqdm(total=len(ids), desc="Adding to Chroma") as progress_bar:
        for batch_index in range(total_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(ids))
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )
            inserted += end - start
            progress_bar.update(end - start)

    return inserted


def query_collection(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query: str,
    top_k: int = 5,
) -> tuple[list[str], list[float], list[dict[str, Any]], list[str]]:
    """Run one semantic query against the research collection."""
    query_text = f"{BGE_QUERY_PREFIX}{query}"
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"],
    )
    return (
        results.get("ids", [[]])[0],
        results.get("distances", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("documents", [[]])[0],
    )


def print_retrieval_result(rank: int, distance: float, metadata: dict[str, Any]) -> None:
    """Print one retrieval hit with the requested metadata fields."""
    page_start = metadata.get("page_start", "")
    page_end = metadata.get("page_end", "")
    page_text = f"{page_start}-{page_end}" if page_end not in ("", page_start) else str(page_start)
    if not page_text:
        page_text = "web"

    print(f"  Rank: {rank}")
    print(f"  Distance: {float(distance):.4f}")
    print(f"  Claim ID: {console_text(metadata.get('claim_id', ''))}")
    print(f"  Title: {console_text(metadata.get('title', ''))}")
    print(f"  Year: {console_text(metadata.get('year', ''))}")
    print(f"  Pages: {page_text}")
    print(f"  Impact type: {console_text(metadata.get('impact_type_clean', ''))}")
    print(f"  Direction: {console_text(metadata.get('impact_direction_clean', ''))}")
    print(f"  AI relevance: {console_text(metadata.get('ai_relevance', ''))}")
    print(f"  Generator use scope: {console_text(metadata.get('generator_use_scope', ''))}")
    print(f"  Claim: {console_text(metadata.get('claim_text', ''))}")
    print(f"  Quote: {console_text(metadata.get('evidence_quote', ''))}")


def validate_collection(
    persist_dir: Path,
    collection_name: str,
    model: SentenceTransformer,
    embedding_dimension: int,
    expected_model: str,
) -> None:
    """Validate collection creation and print sample retrieval results."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=collection_name)
    count = collection.count()

    actual_dimension = embedding_dimension
    try:
        sample = collection.get(limit=1, include=["embeddings"])
        sample_embeddings = sample.get("embeddings")
        if sample_embeddings is not None and len(sample_embeddings) > 0:
            actual_dimension = len(sample_embeddings[0])
    except Exception as exc:
        print(f"Could not inspect stored embedding dimension: {exc}")

    print("\nValidation")
    print(f"Collection exists: {collection.name}")
    print(f"Embedded documents: {count:,}")
    print(f"Embedding dimension: {actual_dimension}")
    metadata = collection.metadata or {}
    metadata_model = clean_string(metadata.get("embedding_model"))
    print(f"Collection metadata model: {metadata_model or '(missing)'}")
    if metadata_model != expected_model:
        raise RuntimeError(
            embedding_model_mismatch_message(
                collection_name,
                metadata_model,
                expected_model,
            )
        )
    if actual_dimension != EXPECTED_EMBEDDING_DIMENSION:
        print(
            f"Warning: expected embedding dimension {EXPECTED_EMBEDDING_DIMENSION}, got {actual_dimension}.",
            file=sys.stderr,
        )

    for query in TEST_QUERIES:
        print("\n" + "-" * 80)
        print(f"Query: {query}")
        ids, distances, metadatas, _documents = query_collection(collection, model, query, top_k=5)
        for rank, (_doc_id, distance, metadata) in enumerate(zip(ids, distances, metadatas), start=1):
            print_retrieval_result(rank, distance, metadata or {})


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Embed clean AI-impact claims into a ChromaDB research collection."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=[DEFAULT_MODEL])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--reset-persist-dir",
        action="store_true",
        help="Delete and recreate only the default chroma_research directory before embedding.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=True,
        help="Delete and recreate the collection if it already exists. This is the default.",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_false",
        dest="recreate",
        help="Use an existing collection instead of deleting it.",
    )
    return parser.parse_args()


def main() -> None:
    """Embed clean research claims and validate retrieval."""
    args = parse_args()
    input_path = resolve_path(args.input)
    persist_dir = resolve_path(args.persist_dir)
    start_time = time.time()

    print("=" * 80)
    print("AI IMPACT CLAIM EMBEDDING")
    print("=" * 80)
    print(f"Input file:       {input_path}")
    print(f"Persist dir:      {persist_dir}")
    print(f"Collection:       {args.collection}")
    print(f"Embedding model:  {args.model}")
    print(f"Batch size:       {args.batch_size}")

    try:
        all_claims, embed_claims = load_claims(input_path)
        ids, documents, metadatas = prepare_documents(embed_claims)
        model = load_model(args.model)
        embedding_dimension = get_embedding_dimension(model)
        embeddings = embed_texts(model, documents, args.batch_size)
        reset_persist_dir_if_requested(persist_dir, args.reset_persist_dir)
        metadata = collection_metadata(args.model, embedding_dimension, input_path)
        collection = create_or_replace_collection(
            persist_dir=persist_dir,
            collection_name=args.collection,
            recreate=args.recreate,
            metadata=metadata,
        )
        inserted = add_documents_in_batches(
            collection=collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            batch_size=args.batch_size,
        )

        elapsed = time.time() - start_time
        print("\nEmbedding complete")
        print(f"Claims loaded:         {len(all_claims):,}")
        print(f"Claims embedded:       {inserted:,}")
        print(f"Collection count:      {collection.count():,}")
        print(f"Collection name:       {args.collection}")
        print(f"Persist directory:     {persist_dir}")
        print(f"Embedding model:       {args.model}")
        print(f"Embedding dimension:   {embedding_dimension}")
        print(f"Elapsed time:          {elapsed:.2f} seconds")

        validate_collection(
            persist_dir=persist_dir,
            collection_name=args.collection,
            model=model,
            embedding_dimension=embedding_dimension,
            expected_model=args.model,
        )

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
