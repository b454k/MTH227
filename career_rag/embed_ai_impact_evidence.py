#!/usr/bin/env python3
"""Embed AI impact evidence and research inference chunks into Chroma."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from career_rag.config import (
        EMBEDDING_MODEL_NAME,
        EXPECTED_EMBEDDING_DIMENSION,
        embedding_model_mismatch_message,
        quiet_huggingface_model_load,
        require_configured_embedding_model,
        require_hf_token,
    )
    from career_rag.ai_exposure_utils import (
        PROJECT_ROOT,
        chroma_scalar,
        one_line,
        read_jsonl,
        resolve_project_path,
    )
except ImportError:  # Allows: py career_rag/embed_ai_impact_evidence.py
    from config import (  # type: ignore
        EMBEDDING_MODEL_NAME,
        EXPECTED_EMBEDDING_DIMENSION,
        embedding_model_mismatch_message,
        quiet_huggingface_model_load,
        require_configured_embedding_model,
        require_hf_token,
    )
    from ai_exposure_utils import (  # type: ignore
        PROJECT_ROOT,
        chroma_scalar,
        one_line,
        read_jsonl,
        resolve_project_path,
    )


DEFAULT_AI_INPUT = PROJECT_ROOT / "data" / "processed" / "ai_impact_evidence_deduped.jsonl"
DEFAULT_AI_CHROMA = PROJECT_ROOT / "chroma_ai_impact"
DEFAULT_AI_COLLECTION = "ai_impact_evidence"
DEFAULT_RESEARCH_CHROMA = PROJECT_ROOT / "chroma_research"
DEFAULT_RESEARCH_COLLECTION = "research_inference"
DEFAULT_RESEARCH_INPUTS = [
    PROJECT_ROOT / "data" / "processed" / "research_inference_chunks.jsonl",
    PROJECT_ROOT / "data" / "processed" / "nber_w31222_method_chunks.jsonl",
]
DEFAULT_MODEL = EMBEDDING_MODEL_NAME
DEFAULT_BATCH_SIZE = 128

AI_METADATA_FIELDS = [
    "doc_id",
    "doc_type",
    "source_id",
    "source_name",
    "source_release",
    "source_file",
    "source_page",
    "soc_code",
    "occupation_title",
    "onet_task_id",
    "task_text",
    "task_cluster",
    "field_or_industry",
    "impact_type",
    "metric_name",
    "metric_value",
    "metric_unit",
    "metric_percentile",
    "confidence",
    "statistics_allowed",
]

RESEARCH_METADATA_FIELDS = [
    "doc_id",
    "doc_type",
    "source_id",
    "source_name",
    "source_url",
    "source_file",
    "page",
    "source_page",
    "allowed_usage",
    "statistics_allowed",
]


def source_page(row: dict[str, Any]) -> Any:
    """Return the best page field."""
    return row.get("source_page") if row.get("source_page") not in (None, "") else row.get("page")


def build_ai_embedding_text(row: dict[str, Any]) -> str:
    """Build the requested AI impact embedding text format."""
    source_release = one_line(row.get("source_release")) or one_line(row.get("source_file"))
    return "\n".join(
        [
            f"Occupation: {one_line(row.get('occupation_title')) or 'N/A'} ({one_line(row.get('soc_code')) or 'N/A'})",
            f"Task: {one_line(row.get('task_text')) or 'N/A'}",
            f"Task cluster: {one_line(row.get('task_cluster')) or 'N/A'}",
            f"Field/industry: {one_line(row.get('field_or_industry')) or 'N/A'}",
            f"Impact type: {one_line(row.get('impact_type')) or 'N/A'}",
            f"Metric: {one_line(row.get('metric_name')) or 'N/A'} = {one_line(row.get('metric_value')) or 'N/A'} {one_line(row.get('metric_unit'))}",
            f"Percentile: {one_line(row.get('metric_percentile')) or 'N/A'}",
            f"Time period: {one_line(row.get('time_period')) or 'N/A'}",
            f"Evidence: {one_line(row.get('evidence_text'))}",
            f"Interpretation: {one_line(row.get('interpretation'))}",
            f"Source: {one_line(row.get('source_name'))}, {source_release}, page {one_line(source_page(row)) or 'N/A'}",
        ]
    )


def build_research_embedding_text(row: dict[str, Any]) -> str:
    """Build embedding text for methodology/caveat chunks."""
    text = row.get("chunk_text") or row.get("evidence_text") or row.get("text") or ""
    return "\n".join(
        [
            f"Research methodology or caveat: {one_line(text)}",
            f"Allowed usage: {one_line(row.get('allowed_usage')) or 'methodology_caveat_inference_only'}",
            f"Statistics allowed: {one_line(row.get('statistics_allowed'))}",
            f"Source: {one_line(row.get('source_name')) or one_line(row.get('source_id'))}, page {one_line(source_page(row)) or 'N/A'}",
        ]
    )


def metadata_for_row(row: dict[str, Any], fields: list[str]) -> dict[str, str | int | float | bool]:
    """Build Chroma-compatible metadata."""
    metadata = {field: chroma_scalar(row.get(field)) for field in fields}
    metadata["source_page"] = chroma_scalar(source_page(row))
    if "page" in fields:
        metadata["page"] = chroma_scalar(source_page(row))
    return metadata


def load_ai_documents(path: Path) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Load AI evidence rows for embedding."""
    rows = read_jsonl(path)
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        doc_id = one_line(row.get("doc_id")) or f"ai_impact_{index:06d}"
        if doc_id in seen_ids:
            doc_id = f"{doc_id}_{index:06d}"
        seen_ids.add(doc_id)
        ids.append(doc_id)
        documents.append(build_ai_embedding_text(row))
        metadatas.append(metadata_for_row(row, AI_METADATA_FIELDS))
    return ids, documents, metadatas


def load_research_documents(paths: list[Path]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Load research inference/methodology chunks for embedding."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    counter = 0
    for path in paths:
        for row in read_jsonl(path):
            counter += 1
            doc_id = one_line(row.get("doc_id")) or f"research_inference_{counter:06d}"
            if doc_id in seen_ids:
                doc_id = f"{doc_id}_{counter:06d}"
            seen_ids.add(doc_id)
            ids.append(doc_id)
            documents.append(build_research_embedding_text(row))
            metadatas.append(metadata_for_row(row, RESEARCH_METADATA_FIELDS))
    return ids, documents, metadatas


def create_or_replace_collection(
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    recreate: bool,
    input_files: list[str],
) -> chromadb.Collection:
    """Create or replace one Chroma collection."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    metadata = {
        "embedding_model": model_name,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "source_files": ";".join(input_files),
    }
    try:
        existing = client.get_collection(collection_name)
    except Exception:
        return client.create_collection(name=collection_name, metadata=metadata)
    if recreate:
        client.delete_collection(name=collection_name)
        return client.create_collection(name=collection_name, metadata=metadata)
    existing_model = one_line((existing.metadata or {}).get("embedding_model"))
    if existing_model != model_name:
        raise RuntimeError(
            embedding_model_mismatch_message(collection_name, existing_model, model_name)
        )
    return client.get_collection(collection_name)


def add_to_collection(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
    batch_size: int,
) -> int:
    """Add documents to Chroma in batches."""
    if not ids:
        return 0
    inserted = 0
    for start in tqdm(range(0, len(ids), batch_size), desc=f"Adding {collection.name}"):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )
        inserted += end - start
    return inserted


def embed_and_add_to_collection(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    batch_size: int,
) -> int:
    """Embed and insert documents batch by batch."""
    if not ids:
        return 0
    inserted = 0
    for start in tqdm(range(0, len(ids), batch_size), desc=f"Embedding {collection.name}"):
        end = min(start + batch_size, len(ids))
        embeddings = model.encode(
            documents[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
        inserted += end - start
    return inserted


def reset_ai_dir_if_requested(path: Path, reset: bool) -> None:
    """Reset only the AI impact Chroma directory when explicitly requested."""
    if not reset:
        return
    resolved = path.resolve()
    expected = DEFAULT_AI_CHROMA.resolve()
    if resolved != expected:
        raise ValueError(f"--reset-ai-persist-dir only supports {expected}")
    if resolved.exists():
        shutil.rmtree(resolved)


def embed_collection(
    model: SentenceTransformer,
    persist_dir: Path,
    collection_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    recreate: bool,
    input_files: list[str],
) -> int:
    """Create embeddings and persist one collection."""
    collection = create_or_replace_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
        recreate=recreate,
        input_files=input_files,
    )
    return embed_and_add_to_collection(
        collection,
        model,
        ids,
        documents,
        metadatas,
        batch_size,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Embed AI impact evidence and research inference chunks.")
    parser.add_argument("--ai-input", default=str(DEFAULT_AI_INPUT))
    parser.add_argument("--ai-persist-dir", default=str(DEFAULT_AI_CHROMA))
    parser.add_argument("--ai-collection", default=DEFAULT_AI_COLLECTION)
    parser.add_argument("--research-inputs", nargs="*", default=[str(path) for path in DEFAULT_RESEARCH_INPUTS])
    parser.add_argument("--research-persist-dir", default=str(DEFAULT_RESEARCH_CHROMA))
    parser.add_argument("--research-collection", default=DEFAULT_RESEARCH_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=[DEFAULT_MODEL])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no-recreate", action="store_true")
    parser.add_argument("--skip-ai", action="store_true")
    parser.add_argument("--skip-research", action="store_true")
    parser.add_argument("--reset-ai-persist-dir", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run embedding."""
    args = parse_args()
    ai_input = resolve_project_path(args.ai_input)
    ai_persist_dir = resolve_project_path(args.ai_persist_dir)
    research_inputs = [resolve_project_path(path) for path in args.research_inputs]
    research_persist_dir = resolve_project_path(args.research_persist_dir)
    recreate = not args.no_recreate

    reset_ai_dir_if_requested(ai_persist_dir, args.reset_ai_persist_dir)
    model_name = require_configured_embedding_model(args.model)
    require_hf_token()
    print(f"Loading embedding model: {model_name}")
    with quiet_huggingface_model_load():
        model = SentenceTransformer(model_name)

    ai_inserted = 0
    research_inserted = 0

    if not args.skip_ai:
        if not ai_input.exists():
            raise FileNotFoundError(f"AI evidence input not found: {ai_input}")
        ids, documents, metadatas = load_ai_documents(ai_input)
        ai_inserted = embed_collection(
            model,
            ai_persist_dir,
            args.ai_collection,
            ids,
            documents,
            metadatas,
            model_name,
            args.batch_size,
            recreate,
            [str(ai_input)],
        )

    if not args.skip_research:
        existing_research_inputs = [path for path in research_inputs if path.exists()]
        ids, documents, metadatas = load_research_documents(existing_research_inputs)
        research_inserted = embed_collection(
            model,
            research_persist_dir,
            args.research_collection,
            ids,
            documents,
            metadatas,
            model_name,
            args.batch_size,
            recreate,
            [str(path) for path in existing_research_inputs],
        )

    print("\nEmbedding complete")
    print(f"AI impact documents embedded: {ai_inserted:,}")
    print(f"AI Chroma path: {ai_persist_dir}")
    print(f"AI collection: {args.ai_collection}")
    print(f"Research inference documents embedded: {research_inserted:,}")
    print(f"Research Chroma path: {research_persist_dir}")
    print(f"Research collection: {args.research_collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
