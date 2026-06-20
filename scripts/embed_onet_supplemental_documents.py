#!/usr/bin/env python3
"""Embed supplemental O*NET documents into a separate ChromaDB collection."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from career_rag.config import (
    EMBEDDING_MODEL_NAME,
    EXPECTED_EMBEDDING_DIMENSION,
    quiet_huggingface_model_load,
    require_hf_token,
)


MODEL_NAME = EMBEDDING_MODEL_NAME
DOCUMENTS_JSONL = Path("data/documents/onet_supplemental_documents.jsonl")
CHROMA_DB_PATH = Path("data/chroma_onet")
COLLECTION_NAME = "onet_supplemental"
DEFAULT_BATCH_SIZE = 50

COLLECTION_METADATA = {
    "embedding_model": MODEL_NAME,
    "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
    "source_file": str(DOCUMENTS_JSONL),
}


def clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Normalize metadata to Chroma-supported scalar values."""
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif value is None:
            cleaned[key] = ""
        else:
            cleaned[key] = str(value)
    return cleaned


def load_documents() -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Load supplemental document IDs, texts, and metadata from JSONL."""
    if not DOCUMENTS_JSONL.exists():
        raise FileNotFoundError(f"Document file not found: {DOCUMENTS_JSONL}")

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    print(f"Loading documents from {DOCUMENTS_JSONL}...")
    with DOCUMENTS_JSONL.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc

            doc_id = str(document.get("id") or "").strip()
            text = str(document.get("text") or "").strip()
            if not doc_id or not text:
                raise ValueError(f"Missing id or text on line {line_number}")

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(clean_metadata(document.get("metadata", {}) or {}))

    print(f"Loaded {len(ids):,} supplemental documents")
    return ids, texts, metadatas


def load_model() -> SentenceTransformer:
    """Load the embedding model used by the existing O*NET collections."""
    print(f"\nLoading embedding model: {MODEL_NAME}")
    require_hf_token()
    with quiet_huggingface_model_load():
        model = SentenceTransformer(MODEL_NAME)
    if hasattr(model, "get_embedding_dimension"):
        embedding_dimension = model.get_embedding_dimension()
    else:
        embedding_dimension = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {embedding_dimension}")
    return model


def create_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Create embeddings for all supplemental documents."""
    print(f"\nEmbedding {len(texts):,} supplemental documents...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return embeddings


def get_or_create_collection() -> chromadb.Collection:
    """Connect to ChromaDB and get or create the supplemental collection."""
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=COLLECTION_METADATA,
    )
    print(f"\nUsing collection: {collection.name}")
    return collection


def get_existing_ids(collection: chromadb.Collection) -> set[str]:
    """Fetch existing document IDs from the supplemental collection."""
    try:
        existing = collection.get()
    except Exception:
        return set()
    return set(existing.get("ids", []) or [])


def insert_batches(
    collection: chromadb.Collection,
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Insert only new supplemental documents into ChromaDB."""
    existing_ids = get_existing_ids(collection)
    if existing_ids:
        print(f"Collection already contains {len(existing_ids):,} documents")

    inserted_count = 0
    skipped_count = 0
    total_batches = (len(ids) + batch_size - 1) // batch_size

    print(f"\nInserting new documents into '{collection.name}'...")
    with tqdm(total=len(ids), desc="Inserting supplemental docs") as progress_bar:
        for batch_index in range(total_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(ids))

            batch_ids = ids[start:end]
            new_indices = [
                index
                for index, doc_id in enumerate(batch_ids)
                if doc_id not in existing_ids
            ]

            if new_indices:
                collection.add(
                    ids=[batch_ids[index] for index in new_indices],
                    documents=[texts[start + index] for index in new_indices],
                    embeddings=[embeddings[start + index] for index in new_indices],
                    metadatas=[metadatas[start + index] for index in new_indices],
                )
                for index in new_indices:
                    existing_ids.add(batch_ids[index])
                inserted_count += len(new_indices)

            skipped_count += len(batch_ids) - len(new_indices)
            progress_bar.update(len(batch_ids))

    if skipped_count:
        print(f"Skipped {skipped_count:,} duplicate IDs already in collection")

    return inserted_count


def print_validation(
    collection: chromadb.Collection,
    total_docs: int,
    inserted_count: int,
    elapsed_time: float,
) -> None:
    """Print the requested validation summary."""
    print("\n" + "=" * 80)
    print("SUPPLEMENTAL COLLECTION VALIDATION")
    print("=" * 80)
    print(f"Total supplemental docs loaded: {total_docs:,}")
    print(f"Total embedded:                 {inserted_count:,}")
    print(f"Collection count:               {collection.count():,}")
    print(f"Elapsed time:                   {elapsed_time:.2f} seconds")

    sample = collection.get(limit=1, include=["metadatas", "documents"])
    if sample.get("ids"):
        print(f"Example document id:            {sample['ids'][0]}")
        print(f"Example metadata:               {sample['metadatas'][0]}")


def main() -> None:
    """Embed supplemental O*NET documents into ChromaDB."""
    print("=" * 80)
    print("O*NET SUPPLEMENTAL DOCUMENT EMBEDDING")
    print("=" * 80)
    print(f"Input file:       {DOCUMENTS_JSONL}")
    print(f"ChromaDB path:    {CHROMA_DB_PATH}")
    print(f"Collection name:  {COLLECTION_NAME}")
    print(f"Embedding model:  {MODEL_NAME}")

    start_time = time.time()

    try:
        ids, texts, metadatas = load_documents()
        model = load_model()
        embeddings = create_embeddings(texts, model)
        collection = get_or_create_collection()
        inserted_count = insert_batches(
            collection=collection,
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            batch_size=DEFAULT_BATCH_SIZE,
        )
        print_validation(
            collection=collection,
            total_docs=len(ids),
            inserted_count=inserted_count,
            elapsed_time=time.time() - start_time,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
