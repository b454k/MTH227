#!/usr/bin/env python3
"""Embed full O*NET occupation documents into a separate ChromaDB collection.

This script reads full occupation documents from JSONL, embeds them with the
same model used for section documents, and stores them in the existing ChromaDB
path under a new collection named ``onet_full_occupations``.

It does not modify or delete the existing ``onet_sections`` collection.
"""

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
    require_hf_token,
)


MODEL_NAME = EMBEDDING_MODEL_NAME
DOCUMENTS_JSONL = Path("data/documents/onet_occupation_documents.jsonl")
CHROMA_DB_PATH = Path("data/chroma_onet")
COLLECTION_NAME = "onet_full_occupations"
DEFAULT_BATCH_SIZE = 50

COLLECTION_METADATA = {
    "embedding_model": MODEL_NAME,
    "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
    "source_file": str(DOCUMENTS_JSONL),
}


def load_documents() -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Load document IDs, texts, and metadata from the JSONL input file."""
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

            try:
                doc_id = document["id"]
                text = document["text"]
            except KeyError as exc:
                raise KeyError(
                    f"Missing required key {exc!s} on line {line_number}"
                ) from exc

            ids.append(str(doc_id))
            texts.append(str(text))
            metadatas.append(document.get("metadata", {}) or {})

    print(f"Loaded {len(ids):,} documents")
    return ids, texts, metadatas


def load_model() -> SentenceTransformer:
    """Load the sentence-transformer model used during indexing."""
    print(f"\nLoading embedding model: {MODEL_NAME}")
    require_hf_token()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def create_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Create embeddings for all document texts."""
    print(f"\nEmbedding {len(texts):,} documents...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return embeddings


def get_or_create_collection(
    chroma_path: Path = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Connect to ChromaDB and get or create the full occupation collection."""
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=COLLECTION_METADATA,
    )
    print(f"\nUsing collection: {collection.name}")
    return collection


def insert_batches(
    collection: chromadb.Collection,
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Insert embedded documents into ChromaDB in batches."""
    print(f"\nInserting documents into '{collection.name}'...")
    print(f"Batch size: {batch_size}")

    existing_docs = collection.get()
    existing_ids = set(existing_docs.get("ids", []))
    if existing_ids:
        print(f"Collection already contains {len(existing_ids):,} documents")

    inserted_count = 0
    skipped_count = 0
    total_batches = (len(ids) + batch_size - 1) // batch_size

    with tqdm(total=len(ids), desc="Inserting documents") as progress_bar:
        for batch_index in range(total_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(ids))

            batch_ids = ids[start:end]
            batch_texts = texts[start:end]
            batch_embeddings = embeddings[start:end]
            batch_metadatas = metadatas[start:end]

            new_indices = [
                index
                for index, doc_id in enumerate(batch_ids)
                if doc_id not in existing_ids
            ]

            if new_indices:
                collection.add(
                    ids=[batch_ids[index] for index in new_indices],
                    documents=[batch_texts[index] for index in new_indices],
                    embeddings=[batch_embeddings[index] for index in new_indices],
                    metadatas=[batch_metadatas[index] for index in new_indices],
                )

                for index in new_indices:
                    existing_ids.add(batch_ids[index])
                inserted_count += len(new_indices)

            skipped_count += len(batch_ids) - len(new_indices)
            progress_bar.update(len(batch_ids))

    if skipped_count:
        print(f"Skipped {skipped_count:,} documents already present in collection")

    return inserted_count


def main() -> None:
    """Embed full occupation documents and store them in ChromaDB."""
    print("=" * 80)
    print("O*NET FULL OCCUPATION DOCUMENT EMBEDDING")
    print("=" * 80)
    print(f"Input file:       {DOCUMENTS_JSONL}")
    print(f"ChromaDB path:    {CHROMA_DB_PATH}")
    print(f"Collection name:  {COLLECTION_NAME}")
    print(f"Embedding model:  {MODEL_NAME}")

    start_time = time.time()

    try:
        ids, texts, metadatas = load_documents()
        model = load_model()
        embeddings = create_embeddings(texts, model, batch_size=DEFAULT_BATCH_SIZE)
        collection = get_or_create_collection()

        inserted_count = insert_batches(
            collection=collection,
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            batch_size=DEFAULT_BATCH_SIZE,
        )

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("EMBEDDING COMPLETE")
        print("=" * 80)
        print(f"Total input documents:   {len(ids):,}")
        print(f"Documents embedded:      {inserted_count:,}")
        print(f"Collection count:        {collection.count():,}")
        print(f"Elapsed time:            {elapsed_time:.2f} seconds")
        print("=" * 80)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
