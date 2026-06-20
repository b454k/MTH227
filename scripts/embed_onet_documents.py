#!/usr/bin/env python3
"""
Create embeddings and store them in ChromaDB for O*NET RAG retrieval.

Reads JSONL documents, creates embeddings using sentence-transformers,
and inserts them into ChromaDB collection for semantic search.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any

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


# Configuration
MODEL_NAME = EMBEDDING_MODEL_NAME
DOCUMENTS_JSONL = Path("data/documents/onet_occupation_section_documents.jsonl")
CHROMA_DB_PATH = Path("data/chroma_onet")
COLLECTION_NAME = "onet_sections"
DEFAULT_BATCH_SIZE = 50
REBUILD_COLLECTION = os.getenv("REBUILD_COLLECTION", "false").lower() == "true"

COLLECTION_METADATA = {
    "embedding_model": MODEL_NAME,
    "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
    "source_file": str(DOCUMENTS_JSONL),
}


def load_documents() -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Load documents from JSONL file.
    
    Returns:
        Tuple of (ids, texts, metadatas)
    """
    ids = []
    texts = []
    metadatas = []
    
    print(f"Loading documents from {DOCUMENTS_JSONL}...")
    
    if not DOCUMENTS_JSONL.exists():
        raise FileNotFoundError(f"Document file not found: {DOCUMENTS_JSONL}")
    
    with open(DOCUMENTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc["id"])
            texts.append(doc["text"])
            metadatas.append(doc.get("metadata", {}))
    
    print(f"Loaded {len(ids)} documents")
    return ids, texts, metadatas


def create_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32
) -> List[List[float]]:
    """
    Create embeddings for texts using SentenceTransformer.
    
    Args:
        texts: List of text strings to embed
        model: SentenceTransformer model instance
        batch_size: Batch size for encoding
    
    Returns:
        List of embedding vectors
    """
    print(f"\nCreating embeddings using {MODEL_NAME}...")
    print(f"Encoding {len(texts)} texts with batch size {batch_size}")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Convert numpy array to list if needed
    if hasattr(embeddings, 'tolist'):
        return embeddings.tolist()
    return embeddings


def create_collection(
    chroma_db_path: Path,
    collection_name: str,
    rebuild: bool = False
) -> chromadb.Collection:
    """
    Create or retrieve ChromaDB collection.
    
    Args:
        chroma_db_path: Path to ChromaDB storage
        collection_name: Name of collection
        rebuild: Whether to rebuild collection if it exists
    
    Returns:
        ChromaDB collection object
    """
    # Create storage directory if needed
    chroma_db_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB client with persistent storage (new API)
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    
    # Check if collection exists
    try:
        existing_collection = client.get_collection(name=collection_name)
        if existing_collection is not None:
            if rebuild:
                print(f"\nCollection '{collection_name}' exists. Deleting (REBUILD_COLLECTION=true)...")
                client.delete_collection(name=collection_name)
                collection = client.create_collection(name=collection_name, metadata=COLLECTION_METADATA)
                print(f"Created new collection '{collection_name}'")
            else:
                print(f"\nCollection '{collection_name}' already exists.")
                user_input = input("Overwrite existing collection? (y/n): ").strip().lower()
                if user_input == "y":
                    print(f"Deleting collection '{collection_name}'...")
                    client.delete_collection(name=collection_name)
                    collection = client.create_collection(name=collection_name, metadata=COLLECTION_METADATA)
                    print(f"Created new collection '{collection_name}'")
                else:
                    print("Keeping existing collection. Exiting.")
                    sys.exit(0)
        else:
            collection = client.create_collection(name=collection_name, metadata=COLLECTION_METADATA)
            print(f"Created new collection '{collection_name}'")
    except Exception:
        # Collection doesn't exist, create it
        collection = client.create_collection(name=collection_name, metadata=COLLECTION_METADATA)
        print(f"Created new collection '{collection_name}'")
    
    return collection


def insert_batches(
    collection: chromadb.Collection,
    ids: List[str],
    embeddings: List[List[float]],
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE
) -> int:
    """
    Insert documents into ChromaDB in batches.
    
    Args:
        collection: ChromaDB collection object
        ids: Document IDs
        embeddings: Embedding vectors
        texts: Document texts
        metadatas: Document metadata
        batch_size: Number of documents per batch
    
    Returns:
        Total number of documents inserted
    """
    print(f"\nInserting {len(ids)} documents into collection '{collection.name}'...")
    print(f"Batch size: {batch_size}")
    
    # Track duplicates and inserted count
    inserted_count = 0
    duplicate_count = 0
    existing_ids = set()
    
    try:
        # Get existing IDs to check for duplicates
        existing_docs = collection.get()
        if existing_docs and "ids" in existing_docs:
            existing_ids = set(existing_docs["ids"])
            print(f"Collection already contains {len(existing_ids)} documents")
    except Exception:
        pass
    
    # Insert in batches
    num_batches = (len(ids) + batch_size - 1) // batch_size
    
    with tqdm(total=len(ids), desc="Inserting documents") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(ids))
            
            batch_ids = ids[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_texts = texts[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]
            
            # Filter out duplicates
            filtered_indices = []
            for i, doc_id in enumerate(batch_ids):
                if doc_id not in existing_ids:
                    filtered_indices.append(i)
                else:
                    duplicate_count += 1
            
            if filtered_indices:
                # Insert non-duplicate documents
                collection.add(
                    ids=[batch_ids[i] for i in filtered_indices],
                    embeddings=[batch_embeddings[i] for i in filtered_indices],
                    documents=[batch_texts[i] for i in filtered_indices],
                    metadatas=[batch_metadatas[i] for i in filtered_indices]
                )
                inserted_count += len(filtered_indices)
                
                # Update existing IDs set
                for i in filtered_indices:
                    existing_ids.add(batch_ids[i])
            
            pbar.update(len(batch_ids))
    
    if duplicate_count > 0:
        print(f"Skipped {duplicate_count} duplicate documents")
    
    return inserted_count


def print_validation(
    collection: chromadb.Collection,
    embedding_dim: int,
    elapsed_time: float,
    inserted_count: int
) -> None:
    """
    Print validation summary after insertion.
    
    Args:
        collection: ChromaDB collection object
        embedding_dim: Dimension of embeddings
        elapsed_time: Total insertion time in seconds
        inserted_count: Number of documents inserted
    """
    print("\n" + "=" * 80)
    print("COLLECTION VALIDATION")
    print("=" * 80)
    
    try:
        collection_info = collection.count()
        print(f"\nCollection name: {collection.name}")
        print(f"Document count: {collection_info}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Documents inserted: {inserted_count}")
        print(f"Insertion time: {elapsed_time:.2f} seconds")
        
        # Get example documents
        try:
            sample_docs = collection.get(limit=1)
            if sample_docs and len(sample_docs["ids"]) > 0:
                print(f"\nExample document ID: {sample_docs['ids'][0]}")
                print(f"Example metadata: {sample_docs['metadatas'][0]}")
        except Exception as e:
            print(f"Could not retrieve sample documents: {e}")
        
        print("\n" + "=" * 80)
        print("COLLECTION READY FOR RAG RETRIEVAL")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error validating collection: {e}")


def main() -> None:
    """Main execution function."""
    print("\n" + "=" * 80)
    print("O*NET DOCUMENT EMBEDDING & CHROMADB INSERTION")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Load documents
        ids, texts, metadatas = load_documents()
        
        # Load embedding model
        print(f"\nLoading embedding model: {MODEL_NAME}")
        require_hf_token()
        with quiet_huggingface_model_load():
            model = SentenceTransformer(MODEL_NAME)
        embedding_dim = model.get_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {embedding_dim}")
        
        # Create embeddings
        embeddings = create_embeddings(texts, model)
        
        # Create or retrieve collection
        collection = create_collection(
            CHROMA_DB_PATH,
            COLLECTION_NAME,
            rebuild=REBUILD_COLLECTION
        )
        
        # Insert batches
        inserted_count = insert_batches(
            collection,
            ids,
            embeddings,
            texts,
            metadatas,
            batch_size=DEFAULT_BATCH_SIZE
        )
        
        elapsed_time = time.time() - start_time
        
        # Print validation
        print_validation(collection, embedding_dim, elapsed_time, inserted_count)
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
