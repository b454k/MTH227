"""Shared configuration constants for the Career RAG project."""

from __future__ import annotations

import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_EMBEDDING_DIMENSION = 384
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def clean_config_text(value: Any) -> str:
    """Return a compact string for configuration and metadata checks."""
    if value is None:
        return ""
    return str(value).strip()


def embedding_model_mismatch_message(
    collection_name: str,
    actual_model: Any,
    expected_model: str = EMBEDDING_MODEL_NAME,
) -> str:
    """Build the standard embedding-model mismatch error message."""
    actual = clean_config_text(actual_model) or "(missing)"
    return (
        "Embedding model mismatch: "
        f"collection '{collection_name}' was built with {actual} but code expects "
        f"{expected_model}. Rebuild this collection."
    )


def require_configured_embedding_model(model_name: Any) -> str:
    """Return the configured model name or raise for unsupported overrides."""
    cleaned = clean_config_text(model_name)
    if cleaned != EMBEDDING_MODEL_NAME:
        raise ValueError(
            f"Unsupported embedding model '{cleaned or '(missing)'}'. "
            f"This project uses only {EMBEDDING_MODEL_NAME}."
        )
    return EMBEDDING_MODEL_NAME


def require_hf_token() -> str:
    """Load .env and require HF_TOKEN before Hugging Face model access."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=ENV_PATH)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Add it to your .env file.")
    return hf_token


@contextmanager
def quiet_huggingface_model_load() -> Any:
    """Suppress Hugging Face/SentenceTransformer model-load progress output."""
    env_updates = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TQDM_DISABLE": "1",
    }
    previous_values = {key: os.environ.get(key) for key in env_updates}
    os.environ.update(env_updates)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
        finally:
            for key, previous_value in previous_values.items():
                if previous_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous_value


def validate_collection_embedding_model(
    collection: Any,
    expected_model: str = EMBEDDING_MODEL_NAME,
) -> None:
    """Raise if a Chroma collection was not built with the configured model."""
    collection_name = clean_config_text(getattr(collection, "name", "")) or "unknown"
    metadata = getattr(collection, "metadata", None) or {}
    actual_model = clean_config_text(metadata.get("embedding_model"))
    if actual_model != expected_model:
        raise RuntimeError(
            embedding_model_mismatch_message(
                collection_name=collection_name,
                actual_model=actual_model,
                expected_model=expected_model,
            )
        )
