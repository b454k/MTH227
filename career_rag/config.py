"""Shared configuration constants for the Career RAG project."""

from __future__ import annotations

import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _path_from_env(name: str, default: Path) -> Path:
    """Return an absolute path from an env var or a project-relative default."""
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


DATA_DIR = _path_from_env("CAREER_RAG_DATA_DIR", PROJECT_ROOT / "data")
ONET_INTEREST_PROFILER_DIR = _path_from_env(
    "CAREER_RAG_IP_DIR",
    PROJECT_ROOT / "onet_interest_profiler",
)

DOCUMENTS_DIR = _path_from_env("CAREER_RAG_DOCUMENTS_DIR", DATA_DIR / "documents")
RESEARCH_DIR = _path_from_env("CAREER_RAG_RESEARCH_DIR", DATA_DIR / "research")
PROCESSED_DIR = _path_from_env("CAREER_RAG_PROCESSED_DIR", DATA_DIR / "processed")

ONET_DUCKDB_PATH = _path_from_env(
    "ONET_DUCKDB_PATH",
    DATA_DIR / "duckdb" / "onet.duckdb",
)
ONET_DOCUMENTS_PATH = _path_from_env(
    "ONET_DOCUMENTS_PATH",
    DOCUMENTS_DIR / "onet_occupation_documents.jsonl",
)
ONET_SECTION_DOCUMENTS_PATH = _path_from_env(
    "ONET_SECTION_DOCUMENTS_PATH",
    DOCUMENTS_DIR / "onet_occupation_section_documents.jsonl",
)
ONET_SUPPLEMENTAL_DOCUMENTS_PATH = _path_from_env(
    "ONET_SUPPLEMENTAL_DOCUMENTS_PATH",
    DOCUMENTS_DIR / "onet_supplemental_documents.jsonl",
)
RESEARCH_DOCUMENTS_PATH = _path_from_env(
    "RESEARCH_DOCUMENTS_PATH",
    RESEARCH_DIR / "ai_impact_claims.jsonl",
)
CHROMA_ONET_DIR = _path_from_env("CHROMA_ONET_DIR", DATA_DIR / "chroma_onet")
CHROMA_RESEARCH_DIR = _path_from_env("CHROMA_RESEARCH_DIR", PROJECT_ROOT / "chroma_research")
CHROMA_AI_IMPACT_DIR = _path_from_env("CHROMA_AI_IMPACT_DIR", PROJECT_ROOT / "chroma_ai_impact")
ANTHROPIC_EVIDENCE_PATH = _path_from_env(
    "ANTHROPIC_EVIDENCE_PATH",
    PROCESSED_DIR / "ai_impact_evidence_deduped.jsonl",
)
ANTHROPIC_RAW_EVIDENCE_PATH = _path_from_env(
    "ANTHROPIC_RAW_EVIDENCE_PATH",
    PROCESSED_DIR / "anthropic_ai_impact.jsonl",
)
PROFILE_RESULT_PATH = _path_from_env(
    "PROFILE_RESULT_PATH",
    ONET_INTEREST_PROFILER_DIR / "ip_profile_result.json",
)
QUESTIONS_PATH = _path_from_env(
    "QUESTIONS_PATH",
    ONET_INTEREST_PROFILER_DIR / "interest_profiler_questions.json",
)
CAREER_LISTINGS_PATH = _path_from_env(
    "CAREER_LISTINGS_PATH",
    ONET_INTEREST_PROFILER_DIR / "ip_career_listings.json",
)
FINAL_REPORT_JSON_PATH = _path_from_env(
    "FINAL_REPORT_JSON_PATH",
    ONET_INTEREST_PROFILER_DIR / "ip_final_career_report.json",
)
FINAL_REPORT_MD_PATH = _path_from_env(
    "FINAL_REPORT_MD_PATH",
    ONET_INTEREST_PROFILER_DIR / "ip_final_career_report.md",
)

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
