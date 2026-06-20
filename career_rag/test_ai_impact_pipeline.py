#!/usr/bin/env python3
"""Run AI-impact pipeline smoke tests and write a consolidated log."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from career_rag.config import (
        EMBEDDING_MODEL_NAME,
        embedding_model_mismatch_message,
    )
except ImportError:  # Allows: py career_rag/test_ai_impact_pipeline.py
    from config import (  # type: ignore
        EMBEDDING_MODEL_NAME,
        embedding_model_mismatch_message,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "test_results_ai_impact.txt"


def load_env_key_present() -> bool:
    """Return True when OPENAI_API_KEY appears in env or .env."""
    if os.getenv("OPENAI_API_KEY"):
        return True
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            return bool(line.split("=", 1)[1].strip())
    return False


def run_command(command: list[str], log: list[str], allow_failure: bool = False) -> int:
    """Run one command and append output to the log."""
    log.append("\n" + "=" * 100)
    log.append("COMMAND: " + " ".join(command))
    log.append("=" * 100)
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - start
    log.append(completed.stdout)
    log.append(f"EXIT CODE: {completed.returncode}")
    log.append(f"ELAPSED: {elapsed:.2f}s")
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed: {' '.join(command)}")
    return completed.returncode


def verify_chroma_embedding_metadata(log: list[str]) -> None:
    """Assert every project Chroma collection uses the configured embedding model."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required for metadata verification.") from exc

    checks = [
        (PROJECT_ROOT / "chroma_research", "research_ai_impact_claims", False),
        (PROJECT_ROOT / "chroma_research", "research_inference", True),
        (PROJECT_ROOT / "chroma_ai_impact", "ai_impact_evidence", True),
        (PROJECT_ROOT / "data" / "chroma_onet", "onet_sections", False),
        (PROJECT_ROOT / "data" / "chroma_onet", "onet_full_occupations", False),
        (PROJECT_ROOT / "data" / "chroma_onet", "onet_supplemental", False),
    ]
    log.append("\n" + "=" * 100)
    log.append("CHROMA EMBEDDING MODEL METADATA CHECK")
    log.append("=" * 100)
    for persist_dir, collection_name, required in checks:
        if not persist_dir.exists():
            message = f"SKIP {collection_name}: persist dir not found: {persist_dir}"
            if required:
                raise FileNotFoundError(message)
            log.append(message)
            continue
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            collection = client.get_collection(collection_name)
        except Exception as exc:
            message = f"SKIP {collection_name}: collection not found in {persist_dir}"
            if required:
                raise RuntimeError(message) from exc
            log.append(message)
            continue
        metadata = collection.metadata or {}
        actual_model = str(metadata.get("embedding_model") or "").strip()
        count = collection.count()
        log.append(
            f"{collection_name}: count={count}; embedding_model={actual_model or '(missing)'}"
        )
        if actual_model != EMBEDDING_MODEL_NAME:
            raise RuntimeError(
                embedding_model_mismatch_message(
                    collection_name,
                    actual_model,
                    EMBEDDING_MODEL_NAME,
                )
            )


def main() -> int:
    """Run the smoke pipeline."""
    py = sys.executable
    log: list[str] = [
        "AI Impact Pipeline Smoke Test",
        f"Python: {py}",
        f"Project root: {PROJECT_ROOT}",
        f"OPENAI_API_KEY present: {load_env_key_present()}",
    ]

    commands = [
        [py, "-m", "career_rag.download_research_sources", "--source-file", "source_url.txt"],
        [py, "-m", "career_rag.extract_research_chunks"],
    ]
    for command in commands:
        run_command(command, log)

    nber_with_llm = [
        py,
        "-m",
        "career_rag.extract_nber_w31222_ai_exposure",
        "--use-llm",
    ]
    nber_without_llm = [py, "-m", "career_rag.extract_nber_w31222_ai_exposure"]
    if load_env_key_present():
        code = run_command(nber_with_llm, log, allow_failure=True)
        if code != 0:
            log.append("LLM extraction failed or was unavailable; retrying NBER extraction without --use-llm.")
            run_command(nber_without_llm, log)
    else:
        log.append("LLM extraction skipped because OPENAI_API_KEY is unavailable.")
        run_command(nber_without_llm, log)

    remaining_commands = [
        [py, "-m", "career_rag.build_anthropic_economic_index", "--download-if-missing"],
        [py, "-m", "career_rag.merge_ai_impact_evidence"],
        [py, "-m", "career_rag.embed_ai_impact_evidence"],
        [
            py,
            "-m",
            "career_rag.ai_impact_retriever",
            "data scientist mathematical modeling ai exposure",
            "--soc-code",
            "15-2051.00",
            "--top-k",
            "8",
            "--debug",
        ],
        [
            py,
            "-m",
            "career_rag.generator",
            "What does data scientists do, how is the ai exposure?",
            "--show-sources",
        ],
    ]
    for command in remaining_commands:
        run_command(command, log)

    verify_chroma_embedding_metadata(log)

    RESULTS_PATH.write_text("\n".join(log), encoding="utf-8")
    print(f"Smoke test log written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
