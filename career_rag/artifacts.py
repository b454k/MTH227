"""Validation helpers for local Career RAG runtime artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from career_rag.config import (
    ANTHROPIC_EVIDENCE_PATH,
    CHROMA_AI_IMPACT_DIR,
    CHROMA_ONET_DIR,
    CHROMA_RESEARCH_DIR,
    ONET_DOCUMENTS_PATH,
    ONET_DUCKDB_PATH,
    ONET_SECTION_DOCUMENTS_PATH,
    ONET_SUPPLEMENTAL_DOCUMENTS_PATH,
    PROFILE_RESULT_PATH,
    PROJECT_ROOT,
    RESEARCH_DOCUMENTS_PATH,
)


SETUP_RESTORE_COMMAND = r"python scripts\archives\restore_data_archives.py"
SETUP_BUILD_COMMAND = r"python scripts\build_all_artifacts.py"
SETUP_CHECK_COMMAND = r"python scripts\check_artifacts.py"

ONET_CHROMA_COLLECTIONS = {
    "onet_sections": CHROMA_ONET_DIR,
    "onet_full_occupations": CHROMA_ONET_DIR,
    "onet_supplemental": CHROMA_ONET_DIR,
}
RESEARCH_CHROMA_COLLECTIONS = {
    "research_ai_impact_claims": CHROMA_RESEARCH_DIR,
    "research_inference": CHROMA_RESEARCH_DIR,
    "ai_impact_evidence": CHROMA_AI_IMPACT_DIR,
}
REQUIRED_CHROMA_COLLECTIONS = {
    **ONET_CHROMA_COLLECTIONS,
    **RESEARCH_CHROMA_COLLECTIONS,
}

REQUIRED_PATH_ARTIFACTS = [
    {
        "id": "onet_duckdb",
        "label": "O*NET DuckDB database",
        "path": ONET_DUCKDB_PATH,
        "kind": "file",
    },
    {
        "id": "onet_full_documents",
        "label": "O*NET full occupation documents",
        "path": ONET_DOCUMENTS_PATH,
        "kind": "file",
    },
    {
        "id": "onet_section_documents",
        "label": "O*NET section documents",
        "path": ONET_SECTION_DOCUMENTS_PATH,
        "kind": "file",
    },
    {
        "id": "onet_supplemental_documents",
        "label": "O*NET supplemental documents",
        "path": ONET_SUPPLEMENTAL_DOCUMENTS_PATH,
        "kind": "file",
    },
    {
        "id": "chroma_onet_dir",
        "label": "O*NET Chroma store",
        "path": CHROMA_ONET_DIR,
        "kind": "dir",
    },
    {
        "id": "research_documents",
        "label": "AI-impact research claim documents",
        "path": RESEARCH_DOCUMENTS_PATH,
        "kind": "file",
    },
    {
        "id": "anthropic_evidence",
        "label": "Structured Anthropic/AI-impact evidence",
        "path": ANTHROPIC_EVIDENCE_PATH,
        "kind": "file",
    },
    {
        "id": "chroma_research_dir",
        "label": "Research Chroma store",
        "path": CHROMA_RESEARCH_DIR,
        "kind": "dir",
    },
    {
        "id": "chroma_ai_impact_dir",
        "label": "Structured AI-impact Chroma store",
        "path": CHROMA_AI_IMPACT_DIR,
        "kind": "dir",
    },
]


class MissingRagArtifactsError(RuntimeError):
    """Raised when local RAG artifacts are missing or unusable."""

    def __init__(self, missing_artifacts: list[dict[str, Any]]) -> None:
        self.missing_artifacts = missing_artifacts
        super().__init__(format_missing_artifacts_error(missing_artifacts))


def project_relative(path: str | Path) -> str:
    """Return a project-relative display path when possible."""
    resolved = Path(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _path_exists(path: Path, kind: str) -> bool:
    if kind == "dir":
        return path.is_dir()
    return path.is_file()


def path_artifact_statuses() -> list[dict[str, Any]]:
    """Return existence status for required file/folder artifacts."""
    statuses: list[dict[str, Any]] = []
    for artifact in REQUIRED_PATH_ARTIFACTS:
        path = Path(artifact["path"])
        statuses.append(
            {
                **artifact,
                "path": str(path),
                "display_path": project_relative(path),
                "exists": _path_exists(path, str(artifact["kind"])),
                "count": None,
                "error": "",
            }
        )
    return statuses


def inspect_chroma_collections(
    collections: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    """Return existence and count status for required Chroma collections."""
    collections = collections or REQUIRED_CHROMA_COLLECTIONS
    statuses: list[dict[str, Any]] = []
    clients_by_dir: dict[str, Any] = {}
    import_error = ""
    try:
        import chromadb
    except ImportError as exc:
        chromadb = None
        import_error = f"chromadb is not installed: {exc}"

    for collection_name, persist_dir in collections.items():
        persist_path = Path(persist_dir)
        status = {
            "id": f"chroma_collection_{collection_name}",
            "label": f"Chroma collection: {collection_name}",
            "path": str(persist_path),
            "display_path": project_relative(persist_path),
            "kind": "chroma_collection",
            "collection": collection_name,
            "exists": False,
            "count": None,
            "error": "",
        }
        if import_error:
            status["error"] = import_error
            statuses.append(status)
            continue
        if not persist_path.is_dir():
            status["error"] = f"Persist directory not found: {project_relative(persist_path)}"
            statuses.append(status)
            continue
        try:
            key = str(persist_path)
            client = clients_by_dir.get(key)
            if client is None:
                client = chromadb.PersistentClient(path=key)
                clients_by_dir[key] = client
            collection = client.get_collection(name=collection_name)
            count = int(collection.count())
            status["exists"] = True
            status["count"] = count
            if count <= 0:
                status["error"] = "Collection exists but contains zero documents."
        except Exception as exc:
            status["error"] = str(exc)
        statuses.append(status)
    return statuses


def inspect_rag_artifacts(check_chroma: bool = True) -> dict[str, Any]:
    """Inspect required local artifacts and return a UI/script-friendly payload."""
    path_statuses = path_artifact_statuses()
    chroma_statuses = inspect_chroma_collections() if check_chroma else []
    all_statuses = [*path_statuses, *chroma_statuses]
    missing = [
        status
        for status in all_statuses
        if not status.get("exists") or (status.get("count") is not None and int(status["count"]) <= 0)
    ]
    return {
        "project_root": str(PROJECT_ROOT),
        "profile_result_path": str(PROFILE_RESULT_PATH),
        "profile_result_display_path": project_relative(PROFILE_RESULT_PATH),
        "statuses": all_statuses,
        "path_statuses": path_statuses,
        "chroma_statuses": chroma_statuses,
        "missing": missing,
        "rag_enabled": not missing,
        "restore_command": SETUP_RESTORE_COMMAND,
        "build_command": SETUP_BUILD_COMMAND,
        "check_command": SETUP_CHECK_COMMAND,
    }


def format_missing_artifacts_error(missing_artifacts: list[dict[str, Any]]) -> str:
    """Build an actionable missing-artifacts message."""
    lines = [
        "RAG artifacts missing. Please run the setup/build command.",
        "",
        "Missing or unusable artifacts:",
    ]
    for artifact in missing_artifacts:
        detail = artifact.get("display_path") or artifact.get("path")
        count = artifact.get("count")
        error = artifact.get("error")
        suffix = ""
        if count is not None:
            suffix += f" (count={count})"
        if error:
            suffix += f" - {error}"
        lines.append(f"- {artifact.get('label')}: {detail}{suffix}")

    lines.extend(
        [
            "",
            "Use prebuilt artifacts when available:",
            f"  {SETUP_RESTORE_COMMAND}",
            "",
            "Or rebuild artifacts from raw/local sources:",
            f"  {SETUP_BUILD_COMMAND}",
            "",
            "Then verify:",
            f"  {SETUP_CHECK_COMMAND}",
        ]
    )
    return "\n".join(lines)


def ensure_required_artifacts(check_chroma: bool = True) -> dict[str, Any]:
    """Raise if any required RAG artifact is missing or empty."""
    inspection = inspect_rag_artifacts(check_chroma=check_chroma)
    missing = inspection["missing"]
    if missing:
        raise MissingRagArtifactsError(missing)
    return inspection
