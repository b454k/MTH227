"""AI-impact evidence helpers for Interest Profiler final reports."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from career_rag.config import (
    ANTHROPIC_EVIDENCE_PATH,
    ANTHROPIC_RAW_EVIDENCE_PATH,
    ONET_DUCKDB_PATH,
)


AI_EVIDENCE_PATHS = [
    ANTHROPIC_EVIDENCE_PATH,
    ANTHROPIC_RAW_EVIDENCE_PATH,
]
DB_PATH = ONET_DUCKDB_PATH

AI_TASK_DOC_TYPES = {"ai_task_impact", "ai_task_penetration"}
AI_TASK_TABLE_LIMIT = 6
NO_LOCAL_ANTHROPIC_EVIDENCE = (
    "No local Anthropic evidence found for this exact O*NET task"
)

ANTHROPIC_ECONOMIC_INDEX_URL = "https://www.anthropic.com/economic-index"
ANTHROPIC_LABOR_MARKET_URL = "https://www.anthropic.com/research/labor-market-impacts"
NBER_RAPID_ADOPTION_URL = "https://www.nber.org/papers/w32966"
NBER_RAPID_ADOPTION_PDF_URL = "https://www.nber.org/system/files/working_papers/w32966/w32966.pdf"


def build_ai_impact_for_occupation(
    display_title: str,
    resolved_onet_title: str,
    onet_soc_code: str,
    onet_tasks: list[Any] | None = None,
    citation_manager: Any | None = None,
) -> dict[str, Any]:
    """Build task-level AI-impact rows from exact local O*NET tasks."""
    anthropic_id, nber_id = _register_sources(citation_manager)
    rows = _local_ai_rows_for_soc(onet_soc_code)
    task_records = _task_records_for_occupation(onet_soc_code, onet_tasks)
    task_rows_by_id = _task_rows_by_task_id(rows)

    task_breakdown = []
    used_retrieved = False
    validation_warnings: list[str] = []
    for task_record in task_records:
        row, validation_warning = _best_evidence_for_task(
            task_record=task_record,
            rows=task_rows_by_id.get(str(task_record.get("task_id") or ""), []),
        )
        if validation_warning:
            validation_warnings.append(validation_warning)
        if row:
            used_retrieved = True
            task_breakdown.append(_format_retrieved_task_row(task_record, row, anthropic_id))
        else:
            task_breakdown.append(
                _format_unobserved_task_row(task_record, validation_warning)
            )

    occupation_rows = [
        row for row in rows if str(row.get("doc_type") or "") == "ai_occupation_impact"
    ]
    future_outlook = _future_outlook_summary(
        display_title=display_title,
        resolved_onet_title=resolved_onet_title,
        occupation_rows=occupation_rows,
        used_retrieved=used_retrieved,
        anthropic_id=anthropic_id,
        nber_id=nber_id,
    )

    retrieved_rows = [_compact_retrieved_row(row) for row in rows[:12]]
    return {
        "evidence_status": (
            "exact_onet_task_evidence"
            if used_retrieved
            else "no_exact_task_anthropic_evidence"
        ),
        "task_breakdown": task_breakdown,
        "future_outlook_summary": future_outlook,
        "retrieved_evidence": retrieved_rows,
        "validation_warnings": validation_warnings,
    }


def _register_sources(citation_manager: Any | None) -> tuple[int | None, int | None]:
    if citation_manager is None:
        return None, None

    anthropic_id = citation_manager.add_source(
        title="Anthropic Economic Index and labor-market impacts",
        source_type="anthropic",
        url=ANTHROPIC_LABOR_MARKET_URL,
        local_file="data/processed/ai_impact_evidence_deduped.jsonl",
        retrieved_section="observed occupation/task exposure and task penetration rows",
        note="Task rows are used only when local rows match the resolved O*NET-SOC code and exact O*NET task ID.",
    )
    nber_id = citation_manager.add_source(
        title="NBER Working Paper w32966 - The Rapid Adoption of Generative AI",
        source_type="nber",
        url=NBER_RAPID_ADOPTION_URL,
        local_file="data/research/pdfs/forklightning_p_the_rapid_adoption_of_generative_728a3da9.pdf",
        retrieved_section="broad GenAI adoption context",
        note="Used as broad labor-market/adoption context, not as task-level automation scoring.",
    )
    return anthropic_id, nber_id


def _marker(source_id: int | None) -> str:
    return f"[{source_id}]" if source_id else ""


@lru_cache(maxsize=1)
def _load_local_ai_rows() -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for path in AI_EVIDENCE_PATHS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = str(row.get("doc_id") or row.get("id") or "")
                if not doc_id:
                    doc_id = f"{row.get('source_file')}::{row.get('soc_code')}::{row.get('task_text')}::{row.get('metric_name')}"
                rows_by_id.setdefault(doc_id, row)
    return list(rows_by_id.values())


def _local_ai_rows_for_soc(onet_soc_code: str) -> list[dict[str, Any]]:
    code = str(onet_soc_code or "").strip()
    if not code:
        return []
    rows = [row for row in _load_local_ai_rows() if str(row.get("soc_code") or "").strip() == code]
    rows.sort(key=_row_rank, reverse=True)
    return rows


def _row_rank(row: dict[str, Any]) -> tuple[float, float]:
    metric = _as_float(row.get("metric_value"))
    doc_type = str(row.get("doc_type") or "")
    type_bonus = 1.0 if doc_type == "ai_occupation_impact" else 0.0
    if metric is None:
        metric = 0.0
    return (type_bonus, metric)


def _task_records_for_occupation(
    onet_soc_code: str,
    onet_tasks: list[Any] | None,
) -> list[dict[str, Any]]:
    records = _fetch_onet_task_records(onet_soc_code, AI_TASK_TABLE_LIMIT)
    if records:
        return records
    return _coerce_task_records(onet_soc_code, onet_tasks, AI_TASK_TABLE_LIMIT)


def _fetch_onet_task_records(onet_soc_code: str, limit: int) -> list[dict[str, Any]]:
    code = str(onet_soc_code or "").strip()
    if not code or not DB_PATH.exists():
        return []
    try:
        import duckdb
    except ImportError:
        return []

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT task_id, task, task_type
            FROM task_statements
            WHERE onetsoc_code = ?
            ORDER BY task_id
            LIMIT ?
            """,
            [code, limit],
        ).fetchall()
    finally:
        conn.close()

    records = []
    for task_id, task, task_type in rows:
        task_text = _clean_text(task)
        if not task_text:
            continue
        records.append(
            {
                "onet_soc_code": code,
                "task_id": _task_id_string(task_id),
                "task": task_text,
                "task_type": _clean_text(task_type) or None,
            }
        )
    return records


def _coerce_task_records(
    onet_soc_code: str,
    onet_tasks: list[Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    records = []
    for item in list(onet_tasks or [])[:limit]:
        if isinstance(item, dict):
            task_text = _clean_text(
                item.get("task")
                or item.get("task_text")
                or item.get("displayed_task")
                or item.get("matched_onet_task")
            )
            task_id = _task_id_string(item.get("task_id") or item.get("onet_task_id"))
            task_type = _clean_text(item.get("task_type")) or None
        else:
            task_text = _clean_text(item)
            task_id = ""
            task_type = None
        if not task_text:
            continue
        records.append(
            {
                "onet_soc_code": str(onet_soc_code or "").strip(),
                "task_id": task_id,
                "task": task_text,
                "task_type": task_type,
            }
        )
    return records


def _task_rows_by_task_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("doc_type") or "") not in AI_TASK_DOC_TYPES:
            continue
        task_id = _task_id_string(row.get("onet_task_id"))
        if not task_id:
            continue
        grouped.setdefault(task_id, []).append(row)
    return grouped


def _best_evidence_for_task(
    task_record: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    displayed_task = _clean_text(task_record.get("task"))
    task_id = _task_id_string(task_record.get("task_id"))
    valid_rows = []
    invalid_rows = []
    for row in rows:
        source_task = _clean_text(row.get("task_text"))
        if source_task and not _same_task_statement(displayed_task, source_task):
            invalid_rows.append(source_task)
            continue
        valid_rows.append(row)

    if valid_rows:
        valid_rows.sort(key=_evidence_rank, reverse=True)
        return valid_rows[0], None

    if invalid_rows:
        warning = (
            f"Validation warning: skipped Anthropic evidence for O*NET task ID {task_id} "
            "because its task statement did not match the displayed O*NET task."
        )
        return None, warning
    return None, None


def _evidence_rank(row: dict[str, Any]) -> tuple[int, float]:
    metric_name = _normalize(row.get("metric_name"))
    impact_type = _normalize(row.get("impact_type"))
    doc_type = str(row.get("doc_type") or "")
    score = _as_float(row.get("metric_value"))
    if metric_name == "directive" or impact_type == "automation":
        metric_priority = 50
    elif doc_type == "ai_task_penetration" or metric_name == "penetration":
        metric_priority = 40
    elif metric_name == "pct":
        metric_priority = 30
    elif score is not None:
        metric_priority = 20
    else:
        metric_priority = 0
    return (metric_priority, score if score is not None else -1.0)


def _format_retrieved_task_row(
    task_record: dict[str, Any],
    row: dict[str, Any],
    anthropic_id: int | None,
) -> dict[str, Any]:
    task_text = _clean_text(task_record.get("task"))
    task_id = _task_id_string(task_record.get("task_id"))
    score = _as_float(row.get("metric_value"))
    metric_name = _clean_text(row.get("metric_name")) or "metric"
    score_display = _score_display(score)
    source_marker = _marker(anthropic_id)
    if score is None:
        score_type = "retrieved_anthropic_metric_missing"
        evidence_status = "exact_task_id_match_metric_missing"
        evidence = (
            f"Local Anthropic evidence matched exact O*NET task ID {task_id}, "
            f"but {metric_name} did not contain a numeric score. {source_marker}"
        ).strip()
    else:
        score_type = f"retrieved_anthropic_{metric_name}"
        evidence_status = "exact_task_id_match"
        evidence = (
            f"Local Anthropic evidence matched exact O*NET task ID {task_id} "
            f"and reports {metric_name} = {score_display}. {source_marker}"
        ).strip()

    return {
        "task": task_text,
        "displayed_task": task_text,
        "onet_task_id": task_id,
        "matched_onet_task": task_text,
        "matched_anthropic_task": task_text,
        "automation_level": _automation_level_from_score(score),
        "score": round(float(score), 4) if score is not None else None,
        "score_display": score_display,
        "score_type": score_type,
        "evidence_status": evidence_status,
        "evidence": evidence,
        "source_ids": [anthropic_id] if anthropic_id else [],
    }


def _format_unobserved_task_row(
    task_record: dict[str, Any],
    validation_warning: str | None = None,
) -> dict[str, Any]:
    task_text = _clean_text(task_record.get("task"))
    evidence = NO_LOCAL_ANTHROPIC_EVIDENCE
    if validation_warning:
        evidence = f"{evidence}. {validation_warning}"
    return {
        "task": task_text,
        "displayed_task": task_text,
        "onet_task_id": _task_id_string(task_record.get("task_id")),
        "matched_onet_task": task_text,
        "matched_anthropic_task": None,
        "automation_level": "Not observed",
        "score": None,
        "score_display": "N/A",
        "score_type": "not_observed",
        "evidence_status": "no_exact_task_evidence",
        "evidence": evidence,
        "source_ids": [],
        "validation_warning": validation_warning,
    }


def _future_outlook_summary(
    display_title: str,
    resolved_onet_title: str,
    occupation_rows: list[dict[str, Any]],
    used_retrieved: bool,
    anthropic_id: int | None,
    nber_id: int | None,
) -> str:
    resolved_note = ""
    if display_title != resolved_onet_title:
        resolved_note = f" O*NET/AI evidence is retrieved from {resolved_onet_title}."

    occupation_metric = _best_occupation_metric(occupation_rows)
    if occupation_metric is not None:
        evidence_sentence = (
            f"The strongest local Anthropic occupation signal reports observed exposure of "
            f"{occupation_metric:.4g} for the resolved O*NET occupation {_marker(anthropic_id)}."
        )
    elif used_retrieved:
        evidence_sentence = (
            f"Local Anthropic task rows were found by exact O*NET task ID for the resolved occupation {_marker(anthropic_id)}."
        )
    else:
        evidence_sentence = (
            "No local Anthropic task metric was found for the displayed O*NET task rows."
        )

    role_sentence = (
        "My read is that AI is more likely to become a helper around repeatable, text-heavy, or digital parts of the work than a clean replacement for the whole occupation. The human side still matters most where the job depends on judgment, context, quality checking, accountability, and knowing what a good result should look like."
    )

    return f"{evidence_sentence}{resolved_note} {role_sentence} Broader GenAI adoption evidence is used as labor-market context, not as task-level scoring {_marker(nber_id)}.".strip()


def _best_occupation_metric(rows: list[dict[str, Any]]) -> float | None:
    values = [_as_float(row.get("metric_value")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def _compact_retrieved_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": row.get("doc_id"),
        "source_id": row.get("source_id"),
        "source_name": row.get("source_name"),
        "source_url": row.get("source_url"),
        "source_file": row.get("source_file"),
        "doc_type": row.get("doc_type"),
        "soc_code": row.get("soc_code"),
        "occupation_title": row.get("occupation_title"),
        "onet_task_id": row.get("onet_task_id"),
        "task_text": row.get("task_text"),
        "impact_type": row.get("impact_type"),
        "metric_name": row.get("metric_name"),
        "metric_value": row.get("metric_value"),
        "evidence_text": _shorten(row.get("evidence_text"), 240),
    }


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _task_id_string(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _same_task_statement(left: Any, right: Any) -> bool:
    return _normalize(left) == _normalize(right)


def _automation_level_from_score(score: float | None) -> str:
    if score is None:
        return "Not observed"
    if score >= 0.67:
        return "High"
    if score >= 0.34:
        return "Medium"
    return "Low"


def _score_display(score: float | None) -> str:
    if score is None:
        return "N/A"
    return f"{score:.4g}"


def _normalize(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shorten(value: Any, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
