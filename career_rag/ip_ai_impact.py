"""AI-impact evidence helpers for Interest Profiler final reports."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from career_rag.config import PROJECT_ROOT


AI_EVIDENCE_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "ai_impact_evidence_deduped.jsonl",
    PROJECT_ROOT / "data" / "processed" / "anthropic_ai_impact.jsonl",
]

ANTHROPIC_ECONOMIC_INDEX_URL = "https://www.anthropic.com/economic-index"
ANTHROPIC_LABOR_MARKET_URL = "https://www.anthropic.com/research/labor-market-impacts"
NBER_RAPID_ADOPTION_URL = "https://www.nber.org/papers/w32966"
NBER_RAPID_ADOPTION_PDF_URL = "https://www.nber.org/system/files/working_papers/w32966/w32966.pdf"


BASE_TASK_CONCEPTS = [
    {
        "task": "Cleaning and preparing data",
        "keywords": ["clean", "prepare", "manipulate", "raw data", "data"],
        "automation_level": "High",
        "score": 0.85,
        "evidence": "Routine data preparation and format conversion are usually strong candidates for AI assistance when data access and validation rules are clear.",
    },
    {
        "task": "Writing SQL queries and standard analysis code",
        "keywords": ["sql", "query", "statistical software", "software", "code", "program"],
        "automation_level": "High",
        "score": 0.80,
        "evidence": "Routine query writing and common analysis-code patterns are highly assistive use cases, while validation remains human-owned.",
    },
    {
        "task": "Creating dashboards and routine reports",
        "keywords": ["dashboard", "report", "reports", "visualization", "disseminate", "information"],
        "automation_level": "Medium",
        "score": 0.60,
        "evidence": "AI can help draft reports and generate dashboard components, but metric choice and interpretation still need business context.",
    },
    {
        "task": "Interpreting business insights and recommending actions",
        "keywords": ["interpret", "recommend", "stakeholder", "business", "strategy", "decision"],
        "automation_level": "Low",
        "score": 0.35,
        "evidence": "The durable part is deciding what the evidence means for a specific organization and explaining tradeoffs to people.",
    },
]

ROLE_TASK_CONCEPTS = {
    "actuar": [
        {
            "task": "Preparing actuarial models and assumptions",
            "keywords": ["actuar", "model", "mortality", "risk", "statistical", "premium"],
            "automation_level": "Medium",
            "score": 0.58,
            "evidence": "AI can assist with model setup, documentation, and sensitivity checks, but assumptions and professional judgment remain central.",
        },
        {
            "task": "Analyzing insurance and financial risk",
            "keywords": ["risk", "insurance", "financial", "statistical", "liabilities"],
            "automation_level": "Medium",
            "score": 0.55,
            "evidence": "Quantitative analysis can be accelerated, but risk interpretation and accountability are not just mechanical calculations.",
        },
        {
            "task": "Explaining risk findings to decision makers",
            "keywords": ["explain", "communicate", "customer", "management", "recommend"],
            "automation_level": "Low",
            "score": 0.30,
            "evidence": "Communication, judgment, and risk ownership are less automatable than calculation support.",
        },
    ],
    "machine learning": [
        {
            "task": "Model training and evaluation",
            "keywords": ["model", "algorithm", "machine learning", "evaluate", "predict"],
            "automation_level": "Medium",
            "score": 0.65,
            "evidence": "AI tools can speed up prototyping, experiment code, and evaluation scaffolding, but robust model selection requires human judgment.",
        },
        {
            "task": "Writing and debugging ML code",
            "keywords": ["software", "code", "program", "computer", "develop", "solution"],
            "automation_level": "High",
            "score": 0.78,
            "evidence": "Routine coding and debugging are strong AI-assistance areas when requirements are clear.",
        },
        {
            "task": "Choosing the problem, data, and risk controls",
            "keywords": ["problem", "research", "innovation", "ethical", "risk", "requirements"],
            "automation_level": "Low",
            "score": 0.32,
            "evidence": "Problem framing, data suitability, ethical judgment, and deployment risk are context-heavy decisions.",
        },
    ],
    "project": [
        {
            "task": "Tracking project milestones and deliverables",
            "keywords": ["milestone", "deliverable", "track", "monitor", "schedule"],
            "automation_level": "Medium",
            "score": 0.62,
            "evidence": "AI can summarize status, draft updates, and flag routine schedule issues, but tradeoffs and prioritization need human ownership.",
        },
        {
            "task": "Coordinating technical teams and stakeholders",
            "keywords": ["coordinate", "communicate", "customer", "personnel", "project"],
            "automation_level": "Low",
            "score": 0.34,
            "evidence": "Negotiation, trust, priorities, and organizational context are less automatable than documentation.",
        },
    ],
    "search marketing": [
        {
            "task": "Creating content and campaign variations",
            "keywords": ["content", "campaign", "marketing", "digital", "search"],
            "automation_level": "High",
            "score": 0.78,
            "evidence": "AI is well suited to drafting and varying text, metadata, and campaign assets, with human review for brand and strategy.",
        },
        {
            "task": "Analyzing traffic and conversion trends",
            "keywords": ["analytics", "traffic", "trend", "market", "performance"],
            "automation_level": "Medium",
            "score": 0.58,
            "evidence": "Routine analysis can be assisted, but strategic implications depend on business context and customer understanding.",
        },
    ],
}


def build_ai_impact_for_occupation(
    display_title: str,
    resolved_onet_title: str,
    onet_soc_code: str,
    onet_tasks: list[str] | None = None,
    citation_manager: Any | None = None,
) -> dict[str, Any]:
    """Build task-level AI-impact rows with retrieved evidence or heuristic fallback."""
    anthropic_id, nber_id = _register_sources(citation_manager)
    rows = _local_ai_rows_for_soc(onet_soc_code)
    concepts = _task_concepts_for_title(display_title, resolved_onet_title)

    task_breakdown = []
    used_retrieved = False
    used_doc_ids: set[str] = set()
    for concept in concepts[:4]:
        row = _best_row_for_concept(rows, concept, used_doc_ids)
        if row:
            used_retrieved = True
            used_doc_ids.add(str(row.get("doc_id") or ""))
            task_breakdown.append(_format_retrieved_task_row(concept, row, anthropic_id))
        else:
            task_breakdown.append(_format_heuristic_task_row(concept, nber_id))

    if len(task_breakdown) < 3:
        for concept in BASE_TASK_CONCEPTS:
            if len(task_breakdown) >= 3:
                break
            if concept["task"] in {item["task"] for item in task_breakdown}:
                continue
            task_breakdown.append(_format_heuristic_task_row(concept, nber_id))

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
            "retrieved_with_heuristic_interpretation"
            if used_retrieved
            else "heuristic_pending_source_retrieval"
        ),
        "task_breakdown": task_breakdown,
        "future_outlook_summary": future_outlook,
        "retrieved_evidence": retrieved_rows,
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
        note="Used only when local rows match the resolved O*NET-SOC code.",
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


def _task_concepts_for_title(display_title: str, resolved_title: str) -> list[dict[str, Any]]:
    text = f"{display_title} {resolved_title}".lower()
    if "actuar" in text:
        return ROLE_TASK_CONCEPTS["actuar"] + BASE_TASK_CONCEPTS[:1]
    if "machine learning" in text or "computer and information research" in text:
        return ROLE_TASK_CONCEPTS["machine learning"] + BASE_TASK_CONCEPTS[:1]
    if "project manager" in text:
        return ROLE_TASK_CONCEPTS["project"] + BASE_TASK_CONCEPTS[2:4]
    if "search marketing" in text or "marketing" in text:
        return ROLE_TASK_CONCEPTS["search marketing"] + BASE_TASK_CONCEPTS[2:4]
    return BASE_TASK_CONCEPTS


def _best_row_for_concept(
    rows: list[dict[str, Any]],
    concept: dict[str, Any],
    used_doc_ids: set[str],
) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if str(row.get("doc_type") or "") == "ai_occupation_impact":
            continue
        doc_id = str(row.get("doc_id") or "")
        if doc_id in used_doc_ids:
            continue
        task_text = str(row.get("task_text") or "")
        similarity = _concept_similarity(task_text, concept)
        if similarity <= 0:
            continue
        metric = _as_float(row.get("metric_value"))
        metric_score = metric if metric is not None else 0.0
        candidates.append((similarity, metric_score, row))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _concept_similarity(task_text: str, concept: dict[str, Any]) -> float:
    haystack = _normalize(task_text)
    if not haystack:
        return 0.0
    keyword_hits = 0
    for keyword in concept.get("keywords", []):
        if _normalize(keyword) in haystack:
            keyword_hits += 1
    if keyword_hits:
        return min(1.0, 0.35 + 0.18 * keyword_hits)

    concept_tokens = set(_normalize(concept.get("task", "")).split())
    task_tokens = set(haystack.split())
    if not concept_tokens or not task_tokens:
        return 0.0
    overlap = len(concept_tokens & task_tokens) / max(len(concept_tokens | task_tokens), 1)
    return overlap if overlap >= 0.16 else 0.0


def _format_retrieved_task_row(
    concept: dict[str, Any],
    row: dict[str, Any],
    anthropic_id: int | None,
) -> dict[str, Any]:
    score = _as_float(row.get("metric_value"))
    metric_name = str(row.get("metric_name") or "metric").strip()
    source_marker = _marker(anthropic_id)
    task_text = str(row.get("task_text") or concept["task"]).strip()
    if score is None:
        score = float(concept["score"])
        score_type = "heuristic_with_retrieved_source_context"
    else:
        score_type = f"retrieved_anthropic_{metric_name}"

    evidence = (
        f"Local Anthropic row for O*NET task \"{_shorten(task_text, 130)}\" reports "
        f"{metric_name} = {score:.4g}. The automation label is a heuristic interpretation of the task type. {source_marker}"
    ).strip()
    return {
        "task": concept["task"],
        "matched_onet_task": task_text,
        "automation_level": concept["automation_level"],
        "score": round(float(score), 4),
        "score_type": score_type,
        "evidence": evidence,
        "source_ids": [anthropic_id] if anthropic_id else [],
    }


def _format_heuristic_task_row(concept: dict[str, Any], nber_id: int | None) -> dict[str, Any]:
    marker = _marker(nber_id)
    evidence = (
        f"{concept['evidence']} No local task-level Anthropic metric was matched for this exact row; "
        f"this score is heuristic pending source retrieval. {marker}"
    ).strip()
    return {
        "task": concept["task"],
        "automation_level": concept["automation_level"],
        "score": round(float(concept["score"]), 2),
        "score_type": "heuristic_or_retrieved",
        "evidence_status": "heuristic_pending_source_retrieval",
        "evidence": evidence,
        "source_ids": [nber_id] if nber_id else [],
    }


def _future_outlook_summary(
    display_title: str,
    resolved_onet_title: str,
    occupation_rows: list[dict[str, Any]],
    used_retrieved: bool,
    anthropic_id: int | None,
    nber_id: int | None,
) -> str:
    display = display_title
    resolved_note = ""
    if display_title != resolved_onet_title:
        resolved_note = f" O*NET/AI evidence is retrieved from {resolved_onet_title}."

    occupation_metric = _best_occupation_metric(occupation_rows)
    if occupation_metric is not None:
        evidence_sentence = (
            f"Local Anthropic occupation evidence reports observed exposure of "
            f"{occupation_metric:.4g} for the resolved O*NET occupation {_marker(anthropic_id)}."
        )
    elif used_retrieved:
        evidence_sentence = (
            f"Local Anthropic task rows were found for the resolved O*NET occupation {_marker(anthropic_id)}."
        )
    else:
        evidence_sentence = (
            "No local occupation-level AI metric was found for this report row, so the task table uses heuristic labels pending source retrieval."
        )

    if _analytics_like(display, resolved_onet_title):
        role_sentence = (
            "AI is more likely to change this path than erase it: repetitive preparation, query, reporting, and prototype-code work can be accelerated, while problem framing, domain judgment, stakeholder communication, and final decisions remain more durable."
        )
    elif "actuar" in f"{display} {resolved_onet_title}".lower():
        role_sentence = (
            "AI can assist actuarial calculation, documentation, and scenario exploration, but pricing assumptions, regulatory context, risk judgment, and accountability remain human-centered."
        )
    elif "project" in f"{display} {resolved_onet_title}".lower():
        role_sentence = (
            "AI can reduce coordination overhead through summaries, status drafting, and routine tracking, but prioritization, conflict resolution, and delivery judgment remain central."
        )
    else:
        role_sentence = (
            "AI can automate or assist routine cognitive tasks, but the more durable work is deciding what problem matters, checking outputs, and applying domain context."
        )

    return f"{evidence_sentence}{resolved_note} {role_sentence} Broader GenAI adoption evidence is used only as labor-market context, not as task-level scoring {_marker(nber_id)}.".strip()


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
        "task_text": row.get("task_text"),
        "impact_type": row.get("impact_type"),
        "metric_name": row.get("metric_name"),
        "metric_value": row.get("metric_value"),
        "evidence_text": _shorten(row.get("evidence_text"), 240),
    }


def _analytics_like(display_title: str, resolved_onet_title: str) -> bool:
    text = f"{display_title} {resolved_onet_title}".lower()
    return any(
        keyword in text
        for keyword in (
            "data",
            "business intelligence",
            "statistic",
            "quantitative",
            "operations research",
            "machine learning",
            "computer and information research",
        )
    )


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
