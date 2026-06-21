"""Final career report builder for the O*NET Interest Profiler flow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from career_rag.artifacts import ensure_required_artifacts
from career_rag.config import (
    CAREER_LISTINGS_PATH,
    FINAL_REPORT_JSON_PATH,
    FINAL_REPORT_MD_PATH,
    ONET_DUCKDB_PATH,
    ONET_INTEREST_PROFILER_DIR,
    PROFILE_RESULT_PATH,
)
from career_rag.interest_profiler_local import (
    RIASEC_INTERESTS,
    make_holland_code,
    prepare_profile_for_rag,
)
from career_rag.ip_ai_impact import build_ai_impact_for_occupation
from career_rag.ip_career_matcher import load_ip_career_listings, match_careers
from career_rag.ip_future_impact import build_future_impact_summary
from career_rag.ip_followup_agent import OpenAIChatProvider
from career_rag.ip_semantic_report import build_semantic_retrieval_report
from career_rag.occupation_aliases import build_occupation_index, resolve_career_alias


DATA_DIR = ONET_INTEREST_PROFILER_DIR
DB_PATH = ONET_DUCKDB_PATH

IP_SHORT_FORM_URL = "https://www.onetcenter.org/dl_tools/ipsf/Interest_Profiler.pdf"
IP_CAREER_LISTINGS_URL = "https://www.onetcenter.org/dl_tools/ipsf/IP_Career_Listings.pdf"
ONET_IP_RESOURCE_URL = "https://www.onetcenter.org/IP.html"

FOUNDATIONAL_SOFT_SKILLS = {
    "active listening",
    "active learning",
    "reading comprehension",
    "writing",
    "speaking",
    "social perceptiveness",
    "monitoring",
    "coordination",
}

JOB_ZONE_PREPARATION_EXPLANATIONS = {
    1: "no experience required",
    2: "high school diploma required",
    3: "associate's degree or vocational training required",
    4: "bachelor's degree required",
    5: "graduate degree required",
}

FINAL_REPORT_SYSTEM_PROMPT = """You are a careful career guidance assistant.
You generate personalized career reports from retrieved evidence.
You must ground factual claims in the provided evidence.
You must not invent O*NET facts, education requirements, AI exposure statistics, or occupation titles.
If evidence is missing, say it is not available or mark the point as an inference.
Use only the submitted Interest Profiler result and actual follow-up answers.
Do not add technology, math, fast-paced, entrepreneurship, or other preferences unless they came from user answers.

Maximum top matches: 10.

For each top match, include:

* why it fits
* O*NET-grounded job details
* AI impact breakdown
* future outlook
* skills to learn
* education needed
* day-in-the-life narrative

Return valid JSON matching the report schema.
Do not include markdown inside JSON unless needed for short text fields.
Use citation IDs like [1], [2] inside strings where appropriate.
Do not include raw source URLs inside main text; sources are listed separately.
"""

class CitationManager:
    """Assign stable numeric citations to report sources."""

    def __init__(self) -> None:
        self._sources: list[dict[str, Any]] = []
        self._key_to_id: dict[tuple[str, str, str, str], int] = {}

    def add_source(
        self,
        title: str,
        source_type: str,
        url: str | None = None,
        local_file: str | None = None,
        retrieved_section: str | None = None,
        note: str | None = None,
    ) -> int:
        """Add or reuse one source and return its numeric ID."""
        key = (
            str(title or "").strip(),
            str(source_type or "").strip(),
            str(url or "").strip(),
            str(local_file or "").strip(),
        )
        existing = self._key_to_id.get(key)
        if existing is not None:
            return existing

        source_id = len(self._sources) + 1
        self._key_to_id[key] = source_id
        self._sources.append(
            {
                "id": source_id,
                "title": str(title or "").strip(),
                "source_type": str(source_type or "").strip(),
                "url": str(url or "").strip() or None,
                "local_file": str(local_file or "").strip() or None,
                "retrieved_section": str(retrieved_section or "").strip() or None,
                "note": str(note or "").strip() or None,
            }
        )
        return source_id

    def marker(self, source_id: int | None) -> str:
        """Return a citation marker like [1]."""
        return f"[{source_id}]" if source_id else ""

    def sources(self) -> list[dict[str, Any]]:
        """Return sources in display order."""
        return list(self._sources)


def load_profile_result(path: str | Path = PROFILE_RESULT_PATH) -> dict[str, Any]:
    """Load the saved Interest Profiler result JSON."""
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("Profile result JSON must contain an object.")
    return value


def build_final_career_report(profile_for_rag: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
    """Build, save, and return the structured final career report."""
    ensure_required_artifacts(check_chroma=True)
    top_k = max(1, min(int(top_k), 10))
    citation_manager = CitationManager()
    day_provider = OpenAIChatProvider()
    ip_source_id = _add_interest_profiler_source(citation_manager)

    profile_used = _build_profile_used(profile_for_rag, citation_manager.marker(ip_source_id))
    occupation_index = build_occupation_index(DB_PATH)
    candidate_groups = _candidate_groups_from_profile(profile_for_rag, profile_used)
    candidate_titles = _candidate_titles_from_groups(candidate_groups)

    top_matches: list[dict[str, Any]] = []
    used_soc_codes: set[str] = set()
    used_candidate_titles: set[str] = set()
    for group in candidate_groups:
        group_limit = int(group.get("limit") or 5)
        group_count = 0
        for candidate in group.get("candidates") or []:
            if len(top_matches) >= top_k or group_count >= group_limit:
                break
            display_title = str(candidate.get("career_title") or candidate.get("title") or "").strip()
            if not display_title:
                continue
            title_key = _title_key(display_title)
            if title_key in used_candidate_titles:
                continue
            resolution = resolve_career_alias(display_title, occupation_index)
            soc_code = resolution.get("onet_soc_code")
            if not soc_code or soc_code in used_soc_codes:
                continue
            details = get_onet_occupation_details(soc_code, citation_manager)
            if not details.get("description"):
                continue

            used_candidate_titles.add(title_key)
            used_soc_codes.add(str(soc_code))
            group_count += 1
            rank = len(top_matches) + 1
            top_matches.append(
                _build_match(
                    rank=rank,
                    display_title=display_title,
                    resolution=resolution,
                    details=details,
                    profile_used=profile_used,
                    citation_manager=citation_manager,
                    day_provider=day_provider,
                    match_group=str(group.get("label") or ""),
                    target_job_zone=group.get("job_zone"),
                )
            )

    alternatives = _build_alternative_careers(
        candidate_titles=candidate_titles,
        occupation_index=occupation_index,
        citation_manager=citation_manager,
        used_candidate_titles=used_candidate_titles,
    )

    if not top_matches:
        raise RuntimeError(
            "Retrieval returned zero occupation-specific O*NET evidence. "
            "Check the local DuckDB/Chroma artifacts with "
            "`python scripts\\check_artifacts.py` before generating a report."
        )

    future_impact_summary = build_future_impact_summary(
        profile_used=profile_used,
        top_matches=top_matches,
        citation_manager=citation_manager,
    )
    semantic_retrieval_report = build_semantic_retrieval_report(
        profile_used=profile_used,
        citation_manager=citation_manager,
    )

    report = {
        "report_generation_method": "template_fallback_local_evidence",
        "top_match_grouping": "current_zone_5_future_zone_5",
        "future_impact_method": "semantic_research_rag",
        "semantic_report_method": "semantic_onet_ai_impact_report",
        "llm_system_prompt_available": True,
        "profile_used": profile_used,
        "core_skills_across_matches": _core_skills_from_matches(top_matches),
        "top_matches": top_matches,
        "alternative_careers": alternatives,
        "future_impact_summary": future_impact_summary,
        "semantic_retrieval_report": semantic_retrieval_report,
        "sources": citation_manager.sources(),
    }

    save_final_report(report)
    save_markdown_report(report)
    return report


def get_onet_occupation_details(
    onet_soc_code: str,
    citation_manager: CitationManager | None = None,
) -> dict[str, Any]:
    """Retrieve one occupation's details from local DuckDB O*NET tables."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required to retrieve O*NET occupation details.") from exc

    if not DB_PATH.exists():
        raise FileNotFoundError(f"O*NET DuckDB database not found: {DB_PATH}")

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        occupation = conn.execute(
            """
            SELECT onetsoc_code, title, description
            FROM occupation_data
            WHERE onetsoc_code = ?
            LIMIT 1
            """,
            [onet_soc_code],
        ).fetchone()
        if not occupation:
            return {}

        code, title, description = occupation
        source_id = None
        if citation_manager is not None:
            source_id = citation_manager.add_source(
                title=f"O*NET occupation data - {title}",
                source_type="onet",
                url=_onet_summary_url(str(code)),
                local_file="data/duckdb/onet.duckdb",
                retrieved_section="description, tasks, skills, knowledge, abilities, education, job zone, work context, software, related occupations",
                note=f"O*NET-SOC {code}",
            )
        citation = citation_manager.marker(source_id) if citation_manager else ""
        task_records = _fetch_task_records(conn, str(code), limit=6)

        details = {
            "occupation_title": str(title),
            "onet_soc_code": str(code),
            "description": _with_marker(description, citation),
            "tasks": _with_marker_list([item["task"] for item in task_records], citation),
            "task_records": task_records,
            "skills": _with_marker_list(_fetch_element_names(conn, "essential_skills", str(code), limit=12), citation),
            "knowledge": _with_marker_list(_fetch_element_names(conn, "knowledge", str(code), limit=8), citation),
            "abilities": _with_marker_list(_fetch_element_names(conn, "abilities", str(code), limit=5), citation),
            "education": _with_marker_list(_fetch_education(conn, str(code), limit=4), citation),
            "job_zone": _fetch_job_zone(conn, str(code), citation),
            "work_context": _with_marker_list(_fetch_work_context(conn, str(code), limit=5), citation),
            "software": _with_marker_list(_fetch_software(conn, str(code), limit=16), citation),
            "related_occupations": _with_marker_list(_fetch_related(conn, str(code), limit=5), citation),
            "citation": citation,
        }
    finally:
        conn.close()

    return details


def save_final_report(
    report: dict[str, Any],
    path: str | Path = FINAL_REPORT_JSON_PATH,
) -> Path:
    """Save the final career report JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=True)
        file.write("\n")
    return output_path


def save_markdown_report(
    report: dict[str, Any],
    path: str | Path = FINAL_REPORT_MD_PATH,
) -> Path:
    """Save a readable markdown version of the final report."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_to_markdown(report), encoding="utf-8")
    return output_path


def report_to_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown version of the report."""
    profile = report.get("profile_used") or {}
    lines = [
        "# Final Career Report",
        "",
        "## Profile Summary",
        f"- Holland code used for matching: {profile.get('final_holland_code', '')}",
        f"- Current Job Zone: {profile.get('current_job_zone', '')}",
        f"- Future Job Zone: {profile.get('future_job_zone', '')}",
        f"- Preferences: {', '.join(profile.get('sub_preferences') or [])}",
        "",
        "## Top Career Matches",
    ]

    for match in report.get("top_matches") or []:
        lines.extend(
            [
                "",
                f"### {match.get('rank')}. {match.get('display_title')}",
                f"Resolved O*NET title: {match.get('resolved_onet_title')}",
                f"Fit score: {match.get('fit_score')}",
                "",
                "Why it fits:",
            ]
        )
        lines.extend(f"- {item}" for item in match.get("why_it_fits") or [])
        lines.extend(
            [
                "",
                f"What this job does: {match.get('onet_details', {}).get('description', '')}",
                "",
                "AI impact:",
            ]
        )
        for row in match.get("ai_impact", {}).get("task_breakdown") or []:
            score = row.get("score_display")
            if not score:
                score = "N/A" if row.get("score") is None else str(row.get("score"))
            lines.append(
                f"- {row.get('task')}: {row.get('automation_level')} "
                f"({score}, {row.get('score_type')}) - {row.get('evidence')}"
            )
        lines.extend(
            [
                "",
                f"Key skills: {', '.join(_flatten_key_skills(match.get('key_skills') or {}))}",
                f"Education needed: {match.get('education_needed')}",
                f"Day in the life: {match.get('day_in_the_life')}",
            ]
        )

    lines.extend(["", "## Alternative Careers"])
    for item in report.get("alternative_careers") or []:
        zone = (item.get("job_zone") or {}).get("zone")
        zone_text = f" Job Zone {zone}." if zone else ""
        source_marker = " ".join(re.findall(r"\[\d+\]", str(item.get("reason") or "")))
        marker_text = f" {source_marker}" if source_marker else ""
        lines.append(f"- {item.get('title')}:{zone_text}{marker_text}")

    future_impact = report.get("future_impact_summary") or {}
    lines.extend(["", "## Future Impact"])
    if future_impact.get("summary"):
        lines.append(str(future_impact["summary"]))
    for item in future_impact.get("takeaways") or []:
        lines.append(f"- {item}")
    if future_impact.get("retrieved_research"):
        lines.append("")
        lines.append("Retrieved research:")
        for item in future_impact.get("retrieved_research") or []:
            page = f", page {item.get('page')}" if item.get("page") else ""
            lines.append(f"- [{item.get('source_id')}] {item.get('title')}{page}: {item.get('snippet')}")

    semantic_report = report.get("semantic_retrieval_report") or {}
    lines.extend(["", "## Semantic Retrieval Report"])
    if semantic_report.get("summary"):
        lines.append(str(semantic_report["summary"]))
    for item in semantic_report.get("takeaways") or []:
        lines.append(f"- {item}")
    if semantic_report.get("semantic_career_signals"):
        lines.append("")
        lines.append("Semantic career signals:")
        for item in semantic_report.get("semantic_career_signals") or []:
            markers = " ".join(f"[{source_id}]" for source_id in item.get("source_ids") or [])
            soc = f" ({item.get('soc_code')})" if item.get("soc_code") else ""
            lines.append(f"- {item.get('title')}{soc}: {markers}")

    lines.extend(["", "## Sources"])
    for source in report.get("sources") or []:
        parts = [f"[{source.get('id')}] {source.get('title')}"]
        if source.get("retrieved_section"):
            parts.append(str(source["retrieved_section"]))
        if source.get("url"):
            parts.append(str(source["url"]))
        if source.get("local_file"):
            parts.append(str(source["local_file"]))
        if source.get("note"):
            parts.append(str(source["note"]))
        lines.append("- " + " - ".join(parts))

    return "\n".join(lines).rstrip() + "\n"


def _add_interest_profiler_source(citation_manager: CitationManager) -> int:
    return citation_manager.add_source(
        title="O*NET Interest Profiler Short Form and Career Listings",
        source_type="local_pdf",
        url=ONET_IP_RESOURCE_URL,
        local_file="onet_interest_profiler/Interest_Profiler.pdf; onet_interest_profiler/IP_Career_Listings.pdf",
        retrieved_section="RIASEC scores, Holland code, Job Zone choices, Interest Area career-listing context",
        note=f"Short form: {IP_SHORT_FORM_URL}; career listings: {IP_CAREER_LISTINGS_URL}",
    )


def _build_profile_used(profile_input: dict[str, Any], ip_marker: str) -> dict[str, Any]:
    is_raw_profile = "raw_riasec_scores" in profile_input
    prepared = prepare_profile_for_rag(profile_input) if is_raw_profile else dict(profile_input)

    raw_scores = prepared.get("riasec_scores") or {}
    raw_scores = {interest: int(raw_scores.get(interest, 0)) for interest in RIASEC_INTERESTS}
    initial_holland = (
        profile_input.get("initial_holland_code")
        if is_raw_profile
        else prepared.get("holland_code")
    )
    initial_holland = str(initial_holland or prepared.get("holland_code") or "")

    followup_refinement = profile_input.get("followup_refinement") or {} if is_raw_profile else {}
    questions_asked = list(followup_refinement.get("questions_asked") or [])
    final_refinement = (followup_refinement.get("final_refinement") or {}) if questions_asked else {}
    raw_sub_preferences = list(
        profile_input.get("preferences_used")
        or final_refinement.get("key_sub_preferences")
        or []
    )

    sub_preferences = _dedupe_text(raw_sub_preferences) or _normalize_sub_preferences(raw_sub_preferences)
    initial_top = list(prepared.get("top_interests") or [])
    final_top = list(final_refinement.get("refined_top_interests") or initial_top)
    final_holland = make_holland_code(final_top[:3]) if final_top else str(prepared.get("holland_code") or "")

    guidance = final_refinement.get("career_matching_guidance") or {}
    current_job_zone = int(prepared.get("current_job_zone") or 0)
    future_job_zone = int(prepared.get("future_job_zone") or 0)
    return {
        "profile_id": profile_input.get("profile_id"),
        "timestamp": profile_input.get("timestamp"),
        "riasec_scores": raw_scores,
        "top_interests": initial_top[:3],
        "holland_code": initial_holland,
        "initial_code": initial_holland,
        "initial_holland_code": initial_holland,
        "final_top_interests": final_top[:3],
        "refined_interests": final_top[:3],
        "final_code": final_holland,
        "final_holland_code": final_holland,
        "current_zone": current_job_zone,
        "future_zone": future_job_zone,
        "current_job_zone": current_job_zone,
        "future_job_zone": future_job_zone,
        "preferences_used": sub_preferences,
        "sub_preferences": sub_preferences,
        "future_vision_summary": str(final_refinement.get("future_vision_summary") or "").strip(),
        "concerns_noted": list(final_refinement.get("concerns_noted") or []),
        "career_matching_guidance": guidance.get("notes")
        or "Use future job zone for aspirational recommendations, current job zone for immediate options.",
        "profile_citation": ip_marker,
    }


def _candidate_titles_from_profile(profile_input: dict[str, Any]) -> list[str]:
    """Return report candidates from the submitted profile's matched careers."""
    return _candidate_titles_from_groups(_candidate_groups_from_profile(profile_input))


def _candidate_groups_from_profile(
    profile_input: dict[str, Any],
    profile_used: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return report candidates split into current-zone and future-zone pools."""
    profile_used = profile_used or {}
    try:
        current_zone = int(profile_input.get("current_job_zone") or profile_used.get("current_job_zone") or 0)
        future_zone = int(profile_input.get("future_job_zone") or profile_used.get("future_job_zone") or 0)
    except (TypeError, ValueError):
        current_zone = int(profile_used.get("current_job_zone") or 0)
        future_zone = int(profile_used.get("future_job_zone") or 0)

    return [
        {
            "label": "Current Job Zone options",
            "job_zone": current_zone,
            "limit": 5,
            "candidates": _candidate_records_for_zone(profile_input, profile_used, current_zone),
        },
        {
            "label": "Future Job Zone options",
            "job_zone": future_zone,
            "limit": 5,
            "candidates": _candidate_records_for_zone(profile_input, profile_used, future_zone),
        },
    ]


def _candidate_records_for_zone(
    profile_input: dict[str, Any],
    profile_used: dict[str, Any],
    zone: int,
) -> list[dict[str, Any]]:
    if zone not in {1, 2, 3, 4, 5}:
        return []

    followup_questions = list((profile_input.get("followup_refinement") or {}).get("questions_asked") or [])
    final_ranked = profile_input.get("final_ranked_matches") or profile_input.get("refined_career_matches") or []
    records = [
        {**item, "career_title": str(item.get("career_title") or item.get("title") or "").strip()}
        for item in final_ranked
        if (
            isinstance(item, dict)
            and str(item.get("career_title") or item.get("title") or "").strip()
            and _safe_int(item.get("job_zone")) == zone
        )
    ]

    if followup_questions and not final_ranked:
        raise RuntimeError(
            "Follow-up answers exist, but refined/final ranked matches are missing. "
            "Finish or recompute the follow-up refinement before generating the report."
        )

    career_matches = profile_input.get("career_matches") or {}
    for key in ("primary_current_zone", "primary_future_zone", "secondary_future_zone", "tertiary_future_zone"):
        for career in career_matches.get(key, []) or []:
            if _safe_int(career.get("job_zone")) == zone:
                records.append({**career, "career_title": str(career.get("career_title") or "").strip()})

    records.extend(_candidate_records_from_current_interests(profile_input, profile_used, zone))

    if not records:
        prepared = prepare_profile_for_rag(profile_input) if "raw_riasec_scores" in profile_input else dict(profile_input)
        records.extend(
            {"career_title": str(title).strip(), "job_zone": zone}
            for title in prepared.get("career_titles_to_retrieve") or []
        )

    return _dedupe_candidate_records(records)


def _candidate_records_from_current_interests(
    profile_input: dict[str, Any],
    profile_used: dict[str, Any],
    zone: int,
) -> list[dict[str, Any]]:
    try:
        listings = load_ip_career_listings(CAREER_LISTINGS_PATH)
    except (OSError, ValueError):
        return []

    top_interests = (
        profile_used.get("final_top_interests")
        or profile_input.get("refined_interests")
        or profile_input.get("final_top_interests")
        or profile_input.get("top_interests")
        or profile_input.get("initial_top_interests")
        or []
    )
    top_interests = [str(interest) for interest in top_interests][:3]
    if not top_interests:
        return []

    records: list[dict[str, Any]] = []
    for interest in top_interests:
        try:
            matches = match_careers(listings, interest, zone)
        except ValueError:
            continue
        records.extend(matches)
    return records


def _candidate_titles_from_groups(groups: list[dict[str, Any]]) -> list[str]:
    titles = []
    for group in groups:
        for candidate in group.get("candidates") or []:
            title = str(candidate.get("career_title") or candidate.get("title") or "").strip()
            if title:
                titles.append(title)
    return _dedupe_text(titles)


def _dedupe_candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for record in records:
        title = str(record.get("career_title") or record.get("title") or "").strip()
        if not title:
            continue
        key = _title_key(title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({**record, "career_title": title})
    return deduped


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _title_key(title: Any) -> str:
    return re.sub(r"\s+", " ", str(title or "").strip()).lower()


def _build_match(
    rank: int,
    display_title: str,
    resolution: dict[str, Any],
    details: dict[str, Any],
    profile_used: dict[str, Any],
    citation_manager: CitationManager,
    day_provider: OpenAIChatProvider | None = None,
    match_group: str = "",
    target_job_zone: Any = None,
) -> dict[str, Any]:
    resolved_title = str(resolution.get("resolved_onet_title") or details.get("occupation_title"))
    soc_code = str(resolution.get("onet_soc_code") or details.get("onet_soc_code"))
    ai_impact = build_ai_impact_for_occupation(
        display_title=display_title,
        resolved_onet_title=resolved_title,
        onet_soc_code=soc_code,
        onet_tasks=details.get("task_records") or [
            _strip_marker(item) for item in details.get("tasks", [])
        ],
        citation_manager=citation_manager,
    )
    fit_score = max(0.72, 0.96 - (rank - 1) * 0.025)

    alias_note = ""
    if display_title != resolved_title:
        alias_note = f"Displayed as {display_title}; O*NET evidence retrieved from {resolved_title}."

    return {
        "rank": rank,
        "display_title": display_title,
        "resolved_onet_title": resolved_title,
        "onet_soc_code": soc_code,
        "alias_resolution": resolution,
        "resolution_note": alias_note,
        "fit_score": round(fit_score, 2),
        "fit_label": _fit_label(fit_score),
        "match_group": match_group,
        "target_job_zone": target_job_zone,
        "why_it_fits": _why_it_fits(display_title, profile_used),
        "onet_details": details,
        "ai_impact": ai_impact,
        "key_skills": _key_skills(details),
        "education_needed": _education_needed(details),
        "day_in_the_life": _day_in_life(details, day_provider),
    }


def _build_alternative_careers(
    candidate_titles: list[str],
    occupation_index: list[dict[str, Any]],
    citation_manager: CitationManager,
    used_candidate_titles: set[str],
) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    for title in candidate_titles:
        if _title_key(title) in used_candidate_titles:
            continue
        resolution = resolve_career_alias(title, occupation_index)
        soc_code = resolution.get("onet_soc_code")
        if not soc_code:
            continue
        resolved_title = str(resolution.get("resolved_onet_title") or title)
        citation_id = citation_manager.add_source(
            title=f"O*NET occupation data - {resolved_title}",
            source_type="onet",
            url=_onet_summary_url(str(soc_code)),
            local_file="data/duckdb/onet.duckdb",
            retrieved_section="occupation title, O*NET-SOC alias resolution, and job zone",
            note=f"O*NET-SOC {soc_code}",
        )
        job_zone = _get_job_zone_for_soc(str(soc_code), citation_manager.marker(citation_id))
        reason = _alternative_reason(title)
        alternatives.append(
            {
                "title": title,
                "resolved_onet_title": resolved_title,
                "onet_soc_code": soc_code,
                "job_zone": job_zone,
                "reason": f"{reason} [{citation_id}]",
            }
        )
        if len(alternatives) >= 5:
            break
    return alternatives


def _fetch_task_records(conn: Any, code: str, limit: int) -> list[dict[str, Any]]:
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
    records = []
    seen_tasks: set[str] = set()
    for task_id, task, task_type in rows:
        task_text = re.sub(r"\s+", " ", str(task or "")).strip()
        if not task_text:
            continue
        task_key = task_text.lower()
        if task_key in seen_tasks:
            continue
        seen_tasks.add(task_key)
        records.append(
            {
                "onet_soc_code": str(code),
                "task_id": _task_id_string(task_id),
                "task": task_text,
                "task_type": re.sub(r"\s+", " ", str(task_type or "")).strip() or None,
            }
        )
    return records


def _fetch_tasks(conn: Any, code: str, limit: int) -> list[str]:
    return [item["task"] for item in _fetch_task_records(conn, code, limit)]


def _fetch_element_names(conn: Any, table_name: str, code: str, limit: int) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT cmr.element_name
        FROM {table_name} AS item
        JOIN content_model_reference AS cmr
            ON item.element_id = cmr.element_id
        WHERE item.onetsoc_code = ?
            AND item.scale_id = 'IM'
        ORDER BY item.data_value DESC, cmr.element_name
        LIMIT ?
        """,
        [code, limit],
    ).fetchall()
    return _dedupe_text(row[0] for row in rows)


def _fetch_education(conn: Any, code: str, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT ec.category_description, e.data_value
        FROM education AS e
        LEFT JOIN education_categories AS ec
            ON e.element_id = ec.element_id
            AND e.scale_id = ec.scale_id
            AND e.category = ec.category
        WHERE e.onetsoc_code = ?
            AND ec.category_description IS NOT NULL
        ORDER BY e.data_value DESC
        LIMIT ?
        """,
        [code, limit],
    ).fetchall()
    values = []
    for description, data_value in rows:
        if description:
            values.append(f"{description} ({float(data_value):.2f}%)")
    return _dedupe_text(values)


def _fetch_job_zone(conn: Any, code: str, citation: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
            CAST(jz.job_zone AS INTEGER),
            jzr.name,
            jzr.experience,
            jzr.education,
            jzr.job_training
        FROM job_zones AS jz
        LEFT JOIN job_zone_reference AS jzr
            ON jz.job_zone = jzr.job_zone
        WHERE jz.onetsoc_code = ?
        LIMIT 1
        """,
        [code],
    ).fetchone()
    if not row:
        return {}
    return {
        "zone": int(row[0]),
        "name": _with_marker(row[1], citation),
        "experience": _with_marker(row[2], citation),
        "education": _with_marker(row[3], citation),
        "training": _with_marker(row[4], citation),
    }


def _get_job_zone_for_soc(onet_soc_code: str, citation: str = "") -> dict[str, Any]:
    try:
        import duckdb
    except ImportError:
        return {}
    if not DB_PATH.exists():
        return {}

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return _fetch_job_zone(conn, str(onet_soc_code), citation)
    finally:
        conn.close()


def _fetch_work_context(conn: Any, code: str, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT cmr.element_name, wcc.category_description
        FROM work_context AS wc
        JOIN content_model_reference AS cmr
            ON wc.element_id = cmr.element_id
        LEFT JOIN work_context_categories AS wcc
            ON wc.element_id = wcc.element_id
            AND wc.scale_id = wcc.scale_id
            AND wc.category = wcc.category
        WHERE wc.onetsoc_code = ?
        ORDER BY wc.data_value DESC
        LIMIT ?
        """,
        [code, limit],
    ).fetchall()
    values = []
    for name, category in rows:
        if category:
            values.append(f"{name}: {category}")
        else:
            values.append(str(name))
    return _dedupe_text(values)


def _fetch_software(conn: Any, code: str, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT workplace_example, in_demand
        FROM software_skills
        WHERE onetsoc_code = ?
            AND workplace_example IS NOT NULL
        ORDER BY
            CASE WHEN UPPER(COALESCE(in_demand, '')) = 'Y' THEN 0 ELSE 1 END,
            workplace_example
        LIMIT ?
        """,
        [code, limit],
    ).fetchall()
    values = []
    for skill, in_demand in rows:
        text = str(skill or "").strip()
        if not text:
            continue
        if str(in_demand or "").strip().upper() == "Y":
            text += " (In demand)"
        values.append(text)
    return _dedupe_text(values)


def _fetch_related(conn: Any, code: str, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT related_title.title
        FROM related_occupations AS related
        JOIN occupation_data AS related_title
            ON related.related_onetsoc_code = related_title.onetsoc_code
        WHERE related.onetsoc_code = ?
        ORDER BY related.related_index, related_title.title
        LIMIT ?
        """,
        [code, limit],
    ).fetchall()
    return _dedupe_text(row[0] for row in rows)


def _normalize_sub_preferences(values: list[Any]) -> list[str]:
    text = " ".join(str(item) for item in values).lower()
    preferences = []
    if not text.strip():
        return []
    if "technology" in text or "tech" in text:
        preferences.append("technology")
    if "data" in text or "math" in text or "analyt" in text or "problem" in text or "technology" in text:
        preferences.append("math and analytical work")
        preferences.append("analytical thinking and problem solving")
    if "fast" in text or "multiple projects" in text or "short time" in text:
        preferences.append("fast-paced short projects")
    if "start" in text or "own thing" in text or "entrepreneur" in text:
        preferences.append("starting own thing")
    if "teaching" in text or "guiding" in text or "listen" in text:
        preferences.append("teaching/guiding with some listening")
    if "social" in text or "people" in text or "teaching" in text:
        preferences.append("lower social interaction than people-heavy careers")
    if "ai" in text:
        preferences.append("AI-aware career planning")
    return _dedupe_text(preferences)


def _core_skills_from_matches(top_matches: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for match in top_matches:
        details = match.get("onet_details") or {}
        values.extend(_strip_marker(item) for item in details.get("skills") or [])
        values.extend(_strip_marker(item) for item in details.get("knowledge") or [])
    return _dedupe_text(values)[:10]


def _why_it_fits(display_title: str, profile_used: dict[str, Any]) -> list[str]:
    marker = profile_used.get("profile_citation") or ""
    preferences = set(profile_used.get("sub_preferences") or [])
    preference_text = " ".join(preferences).lower()
    bullets = []
    if "technology" in preferences or "technology" in preference_text:
        bullets.append(f"You said you prefer technology, and this path keeps the work close to technical systems and tools. {marker}")
    if (
        "math and analytical work" in preferences
        or "analytical thinking and problem solving" in preferences
        or "data" in preference_text
        or "research" in preference_text
        or "problem" in preference_text
    ):
        bullets.append("The role uses analytical thinking, data, math, or structured problem solving.")
    if "fast-paced short projects" in preferences or "fast-paced" in preference_text:
        bullets.append("It can support fast-paced project work where you solve several smaller problems over time.")
    if "starting own thing" in preferences or "leading" in preference_text:
        bullets.append("The skill set can transfer into independent consulting, product ideas, or building your own project.")
    if "lower social interaction than people-heavy careers" in preferences or "lower social" in preference_text:
        bullets.append("It can involve less continuous people-facing interaction than many Social-heavy career paths.")

    if not bullets:
        interests = ", ".join(profile_used.get("final_top_interests") or profile_used.get("top_interests") or [])
        zone = profile_used.get("future_job_zone") or profile_used.get("future_zone")
        bullets.append(f"These roles are included using the submitted Interest Profiler result ({interests}) and selected Job Zones, including future Job Zone {zone}. {marker}")
    return bullets[:4]


def _key_skills(details: dict[str, Any]) -> dict[str, list[str]]:
    raw_skills = list(details.get("skills") or [])
    software = _dedupe_text(details.get("software") or [])[:16]
    knowledge = _dedupe_text(details.get("knowledge") or [])[:8]

    foundational = []
    technical = []
    for skill in raw_skills:
        key = _strip_marker(skill).lower()
        if key in FOUNDATIONAL_SOFT_SKILLS:
            foundational.append(skill)
        else:
            technical.append(skill)

    return {
        "software_tools": software,
        "technical_and_domain": _dedupe_text(technical)[:10],
        "foundational_communication": _dedupe_text(foundational)[:8],
        "knowledge_areas": knowledge,
    }


def _flatten_key_skills(grouped: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "software_tools",
        "technical_and_domain",
        "knowledge_areas",
        "foundational_communication",
    ):
        values.extend(str(item) for item in grouped.get(key) or [])
    return _dedupe_text(values)[:18]


def _education_needed(details: dict[str, Any]) -> str:
    job_zone = details.get("job_zone") or {}
    education = details.get("education") or []
    zone = job_zone.get("zone")
    education_summary = _education_needed_from_percentages(education)
    if education_summary:
        return education_summary
    zone_education = _strip_marker(job_zone.get("education") or "")
    zone_education = re.sub(r",?\s*but some do not\.?", ".", zone_education, flags=re.IGNORECASE)
    if zone and zone_education:
        explanation = JOB_ZONE_PREPARATION_EXPLANATIONS.get(int(zone), "preparation level unavailable")
        return f"O*NET places this occupation in Job Zone {zone} ({explanation}). Typical preparation: {zone_education}"
    return "Education information was limited in the retrieved O*NET rows."


def _education_needed_from_percentages(education: list[Any]) -> str:
    entries = [_parse_education_percentage(item) for item in education]
    entries = [entry for entry in entries if entry]
    if not entries:
        return ""
    leading = sorted(entries, key=lambda item: item[1], reverse=True)[:2]
    return _plain_education_needed_summary(leading)


def _plain_education_needed_summary(entries: list[tuple[str, float]]) -> str:
    positive_entries = [(label, percent) for label, percent in entries if percent > 0]
    if not positive_entries:
        return ""
    top_label = positive_entries[0][0]
    top_level = _degree_level(top_label)
    levels = {_degree_level(label) for label, _percent in positive_entries}
    if top_level == "master":
        if "bachelor" in levels:
            return (
                "Most O*NET responses point to a master's degree for this role; "
                "bachelor's-degree paths are present but less common."
            )
        return "Most O*NET responses point to a master's degree for this role."
    if top_level == "bachelor":
        if "master" in levels:
            return (
                "Most O*NET responses point to a bachelor's degree; "
                "master's-level preparation is also common."
            )
        return "Most O*NET responses point to a bachelor's degree for this role."
    if top_level == "doctoral":
        return "Most O*NET responses point to doctoral or professional-degree preparation for this role."
    if top_level == "associate":
        return "Most O*NET responses point to associate-degree or vocational preparation for this role."
    if top_level == "high_school":
        return "Most O*NET responses point to high-school-level preparation for this role."
    return f"Most O*NET responses point to {top_label.lower()} for this role."


def _degree_level(label: Any) -> str:
    text = str(label or "").lower()
    if "master" in text:
        return "master"
    if "bachelor" in text:
        return "bachelor"
    if "doctoral" in text or "professional degree" in text:
        return "doctoral"
    if "associate" in text or "2-year" in text:
        return "associate"
    if "high school" in text or "ged" in text:
        return "high_school"
    return "other"


def _parse_education_percentage(value: Any) -> tuple[str, float] | None:
    text = _strip_marker(value)
    match = re.match(r"^(?P<label>.+?)\s*\((?P<number>\d+(?:\.\d+)?)%?\)$", text)
    if not match:
        return None
    return match.group("label").strip(), float(match.group("number"))


def _day_in_life(details: dict[str, Any], provider: OpenAIChatProvider | None = None) -> str:
    tasks = [_strip_marker(item) for item in details.get("tasks") or []][:3]
    contexts = [_strip_marker(item) for item in details.get("work_context") or []][:2]
    title = _strip_marker(details.get("occupation_title") or "this role")
    if provider and provider.available and tasks:
        try:
            generated = _llm_day_in_life(
                title=title,
                tasks=tasks,
                contexts=contexts,
                provider=provider,
            )
            if generated:
                return generated
        except Exception:
            pass
    return _template_day_in_life(title, tasks, contexts)


def _llm_day_in_life(
    title: str,
    tasks: list[str],
    contexts: list[str],
    provider: OpenAIChatProvider,
) -> str:
    prompt = (
        "Write one warm, realistic day-in-the-life paragraph for the career below. "
        "Use only the supplied O*NET task and work-context evidence. Do not mention "
        "O*NET, citations, inference, AI, automation, or sources. Do not list the tasks; "
        "turn them into a natural workday. Keep it to 3 short sentences.\n\n"
        f"Career: {title}\n"
        f"Tasks: {json.dumps(tasks, ensure_ascii=True)}\n"
        f"Work context: {json.dumps(contexts, ensure_ascii=True)}"
    )
    text = provider.complete(
        messages=[
            {
                "role": "system",
                "content": "You write concise, human career guidance grounded in supplied evidence.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.45,
    )
    return _clean_generated_day_text(text)


def _template_day_in_life(title: str, tasks: list[str], contexts: list[str]) -> str:
    if not tasks:
        return (
            "The day-to-day rhythm is hard to pin down from the retrieved rows, so this "
            "career deserves more occupation-specific research before deciding."
        )
    first = _activity_phrase(tasks[0])
    second = _activity_phrase(tasks[1]) if len(tasks) > 1 else ""
    third = _activity_phrase(tasks[2]) if len(tasks) > 2 else ""
    context_text = _context_phrase(contexts)
    role_phrase = "this role" if title == "this role" else f"{_article_for_title(title)} {title}"
    opening = (
        f"A typical day as {role_phrase} would probably start with sorting out what needs "
        f"attention first, then spending focused time on {first}."
    )
    if second and third:
        middle = f"As the day develops, the work may shift between {second} and {third}, so the role rewards someone who can keep the details connected."
    elif second:
        middle = f"As the day develops, the work may shift toward {second}, so the role rewards steady attention and practical judgment."
    else:
        middle = "The work is less about one dramatic moment and more about making sound decisions as similar responsibilities come up again and again."
    if context_text:
        closing = f"The setting also points to {context_text}, which shapes the pace and the kind of communication the day asks for."
    else:
        closing = "By the end of the day, the value is in leaving the work clearer, more accurate, and easier for the next person to act on."
    return " ".join([opening, middle, closing])


def _alternative_reason(title: str) -> str:
    del title
    return "Related career from the submitted Interest Profiler matches."


def _fit_label(fit_score: float) -> str:
    if fit_score >= 0.9:
        return "Strong fit"
    if fit_score >= 0.8:
        return "Good fit"
    return "Possible fit"


def _with_marker(value: Any, marker: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if marker and marker not in text:
        return f"{text} {marker}"
    return text


def _with_marker_list(values: list[str], marker: str) -> list[str]:
    return [_with_marker(value, marker) for value in values if str(value or "").strip()]


def _strip_marker(value: Any) -> str:
    return re.sub(r"\s*\[\d+\]\s*$", "", str(value or "")).strip()


def _clean_generated_day_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip("\"'")
    text = re.sub(r"^my inference:\s*", "", text, flags=re.IGNORECASE)
    return text[:900].strip()


def _lower_first(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return value[:1].lower() + value[1:]


def _activity_phrase(value: str) -> str:
    text = _lower_first(_strip_terminal(value))
    replacements = {
        "analyze": "analyzing",
        "analyse": "analyzing",
        "assess": "assessing",
        "build": "building",
        "calculate": "calculating",
        "collect": "collecting",
        "communicate": "communicating",
        "compile": "compiling",
        "conduct": "conducting",
        "coordinate": "coordinating",
        "create": "creating",
        "design": "designing",
        "develop": "developing",
        "evaluate": "evaluating",
        "examine": "examining",
        "identify": "identifying",
        "inspect": "inspecting",
        "interpret": "interpreting",
        "maintain": "maintaining",
        "manage": "managing",
        "monitor": "monitoring",
        "operate": "operating",
        "organize": "organizing",
        "perform": "performing",
        "prepare": "preparing",
        "provide": "providing",
        "review": "reviewing",
        "test": "testing",
        "write": "writing",
    }
    for verb, gerund in replacements.items():
        prefix = f"{verb} "
        if text.startswith(prefix):
            return gerund + text[len(verb):]
    return text


def _context_phrase(values: list[str]) -> str:
    phrases = []
    for value in values:
        text = _lower_first(_strip_terminal(value))
        label = text.split(":", 1)[0].strip()
        if not label:
            continue
        if label == "electronic mail":
            phrases.append("regular email communication")
        elif label == "indoors, environmentally controlled":
            phrases.append("an indoor, controlled workspace")
        else:
            phrases.append(label)
    return _join_natural(_dedupe_text(phrases)[:2])


def _join_natural(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} and {values[-1]}"


def _article_for_title(title: str) -> str:
    first_word = re.sub(r"[^A-Za-z]", "", str(title or "").strip().split(" ", 1)[0]).lower()
    return "an" if first_word[:1] in {"a", "e", "i", "o", "u"} else "a"


def _strip_terminal(value: str) -> str:
    return re.sub(r"[.;:\s]+$", "", str(value or "").strip())


def _onet_summary_url(onet_soc_code: str) -> str:
    return f"https://www.onetonline.org/link/summary/{str(onet_soc_code).strip()}"


def _task_id_string(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _dedupe_text(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
