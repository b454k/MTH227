"""Final career report builder for the O*NET Interest Profiler flow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from career_rag.config import PROJECT_ROOT
from career_rag.interest_profiler_local import (
    RIASEC_INTERESTS,
    make_holland_code,
    prepare_profile_for_rag,
)
from career_rag.ip_ai_impact import build_ai_impact_for_occupation
from career_rag.occupation_aliases import build_occupation_index, resolve_career_alias


DATA_DIR = PROJECT_ROOT / "onet_interest_profiler"
PROFILE_RESULT_PATH = DATA_DIR / "ip_profile_result.json"
FINAL_REPORT_JSON_PATH = DATA_DIR / "ip_final_career_report.json"
FINAL_REPORT_MD_PATH = DATA_DIR / "ip_final_career_report.md"
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "onet.duckdb"

IP_SHORT_FORM_URL = "https://www.onetcenter.org/dl_tools/ipsf/Interest_Profiler.pdf"
IP_CAREER_LISTINGS_URL = "https://www.onetcenter.org/dl_tools/ipsf/IP_Career_Listings.pdf"
ONET_IP_RESOURCE_URL = "https://www.onetcenter.org/IP.html"

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
* day-in-the-life inference

Return valid JSON matching the report schema.
Do not include markdown inside JSON unless needed for short text fields.
Use citation IDs like [1], [2] inside strings where appropriate.
Do not include raw source URLs inside main text; sources are listed separately.
"""

FALLBACK_RANKING = [
    "Data Analyst",
    "Actuary",
    "Machine Learning Engineer",
    "Data Scientist",
    "Business Intelligence Analyst",
    "Operations Research Analyst",
    "Statistician",
    "Financial Quantitative Analyst",
    "Information Technology Project Manager",
    "Search Marketing Strategist",
]

ALTERNATIVE_TITLES = [
    "Operations Research Analyst",
    "Business Intelligence Analyst",
    "Statistician",
    "Financial Quantitative Analyst",
    "Information Technology Project Manager",
]

CORE_SKILLS = [
    "SQL",
    "Python",
    "statistics",
    "Excel/spreadsheets",
    "data visualization",
    "communication of insights",
    "domain knowledge",
]

JOB_SPECIFIC_SKILLS = {
    "Data Analyst": ["dashboard design", "business metrics", "data cleaning workflows"],
    "Actuary": ["probability", "actuarial exams", "risk modeling", "insurance/finance domain knowledge"],
    "Machine Learning Engineer": ["machine learning basics", "model evaluation", "software engineering", "data ethics"],
    "Data Scientist": ["machine learning basics", "experimental design", "feature engineering"],
    "Business Intelligence Analyst": ["BI tools", "data modeling", "KPI design"],
    "Operations Research Analyst": ["optimization", "simulation", "linear programming"],
    "Statistician": ["statistical inference", "experimental design", "survey methods"],
    "Financial Quantitative Analyst": ["financial mathematics", "risk modeling", "time-series analysis"],
    "Information Technology Project Manager": ["agile delivery", "technical planning", "stakeholder management"],
    "Search Marketing Strategist": ["analytics platforms", "A/B testing", "SEO/SEM metrics"],
}


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
    top_k = max(1, min(int(top_k), 10))
    citation_manager = CitationManager()
    ip_source_id = _add_interest_profiler_source(citation_manager)

    profile_used = _build_profile_used(profile_for_rag, citation_manager.marker(ip_source_id))
    occupation_index = build_occupation_index(DB_PATH)

    top_matches: list[dict[str, Any]] = []
    used_soc_codes: set[str] = set()
    for display_title in FALLBACK_RANKING:
        if len(top_matches) >= top_k:
            break
        resolution = resolve_career_alias(display_title, occupation_index)
        soc_code = resolution.get("onet_soc_code")
        if not soc_code:
            continue
        if soc_code in used_soc_codes:
            continue
        details = get_onet_occupation_details(soc_code, citation_manager)
        if not details.get("description"):
            continue

        used_soc_codes.add(str(soc_code))
        rank = len(top_matches) + 1
        top_matches.append(
            _build_match(
                rank=rank,
                display_title=display_title,
                resolution=resolution,
                details=details,
                profile_used=profile_used,
                citation_manager=citation_manager,
            )
        )

    alternatives = _build_alternative_careers(
        occupation_index=occupation_index,
        citation_manager=citation_manager,
        top_soc_codes=used_soc_codes,
    )

    report = {
        "report_generation_method": "template_fallback",
        "llm_system_prompt_available": True,
        "profile_used": profile_used,
        "core_skills_across_matches": CORE_SKILLS,
        "top_matches": top_matches,
        "alternative_careers": alternatives,
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
                url=None,
                local_file="data/duckdb/onet.duckdb",
                retrieved_section="description, tasks, skills, knowledge, abilities, education, job zone, work context, software, related occupations",
                note=f"O*NET-SOC {code}",
            )
        citation = citation_manager.marker(source_id) if citation_manager else ""

        details = {
            "occupation_title": str(title),
            "onet_soc_code": str(code),
            "description": _with_marker(description, citation),
            "tasks": _with_marker_list(_fetch_tasks(conn, str(code), limit=6), citation),
            "skills": _with_marker_list(_fetch_element_names(conn, "essential_skills", str(code), limit=6), citation),
            "knowledge": _with_marker_list(_fetch_element_names(conn, "knowledge", str(code), limit=5), citation),
            "abilities": _with_marker_list(_fetch_element_names(conn, "abilities", str(code), limit=5), citation),
            "education": _with_marker_list(_fetch_education(conn, str(code), limit=4), citation),
            "job_zone": _fetch_job_zone(conn, str(code), citation),
            "work_context": _with_marker_list(_fetch_work_context(conn, str(code), limit=5), citation),
            "software": _with_marker_list(_fetch_software(conn, str(code), limit=8), citation),
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
            lines.append(
                f"- {row.get('task')}: {row.get('automation_level')} "
                f"({row.get('score')}, {row.get('score_type')}) - {row.get('evidence')}"
            )
        lines.extend(
            [
                "",
                f"Skills to learn: {', '.join(match.get('skills_to_learn') or [])}",
                f"Education needed: {match.get('education_needed')}",
                f"Day in the life: {match.get('day_in_the_life')}",
            ]
        )

    lines.extend(["", "## Alternative Careers"])
    for item in report.get("alternative_careers") or []:
        lines.append(f"- {item.get('title')}: {item.get('reason')}")

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
    raw_sub_preferences = list(final_refinement.get("key_sub_preferences") or [])

    sub_preferences = _normalize_sub_preferences(raw_sub_preferences)
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


def _build_match(
    rank: int,
    display_title: str,
    resolution: dict[str, Any],
    details: dict[str, Any],
    profile_used: dict[str, Any],
    citation_manager: CitationManager,
) -> dict[str, Any]:
    resolved_title = str(resolution.get("resolved_onet_title") or details.get("occupation_title"))
    soc_code = str(resolution.get("onet_soc_code") or details.get("onet_soc_code"))
    ai_impact = build_ai_impact_for_occupation(
        display_title=display_title,
        resolved_onet_title=resolved_title,
        onet_soc_code=soc_code,
        onet_tasks=[_strip_marker(item) for item in details.get("tasks", [])],
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
        "why_it_fits": _why_it_fits(display_title, profile_used),
        "onet_details": details,
        "ai_impact": ai_impact,
        "skills_to_learn": _skills_to_learn(display_title, details),
        "education_needed": _education_needed(details),
        "day_in_the_life": _day_in_life(display_title, resolved_title),
    }


def _build_alternative_careers(
    occupation_index: list[dict[str, Any]],
    citation_manager: CitationManager,
    top_soc_codes: set[str],
) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    for title in ALTERNATIVE_TITLES:
        resolution = resolve_career_alias(title, occupation_index)
        soc_code = resolution.get("onet_soc_code")
        if not soc_code:
            continue
        resolved_title = str(resolution.get("resolved_onet_title") or title)
        citation_id = citation_manager.add_source(
            title=f"O*NET occupation data - {resolved_title}",
            source_type="onet",
            local_file="data/duckdb/onet.duckdb",
            retrieved_section="occupation title and O*NET-SOC alias resolution",
            note=f"O*NET-SOC {soc_code}",
        )
        reason = _alternative_reason(title)
        if soc_code in top_soc_codes:
            reason = f"Also relevant as a nearby title: {reason[0].lower() + reason[1:]}"
        alternatives.append(
            {
                "title": title,
                "resolved_onet_title": resolved_title,
                "onet_soc_code": soc_code,
                "reason": f"{reason} [{citation_id}]",
            }
        )
        if len(alternatives) >= 5:
            break
    return alternatives


def _fetch_tasks(conn: Any, code: str, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT task
        FROM task_statements
        WHERE onetsoc_code = ?
        ORDER BY task_id
        LIMIT ?
        """,
        [code, limit],
    ).fetchall()
    return _dedupe_text(row[0] for row in rows)


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
            values.append(f"{description} ({float(data_value):.2f})")
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
        ORDER BY workplace_example
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


def _why_it_fits(display_title: str, profile_used: dict[str, Any]) -> list[str]:
    marker = profile_used.get("profile_citation") or ""
    preferences = set(profile_used.get("sub_preferences") or [])
    bullets = []
    if "technology" in preferences:
        bullets.append(f"You said you prefer technology, and this path keeps the work close to technical systems and tools. {marker}")
    if "math and analytical work" in preferences or "analytical thinking and problem solving" in preferences:
        bullets.append("The role uses analytical thinking, data, math, or structured problem solving.")
    if "fast-paced short projects" in preferences:
        bullets.append("It can support fast-paced project work where you solve several smaller problems over time.")
    if "starting own thing" in preferences:
        bullets.append("The skill set can transfer into independent consulting, product ideas, or building your own project.")
    if "lower social interaction than people-heavy careers" in preferences:
        bullets.append("It can involve less continuous people-facing interaction than many Social-heavy career paths.")

    if not bullets:
        interests = ", ".join(profile_used.get("final_top_interests") or profile_used.get("top_interests") or [])
        zone = profile_used.get("future_job_zone") or profile_used.get("future_zone")
        bullets.append(f"This role is included using the submitted Interest Profiler result ({interests}) and future Job Zone {zone}. {marker}")

    if "Actuary" in display_title and len(bullets) < 4:
        bullets.append("It adds a strong quantitative risk path without requiring a fully people-centered workday.")
    if "Machine Learning" in display_title and len(bullets) < 4:
        bullets.append("It connects technology, problem solving, and project-based experimentation.")
    return bullets[:4]


def _skills_to_learn(display_title: str, details: dict[str, Any]) -> list[str]:
    skills = list(JOB_SPECIFIC_SKILLS.get(display_title, []))
    if "Data" in display_title or "Business Intelligence" in display_title:
        skills = ["SQL", "Python", "statistics", *skills]
    elif "Actuary" in display_title:
        skills = ["statistics", "Excel/spreadsheets", *skills]
    elif "Machine Learning" in display_title:
        skills = ["Python", "statistics", *skills]
    else:
        skills = [*skills, "communication of insights"]

    software = [_strip_marker(item) for item in details.get("software", [])]
    for tool in software[:2]:
        if tool and tool not in skills:
            skills.append(tool)
    return _dedupe_text(skills)[:8]


def _education_needed(details: dict[str, Any]) -> str:
    job_zone = details.get("job_zone") or {}
    education = details.get("education") or []
    zone = job_zone.get("zone")
    zone_name = job_zone.get("name") or ""
    zone_education = job_zone.get("education") or ""
    education_text = "; ".join(education[:2]) if education else "specific degree mix not available in the retrieved rows"
    if zone:
        return (
            f"O*NET lists this occupation as Job Zone {zone} ({zone_name}). "
            f"Common preparation: {zone_education}. Education rows include: {education_text}."
        )
    return f"Education information was limited in the retrieved local O*NET rows. Education rows include: {education_text}."


def _day_in_life(display_title: str, resolved_title: str) -> str:
    if "Actuary" in display_title:
        return (
            "Illustrative inference: a day may involve checking assumptions, updating risk models, reviewing results with a manager, and documenting why a pricing or reserve recommendation makes sense."
        )
    if "Machine Learning" in display_title:
        return (
            "Illustrative inference: a day may involve preparing data, running experiments, debugging model behavior, reading technical notes, and deciding whether a model is useful enough to deploy."
        )
    if "Project Manager" in display_title:
        return (
            "Illustrative inference: a day may involve clarifying priorities, checking delivery risks, translating technical blockers, and keeping a fast-moving project aligned."
        )
    if "Search Marketing" in display_title:
        return (
            "Illustrative inference: a day may involve checking campaign metrics, testing content or keyword ideas, and turning performance data into next actions."
        )
    return (
        "Illustrative inference: a day may involve cleaning data, writing queries, building a small report or model, checking results, and explaining the business meaning of the findings."
    )


def _alternative_reason(title: str) -> str:
    reasons = {
        "Operations Research Analyst": "Uses math, optimization, and analytical problem solving for business and operational decisions.",
        "Business Intelligence Analyst": "Connects data analysis with dashboards, business metrics, and practical recommendations.",
        "Statistician": "Keeps the work quantitative and evidence-focused with strong data interpretation.",
        "Financial Quantitative Analyst": "Combines math, finance, and analytical modeling in a business-facing path.",
        "Information Technology Project Manager": "Keeps the technology/business mix and fast-paced projects, with more coordination work.",
    }
    return reasons.get(title, "Related local O*NET evidence matched this analytical/business-facing direction.")


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
