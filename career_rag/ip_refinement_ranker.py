"""Follow-up-aware ranking for Interest Profiler career candidates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from career_rag.config import PROJECT_ROOT
from career_rag.interest_profiler_local import (
    RIASEC_INTERESTS,
    canonical_interest,
)
from career_rag.occupation_aliases import build_occupation_index, resolve_career_alias


DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "onet.duckdb"
RANKED_MATCH_KEYS = [
    "primary_future_zone",
    "primary_current_zone",
    "secondary_future_zone",
    "tertiary_future_zone",
]

STOPWORDS = {
    "a",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "do",
    "for",
    "from",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "work",
    "working",
    "would",
}

PREFERENCE_KEYWORDS = {
    "ai-aware career planning": ("ai", "automation", "automated", "artificial intelligence", "future proof"),
    "technology systems": ("technology", "tech", "software", "computer", "systems", "digital", "programming"),
    "data and numbers": ("data", "numbers", "analytics", "analysis", "statistical", "statistics", "math"),
    "research and problem solving": ("research", "science", "investigate", "understand", "problem", "solve"),
    "hands-on technical work": ("hands-on", "tools", "machines", "equipment", "build", "repair", "fix"),
    "outdoor or field work": ("outdoor", "outside", "field", "nature", "environment"),
    "creative design or writing": ("creative", "design", "visual", "writing", "art", "music", "perform"),
    "teaching or guiding people": ("teach", "teaching", "guide", "guiding", "help", "support", "mentor"),
    "leading or persuading": ("lead", "leading", "persuade", "sell", "sales", "pitch", "manage"),
    "structured independent work": ("structured", "organized", "records", "process", "independent", "alone"),
    "fast-paced projects": ("fast", "pace", "short projects", "multiple projects", "startup"),
    "job stability concern": ("stability", "stable", "secure", "risk", "worry", "concern", "realistic"),
    "lower social intensity": ("less people", "low social", "alone", "independent", "not talking"),
}


def build_initial_ranked_matches(career_matches: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten original IP match groups into a stable candidate-pool order."""
    flattened: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for group in RANKED_MATCH_KEYS:
        for index, career in enumerate(career_matches.get(group, []) or [], start=1):
            title = _clean_text(career.get("career_title"))
            if not title:
                continue
            title_key = _normalize(title)
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            flattened.append(
                {
                    **career,
                    "career_title": title,
                    "source_match_group": group,
                    "original_rank": len(flattened) + 1,
                    "original_group_rank": index,
                    "ranking_explanation": {
                        "original_profile": _original_profile_reason(career, group),
                        "followup_effects": [],
                    },
                }
            )
    return flattened


def build_refined_career_recommendations(
    profile_result: dict[str, Any],
    followup_refinement: dict[str, Any] | None,
    db_path: str | Path = DB_PATH,
) -> dict[str, Any]:
    """Return profile fields for final follow-up-aware career ranking."""
    initial_matches = build_initial_ranked_matches(profile_result.get("career_matches") or {})
    questions_asked = list((followup_refinement or {}).get("questions_asked") or [])
    final_refinement = (followup_refinement or {}).get("final_refinement") or {}
    preferences_used = extract_structured_preferences(followup_refinement)

    if not questions_asked:
        initial_with_scores = [
            {
                **match,
                "score": round(1.0 - index * 0.001, 4),
                "score_components": _empty_score_components(),
            }
            for index, match in enumerate(initial_matches)
        ]
        return _ranking_payload(
            initial_matches=initial_matches,
            final_matches=initial_with_scores,
            profile_result=profile_result,
            followup_refinement=followup_refinement,
            preferences_used=preferences_used,
            warnings=[],
        )

    target_interests = _target_interests(profile_result, final_refinement)
    target_zone = _target_job_zone(profile_result, final_refinement)
    preference_text = _preference_text(followup_refinement, preferences_used)
    occupation_index = build_occupation_index(db_path)

    scored_matches = []
    try:
        import duckdb
    except ImportError:
        duckdb = None

    conn = duckdb.connect(str(db_path), read_only=True) if duckdb and Path(db_path).exists() else None
    try:
        for match in initial_matches:
            scored_matches.append(
                _score_match(
                    match=match,
                    profile_result=profile_result,
                    target_interests=target_interests,
                    target_zone=target_zone,
                    preference_text=preference_text,
                    preferences_used=preferences_used,
                    occupation_index=occupation_index,
                    conn=conn,
                )
            )
    finally:
        if conn is not None:
            conn.close()

    scored_matches.sort(
        key=lambda item: (
            -float(item.get("score") or 0),
            int(item.get("original_rank") or 9999),
        )
    )

    warnings = []
    if _title_order(scored_matches) == _title_order(initial_matches):
        warnings.append(
            "Follow-up answers exist, but final_ranked_matches is identical to the original career_matches order."
        )

    return _ranking_payload(
        initial_matches=initial_matches,
        final_matches=scored_matches,
        profile_result=profile_result,
        followup_refinement=followup_refinement,
        preferences_used=preferences_used,
        warnings=warnings,
    )


def extract_structured_preferences(
    followup_refinement: dict[str, Any] | None,
) -> list[str]:
    """Convert follow-up answers and final refinement text into reusable preference labels."""
    if not followup_refinement:
        return []
    final_refinement = followup_refinement.get("final_refinement") or {}
    text_parts: list[str] = []
    text_parts.extend(str(item) for item in final_refinement.get("key_sub_preferences") or [])
    text_parts.append(str(final_refinement.get("future_vision_summary") or ""))
    text_parts.extend(str(item) for item in final_refinement.get("concerns_noted") or [])
    guidance = final_refinement.get("career_matching_guidance") or {}
    text_parts.append(str(guidance.get("notes") or ""))
    for item in followup_refinement.get("questions_asked") or []:
        text_parts.append(str(item.get("answer") or ""))

    combined = " ".join(text_parts).lower()
    preferences = [
        label
        for label, keywords in PREFERENCE_KEYWORDS.items()
        if any(keyword in combined for keyword in keywords)
    ]

    for item in final_refinement.get("key_sub_preferences") or []:
        cleaned = _clean_text(item)
        if cleaned and cleaned not in preferences:
            preferences.append(cleaned)
    return preferences[:12]


def _score_match(
    match: dict[str, Any],
    profile_result: dict[str, Any],
    target_interests: list[str],
    target_zone: int,
    preference_text: str,
    preferences_used: list[str],
    occupation_index: list[dict[str, Any]],
    conn: Any | None,
) -> dict[str, Any]:
    resolution = resolve_career_alias(match.get("career_title", ""), occupation_index)
    bundle = _occupation_signal_bundle(conn, resolution.get("onet_soc_code")) if conn else {}
    scores = profile_result.get("riasec_scores") or profile_result.get("raw_riasec_scores") or {}

    components = {
        "riasec_interest_fit": _riasec_interest_fit(match.get("interest"), target_interests, scores),
        "job_zone_fit": _job_zone_fit(match.get("job_zone"), target_zone),
        "task_work_activity_similarity": _text_similarity(
            preference_text,
            [*bundle.get("tasks", []), *bundle.get("work_activities", [])],
        ),
        "skills_knowledge_similarity": _text_similarity(
            preference_text,
            [*bundle.get("skills", []), *bundle.get("knowledge", [])],
        ),
        "work_context_fit": _work_context_fit(
            preference_text,
            preferences_used,
            bundle.get("work_context", []),
        ),
        "ai_impact_preference": _ai_preference_score(
            preference_text,
            resolution.get("onet_soc_code"),
        ),
    }
    weights = {
        "riasec_interest_fit": 0.28,
        "job_zone_fit": 0.18,
        "task_work_activity_similarity": 0.22,
        "skills_knowledge_similarity": 0.17,
        "work_context_fit": 0.10,
        "ai_impact_preference": 0.05,
    }
    total = sum(components[key] * weight for key, weight in weights.items())
    explanation = _ranking_explanation(match, components, target_interests, target_zone)
    return {
        **match,
        "resolved_onet_title": resolution.get("resolved_onet_title"),
        "onet_soc_code": resolution.get("onet_soc_code"),
        "score": round(total, 4),
        "score_components": {key: round(value, 4) for key, value in components.items()},
        "ranking_explanation": explanation,
    }


def _occupation_signal_bundle(conn: Any, soc_code: Any) -> dict[str, list[str]]:
    code = _clean_text(soc_code)
    if not code:
        return {}
    return {
        "tasks": _fetch_column(
            conn,
            """
            SELECT task
            FROM task_statements
            WHERE onetsoc_code = ?
            ORDER BY task_id
            LIMIT 30
            """,
            [code],
        ),
        "work_activities": _fetch_element_names(conn, "work_activities", code, 25),
        "skills": _fetch_element_names(conn, "essential_skills", code, 25),
        "knowledge": _fetch_element_names(conn, "knowledge", code, 25),
        "work_context": _fetch_work_context(conn, code, 25),
    }


def _fetch_element_names(conn: Any, table_name: str, code: str, limit: int) -> list[str]:
    if not _table_exists(conn, table_name):
        return []
    return _fetch_column(
        conn,
        f"""
        SELECT cmr.element_name
        FROM {table_name} AS item
        JOIN content_model_reference AS cmr
            ON item.element_id = cmr.element_id
        WHERE item.onetsoc_code = ?
        ORDER BY item.data_value DESC, cmr.element_name
        LIMIT ?
        """,
        [code, limit],
    )


def _fetch_work_context(conn: Any, code: str, limit: int) -> list[str]:
    if not _table_exists(conn, "work_context"):
        return []
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
        text = _clean_text(f"{name}: {category}" if category else name)
        if text:
            values.append(text)
    return values


def _fetch_column(conn: Any, query: str, params: list[Any]) -> list[str]:
    rows = conn.execute(query, params).fetchall()
    return [_clean_text(row[0]) for row in rows if _clean_text(row[0])]


def _table_exists(conn: Any, table_name: str) -> bool:
    return bool(
        conn.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = ?
            LIMIT 1
            """,
            [table_name],
        ).fetchone()
    )


def _target_interests(profile_result: dict[str, Any], final_refinement: dict[str, Any]) -> list[str]:
    values = (
        (final_refinement.get("career_matching_guidance") or {}).get("prioritize_interests")
        or final_refinement.get("refined_top_interests")
        or profile_result.get("final_top_interests")
        or profile_result.get("initial_top_interests")
        or []
    )
    result = []
    for value in values:
        try:
            result.append(canonical_interest(value))
        except ValueError:
            continue
    return result[:3] or list(profile_result.get("initial_top_interests") or [])[:3]


def _target_job_zone(profile_result: dict[str, Any], final_refinement: dict[str, Any]) -> int:
    guidance = final_refinement.get("career_matching_guidance") or {}
    try:
        zone = int(guidance.get("job_zone_to_use") or profile_result.get("future_job_zone") or 0)
    except (TypeError, ValueError):
        zone = 0
    return zone if zone in {1, 2, 3, 4, 5} else int(profile_result.get("future_job_zone") or 3)


def _riasec_interest_fit(candidate_interest: Any, target_interests: list[str], scores: dict[str, Any]) -> float:
    try:
        interest = canonical_interest(candidate_interest)
    except ValueError:
        return 0.0
    try:
        raw_score = float(scores.get(interest, 0))
    except (TypeError, ValueError):
        raw_score = 0.0
    raw_component = min(1.0, max(0.0, raw_score / 10.0))
    if interest in target_interests:
        rank_component = 1.0 - 0.16 * target_interests.index(interest)
    else:
        rank_component = 0.25
    return max(0.0, min(1.0, 0.55 * rank_component + 0.45 * raw_component))


def _job_zone_fit(candidate_zone: Any, target_zone: int) -> float:
    try:
        zone = int(candidate_zone)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, 1.0 - abs(zone - target_zone) / 4.0)


def _text_similarity(query: str, documents: list[str]) -> float:
    query_tokens = _tokens(query)
    if not query_tokens or not documents:
        return 0.0
    best = 0.0
    for document in documents:
        doc_tokens = _tokens(document)
        if not doc_tokens:
            continue
        overlap = len(query_tokens & doc_tokens)
        if not overlap:
            continue
        score = overlap / max(6, min(len(query_tokens), len(doc_tokens)))
        best = max(best, min(1.0, score))
    return best


def _work_context_fit(
    preference_text: str,
    preferences_used: list[str],
    work_context: list[str],
) -> float:
    if not preferences_used:
        return 0.5
    context_text = " ".join(work_context).lower()
    score = _text_similarity(preference_text, work_context)
    if "outdoor or field work" in preferences_used and "outdoors" in context_text:
        score = max(score, 0.85)
    if "structured independent work" in preferences_used and "freedom to make decisions" in context_text:
        score = max(score, 0.70)
    if "lower social intensity" in preferences_used and "contact with others" in context_text:
        score = min(score, 0.35)
    return max(0.0, min(1.0, score if score else 0.35))


def _ai_preference_score(preference_text: str, soc_code: Any) -> float:
    text = preference_text.lower()
    if not any(term in text for term in ("ai", "automation", "technology", "future proof", "stability")):
        return 0.5
    try:
        from career_rag.ip_ai_impact import _local_ai_rows_for_soc
    except Exception:
        return 0.5
    values = []
    for row in _local_ai_rows_for_soc(str(soc_code or "")):
        if str(row.get("doc_type") or "") != "ai_occupation_impact":
            continue
        try:
            values.append(float(row.get("metric_value")))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.5
    exposure = max(values)
    if any(term in text for term in ("stability", "stable", "secure", "worry", "concern", "automation")):
        return max(0.0, min(1.0, 1.0 - exposure))
    return max(0.0, min(1.0, exposure))


def _ranking_explanation(
    match: dict[str, Any],
    components: dict[str, float],
    target_interests: list[str],
    target_zone: int,
) -> dict[str, Any]:
    effects = []
    if components["task_work_activity_similarity"] >= 0.25:
        effects.append("Follow-up preferences matched this occupation's O*NET tasks or work activities.")
    if components["skills_knowledge_similarity"] >= 0.25:
        effects.append("Follow-up preferences matched its O*NET skills or knowledge areas.")
    if components["work_context_fit"] >= 0.65:
        effects.append("Work-context signals align with the follow-up constraints.")
    if components["job_zone_fit"] < 0.75:
        effects.append(f"Job Zone {match.get('job_zone')} is farther from the follow-up target zone {target_zone}.")
    if components["task_work_activity_similarity"] < 0.10 and components["skills_knowledge_similarity"] < 0.10:
        effects.append("Few O*NET task, skill, or knowledge terms matched the follow-up preferences.")
    return {
        "original_profile": _original_profile_reason(match, match.get("source_match_group")),
        "followup_effects": effects,
        "target_interests": target_interests,
        "target_job_zone": target_zone,
    }


def _original_profile_reason(match: dict[str, Any], group: Any) -> str:
    interest = _clean_text(match.get("interest"))
    zone = _clean_text(match.get("job_zone"))
    group_text = _clean_text(group).replace("_", " ")
    return f"Originally appeared in the O*NET Interest Profiler candidate pool for {interest} / Job Zone {zone} ({group_text})."


def _ranking_payload(
    initial_matches: list[dict[str, Any]],
    final_matches: list[dict[str, Any]],
    profile_result: dict[str, Any],
    followup_refinement: dict[str, Any] | None,
    preferences_used: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    debug = {
        "initial_code": profile_result.get("initial_code") or profile_result.get("initial_holland_code"),
        "final_code": profile_result.get("final_code") or profile_result.get("final_holland_code"),
        "followup_refinement": followup_refinement,
        "preferences_used": preferences_used,
        "top_10_initial_matches": _debug_match_rows(initial_matches[:10]),
        "top_10_refined_matches": _debug_match_rows(final_matches[:10]),
        "score_components": [
            {
                "career_title": item.get("career_title"),
                "score": item.get("score"),
                "score_components": item.get("score_components"),
                "ranking_explanation": item.get("ranking_explanation"),
            }
            for item in final_matches[:10]
        ],
        "validation_warnings": warnings,
    }
    return {
        "preferences_used": preferences_used,
        "refined_career_matches": final_matches,
        "final_ranked_matches": final_matches,
        "refinement_debug": debug,
        "validation_warnings": warnings,
    }


def _debug_match_rows(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "career_title": item.get("career_title"),
            "interest": item.get("interest"),
            "job_zone": item.get("job_zone"),
            "score": item.get("score"),
        }
        for item in matches
    ]


def _empty_score_components() -> dict[str, float]:
    return {
        "riasec_interest_fit": 1.0,
        "job_zone_fit": 1.0,
        "task_work_activity_similarity": 0.0,
        "skills_knowledge_similarity": 0.0,
        "work_context_fit": 0.5,
        "ai_impact_preference": 0.5,
    }


def _preference_text(followup_refinement: dict[str, Any] | None, preferences_used: list[str]) -> str:
    final_refinement = (followup_refinement or {}).get("final_refinement") or {}
    values = list(preferences_used)
    values.extend(str(item) for item in final_refinement.get("key_sub_preferences") or [])
    values.append(str(final_refinement.get("future_vision_summary") or ""))
    values.extend(str(item) for item in final_refinement.get("concerns_noted") or [])
    for item in (followup_refinement or {}).get("questions_asked") or []:
        values.append(str(item.get("answer") or ""))
    return " ".join(values)


def _title_order(matches: list[dict[str, Any]]) -> list[str]:
    return [_normalize(item.get("career_title")) for item in matches]


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _normalize(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
