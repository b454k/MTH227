"""Semantic O*NET + AI-impact comparison report for the Interest Profiler flow."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from career_rag.ai_exposure_utils import one_line
from career_rag.ai_impact_retriever import retrieve_ai_impact
from career_rag.config import ONET_DUCKDB_PATH
from career_rag.ip_followup_agent import OpenAIChatProvider
from career_rag.retriever import OnetRetriever


SEMANTIC_REPORT_METHOD = "semantic_onet_ai_impact_report"
MAX_SNIPPET_CHARS = 260
FOLLOWUP_QUERY_WEIGHT = 1.65
PREFERENCE_QUERY_WEIGHT = 1.35
PROFILE_QUERY_WEIGHT = 0.75
ONET_MAX_ROWS_PER_TITLE = 2
AI_MAX_ROWS_PER_TITLE = 2
JOB_ZONE_MATCH_BOOST = 0.08
JOB_ZONE_MISMATCH_PENALTY = 0.24

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "but",
    "can",
    "career",
    "careers",
    "for",
    "from",
    "have",
    "into",
    "like",
    "more",
    "that",
    "the",
    "their",
    "this",
    "with",
    "work",
    "would",
    "your",
}

DOMAIN_EXPANSIONS = {
    "finance": ["financial", "investment", "banking", "market", "markets", "accounting", "economics"],
    "financial": ["finance", "investment", "banking", "market", "markets", "accounting", "economics"],
    "data": ["analytics", "analysis", "statistics", "modeling"],
    "business": ["management", "strategy", "operations", "market"],
}


def build_semantic_retrieval_report(
    profile_used: dict[str, Any],
    citation_manager: Any,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build a source-cited report from semantic O*NET and AI-impact retrieval."""
    queries = _build_semantic_queries(profile_used)
    query = queries[0]["query"] if queries else _build_profile_query(profile_used)
    followup_terms = _followup_terms(profile_used)
    allowed_zones = _allowed_job_zones(profile_used)
    onet_rows, onet_error = _retrieve_onet_queries(queries, followup_terms, allowed_zones, top_k=top_k)
    ai_rows, ai_error = _retrieve_ai_queries(queries, followup_terms, allowed_zones, top_k=max(6, min(top_k, 10)))

    cited_onet = [_cite_onet_row(row, citation_manager) for row in onet_rows]
    cited_ai = [_cite_ai_row(row, citation_manager) for row in ai_rows]
    career_signals = _career_signals(cited_onet, cited_ai)
    readable_sections = _generate_semantic_sections(
        profile_used=profile_used,
        career_signals=career_signals,
        onet_rows=cited_onet,
        ai_rows=cited_ai,
    )

    status = "ok" if cited_onet or cited_ai else "unavailable"
    return {
        "method": SEMANTIC_REPORT_METHOD,
        "status": status,
        "query": query,
        "queries": queries,
        "followup_terms_used": followup_terms[:20],
        "job_zones_used": sorted(allowed_zones),
        "summary": readable_sections.get("summary", ""),
        "relevant_careers_explanation": readable_sections.get("relevant_careers_explanation", ""),
        "technology_ai_role": readable_sections.get("technology_ai_role", ""),
        "takeaways": readable_sections.get("takeaways", []),
        "semantic_career_signals": career_signals,
        "retrieved_onet": cited_onet,
        "retrieved_ai_impact": cited_ai,
        "errors": {
            "onet": onet_error,
            "ai_impact": ai_error,
        },
    }


def _build_profile_query(profile_used: dict[str, Any]) -> str:
    interests = ", ".join(str(item) for item in profile_used.get("final_top_interests") or [])
    preferences = ", ".join(str(item) for item in profile_used.get("sub_preferences") or [])
    code = str(profile_used.get("final_holland_code") or "")
    parts = [
        "career options, work tasks, skills, education, work context, and AI exposure",
        f"Holland code {code}" if code else "",
        f"interests {interests}" if interests else "",
        f"current Job Zone {profile_used.get('current_job_zone')}",
        f"future Job Zone {profile_used.get('future_job_zone')}",
        f"user preferences {preferences}" if preferences else "",
    ]
    return one_line(". ".join(part for part in parts if part))


def _build_semantic_queries(profile_used: dict[str, Any]) -> list[dict[str, Any]]:
    followup_text = one_line(profile_used.get("followup_answer_text"))
    preferences = ", ".join(str(item) for item in profile_used.get("sub_preferences") or [])
    base_query = _build_profile_query(profile_used)
    queries = []
    if followup_text or preferences:
        queries.append(
            {
                "label": "followup_answers",
                "weight": FOLLOWUP_QUERY_WEIGHT,
                "query": one_line(
                    "career options matching the user's open ended follow-up answers. "
                    f"Only include occupations aligned with current Job Zone {profile_used.get('current_job_zone')} "
                    f"or future Job Zone {profile_used.get('future_job_zone')}. "
                    f"Follow-up answers: {followup_text}. Preferences: {preferences}."
                ),
            }
        )
    if preferences:
        queries.append(
            {
                "label": "preferences",
                "weight": PREFERENCE_QUERY_WEIGHT,
                "query": one_line(
                    "career options, occupations, tasks, skills, and AI exposure matching these user preferences "
                    f"and current/future Job Zones {profile_used.get('current_job_zone')}/{profile_used.get('future_job_zone')}: "
                    + preferences
                ),
            }
        )
    queries.append({"label": "interest_profiler", "weight": PROFILE_QUERY_WEIGHT, "query": base_query})
    return [query for query in queries if query.get("query")]


def _retrieve_onet_queries(
    queries: list[dict[str, Any]],
    followup_terms: list[str],
    allowed_zones: set[int],
    top_k: int,
) -> tuple[list[dict[str, Any]], str]:
    try:
        retriever = OnetRetriever()
    except Exception as exc:
        return [], str(exc)
    rows = []
    errors = []
    for query_spec in queries:
        try:
            for row in retriever.retrieve_smart(str(query_spec["query"]), k=max(top_k * 2, 12)):
                rows.append(_score_row(row, query_spec, followup_terms, allowed_zones))
        except Exception as exc:
            errors.append(f"{query_spec.get('label')}: {exc}")
    return _select_diverse_rows(rows, top_k, title_getter=_onet_row_title, max_per_title=ONET_MAX_ROWS_PER_TITLE), "; ".join(errors)


def _retrieve_ai_queries(
    queries: list[dict[str, Any]],
    followup_terms: list[str],
    allowed_zones: set[int],
    top_k: int,
) -> tuple[list[dict[str, Any]], str]:
    rows = []
    errors = []
    for query_spec in queries:
        try:
            for row in retrieve_ai_impact(str(query_spec["query"]), top_k=max(top_k * 2, 12)):
                rows.append(_score_row(row, query_spec, followup_terms, allowed_zones))
        except Exception as exc:
            errors.append(f"{query_spec.get('label')}: {exc}")
    return _select_diverse_rows(rows, top_k, title_getter=_ai_row_title, max_per_title=AI_MAX_ROWS_PER_TITLE), "; ".join(errors)


def _retrieve_onet(query: str, top_k: int) -> tuple[list[dict[str, Any]], str]:
    try:
        retriever = OnetRetriever()
        return retriever.retrieve_smart(query, k=top_k), ""
    except Exception as exc:
        return [], str(exc)


def _retrieve_ai(query: str, top_k: int) -> tuple[list[dict[str, Any]], str]:
    try:
        return retrieve_ai_impact(query, top_k=top_k), ""
    except Exception as exc:
        return [], str(exc)


def _score_row(
    row: dict[str, Any],
    query_spec: dict[str, Any],
    followup_terms: list[str],
    allowed_zones: set[int],
) -> dict[str, Any]:
    text = _row_text(row)
    overlap = _term_overlap(text, followup_terms)
    base_score = _score_float(row.get("score"))
    query_weight = _positive_float(query_spec.get("weight"), default=1.0)
    overlap_boost = min(0.22, overlap * 0.035)
    zone = _row_job_zone(row)
    zone_adjustment = _job_zone_adjustment(zone, allowed_zones)
    adjusted_score = min(1.0, max(0.0, base_score + overlap_boost + zone_adjustment))
    return {
        **row,
        "query_label": query_spec.get("label"),
        "query_weight": query_spec.get("weight"),
        "followup_overlap": overlap,
        "job_zone": zone,
        "job_zone_match": bool(zone and zone in allowed_zones),
        "adjusted_score": adjusted_score,
        "retrieval_rank_score": adjusted_score * query_weight,
    }


def _select_diverse_rows(
    rows: list[dict[str, Any]],
    top_k: int,
    title_getter: Any,
    max_per_title: int,
) -> list[dict[str, Any]]:
    def rank_key(row: dict[str, Any]) -> tuple[float, float, float, int, float]:
        return (
            _positive_float(row.get("retrieval_rank_score"), default=0.0),
            _score_float(row.get("adjusted_score")),
            _positive_float(row.get("query_weight"), default=1.0),
            int(row.get("followup_overlap") or 0),
            _score_float(row.get("score")),
        )

    matched = [row for row in rows if row.get("job_zone_match")]
    unknown = [row for row in rows if row.get("job_zone") in (None, "")]
    mismatched = [
        row
        for row in rows
        if row not in matched and row not in unknown
    ]
    ranked = []
    for group in (matched, unknown, mismatched):
        ranked.extend(sorted(group, key=rank_key, reverse=True))
    selected = []
    seen_ids = set()
    title_counts: dict[str, int] = {}
    for row in ranked:
        row_id = (str(row.get("collection") or ""), str(row.get("id") or row.get("doc_id") or ""))
        if row_id in seen_ids:
            continue
        title = title_getter(row).lower()
        if title and title_counts.get(title, 0) >= max_per_title:
            continue
        selected.append(row)
        seen_ids.add(row_id)
        if title:
            title_counts[title] = title_counts.get(title, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def _row_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    parts = [
        row.get("text"),
        metadata.get("occupation_title"),
        metadata.get("title"),
        metadata.get("task_text"),
        metadata.get("section"),
        metadata.get("doc_type"),
        metadata.get("impact_type"),
        metadata.get("metric_name"),
    ]
    return " ".join(one_line(part) for part in parts if one_line(part)).lower()


def _onet_row_title(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return one_line(metadata.get("occupation_title") or metadata.get("title") or metadata.get("section"))


def _ai_row_title(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return one_line(metadata.get("occupation_title") or metadata.get("title"))


def _allowed_job_zones(profile_used: dict[str, Any]) -> set[int]:
    zones = set()
    for key in ("current_job_zone", "future_job_zone", "current_zone", "future_zone"):
        try:
            zone = int(profile_used.get(key))
        except (TypeError, ValueError):
            continue
        if 1 <= zone <= 5:
            zones.add(zone)
    return zones


def _row_job_zone(row: dict[str, Any]) -> int | None:
    metadata = row.get("metadata") or {}
    for key in ("job_zone", "zone"):
        try:
            zone = int(metadata.get(key) or row.get(key))
        except (TypeError, ValueError):
            continue
        if 1 <= zone <= 5:
            return zone
    soc = one_line(
        metadata.get("onet_soc_code")
        or metadata.get("soc_code")
        or row.get("soc_code")
        or row.get("onet_soc_code")
    )
    if not soc:
        return None
    return _job_zone_for_soc(soc)


def _job_zone_adjustment(zone: int | None, allowed_zones: set[int]) -> float:
    if not allowed_zones or zone is None:
        return 0.0
    if zone in allowed_zones:
        return JOB_ZONE_MATCH_BOOST
    return -JOB_ZONE_MISMATCH_PENALTY


@lru_cache(maxsize=4096)
def _job_zone_for_soc(onet_soc_code: str) -> int | None:
    try:
        import duckdb
    except ImportError:
        return None
    if not ONET_DUCKDB_PATH.exists():
        return None
    try:
        conn = duckdb.connect(str(ONET_DUCKDB_PATH), read_only=True)
        try:
            row = conn.execute(
                "SELECT job_zone FROM job_zones WHERE onetsoc_code = ? LIMIT 1",
                [onet_soc_code],
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        return None
    if not row:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def _followup_terms(profile_used: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            one_line(profile_used.get("followup_answer_text")),
            " ".join(str(item) for item in profile_used.get("sub_preferences") or []),
            one_line(profile_used.get("future_vision_summary")),
            " ".join(str(item) for item in profile_used.get("concerns_noted") or []),
        ]
    ).lower()
    terms = []
    seen = set()
    for raw_term in re.findall(r"[a-z][a-z0-9+.-]{2,}", text):
        term = raw_term.strip(".-+")
        if not term or term in STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        for expanded in DOMAIN_EXPANSIONS.get(term, []):
            if expanded not in seen:
                seen.add(expanded)
                terms.append(expanded)
    return terms


def _term_overlap(text: str, terms: list[str]) -> int:
    if not text or not terms:
        return 0
    return sum(1 for term in terms if term in text)


def _cite_onet_row(row: dict[str, Any], citation_manager: Any) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    title = (
        one_line(metadata.get("occupation_title"))
        or one_line(metadata.get("title"))
        or one_line(metadata.get("section"))
        or "O*NET semantic document"
    )
    soc = one_line(metadata.get("onet_soc_code") or metadata.get("soc_code"))
    section = one_line(metadata.get("section") or metadata.get("doc_type") or row.get("collection"))
    source_id = citation_manager.add_source(
        title=f"O*NET semantic retrieval - {title}",
        source_type="onet_semantic",
        url=_onet_summary_url(soc) if soc else None,
        local_file="O*NET Chroma collections",
        retrieved_section=section or "semantic O*NET document",
        note=f"O*NET-SOC {soc}" if soc else "Retrieved by semantic profile query.",
    )
    return {
        "source_id": source_id,
        "title": title,
        "soc_code": soc,
        "job_zone": row.get("job_zone"),
        "job_zone_match": row.get("job_zone_match"),
        "section": section,
        "collection": one_line(row.get("collection")),
        "score": row.get("score"),
        "adjusted_score": row.get("adjusted_score"),
        "query_label": row.get("query_label"),
        "query_weight": row.get("query_weight"),
        "snippet": _short_snippet(row.get("text")),
    }


def _cite_ai_row(row: dict[str, Any], citation_manager: Any) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    source_name = one_line(metadata.get("source_name") or metadata.get("source_id")) or "AI-impact evidence"
    occupation = one_line(metadata.get("occupation_title"))
    soc = one_line(metadata.get("soc_code"))
    task = one_line(metadata.get("task_text"))
    impact_type = one_line(metadata.get("impact_type") or metadata.get("metric_name"))
    source_id = citation_manager.add_source(
        title=f"AI-impact semantic retrieval - {source_name}",
        source_type="ai_impact_semantic",
        local_file="AI-impact Chroma collection",
        retrieved_section=impact_type or "semantic AI-impact evidence",
        note=f"{occupation} {soc}".strip() or "Retrieved by semantic profile query.",
    )
    return {
        "source_id": source_id,
        "occupation": occupation,
        "soc_code": soc,
        "job_zone": row.get("job_zone"),
        "job_zone_match": row.get("job_zone_match"),
        "task": task,
        "impact_type": impact_type,
        "score": row.get("score"),
        "reranked_score": row.get("reranked_score"),
        "adjusted_score": row.get("adjusted_score"),
        "query_label": row.get("query_label"),
        "query_weight": row.get("query_weight"),
        "snippet": _short_snippet(row.get("text")),
    }


def _career_signals(
    onet_rows: list[dict[str, Any]],
    ai_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    weighted_scores: dict[str, float] = {}
    weights: dict[str, float] = {}
    source_ids: dict[str, list[int]] = {}
    soc_codes: dict[str, str] = {}
    job_zones: dict[str, int] = {}
    for row in onet_rows:
        title = one_line(row.get("title"))
        if not title:
            continue
        score = _score_float(row.get("adjusted_score") or row.get("score"))
        weight = 2.0 * _positive_float(row.get("query_weight"), default=1.0)
        weighted_scores[title] = weighted_scores.get(title, 0.0) + (weight * score)
        weights[title] = weights.get(title, 0.0) + weight
        source_ids.setdefault(title, []).append(int(row["source_id"]))
        if row.get("soc_code"):
            soc_codes[title] = str(row["soc_code"])
        job_zone = row.get("job_zone")
        if job_zone:
            job_zones[title] = job_zone
    for row in ai_rows:
        title = one_line(row.get("occupation"))
        if not title:
            continue
        score = _score_float(row.get("adjusted_score") or row.get("score"))
        weight = _positive_float(row.get("query_weight"), default=1.0)
        weighted_scores[title] = weighted_scores.get(title, 0.0) + (weight * score)
        weights[title] = weights.get(title, 0.0) + weight
        source_ids.setdefault(title, []).append(int(row["source_id"]))
        if row.get("soc_code"):
            soc_codes[title] = str(row["soc_code"])
        job_zone = row.get("job_zone")
        if job_zone:
            job_zones[title] = job_zone

    signals = []
    ranked_titles = sorted(
        weighted_scores,
        key=lambda title: weighted_scores[title] / max(weights.get(title, 1.0), 1.0),
        reverse=True,
    )
    for title in ranked_titles[:8]:
        ids = []
        seen = set()
        for source_id in source_ids.get(title, []):
            if source_id not in seen:
                seen.add(source_id)
                ids.append(source_id)
        signal = weighted_scores[title] / max(weights.get(title, 1.0), 1.0)
        signals.append(
            {
                "title": title,
                "soc_code": soc_codes.get(title, ""),
                "job_zone": job_zones.get(title),
                "semantic_signal": round(signal, 3),
                "source_ids": ids[:4],
            }
        )
    return signals


def _generate_semantic_sections(
    profile_used: dict[str, Any],
    career_signals: list[dict[str, Any]],
    onet_rows: list[dict[str, Any]],
    ai_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    provider = OpenAIChatProvider()
    if provider.available and (onet_rows or ai_rows):
        try:
            return _llm_semantic_sections(provider, profile_used, career_signals, onet_rows, ai_rows)
        except Exception:
            pass
    return _fallback_semantic_sections(career_signals, onet_rows, ai_rows)


def _llm_semantic_sections(
    provider: OpenAIChatProvider,
    profile_used: dict[str, Any],
    career_signals: list[dict[str, Any]],
    onet_rows: list[dict[str, Any]],
    ai_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    onet_context = "\n".join(
        f"[{row['source_id']}] {row.get('title')} {row.get('section')}: {row.get('snippet')}"
        for row in onet_rows[:8]
    )
    ai_context = "\n".join(
        f"[{row['source_id']}] {row.get('occupation')} {row.get('impact_type')}: {row.get('snippet')}"
        for row in ai_rows[:8]
    )
    signal_context = ", ".join(
        f"{item['title']} Job Zone {item.get('job_zone') or 'unknown'} "
        f"({' '.join(f'[{sid}]' for sid in item.get('source_ids') or [])})"
        for item in career_signals[:6]
    )
    prompt = (
        "Create a comparison report from semantic retrieval only. Do not use the scoring-based "
        "top matches. Treat the user's follow-up answers and stated preferences as the primary "
        "matching signal; keep the selected current/future Job Zones as a preparation constraint. "
        "If a retrieved role is outside the user's selected Job Zones, do not emphasize it. "
        "Use plain language and cite factual sentences with source markers.\n\n"
        f"Final profile: {profile_used}\n\n"
        f"Semantic career signals: {signal_context}\n\n"
        f"O*NET semantic evidence:\n{onet_context}\n\n"
        f"AI-impact semantic evidence:\n{ai_context}\n\n"
        "Return exactly:\n"
        "Relevant Careers: one short paragraph explaining which careers surfaced and why.\n"
        "Role of Technology and AI: one short paragraph explaining AI/technology implications from the retrieved evidence.\n"
        "Takeaways:\n"
        "- first practical takeaway\n"
        "- second practical takeaway\n"
        "- third practical takeaway"
    )
    output = provider.complete(
        [
            {
                "role": "system",
                "content": "You write careful, source-grounded career report comparisons.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return _parse_section_output(output)


def _fallback_semantic_sections(
    career_signals: list[dict[str, Any]],
    onet_rows: list[dict[str, Any]],
    ai_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    markers = _markers([*(row.get("source_id") for row in onet_rows[:2]), *(row.get("source_id") for row in ai_rows[:2])])
    names = [item["title"] for item in career_signals[:4]]
    if names:
        relevant_careers = (
            f"The semantic retrieval report surfaces {', '.join(names)} as careers whose O*NET "
            f"documents and AI-impact evidence are close to the final profile query {markers}."
        )
    else:
        relevant_careers = "Semantic O*NET and AI-impact retrieval did not return enough career evidence to summarize."
    technology_ai_role = (
        f"The AI-impact rows are used to show where technology may affect tasks or work activities, "
        f"while O*NET rows are used to keep the career interpretation grounded in occupation evidence {markers}."
    )
    takeaways = [
        f"Compare these semantic career signals against the scoring-system top matches; agreement suggests a stronger recommendation {markers}.",
        f"Semantic retrieval is useful for surfacing related evidence, but it can mix careers if the profile query is broad {markers}.",
        f"Use the cited O*NET and AI-impact rows as evidence, not as a replacement for the structured score ranking {markers}.",
    ]
    return {
        "summary": relevant_careers,
        "relevant_careers_explanation": relevant_careers,
        "technology_ai_role": technology_ai_role,
        "takeaways": takeaways,
    }


def _parse_section_output(output: str) -> dict[str, Any]:
    text = str(output or "").strip()
    careers_match = re.search(
        r"Relevant Careers:\s*(.*?)(?:\n\s*Role of Technology and AI:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    ai_match = re.search(
        r"Role of Technology and AI:\s*(.*?)(?:\n\s*Takeaways:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    takeaways_match = re.search(r"Takeaways:\s*(.*)\Z", text, flags=re.IGNORECASE | re.DOTALL)
    careers = one_line(careers_match.group(1) if careers_match else text)
    technology_ai_role = one_line(ai_match.group(1) if ai_match else "")
    takeaways_section = takeaways_match.group(1) if takeaways_match else ""
    takeaways = [
        one_line(match.group(1))
        for match in re.finditer(r"^\s*-\s*(.+)$", takeaways_section, flags=re.MULTILINE)
    ]
    return {
        "summary": careers,
        "relevant_careers_explanation": careers,
        "technology_ai_role": technology_ai_role,
        "takeaways": takeaways[:4],
    }


def _markers(source_ids: list[Any]) -> str:
    seen = []
    for value in source_ids:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number not in seen:
            seen.append(number)
    return " ".join(f"[{number}]" for number in seen)


def _score_float(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def _positive_float(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, score)


def _short_snippet(value: Any) -> str:
    text = one_line(value)
    if len(text) <= MAX_SNIPPET_CHARS:
        return text
    return text[: MAX_SNIPPET_CHARS - 3].rstrip() + "..."


def _onet_summary_url(onet_soc_code: str) -> str:
    return f"https://www.onetonline.org/link/summary/{str(onet_soc_code).strip()}"
