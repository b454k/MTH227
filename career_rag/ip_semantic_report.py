"""Semantic O*NET + AI-impact comparison report for the Interest Profiler flow."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from career_rag.ai_exposure_utils import one_line
from career_rag.ai_impact_retriever import retrieve_ai_impact
from career_rag.ip_followup_agent import OpenAIChatProvider
from career_rag.retriever import OnetRetriever


SEMANTIC_REPORT_METHOD = "semantic_onet_ai_impact_report"
MAX_SNIPPET_CHARS = 260


def build_semantic_retrieval_report(
    profile_used: dict[str, Any],
    citation_manager: Any,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build a source-cited report from semantic O*NET and AI-impact retrieval."""
    query = _build_profile_query(profile_used)
    onet_rows, onet_error = _retrieve_onet(query, top_k=top_k)
    ai_rows, ai_error = _retrieve_ai(query, top_k=max(6, min(top_k, 10)))

    cited_onet = [_cite_onet_row(row, citation_manager) for row in onet_rows]
    cited_ai = [_cite_ai_row(row, citation_manager) for row in ai_rows]
    career_signals = _career_signals(cited_onet, cited_ai)
    summary, takeaways = _generate_semantic_summary(
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
        "summary": summary,
        "takeaways": takeaways,
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
        "section": section,
        "collection": one_line(row.get("collection")),
        "score": row.get("score"),
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
        "task": task,
        "impact_type": impact_type,
        "score": row.get("reranked_score") or row.get("score"),
        "snippet": _short_snippet(row.get("text")),
    }


def _career_signals(
    onet_rows: list[dict[str, Any]],
    ai_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    source_ids: dict[str, list[int]] = {}
    soc_codes: dict[str, str] = {}
    for row in onet_rows:
        title = one_line(row.get("title"))
        if not title:
            continue
        counts[title] += 2
        source_ids.setdefault(title, []).append(int(row["source_id"]))
        if row.get("soc_code"):
            soc_codes[title] = str(row["soc_code"])
    for row in ai_rows:
        title = one_line(row.get("occupation"))
        if not title:
            continue
        counts[title] += 1
        source_ids.setdefault(title, []).append(int(row["source_id"]))
        if row.get("soc_code"):
            soc_codes[title] = str(row["soc_code"])

    signals = []
    for title, count in counts.most_common(8):
        ids = []
        seen = set()
        for source_id in source_ids.get(title, []):
            if source_id not in seen:
                seen.add(source_id)
                ids.append(source_id)
        signals.append(
            {
                "title": title,
                "soc_code": soc_codes.get(title, ""),
                "semantic_signal": count,
                "source_ids": ids[:4],
            }
        )
    return signals


def _generate_semantic_summary(
    profile_used: dict[str, Any],
    career_signals: list[dict[str, Any]],
    onet_rows: list[dict[str, Any]],
    ai_rows: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    provider = OpenAIChatProvider()
    if provider.available and (onet_rows or ai_rows):
        try:
            return _llm_semantic_summary(provider, profile_used, career_signals, onet_rows, ai_rows)
        except Exception:
            pass
    return _fallback_semantic_summary(career_signals, onet_rows, ai_rows)


def _llm_semantic_summary(
    provider: OpenAIChatProvider,
    profile_used: dict[str, Any],
    career_signals: list[dict[str, Any]],
    onet_rows: list[dict[str, Any]],
    ai_rows: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    onet_context = "\n".join(
        f"[{row['source_id']}] {row.get('title')} {row.get('section')}: {row.get('snippet')}"
        for row in onet_rows[:8]
    )
    ai_context = "\n".join(
        f"[{row['source_id']}] {row.get('occupation')} {row.get('impact_type')}: {row.get('snippet')}"
        for row in ai_rows[:8]
    )
    signal_context = ", ".join(
        f"{item['title']} ({' '.join(f'[{sid}]' for sid in item.get('source_ids') or [])})"
        for item in career_signals[:6]
    )
    prompt = (
        "Create a comparison report from semantic retrieval only. Do not use the scoring-based "
        "top matches. Explain what careers and AI-impact themes surfaced from the retrieved "
        "O*NET and AI-impact chunks. Cite factual sentences with source markers.\n\n"
        f"Final profile: {profile_used}\n\n"
        f"Semantic career signals: {signal_context}\n\n"
        f"O*NET semantic evidence:\n{onet_context}\n\n"
        f"AI-impact semantic evidence:\n{ai_context}\n\n"
        "Return exactly:\n"
        "Summary: one paragraph, 3-5 sentences.\n"
        "Takeaways:\n"
        "- first comparison takeaway\n"
        "- second comparison takeaway\n"
        "- third comparison takeaway"
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
    return _parse_summary_output(output)


def _fallback_semantic_summary(
    career_signals: list[dict[str, Any]],
    onet_rows: list[dict[str, Any]],
    ai_rows: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    markers = _markers([*(row.get("source_id") for row in onet_rows[:2]), *(row.get("source_id") for row in ai_rows[:2])])
    names = [item["title"] for item in career_signals[:4]]
    if names:
        summary = (
            f"The semantic retrieval report surfaces {', '.join(names)} as careers whose O*NET "
            f"documents and AI-impact evidence are close to the final profile query {markers}."
        )
    else:
        summary = "Semantic O*NET and AI-impact retrieval did not return enough evidence to summarize."
    takeaways = [
        f"Compare these semantic career signals against the scoring-system top matches; agreement suggests a stronger recommendation {markers}.",
        f"Semantic retrieval is useful for surfacing related evidence, but it can mix careers if the profile query is broad {markers}.",
        f"Use the cited O*NET and AI-impact rows as evidence, not as a replacement for the structured score ranking {markers}.",
    ]
    return summary, takeaways


def _parse_summary_output(output: str) -> tuple[str, list[str]]:
    text = str(output or "").strip()
    summary_match = re.search(r"Summary:\s*(.*?)(?:\n\s*Takeaways:|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    summary = one_line(summary_match.group(1) if summary_match else text)
    takeaways_section = text[summary_match.end():] if summary_match else ""
    takeaways = [
        one_line(match.group(1))
        for match in re.finditer(r"^\s*-\s*(.+)$", takeaways_section, flags=re.MULTILINE)
    ]
    return summary, takeaways[:4]


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


def _short_snippet(value: Any) -> str:
    text = one_line(value)
    if len(text) <= MAX_SNIPPET_CHARS:
        return text
    return text[: MAX_SNIPPET_CHARS - 3].rstrip() + "..."


def _onet_summary_url(onet_soc_code: str) -> str:
    return f"https://www.onetonline.org/link/summary/{str(onet_soc_code).strip()}"
