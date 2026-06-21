"""Semantic research RAG for final-report future impact summaries."""

from __future__ import annotations

import re
from typing import Any

from career_rag.ai_exposure_utils import one_line
from career_rag.ai_impact_retriever import retrieve_research_inference
from career_rag.ip_followup_agent import OpenAIChatProvider


FUTURE_IMPACT_METHOD = "semantic_research_rag"
MAX_SNIPPET_CHARS = 280


def build_future_impact_summary(
    profile_used: dict[str, Any],
    top_matches: list[dict[str, Any]],
    citation_manager: Any,
    top_k: int = 8,
) -> dict[str, Any]:
    """Retrieve research chunks semantically and summarize future impact for the profile."""
    query = _build_future_impact_query(profile_used, top_matches)
    try:
        retrieved = retrieve_research_inference(query, top_k=top_k)
    except Exception as exc:
        return {
            "method": FUTURE_IMPACT_METHOD,
            "status": "unavailable",
            "query": query,
            "summary": (
                "Semantic research retrieval is not available right now, so this section "
                "could not be grounded in the extracted AI-impact paper corpus."
            ),
            "takeaways": [],
            "retrieved_research": [],
            "error": str(exc),
        }

    cited_chunks = [_attach_source_id(item, citation_manager) for item in retrieved]
    summary, takeaways = _generate_future_summary(profile_used, top_matches, cited_chunks)
    return {
        "method": FUTURE_IMPACT_METHOD,
        "status": "ok" if cited_chunks else "empty",
        "query": query,
        "summary": summary,
        "takeaways": takeaways,
        "retrieved_research": cited_chunks,
    }


def _build_future_impact_query(
    profile_used: dict[str, Any],
    top_matches: list[dict[str, Any]],
) -> str:
    interests = ", ".join(str(item) for item in profile_used.get("final_top_interests") or [])
    code = str(profile_used.get("final_holland_code") or "")
    preferences = ", ".join(str(item) for item in profile_used.get("sub_preferences") or [])
    careers = ", ".join(
        _match_title(match)
        for match in top_matches[:8]
        if _match_title(match)
    )
    zones = (
        f"current Job Zone {profile_used.get('current_job_zone')}; "
        f"future Job Zone {profile_used.get('future_job_zone')}"
    )
    parts = [
        "future career impact of AI, automation, augmentation, generative AI, labor market change",
        f"Holland code {code}" if code else "",
        f"interests {interests}" if interests else "",
        f"preferences {preferences}" if preferences else "",
        zones,
        f"candidate careers {careers}" if careers else "",
    ]
    return one_line(". ".join(part for part in parts if part))


def _attach_source_id(item: dict[str, Any], citation_manager: Any) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    title = (
        one_line(metadata.get("source_name"))
        or one_line(metadata.get("title"))
        or one_line(metadata.get("source_id"))
        or "AI impact research source"
    )
    page = one_line(metadata.get("page") or metadata.get("source_page") or metadata.get("page_start"))
    section = "semantic research chunk"
    if page:
        section += f", page {page}"
    source_id = citation_manager.add_source(
        title=title,
        source_type="research_rag",
        url=one_line(metadata.get("source_url") or metadata.get("final_url") or metadata.get("url")) or None,
        local_file=one_line(metadata.get("source_file") or metadata.get("local_path")) or None,
        retrieved_section=section,
        note="Retrieved semantically from the extracted AI-impact paper corpus.",
    )
    text = one_line(item.get("text"))
    return {
        "source_id": source_id,
        "title": title,
        "url": one_line(metadata.get("source_url") or metadata.get("final_url") or metadata.get("url")),
        "page": page,
        "score": item.get("score"),
        "snippet": _short_snippet(text),
    }


def _generate_future_summary(
    profile_used: dict[str, Any],
    top_matches: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    if not chunks:
        return (
            "No semantically relevant research chunks were retrieved from the extracted AI-impact paper corpus.",
            [],
        )

    provider = OpenAIChatProvider()
    if provider.available:
        try:
            return _llm_future_summary(provider, profile_used, top_matches, chunks)
        except Exception:
            pass
    return _fallback_future_summary(profile_used, top_matches, chunks)


def _llm_future_summary(
    provider: OpenAIChatProvider,
    profile_used: dict[str, Any],
    top_matches: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    evidence = "\n".join(
        (
            f"[{chunk['source_id']}] {chunk.get('title')}"
            f"{' p. ' + str(chunk.get('page')) if chunk.get('page') else ''}: "
            f"{chunk.get('snippet')}"
        )
        for chunk in chunks[:8]
    )
    profile_context = {
        "final_holland_code": profile_used.get("final_holland_code"),
        "final_top_interests": profile_used.get("final_top_interests") or [],
        "current_job_zone": profile_used.get("current_job_zone"),
        "future_job_zone": profile_used.get("future_job_zone"),
        "preferences": profile_used.get("sub_preferences") or [],
        "top_careers": [_match_title(match) for match in top_matches[:8]],
    }
    prompt = (
        "Write a short future-impact summary for a career report using only the retrieved "
        "research evidence. Explain how AI is likely to affect these career options in a "
        "balanced way: exposure, augmentation, skill adaptation, and uncertainty. "
        "Use plain language for a student. Do not mention retrieval or inference. "
        "Cite every factual sentence with source markers like [3].\n\n"
        f"Final user profile:\n{profile_context}\n\n"
        f"Retrieved research evidence:\n{evidence}\n\n"
        "Return exactly this format:\n"
        "Summary: one paragraph, 3-5 sentences.\n"
        "Takeaways:\n"
        "- first takeaway\n"
        "- second takeaway\n"
        "- third takeaway"
    )
    output = provider.complete(
        [
            {
                "role": "system",
                "content": "You are a careful career guidance writer. Use only provided evidence.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return _parse_summary_output(output)


def _fallback_future_summary(
    profile_used: dict[str, Any],
    top_matches: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    markers = " ".join(f"[{chunk['source_id']}]" for chunk in chunks[:3])
    careers = [name for name in (_match_title(match) for match in top_matches[:4]) if name]
    career_text = ", ".join(careers) if careers else "these careers"
    interests = ", ".join(str(item) for item in profile_used.get("final_top_interests") or [])
    summary = (
        f"For {career_text}, the strongest future-impact theme is not simply replacement; "
        f"the research corpus points to uneven AI exposure where some information-heavy tasks "
        f"can be assisted while human judgment, accountability, and domain context still matter {markers}."
    )
    takeaways = [
        f"Use AI as a work amplifier: learn to check outputs, frame problems clearly, and connect tools to {interests or 'your main interests'} {markers}.",
        f"Prefer roles where technical judgment, communication, and real-world constraints remain central {markers}.",
        f"Treat the future impact as a planning signal, not a fixed prediction; keep building transferable skills across the recommended careers {markers}.",
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


def _match_title(match: dict[str, Any]) -> str:
    return one_line(
        match.get("display_title")
        or match.get("resolved_onet_title")
        or (match.get("onet_details") or {}).get("occupation_title")
    )


def _short_snippet(text: str) -> str:
    text = one_line(text)
    if len(text) <= MAX_SNIPPET_CHARS:
        return text
    return text[: MAX_SNIPPET_CHARS - 3].rstrip() + "..."
