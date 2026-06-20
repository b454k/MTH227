"""Streamlit rendering helpers for the final Interest Profiler career report."""

from __future__ import annotations

import re
from typing import Any

import streamlit as st


JOB_ZONE_PREPARATION_EXPLANATIONS = {
    1: "no experience required",
    2: "high school diploma required",
    3: "associate's degree or vocational training required",
    4: "bachelor's degree required",
    5: "graduate degree required",
}

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


def render_final_career_report(report: dict[str, Any]) -> None:
    """Render the final career report in a clean tabbed Streamlit layout."""
    if not report:
        st.info("No final career report is available yet.")
        return

    tabs = st.tabs(
        [
            "Summary",
            "Top Matches",
            "Alternatives",
            "Sources",
        ]
    )

    with tabs[0]:
        _render_summary(report)
    with tabs[1]:
        _render_top_matches(report)
    with tabs[2]:
        _render_alternatives(report)
    with tabs[3]:
        _render_sources(report)


def _render_summary(report: dict[str, Any]) -> None:
    profile = report.get("profile_used") or {}
    st.subheader("Profile Summary")
    scores = profile.get("riasec_scores") or {}
    if scores:
        st.dataframe(
            [{"Interest Area": key, "Score": value} for key, value in scores.items()],
            hide_index=True,
            use_container_width=True,
        )

    cols = st.columns(4)
    cols[0].metric("Initial Code", profile.get("initial_holland_code", ""))
    cols[1].metric("Final Code", profile.get("final_holland_code", ""))
    cols[2].metric("Current Zone", _job_zone_title(profile.get("current_job_zone")))
    cols[3].metric("Future Zone", _job_zone_title(profile.get("future_job_zone")))

    preferences = profile.get("sub_preferences") or []
    if preferences:
        st.write("Preferences used for matching: " + ", ".join(preferences))
    if profile.get("career_matching_guidance"):
        st.caption(profile["career_matching_guidance"])


def _render_top_matches(report: dict[str, Any]) -> None:
    st.subheader("Top Career Matches")
    for match in report.get("top_matches") or []:
        with st.container(border=True):
            cols = st.columns([3, 1])
            cols[0].markdown(f"**{match.get('rank')}. {_match_title(match)}**")
            cols[1].metric("Fit", match.get("fit_label") or match.get("fit_score"))
            if match.get("resolution_note"):
                st.caption(match["resolution_note"])
            _write_bullets(match.get("why_it_fits") or [])

            with st.expander("Job detail card"):
                _render_job_detail(match)


def _render_job_detail(match: dict[str, Any]) -> None:
    details = match.get("onet_details") or {}
    st.markdown("**What This Job Does**")
    description = details.get("description") or "No local O*NET description was retrieved."
    st.write(_without_citations(description))
    _write_source_indexes(description)

    st.markdown("**Main Tasks**")
    tasks = details.get("tasks") or []
    _write_bullets(tasks)
    _write_source_indexes(tasks)

    st.markdown("**Key Skills**")
    _render_key_skills(match)

    st.markdown("**Education / Job Zone**")
    job_zone = details.get("job_zone") or {}
    if job_zone:
        st.write(_job_zone_detail(job_zone))
    education = match.get("education_needed") or ""
    if education:
        st.write(_without_citations(education))
    _write_source_indexes([job_zone, education, details.get("education") or []])

    st.markdown("**AI Impact Breakdown**")
    _render_ai_table(match)
    ai_summary = match.get("ai_impact", {}).get("future_outlook_summary") or ""
    if ai_summary:
        st.markdown("**Inference**")
        st.info(_without_citations(ai_summary))
        _write_source_indexes(ai_summary)

    st.markdown("**Day In The Life**")
    day_text = match.get("day_in_the_life") or ""
    if day_text:
        st.info(_without_citations(day_text))
        _write_source_indexes([day_text, details.get("tasks") or [], details.get("work_context") or []])


def _render_ai_table(match: dict[str, Any]) -> None:
    rows = []
    for item in match.get("ai_impact", {}).get("task_breakdown") or []:
        score = item.get("score_display")
        if not score:
            score = "N/A" if item.get("score") is None else item.get("score")
        rows.append(
            {
                "O*NET task ID": item.get("onet_task_id"),
                "Task": item.get("task"),
                "AI exposure signal": item.get("automation_level"),
                "Score": score,
                "Source": _format_source_ids(item.get("source_ids") or []),
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No AI impact rows were available for this occupation.")


def _render_alternatives(report: dict[str, Any]) -> None:
    st.subheader("Alternative Careers")
    alternatives = report.get("alternative_careers") or []
    if not alternatives:
        st.info("No alternatives were resolved from local O*NET evidence.")
        return
    for item in alternatives[:5]:
        zone = _job_zone_title((item.get("job_zone") or {}).get("zone"))
        st.markdown(f"**{item.get('title')} ({zone})**")
        if item.get("resolved_onet_title") and item.get("resolved_onet_title") != item.get("title"):
            st.caption(f"O*NET evidence: {item.get('resolved_onet_title')}")
        st.write(_without_citations(item.get("reason") or ""))
        _write_source_indexes(item.get("reason") or "")


def _render_sources(report: dict[str, Any]) -> None:
    st.subheader("Sources")
    with st.expander("View numbered citations", expanded=True):
        for source in report.get("sources") or []:
            label = f"[{source.get('id')}] {source.get('title')}"
            st.markdown(f"**{label}**")
            if source.get("retrieved_section"):
                st.write(source["retrieved_section"])
            if source.get("note"):
                st.caption(source["note"])
            if source.get("url"):
                st.write(source["url"])


def _write_bullets(items: list[Any]) -> None:
    if not items:
        st.write("No local evidence was retrieved for this section.")
        return
    for item in items:
        st.write(f"- {_without_citations(item)}")


def _render_key_skills(match: dict[str, Any]) -> None:
    grouped = match.get("key_skills") or {}
    if not grouped:
        details = match.get("onet_details") or {}
        grouped = _fallback_key_skills(details)

    ordered_groups = [
        ("software_tools", "Software / technical tools"),
        ("technical_and_domain", "Technical and domain skills"),
        ("foundational_communication", "Foundational communication"),
        ("knowledge_areas", "Knowledge areas"),
    ]
    rendered_any = False
    for key, label in ordered_groups:
        values = grouped.get(key) or []
        if not values:
            continue
        rendered_any = True
        st.caption(label)
        st.markdown(_skill_chip_html(values), unsafe_allow_html=True)
        _write_source_indexes(values)
    if not rendered_any:
        st.write("No local evidence was retrieved for this section.")


def _fallback_key_skills(details: dict[str, Any]) -> dict[str, list[str]]:
    skills = [_without_citations(item) for item in details.get("skills") or []]
    software = [_without_citations(item) for item in details.get("software") or []]
    knowledge = [_without_citations(item) for item in details.get("knowledge") or []]
    foundational = [item for item in skills if _skill_key(item) in FOUNDATIONAL_SOFT_SKILLS]
    technical = [item for item in skills if _skill_key(item) not in FOUNDATIONAL_SOFT_SKILLS]
    return {
        "software_tools": software,
        "technical_and_domain": technical,
        "foundational_communication": foundational,
        "knowledge_areas": knowledge,
    }


def _skill_chip_html(values: list[Any]) -> str:
    chips = []
    for value in values:
        text = _without_citations(value)
        if not text:
            continue
        chips.append(f"<span class='skill-chip'>{text}</span>")
    if not chips:
        return ""
    return (
        "<style>"
        ".skill-chip{display:inline-block;margin:0 0.35rem 0.35rem 0;"
        "padding:0.22rem 0.5rem;border:1px solid rgba(49,51,63,.22);"
        "border-radius:0.35rem;background:rgba(240,242,246,.7);font-size:0.9rem;}"
        "</style>"
        + "".join(chips)
    )


def _match_title(match: dict[str, Any]) -> str:
    title = str(match.get("display_title") or "Career")
    details = match.get("onet_details") or {}
    zone = (details.get("job_zone") or {}).get("zone")
    if zone:
        return f"{title} - {_job_zone_title(zone)}"
    return title


def _job_zone_detail(job_zone: dict[str, Any]) -> str:
    zone = job_zone.get("zone")
    name = _without_citations(job_zone.get("name") or "")
    education = _without_citations(job_zone.get("education") or "")
    training = _without_citations(job_zone.get("training") or "")
    parts = [_job_zone_title(zone)]
    if name:
        parts.append(name)
    if education:
        parts.append(f"typical education: {education}")
    if training:
        parts.append(f"training: {training}")
    return ". ".join(parts) + "."


def _job_zone_title(zone: Any) -> str:
    try:
        zone_int = int(zone)
    except (TypeError, ValueError):
        return "Job Zone unavailable"
    explanation = JOB_ZONE_PREPARATION_EXPLANATIONS.get(zone_int, "preparation level unavailable")
    return f"Job Zone {zone_int} ({explanation})"


def _without_citations(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\s*\[\d+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _citation_numbers(value: Any) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                collect(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                collect(nested)
            return
        for raw_number in re.findall(r"\[(\d+)\]", str(item or "")):
            number = int(raw_number)
            if number not in seen:
                seen.add(number)
                numbers.append(number)

    collect(value)
    return numbers


def _write_source_indexes(value: Any) -> None:
    numbers = _citation_numbers(value)
    if numbers:
        st.caption("Source indexes: " + " ".join(f"[{number}]" for number in numbers))


def _format_source_ids(values: list[Any]) -> str:
    ids = []
    for value in values:
        try:
            source_id = int(value)
        except (TypeError, ValueError):
            continue
        ids.append(f"[{source_id}]")
    return " ".join(ids)


def _skill_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
