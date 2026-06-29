"""Streamlit rendering helpers for the final Interest Profiler career report."""

from __future__ import annotations

import re
from html import escape
from typing import Any

import streamlit as st

from career_rag.ui_i18n import (
    display_label,
    dynamic_list,
    dynamic_text,
    dynamic_translation_available,
    interest_label,
    job_zone_preparation,
    normalize_language,
    ui_text,
)


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


def render_final_career_report(report: dict[str, Any], language: str = "en") -> None:
    """Render the final career report in a clean tabbed Streamlit layout."""
    if not report:
        st.info(ui_text("report.no_final", language))
        return
    if normalize_language(language) == "tr" and not dynamic_translation_available(language):
        st.warning(ui_text("translation.unavailable", language))

    tabs = st.tabs(
        [
            ui_text("report.tab.summary", language),
            ui_text("report.tab.top_matches", language),
            ui_text("report.tab.alternatives", language),
            ui_text("report.tab.semantic", language),
            ui_text("report.tab.sources", language),
        ]
    )

    with tabs[0]:
        _render_summary(report, language)
    with tabs[1]:
        _render_top_matches(report, language)
    with tabs[2]:
        _render_alternatives(report, language)
    with tabs[3]:
        _render_semantic_report(report, language)
    with tabs[4]:
        _render_sources(report, language)


def _render_summary(report: dict[str, Any], language: str = "en") -> None:
    profile = report.get("profile_used") or {}
    st.subheader(ui_text("report.summary.header", language))

    top_interests = profile.get("final_top_interests") or profile.get("top_interests") or []
    if top_interests:
        st.caption(ui_text("report.summary.top_interests", language))
        st.write(", ".join(interest_label(item, language) for item in top_interests[:3]))

    cols = st.columns(3)
    cols[0].caption(ui_text("report.summary.final_code", language))
    cols[0].write(f"**{profile.get('final_holland_code', '')}**")
    cols[1].caption(ui_text("report.summary.current_zone", language))
    cols[1].write(_job_zone_title(profile.get("current_job_zone"), language))
    cols[2].caption(ui_text("report.summary.future_zone", language))
    cols[2].write(_job_zone_title(profile.get("future_job_zone"), language))

    preferences = profile.get("sub_preferences") or []
    if preferences:
        st.markdown(f"**{ui_text('report.summary.preferences', language)}**")
        _render_grouped_preferences(preferences, language)
    if profile.get("career_matching_guidance"):
        st.caption(dynamic_text(profile["career_matching_guidance"], language, "career matching guidance"))


def _render_top_matches(report: dict[str, Any], language: str = "en") -> None:
    st.subheader(ui_text("report.top_matches.header", language))
    matches = report.get("top_matches") or []
    st.caption(ui_text("report.top_matches.caption", language))

    for group_label, group_matches in _group_top_matches(matches):
        if group_label:
            st.markdown(f"**{display_label(group_label, language)}**")
        for match in group_matches:
            with st.container(border=True):
                cols = st.columns([3, 1])
                cols[0].markdown(f"**{match.get('rank')}. {_match_title(match, language)}**")
                fit_value = match.get("fit_label") or match.get("fit_score")
                if isinstance(fit_value, str):
                    fit_value = display_label(fit_value, language)
                cols[1].metric(ui_text("report.top_matches.fit", language), fit_value)
                if match.get("resolution_note"):
                    st.caption(dynamic_text(match["resolution_note"], language, "career title resolution note"))

                with st.expander(ui_text("report.top_matches.detail_card", language)):
                    _render_job_detail(match, language)


def _render_job_detail(match: dict[str, Any], language: str = "en") -> None:
    details = match.get("onet_details") or {}
    st.markdown(f"**{ui_text('report.job.what_does', language)}**")
    description = details.get("description") or ui_text("report.job.no_description", language)
    st.write(dynamic_text(_without_citations(description), language, "O*NET occupation description"))
    _write_source_indexes(description)

    st.markdown(f"**{ui_text('report.job.main_tasks', language)}**")
    tasks = details.get("tasks") or []
    _write_bullets(tasks, language)
    _write_source_indexes(tasks)

    st.markdown(f"**{ui_text('report.job.key_skills', language)}**")
    _render_key_skills(match, language)

    st.markdown(f"**{ui_text('report.job.education_zone', language)}**")
    _render_education_job_zone(match, language)

    st.markdown(f"**{ui_text('report.job.ai_impact', language)}**")
    _render_ai_table(match, language)

    st.markdown(f"**{ui_text('report.job.day_life', language)}**")
    day_text = match.get("day_in_the_life") or ""
    if day_text:
        st.info(dynamic_text(_clean_day_text(day_text), language, "day-in-the-life career narrative"))
        _write_source_indexes([day_text, details.get("tasks") or [], details.get("work_context") or []])


def _render_ai_table(match: dict[str, Any], language: str = "en") -> None:
    rows = []
    for item in match.get("ai_impact", {}).get("task_breakdown") or []:
        score = item.get("score_display")
        if not score:
            score = ui_text("common.na", language) if item.get("score") is None else item.get("score")
        rows.append(
            {
                ui_text("report.ai.table.task", language): dynamic_text(
                    item.get("task"), language, "AI-impact task"
                ),
                ui_text("report.ai.table.signal", language): dynamic_text(
                    item.get("automation_level"), language, "AI exposure level"
                ),
                ui_text("report.ai.table.score", language): score,
                ui_text("report.ai.table.source", language): _format_source_ids(item.get("source_ids") or []),
            }
        )
    if rows:
        _render_wrapped_table(rows)
        st.caption(ui_text("report.ai.caption", language))
    else:
        st.info(ui_text("report.ai.no_rows", language))


def _render_alternatives(report: dict[str, Any], language: str = "en") -> None:
    st.subheader(ui_text("report.alternatives.header", language))
    alternatives = report.get("alternative_careers") or []
    if not alternatives:
        st.info(ui_text("report.alternatives.none", language))
        return
    source_map = _source_map(report)
    for item in alternatives[:5]:
        zone = _job_zone_title((item.get("job_zone") or {}).get("zone"), language)
        source_links = _source_links(item, source_map)
        suffix = f" {source_links}" if source_links else ""
        title = dynamic_text(item.get("title"), language, "alternative career title")
        st.markdown(f"- **{title}** - {zone}{suffix}")


def _render_semantic_report(report: dict[str, Any], language: str = "en") -> None:
    st.subheader(ui_text("report.semantic.header", language))
    section = report.get("semantic_retrieval_report") or {}
    if not section:
        st.info(ui_text("report.semantic.need_new_report", language))
        return

    st.caption(ui_text("report.semantic.caption", language))

    summary = section.get("summary") or ""
    if section.get("status") != "ok":
        st.info(
            dynamic_text(_without_citations(summary), language, "semantic report summary")
            or ui_text("report.semantic.not_enough", language)
        )
        errors = section.get("errors") or {}
        for label, error in errors.items():
            if error:
                st.caption(f"{display_label(label, language)}: {error}")
        return

    semantic_markdown = _semantic_report_markdown(section, language)
    if semantic_markdown:
        with st.container(border=True):
            st.markdown(format_semantic_report(semantic_markdown, language))
            _write_source_indexes(
                [
                    section.get("relevant_careers_explanation"),
                    section.get("technology_ai_role"),
                    section.get("takeaways"),
                    summary,
                ]
            )

    signals = section.get("semantic_career_signals") or []
    if signals:
        st.markdown(f"**{ui_text('report.semantic.signals_heading', language)}**")
        st.caption(ui_text("report.semantic.signals_caption", language))
        rows = []
        for item in signals:
            rows.append(
                {
                    ui_text("report.semantic.table.career", language): dynamic_text(
                        item.get("title") or "", language, "semantic career signal title"
                    ),
                    ui_text("report.semantic.table.soc", language): item.get("soc_code") or "",
                    ui_text("report.semantic.table.job_zone", language): item.get("job_zone") or "",
                    ui_text("report.semantic.table.signal", language): item.get("semantic_signal") or "",
                    ui_text("report.semantic.table.sources", language): _format_source_ids(item.get("source_ids") or []),
                }
            )
        _render_wrapped_table(rows)

    onet_rows = section.get("retrieved_onet") or []
    if onet_rows:
        with st.expander(ui_text("report.semantic.retrieved_onet", language), expanded=False):
            _render_wrapped_table(
                [
                    {
                        ui_text("report.semantic.table.source", language): f"[{item.get('source_id')}]",
                        ui_text("report.semantic.table.career_section", language): " - ".join(
                            dynamic_text(part, language, "retrieved O*NET title or section")
                            for part in [item.get("title"), item.get("section")]
                            if part
                        ),
                        ui_text("report.semantic.table.job_zone", language): item.get("job_zone") or "",
                        ui_text("report.semantic.table.passage", language): dynamic_text(
                            item.get("snippet") or "", language, "retrieved O*NET evidence passage"
                        ),
                    }
                    for item in onet_rows
                ]
            )

    ai_rows = section.get("retrieved_ai_impact") or []
    if ai_rows:
        with st.expander(ui_text("report.semantic.retrieved_ai", language), expanded=False):
            _render_wrapped_table(
                [
                    {
                        ui_text("report.semantic.table.source", language): f"[{item.get('source_id')}]",
                        ui_text("report.semantic.table.occupation_signal", language): " - ".join(
                            dynamic_text(part, language, "retrieved AI-impact occupation or signal")
                            for part in [item.get("occupation"), item.get("impact_type")]
                            if part
                        ),
                        ui_text("report.semantic.table.job_zone", language): item.get("job_zone") or "",
                        ui_text("report.semantic.table.task", language): dynamic_text(
                            item.get("task") or "", language, "retrieved AI-impact task"
                        ),
                        ui_text("report.semantic.table.passage", language): dynamic_text(
                            item.get("snippet") or "", language, "retrieved AI-impact evidence passage"
                        ),
                    }
                    for item in ai_rows
                ]
            )


def _render_sources(report: dict[str, Any], language: str = "en") -> None:
    st.subheader(ui_text("report.sources.header", language))
    with st.expander(ui_text("report.sources.expander", language), expanded=True):
        for source in report.get("sources") or []:
            title = dynamic_text(source.get("title"), language, "source title")
            label = f"[{source.get('id')}] {title}"
            if source.get("url"):
                st.markdown(f"**[{label}]({source['url']})**")
            else:
                st.markdown(f"**{label}**")
            if source.get("retrieved_section"):
                st.write(dynamic_text(source["retrieved_section"], language, "retrieved source section"))
            if source.get("note"):
                st.caption(dynamic_text(source["note"], language, "source note"))
            if source.get("url"):
                st.write(source["url"])


def _semantic_report_markdown(section: dict[str, Any], language: str = "en") -> str:
    """Build the semantic report body as markdown before display cleanup."""
    relevant = dynamic_text(
        section.get("relevant_careers_explanation") or section.get("summary") or "",
        language,
        "semantic report relevant careers explanation",
    )
    technology = dynamic_text(
        section.get("technology_ai_role") or "",
        language,
        "semantic report technology and AI role",
    )
    takeaways = section.get("takeaways") or []

    parts = []
    if relevant:
        parts.append(f"{ui_text('report.semantic.relevant_heading', language)} {relevant}")
    if technology:
        parts.append(f"{ui_text('report.semantic.technology_heading', language)} {technology}")
    if takeaways:
        translated_takeaways = dynamic_list(takeaways, language, "semantic report takeaways")
        bullet_text = " ".join(f"- {item}" for item in translated_takeaways if str(item or "").strip())
        parts.append(f"{ui_text('report.semantic.takeaways_heading', language)} {bullet_text}")
    return "\n\n".join(parts)


def format_semantic_report(text: str, language: str = "en") -> str:
    """Clean malformed semantic-report markdown while preserving content."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    heading_titles = {
        "Relevant Careers": ui_text("report.semantic.relevant_heading", language),
        "Role of Technology and AI": ui_text("report.semantic.technology_heading", language),
        "Takeaways": ui_text("report.semantic.takeaways_heading", language),
        ui_text("report.semantic.relevant_heading", language): ui_text(
            "report.semantic.relevant_heading", language
        ),
        ui_text("report.semantic.technology_heading", language): ui_text(
            "report.semantic.technology_heading", language
        ),
        ui_text("report.semantic.takeaways_heading", language): ui_text(
            "report.semantic.takeaways_heading", language
        ),
    }
    heading_lookup = {key.casefold(): value for key, value in heading_titles.items()}
    for title in heading_titles.values():
        escaped = re.escape(title)
        cleaned = re.sub(
            rf"(?i)\b{escaped}\s+{escaped}\b",
            title,
            cleaned,
        )

    heading_pattern = re.compile(
        r"(?i)(^|\s+)(?:#{1,6}\s*)?"
        r"("
        + "|".join(re.escape(title) for title in heading_titles)
        + r")\s*:?\s*"
    )

    def heading_replacement(match: re.Match[str]) -> str:
        raw_title = re.sub(r"\s+", " ", match.group(2)).strip()
        title = heading_lookup.get(raw_title.casefold(), match.group(2).strip())
        prefix = "" if match.start() == 0 else "\n\n"
        return f"{prefix}#### {title}\n\n"

    cleaned = heading_pattern.sub(heading_replacement, cleaned)

    # Put inline markdown headings and bullets on their own lines.
    cleaned = re.sub(r"(?<!\n)\s+(#{1,6}\s+)", r"\n\n\1", cleaned)
    cleaned = re.sub(r"(?<!\n)\s+-\s+", "\n- ", cleaned)
    cleaned = re.sub(r"\n(-\s+)", r"\n\n\1", cleaned, count=1)

    # If an LLM returned all bullets in one paragraph, split each new bullet.
    cleaned = re.sub(r"(?m)([^\n])\s+(-\s+[A-Z0-9])", r"\1\n\2", cleaned)
    cleaned = re.sub(r"(?m)^(#### .+)\n(?!\n)", r"\1\n\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _render_wrapped_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{escape(str(row.get(column) or ''))}</td>"
            for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        (
            "<style>"
            ".wrapped-report-table{width:100%;border-collapse:collapse;font-size:0.92rem;}"
            ".wrapped-report-table th,.wrapped-report-table td{"
            "border:1px solid rgba(49,51,63,.2);padding:0.55rem;vertical-align:top;"
            "white-space:normal;word-break:break-word;overflow-wrap:anywhere;}"
            ".wrapped-report-table th{background:rgba(240,242,246,.75);font-weight:600;}"
            ".wrapped-report-table td:nth-child(1){min-width:18rem;}"
            "</style>"
            f"<table class='wrapped-report-table'><thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
        ),
        unsafe_allow_html=True,
    )


def _write_bullets(items: list[Any], language: str = "en") -> None:
    if not items:
        st.write(ui_text("report.no_local_evidence", language))
        return
    for item in items:
        st.write(f"- {dynamic_text(_without_citations(item), language, 'report bullet')}")


def _render_key_skills(match: dict[str, Any], language: str = "en") -> None:
    grouped = match.get("key_skills") or {}
    if not grouped:
        details = match.get("onet_details") or {}
        grouped = _fallback_key_skills(details)

    ordered_groups = [
        ("software_tools", ui_text("report.skills.software", language)),
        ("technical_and_domain", ui_text("report.skills.technical", language)),
        ("foundational_communication", ui_text("report.skills.foundational", language)),
        ("knowledge_areas", ui_text("report.skills.knowledge", language)),
    ]
    rendered_any = False
    for key, label in ordered_groups:
        values = grouped.get(key) or []
        if not values:
            continue
        rendered_any = True
        st.caption(label)
        st.markdown(
            _skill_chip_html(dynamic_list(values, language, f"{label} skills")),
            unsafe_allow_html=True,
        )
        _write_source_indexes(values)
    if not rendered_any:
        st.write(ui_text("report.no_local_evidence", language))


def _render_grouped_preferences(preferences: list[Any], language: str = "en") -> None:
    grouped = _group_preferences(preferences)
    for label, values in grouped.items():
        if not values:
            continue
        st.caption(display_label(label, language))
        st.markdown(
            _skill_chip_html(dynamic_list(values, language, f"{label} preferences")),
            unsafe_allow_html=True,
        )


def _group_preferences(preferences: list[Any]) -> dict[str, list[str]]:
    groups = {
        "Tasks and tools": [],
        "Work style": [],
        "People and setting": [],
        "Career direction": [],
        "Other": [],
    }
    for preference in _dedupe_text(preferences):
        key = preference.lower()
        if any(term in key for term in ("technology", "data", "math", "analyt", "problem", "research", "ai")):
            groups["Tasks and tools"].append(preference)
        elif any(term in key for term in ("fast", "pace", "structured", "independent", "autonomy", "schedule")):
            groups["Work style"].append(preference)
        elif any(term in key for term in ("social", "people", "teaching", "guiding", "listening", "interaction")):
            groups["People and setting"].append(preference)
        elif any(term in key for term in ("own", "leading", "career", "consult", "security", "impact")):
            groups["Career direction"].append(preference)
        else:
            groups["Other"].append(preference)
    return groups


def _render_education_job_zone(match: dict[str, Any], language: str = "en") -> None:
    details = match.get("onet_details") or {}
    job_zone = details.get("job_zone") or {}
    education = _dedupe_text(details.get("education") or [])
    education_entries = _education_entries(education)

    prep_summary = _education_preparation_summary(job_zone, education_entries, language)
    if prep_summary:
        st.write(ui_text("report.education.typical", language) + prep_summary)

    training = dynamic_text(
        _clean_job_zone_training(job_zone.get("training")),
        language,
        "O*NET job zone training",
    )
    if training:
        st.write(ui_text("report.education.training", language) + training)

    if education:
        st.caption(ui_text("report.education.responses", language))
        _write_bullets([_format_education_entry(item) for item in education_entries], language)

    if not education and not prep_summary and not training:
        st.write(ui_text("report.education.none", language))
    _write_source_indexes([job_zone, education])


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


def _match_title(match: dict[str, Any], language: str = "en") -> str:
    title = dynamic_text(
        match.get("display_title") or ui_text("report.career_fallback", language),
        language,
        "career match title",
    )
    details = match.get("onet_details") or {}
    zone = (details.get("job_zone") or {}).get("zone")
    if zone:
        return f"{title} - {_job_zone_title(zone, language)}"
    return title


def _job_zone_detail(job_zone: dict[str, Any], language: str = "en") -> str:
    zone = job_zone.get("zone")
    name = _without_citations(job_zone.get("name") or "")
    education = _without_citations(job_zone.get("education") or "")
    training = _without_citations(job_zone.get("training") or "")
    parts = [_job_zone_title(zone, language)]
    if name:
        parts.append(dynamic_text(name, language, "O*NET job zone name"))
    if education:
        parts.append(
            ui_text(
                "report.job_zone.typical_education",
                language,
                education=dynamic_text(education, language, "O*NET job zone education"),
            )
        )
    if training:
        parts.append(
            ui_text(
                "report.job_zone.training",
                language,
                training=dynamic_text(training, language, "O*NET job zone training"),
            )
        )
    return ". ".join(parts) + "."


def _job_zone_title(zone: Any, language: str = "en") -> str:
    try:
        zone_int = int(zone)
    except (TypeError, ValueError):
        return ui_text("report.job_zone.unavailable", language)
    if language == "en":
        explanation = JOB_ZONE_PREPARATION_EXPLANATIONS.get(zone_int, "preparation level unavailable")
    else:
        explanation = job_zone_preparation(zone_int, language)
    return ui_text("report.job_zone.title", language, zone=zone_int, explanation=explanation)


def _group_top_matches(matches: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    seen_labels: set[str] = set()
    for match in matches:
        label = str(match.get("match_group") or "").strip()
        if label in seen_labels:
            continue
        seen_labels.add(label)
        groups.append((label, [item for item in matches if str(item.get("match_group") or "").strip() == label]))
    if groups:
        return groups
    return [("", matches)]


def _shared_match_reasons(matches: list[dict[str, Any]]) -> list[str]:
    reasons: list[Any] = []
    for match in matches:
        reasons.extend(match.get("why_it_fits") or [])
    return _dedupe_text(_normalize_shared_reason(_without_citations(reason)) for reason in reasons)[:4]


def _normalize_shared_reason(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"^This role is\b", "These roles are", text, flags=re.IGNORECASE)


def _clean_day_text(value: Any) -> str:
    text = _without_citations(value)
    return re.sub(r"^my inference:\s*", "", text, flags=re.IGNORECASE).strip()


def _education_entries(values: list[Any]) -> list[dict[str, Any]]:
    entries = [_parse_education_entry(value) for value in values]
    return [entry for entry in entries if entry.get("label") or entry.get("text")]


def _parse_education_entry(value: Any) -> dict[str, Any]:
    text = _without_citations(value)
    match = re.match(r"^(?P<label>.+?)\s*\((?P<number>\d+(?:\.\d+)?)%?\)$", text)
    if not match:
        return {"label": text, "percent": None, "text": text}
    number = float(match.group("number"))
    label = match.group("label").strip()
    return {"label": label, "percent": number, "text": f"{label} ({number:g}%)"}


def _format_education_entry(entry: dict[str, Any]) -> str:
    return str(entry.get("text") or entry.get("label") or "")


def _education_preparation_summary(
    job_zone: dict[str, Any],
    education_entries: list[dict[str, Any]],
    language: str = "en",
) -> str:
    entries_with_percent = [
        entry for entry in education_entries if isinstance(entry.get("percent"), (int, float))
    ]
    if entries_with_percent:
        sorted_entries = sorted(
            entries_with_percent,
            key=lambda entry: float(entry.get("percent") or 0),
            reverse=True,
        )
        positive_entries = [
            entry for entry in sorted_entries if float(entry.get("percent") or 0) > 0
        ]
        if positive_entries:
            return _plain_education_summary(positive_entries, language)

    fallback = dynamic_text(
        _clean_job_zone_education(job_zone.get("education")),
        language,
        "O*NET job zone education",
    )
    return fallback


def _is_bachelors_or_higher(label: Any) -> bool:
    text = str(label or "").lower()
    return any(term in text for term in ("bachelor", "master", "doctoral", "professional degree"))


def _plain_education_summary(entries: list[dict[str, Any]], language: str = "en") -> str:
    top = entries[0]
    top_level = _degree_level(top.get("label"))
    bachelor_entry = next((entry for entry in entries if _degree_level(entry.get("label")) == "bachelor"), None)
    master_entry = next((entry for entry in entries if _degree_level(entry.get("label")) == "master"), None)

    if top_level == "master":
        if bachelor_entry:
            return ui_text("report.education.master_with_bachelor", language)
        return ui_text("report.education.master", language)
    if top_level == "bachelor":
        if master_entry:
            return ui_text("report.education.bachelor_with_master", language)
        return ui_text("report.education.bachelor", language)
    if top_level == "doctoral":
        return ui_text("report.education.doctoral", language)
    if top_level == "associate":
        return ui_text("report.education.associate", language)
    if top_level == "high_school":
        return ui_text("report.education.high_school", language)

    label = _without_citations(top.get("label"))
    if label:
        return ui_text("report.education.dynamic", language, label=label.lower())
    return ""


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


def _clean_job_zone_education(value: Any) -> str:
    text = _without_citations(value)
    text = re.sub(r",?\s*but some do not\.?", ".", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _clean_job_zone_training(value: Any) -> str:
    text = _without_citations(value)
    return re.sub(r"\s+", " ", text).strip()


def _join_natural(values: Any) -> str:
    items = [str(value).strip() for value in values if str(value).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


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
        st.caption(" ".join(f"[{number}]" for number in numbers))


def _format_source_ids(values: list[Any]) -> str:
    ids = []
    for value in values:
        try:
            source_id = int(value)
        except (TypeError, ValueError):
            continue
        ids.append(f"[{source_id}]")
    return " ".join(ids)


def _source_map(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    sources: dict[int, dict[str, Any]] = {}
    for source in report.get("sources") or []:
        try:
            source_id = int(source.get("id"))
        except (TypeError, ValueError):
            continue
        sources[source_id] = source
    return sources


def _source_links(item: dict[str, Any], sources: dict[int, dict[str, Any]]) -> str:
    links = []
    for number in _citation_numbers(item):
        source = sources.get(number) or {}
        url = source.get("url")
        if url:
            links.append(f"[[{number}]]({url})")
        else:
            links.append(f"[{number}]")
    return " ".join(links)


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


def _skill_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
