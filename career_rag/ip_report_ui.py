"""Streamlit rendering helpers for the final Interest Profiler career report."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_final_career_report(report: dict[str, Any]) -> None:
    """Render the final career report in a clean tabbed Streamlit layout."""
    if not report:
        st.info("No final career report is available yet.")
        return

    tabs = st.tabs(
        [
            "Summary",
            "Top Matches",
            "AI Impact",
            "Skills & Education",
            "Alternatives",
            "Sources",
        ]
    )

    with tabs[0]:
        _render_summary(report)
    with tabs[1]:
        _render_top_matches(report)
    with tabs[2]:
        _render_ai_impact(report)
    with tabs[3]:
        _render_skills_education(report)
    with tabs[4]:
        _render_alternatives(report)
    with tabs[5]:
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
    cols[2].metric("Current Zone", profile.get("current_job_zone", ""))
    cols[3].metric("Future Zone", profile.get("future_job_zone", ""))

    preferences = profile.get("sub_preferences") or []
    if preferences:
        st.write("Preferences used for matching: " + ", ".join(preferences))
    if profile.get("career_matching_guidance"):
        st.caption(profile["career_matching_guidance"])
    st.caption(f"Generation method: {report.get('report_generation_method', 'unknown')}")


def _render_top_matches(report: dict[str, Any]) -> None:
    st.subheader("Top Career Matches")
    for match in report.get("top_matches") or []:
        with st.container(border=True):
            cols = st.columns([3, 1])
            cols[0].markdown(f"**{match.get('rank')}. {match.get('display_title')}**")
            cols[1].metric("Fit", match.get("fit_label") or match.get("fit_score"))
            if match.get("resolution_note"):
                st.caption(match["resolution_note"])
            _write_bullets(match.get("why_it_fits") or [])

            with st.expander("Job detail card"):
                _render_job_detail(match)


def _render_job_detail(match: dict[str, Any]) -> None:
    details = match.get("onet_details") or {}
    st.markdown("**What This Job Does**")
    st.write(details.get("description") or "No local O*NET description was retrieved.")

    st.markdown("**Main Tasks**")
    _write_bullets(details.get("tasks") or [])

    st.markdown("**Key Skills**")
    _write_bullets(details.get("skills") or [])

    st.markdown("**Education / Job Zone**")
    job_zone = details.get("job_zone") or {}
    if job_zone:
        st.write(
            f"Job Zone {job_zone.get('zone')}: {job_zone.get('name')} "
            f"Preparation: {job_zone.get('education')}"
        )
    st.write(match.get("education_needed") or "")

    st.markdown("**AI Impact Breakdown**")
    _render_ai_table(match)
    st.write(match.get("ai_impact", {}).get("future_outlook_summary") or "")

    st.markdown("**Skills To Learn**")
    _write_bullets(match.get("skills_to_learn") or [])

    st.markdown("**Day In The Life**")
    st.write(match.get("day_in_the_life") or "")


def _render_ai_impact(report: dict[str, Any]) -> None:
    matches = report.get("top_matches") or []
    if not matches:
        st.info("No top matches are available.")
        return

    labels = [f"{match.get('rank')}. {match.get('display_title')}" for match in matches]
    selected = st.selectbox("Career", labels)
    index = labels.index(selected)
    match = matches[index]

    st.subheader(match.get("display_title"))
    if match.get("resolution_note"):
        st.caption(match["resolution_note"])
    _render_ai_table(match)
    st.write(match.get("ai_impact", {}).get("future_outlook_summary") or "")


def _render_ai_table(match: dict[str, Any]) -> None:
    rows = []
    for item in match.get("ai_impact", {}).get("task_breakdown") or []:
        rows.append(
            {
                "Task": item.get("task"),
                "Automation level": item.get("automation_level"),
                "Score": item.get("score"),
                "Evidence": item.get("evidence"),
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No AI impact rows were available for this occupation.")


def _render_skills_education(report: dict[str, Any]) -> None:
    st.subheader("Core Skills Across Your Matches")
    _write_bullets(report.get("core_skills_across_matches") or [])

    st.subheader("Job-Specific Skills And Education")
    for match in report.get("top_matches") or []:
        with st.expander(match.get("display_title", "Career")):
            st.markdown("**Skills to learn**")
            _write_bullets(match.get("skills_to_learn") or [])
            st.markdown("**Education needed**")
            st.write(match.get("education_needed") or "")


def _render_alternatives(report: dict[str, Any]) -> None:
    st.subheader("Alternative Careers")
    alternatives = report.get("alternative_careers") or []
    if not alternatives:
        st.info("No alternatives were resolved from local O*NET evidence.")
        return
    for item in alternatives[:5]:
        st.markdown(f"**{item.get('title')}**")
        if item.get("resolved_onet_title") and item.get("resolved_onet_title") != item.get("title"):
            st.caption(f"O*NET evidence: {item.get('resolved_onet_title')}")
        st.write(item.get("reason") or "")


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
            if source.get("local_file"):
                st.caption(f"Local: {source['local_file']}")


def _write_bullets(items: list[Any]) -> None:
    if not items:
        st.write("No local evidence was retrieved for this section.")
        return
    for item in items:
        st.write(f"- {item}")
