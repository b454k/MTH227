"""Streamlit UI for the local O*NET Interest Profiler short form."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from career_rag.config import PROJECT_ROOT
from career_rag.interest_profiler_local import (
    RIASEC_INTERESTS,
    build_profile_result,
    calculate_riasec_scores,
    prepare_profile_for_rag,
    save_profile_result,
    update_profile_with_followup,
)
from career_rag.ip_career_matcher import (
    JOB_ZONE_LABELS,
    build_career_matches,
    load_ip_career_listings,
    match_nearby_careers,
)
from career_rag.ip_followup_agent import (
    MAX_FOLLOWUP_QUESTIONS,
    build_followup_question_plan,
    complete_followup_refinement,
    get_next_followup_question,
    record_followup_answer,
)
from career_rag.ip_final_report import FINAL_REPORT_JSON_PATH, build_final_career_report
from career_rag.ip_report_ui import render_final_career_report


DATA_DIR = PROJECT_ROOT / "onet_interest_profiler"
QUESTIONS_PATH = DATA_DIR / "interest_profiler_questions.json"
CAREER_LISTINGS_PATH = DATA_DIR / "ip_career_listings.json"
RESULT_PATH = DATA_DIR / "ip_profile_result.json"
FINAL_REPORT_PATH = FINAL_REPORT_JSON_PATH


@st.cache_data
def _load_questions() -> list[dict[str, Any]]:
    from career_rag.interest_profiler_local import load_interest_profiler_questions

    return load_interest_profiler_questions(QUESTIONS_PATH)


@st.cache_data
def _load_career_listings() -> list[dict[str, Any]]:
    return load_ip_career_listings(CAREER_LISTINGS_PATH)


def _group_questions_by_interest(questions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {interest: [] for interest in RIASEC_INTERESTS}
    for question in questions:
        grouped[question["area"]].append(question)
    return grouped


def _format_job_zone(zone: int) -> str:
    return f"{zone} - {JOB_ZONE_LABELS[int(zone)]}"


def _score_rows(scores: dict[str, int]) -> list[dict[str, int | str]]:
    return [{"Interest Area": interest, "Score": scores[interest]} for interest in RIASEC_INTERESTS]


def _career_rows(careers: list[dict[str, Any]], nearby: bool = False) -> list[dict[str, Any]]:
    rows = []
    for career in careers:
        note = career.get("assignment_note")
        if note == "assigned_based_on_second_highest_interest":
            note_text = "Assigned based on second highest interest"
        elif note == "assigned_based_on_third_highest_interest":
            note_text = "Assigned based on third highest interest"
        else:
            note_text = ""

        row = {
            "Career": career["career_title"],
            "Job Zone": career["job_zone"],
            "Note": note_text,
        }
        if nearby:
            row["Nearby Alternative For"] = career.get("nearby_alternative_for_job_zone")
        rows.append(row)
    return rows


def _show_career_matches(
    title: str,
    careers: list[dict[str, Any]],
    all_careers: list[dict[str, Any]],
    interest: str,
    job_zone: int,
) -> None:
    st.subheader(title)
    if careers:
        st.dataframe(_career_rows(careers), hide_index=True, use_container_width=True)
        return

    st.info("No careers are listed for this Interest Area and Job Zone in the local PDF data.")
    nearby = match_nearby_careers(all_careers, interest, job_zone)
    if nearby:
        st.caption("Nearby Job Zone alternatives from the same local PDF data")
        st.dataframe(_career_rows(nearby, nearby=True), hide_index=True, use_container_width=True)


def _reset_followup_state() -> None:
    st.session_state["ip_followup_active"] = False
    st.session_state["ip_followup_complete"] = False
    st.session_state["ip_followup_plan"] = []
    st.session_state["ip_followup_answers"] = []


def _set_profile_result(result: dict[str, Any], source: str) -> None:
    st.session_state["profile_result"] = result
    st.session_state["ip_profile_result"] = result
    st.session_state["profile_result_source"] = source


def _clear_old_final_report_state() -> None:
    st.session_state["final_report"] = None
    st.session_state["ip_final_career_report"] = None
    st.session_state["followup_answers"] = None
    st.session_state["ip_followup_answers"] = None


def _load_profile_result_from_json() -> dict[str, Any] | None:
    if not RESULT_PATH.exists():
        return None
    try:
        with RESULT_PATH.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _get_current_profile_result() -> tuple[dict[str, Any] | None, str]:
    result = st.session_state.get("profile_result")
    if isinstance(result, dict):
        return result, "session_state"

    legacy_result = st.session_state.get("ip_profile_result")
    if isinstance(legacy_result, dict):
        _set_profile_result(legacy_result, "session_state")
        return legacy_result, "session_state"

    saved_result = _load_profile_result_from_json()
    if saved_result:
        _set_profile_result(saved_result, "JSON")
        return saved_result, "JSON"
    return None, "none"


def _render_profile_source_debug(source: str, result: dict[str, Any]) -> None:
    st.caption(
        "Profile source used: "
        f"{source}; profile_id={result.get('profile_id', '(missing)')}; "
        f"timestamp={result.get('timestamp', '(missing)')}"
    )


def _has_followup_answer(result: dict[str, Any]) -> bool:
    followup = result.get("followup_refinement") or {}
    return bool(followup.get("questions_asked"))


def _validate_report_matches_profile(report: dict[str, Any], profile_result: dict[str, Any]) -> list[str]:
    profile_used = report.get("profile_used") or {}
    expected_initial = str(
        profile_result.get("holland_code")
        or profile_result.get("initial_code")
        or profile_result.get("initial_holland_code")
        or ""
    )
    expected_current = int(profile_result.get("current_job_zone") or 0)
    expected_future = int(profile_result.get("future_job_zone") or 0)
    actual_initial = str(profile_used.get("initial_code") or profile_used.get("initial_holland_code") or "")
    actual_current = int(profile_used.get("current_zone") or profile_used.get("current_job_zone") or 0)
    actual_future = int(profile_used.get("future_zone") or profile_used.get("future_job_zone") or 0)
    actual_final = str(profile_used.get("final_code") or profile_used.get("final_holland_code") or "")

    errors = []
    if actual_initial != expected_initial:
        errors.append(f"initial_code mismatch: report={actual_initial}, profile={expected_initial}")
    if actual_current != expected_current:
        errors.append(f"current_zone mismatch: report={actual_current}, profile={expected_current}")
    if actual_future != expected_future:
        errors.append(f"future_zone mismatch: report={actual_future}, profile={expected_future}")
    if not _has_followup_answer(profile_result) and actual_final != expected_initial:
        errors.append(f"final_code mismatch without follow-up: report={actual_final}, initial={expected_initial}")
    return errors


def _render_initial_form(questions: list[dict[str, Any]], career_listings: list[dict[str, Any]]) -> None:
    grouped_questions = _group_questions_by_interest(questions)

    st.header("Intro")
    st.write(
        "Check the box by the activities you would like to do. Do not think about "
        "how much education/training is needed or how much money you will make."
    )

    with st.form("interest_profiler_form"):
        st.header("60 Work Activities")
        checked_ids: list[int] = []
        for interest in RIASEC_INTERESTS:
            st.subheader(interest)
            columns = st.columns(2)
            for index, question in enumerate(grouped_questions[interest]):
                with columns[index % 2]:
                    checked = st.checkbox(
                        question["text"],
                        key=f"activity_{question['id']}",
                    )
                if checked:
                    checked_ids.append(question["id"])

        st.header("Job Zone Questions")
        st.markdown("**A. Current Job Zone**")
        current_job_zone = st.selectbox(
            "What level of education, training, and experience do you currently have?",
            options=list(JOB_ZONE_LABELS),
            format_func=_format_job_zone,
            index=2,
        )
        st.markdown("**B. Future Job Zone**")
        future_job_zone = st.selectbox(
            "What level of education, training, and experience are you willing to work toward?",
            options=list(JOB_ZONE_LABELS),
            format_func=_format_job_zone,
            index=3,
        )
        submitted = st.form_submit_button("Submit")

    if not submitted:
        return

    scores = calculate_riasec_scores(questions, checked_ids)
    career_matches = build_career_matches(
        scores=scores,
        current_job_zone=current_job_zone,
        future_job_zone=future_job_zone,
        career_listings=career_listings,
    )
    result = build_profile_result(
        scores=scores,
        current_job_zone=current_job_zone,
        future_job_zone=future_job_zone,
        career_matches=career_matches,
    )
    _clear_old_final_report_state()
    save_profile_result(result, RESULT_PATH)
    _set_profile_result(result, "session_state")
    _reset_followup_state()
    st.success(f"Saved profile result to {Path(RESULT_PATH).as_posix()}")


def _render_profile_result(result: dict[str, Any], career_listings: list[dict[str, Any]]) -> None:
    scores = result.get("riasec_scores") or result["raw_riasec_scores"]
    top_interests = result.get("top_interests") or result["initial_top_interests"]
    current_job_zone = int(result["current_job_zone"])
    future_job_zone = int(result["future_job_zone"])
    career_matches = result["career_matches"]

    st.header("Profile Result")
    st.subheader("RIASEC Scores")
    st.dataframe(_score_rows(scores), hide_index=True, use_container_width=True)

    ambiguity = result.get("score_ambiguity") or {}
    if ambiguity.get("has_ambiguity"):
        st.warning(
            "Top categories are tied or within 1 point: "
            + ", ".join(ambiguity.get("ambiguous_categories", []))
        )

    st.subheader("Top 3 Interests")
    st.write(", ".join(top_interests))
    st.metric("Holland Code", result.get("holland_code") or result["initial_holland_code"])

    zone_columns = st.columns(2)
    zone_columns[0].metric("Current Job Zone", _format_job_zone(current_job_zone))
    zone_columns[1].metric("Future Job Zone", _format_job_zone(future_job_zone))

    _show_career_matches(
        f"Primary Interest + Current Job Zone: {top_interests[0]} / Job Zone {current_job_zone}",
        career_matches["primary_current_zone"],
        career_listings,
        top_interests[0],
        current_job_zone,
    )
    _show_career_matches(
        f"Primary Interest + Future Job Zone: {top_interests[0]} / Job Zone {future_job_zone}",
        career_matches["primary_future_zone"],
        career_listings,
        top_interests[0],
        future_job_zone,
    )
    _show_career_matches(
        f"Secondary Interest + Future Job Zone: {top_interests[1]} / Job Zone {future_job_zone}",
        career_matches["secondary_future_zone"],
        career_listings,
        top_interests[1],
        future_job_zone,
    )
    _show_career_matches(
        f"Tertiary Interest + Future Job Zone: {top_interests[2]} / Job Zone {future_job_zone}",
        career_matches["tertiary_future_zone"],
        career_listings,
        top_interests[2],
        future_job_zone,
    )

    _render_followup(result)
    _render_final_report_section(st.session_state.get("profile_result", result))

    st.download_button(
        "Download JSON Result",
        data=json.dumps(st.session_state.get("profile_result", result), indent=2, ensure_ascii=True),
        file_name="ip_profile_result.json",
        mime="application/json",
    )


def _render_followup(result: dict[str, Any]) -> None:
    ambiguity = result.get("score_ambiguity") or {}
    followup_refinement = result.get("followup_refinement")

    st.header("Follow-Up Refinement")
    if followup_refinement:
        _show_followup_refinement(followup_refinement)
        return

    if st.session_state.get("ip_followup_active"):
        _render_followup_question(result)
        return

    if ambiguity.get("has_ambiguity"):
        st.info(
            "Your profile has some close or tied interest areas. You can answer a few "
            "follow-up questions to refine your result."
        )
    else:
        st.info(
            "Your scores are clear, but you can answer a few questions to personalize "
            "the recommendations further."
        )

    continue_col, skip_col = st.columns(2)
    with continue_col:
        if st.button("Continue with follow-up questions"):
            st.session_state["ip_followup_plan"] = build_followup_question_plan(result)
            st.session_state["ip_followup_answers"] = []
            st.session_state["ip_followup_active"] = True
            st.session_state["ip_followup_complete"] = False
            st.rerun()
    with skip_col:
        if st.button("Skip follow-up and use initial result"):
            st.success("Using the initial deterministic O*NET result.")


def _render_followup_question(result: dict[str, Any]) -> None:
    question_plan = st.session_state.get("ip_followup_plan") or build_followup_question_plan(result)
    questions_asked = st.session_state.get("ip_followup_answers") or []
    current = get_next_followup_question(question_plan, questions_asked)

    if current is None:
        refinement = complete_followup_refinement(result, questions_asked)
        updated = update_profile_with_followup(result, refinement)
        save_profile_result(updated, RESULT_PATH)
        _set_profile_result(updated, "session_state")
        st.session_state["final_report"] = None
        st.session_state["ip_final_career_report"] = None
        st.session_state["ip_followup_active"] = False
        st.session_state["ip_followup_complete"] = True
        st.rerun()
        return

    st.caption(f"Question {len(questions_asked) + 1} of max {MAX_FOLLOWUP_QUESTIONS}")
    st.write(current["question"])
    with st.form(f"followup_question_{len(questions_asked)}"):
        answer = st.text_area("Your answer", key=f"followup_answer_{len(questions_asked)}")
        submitted = st.form_submit_button("Submit answer")

    if not submitted:
        return
    if not answer.strip():
        st.warning("Please add a short answer before continuing.")
        return

    updated_plan, updated_answers = record_followup_answer(question_plan, questions_asked, answer)
    st.session_state["ip_followup_plan"] = updated_plan
    st.session_state["ip_followup_answers"] = updated_answers
    st.session_state["followup_answers"] = updated_answers
    st.rerun()


def _show_followup_refinement(followup_refinement: dict[str, Any]) -> None:
    final = followup_refinement.get("final_refinement") or {}
    st.caption(f"Method: {followup_refinement.get('method', 'unknown')}")
    if not followup_refinement.get("json_valid", True):
        st.warning("The LLM response was not valid JSON after repair. A conservative fallback was saved.")

    st.subheader("Refined Top Interests")
    st.write(", ".join(final.get("refined_top_interests", [])))
    st.metric("Refined Holland Code", final.get("refined_holland_code", ""))

    st.subheader("Key Sub-Preferences")
    for item in final.get("key_sub_preferences", []):
        st.write(f"- {item}")

    st.subheader("Future Vision Summary")
    st.write(final.get("future_vision_summary", ""))

    st.subheader("Recommended Career Matching Strategy")
    guidance = final.get("career_matching_guidance") or {}
    st.write(guidance.get("notes", ""))
    if guidance.get("prioritize_interests"):
        st.write("Prioritize: " + ", ".join(guidance["prioritize_interests"]))
    if guidance.get("job_zone_to_use"):
        st.write(f"Job Zone to use: {_format_job_zone(int(guidance['job_zone_to_use']))}")

    with st.expander("Structured profile prepared for later RAG integration"):
        profile_for_rag = prepare_profile_for_rag(st.session_state["profile_result"])
        st.json(profile_for_rag)


def _load_saved_final_report() -> dict[str, Any] | None:
    if not FINAL_REPORT_PATH.exists():
        return None
    try:
        with FINAL_REPORT_PATH.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _render_final_report_section(result: dict[str, Any]) -> None:
    st.header("Final Career Report")
    st.write(
        "Generate a source-grounded report from the saved Interest Profiler profile, "
        "local O*NET data, and available AI-impact evidence."
    )

    if st.button("Generate Final Career Report"):
        with st.spinner("Retrieving O*NET and AI-impact evidence..."):
            try:
                report = build_final_career_report(result, top_k=10)
                validation_errors = _validate_report_matches_profile(report, result)
                if validation_errors:
                    st.error("Final report validation failed: " + "; ".join(validation_errors))
                    return
            except Exception as exc:
                st.error(f"Could not generate the final career report: {exc}")
                return
        st.session_state["final_report"] = report
        st.session_state["ip_final_career_report"] = report
        st.success(f"Saved final report to {Path(FINAL_REPORT_PATH).as_posix()}")

    report = st.session_state.get("final_report") or st.session_state.get("ip_final_career_report")
    if not report and not result:
        report = _load_saved_final_report()
    if report:
        validation_errors = _validate_report_matches_profile(report, result)
        if validation_errors:
            st.warning("Hidden stale final report because it does not match the current profile: " + "; ".join(validation_errors))
            return
    if report:
        render_final_career_report(report)


def main() -> None:
    st.set_page_config(page_title="O*NET Interest Profiler", layout="wide")
    st.title("O*NET Interest Profiler Short Form")

    questions = _load_questions()
    career_listings = _load_career_listings()

    _render_initial_form(questions, career_listings)
    result, source = _get_current_profile_result()
    if result:
        _render_profile_source_debug(source, result)
        _render_profile_result(result, career_listings)


if __name__ == "__main__":
    main()
