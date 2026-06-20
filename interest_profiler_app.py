"""Streamlit UI for the local O*NET Interest Profiler short form."""

from __future__ import annotations

import json
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

JOB_ZONE_PREPARATION_EXPLANATIONS = {
    1: "no experience required",
    2: "high school diploma required",
    3: "associate's degree or vocational training required",
    4: "bachelor's degree required",
    5: "graduate degree required",
}

INTEREST_RANK_LABELS = {
    0: "top interest",
    1: "second interest",
    2: "third interest",
}


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
    zone = int(zone)
    return f"{zone} - {JOB_ZONE_LABELS[zone]} ({_job_zone_explanation(zone)})"


def _job_zone_explanation(zone: Any) -> str:
    try:
        zone_int = int(zone)
    except (TypeError, ValueError):
        return "preparation level unavailable"
    return JOB_ZONE_PREPARATION_EXPLANATIONS.get(zone_int, "preparation level unavailable")


def _job_zone_title(zone: Any) -> str:
    try:
        zone_int = int(zone)
    except (TypeError, ValueError):
        return "Job Zone unavailable"
    return f"Job Zone {zone_int} ({_job_zone_explanation(zone_int)})"


def _match_table_title(interest: str, rank_index: int, job_zone: int) -> str:
    rank_label = INTEREST_RANK_LABELS.get(rank_index, "interest")
    return f"{interest} ({rank_label}) + {_job_zone_title(job_zone)}"


def _score_rows(scores: dict[str, int]) -> list[dict[str, int | str]]:
    return [{"Interest Area": interest, "Score": scores[interest]} for interest in RIASEC_INTERESTS]


def _career_rows(careers: list[dict[str, Any]], nearby: bool = False) -> list[dict[str, Any]]:
    rows = []
    for career in careers:
        row = {
            "Career": career["career_title"],
        }
        if nearby:
            row["Nearby Alternative"] = _job_zone_title(career.get("job_zone"))
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
    st.success("Profile submitted.")


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
        _match_table_title(top_interests[0], 0, current_job_zone),
        career_matches["primary_current_zone"],
        career_listings,
        top_interests[0],
        current_job_zone,
    )
    _show_career_matches(
        _match_table_title(top_interests[0], 0, future_job_zone),
        career_matches["primary_future_zone"],
        career_listings,
        top_interests[0],
        future_job_zone,
    )
    _show_career_matches(
        _match_table_title(top_interests[1], 1, future_job_zone),
        career_matches["secondary_future_zone"],
        career_listings,
        top_interests[1],
        future_job_zone,
    )
    _show_career_matches(
        _match_table_title(top_interests[2], 2, future_job_zone),
        career_matches["tertiary_future_zone"],
        career_listings,
        top_interests[2],
        future_job_zone,
    )

    _render_followup(result)
    _render_final_ranked_matches(st.session_state.get("profile_result", result))
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
    if followup_refinement and followup_refinement.get("method") == "in_progress":
        st.info("Follow-up answers are being saved as you go. Continue to finish the refinement.")
        st.session_state["ip_followup_plan"] = (
            st.session_state.get("ip_followup_plan")
            or followup_refinement.get("question_plan")
            or build_followup_question_plan(result)
        )
        st.session_state["ip_followup_answers"] = (
            st.session_state.get("ip_followup_answers")
            or followup_refinement.get("questions_asked")
            or []
        )
        st.session_state["ip_followup_active"] = True
        _render_followup_question(result)
        return

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
    in_progress = result.get("followup_refinement") or {}
    question_plan = (
        st.session_state.get("ip_followup_plan")
        or in_progress.get("question_plan")
        or build_followup_question_plan(result)
    )
    questions_asked = (
        st.session_state.get("ip_followup_answers")
        or in_progress.get("questions_asked")
        or []
    )
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
    partial = {
        **result,
        "followup_refinement": {
            "method": "in_progress",
            "question_plan": updated_plan,
            "questions_asked": updated_answers,
            "final_refinement": {},
            "json_valid": True,
        },
    }
    save_profile_result(partial, RESULT_PATH)
    _set_profile_result(partial, "session_state")
    st.rerun()


def _show_followup_refinement(followup_refinement: dict[str, Any]) -> None:
    if not followup_refinement.get("json_valid", True):
        st.warning("The LLM response was not valid JSON after repair. A conservative fallback was saved.")

    st.success("Follow-up answers saved and the recommendations were updated.")
    with st.expander("Profile JSON for presentation", expanded=False):
        profile_for_rag = prepare_profile_for_rag(st.session_state["profile_result"])
        st.json(profile_for_rag)


def _render_final_ranked_matches(result: dict[str, Any]) -> None:
    ranked = result.get("final_ranked_matches") or []
    if not ranked:
        return
    st.header("Final Ranked Career Recommendations")
    warnings = result.get("validation_warnings") or []
    for warning in warnings:
        st.warning(warning)

    rows = []
    has_followup = bool((result.get("followup_refinement") or {}).get("questions_asked"))
    for item in ranked[:10]:
        rows.append(
            {
                "Career": f"{item.get('career_title')} ({_job_zone_title(item.get('job_zone'))})",
                "Fit score": item.get("score"),
                "Original profile match": _short_original_match(item),
                "Follow-up effect": _short_followup_effect(item, has_followup),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    source_notes = ["[1] O*NET Interest Profiler career listings and selected Job Zones."]
    if has_followup:
        source_notes.append("[2] Your follow-up answers and derived preference signals.")
    st.caption("Sources: " + " ".join(source_notes))


def _short_original_match(item: dict[str, Any]) -> str:
    interest = item.get("interest") or "interest"
    group = str(item.get("source_match_group") or "").replace("_", " ")
    group_text = f"; {group}" if group else ""
    return f"{interest} match at {_job_zone_title(item.get('job_zone'))}{group_text} [1]"


def _short_followup_effect(item: dict[str, Any], has_followup: bool) -> str:
    if not has_followup:
        return "Follow-up was skipped, so the original Interest Profiler order is used [1]"

    effects = (item.get("ranking_explanation") or {}).get("followup_effects") or []
    readable: list[str] = []
    for effect in effects:
        effect_text = str(effect)
        if "tasks or work activities" in effect_text:
            readable.append("your answers resemble the job's tasks")
        elif "skills or knowledge" in effect_text:
            readable.append("your answers match its skills or knowledge areas")
        elif "Work-context" in effect_text or "work-context" in effect_text:
            readable.append("the work setting fits your constraints")
        elif "farther" in effect_text and "Job Zone" in effect_text:
            readable.append("the preparation level is a weaker fit")
        elif "Few O*NET" in effect_text:
            readable.append("limited overlap with your follow-up wording")

    if not readable:
        return "Follow-up kept it relevant without adding a strong new signal [2]"
    return "; ".join(readable[:2]) + " [2]"


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

    if (result.get("followup_refinement") or {}).get("method") == "in_progress":
        st.info("Finish the follow-up questions to compute the refined ranking before generating the final report.")
        return

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
        st.success("Final career report generated.")

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
    result, _source = _get_current_profile_result()
    if result:
        _render_profile_result(result, career_listings)


if __name__ == "__main__":
    main()
