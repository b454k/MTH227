"""Streamlit UI for the local O*NET Interest Profiler short form."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from career_rag.artifacts import (
    MissingRagArtifactsError,
    format_missing_artifacts_error,
    inspect_rag_artifacts,
)
from career_rag.config import (
    CAREER_LISTINGS_PATH,
    FINAL_REPORT_JSON_PATH,
    PROFILE_RESULT_PATH,
    QUESTIONS_PATH,
)
from career_rag.interest_profiler_local import (
    RIASEC_INTERESTS,
    build_profile_result,
    calculate_riasec_scores,
    save_profile_result,
    update_profile_with_followup,
)
from career_rag.ip_career_matcher import (
    JOB_ZONE_LABELS,
    build_career_matches,
    load_ip_career_listings,
)
from career_rag.ip_followup_agent import (
    MAX_FOLLOWUP_QUESTIONS,
    build_followup_question_plan,
    complete_followup_refinement,
    get_next_followup_question,
    record_followup_answer,
)
from career_rag.ip_final_report import build_final_career_report
from career_rag.ip_report_ui import render_final_career_report
from career_rag.ui_i18n import (
    LANGUAGE_BUTTON_LABELS,
    LANGUAGE_SESSION_KEY,
    artifact_label,
    followup_question_text,
    interest_label,
    job_zone_label,
    job_zone_preparation,
    normalize_language,
    question_text,
    ui_text,
)


RESULT_PATH = PROFILE_RESULT_PATH
FINAL_REPORT_PATH = FINAL_REPORT_JSON_PATH

JOB_ZONE_PREPARATION_EXPLANATIONS = {
    1: "no experience required",
    2: "high school diploma required",
    3: "associate's degree or vocational training required",
    4: "bachelor's degree required",
    5: "graduate degree required",
}

METHOD_REFERENCES = [
    {
        "id": 1,
        "label": (
            "Renji, N. M., Rao, B., & Lipizzi, C. (2025). Steve: LLM powered chatbot "
            "for career progression. arXiv:2504.03789."
        ),
        "url": "https://arxiv.org/abs/2504.03789",
    },
    {
        "id": 2,
        "label": (
            "Jeon, H., et al. (2025). Letters from future self: Supporting young adults' "
            "career exploration with LLM-powered agents. CHI 2025."
        ),
        "url": "https://arxiv.org/abs/2502.18881",
    },
    {
        "id": 3,
        "label": "Anthropic Economic Index and labor-market impact research.",
        "url": "https://www.anthropic.com/research/labor-market-impacts",
    },
    {
        "id": 4,
        "label": "NBER Working Paper w32966 - The Rapid Adoption of Generative AI.",
        "url": "https://www.nber.org/papers/w32966",
    },
]


@st.cache_data
def _load_questions() -> list[dict[str, Any]]:
    from career_rag.interest_profiler_local import load_interest_profiler_questions

    return load_interest_profiler_questions(QUESTIONS_PATH)


@st.cache_data
def _load_career_listings() -> list[dict[str, Any]]:
    return load_ip_career_listings(CAREER_LISTINGS_PATH)


@st.cache_data(ttl=30)
def _inspect_rag_artifacts_cached() -> dict[str, Any]:
    return inspect_rag_artifacts(check_chroma=True)


def _group_questions_by_interest(questions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {interest: [] for interest in RIASEC_INTERESTS}
    for question in questions:
        grouped[question["area"]].append(question)
    return grouped


def _format_job_zone(zone: int) -> str:
    zone = int(zone)
    return f"{zone} - {JOB_ZONE_LABELS[zone]} ({_job_zone_explanation(zone)})"


def _format_job_zone_for_language(zone: int, language: str) -> str:
    zone = int(zone)
    return f"{zone} - {job_zone_label(zone, language)} ({_job_zone_explanation(zone, language)})"


def _job_zone_explanation(zone: Any, language: str = "en") -> str:
    try:
        zone_int = int(zone)
    except (TypeError, ValueError):
        return ui_text("report.job_zone.preparation_unavailable", language)
    if normalize_language(language) == "en":
        return JOB_ZONE_PREPARATION_EXPLANATIONS.get(zone_int, "preparation level unavailable")
    return job_zone_preparation(zone_int, language)


def _current_language() -> str:
    language = normalize_language(st.session_state.get(LANGUAGE_SESSION_KEY, "en"))
    st.session_state[LANGUAGE_SESSION_KEY] = language
    return language


def _render_language_switch() -> str:
    language = _current_language()
    _, switch_column = st.columns([5, 1.25])
    with switch_column:
        st.caption(ui_text("language.label", language))
        english_column, turkish_column = st.columns(2)
        with english_column:
            if st.button(
                LANGUAGE_BUTTON_LABELS["en"],
                key="language_switch_en",
                type="primary" if language == "en" else "secondary",
                use_container_width=True,
            ):
                st.session_state[LANGUAGE_SESSION_KEY] = "en"
                st.rerun()
        with turkish_column:
            if st.button(
                LANGUAGE_BUTTON_LABELS["tr"],
                key="language_switch_tr",
                type="primary" if language == "tr" else "secondary",
                use_container_width=True,
            ):
                st.session_state[LANGUAGE_SESSION_KEY] = "tr"
                st.rerun()
    return _current_language()


def _render_app_intro(language: str) -> None:
    st.title(ui_text("app.title", language))
    st.caption(ui_text("app.caption", language))
    with st.expander(ui_text("app.method_references", language), expanded=False):
        for source in METHOD_REFERENCES:
            st.markdown(f"[{source['id']}] [{source['label']}]({source['url']})")


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


def _render_initial_form(
    questions: list[dict[str, Any]],
    career_listings: list[dict[str, Any]],
    language: str,
) -> None:
    grouped_questions = _group_questions_by_interest(questions)

    st.write(ui_text("initial.instructions", language))

    with st.form("interest_profiler_form"):
        st.header(ui_text("initial.activities_header", language))
        checked_ids: list[int] = []
        for interest in RIASEC_INTERESTS:
            st.subheader(interest_label(interest, language))
            columns = st.columns(2)
            for index, question in enumerate(grouped_questions[interest]):
                with columns[index % 2]:
                    checked = st.checkbox(
                        question_text(question, language),
                        key=f"activity_{question['id']}",
                    )
                if checked:
                    checked_ids.append(question["id"])

        st.header(ui_text("initial.job_zone_header", language))
        st.markdown(f"**{ui_text('initial.current_job_zone_heading', language)}**")
        current_job_zone = st.selectbox(
            ui_text("initial.current_job_zone_question", language),
            options=list(JOB_ZONE_LABELS),
            format_func=lambda zone: _format_job_zone_for_language(zone, language),
            index=2,
        )
        st.markdown(f"**{ui_text('initial.future_job_zone_heading', language)}**")
        future_job_zone = st.selectbox(
            ui_text("initial.future_job_zone_question", language),
            options=list(JOB_ZONE_LABELS),
            format_func=lambda zone: _format_job_zone_for_language(zone, language),
            index=3,
        )
        submitted = st.form_submit_button(ui_text("initial.submit", language))

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


def _render_profile_result(result: dict[str, Any], language: str) -> None:
    _render_followup(result, language)
    _render_final_report_section(st.session_state.get("profile_result", result), language)


def _render_followup(result: dict[str, Any], language: str) -> None:
    ambiguity = result.get("score_ambiguity") or {}
    followup_refinement = result.get("followup_refinement")

    st.header(ui_text("followup.header", language))
    rag_ready, inspection = _rag_artifacts_ready()
    if not rag_ready and not followup_refinement:
        _render_missing_artifacts_message(inspection, language)
        return

    if followup_refinement and followup_refinement.get("method") == "in_progress":
        if not rag_ready:
            _render_missing_artifacts_message(inspection, language)
            return
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
        _render_followup_question(result, language)
        return

    if followup_refinement:
        _show_followup_refinement(followup_refinement, language)
        return

    if st.session_state.get("ip_followup_active"):
        if not rag_ready:
            _render_missing_artifacts_message(inspection, language)
            return
        _render_followup_question(result, language)
        return

    if st.session_state.get("ip_followup_complete"):
        return

    if ambiguity.get("has_ambiguity"):
        st.info(ui_text("followup.ambiguity_info", language))
    else:
        st.info(ui_text("followup.clear_info", language))

    continue_col, skip_col = st.columns(2)
    with continue_col:
        if st.button(ui_text("followup.continue", language)):
            st.session_state["ip_followup_plan"] = build_followup_question_plan(result)
            st.session_state["ip_followup_answers"] = []
            st.session_state["ip_followup_active"] = True
            st.session_state["ip_followup_complete"] = False
            st.rerun()
    with skip_col:
        if st.button(ui_text("followup.skip", language)):
            st.session_state["ip_followup_complete"] = True
            st.rerun()


def _render_followup_question(result: dict[str, Any], language: str) -> None:
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
        try:
            refinement = complete_followup_refinement(result, questions_asked)
            updated = update_profile_with_followup(result, refinement)
        except MissingRagArtifactsError as exc:
            st.error(_format_missing_artifacts_error_for_ui({"missing": exc.missing_artifacts}, language))
            return
        except Exception as exc:
            st.error(ui_text("followup.rag_error_prefix", language) + str(exc))
            return
        save_profile_result(updated, RESULT_PATH)
        _set_profile_result(updated, "session_state")
        st.session_state["final_report"] = None
        st.session_state["ip_final_career_report"] = None
        st.session_state["ip_followup_active"] = False
        st.session_state["ip_followup_complete"] = True
        st.rerun()
        return

    st.caption(
        ui_text(
            "followup.question_progress",
            language,
            current=len(questions_asked) + 1,
            maximum=MAX_FOLLOWUP_QUESTIONS,
        )
    )
    st.write(followup_question_text(current, language))
    with st.form(f"followup_question_{len(questions_asked)}"):
        answer = st.text_area(ui_text("followup.answer_label", language), key=f"followup_answer_{len(questions_asked)}")
        submitted = st.form_submit_button(ui_text("followup.submit_answer", language))

    if not submitted:
        return
    if not answer.strip():
        st.warning(ui_text("followup.empty_answer_warning", language))
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


def _show_followup_refinement(followup_refinement: dict[str, Any], language: str) -> None:
    if not followup_refinement.get("json_valid", True):
        st.warning(ui_text("followup.invalid_json_warning", language))


def _rag_artifacts_ready() -> tuple[bool, dict[str, Any]]:
    inspection = _inspect_rag_artifacts_cached()
    return bool(inspection.get("rag_enabled")), inspection


def _render_missing_artifacts_message(inspection: dict[str, Any], language: str) -> None:
    st.error(_format_missing_artifacts_error_for_ui(inspection, language))


def _format_missing_artifacts_error_for_ui(inspection: dict[str, Any], language: str) -> str:
    missing = list(inspection.get("missing") or [])
    if normalize_language(language) == "en":
        return format_missing_artifacts_error(missing)

    restore_command = inspection.get("restore_command") or r"python scripts\archives\restore_data_archives.py"
    build_command = inspection.get("build_command") or r"python scripts\build_all_artifacts.py"
    check_command = inspection.get("check_command") or r"python scripts\check_artifacts.py"
    lines = [
        ui_text("artifacts.missing_title", language),
        "",
        ui_text("artifacts.missing_list", language),
    ]
    for artifact in missing:
        detail = artifact.get("display_path") or artifact.get("path")
        count = artifact.get("count")
        error = artifact.get("error")
        suffix = ""
        if count is not None:
            suffix += f" (count={count})"
        if error:
            suffix += f" - {error}"
        lines.append(f"- {artifact_label(artifact, language)}: {detail}{suffix}")

    lines.extend(
        [
            "",
            ui_text("artifacts.use_prebuilt", language),
            f"  {restore_command}",
            "",
            ui_text("artifacts.rebuild", language),
            f"  {build_command}",
            "",
            ui_text("artifacts.verify", language),
            f"  {check_command}",
        ]
    )
    return "\n".join(lines)


def _load_saved_final_report() -> dict[str, Any] | None:
    if not FINAL_REPORT_PATH.exists():
        return None
    try:
        with FINAL_REPORT_PATH.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _render_final_report_section(result: dict[str, Any], language: str) -> None:
    st.header(ui_text("final.header", language))
    st.write(ui_text("final.description", language))

    if (result.get("followup_refinement") or {}).get("method") == "in_progress":
        st.info(ui_text("final.finish_followup_info", language))
        return

    rag_ready, inspection = _rag_artifacts_ready()
    if not rag_ready:
        _render_missing_artifacts_message(inspection, language)
        return

    if st.button(ui_text("final.generate_button", language)):
        with st.spinner(ui_text("final.spinner", language)):
            try:
                report = build_final_career_report(result, top_k=10)
                validation_errors = _validate_report_matches_profile(report, result)
                if validation_errors:
                    st.error(ui_text("final.validation_failed_prefix", language) + "; ".join(validation_errors))
                    return
            except Exception as exc:
                st.error(ui_text("final.generation_error_prefix", language) + str(exc))
                return
        st.session_state["final_report"] = report
        st.session_state["ip_final_career_report"] = report
        st.success(ui_text("final.success", language))

    report = st.session_state.get("final_report") or st.session_state.get("ip_final_career_report")
    if not report and not result:
        report = _load_saved_final_report()
    if report:
        validation_errors = _validate_report_matches_profile(report, result)
        if validation_errors:
            st.warning(ui_text("final.stale_warning_prefix", language) + "; ".join(validation_errors))
            return
        if not _report_uses_current_future_grouping(report):
            st.session_state["final_report"] = None
            st.session_state["ip_final_career_report"] = None
            report = None
    if report:
        render_final_career_report(report, language=language)


def _report_uses_current_future_grouping(report: dict[str, Any]) -> bool:
    if report.get("top_match_grouping") != "current_zone_5_future_zone_5":
        return False
    semantic_report = report.get("semantic_retrieval_report") or {}
    if report.get("semantic_report_method") != "semantic_onet_ai_impact_report":
        return False
    if semantic_report.get("method") != "semantic_onet_ai_impact_report":
        return False
    return all(match.get("match_group") for match in report.get("top_matches") or [])


def main() -> None:
    st.set_page_config(page_title="AI Aware Career Guide", layout="wide")
    language = _render_language_switch()
    _render_app_intro(language)

    questions = _load_questions()
    career_listings = _load_career_listings()

    _render_initial_form(questions, career_listings, language)
    result, _source = _get_current_profile_result()
    if result:
        _render_profile_result(result, language)


if __name__ == "__main__":
    main()
