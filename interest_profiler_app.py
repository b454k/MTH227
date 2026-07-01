"""Streamlit UI for the local O*NET Interest Profiler short form."""

from __future__ import annotations

import json
from html import escape
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


def _mobile_copy(key: str, language: str) -> str:
    """Return presentation-only labels for the mobile-style Streamlit shell."""
    labels = {
        "en": {
            "hero.eyebrow": "AI career guide",
            "hero.status": "Interest profile",
            "assessment.header": "Start with quick swipes",
            "assessment.caption": "Answer each work activity with a simple yes or no.",
            "assessment.activity_prompt": "Would you enjoy this activity?",
            "assessment.progress": "Question {current} of {total}",
            "assessment.previous_yes": "Previous answer: Yes",
            "assessment.previous_no": "Previous answer: No",
            "assessment.no": "No",
            "assessment.yes": "Yes",
            "assessment.back": "Back",
            "assessment.reset": "Reset",
            "assessment.done_title": "All activity cards answered",
            "assessment.done_text": "Choose the preparation levels to create your career matches.",
            "assessment.completed": "{answered} of {total} cards answered",
            "assessment.submit": "Build my matches",
            "assessment.spinner": "Building your matches...",
            "assessment.incomplete": "Answer every activity card before building matches.",
            "assessment.saved": "Profile updated from your card choices.",
            "profile.ready": "Profile ready",
            "profile.code": "Career code",
            "profile.current_zone": "Current zone",
            "profile.future_zone": "Future zone",
            "followup.kicker": "Refine",
            "final.kicker": "Report",
        },
        "tr": {
            "hero.eyebrow": "YZ kariyer rehberi",
            "hero.status": "Ilgi profili",
            "assessment.header": "Hizli kartlarla basla",
            "assessment.caption": "Her is etkinligini yalnizca evet veya hayir ile yanitla.",
            "assessment.activity_prompt": "Bu etkinligi yapmak ister miydin?",
            "assessment.progress": "Soru {current} / {total}",
            "assessment.previous_yes": "Onceki yanit: Evet",
            "assessment.previous_no": "Onceki yanit: Hayir",
            "assessment.no": "Hayir",
            "assessment.yes": "Evet",
            "assessment.back": "Geri",
            "assessment.reset": "Sifirla",
            "assessment.done_title": "Tum etkinlik kartlari tamamlandi",
            "assessment.done_text": "Kariyer eslesmelerini olusturmak icin hazirlik duzeylerini sec.",
            "assessment.completed": "{answered} / {total} kart yanitlandi",
            "assessment.submit": "Eslesmelerimi olustur",
            "assessment.spinner": "Eslesmelerin olusturuluyor...",
            "assessment.incomplete": "Eslesmeleri olusturmadan once tum etkinlik kartlarini yanitla.",
            "assessment.saved": "Profil kart secimlerinden guncellendi.",
            "profile.ready": "Profil hazir",
            "profile.code": "Kariyer kodu",
            "profile.current_zone": "Mevcut bolge",
            "profile.future_zone": "Hedef bolge",
            "followup.kicker": "Iyilestir",
            "final.kicker": "Rapor",
        },
    }
    normalized = normalize_language(language)
    return labels.get(normalized, labels["en"]).get(key, labels["en"].get(key, key))


def _safe_html(value: Any) -> str:
    return escape(str(value or ""))


def _render_global_styles() -> None:
    """Apply the dark phone-shell styling shared by the app and report."""
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #050505;
            --phone-bg: #0b0b0c;
            --panel: #141416;
            --panel-2: #1b1b1e;
            --line: #2a2a2f;
            --line-strong: #3a3a40;
            --text: #f4f4f5;
            --muted: #a4a4aa;
            --soft: #d6d6da;
            --white: #ffffff;
            --black: #050505;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--app-bg);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stHeader"], [data-testid="stToolbar"], footer {
            background: transparent;
        }

        .block-container {
            max-width: 460px;
            min-height: 100vh;
            padding: 1.2rem 1.05rem 3rem;
            background: var(--phone-bg);
            color: var(--text);
            border-left: 1px solid var(--line);
            border-right: 1px solid var(--line);
        }

        .block-container::before {
            content: "";
            display: block;
            width: 72px;
            height: 5px;
            margin: 0.1rem auto 1.15rem;
            border-radius: 999px;
            background: #2d2d31;
        }

        @media (min-width: 720px) {
            .block-container {
                min-height: 900px;
                margin-top: 1.8rem;
                margin-bottom: 1.8rem;
                border: 1px solid var(--line);
                border-radius: 34px;
                box-shadow: 0 28px 80px rgba(0, 0, 0, 0.55);
            }
        }

        @media (max-width: 560px) {
            .block-container {
                max-width: 100%;
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            letter-spacing: 0;
        }

        h1, h2, h3 {
            color: var(--text);
            font-weight: 800;
        }

        p, li, .stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {
            color: var(--soft);
        }

        a {
            color: var(--white) !important;
            text-decoration-color: #777;
        }

        hr {
            border-color: var(--line);
        }

        .mobile-hero {
            padding: 1.1rem 0 1.25rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
        }

        .mobile-eyebrow, .section-kicker, .card-kicker {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .mobile-hero h1 {
            font-size: 2rem;
            line-height: 1.02;
            margin: 0 0 0.65rem;
            color: var(--white);
        }

        .mobile-hero p {
            margin: 0;
            color: var(--muted);
            line-height: 1.5;
            font-size: 0.94rem;
        }

        .section-heading {
            margin: 1.15rem 0 0.7rem;
        }

        .section-heading h2 {
            font-size: 1.3rem;
            margin: 0.1rem 0 0.25rem;
            color: var(--white);
        }

        .section-heading p {
            margin: 0;
            color: var(--muted);
            line-height: 1.45;
        }

        .phone-card, .swipe-card, .report-card, .match-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 1rem;
            margin: 0.72rem 0;
        }

        .swipe-card {
            min-height: 300px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            background: #161618;
            border-color: var(--line-strong);
        }

        .swipe-progress-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .activity-prompt {
            margin: 1.6rem 0 0.4rem;
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 750;
        }

        .activity-text {
            color: var(--white);
            font-size: 1.78rem;
            line-height: 1.08;
            font-weight: 850;
            margin: 0;
            overflow-wrap: anywhere;
        }

        .answer-memory {
            margin-top: 1rem;
            color: var(--muted);
            font-size: 0.84rem;
        }

        .completion-title {
            margin: 0 0 0.4rem;
            color: var(--white);
            font-size: 1.25rem;
            font-weight: 850;
        }

        .completion-text {
            margin: 0;
            color: var(--muted);
            line-height: 1.45;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.75rem;
        }

        .metric-tile {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel-2);
            padding: 0.7rem 0.6rem;
            min-width: 0;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--white);
            font-size: 1rem;
            font-weight: 850;
            margin-top: 0.25rem;
            overflow-wrap: anywhere;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.75rem;
        }

        .ui-badge, .impact-badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--line-strong);
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            color: var(--white);
            background: #202024;
            font-size: 0.74rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .impact-high {
            background: #f4f4f5;
            border-color: #f4f4f5;
            color: #050505;
        }

        .impact-medium {
            background: #b8b8bf;
            border-color: #b8b8bf;
            color: #050505;
        }

        .impact-low, .impact-available {
            background: #242428;
            border-color: #4a4a50;
            color: #f4f4f5;
        }

        div.stButton > button,
        div.stFormSubmitButton > button {
            min-height: 3rem;
            border-radius: 8px;
            border: 1px solid var(--line-strong);
            background: #1b1b1e;
            color: var(--white);
            font-weight: 800;
            letter-spacing: 0;
            transition: border-color 120ms ease, background 120ms ease, transform 120ms ease;
        }

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {
            border-color: #f4f4f5;
            background: #222226;
            color: var(--white);
            transform: translateY(-1px);
        }

        div.stButton > button[kind="primary"],
        div.stFormSubmitButton > button[kind="primary"] {
            background: var(--white);
            color: var(--black);
            border-color: var(--white);
        }

        div.stButton > button[kind="primary"]:hover,
        div.stFormSubmitButton > button[kind="primary"]:hover {
            background: #dddddf;
            color: var(--black);
            border-color: #dddddf;
        }

        [data-testid="stProgress"] > div > div > div {
            background: var(--white);
        }

        [data-testid="stProgress"] > div > div {
            background: #242428;
        }

        [data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 1rem;
        }

        [data-baseweb="select"] > div,
        textarea,
        input {
            background: #101012 !important;
            border-color: var(--line-strong) !important;
            color: var(--white) !important;
            border-radius: 8px !important;
        }

        [data-baseweb="select"] span,
        [data-baseweb="select"] svg {
            color: var(--white) !important;
            fill: var(--white) !important;
        }

        textarea:focus,
        input:focus {
            border-color: var(--white) !important;
            box-shadow: none !important;
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--line-strong);
            background: #171719;
            color: var(--text);
        }

        [data-testid="stSpinner"] {
            color: var(--white);
        }

        [data-testid="stSpinner"] svg {
            color: var(--white);
            fill: var(--white);
        }

        details {
            border: 1px solid var(--line) !important;
            border-radius: 8px !important;
            background: var(--panel) !important;
        }

        details summary {
            color: var(--white) !important;
            font-weight: 800;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            overflow-x: auto;
        }

        .stTabs [data-baseweb="tab"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #151517;
            color: var(--muted);
            padding: 0.45rem 0.75rem;
            min-width: max-content;
        }

        .stTabs [aria-selected="true"] {
            background: var(--white);
            color: var(--black);
            border-color: var(--white);
        }

        .wrapped-report-table {
            background: #101012;
            color: var(--soft);
        }

        .wrapped-report-table th {
            background: #202024 !important;
            color: var(--white);
            border-color: var(--line-strong) !important;
        }

        .wrapped-report-table td {
            border-color: var(--line) !important;
        }

        .skill-chip {
            border-color: var(--line-strong) !important;
            background: #202024 !important;
            color: var(--white) !important;
            border-radius: 999px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_header_html(kicker: str, title: str, caption: str = "") -> str:
    caption_html = f"<p>{_safe_html(caption)}</p>" if caption else ""
    return (
        "<div class='section-heading'>"
        f"<div class='section-kicker'>{_safe_html(kicker)}</div>"
        f"<h2>{_safe_html(title)}</h2>"
        f"{caption_html}"
        "</div>"
    )


def _metric_tile_html(label: str, value: Any) -> str:
    return (
        "<div class='metric-tile'>"
        f"<div class='metric-label'>{_safe_html(label)}</div>"
        f"<div class='metric-value'>{_safe_html(value)}</div>"
        "</div>"
    )


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
    st.markdown(
        (
            "<div class='mobile-hero'>"
            f"<div class='mobile-eyebrow'>{_safe_html(_mobile_copy('hero.eyebrow', language))}</div>"
            f"<h1>{_safe_html(ui_text('app.title', language))}</h1>"
            f"<p>{_safe_html(ui_text('app.caption', language))}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
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


def _ensure_interest_card_state(questions: list[dict[str, Any]]) -> None:
    valid_ids = {int(question["id"]) for question in questions}
    raw_answers = st.session_state.get("ip_card_answers")
    if not isinstance(raw_answers, dict):
        raw_answers = {}

    answers: dict[int, bool] = {}
    for raw_id, liked in raw_answers.items():
        try:
            question_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if question_id in valid_ids and isinstance(liked, bool):
            answers[question_id] = liked
    st.session_state["ip_card_answers"] = answers

    try:
        index = int(st.session_state.get("ip_card_index", 0))
    except (TypeError, ValueError):
        index = 0
    index = max(0, min(index, len(questions)))
    if index >= len(questions) and len(answers) < len(questions):
        index = _first_unanswered_index(questions, answers)
    st.session_state["ip_card_index"] = index


def _first_unanswered_index(questions: list[dict[str, Any]], answers: dict[int, bool]) -> int:
    for index, question in enumerate(questions):
        if int(question["id"]) not in answers:
            return index
    return len(questions)


def _answered_card_count(questions: list[dict[str, Any]]) -> int:
    answers = st.session_state.get("ip_card_answers") or {}
    valid_ids = {int(question["id"]) for question in questions}
    return sum(1 for question_id in valid_ids if question_id in answers)


def _liked_question_ids(questions: list[dict[str, Any]]) -> list[int]:
    answers = st.session_state.get("ip_card_answers") or {}
    return [
        int(question["id"])
        for question in questions
        if answers.get(int(question["id"])) is True
    ]


def _record_interest_card_answer(question_id: int, liked: bool, questions: list[dict[str, Any]]) -> None:
    answers = st.session_state.setdefault("ip_card_answers", {})
    answers[int(question_id)] = bool(liked)
    current_index = int(st.session_state.get("ip_card_index", 0))
    next_index = len(questions)
    for index in range(current_index + 1, len(questions)):
        if int(questions[index]["id"]) not in answers:
            next_index = index
            break
    if next_index == len(questions) and current_index + 1 < len(questions):
        next_index = current_index + 1
    st.session_state["ip_card_index"] = min(next_index, len(questions))
    st.session_state["ip_assessment_in_progress"] = True
    st.session_state["ip_assessment_submitted"] = False


def _reset_interest_cards() -> None:
    st.session_state["ip_card_answers"] = {}
    st.session_state["ip_card_index"] = 0
    st.session_state["ip_assessment_in_progress"] = False
    st.session_state["ip_assessment_submitted"] = False


def _submit_interest_profile(
    questions: list[dict[str, Any]],
    career_listings: list[dict[str, Any]],
    current_job_zone: int,
    future_job_zone: int,
) -> None:
    checked_ids = _liked_question_ids(questions)
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
    st.session_state["ip_assessment_in_progress"] = False
    st.session_state["ip_assessment_submitted"] = True


def _render_profile_summary_card(result: dict[str, Any], language: str) -> None:
    top_interests = result.get("final_top_interests") or result.get("top_interests") or []
    code = result.get("final_holland_code") or result.get("holland_code") or ""
    current_zone = result.get("current_job_zone") or result.get("current_zone") or ""
    future_zone = result.get("future_job_zone") or result.get("future_zone") or ""
    badges = "".join(
        f"<span class='ui-badge'>{_safe_html(interest_label(interest, language))}</span>"
        for interest in top_interests[:3]
    )
    metrics = "".join(
        [
            _metric_tile_html(_mobile_copy("profile.code", language), code),
            _metric_tile_html(_mobile_copy("profile.current_zone", language), current_zone),
            _metric_tile_html(_mobile_copy("profile.future_zone", language), future_zone),
        ]
    )
    st.markdown(
        (
            "<div class='phone-card'>"
            f"<div class='card-kicker'>{_safe_html(_mobile_copy('hero.status', language))}</div>"
            f"<div class='completion-title'>{_safe_html(_mobile_copy('profile.ready', language))}</div>"
            f"<div class='badge-row'>{badges}</div>"
            f"<div class='metric-strip'>{metrics}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_initial_form(
    questions: list[dict[str, Any]],
    career_listings: list[dict[str, Any]],
    language: str,
) -> None:
    if st.session_state.get("ip_assessment_submitted") and st.session_state.get("profile_result"):
        st.session_state["ip_assessment_in_progress"] = False
        return

    _ensure_interest_card_state(questions)
    total = len(questions)
    answered = _answered_card_count(questions)

    st.markdown(
        _section_header_html(
            ui_text("initial.activities_header", language),
            _mobile_copy("assessment.header", language),
            _mobile_copy("assessment.caption", language),
        ),
        unsafe_allow_html=True,
    )

    if not questions:
        st.warning(ui_text("report.no_local_evidence", language))
        return

    if answered < total:
        st.session_state["ip_assessment_in_progress"] = answered > 0
        index = int(st.session_state.get("ip_card_index", 0))
        index = max(0, min(index, total - 1))
        question = questions[index]
        question_id = int(question["id"])
        answers = st.session_state.get("ip_card_answers") or {}
        previous_answer = answers.get(question_id)
        previous_html = ""
        if previous_answer is True:
            previous_html = (
                f"<div class='answer-memory'>{_safe_html(_mobile_copy('assessment.previous_yes', language))}</div>"
            )
        elif previous_answer is False:
            previous_html = (
                f"<div class='answer-memory'>{_safe_html(_mobile_copy('assessment.previous_no', language))}</div>"
            )

        current_label = _mobile_copy("assessment.progress", language).format(
            current=index + 1,
            total=total,
        )
        answered_label = _mobile_copy("assessment.completed", language).format(
            answered=answered,
            total=total,
        )
        st.markdown(
            (
                "<div class='swipe-card'>"
                "<div>"
                "<div class='swipe-progress-row'>"
                f"<span>{_safe_html(current_label)}</span>"
                f"<span>{_safe_html(answered_label)}</span>"
                "</div>"
                f"<div class='activity-prompt'>{_safe_html(_mobile_copy('assessment.activity_prompt', language))}</div>"
                f"<p class='activity-text'>{_safe_html(question_text(question, language))}</p>"
                "</div>"
                f"{previous_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.progress(answered / total)

        no_col, yes_col = st.columns(2)
        with no_col:
            if st.button(
                _mobile_copy("assessment.no", language),
                key=f"activity_no_{question_id}",
                use_container_width=True,
            ):
                _record_interest_card_answer(question_id, False, questions)
                st.rerun()
        with yes_col:
            if st.button(
                _mobile_copy("assessment.yes", language),
                key=f"activity_yes_{question_id}",
                type="primary",
                use_container_width=True,
            ):
                _record_interest_card_answer(question_id, True, questions)
                st.rerun()

        back_col, reset_col = st.columns(2)
        with back_col:
            if st.button(
                _mobile_copy("assessment.back", language),
                key="activity_back",
                disabled=index <= 0,
                use_container_width=True,
            ):
                st.session_state["ip_card_index"] = max(0, index - 1)
                st.rerun()
        with reset_col:
            if st.button(
                _mobile_copy("assessment.reset", language),
                key="activity_reset",
                use_container_width=True,
            ):
                _reset_interest_cards()
                st.rerun()
        return

    st.session_state["ip_assessment_in_progress"] = True
    st.progress(1.0)
    st.markdown(
        (
            "<div class='phone-card'>"
            f"<div class='completion-title'>{_safe_html(_mobile_copy('assessment.done_title', language))}</div>"
            f"<p class='completion-text'>{_safe_html(_mobile_copy('assessment.done_text', language))}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.form("interest_profiler_zone_form"):
        st.markdown(f"**{ui_text('initial.current_job_zone_heading', language)}**")
        current_job_zone = st.selectbox(
            ui_text("initial.current_job_zone_question", language),
            options=list(JOB_ZONE_LABELS),
            format_func=lambda zone: _format_job_zone_for_language(zone, language),
            index=2,
            key="ip_current_job_zone",
        )
        st.markdown(f"**{ui_text('initial.future_job_zone_heading', language)}**")
        future_job_zone = st.selectbox(
            ui_text("initial.future_job_zone_question", language),
            options=list(JOB_ZONE_LABELS),
            format_func=lambda zone: _format_job_zone_for_language(zone, language),
            index=3,
            key="ip_future_job_zone",
        )
        submitted = st.form_submit_button(
            _mobile_copy("assessment.submit", language),
            use_container_width=True,
        )

    reset_col, _ = st.columns([1, 1])
    with reset_col:
        if st.button(
            _mobile_copy("assessment.reset", language),
            key="activity_reset_after_complete",
            use_container_width=True,
        ):
            _reset_interest_cards()
            st.rerun()

    if not submitted:
        return
    if _answered_card_count(questions) < total:
        st.warning(_mobile_copy("assessment.incomplete", language))
        return
    with st.spinner(_mobile_copy("assessment.spinner", language)):
        _submit_interest_profile(questions, career_listings, current_job_zone, future_job_zone)
    st.success(_mobile_copy("assessment.saved", language))
    st.rerun()


def _render_profile_result(result: dict[str, Any], language: str) -> None:
    _render_profile_summary_card(result, language)
    _render_followup(result, language)
    _render_final_report_section(st.session_state.get("profile_result", result), language)


def _render_followup(result: dict[str, Any], language: str) -> None:
    ambiguity = result.get("score_ambiguity") or {}
    followup_refinement = result.get("followup_refinement")

    st.markdown(
        _section_header_html(
            _mobile_copy("followup.kicker", language),
            ui_text("followup.header", language),
        ),
        unsafe_allow_html=True,
    )
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
        if st.button(ui_text("followup.continue", language), use_container_width=True):
            st.session_state["ip_followup_plan"] = build_followup_question_plan(result)
            st.session_state["ip_followup_answers"] = []
            st.session_state["ip_followup_active"] = True
            st.session_state["ip_followup_complete"] = False
            st.rerun()
    with skip_col:
        if st.button(ui_text("followup.skip", language), use_container_width=True):
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

    progress_text = ui_text(
        "followup.question_progress",
        language,
        current=len(questions_asked) + 1,
        maximum=MAX_FOLLOWUP_QUESTIONS,
    )
    st.markdown(
        (
            "<div class='phone-card'>"
            f"<div class='card-kicker'>{_safe_html(progress_text)}</div>"
            f"<p class='activity-text'>{_safe_html(followup_question_text(current, language))}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.form(f"followup_question_{len(questions_asked)}"):
        answer = st.text_area(ui_text("followup.answer_label", language), key=f"followup_answer_{len(questions_asked)}")
        submitted = st.form_submit_button(ui_text("followup.submit_answer", language), use_container_width=True)

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
    st.markdown(
        _section_header_html(
            _mobile_copy("final.kicker", language),
            ui_text("final.header", language),
            ui_text("final.description", language),
        ),
        unsafe_allow_html=True,
    )

    if (result.get("followup_refinement") or {}).get("method") == "in_progress":
        st.info(ui_text("final.finish_followup_info", language))
        return

    rag_ready, inspection = _rag_artifacts_ready()
    if not rag_ready:
        _render_missing_artifacts_message(inspection, language)
        return

    if st.button(ui_text("final.generate_button", language), type="primary", use_container_width=True):
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
    _render_global_styles()
    language = _render_language_switch()
    _render_app_intro(language)

    questions = _load_questions()
    career_listings = _load_career_listings()

    _render_initial_form(questions, career_listings, language)
    result, _source = _get_current_profile_result()
    if result and not st.session_state.get("ip_assessment_in_progress"):
        _render_profile_result(result, language)


if __name__ == "__main__":
    main()
