"""Smoke tests for the local PDF-based O*NET Interest Profiler flow."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from career_rag.config import PROJECT_ROOT
from career_rag.interest_profiler_local import (
    RIASEC_INTERESTS,
    build_profile_result,
    calculate_riasec_scores,
    detect_ambiguous_categories,
    load_interest_profiler_questions,
    prepare_profile_for_rag,
    save_profile_result,
    update_profile_with_followup,
)
from career_rag.ip_career_matcher import (
    build_career_matches,
    load_ip_career_listings,
    match_careers,
)
from career_rag.ip_followup_agent import (
    build_followup_question_plan,
    complete_followup_refinement,
)


DATA_DIR = PROJECT_ROOT / "onet_interest_profiler"
QUESTIONS_PATH = DATA_DIR / "interest_profiler_questions.json"
CAREER_LISTINGS_PATH = DATA_DIR / "ip_career_listings.json"
RESULT_PATH = DATA_DIR / "ip_profile_result.json"


class UnavailableProvider:
    available = False


def _assert_interest_counts(questions: list[dict]) -> None:
    counts = Counter(question["area"] for question in questions)
    assert len(questions) == 60
    for interest in RIASEC_INTERESTS:
        assert counts[interest] == 10, f"{interest} count was {counts[interest]}"


def _question_ids_for_area(questions: list[dict], area: str, count: int) -> list[int]:
    return [question["id"] for question in questions if question["area"] == area][:count]


def test_scoring_and_ambiguity(questions: list[dict]) -> None:
    checked_ids = [
        *_question_ids_for_area(questions, "Investigative", 5),
        *_question_ids_for_area(questions, "Conventional", 5),
        *_question_ids_for_area(questions, "Social", 4),
    ]
    scores = calculate_riasec_scores(questions, checked_ids)
    assert scores["Investigative"] == 5
    assert scores["Conventional"] == 5
    assert scores["Social"] == 4
    exact_tie = detect_ambiguous_categories(scores)
    assert exact_tie["has_ambiguity"] is True
    assert "Investigative" in exact_tie["ambiguous_categories"]
    assert "Conventional" in exact_tie["ambiguous_categories"]

    near_tie_scores = {
        "Realistic": 2,
        "Investigative": 8,
        "Artistic": 3,
        "Social": 4,
        "Enterprising": 2,
        "Conventional": 7,
    }
    near_tie = detect_ambiguous_categories(near_tie_scores)
    assert near_tie["has_ambiguity"] is True
    assert near_tie["ambiguous_categories"] == ["Investigative", "Conventional"]

    clear_scores = {
        "Realistic": 1,
        "Investigative": 9,
        "Artistic": 2,
        "Social": 4,
        "Enterprising": 1,
        "Conventional": 6,
    }
    no_tie = detect_ambiguous_categories(clear_scores)
    assert no_tie["has_ambiguity"] is False
    assert no_tie["ambiguous_categories"] == []

    empty_scores = calculate_riasec_scores(questions, [])
    assert empty_scores == {interest: 0 for interest in RIASEC_INTERESTS}
    assert detect_ambiguous_categories(empty_scores)["has_ambiguity"] is True


def test_career_matching_and_profile(questions: list[dict], listings: list[dict]) -> None:
    checked_ids = [
        *_question_ids_for_area(questions, "Investigative", 8),
        *_question_ids_for_area(questions, "Conventional", 7),
        *_question_ids_for_area(questions, "Social", 5),
        *_question_ids_for_area(questions, "Realistic", 3),
        *_question_ids_for_area(questions, "Enterprising", 4),
        *_question_ids_for_area(questions, "Artistic", 2),
    ]
    scores = calculate_riasec_scores(questions, checked_ids)
    direct_matches = match_careers(listings, "Investigative", 4)
    assert all(item["interest"] == "Investigative" and item["job_zone"] == 4 for item in direct_matches)

    career_matches = build_career_matches(scores, current_job_zone=4, future_job_zone=5, career_listings=listings)
    result = build_profile_result(scores, 4, 5, career_matches)
    assert result["source"] == "onet_interest_profiler_short_form_pdf_local"
    assert result["raw_riasec_scores"] == scores
    assert result["followup_refinement"] is None
    assert result["initial_top_interests"] == ["Investigative", "Conventional", "Social"]
    assert result["final_top_interests"] == result["initial_top_interests"]
    assert result["ready_for_rag"] is True

    saved_path = save_profile_result(result, RESULT_PATH)
    assert saved_path.exists()

    plan = build_followup_question_plan(result)
    assert plan
    questions_asked = [
        {"stage": plan[0]["stage"], "question": plan[0]["question"], "answer": "I research first."}
    ]
    template_refinement = complete_followup_refinement(
        result,
        questions_asked,
        provider=UnavailableProvider(),
    )
    updated = update_profile_with_followup(result, template_refinement)
    assert updated["raw_riasec_scores"] == scores
    assert updated["initial_top_interests"] == ["Investigative", "Conventional", "Social"]
    assert updated["followup_refinement"]["method"] == "template_fallback"

    profile_for_rag = prepare_profile_for_rag(updated)
    assert profile_for_rag["holland_code"] == updated["final_holland_code"]
    assert isinstance(profile_for_rag["career_titles_to_retrieve"], list)


def test_invalid_job_zone(listings: list[dict]) -> None:
    try:
        match_careers(listings, "Investigative", 6)
    except ValueError as exc:
        assert "Job Zone" in str(exc)
    else:
        raise AssertionError("Invalid Job Zone did not raise ValueError.")


def main() -> int:
    questions = load_interest_profiler_questions(QUESTIONS_PATH)
    listings = load_ip_career_listings(CAREER_LISTINGS_PATH)

    _assert_interest_counts(questions)
    test_scoring_and_ambiguity(questions)
    test_career_matching_and_profile(questions, listings)
    test_invalid_job_zone(listings)

    print("Interest Profiler local smoke test passed.")
    print(f"Questions: {len(questions)}")
    print(f"Career listings: {len(listings)}")
    print(f"Saved result: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
