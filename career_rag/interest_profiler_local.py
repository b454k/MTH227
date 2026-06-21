"""Local O*NET Interest Profiler scoring and profile-result helpers."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


RIASEC_INTERESTS = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
]
RIASEC_CODES = {
    "Realistic": "R",
    "Investigative": "I",
    "Artistic": "A",
    "Social": "S",
    "Enterprising": "E",
    "Conventional": "C",
}
RIASEC_BY_CODE = {code: interest for interest, code in RIASEC_CODES.items()}
PROFILE_SOURCE = "onet_interest_profiler_short_form_pdf_local"
CAREER_MATCH_KEYS = [
    "primary_current_zone",
    "primary_future_zone",
    "secondary_future_zone",
    "tertiary_future_zone",
]
RANKED_MATCH_KEYS = [
    "primary_future_zone",
    "primary_current_zone",
    "secondary_future_zone",
    "tertiary_future_zone",
]


def canonical_interest(value: str) -> str:
    """Return a normalized RIASEC interest name from a name or one-letter code."""
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("RIASEC interest cannot be empty.")

    upper = normalized.upper()
    if upper in RIASEC_BY_CODE:
        return RIASEC_BY_CODE[upper]

    for interest in RIASEC_INTERESTS:
        if interest.lower() == normalized.lower():
            return interest
    raise ValueError(f"Unknown RIASEC interest area: {value!r}")


def validate_interest_profiler_questions(questions: Any) -> list[dict[str, Any]]:
    """Validate and return the 60 local O*NET Interest Profiler activities."""
    if not isinstance(questions, list):
        raise ValueError("Interest Profiler questions JSON must contain a list.")
    if len(questions) != 60:
        raise ValueError(f"Expected exactly 60 Interest Profiler questions, got {len(questions)}.")

    ids: list[int] = []
    area_counts: Counter[str] = Counter()
    validated: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} must be an object.")

        missing = {"id", "area", "area_code", "text"} - set(question)
        if missing:
            raise ValueError(f"Question {index} is missing fields: {sorted(missing)}.")

        question_id = question["id"]
        if not isinstance(question_id, int):
            raise ValueError(f"Question {index} id must be an integer.")
        ids.append(question_id)

        area = canonical_interest(str(question["area"]))
        area_code = str(question["area_code"]).strip().upper()
        if area_code != RIASEC_CODES[area]:
            raise ValueError(
                f"Question {question_id} has area_code {area_code!r}, expected {RIASEC_CODES[area]!r}."
            )

        text = str(question["text"]).strip()
        if not text:
            raise ValueError(f"Question {question_id} text cannot be empty.")

        area_counts[area] += 1
        validated.append(
            {
                "id": question_id,
                "area": area,
                "area_code": area_code,
                "text": text,
            }
        )

    expected_ids = list(range(1, 61))
    if ids != expected_ids:
        raise ValueError("Interest Profiler question IDs must be ordered from 1 to 60.")
    if len(set(ids)) != 60:
        raise ValueError("Interest Profiler question IDs must be unique.")

    invalid_counts = {
        area: area_counts[area]
        for area in RIASEC_INTERESTS
        if area_counts[area] != 10
    }
    if invalid_counts:
        raise ValueError(f"Each RIASEC area must have exactly 10 questions: {invalid_counts}.")

    return validated


def load_interest_profiler_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load the 60 short-form Interest Profiler activities from local JSON."""
    questions_path = Path(path)
    with questions_path.open("r", encoding="utf-8") as file:
        questions = json.load(file)
    return validate_interest_profiler_questions(questions)


def _canonical_scores(scores: dict[str, Any]) -> dict[str, int]:
    canonical_scores = {interest: 0 for interest in RIASEC_INTERESTS}
    for interest, score in scores.items():
        canonical_scores[canonical_interest(str(interest))] = int(score)
    return canonical_scores


def calculate_riasec_scores(
    questions: list[dict[str, Any]],
    checked_ids: set[int | str] | list[int | str] | tuple[int | str, ...],
) -> dict[str, int]:
    """Return RIASEC scores, where each checked activity counts as one point."""
    validated_questions = validate_interest_profiler_questions(questions)
    valid_ids = {question["id"] for question in validated_questions}

    checked: set[int] = set()
    for question_id in checked_ids:
        try:
            checked_id = int(question_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checked question id must be an integer: {question_id!r}") from exc
        if checked_id not in valid_ids:
            raise ValueError(f"Checked question id is not in the profiler: {checked_id}.")
        checked.add(checked_id)

    scores = {interest: 0 for interest in RIASEC_INTERESTS}
    for question in validated_questions:
        if question["id"] in checked:
            scores[question["area"]] += 1

    return scores


def get_top_interests(scores: dict[str, Any], top_n: int = 3) -> list[str]:
    """Return top RIASEC interests using fixed RIASEC order for deterministic ties."""
    if top_n < 1:
        return []

    canonical_scores = _canonical_scores(scores)
    sorted_interests = sorted(
        RIASEC_INTERESTS,
        key=lambda interest: (-canonical_scores[interest], RIASEC_INTERESTS.index(interest)),
    )
    return sorted_interests[:top_n]


def make_holland_code(top_interests: list[str] | tuple[str, ...]) -> str:
    """Convert interest names to their Holland code letters."""
    return "".join(RIASEC_CODES[canonical_interest(interest)] for interest in top_interests)


def detect_ambiguous_categories(scores: dict[str, Any]) -> dict[str, Any]:
    """Detect exact ties or near ties among the highest-scoring RIASEC categories."""
    canonical_scores = _canonical_scores(scores)
    top_score = max(canonical_scores.values())
    ambiguous_categories = [
        interest
        for interest in RIASEC_INTERESTS
        if top_score - canonical_scores[interest] <= 1
    ]
    has_ambiguity = len(ambiguous_categories) >= 2
    return {
        "has_ambiguity": has_ambiguity,
        "reason": "Top categories are tied or within 1 point." if has_ambiguity else "",
        "ambiguous_categories": ambiguous_categories if has_ambiguity else [],
    }


def _validate_job_zone(job_zone: Any) -> int:
    try:
        zone = int(job_zone)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Job Zone must be an integer 1-5: {job_zone!r}") from exc
    if zone not in {1, 2, 3, 4, 5}:
        raise ValueError(f"Job Zone must be an integer 1-5: {zone}.")
    return zone


def _normalize_career_matches(
    career_matches: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    career_matches = career_matches or {}
    normalized: dict[str, list[dict[str, Any]]] = {}
    for key in CAREER_MATCH_KEYS:
        matches = career_matches.get(key, [])
        if not isinstance(matches, list):
            raise ValueError(f"career_matches[{key!r}] must be a list.")
        normalized[key] = matches
    return normalized


def build_profile_result(
    scores: dict[str, Any],
    current_job_zone: int,
    future_job_zone: int,
    career_matches: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    """Build the structured local Interest Profiler result object."""
    canonical_scores = _canonical_scores(scores)
    top_interests = get_top_interests(canonical_scores, top_n=3)
    holland_code = make_holland_code(top_interests)
    timestamp = datetime.now(timezone.utc).isoformat()
    profile_id = uuid4().hex
    normalized_matches = _normalize_career_matches(career_matches)
    initial_ranked_matches = _flatten_career_match_records(normalized_matches)

    return {
        "source": PROFILE_SOURCE,
        "profile_id": profile_id,
        "timestamp": timestamp,
        "riasec_scores": canonical_scores,
        "raw_riasec_scores": canonical_scores,
        "score_ambiguity": detect_ambiguous_categories(canonical_scores),
        "top_interests": top_interests,
        "initial_top_interests": top_interests,
        "holland_code": holland_code,
        "initial_code": holland_code,
        "initial_holland_code": holland_code,
        "current_job_zone": _validate_job_zone(current_job_zone),
        "future_job_zone": _validate_job_zone(future_job_zone),
        "career_matches": normalized_matches,
        "followup_refinement": None,
        "refined_interests": top_interests,
        "final_code": holland_code,
        "preferences_used": [],
        "final_top_interests": top_interests,
        "final_holland_code": holland_code,
        "refined_career_matches": initial_ranked_matches,
        "final_ranked_matches": initial_ranked_matches,
        "refinement_debug": {
            "initial_code": holland_code,
            "final_code": holland_code,
            "followup_refinement": None,
            "preferences_used": [],
            "top_10_initial_matches": _debug_match_rows(initial_ranked_matches[:10]),
            "top_10_refined_matches": _debug_match_rows(initial_ranked_matches[:10]),
            "score_components": [],
            "validation_warnings": [],
        },
        "validation_warnings": [],
        "ready_for_rag": True,
    }


def update_profile_with_followup(
    profile_result: dict[str, Any],
    followup_refinement: dict[str, Any],
) -> dict[str, Any]:
    """Attach follow-up refinement and rebuild final career ranking."""
    updated = dict(profile_result)
    updated["followup_refinement"] = followup_refinement

    final_refinement = followup_refinement.get("final_refinement") or {}
    questions_asked = list(followup_refinement.get("questions_asked") or [])
    initial_top = [canonical_interest(interest) for interest in updated.get("initial_top_interests", [])]
    proposed_top = final_refinement.get("refined_top_interests") or initial_top
    if questions_asked and _refined_interests_justified(updated, final_refinement, proposed_top):
        refined_top = [canonical_interest(interest) for interest in proposed_top][:3]
    else:
        refined_top = initial_top
    refined_holland = make_holland_code(refined_top)

    updated["final_top_interests"] = refined_top
    updated["final_holland_code"] = refined_holland
    updated["refined_interests"] = updated["final_top_interests"]
    updated["final_code"] = updated["final_holland_code"]

    from career_rag.ip_refinement_ranker import build_refined_career_recommendations

    ranking_update = build_refined_career_recommendations(updated, followup_refinement)

    updated["preferences_used"] = ranking_update.get("preferences_used", [])
    updated["refined_career_matches"] = ranking_update.get("refined_career_matches", [])
    updated["final_ranked_matches"] = ranking_update.get("final_ranked_matches", [])
    updated["refinement_debug"] = ranking_update.get("refinement_debug", {})
    updated["validation_warnings"] = ranking_update.get("validation_warnings", [])
    updated["ready_for_rag"] = True
    return updated


def save_profile_result(result: dict[str, Any], path: str | Path) -> Path:
    """Save the structured Interest Profiler result as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=True)
        file.write("\n")
    return output_path


def _flatten_career_titles(career_matches: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for key in RANKED_MATCH_KEYS:
        for career in career_matches.get(key, []) or []:
            title = str(career.get("career_title") or "").strip()
            if title and title.lower() not in seen:
                seen.add(title.lower())
                titles.append(title)
    return titles


def _flatten_career_match_records(career_matches: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in RANKED_MATCH_KEYS:
        for group_rank, career in enumerate(career_matches.get(key, []) or [], start=1):
            title = str(career.get("career_title") or "").strip()
            if not title:
                continue
            title_key = title.lower()
            if title_key in seen:
                continue
            seen.add(title_key)
            records.append(
                {
                    **career,
                    "career_title": title,
                    "source_match_group": key,
                    "original_rank": len(records) + 1,
                    "original_group_rank": group_rank,
                    "ranking_explanation": {
                        "original_profile": (
                            f"Originally appeared in the O*NET Interest Profiler candidate pool "
                            f"for {career.get('interest')} / Job Zone {career.get('job_zone')}."
                        ),
                        "followup_effects": [],
                    },
                }
            )
    return records


def _debug_match_rows(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "career_title": item.get("career_title"),
            "interest": item.get("interest"),
            "job_zone": item.get("job_zone"),
            "score": item.get("score"),
        }
        for item in matches
    ]


def prepare_profile_for_rag(profile_result: dict[str, Any]) -> dict[str, Any]:
    """Prepare structured profile context for later career RAG integration."""
    raw_scores = profile_result.get("riasec_scores") or profile_result.get("raw_riasec_scores") or {}
    scores = _canonical_scores(raw_scores)

    followup_refinement = profile_result.get("followup_refinement")
    if followup_refinement:
        top_interests = (
            profile_result.get("refined_interests")
            or profile_result.get("final_top_interests")
            or profile_result.get("top_interests")
            or profile_result.get("initial_top_interests")
        )
        holland_code = (
            profile_result.get("final_code")
            or profile_result.get("final_holland_code")
            or profile_result.get("holland_code")
            or profile_result.get("initial_holland_code")
        )
    else:
        top_interests = profile_result.get("top_interests") or profile_result.get("initial_top_interests") or get_top_interests(scores, top_n=3)
        holland_code = (
            profile_result.get("holland_code")
            or profile_result.get("initial_code")
            or profile_result.get("initial_holland_code")
            or make_holland_code(top_interests)
        )

    top_interests = [canonical_interest(interest) for interest in top_interests]
    holland_code = str(holland_code or make_holland_code(top_interests))
    current_job_zone = _validate_job_zone(profile_result.get("current_job_zone"))
    future_job_zone = _validate_job_zone(profile_result.get("future_job_zone"))
    career_titles = [
        str(item.get("career_title") or "").strip()
        for item in profile_result.get("final_ranked_matches") or []
        if str(item.get("career_title") or "").strip()
    ]
    if not career_titles:
        career_titles = _flatten_career_titles(profile_result.get("career_matches") or {})

    final_refinement = (followup_refinement or {}).get("final_refinement") or {}
    guidance = final_refinement.get("career_matching_guidance") or {}
    notes = str(guidance.get("notes") or "").strip()
    sub_preferences = final_refinement.get("key_sub_preferences") or []

    score_text = ", ".join(f"{interest}: {scores[interest]}" for interest in RIASEC_INTERESTS)
    summary_parts = [
        f"Raw RIASEC scores: {score_text}.",
        f"Top interests for prompting: {', '.join(top_interests)} ({holland_code}).",
        f"Current Job Zone: {current_job_zone}; Future Job Zone: {future_job_zone}.",
    ]
    if sub_preferences:
        summary_parts.append("Follow-up sub-preferences: " + "; ".join(map(str, sub_preferences)) + ".")
    if notes:
        summary_parts.append(f"Career matching guidance: {notes}")

    return {
        "riasec_scores": scores,
        "top_interests": top_interests,
        "holland_code": holland_code,
        "current_job_zone": current_job_zone,
        "future_job_zone": future_job_zone,
        "career_titles_to_retrieve": career_titles,
        "profile_summary_for_prompt": " ".join(summary_parts),
    }


def _refined_interests_justified(
    profile_result: dict[str, Any],
    final_refinement: dict[str, Any],
    proposed_top: Any,
) -> bool:
    try:
        proposed = [canonical_interest(interest) for interest in list(proposed_top)[:3]]
    except (TypeError, ValueError):
        return False
    initial_top = [canonical_interest(interest) for interest in profile_result.get("initial_top_interests", [])]
    if proposed == initial_top:
        return True

    ambiguity = profile_result.get("score_ambiguity") or {}
    ambiguous = [canonical_interest(interest) for interest in ambiguity.get("ambiguous_categories", [])]
    guidance = final_refinement.get("career_matching_guidance") or {}
    prioritized = []
    for interest in guidance.get("prioritize_interests") or []:
        try:
            prioritized.append(canonical_interest(interest))
        except ValueError:
            continue
    tie_resolution = final_refinement.get("tie_resolution") or {}
    resolved_order = []
    for interest in tie_resolution.get("resolved_order") or []:
        try:
            resolved_order.append(canonical_interest(interest))
        except ValueError:
            continue

    if ambiguity.get("has_ambiguity") and set(proposed).issubset(set(initial_top) | set(ambiguous)):
        return bool(resolved_order or prioritized)
    return bool(prioritized and proposed[: len(prioritized)] == prioritized[: len(proposed)])
