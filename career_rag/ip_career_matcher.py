"""Local career matching for O*NET Interest Profiler results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from career_rag.interest_profiler_local import (
    RIASEC_CODES,
    RIASEC_INTERESTS,
    canonical_interest,
    get_top_interests,
)


JOB_ZONE_LABELS = {
    1: "Little or No Preparation Needed",
    2: "Some Preparation Needed",
    3: "Medium Preparation Needed",
    4: "Considerable Preparation Needed",
    5: "Extensive Preparation Needed",
}


def validate_job_zone(job_zone: Any) -> int:
    """Validate an O*NET Job Zone value."""
    try:
        zone = int(job_zone)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Job Zone must be an integer 1-5: {job_zone!r}") from exc
    if zone not in JOB_ZONE_LABELS:
        raise ValueError(f"Job Zone must be an integer 1-5: {zone}.")
    return zone


def _validate_assignment_note(note: Any) -> str | None:
    if note is None:
        return None
    note_text = str(note).strip()
    allowed = {
        "assigned_based_on_second_highest_interest",
        "assigned_based_on_third_highest_interest",
    }
    if note_text not in allowed:
        raise ValueError(f"Unknown Interest Profiler assignment note: {note!r}")
    return note_text


def validate_ip_career_listings(listings: Any) -> list[dict[str, Any]]:
    """Validate PDF-derived Interest Profiler career listings."""
    if not isinstance(listings, list):
        raise ValueError("IP career listings JSON must contain a list.")

    required_keys = {
        "interest",
        "interest_code",
        "job_zone",
        "job_zone_label",
        "career_title",
    }
    validated: list[dict[str, Any]] = []

    for index, listing in enumerate(listings, start=1):
        if not isinstance(listing, dict):
            raise ValueError(f"Career listing {index} must be an object.")

        missing = required_keys - set(listing)
        if missing:
            raise ValueError(f"Career listing {index} is missing fields: {sorted(missing)}.")

        interest = canonical_interest(str(listing["interest"]))
        interest_code = str(listing["interest_code"]).strip().upper()
        if interest_code != RIASEC_CODES[interest]:
            raise ValueError(
                f"Career listing {index} has interest_code {interest_code!r}, "
                f"expected {RIASEC_CODES[interest]!r}."
            )

        zone = validate_job_zone(listing["job_zone"])
        label = str(listing["job_zone_label"]).strip()
        if label != JOB_ZONE_LABELS[zone]:
            raise ValueError(
                f"Career listing {index} has job_zone_label {label!r}, "
                f"expected {JOB_ZONE_LABELS[zone]!r}."
            )

        title = str(listing["career_title"]).strip()
        if not title:
            raise ValueError(f"Career listing {index} career_title cannot be empty.")
        if "**" in title:
            raise ValueError(f"Career listing {index} career_title must not include asterisks.")

        validated.append(
            {
                "interest": interest,
                "interest_code": interest_code,
                "job_zone": zone,
                "job_zone_label": label,
                "career_title": title,
                "assignment_note": _validate_assignment_note(listing.get("assignment_note")),
            }
        )

    return validated


def load_ip_career_listings(path: str | Path) -> list[dict[str, Any]]:
    """Load the local PDF-derived O*NET Interest Profiler career listings."""
    listings_path = Path(path)
    with listings_path.open("r", encoding="utf-8") as file:
        listings = json.load(file)
    return validate_ip_career_listings(listings)


def match_careers(
    career_listings: list[dict[str, Any]],
    interest: str,
    job_zone: int,
) -> list[dict[str, Any]]:
    """Return all career listings matching one RIASEC interest and Job Zone."""
    validated_listings = validate_ip_career_listings(career_listings)
    canonical = canonical_interest(interest)
    zone = validate_job_zone(job_zone)
    return [
        listing
        for listing in validated_listings
        if listing["interest"] == canonical and listing["job_zone"] == zone
    ]


def build_career_matches(
    scores: dict[str, Any],
    current_job_zone: int,
    future_job_zone: int,
    career_listings: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build the four requested career-match groups from raw RIASEC scores."""
    top_interests = get_top_interests(scores, top_n=3)
    current_zone = validate_job_zone(current_job_zone)
    future_zone = validate_job_zone(future_job_zone)
    validated_listings = validate_ip_career_listings(career_listings)

    return {
        "primary_current_zone": match_careers(validated_listings, top_interests[0], current_zone),
        "primary_future_zone": match_careers(validated_listings, top_interests[0], future_zone),
        "secondary_future_zone": match_careers(validated_listings, top_interests[1], future_zone),
        "tertiary_future_zone": match_careers(validated_listings, top_interests[2], future_zone),
    }


def match_nearby_careers(
    career_listings: list[dict[str, Any]],
    interest: str,
    job_zone: int,
) -> list[dict[str, Any]]:
    """Return clearly labeled adjacent Job Zone alternatives for empty exact matches."""
    zone = validate_job_zone(job_zone)
    nearby_zones = [candidate for candidate in (zone - 1, zone + 1) if candidate in JOB_ZONE_LABELS]
    alternatives: list[dict[str, Any]] = []
    for nearby_zone in nearby_zones:
        for career in match_careers(career_listings, interest, nearby_zone):
            alternatives.append({**career, "nearby_alternative_for_job_zone": zone})
    return alternatives

