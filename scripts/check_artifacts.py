#!/usr/bin/env python3
"""Smoke-check local artifacts required by the Career RAG app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from career_rag.artifacts import (  # noqa: E402
    MissingRagArtifactsError,
    ensure_required_artifacts,
    format_missing_artifacts_error,
    inspect_rag_artifacts,
)
from career_rag.interest_profiler_local import RIASEC_INTERESTS  # noqa: E402
from career_rag.ip_refinement_ranker import build_refined_career_recommendations  # noqa: E402
from career_rag.occupation_aliases import build_occupation_index, resolve_career_alias  # noqa: E402


def print_artifact_table(inspection: dict[str, Any]) -> None:
    print("Artifact status")
    print("-" * 100)
    for status in inspection.get("statuses") or []:
        exists = "OK" if status.get("exists") else "MISSING"
        count = status.get("count")
        count_text = "" if count is None else f" count={count}"
        error = status.get("error")
        error_text = f" error={error}" if error else ""
        print(
            f"{exists:7} {status.get('label')}: "
            f"{status.get('display_path') or status.get('path')}{count_text}{error_text}"
        )
    print("-" * 100)


def sample_onet_retrieval() -> None:
    from career_rag.retriever import OnetRetriever

    retriever = OnetRetriever()
    results = retriever.retrieve_smart("tasks and skills for data scientists", k=3)
    if not results:
        raise RuntimeError("Sample O*NET retrieval returned zero results.")
    first = results[0]
    metadata = first.get("metadata") or {}
    print(
        "Sample O*NET retrieval OK: "
        f"{metadata.get('occupation_title') or 'unknown occupation'} "
        f"from {first.get('collection')}"
    )


def sample_followup_ranking() -> None:
    occupation_index = build_occupation_index()
    resolution = resolve_career_alias("Data Scientists", occupation_index)
    if not resolution.get("onet_soc_code"):
        resolution = resolve_career_alias("Actuaries", occupation_index)
    if not resolution.get("onet_soc_code"):
        raise RuntimeError("Could not resolve a sample occupation from local O*NET data.")

    title = str(resolution.get("resolved_onet_title") or "Data Scientists")
    profile_result = {
        "riasec_scores": {interest: 0 for interest in RIASEC_INTERESTS},
        "initial_top_interests": ["Investigative", "Conventional", "Realistic"],
        "final_top_interests": ["Investigative", "Conventional", "Realistic"],
        "initial_holland_code": "ICR",
        "final_holland_code": "ICR",
        "current_job_zone": 3,
        "future_job_zone": 4,
        "career_matches": {
            "primary_future_zone": [
                {
                    "career_title": title,
                    "interest": "Investigative",
                    "job_zone": 4,
                }
            ],
            "primary_current_zone": [],
            "secondary_future_zone": [],
            "tertiary_future_zone": [],
        },
    }
    followup_refinement = {
        "method": "smoke_test",
        "questions_asked": [
            {
                "question": "What kind of work sounds most appealing?",
                "answer": "I like data analysis, technology systems, and structured problem solving.",
            }
        ],
        "final_refinement": {
            "refined_top_interests": ["Investigative", "Conventional", "Realistic"],
            "key_sub_preferences": [
                "data and numbers",
                "technology systems",
                "research and problem solving",
            ],
            "concerns_noted": [],
            "future_vision_summary": "Evidence-grounded analytical work.",
            "career_matching_guidance": {
                "prioritize_interests": ["Investigative"],
                "job_zone_to_use": 4,
                "notes": "Prefer data and technology-heavy work.",
            },
        },
        "json_valid": True,
    }
    payload = build_refined_career_recommendations(profile_result, followup_refinement)
    matches = payload.get("final_ranked_matches") or []
    if not matches:
        raise RuntimeError("Follow-up ranking smoke test returned zero matches.")
    top = matches[0]
    if not top.get("onet_soc_code"):
        raise RuntimeError("Follow-up ranking did not attach O*NET occupation evidence.")
    print(
        "Follow-up ranking OK: "
        f"{top.get('career_title')} -> {top.get('onet_soc_code')} "
        f"score={top.get('score')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Only check paths and Chroma collection counts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inspection = inspect_rag_artifacts(check_chroma=True)
    print_artifact_table(inspection)

    try:
        ensure_required_artifacts(check_chroma=True)
    except MissingRagArtifactsError as exc:
        print(format_missing_artifacts_error(exc.missing_artifacts), file=sys.stderr)
        return 1

    if args.skip_retrieval:
        print("Artifact path and Chroma count checks passed.")
        return 0

    try:
        sample_onet_retrieval()
        sample_followup_ranking()
    except Exception as exc:
        print(f"Smoke test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("All artifact checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
