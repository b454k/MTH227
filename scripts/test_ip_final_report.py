"""Smoke test for the Interest Profiler final career report stage."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from career_rag.config import PROJECT_ROOT
from career_rag.ip_ai_impact import build_ai_impact_for_occupation
from career_rag.ip_final_report import (
    FINAL_REPORT_JSON_PATH,
    build_final_career_report,
    get_onet_occupation_details,
    load_profile_result,
)
from career_rag.occupation_aliases import build_occupation_index, resolve_career_alias


PROFILE_PATH = PROJECT_ROOT / "onet_interest_profiler" / "ip_profile_result.json"


def main() -> int:
    profile = load_profile_result(PROFILE_PATH)
    occupation_index = build_occupation_index()

    data_analyst = resolve_career_alias("Data Analyst", occupation_index)
    actuary = resolve_career_alias("Actuary", occupation_index)
    ml_engineer = resolve_career_alias("Machine Learning Engineer", occupation_index)

    assert data_analyst["onet_soc_code"], "Data Analyst did not resolve."
    assert actuary["resolved_onet_title"] == "Actuaries", "Actuary should resolve to Actuaries."
    assert ml_engineer["onet_soc_code"], "Machine Learning Engineer did not resolve."

    for resolution in (data_analyst, actuary, ml_engineer):
        details = get_onet_occupation_details(resolution["onet_soc_code"])
        assert details.get("description"), f"Missing description for {resolution}"
        assert details.get("tasks"), f"Missing tasks for {resolution}"
        ai_impact = build_ai_impact_for_occupation(
            display_title=resolution["requested_title"],
            resolved_onet_title=resolution["resolved_onet_title"],
            onet_soc_code=resolution["onet_soc_code"],
            onet_tasks=details.get("tasks"),
        )
        assert ai_impact.get("task_breakdown"), f"Missing AI impact for {resolution}"

    report = build_final_career_report(profile, top_k=10)
    top_matches = report.get("top_matches") or []
    titles = [match.get("display_title") for match in top_matches]

    assert len(top_matches) <= 10
    assert top_matches[0]["display_title"] == "Data Analyst"
    assert "Actuary" in titles
    if ml_engineer["onet_soc_code"]:
        assert "Machine Learning Engineer" in titles

    for match in top_matches:
        assert match.get("why_it_fits"), f"Missing why_it_fits for {match.get('display_title')}"
        assert (match.get("ai_impact") or {}).get("task_breakdown"), (
            f"Missing AI impact breakdown for {match.get('display_title')}"
        )

    assert report.get("sources"), "Report sources should not be empty."
    assert FINAL_REPORT_JSON_PATH.exists(), "Report JSON was not saved."

    print("Final career report smoke test passed.")
    print(f"Top matches: {', '.join(titles)}")
    print(f"Saved report: {FINAL_REPORT_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
