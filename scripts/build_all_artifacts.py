#!/usr/bin/env python3
"""Build or rebuild the local artifacts needed by the Career RAG app."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, args: list[str], env: dict[str, str] | None = None) -> None:
    """Run one build step and stop with a clear message on failure."""
    print("\n" + "=" * 100)
    print(f"STEP: {label}")
    print("=" * 100)
    print("Command: " + " ".join(args))

    step_env = os.environ.copy()
    step_env["PYTHONPATH"] = str(PROJECT_ROOT)
    if env:
        step_env.update(env)

    completed = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        env=step_env,
        check=False,
    )
    if completed.returncode != 0:
        print(f"FAILED: {label} (exit code {completed.returncode})", file=sys.stderr)
        raise SystemExit(completed.returncode)
    print(f"SUCCESS: {label}")


def script(path: str) -> list[str]:
    return [sys.executable, path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore-archives-first",
        action="store_true",
        help="Restore prebuilt data_archives before running build steps.",
    )
    parser.add_argument(
        "--skip-onet",
        action="store_true",
        help="Skip DuckDB, O*NET document, and O*NET Chroma steps.",
    )
    parser.add_argument(
        "--skip-ai-impact",
        action="store_true",
        help="Skip structured Anthropic/NBER AI-impact evidence steps.",
    )
    parser.add_argument(
        "--skip-research-claims",
        action="store_true",
        help="Skip research claim Chroma embedding.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Do not run scripts/check_artifacts.py at the end.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.restore_archives_first:
        run_step(
            "Restore prebuilt local data archive",
            script("scripts/archives/restore_data_archives.py"),
        )

    if not args.skip_onet:
        run_step("Create/open O*NET DuckDB database", script("scripts/onet/create_db.py"))
        run_step("Import O*NET SQL tables into DuckDB", script("scripts/onet/import_onet.py"))
        run_step(
            "Import supplemental O*NET SQL tables",
            script("scripts/onet/import_missing_onet_tables.py"),
        )
        run_step("Generate O*NET occupation documents", script("scripts/onet/generate_onet_documents.py"))
        run_step(
            "Generate O*NET supplemental documents",
            script("scripts/onet/generate_onet_supplemental_documents.py"),
        )
        run_step(
            "Build O*NET section Chroma collection",
            script("scripts/onet/embed_onet_documents.py"),
            env={"REBUILD_COLLECTION": "true"},
        )
        run_step(
            "Build O*NET full occupation Chroma collection",
            script("scripts/onet/embed_onet_full_documents.py"),
        )
        run_step(
            "Build O*NET supplemental Chroma collection",
            script("scripts/onet/embed_onet_supplemental_documents.py"),
        )

    if not args.skip_ai_impact:
        run_step(
            "Generate Anthropic Economic Index evidence rows",
            script("scripts/ai_impact/build_anthropic_economic_index.py"),
        )
        run_step(
            "Extract NBER W31222 AI-exposure evidence",
            script("scripts/ai_impact/extract_nber_w31222_ai_exposure.py"),
        )
        run_step(
            "Merge structured AI-impact evidence",
            script("scripts/ai_impact/merge_ai_impact_evidence.py"),
        )
        run_step(
            "Build structured AI-impact and research-inference Chroma collections",
            script("scripts/ai_impact/embed_ai_impact_evidence.py"),
        )

    if not args.skip_research_claims:
        run_step(
            "Build research claim Chroma collection",
            script("scripts/research/embed_ai_impact_claims.py"),
        )

    if not args.skip_smoke_test:
        run_step("Run artifact smoke checks", script("scripts/check_artifacts.py"))

    print("\nAll requested artifact build steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
