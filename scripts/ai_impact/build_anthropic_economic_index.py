#!/usr/bin/env python3
"""Build normalized Anthropic Economic Index AI-impact evidence rows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from career_rag.ai_exposure_utils import (
    clean_text,
    compact_dict_text,
    created_at,
    normalize_soc_code,
    normalize_text,
    one_line,
    parse_float,
    resolve_project_path,
    stable_doc_id,
    write_jsonl,
)


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "anthropic_economic_index"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "anthropic_ai_impact.jsonl"
DEFAULT_ERRORS = PROJECT_ROOT / "data" / "processed" / "anthropic_ai_impact_errors.jsonl"
DEFAULT_DUCKDB = PROJECT_ROOT / "data" / "duckdb" / "onet.duckdb"
HF_TREE_URL = "https://huggingface.co/api/datasets/Anthropic/EconomicIndex/tree/main?recursive=1"
HF_RESOLVE_BASE = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main"

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx"}
RELEVANT_PATH_TERMS = (
    "labor_market_impacts",
    "onet",
    "soc",
    "occupation",
    "automation",
    "augmentation",
    "penetration",
    "exposure",
    "cluster",
    "output",
)
SKIP_PATH_TERMS = (
    "plot",
    ".png",
    ".jpg",
    ".jpeg",
    ".ipynb",
    "/code/",
    "request_hierarchy_tree",
)

SOC_COLS = ("soc_code", "onet_soc_code", "occupation_code", "occ_code", "soc", "SOC", "soc_2018_code", "onetsoc_code")
OCCUPATION_COLS = ("occupation", "occupation_title", "onet_occupation", "title", "job_title", "soc_name")
TASK_COLS = ("task", "task_text", "onet_task", "task_name", "task_description", "task_statement", "description")
TASK_ID_COLS = ("task_id", "onet_task_id", "task_number", "onet_task_code")
CLUSTER_COLS = ("task_cluster", "cluster", "cluster_name", "category", "request_cluster")
FIELD_COLS = ("field_or_industry", "industry", "field", "sector", "domain")
TIME_COLS = ("time_period", "date_range", "release_date", "period", "week")

METRIC_TERMS = (
    "automation",
    "automate",
    "directive",
    "augmentation",
    "augment",
    "collaboration",
    "feedback",
    "iteration",
    "iterative",
    "learning",
    "validation",
    "penetration",
    "exposure",
    "percent",
    "percentage",
    "share",
    "rate",
    "score",
    "value",
    "pct",
    "usage",
    "count",
    "conversation",
    "aui",
)

IDENTIFIER_TERMS = (
    "soc",
    "code",
    "id",
    "title",
    "occupation",
    "task",
    "description",
    "name",
    "date",
    "url",
    "file",
    "text",
)


def load_onet_lookup(duckdb_path: Path) -> dict[str, Any]:
    """Load O*NET task/occupation lookup tables from DuckDB."""
    lookup = {
        "tasks": [],
        "tasks_by_id": {},
        "tasks_by_norm": {},
        "occupation_by_soc": {},
        "occupation_by_title": {},
    }
    if not duckdb_path.exists():
        return lookup

    try:
        import duckdb
    except ImportError:
        return lookup

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        occupation_rows = con.execute(
            "select onetsoc_code, title from occupation_data"
        ).fetchall()
        for soc, title in occupation_rows:
            soc_norm = normalize_soc_code(soc) or one_line(soc)
            title_text = one_line(title)
            lookup["occupation_by_soc"][soc_norm] = title_text
            lookup["occupation_by_title"][normalize_text(title_text)] = soc_norm

        task_rows = con.execute(
            """
            select ts.onetsoc_code, od.title, ts.task_id, ts.task, ts.task_type
            from task_statements ts
            left join occupation_data od on ts.onetsoc_code = od.onetsoc_code
            """
        ).fetchall()
        for soc, title, task_id, task, task_type in task_rows:
            soc_norm = normalize_soc_code(soc) or one_line(soc)
            row = {
                "soc_code": soc_norm,
                "occupation_title": one_line(title),
                "onet_task_id": str(int(task_id)) if task_id is not None and float(task_id).is_integer() else one_line(task_id),
                "task_text": one_line(task),
                "task_type": one_line(task_type),
                "task_norm": normalize_text(task),
            }
            lookup["tasks"].append(row)
            lookup["tasks_by_id"][(soc_norm, row["onet_task_id"])] = row
            lookup["tasks_by_norm"].setdefault(row["task_norm"], []).append(row)
    finally:
        con.close()
    return lookup


def local_data_files(input_dir: Path) -> list[Path]:
    """Return supported local dataset files."""
    if not input_dir.exists():
        return []
    files = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        lowered = path.as_posix().lower()
        if any(term in lowered for term in SKIP_PATH_TERMS):
            continue
        if "onet_task_statements" in lowered or "soc_structure" in lowered:
            continue
        files.append(path)
    return sorted(files)


def relevant_hf_file(file_info: dict[str, Any], max_bytes: int) -> bool:
    """Return True for a Hugging Face file worth downloading for this pipeline."""
    if file_info.get("type") != "file":
        return False
    path = str(file_info.get("path") or "")
    lowered = path.lower().replace("\\", "/")
    if any(term in lowered for term in SKIP_PATH_TERMS):
        return False
    if Path(path).suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    if not any(term in lowered for term in RELEVANT_PATH_TERMS):
        return False
    size = int(file_info.get("size") or 0)
    if size > max_bytes and "labor_market_impacts" not in lowered:
        return False
    return True


def download_hf_dataset(input_dir: Path, max_download_mb: float) -> list[Path]:
    """Download selected Anthropic Economic Index files from Hugging Face."""
    input_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_download_mb * 1024 * 1024)
    response = requests.get(HF_TREE_URL, timeout=30)
    response.raise_for_status()
    files = response.json()
    downloaded: list[Path] = []

    for file_info in files:
        if not relevant_hf_file(file_info, max_bytes=max_bytes):
            continue
        rel_path = str(file_info["path"])
        output_path = input_dir / rel_path
        if output_path.exists() and output_path.stat().st_size > 0:
            downloaded.append(output_path)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_RESOLVE_BASE}/{quote(rel_path)}?download=true"
        print(f"Downloading {rel_path} ({int(file_info.get('size') or 0):,} bytes)")
        file_response = requests.get(url, timeout=120)
        file_response.raise_for_status()
        output_path.write_bytes(file_response.content)
        downloaded.append(output_path)
    return downloaded


def read_dataset_file(path: Path) -> pd.DataFrame:
    """Read a supported dataset file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if suffix == ".json":
        try:
            return pd.read_json(path, dtype=False)
        except ValueError:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        return pd.DataFrame(value)
                return pd.json_normalize(data)
            return pd.DataFrame()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".xlsx":
        try:
            return pd.read_excel(path, dtype=str, keep_default_na=False)
        except ImportError as exc:
            raise RuntimeError("openpyxl is required to read XLSX files.") from exc
    return pd.DataFrame()


def release_from_path(path: Path, input_dir: Path) -> str | None:
    """Infer source release from the relative file path."""
    try:
        rel = path.relative_to(input_dir).as_posix()
    except ValueError:
        rel = path.as_posix()
    match = re.search(r"release_\d{4}_\d{2}_\d{2}", rel)
    if match:
        return match.group(0)
    if rel.startswith("labor_market_impacts/"):
        return "labor_market_impacts"
    return None


def relative_source_file(path: Path) -> str:
    """Return a stable source_file path."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def lower_column_map(columns: list[Any]) -> dict[str, Any]:
    """Map lowercase column names to original names."""
    return {str(column).lower(): column for column in columns}


def find_column(columns: list[Any], candidates: tuple[str, ...]) -> Any | None:
    """Find a column by exact lowercase candidate name."""
    lower_map = lower_column_map(columns)
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for column in columns:
        lowered = str(column).lower()
        if any(candidate.lower() == lowered.replace(" ", "_") for candidate in candidates):
            return column
    return None


def columns_containing(columns: list[Any], terms: tuple[str, ...]) -> list[Any]:
    """Find columns whose names contain any term."""
    matches = []
    for column in columns:
        lowered = str(column).lower()
        if any(term in lowered for term in terms):
            matches.append(column)
    return matches


def is_metric_column(column: Any, df: pd.DataFrame) -> bool:
    """Return True when a column is likely a usable metric."""
    lowered = str(column).lower()
    if not any(term in lowered for term in METRIC_TERMS):
        return False
    if any(term in lowered for term in IDENTIFIER_TERMS) and not any(
        term in lowered for term in ("score", "value", "pct", "percent", "share", "rate", "count", "usage", "exposure", "penetration", "automation", "augmentation")
    ):
        return False
    sample = df[column].head(200)
    numeric_count = sum(parse_float(value) is not None for value in sample)
    return numeric_count > 0 or any(term in lowered for term in ("automation", "augmentation", "exposure", "penetration"))


def likely_metric_columns(df: pd.DataFrame) -> list[Any]:
    """Choose metric columns dynamically."""
    metrics = [column for column in df.columns if is_metric_column(column, df)]
    if not metrics and "pct" in lower_column_map(list(df.columns)):
        metrics.append(lower_column_map(list(df.columns))["pct"])
    return metrics


def value_from_row(row: pd.Series, column: Any | None) -> str:
    """Read a row value from a maybe-missing column."""
    if column is None:
        return ""
    return one_line(row.get(column))


def infer_impact_type(metric_name: str, row: pd.Series | None = None, interaction_column: Any | None = None) -> str:
    """Infer the normalized impact_type label."""
    text = metric_name.lower()
    if row is not None and interaction_column is not None:
        text = f"{text} {one_line(row.get(interaction_column)).lower()}"
    if any(term in text for term in ("automation", "automate", "directive")):
        return "automation"
    if any(term in text for term in ("augmentation", "augment", "collaboration", "feedback", "iterative", "iteration", "learning", "validation")):
        return "augmentation"
    if "penetration" in text:
        return "task_penetration"
    if "job_exposure" in text or "job exposure" in text:
        return "job_exposure"
    if "exposure" in text:
        return "exposure"
    if any(term in text for term in ("usage", "count", "conversation", "aui")):
        return "observed_usage"
    return "mixed"


def infer_doc_type(impact_type: str, soc_code: str, occupation_title: str, task_text: str, task_id: str) -> str:
    """Infer the normalized doc_type."""
    if impact_type == "task_penetration":
        return "ai_task_penetration"
    if impact_type == "job_exposure":
        return "ai_job_exposure"
    if task_text or task_id:
        return "ai_task_impact"
    if soc_code or occupation_title:
        return "ai_occupation_impact"
    return "ai_task_impact" if impact_type in {"automation", "augmentation"} else "ai_occupation_impact"


def metric_unit(metric_name: str, value: Any) -> str | None:
    """Infer metric unit."""
    lowered = metric_name.lower()
    value_text = one_line(value).lower()
    if "percentile" in lowered:
        return "percentile"
    if any(term in lowered for term in ("percent", "percentage", "pct", "share", "rate")):
        return "percent"
    if "%" in value_text:
        return "percent"
    if "count" in lowered or "conversation" in lowered:
        return "count"
    return None


def find_percentile(row: pd.Series, df: pd.DataFrame) -> float | None:
    """Find a percentile value in the row when present."""
    for column in df.columns:
        if "percentile" in str(column).lower():
            return parse_float(row.get(column))
    return None


def task_id_text(value: str) -> str:
    """Normalize task IDs from source data."""
    numeric = parse_float(value)
    if numeric is not None and numeric.is_integer():
        return str(int(numeric))
    return one_line(value)


def match_onet_task(
    lookup: dict[str, Any],
    soc_code: str,
    occupation_title: str,
    task_id: str,
    task_text: str,
) -> tuple[dict[str, Any] | None, str, str | None]:
    """Match a source row to an O*NET task when possible."""
    warnings: list[str] = []
    task_id = task_id_text(task_id)
    if soc_code and task_id:
        match = lookup["tasks_by_id"].get((soc_code, task_id))
        if match:
            return match, "high", None

    normalized_task = normalize_text(task_text)
    if normalized_task:
        exact_matches = lookup["tasks_by_norm"].get(normalized_task, [])
        if exact_matches:
            if soc_code:
                for match in exact_matches:
                    if match["soc_code"] == soc_code:
                        return match, "high", None
            return exact_matches[0], "medium", "Exact task text matched a different or missing SOC."

    if normalized_task and lookup["tasks"] and (soc_code or occupation_title):
        candidates = lookup["tasks"]
        if soc_code:
            candidates = [task for task in candidates if task["soc_code"] == soc_code] or candidates
        elif occupation_title:
            title_norm = normalize_text(occupation_title)
            candidates = [task for task in candidates if normalize_text(task["occupation_title"]) == title_norm] or candidates

        best: tuple[float, dict[str, Any] | None] = (0.0, None)
        for task in candidates:
            score = SequenceMatcher(None, normalized_task, task["task_norm"]).ratio()
            if score > best[0]:
                best = (score, task)
        if best[1] and best[0] >= 0.88:
            return best[1], "medium", f"Task matched by text similarity {best[0]:.2f}."
        if best[1] and best[0] >= 0.78:
            return best[1], "low", f"Uncertain task similarity match {best[0]:.2f}."

    if task_text:
        warnings.append("No confident O*NET task match.")
    return None, "low" if task_text else "medium", "; ".join(warnings) if warnings else None


def source_url_for_row() -> str:
    """Return canonical Anthropic dataset URL."""
    return "https://huggingface.co/datasets/Anthropic/EconomicIndex"


def interpretation(impact_type: str, doc_type: str, metric_name: str) -> str:
    """Write cautious interpretation for one normalized row."""
    if impact_type == "automation":
        return "Anthropic labels this as more directive/automation-like usage; it does not mean the whole job is automated."
    if impact_type == "augmentation":
        return "Anthropic labels this as more collaborative/augmentation-like usage."
    if impact_type == "task_penetration":
        return "This is observed Claude task penetration for the task or mapped task group."
    if impact_type == "job_exposure":
        return "This is occupation/job exposure evidence and should not be presented as a specific task metric."
    if impact_type == "observed_usage":
        return "This reflects observed Claude usage in the dataset, not a forecast of job loss."
    if doc_type == "ai_task_impact":
        return "This is task-level Anthropic evidence; interpret within the task scope."
    return f"This Anthropic metric should be interpreted in its source scope: {metric_name}."


def evidence_text_for_row(
    source_file: str,
    row: pd.Series,
    metric_column: Any,
    metric_value: float | None,
    columns: dict[str, Any],
) -> str:
    """Build concise evidence text from selected row fields."""
    keys = [
        "soc_code",
        "occupation_title",
        "task_text",
        "task_cluster",
        "field_or_industry",
        "time_period",
    ]
    values = {
        "soc_code": columns.get("soc_code_value"),
        "occupation_title": columns.get("occupation_title_value"),
        "task_text": columns.get("task_text_value"),
        "task_cluster": columns.get("task_cluster_value"),
        "field_or_industry": columns.get("field_or_industry_value"),
        "time_period": columns.get("time_period_value"),
    }
    context = compact_dict_text(values, keys)
    metric_value_text = "null" if metric_value is None else str(metric_value)
    return f"Anthropic Economic Index row from {source_file}: {context}; {metric_column} = {metric_value_text}."


def normalize_file_rows(path: Path, df: pd.DataFrame, input_dir: Path, lookup: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize one source file into AI-impact evidence rows."""
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if df.empty:
        errors.append(error_row(path, "empty_file", "No rows found."))
        return rows, errors

    columns = list(df.columns)
    print(f"\nFile: {relative_source_file(path)}")
    print(f"Rows: {len(df):,}")
    print("Columns:", ", ".join(str(column) for column in columns))

    soc_col = find_column(columns, SOC_COLS)
    occupation_col = find_column(columns, OCCUPATION_COLS)
    task_col = find_column(columns, TASK_COLS)
    task_id_col = find_column(columns, TASK_ID_COLS)
    cluster_col = find_column(columns, CLUSTER_COLS)
    field_col = find_column(columns, FIELD_COLS)
    time_col = find_column(columns, TIME_COLS)
    interaction_col = find_column(columns, ("interaction_type", "interaction", "behavior", "collaboration_type"))
    metric_cols = likely_metric_columns(df)

    if not metric_cols:
        errors.append(error_row(path, "no_metric_columns", "No likely metric columns detected.", columns=columns))
        return rows, errors

    release = release_from_path(path, input_dir)
    source_file = relative_source_file(path)

    for row_index, source_row in df.iterrows():
        raw_soc = value_from_row(source_row, soc_col)
        soc_code = normalize_soc_code(raw_soc)
        occupation_title = value_from_row(source_row, occupation_col)
        if not occupation_title and soc_code:
            occupation_title = lookup["occupation_by_soc"].get(soc_code, "")
        if not soc_code and occupation_title:
            soc_code = lookup["occupation_by_title"].get(normalize_text(occupation_title), "")

        task_text = value_from_row(source_row, task_col)
        task_id = task_id_text(value_from_row(source_row, task_id_col))
        match, match_confidence, warning = match_onet_task(
            lookup,
            soc_code=soc_code,
            occupation_title=occupation_title,
            task_id=task_id,
            task_text=task_text,
        )
        if match:
            soc_code = soc_code or match["soc_code"]
            occupation_title = occupation_title or match["occupation_title"]
            task_id = task_id or match["onet_task_id"]
            task_text = task_text or match["task_text"]
        if warning:
            errors.append(error_row(path, "mapping_warning", warning, row_index=int(row_index)))

        task_cluster = value_from_row(source_row, cluster_col)
        field_or_industry = value_from_row(source_row, field_col)
        time_period = value_from_row(source_row, time_col)

        for metric_column in metric_cols:
            metric_raw = source_row.get(metric_column)
            metric_value = parse_float(metric_raw)
            if metric_value is None and not one_line(metric_raw):
                continue
            impact_type = infer_impact_type(str(metric_column), source_row, interaction_col)
            doc_type = infer_doc_type(impact_type, soc_code, occupation_title, task_text, task_id)
            confidence = confidence_for_row(metric_value, match_confidence, soc_code, task_text, doc_type)
            evidence_text = evidence_text_for_row(
                source_file,
                source_row,
                metric_column,
                metric_value,
                {
                    "soc_code_value": soc_code,
                    "occupation_title_value": occupation_title,
                    "task_text_value": task_text,
                    "task_cluster_value": task_cluster,
                    "field_or_industry_value": field_or_industry,
                    "time_period_value": time_period,
                },
            )
            rows.append(
                {
                    "doc_id": stable_doc_id(
                        "anthropic",
                        source_file,
                        row_index,
                        str(metric_column),
                        soc_code,
                        task_id,
                        task_text[:80],
                        prefix="anthropic",
                    ),
                    "doc_type": doc_type,
                    "source_id": "anthropic_economic_index",
                    "source_name": "Anthropic Economic Index",
                    "source_release": release,
                    "source_file": source_file,
                    "source_url": source_url_for_row(),
                    "soc_code": soc_code or None,
                    "occupation_title": occupation_title or None,
                    "onet_task_id": task_id or None,
                    "task_text": task_text or None,
                    "task_cluster": task_cluster or None,
                    "field_or_industry": field_or_industry or None,
                    "impact_type": impact_type,
                    "metric_name": one_line(metric_column),
                    "metric_value": metric_value,
                    "metric_unit": metric_unit(str(metric_column), metric_raw),
                    "metric_percentile": find_percentile(source_row, df),
                    "time_period": time_period or None,
                    "evidence_text": evidence_text,
                    "interpretation": interpretation(impact_type, doc_type, str(metric_column)),
                    "confidence": confidence,
                    "statistics_allowed": True,
                    "created_at": created_at(),
                }
            )

    return rows, errors


def confidence_for_row(metric_value: float | None, match_confidence: str, soc_code: str, task_text: str, doc_type: str) -> str:
    """Assign a conservative confidence label."""
    if metric_value is None:
        return "low"
    if doc_type in {"ai_task_impact", "ai_task_penetration"} and task_text and match_confidence in {"high", "medium"}:
        return match_confidence
    if soc_code:
        return "medium"
    return "low"


def error_row(path: Path, stage: str, message: str, row_index: int | None = None, columns: list[Any] | None = None) -> dict[str, Any]:
    """Build an error/warning row."""
    return {
        "source_id": "anthropic_economic_index",
        "source_file": relative_source_file(path),
        "stage": stage,
        "row_index": row_index,
        "columns": [str(column) for column in columns] if columns is not None else None,
        "error_message": message,
        "created_at": created_at(),
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove exact duplicate normalized rows."""
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            one_line(row.get("source_file")).lower(),
            one_line(row.get("soc_code")).lower(),
            one_line(row.get("occupation_title")).lower(),
            one_line(row.get("onet_task_id")).lower(),
            normalize_text(row.get("task_text")),
            one_line(row.get("impact_type")).lower(),
            one_line(row.get("metric_name")).lower(),
            one_line(row.get("metric_value")).lower(),
            normalize_text(row.get("evidence_text")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Normalize Anthropic Economic Index data.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--errors", default=str(DEFAULT_ERRORS))
    parser.add_argument("--duckdb-path", default=str(DEFAULT_DUCKDB))
    parser.add_argument("--max-download-mb", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    """Run the Anthropic ingestion."""
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_path = resolve_project_path(args.output)
    errors_path = resolve_project_path(args.errors)
    duckdb_path = resolve_project_path(args.duckdb_path)

    files = local_data_files(input_dir)
    if not files and args.download_if_missing:
        try:
            download_hf_dataset(input_dir, max_download_mb=args.max_download_mb)
        except Exception as exc:
            print(f"Download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"Place Anthropic Economic Index files under: {input_dir}", file=sys.stderr)
        files = local_data_files(input_dir)

    if not files:
        print(f"No Anthropic files found under {input_dir}.", file=sys.stderr)
        print(f"Place files under {input_dir} or rerun with --download-if-missing.", file=sys.stderr)
        write_jsonl([], output_path)
        write_jsonl([error_row(input_dir, "missing_input", "No local Anthropic files found.")], errors_path)
        return 1

    print(f"Anthropic input directory: {input_dir}")
    print(f"Files to inspect: {len(files)}")
    lookup = load_onet_lookup(duckdb_path)
    print(f"O*NET tasks loaded for mapping: {len(lookup['tasks']):,}")

    all_rows: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    for path in files:
        try:
            df = read_dataset_file(path)
            rows, errors = normalize_file_rows(path, df, input_dir, lookup)
            all_rows.extend(rows)
            all_errors.extend(errors)
        except Exception as exc:
            all_errors.append(error_row(path, "file_error", f"{type(exc).__name__}: {exc}"))

    all_rows = dedupe_rows(all_rows)
    write_jsonl(all_rows, output_path)
    write_jsonl(all_errors, errors_path)

    print("\nAnthropic Economic Index build complete")
    print(f"Files inspected: {len(files)}")
    print(f"Evidence rows written: {len(all_rows):,}")
    print(f"Errors/warnings written: {len(all_errors):,}")
    print(f"Rows with SOC: {sum(1 for row in all_rows if row.get('soc_code')):,}")
    print(f"Rows with task_text: {sum(1 for row in all_rows if row.get('task_text')):,}")
    print(f"Rows with metric_value: {sum(1 for row in all_rows if row.get('metric_value') is not None):,}")
    print(f"Output: {output_path}")
    print(f"Errors: {errors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
