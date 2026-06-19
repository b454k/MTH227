#!/usr/bin/env python3
"""Import selected supplemental O*NET SQL tables into the existing DuckDB DB.

This script is intentionally additive: it connects to the existing
``data/duckdb/onet.duckdb`` file and imports only the useful missing tables
listed below. Existing tables are skipped by default.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "onet.duckdb"
SQL_FOLDER = PROJECT_ROOT / "onet_sql"
REPORT_PATH = PROJECT_ROOT / "data" / "duckdb" / "missing_tables_import_report.txt"

TARGET_SQL_FILES = [
    "31_gwas_to_iwas.sql",
    "32_gwas_to_iwas_to_dwas.sql",
    "33_tasks_to_dwas.sql",
    "34_emerging_tasks.sql",
    "35_related_occupations.sql",
    "36_job_titles.sql",
    "37_sample_of_reported_titles.sql",
    "38_abilities_to_work_activities.sql",
    "39_abilities_to_work_context.sql",
    "40_essential_skills_to_work_activities.sql",
    "41_essential_skills_to_work_context.sql",
    "42_transferable_skills_to_work_activities.sql",
    "43_transferable_skills_to_work_context.sql",
    "44_work_styles_to_work_activities.sql",
    "45_work_styles_to_work_context.sql",
]

VALIDATION_TABLES = [
    "gwas_to_iwas",
    "gwas_to_iwas_to_dwas",
    "tasks_to_dwas",
    "related_occupations",
    "job_titles",
    "sample_of_reported_titles",
    "abilities_to_work_activities",
    "abilities_to_work_context",
    "essential_skills_to_work_activities",
    "essential_skills_to_work_context",
    "transferable_skills_to_work_activities",
    "transferable_skills_to_work_context",
    "work_styles_to_work_activities",
    "work_styles_to_work_context",
    "emerging_tasks",
]

REIMPORT = os.getenv("REIMPORT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

CONSTRAINT_PREFIXES = (
    "PRIMARY KEY",
    "FOREIGN KEY",
    "KEY ",
    "INDEX ",
    "CONSTRAINT ",
    "UNIQUE ",
)


def infer_table_name(sql_file_name: str) -> str:
    """Infer a table name from O*NET SQL file names like 31_gwas_to_iwas.sql."""
    stem = Path(sql_file_name).stem
    return re.sub(r"^\d+_", "", stem)


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier safely."""
    return '"' + identifier.replace('"', '""') + '"'


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return True when a table exists in the current DuckDB database."""
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    return table_name in tables


def list_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """List current DuckDB tables in sorted order."""
    return sorted(row[0] for row in conn.execute("SHOW TABLES").fetchall())


def row_count(conn: duckdb.DuckDBPyConnection, table_name: str) -> int:
    """Return the row count for a table."""
    result = conn.execute(
        f"SELECT COUNT(*) FROM {quote_identifier(table_name)}"
    ).fetchone()
    return int(result[0]) if result else 0


def remove_mysql_versioned_comments(sql: str) -> str:
    """Drop MySQL versioned comments such as /*! START TRANSACTION */;."""
    return re.sub(r"/\*![\s\S]*?\*/;?", "", sql)


def strip_create_table_constraints(sql: str) -> str:
    """Remove table-level constraints from a CREATE TABLE statement.

    The O*NET exports use one column or constraint per line, which lets us keep
    all column definitions while removing foreign keys, primary keys, indexes,
    and named constraints that DuckDB does not need for this RAG database.
    """
    create_match = re.search(
        r"CREATE\s+TABLE\s+([^\s(]+)\s*\((.*?)\)\s*;",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not create_match:
        return sql

    table_name = create_match.group(1)
    body = create_match.group(2)
    kept_definitions: list[str] = []

    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        definition = stripped.rstrip(",").strip()
        upper_definition = definition.upper()
        if upper_definition.startswith(CONSTRAINT_PREFIXES):
            continue

        kept_definitions.append(definition)

    cleaned_body = ",\n  ".join(kept_definitions)
    cleaned_create = f"CREATE TABLE {table_name} (\n  {cleaned_body}\n);"

    return sql[: create_match.start()] + cleaned_create + sql[create_match.end() :]


def clean_sql_for_fallback(sql: str) -> str:
    """Clean O*NET SQL enough for DuckDB while preserving columns and rows."""
    sql = remove_mysql_versioned_comments(sql)
    sql = strip_create_table_constraints(sql)
    return sql


def drop_table_if_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Drop one target table. Used only for REIMPORT or failed partial imports."""
    conn.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")


def import_sql_file(
    conn: duckdb.DuckDBPyConnection,
    sql_path: Path,
    table_name: str,
) -> tuple[int, str]:
    """Import one SQL file and return ``(row_count, import_method)``."""
    sql = sql_path.read_text(encoding="utf-8", errors="ignore")
    direct_sql = remove_mysql_versioned_comments(sql)

    try:
        conn.execute(direct_sql)
        return row_count(conn, table_name), "direct"
    except Exception as direct_error:
        print(f"  Direct import failed: {direct_error}")
        print("  Retrying without table constraints...")
        drop_table_if_exists(conn, table_name)

    fallback_sql = clean_sql_for_fallback(sql)
    conn.execute(fallback_sql)
    return row_count(conn, table_name), "fallback_without_constraints"


def write_report(
    imported: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> None:
    """Write a compact import report to disk."""
    lines = [
        "O*NET Missing Tables Import Report",
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Database: {DB_PATH}",
        f"SQL folder: {SQL_FOLDER}",
        f"REIMPORT: {REIMPORT}",
        "",
        "Imported tables:",
    ]

    if imported:
        for item in imported:
            lines.append(
                f"- {item['table_name']}: {item['row_count']:,} rows "
                f"({item['method']}, {item['file_name']})"
            )
    else:
        lines.append("- None")

    lines.extend(["", "Skipped tables:"])
    if skipped:
        for item in skipped:
            lines.append(
                f"- {item['table_name']}: {item['row_count']:,} existing rows "
                f"({item['file_name']})"
            )
    else:
        lines.append("- None")

    lines.extend(["", "Failed tables:"])
    if failed:
        for item in failed:
            lines.append(f"- {item['table_name']} ({item['file_name']}): {item['error']}")
    else:
        lines.append("- None")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_counts(conn: duckdb.DuckDBPyConnection) -> None:
    """Print counts for supplemental tables that exist."""
    print("\nValidation counts:")
    for table_name in VALIDATION_TABLES:
        if table_exists(conn, table_name):
            print(f"  {table_name}: {row_count(conn, table_name):,}")


def main() -> None:
    """Import the selected supplemental O*NET tables."""
    print("=" * 80)
    print("IMPORTING USEFUL MISSING O*NET TABLES")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"SQL folder: {SQL_FOLDER}")
    print(f"REIMPORT: {REIMPORT}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Existing DuckDB database not found: {DB_PATH}")
    if not SQL_FOLDER.exists():
        raise FileNotFoundError(f"O*NET SQL folder not found: {SQL_FOLDER}")

    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    conn = duckdb.connect(str(DB_PATH))
    try:
        existing_tables = list_tables(conn)
        print(f"\nExisting tables before import ({len(existing_tables)}):")
        for table_name in existing_tables:
            print(f"  - {table_name}")

        for file_name in TARGET_SQL_FILES:
            sql_path = SQL_FOLDER / file_name
            table_name = infer_table_name(file_name)
            print("\n" + "-" * 80)
            print(f"File:  {file_name}")
            print(f"Table: {table_name}")

            if not sql_path.exists():
                message = f"SQL file not found: {sql_path}"
                print(f"  FAILED: {message}")
                failed.append(
                    {"file_name": file_name, "table_name": table_name, "error": message}
                )
                continue

            if table_exists(conn, table_name):
                existing_count = row_count(conn, table_name)
                if not REIMPORT:
                    print(f"  Skipping existing table ({existing_count:,} rows)")
                    skipped.append(
                        {
                            "file_name": file_name,
                            "table_name": table_name,
                            "row_count": existing_count,
                        }
                    )
                    continue

                print("  REIMPORT=true, dropping this target table first")
                drop_table_if_exists(conn, table_name)

            try:
                count, method = import_sql_file(conn, sql_path, table_name)
                print(f"  Imported {count:,} rows using {method}")
                imported.append(
                    {
                        "file_name": file_name,
                        "table_name": table_name,
                        "row_count": count,
                        "method": method,
                    }
                )
            except Exception as exc:
                drop_table_if_exists(conn, table_name)
                print(f"  FAILED: {exc}")
                failed.append(
                    {
                        "file_name": file_name,
                        "table_name": table_name,
                        "error": str(exc),
                    }
                )

        validate_counts(conn)
    finally:
        conn.close()

    write_report(imported=imported, skipped=skipped, failed=failed)
    print("\n" + "=" * 80)
    print("IMPORT COMPLETE")
    print("=" * 80)
    print(f"Imported: {len(imported)}")
    print(f"Skipped:  {len(skipped)}")
    print(f"Failed:   {len(failed)}")
    print(f"Report:   {REPORT_PATH}")


if __name__ == "__main__":
    main()
