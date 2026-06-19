#!/usr/bin/env python3
"""Generate supplemental RAG documents from newly imported O*NET tables."""

from __future__ import annotations

import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "onet.duckdb"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "documents" / "onet_supplemental_documents.jsonl"
)

MAX_ALIAS_TITLES = 80
MAX_REPORTED_TITLES = 40
MAX_RELATED_OCCUPATIONS = 40
MAX_TASKS_PER_OCCUPATION = 35
MAX_DWAS_PER_TASK = 8
MAX_TARGETS_PER_LINKAGE_SECTION = 80

LINKAGE_FAMILIES = [
    {
        "label": "Ability",
        "noun": "ability",
        "activity_table": "abilities_to_work_activities",
        "context_table": "abilities_to_work_context",
        "source_col": "abilities_element_id",
    },
    {
        "label": "Essential Skill",
        "noun": "essential skill",
        "activity_table": "essential_skills_to_work_activities",
        "context_table": "essential_skills_to_work_context",
        "source_col": "essential_skills_element_id",
    },
    {
        "label": "Transferable Skill",
        "noun": "transferable skill",
        "activity_table": "transferable_skills_to_work_activities",
        "context_table": "transferable_skills_to_work_context",
        "source_col": "transferable_skills_element_id",
    },
    {
        "label": "Work Style",
        "noun": "work style",
        "activity_table": "work_styles_to_work_activities",
        "context_table": "work_styles_to_work_context",
        "source_col": "work_styles_element_id",
    },
]


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier safely."""
    return '"' + identifier.replace('"', '""') + '"'


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return True when a table exists."""
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    return table_name in tables


def get_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    """Return column names for a table."""
    if not table_exists(conn, table_name):
        return set()
    rows = conn.execute(f"DESCRIBE {quote_identifier(table_name)}").fetchall()
    return {str(row[0]) for row in rows}


def has_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    required_columns: Iterable[str],
) -> bool:
    """Return True when a table exists and has all required columns."""
    columns = get_columns(conn, table_name)
    return bool(columns) and all(column in columns for column in required_columns)


def text_value(value: Any) -> str:
    """Convert database values to clean strings."""
    if value is None:
        return ""
    return str(value).strip()


def dedupe_strings(values: Iterable[Any]) -> list[str]:
    """Deduplicate non-empty strings while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = text_value(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def sanitize_id(value: str) -> str:
    """Make a stable, Chroma-safe document ID component."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"


def clean_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    """Chroma-friendly metadata: strings only, no None values."""
    return {key: text_value(value) for key, value in metadata.items()}


def make_document(doc_id: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Build one JSONL document with normalized metadata."""
    return {
        "id": doc_id,
        "text": text.strip(),
        "metadata": clean_metadata(metadata),
    }


def append_limited_list(
    lines: list[str],
    values: list[str],
    limit: int,
    empty_message: str = "No data available.",
) -> None:
    """Append bullet lines with a short note when values are truncated."""
    if not values:
        lines.append(empty_message)
        return

    for value in values[:limit]:
        lines.append(f"- {value}")

    remaining = len(values) - limit
    if remaining > 0:
        lines.append(f"- ... {remaining:,} more not shown")


def get_occupations(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Return O*NET occupation codes and titles."""
    if not has_columns(conn, "occupation_data", ["onetsoc_code", "title"]):
        print("Missing occupation_data(onetsoc_code, title); occupation docs skipped.")
        return []

    rows = conn.execute(
        """
        SELECT onetsoc_code, title
        FROM occupation_data
        ORDER BY onetsoc_code
        """
    ).fetchall()
    return [(text_value(code), text_value(title)) for code, title in rows]


def element_name(conn: duckdb.DuckDBPyConnection, element_id: str) -> str:
    """Look up a content model element name, falling back to the ID."""
    if not has_columns(conn, "content_model_reference", ["element_id", "element_name"]):
        return element_id

    row = conn.execute(
        """
        SELECT element_name
        FROM content_model_reference
        WHERE element_id = ?
        LIMIT 1
        """,
        [element_id],
    ).fetchone()
    return text_value(row[0]) if row and row[0] else element_id


def generate_alias_documents(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Create one title-alias document per occupation where aliases exist."""
    documents: list[dict[str, Any]] = []
    occupations = get_occupations(conn)

    has_job_titles = has_columns(conn, "job_titles", ["onetsoc_code", "job_title"])
    has_reported_titles = has_columns(
        conn,
        "sample_of_reported_titles",
        ["onetsoc_code", "reported_job_title"],
    )

    if not has_job_titles and not has_reported_titles:
        print("Alias tables are missing; occupation_aliases documents skipped.")
        return documents

    for onet_code, title in occupations:
        alias_values: list[str] = []
        reported_values: list[str] = []

        if has_job_titles:
            short_title_expr = (
                "short_title"
                if "short_title" in get_columns(conn, "job_titles")
                else "NULL AS short_title"
            )
            rows = conn.execute(
                f"""
                SELECT job_title, {short_title_expr}
                FROM job_titles
                WHERE onetsoc_code = ?
                ORDER BY job_title
                """,
                [onet_code],
            ).fetchall()
            for job_title, short_title in rows:
                alias_values.extend([job_title, short_title])

        if has_reported_titles:
            rows = conn.execute(
                """
                SELECT reported_job_title
                FROM sample_of_reported_titles
                WHERE onetsoc_code = ?
                ORDER BY reported_job_title
                """,
                [onet_code],
            ).fetchall()
            reported_values = [row[0] for row in rows]

        aliases = dedupe_strings(alias_values)
        reported_titles = dedupe_strings(reported_values)
        if not aliases and not reported_titles:
            continue

        lines = [
            f"Occupation Title Aliases: {title}",
            f"O*NET-SOC Code: {onet_code}",
            "",
            "Official title:",
            title,
            "",
            "Alternative job titles:",
        ]
        append_limited_list(lines, aliases, MAX_ALIAS_TITLES)
        lines.extend(["", "Reported titles:"])
        append_limited_list(lines, reported_titles, MAX_REPORTED_TITLES)

        documents.append(
            make_document(
                doc_id=f"supp-{sanitize_id(onet_code)}-occupation-aliases",
                text="\n".join(lines),
                metadata={
                    "doc_type": "occupation_aliases",
                    "section": "Occupation Titles",
                    "occupation_title": title,
                    "onet_soc_code": onet_code,
                    "source_table": "job_titles/sample_of_reported_titles",
                },
            )
        )

    print(f"Generated {len(documents):,} occupation alias documents")
    return documents


def generate_related_occupation_documents(
    conn: duckdb.DuckDBPyConnection,
) -> list[dict[str, Any]]:
    """Create related-occupation documents."""
    documents: list[dict[str, Any]] = []
    if not has_columns(
        conn,
        "related_occupations",
        ["onetsoc_code", "related_onetsoc_code"],
    ):
        print("related_occupations table is missing; related docs skipped.")
        return documents

    occupations = get_occupations(conn)
    for onet_code, title in occupations:
        rows = conn.execute(
            """
            SELECT
                related.related_onetsoc_code,
                COALESCE(related_title.title, related.related_onetsoc_code) AS related_title,
                related.relatedness_tier,
                related.related_index
            FROM related_occupations AS related
            LEFT JOIN occupation_data AS related_title
                ON related.related_onetsoc_code = related_title.onetsoc_code
            WHERE related.onetsoc_code = ?
            ORDER BY related.related_index, related.related_onetsoc_code
            """,
            [onet_code],
        ).fetchall()

        if not rows:
            continue

        related_lines = []
        for related_code, related_title, tier, index in rows[:MAX_RELATED_OCCUPATIONS]:
            detail_parts = [text_value(related_title), f"({text_value(related_code)})"]
            if tier:
                detail_parts.append(f"- {text_value(tier)}")
            if index is not None:
                detail_parts.append(f"rank {text_value(index)}")
            related_lines.append(" ".join(detail_parts))

        lines = [
            f"Related Occupations: {title}",
            f"O*NET-SOC Code: {onet_code}",
            "",
            "Related careers:",
        ]
        append_limited_list(lines, related_lines, MAX_RELATED_OCCUPATIONS)

        documents.append(
            make_document(
                doc_id=f"supp-{sanitize_id(onet_code)}-related-occupations",
                text="\n".join(lines),
                metadata={
                    "doc_type": "related_occupations",
                    "section": "Related Occupations",
                    "occupation_title": title,
                    "onet_soc_code": onet_code,
                    "source_table": "related_occupations",
                },
            )
        )

    print(f"Generated {len(documents):,} related occupation documents")
    return documents


def generate_task_dwa_documents(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Create one compact Task-to-DWA document per occupation."""
    documents: list[dict[str, Any]] = []
    if not has_columns(
        conn,
        "tasks_to_dwas",
        ["onetsoc_code", "task_id", "dwa_element_id"],
    ):
        print("tasks_to_dwas table is missing; task-DWA documents skipped.")
        return documents

    gwas_join = ""
    gwas_name_expr = "NULL"
    if has_columns(conn, "gwas_to_iwas_to_dwas", ["dwa_element_id", "dwa_element_name"]):
        gwas_join = """
            LEFT JOIN gwas_to_iwas_to_dwas AS hierarchy
                ON task_map.dwa_element_id = hierarchy.dwa_element_id
        """
        gwas_name_expr = "hierarchy.dwa_element_name"

    rows = conn.execute(
        f"""
        SELECT
            task_map.onetsoc_code,
            COALESCE(occupation.title, task_map.onetsoc_code) AS occupation_title,
            task_map.task_id,
            COALESCE(task_statement.task, 'Task ID ' || CAST(task_map.task_id AS VARCHAR)) AS task_text,
            task_map.dwa_element_id,
            COALESCE(content.element_name, {gwas_name_expr}, task_map.dwa_element_id) AS dwa_name
        FROM tasks_to_dwas AS task_map
        LEFT JOIN occupation_data AS occupation
            ON task_map.onetsoc_code = occupation.onetsoc_code
        LEFT JOIN task_statements AS task_statement
            ON task_map.task_id = task_statement.task_id
            AND task_map.onetsoc_code = task_statement.onetsoc_code
        LEFT JOIN content_model_reference AS content
            ON task_map.dwa_element_id = content.element_id
        {gwas_join}
        ORDER BY task_map.onetsoc_code, task_map.task_id, dwa_name
        """
    ).fetchall()

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for (
        onet_code,
        occupation_title,
        task_id,
        task_text,
        dwa_element_id,
        dwa_name,
    ) in rows:
        code = text_value(onet_code)
        occupation = grouped.setdefault(
            code,
            {
                "title": text_value(occupation_title),
                "tasks": OrderedDict(),
            },
        )
        task_key = text_value(task_id)
        task = occupation["tasks"].setdefault(
            task_key,
            {"task_text": text_value(task_text), "dwas": []},
        )
        task["dwas"].append(f"{text_value(dwa_name)} ({text_value(dwa_element_id)})")

    for onet_code, occupation in grouped.items():
        tasks: OrderedDict[str, dict[str, Any]] = occupation["tasks"]
        lines = [
            f"Task and Detailed Work Activity Mapping: {occupation['title']}",
            f"O*NET-SOC Code: {onet_code}",
            "",
        ]

        for index, task in enumerate(tasks.values()):
            if index >= MAX_TASKS_PER_OCCUPATION:
                remaining = len(tasks) - MAX_TASKS_PER_OCCUPATION
                lines.append(f"... {remaining:,} more task mappings not shown")
                break

            lines.extend(["Task:", f"- {task['task_text']}"])
            lines.append("Mapped Detailed Work Activities:")
            append_limited_list(
                lines,
                dedupe_strings(task["dwas"]),
                MAX_DWAS_PER_TASK,
            )
            lines.append("")

        documents.append(
            make_document(
                doc_id=f"supp-{sanitize_id(onet_code)}-task-dwa-mapping",
                text="\n".join(lines),
                metadata={
                    "doc_type": "task_dwa_mapping",
                    "section": "Task-DWA Mapping",
                    "occupation_title": occupation["title"],
                    "onet_soc_code": onet_code,
                    "source_table": "tasks_to_dwas",
                },
            )
        )

    print(f"Generated {len(documents):,} task-DWA mapping documents")
    return documents


def generate_work_activity_hierarchy_documents(
    conn: duckdb.DuckDBPyConnection,
) -> list[dict[str, Any]]:
    """Create GWA-IWA-DWA hierarchy documents grouped by intermediate activity."""
    documents: list[dict[str, Any]] = []
    if not has_columns(
        conn,
        "gwas_to_iwas",
        ["gwa_element_id", "iwa_element_id", "iwa_element_name"],
    ):
        print("gwas_to_iwas table is missing; hierarchy documents skipped.")
        return documents

    has_dwas = has_columns(
        conn,
        "gwas_to_iwas_to_dwas",
        ["gwa_element_id", "iwa_element_id", "dwa_element_id", "dwa_element_name"],
    )

    dwa_join = ""
    dwa_select = "NULL AS dwa_element_id, NULL AS dwa_name"
    if has_dwas:
        dwa_join = """
            LEFT JOIN gwas_to_iwas_to_dwas AS dwas
                ON iwas.gwa_element_id = dwas.gwa_element_id
                AND iwas.iwa_element_id = dwas.iwa_element_id
            LEFT JOIN content_model_reference AS dwa_content
                ON dwas.dwa_element_id = dwa_content.element_id
        """
        dwa_select = """
            dwas.dwa_element_id,
            COALESCE(dwa_content.element_name, dwas.dwa_element_name, dwas.dwa_element_id) AS dwa_name
        """

    rows = conn.execute(
        f"""
        SELECT
            iwas.gwa_element_id,
            COALESCE(gwa_content.element_name, iwas.gwa_element_id) AS gwa_name,
            iwas.iwa_element_id,
            COALESCE(iwa_content.element_name, iwas.iwa_element_name, iwas.iwa_element_id) AS iwa_name,
            {dwa_select}
        FROM gwas_to_iwas AS iwas
        LEFT JOIN content_model_reference AS gwa_content
            ON iwas.gwa_element_id = gwa_content.element_id
        LEFT JOIN content_model_reference AS iwa_content
            ON iwas.iwa_element_id = iwa_content.element_id
        {dwa_join}
        ORDER BY iwas.gwa_element_id, iwas.iwa_element_id, dwa_name
        """
    ).fetchall()

    grouped: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for gwa_id, gwa_name, iwa_id, iwa_name, dwa_id, dwa_name in rows:
        key = (text_value(gwa_id), text_value(iwa_id))
        item = grouped.setdefault(
            key,
            {
                "gwa_id": text_value(gwa_id),
                "gwa_name": text_value(gwa_name),
                "iwa_id": text_value(iwa_id),
                "iwa_name": text_value(iwa_name),
                "dwas": [],
            },
        )
        if dwa_id or dwa_name:
            item["dwas"].append(f"{text_value(dwa_name)} ({text_value(dwa_id)})")

    for item in grouped.values():
        lines = [
            "Work Activity Hierarchy",
            "",
            "General Work Activity:",
            f"{item['gwa_name']} ({item['gwa_id']})",
            "",
            "Intermediate Work Activity:",
            f"{item['iwa_name']} ({item['iwa_id']})",
            "",
            "Detailed Work Activities:",
        ]
        append_limited_list(lines, dedupe_strings(item["dwas"]), 120)

        documents.append(
            make_document(
                doc_id=(
                    "supp-work-activity-hierarchy-"
                    f"{sanitize_id(item['gwa_id'])}-{sanitize_id(item['iwa_id'])}"
                ),
                text="\n".join(lines),
                metadata={
                    "doc_type": "work_activity_hierarchy",
                    "section": "GWA-IWA-DWA Hierarchy",
                    "occupation_title": "",
                    "onet_soc_code": "",
                    "source_table": "gwas_to_iwas/gwas_to_iwas_to_dwas",
                    "gwa_element_id": item["gwa_id"],
                    "iwa_element_id": item["iwa_id"],
                    "element_name": item["iwa_name"],
                },
            )
        )

    print(f"Generated {len(documents):,} work activity hierarchy documents")
    return documents


def fetch_target_names(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    source_col: str,
    target_col: str,
    source_id: str,
) -> list[str]:
    """Fetch linked target names from one linkage table."""
    if not has_columns(conn, table_name, [source_col, target_col]):
        return []

    rows = conn.execute(
        f"""
        SELECT DISTINCT
            COALESCE(content.element_name, links.{quote_identifier(target_col)}) AS target_name
        FROM {quote_identifier(table_name)} AS links
        LEFT JOIN content_model_reference AS content
            ON links.{quote_identifier(target_col)} = content.element_id
        WHERE links.{quote_identifier(source_col)} = ?
        ORDER BY target_name
        """,
        [source_id],
    ).fetchall()
    return dedupe_strings(row[0] for row in rows)


def fetch_source_ids(
    conn: duckdb.DuckDBPyConnection,
    table_names: list[str],
    source_col: str,
) -> list[str]:
    """Fetch source element IDs from available linkage tables."""
    source_ids: list[str] = []
    for table_name in table_names:
        if not has_columns(conn, table_name, [source_col]):
            continue
        rows = conn.execute(
            f"""
            SELECT DISTINCT {quote_identifier(source_col)}
            FROM {quote_identifier(table_name)}
            ORDER BY {quote_identifier(source_col)}
            """
        ).fetchall()
        source_ids.extend(row[0] for row in rows)
    return dedupe_strings(source_ids)


def generate_linkage_documents(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Create content model linkage documents for abilities, skills, and styles."""
    documents: list[dict[str, Any]] = []

    for family in LINKAGE_FAMILIES:
        activity_table = family["activity_table"]
        context_table = family["context_table"]
        source_col = family["source_col"]
        available_tables = [
            table
            for table in [activity_table, context_table]
            if table_exists(conn, table)
        ]

        if not available_tables:
            print(f"{family['label']} linkage tables are missing; skipped.")
            continue

        source_ids = fetch_source_ids(conn, available_tables, source_col)
        for source_id in source_ids:
            activities = fetch_target_names(
                conn,
                activity_table,
                source_col,
                "work_activities_element_id",
                source_id,
            )
            contexts = fetch_target_names(
                conn,
                context_table,
                source_col,
                "work_context_element_id",
                source_id,
            )
            if not activities and not contexts:
                continue

            source_name = element_name(conn, source_id)
            noun = family["noun"]
            lines = [
                f"O*NET Linkage: {source_name}",
                "",
                f"Element type: {family['label']}",
                f"Element ID: {source_id}",
                "",
                f"This {noun} is linked to work activities such as:",
            ]
            append_limited_list(lines, activities, MAX_TARGETS_PER_LINKAGE_SECTION)
            lines.extend(["", f"This {noun} is linked to work contexts such as:"])
            append_limited_list(lines, contexts, MAX_TARGETS_PER_LINKAGE_SECTION)

            source_table = "/".join(available_tables)
            documents.append(
                make_document(
                    doc_id=(
                        f"supp-content-model-linkage-"
                        f"{sanitize_id(source_id)}-{sanitize_id(source_table)}"
                    ),
                    text="\n".join(lines),
                    metadata={
                        "doc_type": "content_model_linkage",
                        "section": "Skills Abilities Styles Linkages",
                        "occupation_title": "",
                        "onet_soc_code": "",
                        "element_name": source_name,
                        "element_id": source_id,
                        "source_table": source_table,
                    },
                )
            )

    print(f"Generated {len(documents):,} content model linkage documents")
    return documents


def write_jsonl(path: Path, documents: list[dict[str, Any]]) -> None:
    """Write documents to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            json.dump(document, file, ensure_ascii=False)
            file.write("\n")


def print_example(documents: list[dict[str, Any]]) -> None:
    """Print one example document for quick inspection."""
    if not documents:
        return

    example = documents[0]
    print("\nExample document:")
    print(f"  ID: {example['id']}")
    print(f"  Metadata: {example['metadata']}")
    print("  Text preview:")
    print(example["text"][:900])


def main() -> None:
    """Generate supplemental documents."""
    print("=" * 80)
    print("GENERATING O*NET SUPPLEMENTAL DOCUMENTS")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Output:   {OUTPUT_PATH}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Existing DuckDB database not found: {DB_PATH}")

    documents: list[dict[str, Any]] = []
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        generators = [
            generate_alias_documents,
            generate_related_occupation_documents,
            generate_task_dwa_documents,
            generate_work_activity_hierarchy_documents,
            generate_linkage_documents,
        ]

        for generator in generators:
            try:
                documents.extend(generator(conn))
            except Exception as exc:
                print(f"WARNING: {generator.__name__} failed: {exc}")
    finally:
        conn.close()

    if not documents:
        print("No supplemental documents were generated.")
        return

    by_doc_type: dict[str, int] = defaultdict(int)
    for document in documents:
        by_doc_type[document["metadata"].get("doc_type", "unknown")] += 1

    print("\nDocument counts by type:")
    for doc_type, count in sorted(by_doc_type.items()):
        print(f"  {doc_type}: {count:,}")

    print(f"\nWriting {len(documents):,} documents...")
    write_jsonl(OUTPUT_PATH, documents)
    print_example(documents)

    print("\n" + "=" * 80)
    print("SUPPLEMENTAL DOCUMENT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Documents written: {len(documents):,}")
    print(f"Output file:       {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
