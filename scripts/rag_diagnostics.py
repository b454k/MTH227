#!/usr/bin/env python3
"""Consolidated diagnostics for existing Career RAG data and retrieval outputs.

This script only reads existing project artifacts unless a report output path is
explicitly requested. It replaces the older one-off check/inspect/test scripts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "onet.duckdb"
FULL_DOCS_PATH = PROJECT_ROOT / "data" / "documents" / "onet_occupation_documents.jsonl"
SECTION_DOCS_PATH = (
    PROJECT_ROOT / "data" / "documents" / "onet_occupation_section_documents.jsonl"
)
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_onet"

SUPPLEMENTAL_TABLES = [
    "gwas_to_iwas",
    "gwas_to_iwas_to_dwas",
    "tasks_to_dwas",
    "emerging_tasks",
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
]

RETRIEVAL_QUERIES = [
    "careers involving mathematics",
    "careers involving machine learning",
    "jobs with analytical thinking",
    "day in the life of a data scientist",
    "software used by actuaries",
]

SUPPLEMENTAL_QUERIES = [
    "BI analyst",
    "business intelligence analyst alternative careers",
    "related careers for actuaries",
    "task DWA mapping for data scientists",
    "what detailed work activities are linked to data scientist tasks",
    "what work activities are linked to mathematics skill",
]


def main() -> int:
    """Run the requested diagnostic command."""
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    add_simple_command(subparsers, "tables", "List DuckDB tables.", list_tables)

    schema_parser = add_simple_command(
        subparsers,
        "schema",
        "Print or save DuckDB schema details.",
        inspect_schema,
    )
    schema_parser.add_argument("--output", type=Path, default=None)

    supplemental_parser = add_simple_command(
        subparsers,
        "supplemental-schema",
        "Inspect supplemental O*NET tables.",
        inspect_supplemental_schema,
    )
    supplemental_parser.add_argument("--output", type=Path, default=None)

    occupation_parser = add_simple_command(
        subparsers,
        "occupation",
        "Explore major O*NET tables for one occupation.",
        explore_occupation,
    )
    occupation_parser.add_argument("--code", default="11-2021.00")
    occupation_parser.add_argument("--limit", type=int, default=20)

    codes_parser = add_simple_command(
        subparsers,
        "codes",
        "Show occupation codes with broad table coverage.",
        check_codes,
    )
    codes_parser.add_argument("--limit", type=int, default=30)

    docs_parser = add_simple_command(
        subparsers,
        "documents",
        "Summarize generated O*NET documents.",
        inspect_documents,
    )
    docs_parser.add_argument(
        "--codes",
        nargs="*",
        default=["15-2051.00", "11-2021.00", "15-1252.00", "15-2011.00"],
    )
    docs_parser.add_argument("--show-full-text", action="store_true")

    sample_parser = add_simple_command(
        subparsers,
        "sample-sections",
        "Show raw sections from one generated occupation document.",
        sample_sections,
    )
    sample_parser.add_argument("--index", type=int, default=100)

    add_simple_command(
        subparsers,
        "validate-documents",
        "Print document counts and sample previews.",
        validate_documents,
    )

    verify_parser = add_simple_command(
        subparsers,
        "verify-documents",
        "Verify generated document cleanup quality.",
        verify_documents,
    )
    verify_parser.add_argument("--indices", type=int, nargs="*", default=[100, 500, 900])

    chroma_parser = add_simple_command(
        subparsers,
        "chroma-sections",
        "Count section metadata values in Chroma.",
        inspect_chroma_sections,
    )
    chroma_parser.add_argument("--limit", type=int, default=None)
    chroma_parser.add_argument("--batch-size", type=int, default=1000)

    retrieval_parser = add_simple_command(
        subparsers,
        "retrieval",
        "Run O*NET retrieval smoke queries.",
        run_retrieval,
    )
    retrieval_parser.add_argument("--k", type=int, default=5)

    supplemental_retrieval_parser = add_simple_command(
        subparsers,
        "supplemental-retrieval",
        "Run supplemental O*NET retrieval smoke queries.",
        run_supplemental_retrieval,
    )
    supplemental_retrieval_parser.add_argument("--k", type=int, default=5)

    research_parser = add_simple_command(
        subparsers,
        "research",
        "Run AI-impact research claim retrieval.",
        run_research_retrieval,
    )
    research_parser.add_argument("query")
    research_parser.add_argument("--top-k", type=int, default=8)

    combined_parser = add_simple_command(
        subparsers,
        "combined-retrieval",
        "Run O*NET retrieval plus research claim retrieval.",
        run_combined_retrieval,
    )
    combined_parser.add_argument("query")
    combined_parser.add_argument("--onet-k", type=int, default=5)
    combined_parser.add_argument("--research-top-k", type=int, default=8)

    return parser


def add_simple_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
    func: Any,
) -> argparse.ArgumentParser:
    """Register one subcommand."""
    parser = subparsers.add_parser(name, help=help_text, description=help_text)
    parser.set_defaults(func=func)
    return parser


def connect_duckdb(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Connect to the existing O*NET DuckDB database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier safely."""
    return '"' + identifier.replace('"', '""') + '"'


def list_tables(_args: argparse.Namespace) -> int:
    """List DuckDB tables."""
    with connect_duckdb() as conn:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    print("Tables:")
    for table in tables:
        print(f"  {table}")
    return 0


def inspect_schema(args: argparse.Namespace) -> int:
    """Print or save the DuckDB schema."""
    with connect_duckdb() as conn:
        lines = build_schema_report(conn, None)
    emit_report(lines, args.output)
    return 0


def inspect_supplemental_schema(args: argparse.Namespace) -> int:
    """Print or save schema and samples for supplemental tables."""
    with connect_duckdb() as conn:
        lines = [
            "O*NET Supplemental Tables Schema Report",
            f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"Database: {DB_PATH}",
        ]
        existing_tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        for table in SUPPLEMENTAL_TABLES:
            if table in existing_tables:
                lines.extend(build_schema_report(conn, [table], sample_rows=5))
            else:
                lines.extend(["", f"TABLE: {table}", "  Missing, not inspected."])

    emit_report(lines, args.output)
    return 0


def build_schema_report(
    conn: duckdb.DuckDBPyConnection,
    tables: list[str] | None,
    sample_rows: int = 0,
) -> list[str]:
    """Build schema lines for selected tables or all tables."""
    if tables is None:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

    lines: list[str] = []
    for table in tables:
        quoted = quote_identifier(table)
        lines.extend(["", "=" * 80, f"TABLE: {table}", "=" * 80])
        count = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
        lines.append(f"Rows: {count:,}")
        lines.append("Columns:")
        for column in conn.execute(f"DESCRIBE {quoted}").fetchall():
            lines.append(f"  - {column[0]}: {column[1]}")
        if sample_rows:
            lines.append(f"First {sample_rows} rows:")
            rows = conn.execute(f"SELECT * FROM {quoted} LIMIT {sample_rows}").fetchall()
            if rows:
                for row in rows:
                    lines.append("  " + " | ".join("" if value is None else str(value) for value in row))
            else:
                lines.append("  (no rows)")
    return lines


def emit_report(lines: list[str], output: Path | None) -> None:
    """Print a report and optionally write it to disk."""
    text = "\n".join(lines).strip() + "\n"
    print(text)
    if output:
        output = output if output.is_absolute() else PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Report saved to {output}")


def explore_occupation(args: argparse.Namespace) -> int:
    """Explore major O*NET tables for one occupation."""
    code = args.code.strip()
    if not code:
        raise ValueError("--code must not be empty")

    with connect_duckdb() as conn:
        title = conn.execute(
            "SELECT title FROM occupation_data WHERE onetsoc_code = ?",
            [code],
        ).fetchone()
        print("=" * 80)
        print(f"O*NET DATABASE EXPLORATION FOR {code}")
        print("=" * 80)
        print(f"Title: {title[0] if title else 'N/A'}")

        for table, label, order_scale in [
            ("essential_skills", "ESSENTIAL SKILLS", "IM"),
            ("transferable_skills", "TRANSFERABLE SKILLS", "IM"),
            ("knowledge", "KNOWLEDGE AREAS", "IM"),
            ("abilities", "ABILITIES", "IM"),
            ("work_activities", "WORK ACTIVITIES", "IM"),
            ("work_styles", "WORK STYLES", None),
        ]:
            print_ranked_content_model(conn, table, label, code, args.limit, order_scale)

        print_work_context(conn, code, args.limit)

    return 0


def print_ranked_content_model(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    label: str,
    code: str,
    limit: int,
    scale_id: str | None,
) -> None:
    """Print ranked content-model rows for one occupation."""
    where = "WHERE source.onetsoc_code = ?"
    params: list[Any] = [code]
    if scale_id:
        where += " AND source.scale_id = ?"
        params.append(scale_id)
    params.append(limit)

    query = f"""
        SELECT source.element_id, cmr.element_name, source.data_value, source.scale_id
        FROM {quote_identifier(table)} source
        JOIN content_model_reference cmr ON source.element_id = cmr.element_id
        {where}
        ORDER BY source.data_value DESC
        LIMIT ?
    """
    rows = conn.execute(query, params).fetchall()
    print_section_rows(label, ["element_id", "element_name", "data_value", "scale_id"], rows)


def print_work_context(conn: duckdb.DuckDBPyConnection, code: str, limit: int) -> None:
    """Print ranked work-context rows for one occupation."""
    rows = conn.execute(
        """
        SELECT wc.element_id, cmr.element_name, wc.data_value, wc.scale_id,
               wc.category, wcc.category_description
        FROM work_context wc
        JOIN content_model_reference cmr ON wc.element_id = cmr.element_id
        LEFT JOIN work_context_categories wcc
          ON wc.element_id = wcc.element_id
         AND wc.scale_id = wcc.scale_id
         AND wc.category = wcc.category
        WHERE wc.onetsoc_code = ?
        ORDER BY wc.data_value DESC
        LIMIT ?
        """,
        [code, limit],
    ).fetchall()
    print_section_rows(
        "WORK CONTEXT ITEMS",
        ["element_id", "element_name", "data_value", "scale_id", "category", "description"],
        rows,
    )


def print_section_rows(label: str, headers: list[str], rows: list[tuple[Any, ...]]) -> None:
    """Print a small table-like section."""
    print("\n" + label)
    print("-" * 80)
    print(" | ".join(headers))
    for row in rows:
        print(" | ".join("" if value is None else str(value) for value in row))
    if not rows:
        print("(no rows)")


def check_codes(args: argparse.Namespace) -> int:
    """Show occupation codes and broad table coverage."""
    tables = [
        "essential_skills",
        "transferable_skills",
        "knowledge",
        "abilities",
        "work_activities",
        "work_styles",
        "work_context",
    ]
    with connect_duckdb() as conn:
        print("First occupation codes:")
        for row in conn.execute(
            "SELECT onetsoc_code, title FROM occupation_data ORDER BY onetsoc_code LIMIT 5"
        ).fetchall():
            print(f"  {row[0]} | {row[1]}")

        codes = conn.execute(
            "SELECT DISTINCT onetsoc_code FROM occupation_data LIMIT ?",
            [args.limit],
        ).fetchall()
        best_code = None
        best_count = 0
        print("\nOccupation data coverage:")
        for (code,) in codes:
            counts = {
                table: conn.execute(
                    f"SELECT COUNT(*) FROM {quote_identifier(table)} WHERE onetsoc_code = ?",
                    [code],
                ).fetchone()[0]
                for table in tables
            }
            coverage_count = sum(1 for count in counts.values() if count > 0)
            if coverage_count >= 5:
                print(f"\n{code}:")
                for table, count in counts.items():
                    print(f"  {table}: {count}")
            if coverage_count > best_count:
                best_count = coverage_count
                best_code = code
        print(f"\nBest occupation code with broad data coverage: {best_code}")
    return 0


def inspect_documents(args: argparse.Namespace) -> int:
    """Summarize generated full occupation documents."""
    docs = load_jsonl(FULL_DOCS_PATH)
    found_by_code = {
        doc.get("metadata", {}).get("onet_soc_code"): doc
        for doc in docs
        if doc.get("metadata")
    }
    total_chars = sum(len(doc.get("text") or "") for doc in docs)
    print("=" * 80)
    print("O*NET OCCUPATION DOCUMENTS")
    print("=" * 80)
    print(f"Total documents: {len(docs):,}")
    print(f"Total text chars: {total_chars:,}")
    print(f"Average text chars: {total_chars / len(docs):.0f}" if docs else "Average text chars: 0")
    for code in args.codes:
        doc = found_by_code.get(code)
        print("\n" + "-" * 80)
        if not doc:
            print(f"{code}: not found")
            continue
        metadata = doc.get("metadata") or {}
        text = doc.get("text") or ""
        print(f"{code} | {metadata.get('occupation_title')}")
        print(f"Text length: {len(text):,}")
        print(text if args.show_full_text else preview(text, 800))
    return 0


def sample_sections(args: argparse.Namespace) -> int:
    """Show raw sections from a selected full occupation document."""
    docs = load_jsonl(FULL_DOCS_PATH)
    if args.index < 0 or args.index >= len(docs):
        raise ValueError(f"--index must be between 0 and {len(docs) - 1}")
    doc = docs[args.index]
    text = doc.get("text") or ""
    metadata = doc.get("metadata") or {}
    print(f"Document index: {args.index}")
    print(f"Occupation: {metadata.get('onet_soc_code')} | {metadata.get('occupation_title')}")
    for heading in ("Career Interests:", "Software Skills:"):
        print("\n" + "=" * 80)
        print(heading.rstrip(":").upper())
        print("=" * 80)
        print(extract_section(text, heading) or "(section not found)")
    return 0


def validate_documents(_args: argparse.Namespace) -> int:
    """Print samples and counts for generated full and section documents."""
    full_docs = load_jsonl(FULL_DOCS_PATH)
    section_docs = load_jsonl(SECTION_DOCS_PATH)
    print("Sample full documents:")
    for index, doc in enumerate(full_docs[:2], 1):
        print(f"{index}. {doc.get('id')}")
        print(f"   Metadata: {doc.get('metadata')}")
        print(f"   Text length: {len(doc.get('text') or ''):,}")
        print(f"   Preview: {preview(doc.get('text'), 160)}")
    print("\nSample section documents:")
    for doc in section_docs[:5]:
        metadata = doc.get("metadata") or {}
        print(f"{doc.get('id')} ({metadata.get('section')}) - {len(doc.get('text') or ''):,} chars")
    print("\nCounts:")
    print(f"Full documents: {len(full_docs):,}")
    print(f"Section documents: {len(section_docs):,}")
    print(f"Total documents: {len(full_docs) + len(section_docs):,}")
    return 0


def verify_documents(args: argparse.Namespace) -> int:
    """Verify common generated-document cleanup properties."""
    docs = load_jsonl(FULL_DOCS_PATH)
    print("=" * 80)
    print("GENERATED DOCUMENT QUALITY CHECKS")
    print("=" * 80)
    for index in args.indices:
        if index < 0 or index >= len(docs):
            continue
        doc = docs[index]
        text = doc.get("text") or ""
        metadata = doc.get("metadata") or {}
        print("\n" + "-" * 80)
        print(f"{index}: {metadata.get('onet_soc_code')} | {metadata.get('occupation_title')}")
        verify_section_items(text, "Work Styles:", check_duplicates=True)
        verify_section_items(text, "Career Interests:", riasec=True)
        verify_section_items(text, "Software Skills:", in_demand=True)
        verify_section_items(text, "Education:", value_markers=True)
        print(f"Consecutive blank line runs: {text.count(chr(10) * 3)}")
    return 0


def verify_section_items(
    text: str,
    heading: str,
    check_duplicates: bool = False,
    riasec: bool = False,
    in_demand: bool = False,
    value_markers: bool = False,
) -> None:
    """Print simple quality checks for one generated-document section."""
    section = extract_section(text, heading)
    if not section:
        return
    lines = [line.strip() for line in section.splitlines() if line.strip().startswith("-")]
    print(f"\n{heading.rstrip(':')} ({len(lines)} items)")
    if check_duplicates:
        names = [re.sub(r":.*", "", line[2:]).strip() for line in lines]
        print("  Duplicate names:", "no" if len(names) == len(set(names)) else "yes")
    if riasec:
        riasec_names = {"Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"}
        print("  RIASEC-only:", all(any(name in line for name in riasec_names) for line in lines))
    if in_demand:
        print("  In-demand examples:")
        for line in lines[:5]:
            print(f"    {'yes' if '(In Demand)' in line else 'no '} | {line}")
    if value_markers:
        print("  Value markers present:", all("(" in line and ")" in line for line in lines))


def inspect_chroma_sections(args: argparse.Namespace) -> int:
    """Count section metadata values in the existing section Chroma collection."""
    import chromadb

    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(f"ChromaDB path not found: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(name="onet_sections")
    count = collection.count()
    target_count = min(args.limit, count) if args.limit else count
    section_counts: Counter[str] = Counter()
    missing = 0
    retrieved = 0
    for offset in range(0, target_count, args.batch_size):
        batch = collection.get(
            limit=min(args.batch_size, target_count - offset),
            offset=offset,
            include=["metadatas"],
        )
        for metadata in batch.get("metadatas") or []:
            retrieved += 1
            section = (metadata or {}).get("section")
            if section:
                section_counts[str(section)] += 1
            else:
                missing += 1
    print("=" * 80)
    print("O*NET CHROMADB SECTION METADATA")
    print("=" * 80)
    print(f"Collection count: {count:,}")
    print(f"Metadata inspected: {retrieved:,}")
    for section, section_count in section_counts.most_common():
        print(f"  {section}: {section_count:,}")
    if missing:
        print(f"Rows missing section metadata: {missing:,}")
    return 0


def run_retrieval(args: argparse.Namespace) -> int:
    """Run normal O*NET retrieval smoke queries through the reusable retriever."""
    ensure_project_on_path()
    from career_rag.retriever import OnetRetriever, format_results

    retriever = OnetRetriever()
    for query in RETRIEVAL_QUERIES:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
        print(format_results(retriever.retrieve_smart(query, k=args.k)))
    return 0


def run_supplemental_retrieval(args: argparse.Namespace) -> int:
    """Run supplemental O*NET retrieval smoke queries."""
    ensure_project_on_path()
    from career_rag.retriever import OnetRetriever

    retriever = OnetRetriever()
    if retriever.supplemental_collection is None:
        print("Supplemental collection 'onet_supplemental' was not found.")
        return 1
    for query in SUPPLEMENTAL_QUERIES:
        doc_type = choose_supplemental_doc_type(query)
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print(f"Doc type filter: {doc_type or 'None'}")
        print("=" * 80)
        if query.lower() == "bi analyst":
            results = retriever.find_occupation_alias(query, k=args.k)
        else:
            results = retriever.retrieve_supplemental(query, k=args.k, doc_type=doc_type)
        for rank, result in enumerate(results, 1):
            metadata = result.get("metadata") or {}
            print(f"{rank}. {result.get('collection')} | {metadata.get('doc_type')} | {metadata.get('occupation_title')} | {float(result.get('score') or 0):.4f}")
            print(f"   {preview(result.get('text'), 500)}")
    return 0


def run_research_retrieval(args: argparse.Namespace) -> int:
    """Run AI-impact research claim retrieval."""
    ensure_project_on_path()
    from career_rag.research_retriever import format_pages, retrieve_research_claims

    claims = retrieve_research_claims(args.query, top_k=args.top_k)
    print("=" * 80)
    print("AI IMPACT RESEARCH CLAIM RETRIEVAL")
    print("=" * 80)
    print(f"Query: {args.query}")
    for rank, claim in enumerate(claims, 1):
        print_research_claim(rank, claim, format_pages)
    return 0


def run_combined_retrieval(args: argparse.Namespace) -> int:
    """Run O*NET retrieval and research claim retrieval for one query."""
    ensure_project_on_path()
    from career_rag.research_retriever import format_pages, retrieve_research_claims
    from career_rag.retriever import OnetRetriever

    retriever = OnetRetriever()
    onet_results = retriever.retrieve_smart(args.query, k=args.onet_k)
    research_claims = retrieve_research_claims(
        args.query,
        top_k=args.research_top_k,
    )

    print("=== QUERY ===")
    print(args.query)
    print("\n=== O*NET RESULTS ===")
    for rank, result in enumerate(onet_results, 1):
        metadata = result.get("metadata") or {}
        section_or_doc_type = (
            metadata.get("section")
            or metadata.get("doc_type")
            or metadata.get("source_type")
            or "N/A"
        )
        print(f"\nRank {rank}")
        print(f"occupation title: {metadata.get('occupation_title') or 'N/A'}")
        print(f"section/doc_type: {section_or_doc_type}")
        print(f"score: {float(result.get('score') or 0.0):.4f}")
        print(f"text snippet: {preview(result.get('text'), 500)}")

    print("\n=== RESEARCH CLAIM RESULTS ===")
    for rank, claim in enumerate(research_claims, 1):
        print_research_claim(rank, claim, format_pages)
    return 0


def print_research_claim(rank: int, claim: dict[str, Any], format_pages_func: Any) -> None:
    """Print one research claim in a compact diagnostic format."""
    print(f"\nRank {rank}")
    print(f"score: {float(claim.get('score') or 0.0):.4f}")
    print(f"claim_id: {claim.get('claim_id') or 'N/A'}")
    print(f"title: {claim.get('title') or 'N/A'}")
    print(f"year: {claim.get('year') or 'N/A'}")
    print(f"pages: {format_pages_func(claim) or 'N/A'}")
    print(f"impact_type_clean: {claim.get('impact_type_clean') or 'N/A'}")
    print(f"impact_direction_clean: {claim.get('impact_direction_clean') or 'N/A'}")
    print(f"ai_relevance: {claim.get('ai_relevance') or 'N/A'}")
    print(f"generator_use_scope: {claim.get('generator_use_scope') or 'N/A'}")
    print(f"claim_text: {claim.get('claim_text') or 'N/A'}")
    print(f"evidence_quote: {claim.get('evidence_quote') or 'N/A'}")


def choose_supplemental_doc_type(query: str) -> str | None:
    """Choose a focused supplemental doc_type for a validation query."""
    query_lower = query.lower()
    if query_lower == "bi analyst":
        return "occupation_aliases"
    if "alternative careers" in query_lower or "related careers" in query_lower:
        return "related_occupations"
    if "task" in query_lower or "dwa" in query_lower or "detailed work" in query_lower:
        return "task_dwa_mapping"
    if "mathematics" in query_lower or "linked to" in query_lower:
        return "content_model_linkage"
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory for diagnostics."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def extract_section(text: str, heading: str) -> str:
    """Extract one double-newline-delimited section from generated text."""
    start = text.find(heading)
    if start == -1:
        return ""
    end = text.find("\n\n", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def preview(value: Any, limit: int = 500) -> str:
    """Return a one-line preview."""
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0].rstrip() + "..."
    return text


def ensure_project_on_path() -> None:
    """Allow imports from the project root when run as a script."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
