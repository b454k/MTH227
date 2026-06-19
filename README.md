# Career RAG

Career RAG is a retrieval-augmented career guidance project. It combines:

- O*NET occupation evidence for tasks, skills, knowledge, software, education, work context, and related occupations.
- AI/labor-market research evidence for automation exposure, augmentation, job loss, job creation, skill change, wage effects, productivity effects, and uncertainty.
- A generator CLI that answers career questions using retrieved evidence.

## Top-Level Items

| Item | Purpose |
| --- | --- |
| `.env` | Local environment variables, especially `OPENAI_API_KEY` and optional model settings. This is ignored by git. |
| `.gitignore` | Files and folders excluded from git, including `.env`, virtual environments, and Python caches. |
| `.venv/` | Local Python virtual environment. Not part of the project logic. |
| `.vscode/` | VS Code workspace settings. It points VS Code at `.venv\Scripts\python.exe` and auto-activates the environment. |
| `requirements.txt` | Python dependencies required by the project. |
| `career_rag/` | Main Python package for research collection/extraction, retrievers, and answer generation. |
| `scripts/` | O*NET build/embed scripts plus one consolidated diagnostics CLI. |
| `data/` | Local generated data, databases, Chroma stores, downloaded research files, and JSONL artifacts. |
| `onet_sql/` | Source O*NET SQL dump files imported into DuckDB. |
| `chroma_research/` | Persistent ChromaDB collection for AI-impact research claims. |

## Main Package: `career_rag/`

| File | Purpose |
| --- | --- |
| `__init__.py` | Package exports for the reusable O*NET retriever helpers. |
| `retriever.py` | Reusable O*NET Chroma retriever. Searches section, full-occupation, and supplemental O*NET collections. |
| `research_retriever.py` | Reusable AI-impact research claim retriever. Searches `chroma_research/research_ai_impact_claims`. |
| `generator.py` | CLI and generator layer. Retrieves O*NET evidence, optionally retrieves research evidence for AI-impact questions, and calls OpenAI. |
| `collect_research_sources.py` | Downloads PDFs/web snapshots from `data/research/source_urls.txt` and writes a source inventory. |
| `enrich_research_metadata.py` | Enriches collected research source metadata by inspecting PDFs and text. |
| `extract_research_chunks.py` | Extracts and chunks text from saved research PDFs/web snapshots. |
| `extract_ai_impact_claims.py` | Uses the OpenAI API to extract AI/job-impact claims from research chunks. |
| `postprocess_ai_impact_claims.py` | Cleans labels, validates quotes, and prepares final research claim JSONL without API calls. |
| `embed_ai_impact_claims.py` | Embeds final AI-impact claims into ChromaDB using `BAAI/bge-small-en-v1.5`. |

## Scripts: `scripts/`

| File | Purpose |
| --- | --- |
| `create_db.py` | Creates or opens `data/duckdb/onet.duckdb`. Mostly a tiny bootstrap helper. |
| `import_onet.py` | Imports all SQL files from `onet_sql/` into DuckDB. |
| `import_missing_onet_tables.py` | Additively imports supplemental O*NET SQL tables not covered by the first import step. |
| `generate_onet_documents.py` | Builds full and section-level O*NET occupation JSONL documents. |
| `generate_onet_supplemental_documents.py` | Builds supplemental JSONL documents for aliases, related occupations, task-DWA links, and content-model links. |
| `embed_onet_documents.py` | Embeds section-level O*NET documents into Chroma collection `onet_sections`. |
| `embed_onet_full_documents.py` | Embeds full O*NET occupation documents into Chroma collection `onet_full_occupations`. |
| `embed_onet_supplemental_documents.py` | Embeds supplemental O*NET documents into Chroma collection `onet_supplemental`. |
| `rag_diagnostics.py` | Consolidated read-only diagnostics CLI replacing old check/inspect/test scripts. |

Useful diagnostics examples:

```powershell
.venv\Scripts\python.exe scripts\rag_diagnostics.py tables
.venv\Scripts\python.exe scripts\rag_diagnostics.py documents
.venv\Scripts\python.exe scripts\rag_diagnostics.py combined-retrieval "How will AI affect office support workers?"
.venv\Scripts\python.exe scripts\rag_diagnostics.py research "AI impact on programmers"
```

## Data: `data/`

| Item | Purpose |
| --- | --- |
| `data/duckdb/onet.duckdb` | DuckDB database built from O*NET SQL files. |
| `data/duckdb/missing_tables_import_report.txt` | Report from supplemental O*NET table import. |
| `data/documents/onet_occupation_documents.jsonl` | Full occupation documents used for broad O*NET retrieval. |
| `data/documents/onet_occupation_section_documents.jsonl` | Section-level O*NET documents used for focused retrieval. |
| `data/documents/onet_supplemental_documents.jsonl` | Supplemental O*NET documents for aliases, related occupations, task mappings, and linkages. |
| `data/chroma_onet/` | Persistent ChromaDB store for O*NET collections. |
| `data/research/source_urls.txt` | Input URL list for research collection. |
| `data/research/research_sources_enriched.csv` | Research source inventory after metadata enrichment. |
| `data/research/research_chunks.jsonl` | Text chunks extracted from research PDFs/web snapshots. |
| `data/research/research_chunk_summary.csv` | Summary of chunk extraction by source. |
| `data/research/ai_impact_claims_raw.jsonl` | Raw per-chunk OpenAI claim extraction output. |
| `data/research/ai_impact_claims_postprocessed.jsonl` | Postprocessed AI-impact claims. |
| `data/research/ai_impact_claims.jsonl` | Final clean research claims used for embedding and retrieval. |
| `data/research/ai_impact_claims_failed_quotes.jsonl` | Claims that failed quote validation. |
| `data/research/ai_impact_claims_postprocess_summary.csv` | Summary of postprocessing decisions. |
| `data/research/claim_extraction_summary.csv` | Summary of claim extraction runs. |
| `data/research/label_cleaning_warnings.csv` | Warnings produced during label cleanup. |
| `data/research/ai_exposure_data.xlsx` | Spreadsheet containing AI exposure data used for analysis/reference. |
| `data/research/pdfs/` | Downloaded research PDFs. |
| `data/research/web_snapshots/` | Saved HTML/TXT web snapshots. |

## O*NET SQL: `onet_sql/`

The numbered SQL files are the source O*NET tables. They are imported in filename order by `scripts/import_onet.py`.

- `01_...` through `30_...` contain core O*NET tables such as occupation data, skills, knowledge, abilities, tasks, work activities, work context, work styles, education, interests, and software.
- `31_...` through `45_...` contain supplemental tables such as task-to-DWA mappings, related occupations, job titles, and content-model linkages.
- `Read Me.txt` is the O*NET-provided reference note for the SQL dump.

## Chroma Stores

| Item | Purpose |
| --- | --- |
| `data/chroma_onet/` | O*NET ChromaDB persistence directory. Contains `onet_sections`, `onet_full_occupations`, and `onet_supplemental`. |
| `chroma_research/` | Research ChromaDB persistence directory. Contains `research_ai_impact_claims`. |

Both sides use the embedding model:

```text
BAAI/bge-small-en-v1.5
```

Research retrieval uses the BGE query prefix:

```text
Represent this sentence for searching relevant passages: {query}
```

## Typical Build Flow

O*NET side:

```powershell
.venv\Scripts\python.exe scripts\import_onet.py
.venv\Scripts\python.exe scripts\import_missing_onet_tables.py
.venv\Scripts\python.exe scripts\generate_onet_documents.py
.venv\Scripts\python.exe scripts\generate_onet_supplemental_documents.py
.venv\Scripts\python.exe scripts\embed_onet_documents.py
.venv\Scripts\python.exe scripts\embed_onet_full_documents.py
.venv\Scripts\python.exe scripts\embed_onet_supplemental_documents.py
```

Research side:

```powershell
.venv\Scripts\python.exe career_rag\collect_research_sources.py
.venv\Scripts\python.exe career_rag\enrich_research_metadata.py
.venv\Scripts\python.exe career_rag\extract_research_chunks.py
.venv\Scripts\python.exe career_rag\extract_ai_impact_claims.py
.venv\Scripts\python.exe career_rag\postprocess_ai_impact_claims.py
.venv\Scripts\python.exe career_rag\embed_ai_impact_claims.py
```

Normal querying:

```powershell
.venv\Scripts\python.exe career_rag\generator.py "What software is commonly used by actuaries?" --show-sources
```

AI-impact querying:

```powershell
.venv\Scripts\python.exe career_rag\generator.py "How will AI affect office support workers?" --show-sources
```

Force research retrieval:

```powershell
.venv\Scripts\python.exe career_rag\generator.py "What skills matter for actuaries?" --use-research --show-sources
```

## Important Notes

- Do not edit ChromaDB files directly.
- Do not edit DuckDB files directly.
- Do not commit `.env` or `.venv/`.
- Large local artifacts are intentionally ignored by git, including DuckDB files, ChromaDB stores, downloaded PDFs, generated JSONL/CSV/XLSX research outputs, and O*NET SQL dumps.
- To reproduce a fresh clone, place/download the O*NET SQL source files under `onet_sql/`, keep research URLs in `data/research/source_urls.txt`, then run the build flow above.
- `generator.py` is the user-facing answer path.
- `retriever.py` and `research_retriever.py` are reusable retrieval layers.
- `rag_diagnostics.py` is for inspection and smoke checks only; it should not be needed for normal answers.
