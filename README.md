# AI Aware Career  Guide

Install dependencies, restore/check local RAG artifacts, and run the Streamlit app from the project root:

```powershell
python -m pip install -r requirements.txt
python scripts\archives\restore_data_archives.py
python scripts\check_artifacts.py
python -m streamlit run interest_profiler_app.py
```

If artifacts are missing and you have raw/local data available, rebuild them:

```powershell
python scripts\build_all_artifacts.py
```

## Project Overview

AI Aware Career Guide is a local career-guidance application built for an Applied NLP workflow. It combines the O*NET Interest Profiler, O*NET occupation data, vector retrieval, and AI labor-market evidence to help a user move from interests to grounded career recommendations.

The final app is `interest_profiler_app.py`. It asks the O*NET Interest Profiler short-form questions, computes RIASEC scores, lets the user choose current and future Job Zones, optionally asks follow-up questions, ranks matching careers, and generates a cited final career report. Follow-up ranking and report generation require local O*NET DuckDB, JSONL, Chroma, and AI-impact evidence artifacts; if they are missing, the app stops and shows rebuild instructions.

The project can run without an OpenAI key for deterministic scoring and template fallback behavior. LLM-powered follow-up interpretation, claim extraction, and richer report text require `OPENAI_API_KEY` in `.env`.

## Quick Start

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\archives\restore_data_archives.py
python scripts\check_artifacts.py
python -m streamlit run interest_profiler_app.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python scripts/archives/restore_data_archives.py
python -m streamlit run interest_profiler_app.py
```

Keep `.env` local. Do not commit API keys or machine-specific paths.

## Fresh Clone Setup

Windows PowerShell:

```powershell
git clone <repo-url>
cd career-rag
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python scripts\archives\restore_data_archives.py
python scripts\check_artifacts.py
python -m streamlit run interest_profiler_app.py
```

Fill `HF_TOKEN` in `.env` before running retrieval smoke tests or rebuilding Chroma collections. Fill `OPENAI_API_KEY` only if you want LLM-powered follow-up interpretation or research-claim extraction.

If the split archive bundle is not available, place the raw O*NET SQL dump under `onet_sql/`, place/download research and Anthropic source files under `data/`, then run:

```powershell
python scripts\build_all_artifacts.py
python scripts\check_artifacts.py
```

For a quick path/count-only check that does not load the embedding model:

```powershell
python scripts\check_artifacts.py --skip-retrieval
```

Expected behavior after a clone is explicit: the app either works with restored artifacts, or it shows `RAG artifacts missing. Please run the setup/build command.` with the exact missing files/folders.

## Artifact Policy

Generated DuckDB files, JSONL documents, Chroma stores, downloaded research files, profile results, final reports, caches, and logs are ignored by git. The repo includes `data_archives/` split zip parts so teammates can restore the large local runtime layout without committing uncompressed generated data.

Versioned artifact bundle:

```powershell
python scripts\archives\restore_data_archives.py
```

Rebuild from source/local inputs:

```powershell
python scripts\build_all_artifacts.py
```

If the archive parts become too large for normal GitHub storage, move them to Git LFS or a downloadable release zip and keep `data_archives/manifest.json` plus restore instructions updated.

## Troubleshooting Missing RAG Artifacts

- Missing `data/duckdb/onet.duckdb`: run `python scripts\archives\restore_data_archives.py`, or rebuild with `python scripts\build_all_artifacts.py` after placing O*NET SQL files in `onet_sql/`.
- Missing `data/documents/*.jsonl`: run `python scripts\onet\generate_onet_documents.py` and `python scripts\onet\generate_onet_supplemental_documents.py`.
- Missing or empty `data/chroma_onet/`: run the three O*NET embedding scripts or `python scripts\build_all_artifacts.py`.
- Missing `data/processed/ai_impact_evidence_deduped.jsonl`: restore archives or run the structured AI-impact build steps.
- Missing or empty `chroma_research/` or `chroma_ai_impact/`: restore archives or rerun the AI-impact/research embedding scripts.
- `HF_TOKEN is missing`: add it to `.env` before embedding or running retrieval smoke tests.
- Follow-up results look identical to initial matches: open `Debug: RAG artifacts` in the app and run `python scripts\check_artifacts.py`; the app now warns when follow-up answers exist but the refined order does not change.

## What The App Does

The Streamlit flow is:

1. Load the local O*NET Interest Profiler short-form questions from `onet_interest_profiler/interest_profiler_questions.json`.
2. Score the six Holland/RIASEC interests: Realistic, Investigative, Artistic, Social, Enterprising, and Conventional.
3. Detect clear, tied, or near-tied interest profiles.
4. Ask current and future Job Zone preferences to respect education, training, and preparation constraints.
5. Match careers from the local O*NET Interest Profiler career listing JSON.
6. Optionally ask follow-up questions to refine ambiguous or personalized preferences.
7. Resolve career titles to O*NET occupations and SOC codes.
8. Retrieve local O*NET evidence and AI-impact evidence.
9. Render a final report with ranked careers, explanations, task/skill evidence, AI-impact notes, alternatives, and citations.

## How The App Works

`interest_profiler_app.py` is the user-facing Streamlit application. It keeps the current profile, follow-up state, and generated report in Streamlit session state, while also saving durable JSON outputs under `onet_interest_profiler/`.

The app works in four stages:

1. **Interest Profiler scoring**: the app loads `interest_profiler_questions.json`, records checked activities, and uses `career_rag/interest_profiler_local.py` to compute six RIASEC scores and a Holland code.
2. **Career candidate matching**: `career_rag/ip_career_matcher.py` matches the highest RIASEC interests against `ip_career_listings.json`, filtered by current and future O*NET Job Zones.
3. **Optional follow-up refinement**: `career_rag/ip_followup_agent.py` asks open-ended questions, optionally uses OpenAI to interpret answers, and `career_rag/ip_refinement_ranker.py` adjusts the ranked career list using preferences, Job Zones, O*NET occupation details, and follow-up text.
4. **Final report generation**: `career_rag/ip_final_report.py` resolves titles to O*NET-SOC codes, pulls occupation details from DuckDB, retrieves semantic O*NET and AI-impact evidence, builds citations, and saves `ip_final_career_report.json` and `ip_final_career_report.md`. `career_rag/ip_report_ui.py` renders that report in the app.

The app checks required artifacts before RAG-heavy steps. If the DuckDB database, JSONL documents, or Chroma collections are missing, it shows rebuild/restore instructions instead of silently producing weak results.

## RAG Implementation

This project uses retrieval-augmented generation in two related ways:

1. **O*NET occupation RAG**: O*NET SQL tables are imported into DuckDB, transformed into JSONL documents, embedded with SentenceTransformers, and indexed in ChromaDB. Runtime code retrieves occupation sections, full occupation profiles, and supplemental documents for the user's career question or profile.
2. **AI-impact RAG**: Anthropic Economic Index evidence, NBER-derived evidence, and selected research/methodology chunks are normalized into JSONL rows, embedded, and indexed in separate Chroma collections. Runtime code retrieves these rows for AI exposure, task penetration, methodology, and caveats.

The retrieval flow is:

1. Source data is converted to typed local artifacts: DuckDB tables for structured O*NET data, JSONL files for retrievable text documents, and Chroma persistent collections for vector search.
2. Queries are formed from either a user question, a resolved occupation, or the Interest Profiler result: RIASEC scores, Holland code, Job Zones, and follow-up answer text.
3. Queries are embedded with `BAAI/bge-small-en-v1.5`.
4. Chroma returns nearest-neighbor documents plus metadata such as occupation title, O*NET-SOC code, section, document type, source, task ID, page, metric, and score.
5. The generator/report builder reranks or filters results by occupation code, title, task match, Job Zone, follow-up term overlap, source type, and evidence confidence.
6. Retrieved evidence is passed to report templates or OpenAI prompts with strict grounding instructions and citation IDs.

### Models, Technologies, Databases, And Data Types

| Part | Model / Technology | Main data types | Used by |
| --- | --- | --- | --- |
| Interest scoring | Deterministic Python scoring | JSON questions, checked activity IDs, integer RIASEC scores, Holland code strings | `interest_profiler_app.py`, `interest_profiler_local.py` |
| Local occupation database | DuckDB | O*NET SQL-imported relational tables, O*NET-SOC codes, task/skill/knowledge rows | `ip_final_report.py`, `ip_refinement_ranker.py`, `occupation_aliases.py` |
| O*NET document retrieval | ChromaDB + `BAAI/bge-small-en-v1.5` | JSONL text chunks, Chroma documents, metadata dictionaries, 384-dimensional embeddings | `retriever.py`, `ip_semantic_report.py`, `generator.py` |
| Research and AI-impact retrieval | ChromaDB + `BAAI/bge-small-en-v1.5` | JSONL evidence rows, claim rows, research chunks, metric fields, source metadata | `ai_impact_retriever.py`, `research_retriever.py`, `ip_ai_impact.py` |
| LLM query rewriting and generation | OpenAI chat model, default `gpt-4o-mini` via `OPENAI_MODEL` | Chat messages, JSON outputs, grounded report strings | `generator.py`, `ip_followup_agent.py`, parts of `ip_final_report.py` |
| App UI | Streamlit | Session-state dictionaries, forms, tabs, tables, report JSON | `interest_profiler_app.py`, `ip_report_ui.py` |
| Build and diagnostics | Python CLI scripts | DuckDB files, JSONL, CSV, PDF/text source files, Chroma stores, zip archive parts | `scripts/` |

Important persistent artifacts:

| Artifact | Type | Role |
| --- | --- | --- |
| `data/duckdb/onet.duckdb` | DuckDB database | Structured O*NET facts used for occupation details, tasks, aliases, and report grounding. |
| `data/documents/onet_occupation_documents.jsonl` | JSONL | Full occupation documents for broad retrieval. |
| `data/documents/onet_occupation_section_documents.jsonl` | JSONL | Section-level occupation chunks for tasks, skills, education, interests, software, and work context retrieval. |
| `data/documents/onet_supplemental_documents.jsonl` | JSONL | Aliases, related occupations, task-DWA mappings, and content-model linkages. |
| `data/chroma_onet/` | Chroma persistent DB | `onet_sections`, `onet_full_occupations`, and `onet_supplemental` collections. |
| `data/processed/ai_impact_evidence_deduped.jsonl` | JSONL | Structured Anthropic/NBER AI-impact evidence rows. |
| `chroma_ai_impact/` | Chroma persistent DB | `ai_impact_evidence` collection for structured AI-impact retrieval. |
| `chroma_research/` | Chroma persistent DB | `research_ai_impact_claims` and `research_inference` collections. |
| `onet_interest_profiler/ip_profile_result.json` | JSON | Saved user profile, scores, Job Zones, matches, and follow-up refinement. |
| `onet_interest_profiler/ip_final_career_report.json` | JSON | Structured final report rendered by the Streamlit UI. |
| `onet_interest_profiler/ip_final_career_report.md` | Markdown | Human-readable final report export. |

## Essential Main-Work Files

These are the files that do the core retrieval and inference work:

| File | Why it matters |
| --- | --- |
| `interest_profiler_app.py` | Main app orchestrator: collects inputs, triggers scoring, follow-up refinement, artifact checks, report generation, and report rendering. |
| `career_rag/retriever.py` | Main O*NET vector retriever. It loads the embedding model, connects to Chroma, routes queries across section/full/supplemental collections, applies metadata filters, and returns normalized evidence chunks. |
| `career_rag/ai_impact_retriever.py` | Main structured AI-impact retriever. It searches `ai_impact_evidence` and prioritizes exact SOC, exact title, task-level evidence, source type, metric availability, and confidence. |
| `career_rag/research_retriever.py` | Retrieves legacy or supplemental AI-impact research claims from `research_ai_impact_claims`. |
| `career_rag/generator.py` | General RAG answer generator and CLI. It rewrites queries when OpenAI is available, retrieves O*NET and AI-impact evidence, formats context, calls the LLM, and falls back when needed. |
| `career_rag/ip_final_report.py` | Final Interest Profiler report builder. It combines profile data, career matches, DuckDB occupation details, semantic retrieval, AI-impact evidence, citations, and optional LLM-written narratives. |
| `career_rag/ip_semantic_report.py` | Builds the semantic comparison section of the final report by querying O*NET and AI-impact collections from profile/follow-up text. |
| `career_rag/ip_ai_impact.py` | Builds occupation-specific AI-impact sections from local evidence, especially exact O*NET task rows when available. |
| `career_rag/ip_followup_agent.py` | Generates and interprets open-ended follow-up questions, using deterministic fallback logic or OpenAI JSON output. |
| `career_rag/ip_refinement_ranker.py` | Converts follow-up answers into structured preferences and reranks candidate careers using local O*NET evidence. |

For rebuilding the RAG indexes, the most important scripts are:

| Script | Main output |
| --- | --- |
| `scripts/onet/import_onet.py` | `data/duckdb/onet.duckdb` |
| `scripts/onet/generate_onet_documents.py` | Full and section-level O*NET JSONL documents |
| `scripts/onet/generate_onet_supplemental_documents.py` | Supplemental O*NET JSONL documents |
| `scripts/onet/embed_onet_documents.py` | Chroma `onet_sections` collection |
| `scripts/onet/embed_onet_full_documents.py` | Chroma `onet_full_occupations` collection |
| `scripts/onet/embed_onet_supplemental_documents.py` | Chroma `onet_supplemental` collection |
| `scripts/ai_impact/merge_ai_impact_evidence.py` | Deduplicated structured AI-impact JSONL |
| `scripts/ai_impact/embed_ai_impact_evidence.py` | Chroma `ai_impact_evidence` and research-inference collections |
| `scripts/research/embed_ai_impact_claims.py` | Chroma `research_ai_impact_claims` collection |
| `scripts/build_all_artifacts.py` | End-to-end local artifact rebuild |
| `scripts/check_artifacts.py` | Artifact and retrieval smoke test |

## Compact Structure

```text
career-rag/
  README.md                       Project setup, architecture, scripts, and references
  requirements.txt                Python package dependencies
  .env.example                    Environment variable template
  interest_profiler_app.py        Streamlit entrypoint
  career_rag/                     Runtime package used by the app and generator
  scripts/
    onet/                         O*NET database, document, and embedding builders
    research/                     Research source and claim-processing utilities
    ai_impact/                    Structured AI-exposure evidence pipeline
    diagnostics/                  Read-only inspection CLI
    archives/                     Data archive restore/package helpers
  onet_interest_profiler/         Local PDF-derived profiler assets and app outputs
  data/                           Ignored generated data restored from archives
  data_archives/                  Split archive bundle for generated data
  onet_sql/                       O*NET SQL source dump location
  chroma_ai_impact/               Ignored Chroma store for AI-impact evidence
  chroma_research/                Ignored Chroma store for research evidence
```

## Runtime Package

| File | Purpose |
| --- | --- |
| `interest_profiler_app.py` | Streamlit UI for the local O*NET Interest Profiler and final report workflow. |
| `career_rag/__init__.py` | Package exports for reusable O*NET retrieval helpers. |
| `career_rag/config.py` | Shared project paths, environment loading, and embedding-model configuration. |
| `career_rag/artifacts.py` | Central artifact validation, Chroma count inspection, and missing-artifact messages. |
| `career_rag/interest_profiler_local.py` | Local O*NET Interest Profiler scoring, profile JSON creation, and RIASEC helper logic. |
| `career_rag/ip_career_matcher.py` | Local career matching for Interest Profiler scores and Job Zones. |
| `career_rag/ip_followup_agent.py` | Optional follow-up question planning and refinement for ambiguous or personalized profiles. |
| `career_rag/ip_refinement_ranker.py` | Follow-up-aware ranking adjustments for candidate careers. |
| `career_rag/ip_final_report.py` | Builds and saves the final career report JSON/Markdown with citations and local evidence. |
| `career_rag/ip_report_ui.py` | Streamlit rendering helpers for the final report tabs, cards, tables, and sources. |
| `career_rag/ip_ai_impact.py` | AI-impact helpers for final reports, using local structured evidence with labeled fallback rows. |
| `career_rag/occupation_aliases.py` | Resolves user-facing career titles to local O*NET occupations and SOC codes. |
| `career_rag/retriever.py` | Reusable Chroma retriever for O*NET occupation, section, and supplemental documents. |
| `career_rag/research_retriever.py` | Reusable retriever for AI-impact research claims in `chroma_research`. |
| `career_rag/ai_impact_retriever.py` | Prioritized retriever for structured AI-impact evidence and research inference chunks. |
| `career_rag/ai_exposure_utils.py` | Shared normalization, hashing, JSONL, and source-policy helpers for AI-exposure pipelines. |
| `career_rag/generator.py` | CLI/generator layer for answering career questions using retrieved O*NET and research evidence. |

## Script Groups

Run scripts from the project root.

### O*NET Build Scripts

| Script | Purpose |
| --- | --- |
| `scripts/onet/create_db.py` | Creates or opens `data/duckdb/onet.duckdb`. |
| `scripts/onet/import_onet.py` | Imports all O*NET SQL files from `onet_sql/` into DuckDB in filename order. |
| `scripts/onet/import_missing_onet_tables.py` | Additively imports supplemental O*NET tables into the existing DuckDB database. |
| `scripts/onet/generate_onet_documents.py` | Builds full and section-level O*NET occupation JSONL documents for RAG retrieval. |
| `scripts/onet/generate_onet_supplemental_documents.py` | Builds supplemental documents for aliases, related occupations, task-DWA links, and content-model links. |
| `scripts/onet/embed_onet_documents.py` | Embeds section-level O*NET documents into Chroma collection `onet_sections`. |
| `scripts/onet/embed_onet_full_documents.py` | Embeds full occupation documents into Chroma collection `onet_full_occupations`. |
| `scripts/onet/embed_onet_supplemental_documents.py` | Embeds supplemental O*NET documents into Chroma collection `onet_supplemental`. |

Typical O*NET rebuild:

```powershell
python scripts\onet\import_onet.py
python scripts\onet\import_missing_onet_tables.py
python scripts\onet\generate_onet_documents.py
python scripts\onet\generate_onet_supplemental_documents.py
python scripts\onet\embed_onet_documents.py
python scripts\onet\embed_onet_full_documents.py
python scripts\onet\embed_onet_supplemental_documents.py
```

### Research Claim Scripts

| Script | Purpose |
| --- | --- |
| `scripts/research/collect_research_sources.py` | Reads `data/research/source_urls.txt`, downloads PDFs or webpage snapshots, and writes a source inventory CSV. |
| `scripts/research/enrich_research_metadata.py` | Inspects collected PDFs/text and enriches missing title, author, and year metadata. |
| `scripts/research/extract_ai_impact_claims.py` | Uses the OpenAI API to extract AI/job-impact claims from prepared research chunks. |
| `scripts/research/postprocess_ai_impact_claims.py` | Cleans labels, validates evidence quotes, deduplicates claims, and writes final claim JSONL/CSV summaries. |
| `scripts/research/embed_ai_impact_claims.py` | Embeds clean AI-impact claims into Chroma collection `research_ai_impact_claims`. |
| `scripts/research/extract_research_chunks.py` | Extracts policy-limited inference chunks from downloaded research-source manifests for the structured AI-impact pipeline. |

### Structured AI-Impact Scripts

| Script | Purpose |
| --- | --- |
| `scripts/ai_impact/download_research_sources.py` | Downloads source URLs and writes a source-policy manifest for AI-exposure work. |
| `scripts/ai_impact/extract_nber_w31222_ai_exposure.py` | Extracts structured AI-exposure evidence from NBER Working Paper 31222. |
| `scripts/ai_impact/build_anthropic_economic_index.py` | Normalizes Anthropic Economic Index data into AI-impact evidence rows. |
| `scripts/ai_impact/merge_ai_impact_evidence.py` | Merges and deduplicates Anthropic and NBER structured evidence. |
| `scripts/ai_impact/embed_ai_impact_evidence.py` | Embeds structured AI-impact evidence and research inference chunks into Chroma. |

Typical structured AI-impact rebuild:

```powershell
python scripts\ai_impact\download_research_sources.py
python scripts\research\extract_research_chunks.py
python scripts\ai_impact\extract_nber_w31222_ai_exposure.py --use-llm
python scripts\ai_impact\build_anthropic_economic_index.py --download-if-missing
python scripts\ai_impact\merge_ai_impact_evidence.py
python scripts\ai_impact\embed_ai_impact_evidence.py
```

If LLM extraction is unavailable, run NBER extraction without `--use-llm`.

### Diagnostics And Archives

| Script | Purpose |
| --- | --- |
| `scripts/diagnostics/rag_diagnostics.py` | Read-only diagnostics for DuckDB tables, documents, Chroma retrieval, and combined retrieval outputs. |
| `scripts/check_artifacts.py` | Verifies required paths, Chroma collection counts, sample retrieval, and follow-up ranking. |
| `scripts/build_all_artifacts.py` | Runs the O*NET, AI-impact, Chroma, and smoke-test build steps in order. |
| `scripts/archives/restore_data_archives.py` | Restores ignored local runtime data from split zip parts in `data_archives/`. |
| `scripts/archives/package_data_archives.py` | Packages generated local data into GitHub-friendly split zip archives. |

Useful diagnostics:

```powershell
python scripts\diagnostics\rag_diagnostics.py tables
python scripts\diagnostics\rag_diagnostics.py documents
python scripts\diagnostics\rag_diagnostics.py combined-retrieval "How will AI affect office support workers?"
python scripts\diagnostics\rag_diagnostics.py research "AI impact on programmers"
python scripts\check_artifacts.py
```

## Data And Evidence

| Path | Purpose |
| --- | --- |
| `onet_interest_profiler/Interest_Profiler.pdf` | Source PDF for the 60 short-form activities. |
| `onet_interest_profiler/IP_Career_Listings.pdf` | Source PDF for Interest Area and Job Zone career listings. |
| `onet_interest_profiler/interest_profiler_questions.json` | Runtime JSON for the 60 profiler activities. |
| `onet_interest_profiler/ip_career_listings.json` | Runtime JSON for local career matching. |
| `data/duckdb/onet.duckdb` | DuckDB database built from O*NET SQL tables. |
| `data/documents/*.jsonl` | Generated O*NET documents used for retrieval. |
| `data/chroma_onet/` | Chroma collections for O*NET section, full occupation, and supplemental retrieval. |
| `data/research/source_urls.txt` | URL input list for research source collection. |
| `data/research/*.jsonl` and `*.csv` | Research-claim extraction, cleanup, and summary outputs. |
| `data/processed/*.jsonl` | Structured AI-impact evidence and method/inference chunks. |
| `chroma_research/` | Research Chroma collections. |
| `chroma_ai_impact/` | Structured AI-impact Chroma collections. |
| `data_archives/` | Split archive bundle for sharing ignored generated data. |

All Chroma collections use the embedding model configured in `career_rag/config.py`:

```text
BAAI/bge-small-en-v1.5
```

SentenceTransformer model loading may require `HF_TOKEN` in `.env`.

## Generator CLI

The Streamlit app is the main deliverable, but the package also includes a CLI generator:

```powershell
python -m career_rag.generator "What software is commonly used by actuaries?" --show-sources
python -m career_rag.generator "How will AI affect office support workers?" --show-sources
python -m career_rag.generator "What skills matter for actuaries?" --use-research --show-sources
```

## Important Implementation Points

- User profile answers are saved as structured JSON, not embedded into Chroma.
- Raw RIASEC scores are preserved separately from follow-up refinements.
- Job Zone constraints are part of matching, so recommendations respect current and future preparation levels.
- O*NET occupation facts come from local DuckDB tables and generated JSONL/Chroma documents.
- AI-impact rows distinguish observed exposure, task penetration, adoption context, and fallback/inferred explanations.
- Exposure is not treated as job loss; broad labor-market evidence is not converted into task-level automation scores unless the local data supports that level.
- Generated app outputs such as `ip_profile_result.json` and `ip_final_career_report.json` are ignored by git.
- Generated logs, smoke-test outputs, caches, Chroma stores, DuckDB files, downloaded PDFs, and large data files are intentionally excluded from source control.

## Applied NLP Course Summary

Problem addressed: students and career changers need career recommendations that are personalized, explainable, and grounded in trusted occupation data, while also reflecting current uncertainty around AI exposure.

Solution: this project builds a retrieval-augmented career guidance system. It scores user interests with the O*NET Interest Profiler, retrieves relevant local O*NET evidence, adds structured AI-impact evidence, and produces a cited report that explains why each career is recommended.

Core NLP steps:

1. Data collection from O*NET PDFs, O*NET SQL tables, research URLs, NBER evidence, and Anthropic Economic Index data.
2. Text cleaning, metadata enrichment, chunking, and JSONL document construction.
3. Semantic embedding with `BAAI/bge-small-en-v1.5`.
4. Vector indexing and retrieval with ChromaDB.
5. Query/profile structuring from RIASEC scores, Job Zones, and follow-up answers.
6. Retrieval-augmented generation with OpenAI models when keys are available.
7. Citation-aware final reporting with deterministic fallback behavior.

Technologies used: Python, Streamlit, DuckDB, ChromaDB, SentenceTransformers, OpenAI API, pandas, PyMuPDF, BeautifulSoup, requests, and local O*NET/AI-impact data artifacts.

In short, AI Aware Career Guide turns an interest-profile questionnaire into an evidence-grounded career report by combining classic NLP preprocessing, embeddings, vector retrieval, structured occupation databases, and optional LLM generation.

## References

Primary O*NET and Interest Profiler sources:

- O*NET Resource Center: https://www.onetcenter.org/
- O*NET OnLine: https://www.onetonline.org/
- O*NET Database downloads: https://www.onetcenter.org/database.html
- O*NET Interest Profiler resources: https://www.onetcenter.org/IP.html
- O*NET Interest Profiler web app: https://onetinterestprofiler.org/
- My Next Move interest browser: https://www.mynextmove.org/find/interests
- O*NET Interest Profiler Short Form PDF: https://www.onetcenter.org/dl_tools/ipsf/Interest_Profiler.pdf
- O*NET Interest Profiler Career Listings PDF: https://www.onetcenter.org/dl_tools/ipsf/IP_Career_Listings.pdf
- O*NET Interest Profiler Score Report PDF: https://www.onetcenter.org/dl_tools/ipsf/IP_Score_Report.pdf
- O*NET Interest Profiler Score Report Starter PDF: https://www.onetcenter.org/dl_tools/ipsf/IP_Score_Report_Starter.pdf
- O*NET Interest Profiler Short Form psychometric report: https://www.onetcenter.org/reports/IPSF_Psychometric.html

AI-impact and labor-market evidence sources:

- Anthropic Economic Index: https://www.anthropic.com/economic-index
- Anthropic Economic Index dataset on Hugging Face: https://huggingface.co/datasets/Anthropic/EconomicIndex
- Anthropic labor-market impacts of AI: https://www.anthropic.com/research/labor-market-impacts
- NBER Working Paper w32966, "The Rapid Adoption of Generative AI": https://www.nber.org/papers/w32966
- NBER Working Paper w32966 PDF: https://www.nber.org/system/files/working_papers/w32966/w32966.pdf
- NBER Working Paper w31222, "Generative AI and Firm Values": https://www.nber.org/papers/w31222
- NBER Working Paper w31222 PDF: https://www.nber.org/system/files/working_papers/w31222/w31222.pdf

Career-guidance and LLM-related background sources used during development:

- Hua et al. (2024), "The Career Interests of Large Language Models": https://arxiv.org/abs/2407.08564
- Renji et al. (2025), "Steve: LLM Powered ChatBot for Career Progression": https://arxiv.org/abs/2504.03789
- Jeon et al. (2025), "Letters from Future Self": https://arxiv.org/abs/2502.18881

The expanded research corpus used by the collection scripts is listed in `data/research/source_urls.txt`.
