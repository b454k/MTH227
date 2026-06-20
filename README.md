# Career RAG

Install dependencies and run the Streamlit app from the project root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run interest_profiler_app.py
```

If this is a fresh clone, restore the bundled local data first:

```powershell
python scripts\archives\restore_data_archives.py
```

## Project Overview

Career RAG is a local career-guidance application built for an Applied NLP workflow. It combines the O*NET Interest Profiler, O*NET occupation data, vector retrieval, and AI labor-market evidence to help a user move from interests to grounded career recommendations.

The final app is `interest_profiler_app.py`. It asks the O*NET Interest Profiler short-form questions, computes RIASEC scores, lets the user choose current and future Job Zones, optionally asks follow-up questions, ranks matching careers, and generates a cited final career report. The report uses local O*NET evidence from DuckDB and AI-impact evidence from local JSONL/Chroma artifacts when available.

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

## Compact Structure

```text
career-rag/
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
| `scripts/archives/restore_data_archives.py` | Restores ignored local runtime data from split zip parts in `data_archives/`. |
| `scripts/archives/package_data_archives.py` | Packages generated local data into GitHub-friendly split zip archives. |

Useful diagnostics:

```powershell
python scripts\diagnostics\rag_diagnostics.py tables
python scripts\diagnostics\rag_diagnostics.py documents
python scripts\diagnostics\rag_diagnostics.py combined-retrieval "How will AI affect office support workers?"
python scripts\diagnostics\rag_diagnostics.py research "AI impact on programmers"
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

In short, Career RAG turns an interest-profile questionnaire into an evidence-grounded career report by combining classic NLP preprocessing, embeddings, vector retrieval, structured occupation databases, and optional LLM generation.
