# Career RAG

Career RAG is a retrieval-augmented career guidance project. It combines:

- O*NET occupation evidence for tasks, skills, knowledge, software, education, work context, and related occupations.
- AI/labor-market research evidence for automation exposure, augmentation, job loss, job creation, skill change, wage effects, productivity effects, and uncertainty.
- A generator CLI that answers career questions using retrieved evidence.

## Quick Start on a New Machine

Use Python 3.11 or 3.12 if possible. The app can run without an OpenAI key, but LLM-powered follow-up/report text needs `OPENAI_API_KEY`.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python scripts/restore_data_archives.py
python -m streamlit run interest_profiler_app.py
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\restore_data_archives.py
python -m streamlit run interest_profiler_app.py
```

Fill in `.env` only with local secrets. Do not commit `.env`.

## Top-Level Items

| Item | Purpose |
| --- | --- |
| `.env` | Local environment variables, especially `OPENAI_API_KEY` and optional model settings. This is ignored by git. |
| `.env.example` | Safe template for teammate setup. Copy it to `.env` locally. |
| `.gitignore` | Files and folders excluded from git, including `.env`, virtual environments, and Python caches. |
| `.venv/` | Local Python virtual environment. Not part of the project logic. |
| `.vscode/` | VS Code workspace settings. It points VS Code at `.venv\Scripts\python.exe` and auto-activates the environment. |
| `requirements.txt` | Python dependencies required by the project. |
| `career_rag/` | Main Python package for research collection/extraction, retrievers, and answer generation. |
| `scripts/` | O*NET build/embed scripts plus one consolidated diagnostics CLI. |
| `data/` | Local generated data, databases, Chroma stores, downloaded research files, and JSONL artifacts. |
| `data_archives/` | Split zip bundle of ignored local data. Run `python scripts/restore_data_archives.py` after cloning. |
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
| `config.py` | Shared project constants, including the single SentenceTransformer embedding model name. |
| `occupation_aliases.py` | Resolves requested career titles such as Data Analyst or Machine Learning Engineer to local O*NET occupations and SOC codes. |
| `ip_ai_impact.py` | Builds final-report AI-impact breakdowns from local Anthropic evidence when available, with clearly labeled heuristic fallback rows. |
| `ip_final_report.py` | Builds and saves the final Interest Profiler career report JSON/Markdown using local O*NET DuckDB evidence and citations. |
| `ip_report_ui.py` | Streamlit renderer for the final career report tabs, detail cards, AI tables, alternatives, and sources. |

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
| `package_data_archives.py` | Creates split zip parts under `data_archives/` from ignored local runtime data. |
| `restore_data_archives.py` | Restores ignored runtime data from `data_archives/` after a fresh clone. |

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
| `chroma_research/` | Research ChromaDB persistence directory. Contains `research_ai_impact_claims` and `research_inference`. |
| `chroma_ai_impact/` | Structured AI-impact evidence ChromaDB persistence directory. Contains `ai_impact_evidence`. |

All Chroma collections use the embedding model defined in `career_rag/config.py`:

```text
BAAI/bge-small-en-v1.5
```

SentenceTransformer model loading requires `HF_TOKEN` in the project `.env`:

```text
HF_TOKEN=your_huggingface_token
```

Research retrieval uses the BGE query prefix:

```text
Represent this sentence for searching relevant passages: {query}
```

## Sharing Large Local Data

Large runtime folders are ignored uncompressed:

- `data/`
- `chroma_ai_impact/`
- `chroma_research/`
- `onet_sql/*.sql`

They are shared through `data_archives/` instead. The archive bundle preserves the same project-relative paths, so restoring it does not require code changes:

```bash
python scripts/restore_data_archives.py
```

To refresh the bundle after rebuilding data locally:

```bash
python scripts/package_data_archives.py
```

The default archive part size is below GitHub's hard single-file limit. Keep using git from the command line for upload/clone; GitHub's browser uploader may reject medium-size files even when git accepts them.

## O*NET Interest Profiler Local PDF Version

This project includes a local Streamlit implementation of the O*NET Interest Profiler Short Form. It uses PDF-derived JSON files from `onet_interest_profiler/` and does not call the O*NET Web Services API. This is the local PDF-based implementation until O*NET Web Services API approval is available.

Run the UI:

```powershell
.venv\Scripts\python.exe -m streamlit run interest_profiler_app.py
```

Local files:

| File | Purpose |
| --- | --- |
| `onet_interest_profiler/Interest_Profiler.pdf` | Source PDF for the 60 short-form activities. |
| `onet_interest_profiler/IP_Career_Listings.pdf` | Source PDF for Interest Area + Job Zone career listings. |
| `onet_interest_profiler/interest_profiler_questions.json` | Runtime JSON for the 60 activities; the app does not parse the PDF at runtime. |
| `onet_interest_profiler/ip_career_listings.json` | Runtime JSON for local career matching; career titles only come from this file. |
| `onet_interest_profiler/source_urls.txt` | Research/source links for the local profiler and follow-up design notes. |
| `onet_interest_profiler/ip_profile_result.json` | Saved user profile result after submitting the Streamlit form. |
| `onet_interest_profiler/ip_final_career_report.json` | Saved final career report after retrieval and report generation. |
| `onet_interest_profiler/ip_final_career_report.md` | Optional readable markdown copy of the final report. |

Scoring and matching:

- The 60 activities are split evenly across Realistic, Investigative, Artistic, Social, Enterprising, and Conventional.
- Each checked activity counts as 1 point; unchecked activities count as 0.
- Each RIASEC score ranges from 0 to 10.
- Exact ties and near ties within 1 point are marked as ambiguous instead of being treated as solved.
- The initial Holland code is built from the top three deterministic interests, such as `ICS`.
- Current Job Zone is used for immediate primary-interest matches; Future Job Zone is used for primary, secondary, and tertiary aspirational matches.
- Job Zones 1-5 use the O*NET labels from little/no preparation through extensive preparation.

LLM follow-up:

- After the initial result, the app offers follow-up questions for tied/close profiles and optional personalization for clear profiles.
- The follow-up asks tie-breaking, dynamic deepening, and future-vision questions one at a time.
- If `OPENAI_API_KEY` is available in `.env`, the final refinement JSON can be produced by the LLM using `OPENAI_MODEL` or `gpt-4o-mini`.
- If no LLM key is available, the app still works in `template_fallback` mode and records the answers without trying to infer deeply.
- Raw O*NET scores and initial interests are preserved in `raw_riasec_scores` and `initial_top_interests`; LLM/template refinement is stored separately in `followup_refinement`.
- The saved profile JSON is intentionally kept separate from Chroma. `prepare_profile_for_rag(profile_result)` prepares structured profile context for later generator integration without calling the retriever.

Smoke test:

```powershell
.venv\Scripts\python.exe scripts\test_interest_profiler_local.py
```

## Final Career Report After Interest Profiler

After the Streamlit Interest Profiler and optional follow-up questions are complete, click **Generate Final Career Report** in the app. The report builder loads `onet_interest_profiler/ip_profile_result.json` or the in-memory profile result, keeps raw RIASEC scores unchanged, and uses the follow-up refinement fields when present. The profile is used only as structured input; user profile data is not embedded into Chroma.

The final report is built by `career_rag/ip_final_report.py`:

- `prepare_profile_for_rag(profile_result)` supplies the RIASEC scores, Holland code, Job Zones, career titles, and prompt summary.
- Follow-up fields such as `refined_top_interests`, `refined_holland_code`, `key_sub_preferences`, `future_vision_summary`, `concerns_noted`, and `career_matching_guidance` personalize the ranking.
- For the current sample profile, the deterministic ranking prioritizes Data Analyst, Actuary, and Machine Learning Engineer before related analytics, math, technology, and business-facing roles.

Occupation retrieval and alias mapping:

- `career_rag/occupation_aliases.py` builds a local occupation index from `occupation_data`, `job_titles`, and `sample_of_reported_titles` in `data/duckdb/onet.duckdb`.
- `resolve_career_alias(title, occupation_index)` returns the requested title, resolved O*NET title, O*NET-SOC code, resolution method, and confidence.
- Alias display titles are kept readable. For example, a report can display "Machine Learning Engineer" while noting that O*NET evidence was retrieved from the closest local occupation title.
- O*NET details come from DuckDB tables, not just the Interest Profiler career listing PDF: description, tasks, skills, knowledge, abilities, education, Job Zone, work context, software, and related occupations.

AI-impact evidence:

- `career_rag/ip_ai_impact.py` first looks for local structured Anthropic evidence in `data/processed/ai_impact_evidence_deduped.jsonl` and `data/processed/anthropic_ai_impact.jsonl`.
- Retrieved Anthropic metrics are labeled by their real metric type, such as observed exposure or task penetration.
- When no local task-level metric matches, the report uses `evidence_status = "heuristic_pending_source_retrieval"` and labels scores as heuristic.
- NBER "The Rapid Adoption of Generative AI" is used only as broad labor-market/adoption context, not as task-level automation scoring.

Citations:

- `CitationManager` assigns numbered citation IDs such as `[1]`, `[2]`, and stores source metadata.
- O*NET occupation facts cite the local DuckDB source.
- Interest Profiler scores, Job Zones, and career-listing context cite the local O*NET PDFs.
- Anthropic evidence cites the local structured evidence and source URL when rows are used.
- NBER evidence is listed as broad context and is not presented as task-level automation data.
- Inferred "day in the life" paragraphs are labeled as illustrative inference.

Run the final-report smoke test without an LLM API key:

```powershell
.venv\Scripts\python.exe scripts\test_ip_final_report.py
```

The report still renders when `OPENAI_API_KEY` is missing. In that case, `report_generation_method` is `template_fallback`, using deterministic templates plus retrieved local evidence.

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

## AI Exposure Evidence Pipeline

This newer pipeline keeps AI-exposure statistics separate from general research text.

Statistics sources:

- NBER Working Paper 31222
- Anthropic Economic Index

Non-statistics research sources:

- Used only for methodology, definitions, caveats, and short inference.
- Not allowed to produce numeric AI-impact claims.

Command order:

```powershell
.venv\Scripts\python.exe -m career_rag.download_research_sources --source-file source_url.txt
.venv\Scripts\python.exe -m career_rag.extract_research_chunks
.venv\Scripts\python.exe -m career_rag.extract_nber_w31222_ai_exposure --use-llm
.venv\Scripts\python.exe -m career_rag.build_anthropic_economic_index --download-if-missing
.venv\Scripts\python.exe -m career_rag.merge_ai_impact_evidence
.venv\Scripts\python.exe -m career_rag.embed_ai_impact_evidence
.venv\Scripts\python.exe -m career_rag.ai_impact_retriever "data scientist mathematical modeling ai exposure" --soc-code 15-2051.00 --top-k 8 --debug
.venv\Scripts\python.exe -m career_rag.generator "What does data scientists do, how is the ai exposure?" --show-sources
```

If LLM extraction is unavailable, run NBER extraction without `--use-llm`:

```powershell
.venv\Scripts\python.exe -m career_rag.extract_nber_w31222_ai_exposure
```

Inputs and outputs:

| Step | Input | Output |
| --- | --- | --- |
| Download sources | `data/research/source_urls.txt` or `source_url.txt` | `data/research_sources/raw/`, `data/research_sources/source_manifest.jsonl` |
| Research chunks | `data/research_sources/source_manifest.jsonl` | `data/processed/research_inference_chunks.jsonl` |
| NBER W31222 extraction | NBER row in manifest | `data/processed/nber_w31222_ai_exposure.jsonl`, `data/processed/nber_w31222_method_chunks.jsonl`, `data/processed/nber_w31222_extraction_errors.jsonl` |
| Anthropic ingestion | `data/anthropic_economic_index/` or Hugging Face download | `data/processed/anthropic_ai_impact.jsonl`, `data/processed/anthropic_ai_impact_errors.jsonl` |
| Merge/dedup | Anthropic and NBER structured JSONL | `data/processed/ai_impact_evidence.jsonl`, `data/processed/ai_impact_evidence_deduped.jsonl` |
| Embed | Deduped evidence and inference chunks | `chroma_ai_impact/ai_impact_evidence`, `chroma_research/research_inference` |

Chroma collections:

- `data/chroma_onet/onet_sections`: O*NET section evidence.
- `chroma_ai_impact/ai_impact_evidence`: structured statistics from Anthropic Economic Index and NBER W31222.
- `chroma_research/research_inference`: methodology/caveat chunks only.

Generator usage:

```powershell
.venv\Scripts\python.exe -m career_rag.generator
.venv\Scripts\python.exe -m career_rag.generator "What does a data scientist do, and how is the AI exposure?" --show-sources
```

When no query is passed, the CLI prompts:

```text
Enter your career question:
```

Limitations:

- Exposure does not equal job loss.
- Occupation-level exposure should not be presented as task-level exposure.
- Firm-level exposure should not be converted into task-level exposure.
- Non-statistics papers are not allowed to produce numeric AI-impact claims.
- LLM extraction from NBER W31222 must be checked because it may miss or misread values.
- Anthropic data reflects observed Claude usage and source-specific task mappings, not a complete forecast of labor-market outcomes.

Smoke test:

```powershell
.venv\Scripts\python.exe -m career_rag.test_ai_impact_pipeline
```

The smoke test writes `test_results_ai_impact.txt`.

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
- Large local artifacts are intentionally ignored uncompressed by git, including DuckDB files, ChromaDB stores, downloaded PDFs, generated JSONL/CSV/XLSX research outputs, and O*NET SQL dumps.
- To reproduce the current local dataset from a fresh clone, run `python scripts/restore_data_archives.py`.
- To rebuild from source instead, place/download the O*NET SQL source files under `onet_sql/`, keep research URLs in `data/research/source_urls.txt`, then run the build flow above.
- `generator.py` is the user-facing answer path.
- `retriever.py` and `research_retriever.py` are reusable retrieval layers.
- `rag_diagnostics.py` is for inspection and smoke checks only; it should not be needed for normal answers.
