#!/usr/bin/env python3
"""Generator layer and simple CLI for the O*NET career guidance RAG system.

Flow:
user question -> optional query rewrite -> retrieve O*NET evidence -> generate answer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

try:
    from career_rag.retriever import OnetRetriever, detect_query_type
    from career_rag.ai_impact_retriever import (
        DEFAULT_AI_COLLECTION as AI_IMPACT_COLLECTION_NAME,
        DEFAULT_RESEARCH_COLLECTION as RESEARCH_INFERENCE_COLLECTION_NAME,
        retrieve_ai_impact,
        retrieve_research_inference,
    )
    from career_rag.research_retriever import (
        DEFAULT_COLLECTION as RESEARCH_COLLECTION_NAME,
        format_pages as format_research_pages,
        format_research_source,
        retrieve_research_claims,
    )
except ImportError:  # Allows: py career_rag/generator.py
    from retriever import OnetRetriever, detect_query_type  # type: ignore
    from ai_impact_retriever import (  # type: ignore
        DEFAULT_AI_COLLECTION as AI_IMPACT_COLLECTION_NAME,
        DEFAULT_RESEARCH_COLLECTION as RESEARCH_INFERENCE_COLLECTION_NAME,
        retrieve_ai_impact,
        retrieve_research_inference,
    )
    from research_retriever import (  # type: ignore
        DEFAULT_COLLECTION as RESEARCH_COLLECTION_NAME,
        format_pages as format_research_pages,
        format_research_source,
        retrieve_research_claims,
    )


DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_K = 8
DEFAULT_RESEARCH_TOP_K = 8
DEFAULT_AI_IMPACT_TOP_K = 8
DEFAULT_RESEARCH_INFERENCE_TOP_K = 4

MISSING_API_KEY_MESSAGE = (
    "OPENAI_API_KEY was not found. Check that .env exists in the project root "
    "and contains OPENAI_API_KEY=..."
)
QUOTA_MESSAGE = "OpenAI API quota/billing issue. Check Platform billing and usage limits."

SYSTEM_PROMPT = """You are a career guidance assistant.

Use retrieved O*NET evidence as the primary career information.
Do not invent facts.
If the evidence is insufficient, say what is missing.
Explain clearly and practically.
When useful, organize the answer into sections.

You are given two types of evidence.

O*NET evidence:
- Describes what an occupation does.
- Use it for tasks, skills, knowledge, software, education, work context, work styles, and day-to-day work.

Research evidence:
- Structured AI impact evidence comes only from Anthropic Economic Index and NBER Working Paper 31222.
- Research inference chunks are only for methodology, definitions, caveats, and limitations.
- Do not use numeric statistics from any other source.

If supplemental evidence is provided, use it for:
- occupation aliases
- related occupations / alternative careers
- task-to-DWA mappings
- skill/ability/work-style linkages

You may synthesize and explain, but do not confuse related occupations with exact matches.
For alternative career questions, prioritize the related occupations evidence.
For task-DWA questions, explicitly mention task statements and mapped detailed work activities.

For career questions, prefer these sections:
- Short answer
- Why it fits
- Relevant tasks
- Skills/knowledge needed
- Work style/work context
- Education/training
- Caveats

Strict grounding rules:
1. Do not make AI-impact claims unless supported by research evidence.
2. Every numeric AI-impact value must come from retrieved Anthropic Economic Index or NBER W31222 metadata/evidence.
3. Do not treat automation exposure as guaranteed job loss.
4. Distinguish automation exposure, augmentation potential, task transformation, job loss, job creation, skill change, and worker vulnerability.
5. Use O*NET evidence for what the occupation involves.
6. Use research evidence for how AI may affect it.
7. If evidence is mixed or uncertain, say so.
8. Cite research sources using title, year, and page metadata.
9. If evidence is task-level, occupation-level, or firm-level, state that scope.
10. Do not make layoffs or job-disappearance claims unless retrieved evidence explicitly supports them.

Useful wording for AI-impact answers:
- The research evidence suggests exposure rather than guaranteed replacement.
- This claim is task-level rather than occupation-specific, so it should be interpreted as an indirect signal.
- I did not find an indexed task-level AI exposure statistic for this specific task.
- The indexed source supports the exposure method, but does not provide a specific numeric value for this task.

O*NET evidence describes occupations, tasks, skills, knowledge, work context, interests, education, and software. It does not by itself prove AI automation risk.
"""

AI_IMPACT_KEYWORDS = (
    "artificial intelligence",
    "generative ai",
    "gen ai",
    "large language model",
    "automation",
    "automate",
    "automated",
    "ai exposure",
    "replaced by ai",
    "future of work",
    "future of this career",
    "job impact",
    "job loss",
    "job creation",
    "skill change",
    "augmentation",
    "task impact",
    "displacement",
    "exposure",
    "replace",
    "replacement",
    "at risk",
    "generative ai",
    "claude",
    "chatgpt",
    "artificial intelligence",
)


def _configure_console_encoding() -> None:
    """Keep Windows consoles from failing on Unicode titles or answers."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()

ALTERNATIVE_CAREER_KEYWORDS = (
    "alternative careers",
    "similar careers",
    "related careers",
    "related jobs",
    "career alternatives",
    "jobs like",
    "similar jobs",
)

TASK_DWA_KEYWORDS = (
    "dwa",
    "detailed work activities",
    "task mapping",
    "task-dwa",
    "task dwa",
    "connected to tasks",
    "connected to data scientist tasks",
    "work activities connected to tasks",
)

CONTENT_MODEL_LINKAGE_KEYWORDS = (
    "linked to skill",
    "linked to mathematics",
    "work activities are linked to",
    "skills linked to work context",
    "abilities linked to work activities",
    "work styles linked to work context",
    "linked to work activities",
    "linked to work context",
)

ALIAS_HINT_KEYWORDS = (
    "bi analyst",
    "cfo",
    "ceo",
    "cto",
    "cio",
    "ciso",
    "actuarial clerk",
    "business analyst",
    "business intelligence analyst",
    "software engineer",
    "systems analyst",
    "data analyst",
)

KNOWN_OCCUPATION_HINTS = {
    "actuar": "Actuaries",
    "bi analyst": "Business Intelligence Analysts",
    "business intelligence": "Business Intelligence Analysts",
    "data scientist": "Data Scientists",
    "data scientists": "Data Scientists",
    "software engineer": "Software Developers",
    "software developer": "Software Developers",
    "business analyst": "Business Intelligence Analysts",
}


def is_ai_impact_query(user_question: str) -> bool:
    """Return True when a query asks about AI, automation, or job impact."""
    query_lower = user_question.lower()
    if re.search(r"\b(ai|llm|llms)\b", query_lower):
        return True
    return _contains_any(query_lower, AI_IMPACT_KEYWORDS)


def analyze_query(user_question: str) -> dict[str, Any]:
    """Build a retrieval plan for normal and supplemental O*NET evidence."""
    query = user_question.strip()
    query_lower = query.lower()
    normal_intent = _map_detected_intent_for_analysis(detect_query_type(query))

    intent = normal_intent
    supplemental_needed = False
    supplemental_doc_type: str | None = None

    if _contains_any(query_lower, ALTERNATIVE_CAREER_KEYWORDS):
        intent = "alternatives"
        supplemental_needed = True
        supplemental_doc_type = "related_occupations"
    elif _contains_any(query_lower, TASK_DWA_KEYWORDS):
        intent = "task_dwa"
        supplemental_needed = True
        supplemental_doc_type = "task_dwa_mapping"
    elif _contains_any(query_lower, CONTENT_MODEL_LINKAGE_KEYWORDS):
        intent = "content_model_linkage"
        supplemental_needed = True
        supplemental_doc_type = "content_model_linkage"
    elif _looks_like_alias_query(query):
        intent = "aliases"
        supplemental_needed = True
        supplemental_doc_type = "occupation_aliases"

    occupation_hint = _infer_simple_occupation_hint(query_lower)
    search_queries = [query]
    if occupation_hint:
        search_queries.append(f"{occupation_hint} {query}")
    if supplemental_doc_type:
        search_queries.append(f"{query} {supplemental_doc_type.replace('_', ' ')}")

    return {
        "intent": intent,
        "occupation_hint": occupation_hint,
        "supplemental_needed": supplemental_needed,
        "supplemental_doc_type": supplemental_doc_type,
        "ai_impact_query": is_ai_impact_query(query),
        "search_queries": _dedupe_texts(search_queries),
    }


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return True when any keyword phrase appears in text."""
    return any(keyword in text for keyword in keywords)


def _looks_like_alias_query(query: str) -> bool:
    """Detect informal job titles and abbreviations worth resolving via aliases."""
    query_lower = query.lower()
    if _contains_any(query_lower, ALIAS_HINT_KEYWORDS):
        return True

    tokens = [token.strip(".,?!:;()[]{}") for token in query.split()]
    has_abbreviation = any(
        token.isupper() and 2 <= len(token) <= 5
        for token in tokens
        if token
    )
    has_title_word = any(
        word in query_lower
        for word in ("analyst", "engineer", "clerk", "officer", "manager")
    )
    return has_abbreviation and has_title_word


def _infer_simple_occupation_hint(query_lower: str) -> str | None:
    """Extract a small set of common occupation hints without calling an LLM."""
    for phrase, title in KNOWN_OCCUPATION_HINTS.items():
        if phrase in query_lower:
            return title
    return None


def _map_detected_intent_for_analysis(detected_intent: str) -> str:
    """Map retriever query types to generator analysis intents."""
    if detected_intent == "technology_career":
        return "occupation_info"
    if detected_intent in {"tasks", "interests"}:
        return "occupation_info"
    if detected_intent in {"skills", "software", "education", "day_in_life"}:
        return detected_intent
    return "general"


def _dedupe_texts(values: list[Any]) -> list[str]:
    """Deduplicate non-empty text values while preserving order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


class CareerRAGGenerator:
    """Retrieve O*NET evidence and generate career guidance answers."""

    def __init__(
        self,
        retriever: OnetRetriever | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        use_query_rewriting: bool = True,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        """Initialize the OpenAI client and O*NET retriever."""
        self._load_dotenv_if_available()

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        self.client_setup_warning = ""
        if api_key:
            try:
                from openai import OpenAI
            except ImportError:
                self.client_setup_warning = (
                    "The OpenAI Python package is not installed; using fallback generation."
                )
            else:
                self.client = OpenAI(api_key=api_key)
        else:
            self.client_setup_warning = (
                f"{MISSING_API_KEY_MESSAGE} Using fallback generation."
            )
        self.model_name = os.getenv("OPENAI_MODEL") or model_name
        self.use_query_rewriting = use_query_rewriting
        self.max_context_chars = max_context_chars
        self.retriever = retriever or OnetRetriever()
        self.last_rewrite_result: dict[str, Any] | None = None
        self.last_query_analysis: dict[str, Any] | None = None

    def rewrite_query(self, user_question: str) -> dict[str, Any]:
        """Rewrite a user question into better retrieval queries."""
        self._validate_question(user_question)

        fallback = self._fallback_rewrite(user_question)
        if not self.use_query_rewriting or self.client is None:
            return fallback

        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite career guidance questions for retrieval over "
                    "O*NET occupation evidence. Return only valid JSON. Do not "
                    "answer the user's question."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return a JSON object with these keys:\n"
                    "- original_question: string\n"
                    "- search_queries: array of 2 to 4 strings\n"
                    "- detected_intent: one of skills, software, day_in_life, "
                    "education, career_match, general\n"
                    "- occupation_hint: occupation name string or null\n\n"
                    "Rules:\n"
                    "1. Keep the original question exactly.\n"
                    "2. Generate 2-4 improved search queries.\n"
                    "3. Extract an occupation only if the user mentioned one.\n"
                    "4. Do not hallucinate an occupation.\n"
                    "5. Do not answer the question.\n"
                    "6. Only improve retrieval.\n\n"
                    f"User question: {user_question}"
                ),
            },
        ]

        try:
            content = self._chat_completion(
                messages=messages,
                temperature=0.1,
                response_format_json=True,
            )
            parsed = self._parse_json_object(content)
            return self._normalize_rewrite_result(parsed, user_question)
        except Exception:
            return fallback

    def retrieve_evidence(
        self,
        user_question: str,
        k_per_query: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve, merge, deduplicate, and rank O*NET evidence chunks."""
        self._validate_question(user_question)
        self._validate_k(k_per_query)

        rewrite_result = self.rewrite_query(user_question)
        query_analysis = self._build_query_analysis(user_question, rewrite_result)
        self.last_rewrite_result = rewrite_result
        self.last_query_analysis = query_analysis
        return self._retrieve_evidence_from_plan(
            query_analysis=query_analysis,
            rewrite_result=rewrite_result,
            k_per_query=k_per_query,
        )

    def format_context(self, evidence: list[dict[str, Any]]) -> str:
        """Format retrieved evidence into a compact source block for the LLM."""
        if not evidence:
            return "No retrieved O*NET evidence was found."

        context_parts: list[str] = []
        used_chars = 0

        for index, item in enumerate(evidence, 1):
            metadata = item.get("metadata") or {}
            text = item.get("text") or ""
            doc_type = self._display_doc_type(item, metadata)
            source_text = (
                f"[Source {index}]\n"
                f"Occupation: {metadata.get('occupation_title') or 'N/A'}\n"
                f"O*NET Code: {metadata.get('onet_soc_code') or 'N/A'}\n"
                f"Section: {metadata.get('section') or 'N/A'}\n"
                f"Doc Type: {doc_type or 'N/A'}\n"
                f"Collection: {item.get('collection') or 'N/A'}\n"
                f"Document ID: {item.get('id') or 'N/A'}\n"
                f"Score: {float(item.get('score', 0.0)):.4f}\n"
                f"Text:\n{text}\n"
            )

            remaining_chars = self.max_context_chars - used_chars
            if remaining_chars <= 0:
                break

            if len(source_text) > remaining_chars:
                source_text = source_text[:remaining_chars].rstrip()

            context_parts.append(source_text)
            used_chars += len(source_text)

        return "\n\n".join(context_parts)

    def retrieve_research_evidence(
        self,
        user_question: str,
        onet_evidence: list[dict[str, Any]],
        top_k: int = DEFAULT_RESEARCH_TOP_K,
    ) -> list[dict[str, Any]]:
        """Retrieve research claims using the original and O*NET-expanded queries."""
        self._validate_question(user_question)
        self._validate_k(top_k)

        expanded_query = self._build_expanded_research_query(
            user_question,
            onet_evidence,
        )
        research_queries = self._dedupe_queries([user_question, expanded_query])
        best_by_claim_id: dict[str, dict[str, Any]] = {}

        for query in research_queries:
            claims = retrieve_research_claims(query, top_k=top_k)
            for claim in claims:
                claim_id = str(claim.get("claim_id") or claim.get("id") or "")
                if not claim_id:
                    continue
                current_best = best_by_claim_id.get(claim_id)
                if current_best is None or self._research_rank_value(
                    claim
                ) > self._research_rank_value(current_best):
                    best_by_claim_id[claim_id] = claim

        ranked = list(best_by_claim_id.values())
        ranked.sort(key=self._research_rank_value, reverse=True)
        return ranked[:top_k]

    def format_research_context(self, claims: list[dict[str, Any]]) -> str:
        """Format research claims into the LLM prompt block."""
        if not claims:
            return "No retrieved research evidence was found."

        context_parts: list[str] = []
        for index, claim in enumerate(claims, 1):
            affected_entity = self._format_affected_entity(claim)
            source = self._format_research_prompt_source(claim)
            context_parts.append(
                "\n".join(
                    [
                        f"[Research Claim R{index}]",
                        f"Claim: {self._one_line(claim.get('claim_text'))}",
                        f"Evidence quote: {self._one_line(claim.get('evidence_quote'))}",
                        f"Impact type: {self._one_line(claim.get('impact_type_clean'))}",
                        f"Direction: {self._one_line(claim.get('impact_direction_clean'))}",
                        f"Affected entity: {affected_entity}",
                        f"AI relevance: {self._one_line(claim.get('ai_relevance'))}",
                        f"Scope: {self._one_line(claim.get('generator_use_scope'))}",
                        f"Source: {source}",
                    ]
                )
            )

        return "\n\n".join(context_parts)

    def format_ai_impact_context(self, evidence: list[dict[str, Any]]) -> str:
        """Format structured AI-impact evidence for the LLM prompt."""
        if not evidence:
            return "No retrieved structured AI-impact evidence was found."

        context_parts: list[str] = []
        for index, item in enumerate(evidence, 1):
            metadata = item.get("metadata") or {}
            context_parts.append(
                "\n".join(
                    [
                        f"[AI Impact Evidence A{index}]",
                        f"Source: {self._one_line(metadata.get('source_name') or metadata.get('source_id'))}",
                        f"Source file/release: {self._one_line(metadata.get('source_release') or metadata.get('source_file'))}",
                        f"Page: {self._one_line(metadata.get('source_page')) or 'N/A'}",
                        f"Doc type: {self._one_line(metadata.get('doc_type'))}",
                        f"Occupation: {self._one_line(metadata.get('occupation_title')) or 'N/A'}",
                        f"SOC: {self._one_line(metadata.get('soc_code')) or 'N/A'}",
                        f"Task: {self._one_line(metadata.get('task_text')) or 'N/A'}",
                        f"Impact type: {self._one_line(metadata.get('impact_type'))}",
                        f"Metric: {self._one_line(metadata.get('metric_name'))} = {self._one_line(metadata.get('metric_value')) or 'N/A'} {self._metric_unit_for_display(metadata)}",
                        f"Metric caution: {self._metric_caution(metadata)}",
                        f"Confidence: {self._one_line(metadata.get('confidence'))}",
                        f"Evidence: {self._one_line(item.get('text'))}",
                    ]
                )
            )

        return "\n\n".join(context_parts)

    def format_research_inference_context(self, chunks: list[dict[str, Any]]) -> str:
        """Format research methodology/caveat chunks for the LLM prompt."""
        if not chunks:
            return "No retrieved research inference evidence was found."

        context_parts: list[str] = []
        for index, item in enumerate(chunks, 1):
            metadata = item.get("metadata") or {}
            context_parts.append(
                "\n".join(
                    [
                        f"[Research Inference C{index}]",
                        f"Source: {self._one_line(metadata.get('source_name') or metadata.get('source_id'))}",
                        f"Page: {self._one_line(metadata.get('source_page') or metadata.get('page')) or 'N/A'}",
                        f"Allowed usage: {self._one_line(metadata.get('allowed_usage'))}",
                        f"Statistics allowed: {self._one_line(metadata.get('statistics_allowed'))}",
                        "Numeric-use rule: Do not use numbers from this block as AI-impact statistics.",
                        f"Text: {self._redact_numbers_for_inference(self._one_line(item.get('text')))}",
                    ]
                )
            )

        return "\n\n".join(context_parts)

    def _build_expanded_research_query(
        self,
        user_question: str,
        onet_evidence: list[dict[str, Any]],
    ) -> str:
        """Build a research query enriched with retrieved O*NET context."""
        titles: list[str] = []
        snippets: list[str] = []

        for item in onet_evidence[:6]:
            metadata = item.get("metadata") or {}
            title = self._one_line(metadata.get("occupation_title"))
            if title and title not in titles:
                titles.append(title)

            text = self._one_line(item.get("text"))
            if text and len(snippets) < 3:
                snippets.append(self._truncate(text, 260))

        query_parts = [f"AI impact on {user_question}"]
        if titles:
            query_parts.append(f"Related O*NET occupations: {'; '.join(titles[:5])}")
        if snippets:
            query_parts.append(f"O*NET evidence snippets: {' '.join(snippets)}")

        return ". ".join(part for part in query_parts if part).strip()

    def _build_research_summary(
        self,
        claims: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build compact research source metadata for display or JSON output."""
        summaries: list[dict[str, Any]] = []

        for index, claim in enumerate(claims, 1):
            summaries.append(
                {
                    "rank": index,
                    "collection": claim.get("collection") or RESEARCH_COLLECTION_NAME,
                    "claim_id": claim.get("claim_id"),
                    "title": claim.get("title"),
                    "year": claim.get("year"),
                    "pages": self._page_range(claim),
                    "impact_type_clean": claim.get("impact_type_clean"),
                    "impact_direction_clean": claim.get("impact_direction_clean"),
                    "ai_relevance": claim.get("ai_relevance"),
                    "generator_use_scope": claim.get("generator_use_scope"),
                    "score": claim.get("score"),
                    "distance": claim.get("distance"),
                    "source": format_research_source(claim),
                }
            )

        return summaries

    def _build_ai_impact_summary(
        self,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build source summaries for structured AI-impact evidence."""
        summaries: list[dict[str, Any]] = []
        for index, item in enumerate(evidence, 1):
            metadata = item.get("metadata") or {}
            summaries.append(
                {
                    "rank": index,
                    "collection": item.get("collection") or AI_IMPACT_COLLECTION_NAME,
                    "source_name": metadata.get("source_name"),
                    "source_release": metadata.get("source_release"),
                    "source_file": metadata.get("source_file"),
                    "page": metadata.get("source_page"),
                    "doc_type": metadata.get("doc_type"),
                    "occupation_title": metadata.get("occupation_title"),
                    "soc_code": metadata.get("soc_code"),
                    "task_text": metadata.get("task_text"),
                    "impact_type": metadata.get("impact_type"),
                    "metric_name": metadata.get("metric_name"),
                    "metric_value": metadata.get("metric_value"),
                    "metric_unit": self._metric_unit_for_display(metadata),
                    "confidence": metadata.get("confidence"),
                    "score": item.get("reranked_score") or item.get("score"),
                    "distance": item.get("distance"),
                }
            )
        return summaries

    def _build_research_inference_summary(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build source summaries for methodology/caveat chunks."""
        summaries: list[dict[str, Any]] = []
        for index, item in enumerate(chunks, 1):
            metadata = item.get("metadata") or {}
            summaries.append(
                {
                    "rank": index,
                    "collection": item.get("collection") or RESEARCH_INFERENCE_COLLECTION_NAME,
                    "source_name": metadata.get("source_name"),
                    "page": metadata.get("source_page") or metadata.get("page"),
                    "allowed_usage": metadata.get("allowed_usage"),
                    "score": item.get("score"),
                    "distance": item.get("distance"),
                }
            )
        return summaries

    def retrieve_structured_ai_evidence(
        self,
        user_question: str,
        onet_evidence: list[dict[str, Any]],
        top_k: int = DEFAULT_AI_IMPACT_TOP_K,
    ) -> list[dict[str, Any]]:
        """Retrieve prioritized structured AI-impact evidence."""
        occupation_context = self._infer_occupation_context(onet_evidence)
        query = self._build_expanded_research_query(user_question, onet_evidence)
        try:
            return retrieve_ai_impact(
                query,
                soc_code=occupation_context.get("soc_code"),
                occupation_title=occupation_context.get("occupation_title"),
                task_texts=occupation_context.get("task_texts") or [],
                top_k=top_k,
            )
        except Exception:
            return []

    def retrieve_research_inference_evidence(
        self,
        user_question: str,
        onet_evidence: list[dict[str, Any]],
        top_k: int = DEFAULT_RESEARCH_INFERENCE_TOP_K,
    ) -> list[dict[str, Any]]:
        """Retrieve short methodology/caveat chunks."""
        query = self._build_expanded_research_query(user_question, onet_evidence)
        try:
            return retrieve_research_inference(query, top_k=top_k)
        except Exception:
            return []

    def _infer_occupation_context(
        self,
        onet_evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Infer the leading occupation/SOC/tasks from retrieved O*NET evidence."""
        title_counts: dict[str, int] = {}
        soc_counts: dict[str, int] = {}
        task_texts: list[str] = []
        first_title: str | None = None
        first_soc: str | None = None

        for item in onet_evidence:
            metadata = item.get("metadata") or {}
            title = self._one_line(metadata.get("occupation_title"))
            soc = self._one_line(metadata.get("onet_soc_code") or metadata.get("soc_code"))
            if title and not first_title:
                first_title = title
            if soc and not first_soc:
                first_soc = soc
            if first_title and first_soc:
                break

        for item in onet_evidence:
            metadata = item.get("metadata") or {}
            title = self._one_line(metadata.get("occupation_title"))
            soc = self._one_line(metadata.get("onet_soc_code") or metadata.get("soc_code"))
            if title:
                title_counts[title] = title_counts.get(title, 0) + 1
            if soc:
                soc_counts[soc] = soc_counts.get(soc, 0) + 1
            if (first_soc and soc == first_soc) or (not first_soc and title == first_title):
                task_texts.extend(self._extract_task_lines(item.get("text")))

        occupation_title = first_title or (max(title_counts, key=title_counts.get) if title_counts else None)
        soc_code = first_soc or (max(soc_counts, key=soc_counts.get) if soc_counts else None)
        return {
            "occupation_title": occupation_title,
            "soc_code": soc_code,
            "task_texts": self._dedupe_queries(task_texts)[:12],
        }

    def _extract_task_lines(self, text: Any) -> list[str]:
        """Extract likely task statements from O*NET evidence text."""
        lines: list[str] = []
        for line in str(text or "").splitlines():
            cleaned = self._one_line(line)
            if not cleaned.startswith("- "):
                continue
            task = cleaned[2:].strip()
            if len(task.split()) < 4:
                continue
            if ":" in task and re.match(r"^[A-Z][A-Za-z /&-]{2,40}:", task):
                continue
            lines.append(task)
        return lines

    def _build_ai_answer_prompt(
        self,
        user_question: str,
        query_analysis: dict[str, Any],
        onet_context: str,
        ai_impact_context: str,
        research_inference_context: str,
    ) -> list[dict[str, str]]:
        """Build the stricter prompt for AI-exposure answers."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Answer the user's career question in about 300-500 words.\n"
                    "Use this structure with short headings:\n"
                    "1. Short career explanation\n"
                    "2. Main tasks\n"
                    "3. AI exposure by task\n"
                    "4. Inference\n"
                    "5. Future outlook\n"
                    "6. Sources\n\n"
                    "Rules:\n"
                    "- Use O*NET evidence for what the occupation does and its tasks.\n"
                    "- Use structured AI impact evidence for AI statistics.\n"
                    "- Prefer Anthropic task-level rows for automation, augmentation, task penetration, observed usage, and job exposure.\n"
                    "- Use NBER W31222 for exposure methodology, direct/indirect exposure, and core/supplemental distinctions.\n"
                    "- Research inference chunks are only for caveats or methodology, not numeric statistics.\n"
                    "- Do not fabricate values. Every numeric AI-impact value must appear in the structured AI evidence block.\n"
                    "- Do not convert share/score metrics into percentages unless metric_unit is percent.\n"
                    "- A metric value of 0.0 share means 0.0 in the indexed Anthropic measure; do not call it no AI impact, no exposure, or no automation unless impact_type is explicitly no_exposure.\n"
                    "- For 0.0 share rows, say '0.0 share in the indexed measure'; do not say 'no current AI penetration' or imply absence of AI impact.\n"
                    "- Do not say exposure means replacement, disappearance, layoffs, or safety from AI.\n"
                    "- In Future outlook, discuss task change and skill adaptation only. Do not forecast demand, hiring, employment, openings, growth, or decline unless retrieved evidence explicitly says so.\n"
                    "- Only list NBER W31222 in Sources if the answer uses a specific NBER structured claim or NBER method distinction.\n"
                    "- In Sources, list only sources that support claims actually used in the answer.\n"
                    "- Never use research inference chunks for numeric AI-impact claims; those chunks are only caveats/methodology.\n"
                    "- If no indexed task-level AI exposure statistic is found for a task, say so.\n\n"
                    f"USER QUESTION:\n{user_question}\n\n"
                    "RETRIEVAL PLAN:\n"
                    f"{json.dumps(query_analysis, ensure_ascii=False)}\n\n"
                    f"RETRIEVED O*NET EVIDENCE:\n{onet_context}\n\n"
                    f"STRUCTURED AI IMPACT EVIDENCE:\n{ai_impact_context}\n\n"
                    f"RESEARCH INFERENCE / METHODOLOGY EVIDENCE:\n{research_inference_context}"
                ),
            },
        ]

    def _fallback_answer(
        self,
        user_question: str,
        onet_evidence: list[dict[str, Any]],
        ai_impact_evidence: list[dict[str, Any]],
        research_inference: list[dict[str, Any]],
        ai_impact_needed: bool,
    ) -> str:
        """Build a deterministic answer if the LLM is unavailable."""
        occupation_context = self._infer_occupation_context(onet_evidence)
        occupation_title = occupation_context.get("occupation_title") or "this occupation"
        description = self._first_occupation_description(onet_evidence)
        tasks = occupation_context.get("task_texts") or []
        task_lines = tasks[:5]

        parts = [
            "Short career explanation",
            description
            or f"{occupation_title} work on the tasks and responsibilities described in the retrieved O*NET evidence.",
            "",
            "Main tasks",
        ]
        if task_lines:
            parts.extend(f"- {task}" for task in task_lines)
        else:
            parts.append("- I found O*NET occupation evidence, but not a clean task list in the retrieved chunks.")

        if ai_impact_needed:
            parts.extend(["", "AI exposure by task"])
            metric_lines = self._fallback_ai_metric_lines(ai_impact_evidence)
            if metric_lines:
                parts.extend(metric_lines[:5])
            else:
                parts.append("- I did not find an indexed task-level AI exposure statistic for this specific task.")

            parts.extend(
                [
                    "",
                    "Inference",
                    "The retrieved evidence should be read as exposure or observed usage, not as proof that the occupation will be replaced. Coding, modeling, drafting, and repeatable information-processing tasks are usually the parts to inspect first; judgment-heavy communication and accountable decisions need more caution.",
                    "",
                    "Future outlook",
                    "Expect task change rather than a simple job-disappearance conclusion. The safest reading is that AI may augment or automate parts of the work depending on the specific task and metric scope.",
                    "",
                    "Sources",
                ]
            )
            source_names = self._fallback_source_names(ai_impact_evidence, research_inference)
            parts.extend(f"- {source}" for source in source_names)
        else:
            parts.extend(
                [
                    "",
                    "Sources",
                    "- O*NET retrieved occupation evidence.",
                ]
            )

        if self.client_setup_warning:
            parts.append("")
            parts.append(f"Note: {self.client_setup_warning}")
        return "\n".join(parts)

    def _first_occupation_description(self, onet_evidence: list[dict[str, Any]]) -> str:
        """Extract a short occupation description from retrieved O*NET text."""
        for item in onet_evidence:
            text = str(item.get("text") or "")
            for line in text.splitlines():
                cleaned = self._one_line(line)
                if not cleaned or cleaned.startswith("- "):
                    continue
                if "O*NET-SOC Code:" in cleaned or cleaned.endswith(" - Tasks"):
                    continue
                if len(cleaned.split()) >= 8:
                    return self._truncate(cleaned, 280)
        return ""

    def _fallback_ai_metric_lines(
        self,
        ai_impact_evidence: list[dict[str, Any]],
    ) -> list[str]:
        """Format direct metric lines for fallback generation."""
        lines: list[str] = []
        for item in ai_impact_evidence:
            metadata = item.get("metadata") or {}
            source = self._one_line(metadata.get("source_name") or metadata.get("source_id"))
            task = self._one_line(metadata.get("task_text")) or "occupation-level evidence"
            metric_name = self._one_line(metadata.get("metric_name"))
            metric_value = self._one_line(metadata.get("metric_value"))
            metric_unit = self._metric_unit_for_display(metadata)
            impact_type = self._one_line(metadata.get("impact_type"))
            if metric_name and metric_value:
                caution = ""
                if self._metric_caution(metadata).startswith("0.0 share"):
                    caution = " This is not by itself evidence of no AI impact."
                lines.append(
                    f"- For {task}, {source} reports {impact_type} metric {metric_name} = {metric_value} {metric_unit}.{caution}".rstrip()
                )
            elif source:
                lines.append(
                    f"- {source} provides {impact_type or 'methodology'} evidence for {task}, but no specific numeric value is available in this row."
                )
        return lines

    def _fallback_source_names(
        self,
        ai_impact_evidence: list[dict[str, Any]],
        research_inference: list[dict[str, Any]],
    ) -> list[str]:
        """Collect compact source names for fallback answers."""
        sources: list[str] = []
        for item in ai_impact_evidence + research_inference:
            metadata = item.get("metadata") or {}
            source = self._one_line(metadata.get("source_name") or metadata.get("source_id"))
            if source and source not in sources:
                sources.append(source)
        return sources or ["O*NET retrieved occupation evidence."]

    def _postprocess_ai_answer(self, answer: str, ai_impact_needed: bool) -> str:
        """Apply final wording guards for structured AI-impact answers."""
        if not ai_impact_needed or not answer:
            return answer
        guarded = self._guard_zero_share_language(answer)
        guarded = self._guard_labor_demand_forecasts(guarded)
        guarded = self._remove_unused_nber_source(guarded)
        return guarded

    @staticmethod
    def _guard_zero_share_language(answer: str) -> str:
        """Keep 0.0 share wording scoped to the indexed metric."""
        replacements = [
            (
                r"\bno current AI penetration\b",
                "0.0 share in the indexed measure",
            ),
            (
                r"\bno observed AI penetration\b",
                "0.0 share in the indexed measure",
            ),
            (
                r"\bno significant AI penetration\b",
                "low or zero indexed penetration in the retrieved measure",
            ),
            (
                r"\bno current AI exposure\b",
                "0.0 share in the indexed exposure measure",
            ),
            (
                r"\bnot currently exposed to AI\b",
                "reported as 0.0 share in the indexed measure",
            ),
        ]
        guarded = answer
        for pattern, replacement in replacements:
            guarded = re.sub(pattern, replacement, guarded, flags=re.IGNORECASE)
        return guarded

    @staticmethod
    def _guard_labor_demand_forecasts(answer: str) -> str:
        """Replace unsupported labor-market forecasts in the outlook section."""
        lines = answer.splitlines()
        guarded_lines: list[str] = []
        in_future_outlook = False
        section_names = {
            "shortcareerexplanation",
            "maintasks",
            "aiexposurebytask",
            "inference",
            "futureoutlook",
            "sources",
        }
        labor_terms = re.compile(
            r"\b(demand|employment|job openings|openings|hiring|jobs?)\b",
            flags=re.IGNORECASE,
        )
        forecast_terms = re.compile(
            r"\b(will|likely|expected|projected|increase|decrease|grow|growth|"
            r"shrink|decline|rise|fall|expand|contract)\b",
            flags=re.IGNORECASE,
        )
        replacement = (
            "The retrieved evidence supports task-change discussion, not a "
            "labor-market demand forecast."
        )
        for line in lines:
            stripped = line.strip()
            heading_key = re.sub(r"[^a-z]", "", stripped.lower())
            if heading_key in section_names:
                in_future_outlook = heading_key == "futureoutlook"

            if (
                in_future_outlook
                and labor_terms.search(stripped)
                and forecast_terms.search(stripped)
            ):
                prefix_match = re.match(r"^(\s*(?:[-*]\s*)?)", line)
                prefix = prefix_match.group(1) if prefix_match else ""
                guarded_lines.append(f"{prefix}{replacement}")
                continue
            guarded_lines.append(line)
        return "\n".join(guarded_lines)

    @staticmethod
    def _remove_unused_nber_source(answer: str) -> str:
        """Omit NBER from Sources when no NBER claim is used in the body."""
        lines = answer.splitlines()
        sources_index: int | None = None
        for index, line in enumerate(lines):
            heading_key = re.sub(r"[^a-z]", "", line.strip().lower())
            if heading_key == "sources":
                sources_index = index
                break
        if sources_index is None:
            return answer

        body = "\n".join(lines[:sources_index])
        nber_marker = re.compile(r"\b(nber|w31222)\b", flags=re.IGNORECASE)
        nber_specific_terms = re.compile(
            r"\b(direct exposure|indirect exposure|"
            r"core task exposure|supplemental task exposure|core/supplemental|"
            r"core and supplemental|firm exposure|firm value|firm valuation|"
            r"highest-exposure quintile|job postings|hourly wage|labor demand|"
            r"analyst forecasts|profitability|abnormal returns|"
            r"exposure methodology|method distinction)\b",
            flags=re.IGNORECASE,
        )
        if nber_marker.search(body) and nber_specific_terms.search(body):
            return answer

        filtered_lines = [
            line
            for index, line in enumerate(lines)
            if not (
                index > sources_index
                and re.search(r"\b(nber|w31222)\b", line, flags=re.IGNORECASE)
            )
        ]
        return "\n".join(filtered_lines)

    def _metric_unit_for_display(self, metadata: dict[str, Any]) -> str:
        """Display conservative units for common AI metric names."""
        unit = self._one_line(metadata.get("metric_unit"))
        if unit:
            return unit
        metric_name = self._one_line(metadata.get("metric_name")).lower()
        impact_type = self._one_line(metadata.get("impact_type")).lower()
        if "penetration" in metric_name or "penetration" in impact_type:
            return "share"
        if "exposure" in metric_name or "exposure" in impact_type:
            return "share"
        return ""

    def _metric_caution(self, metadata: dict[str, Any]) -> str:
        """Return caution text for metric interpretation."""
        metric_value = self._one_line(metadata.get("metric_value"))
        impact_type = self._one_line(metadata.get("impact_type")).lower()
        unit = self._metric_unit_for_display(metadata)
        if metric_value in {"0", "0.0", "0.00"} and unit == "share":
            return (
                "0.0 share is the value in the indexed measure; it does not by "
                "itself prove no AI impact or no future exposure."
            )
        if impact_type == "no_exposure":
            return "This row is explicitly labeled no_exposure by the source extraction."
        return "Use the metric only in its source scope."

    @staticmethod
    def _redact_numbers_for_inference(text: str) -> str:
        """Prevent research inference chunks from supplying numeric AI-impact claims."""
        return re.sub(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?%?", "[number omitted]", text)

    @staticmethod
    def _research_rank_value(claim: dict[str, Any]) -> float:
        """Rank research claims by score with a small direct-AI preference."""
        score = float(claim.get("score") or 0.0)
        relevance = str(claim.get("ai_relevance") or "").strip()
        if relevance in {"direct_ai", "automation_or_technology"}:
            score += 0.03
        return score

    @staticmethod
    def _format_affected_entity(claim: dict[str, Any]) -> str:
        """Format affected entity type and text for the research prompt."""
        entity_type = CareerRAGGenerator._one_line(claim.get("affected_entity_type"))
        entity_text = CareerRAGGenerator._one_line(claim.get("affected_entity_text"))
        if entity_type and entity_text:
            return f"{entity_type}: {entity_text}"
        return entity_text or entity_type or "N/A"

    @staticmethod
    def _format_research_prompt_source(claim: dict[str, Any]) -> str:
        """Format the richer source line used in the LLM research block."""
        title = CareerRAGGenerator._one_line(
            claim.get("title") or claim.get("source_id") or claim.get("claim_id")
        )
        authors = CareerRAGGenerator._one_line(claim.get("authors"))
        year = CareerRAGGenerator._one_line(claim.get("year"))
        pages = format_research_pages(claim)
        url = CareerRAGGenerator._one_line(
            claim.get("final_url") or claim.get("url")
        )
        parts = [part for part in (title, authors, year, pages, url) if part]
        return ", ".join(parts) if parts else "N/A"

    @staticmethod
    def _page_range(claim: dict[str, Any]) -> str:
        """Format page metadata for the CLI source table."""
        page_start = CareerRAGGenerator._one_line(claim.get("page_start"))
        page_end = CareerRAGGenerator._one_line(claim.get("page_end"))
        if page_start.endswith(".0"):
            page_start = page_start[:-2]
        if page_end.endswith(".0"):
            page_end = page_end[:-2]
        if not page_start and not page_end:
            return "N/A"
        if not page_start:
            return page_end
        if not page_end or page_start == page_end:
            return page_start
        return f"{page_start}-{page_end}"

    @staticmethod
    def _one_line(value: Any) -> str:
        """Return compact single-line text for prompt and CLI display."""
        if value is None:
            return ""
        return " ".join(str(value).split())

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """Truncate text without splitting the final word when possible."""
        if len(text) <= limit:
            return text
        shortened = text[:limit].rsplit(" ", 1)[0].rstrip()
        return f"{shortened}..."

    def generate_answer(
        self,
        user_question: str,
        k: int = DEFAULT_K,
        use_research: bool = False,
        research_top_k: int = DEFAULT_RESEARCH_TOP_K,
    ) -> dict[str, Any]:
        """Generate an answer using O*NET evidence and optional research evidence."""
        self._validate_question(user_question)
        self._validate_k(k)
        self._validate_k(research_top_k)

        rewrite_result = self.rewrite_query(user_question)
        query_analysis = self._build_query_analysis(user_question, rewrite_result)
        ai_impact_needed = bool(query_analysis.get("ai_impact_query"))
        old_research_needed = bool(use_research and not ai_impact_needed)
        query_analysis["use_research"] = bool(old_research_needed or ai_impact_needed)
        query_analysis["use_structured_ai_impact"] = ai_impact_needed
        self.last_rewrite_result = rewrite_result
        self.last_query_analysis = query_analysis

        evidence = self._retrieve_evidence_from_plan(
            query_analysis=query_analysis,
            rewrite_result=rewrite_result,
            k_per_query=k,
        )[:k]
        context = self.format_context(evidence)
        research_claims: list[dict[str, Any]] = []
        research_context = "No retrieved research evidence was requested."
        ai_impact_evidence: list[dict[str, Any]] = []
        ai_impact_context = "No structured AI-impact evidence was requested."
        research_inference: list[dict[str, Any]] = []
        research_inference_context = "No research inference evidence was requested."
        expanded_research_query = ""

        if ai_impact_needed:
            ai_impact_evidence = self.retrieve_structured_ai_evidence(
                user_question=user_question,
                onet_evidence=evidence,
                top_k=research_top_k,
            )
            research_inference = self.retrieve_research_inference_evidence(
                user_question=user_question,
                onet_evidence=evidence,
                top_k=DEFAULT_RESEARCH_INFERENCE_TOP_K,
            )
            expanded_research_query = self._build_expanded_research_query(
                user_question,
                evidence,
            )
            ai_impact_context = self.format_ai_impact_context(ai_impact_evidence)
            research_inference_context = self.format_research_inference_context(
                research_inference
            )
            messages = self._build_ai_answer_prompt(
                user_question=user_question,
                query_analysis=query_analysis,
                onet_context=context,
                ai_impact_context=ai_impact_context,
                research_inference_context=research_inference_context,
            )
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Answer the user's question using the retrieved evidence "
                        "below. Use O*NET evidence for what the occupation involves "
                        "and research evidence for AI-impact claims.\n\n"
                        f"USER QUESTION:\n{user_question}\n\n"
                        "RETRIEVAL PLAN:\n"
                        f"{json.dumps(query_analysis, ensure_ascii=False)}\n\n"
                        f"RETRIEVED O*NET EVIDENCE:\n{context}\n\n"
                        f"RETRIEVED RESEARCH EVIDENCE:\n{research_context}"
                    ),
                },
            ]

        if old_research_needed:
            research_claims = self.retrieve_research_evidence(
                user_question=user_question,
                onet_evidence=evidence,
                top_k=research_top_k,
            )
            expanded_research_query = self._build_expanded_research_query(
                user_question,
                evidence,
            )
            research_context = self.format_research_context(research_claims)
            messages[1]["content"] = messages[1]["content"].replace(
                "RETRIEVED RESEARCH EVIDENCE:\nNo retrieved research evidence was requested.",
                f"RETRIEVED RESEARCH EVIDENCE:\n{research_context}",
            )

        try:
            answer = self._chat_completion(messages=messages, temperature=0.2)
        except Exception:
            answer = self._fallback_answer(
                user_question=user_question,
                onet_evidence=evidence,
                ai_impact_evidence=ai_impact_evidence,
                research_inference=research_inference,
                ai_impact_needed=ai_impact_needed,
            )
        answer = self._postprocess_ai_answer(answer, ai_impact_needed=ai_impact_needed)

        return {
            "question": user_question,
            "answer": answer,
            "rewritten_queries": rewrite_result.get("search_queries", []),
            "query_analysis": query_analysis,
            "evidence": evidence,
            "evidence_summary": self._build_evidence_summary(evidence),
            "research_claims": research_claims,
            "research_summary": self._build_research_summary(research_claims),
            "ai_impact_evidence": ai_impact_evidence,
            "ai_impact_summary": self._build_ai_impact_summary(ai_impact_evidence),
            "research_inference": research_inference,
            "research_inference_summary": self._build_research_inference_summary(
                research_inference
            ),
            "used_research": bool(old_research_needed or ai_impact_needed),
            "used_structured_ai_impact": ai_impact_needed,
            "expanded_research_query": expanded_research_query,
        }

    def _build_query_analysis(
        self,
        user_question: str,
        rewrite_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze routing and fold rewritten searches into the plan."""
        query_analysis = analyze_query(user_question)
        rewritten_queries = rewrite_result.get("search_queries") or []
        query_analysis["search_queries"] = self._dedupe_queries(
            [
                user_question,
                *query_analysis.get("search_queries", []),
                *rewritten_queries,
            ]
        )
        return query_analysis

    def _retrieve_evidence_from_plan(
        self,
        query_analysis: dict[str, Any],
        rewrite_result: dict[str, Any],
        k_per_query: int,
    ) -> list[dict[str, Any]]:
        """Retrieve evidence using normal and supplemental routing."""
        if not query_analysis.get("supplemental_needed"):
            return self._retrieve_evidence_from_rewrite(rewrite_result, k_per_query)

        intent = str(query_analysis.get("intent") or "general")
        if intent == "alternatives":
            return self._retrieve_alternative_career_evidence(
                query_analysis,
                rewrite_result,
                k_per_query,
            )
        if intent == "aliases":
            return self._retrieve_alias_aware_evidence(
                query_analysis,
                rewrite_result,
                k_per_query,
            )
        if intent == "task_dwa":
            return self._retrieve_task_dwa_evidence(
                query_analysis,
                rewrite_result,
                k_per_query,
            )
        if intent == "content_model_linkage":
            return self._retrieve_content_model_linkage_evidence(
                query_analysis,
                rewrite_result,
                k_per_query,
            )

        return self._retrieve_evidence_from_rewrite(rewrite_result, k_per_query)

    def _retrieve_alternative_career_evidence(
        self,
        query_analysis: dict[str, Any],
        rewrite_result: dict[str, Any],
        k_per_query: int,
    ) -> list[dict[str, Any]]:
        """Retrieve related-occupation evidence and light normal context."""
        query = self._analysis_query(query_analysis, rewrite_result)
        alias_results, occupation_code, occupation_title = self._resolve_occupation(query)

        evidence_groups: list[list[dict[str, Any]]] = []
        if occupation_code:
            related_results = self._safe_retrieve(
                lambda: self.retriever.get_related_occupations(
                    occupation_code,
                    k=max(k_per_query, 10),
                )
            )
            evidence_groups.append(related_results)
            evidence_groups.append(alias_results[:1])
            evidence_groups.append(
                self._normal_occupation_evidence(
                    query=f"{occupation_title or occupation_code} occupation overview",
                    occupation_code=occupation_code,
                    k=min(3, k_per_query),
                )
            )
        else:
            evidence_groups.append(
                self._safe_retrieve(
                    lambda: self.retriever.retrieve_supplemental(
                        query,
                        k=k_per_query,
                        doc_type="related_occupations",
                    )
                )
            )
            evidence_groups.append(
                self._retrieve_evidence_from_rewrite(rewrite_result, k_per_query)
            )

        return self._merge_evidence_groups(evidence_groups, k_per_query)

    def _retrieve_alias_aware_evidence(
        self,
        query_analysis: dict[str, Any],
        rewrite_result: dict[str, Any],
        k_per_query: int,
    ) -> list[dict[str, Any]]:
        """Use alias docs to identify the official occupation, then retrieve normal evidence."""
        query = self._analysis_query(query_analysis, rewrite_result)
        alias_results, occupation_code, occupation_title = self._resolve_occupation(query)

        evidence_groups: list[list[dict[str, Any]]] = [alias_results[:2]]
        if occupation_code:
            normal_query = f"{occupation_title or ''} {query}".strip()
            evidence_groups.append(
                self._normal_occupation_evidence(
                    query=normal_query,
                    occupation_code=occupation_code,
                    k=k_per_query,
                )
            )
        else:
            evidence_groups.append(
                self._retrieve_evidence_from_rewrite(rewrite_result, k_per_query)
            )

        return self._merge_evidence_groups(evidence_groups, k_per_query)

    def _retrieve_task_dwa_evidence(
        self,
        query_analysis: dict[str, Any],
        rewrite_result: dict[str, Any],
        k_per_query: int,
    ) -> list[dict[str, Any]]:
        """Retrieve task-to-DWA mapping evidence and supporting normal context."""
        query = self._analysis_query(query_analysis, rewrite_result)
        alias_results, occupation_code, occupation_title = self._resolve_occupation(query)

        evidence_groups: list[list[dict[str, Any]]] = []
        if occupation_code:
            evidence_groups.append(
                self._safe_retrieve(
                    lambda: self.retriever.get_task_dwa_mapping(
                        occupation_code,
                        k=max(k_per_query, 10),
                    )
                )
            )
            evidence_groups.append(alias_results[:1])
            evidence_groups.append(
                self._normal_occupation_evidence(
                    query=f"{occupation_title or ''} tasks work activities".strip(),
                    occupation_code=occupation_code,
                    k=min(4, k_per_query),
                )
            )
        else:
            evidence_groups.append(
                self._safe_retrieve(
                    lambda: self.retriever.retrieve_supplemental(
                        query,
                        k=k_per_query,
                        doc_type="task_dwa_mapping",
                    )
                )
            )

        return self._merge_evidence_groups(evidence_groups, k_per_query)

    def _retrieve_content_model_linkage_evidence(
        self,
        query_analysis: dict[str, Any],
        rewrite_result: dict[str, Any],
        k_per_query: int,
    ) -> list[dict[str, Any]]:
        """Retrieve skill/ability/style linkage documents."""
        query = self._analysis_query(query_analysis, rewrite_result)
        supplemental_results = self._safe_retrieve(
            lambda: self.retriever.retrieve_supplemental(
                query,
                k=k_per_query,
                doc_type="content_model_linkage",
            )
        )

        if len(supplemental_results) >= k_per_query:
            return self._merge_evidence_groups([supplemental_results], k_per_query)

        fallback_results = self._retrieve_evidence_from_rewrite(rewrite_result, k_per_query)
        return self._merge_evidence_groups(
            [supplemental_results, fallback_results],
            k_per_query,
        )

    def _resolve_occupation(
        self,
        query: str,
    ) -> tuple[list[dict[str, Any]], str | None, str | None]:
        """Use supplemental alias documents to infer an official occupation."""
        alias_results = self._safe_retrieve(
            lambda: self.retriever.find_occupation_alias(query, k=5)
        )
        if not alias_results:
            return [], None, None

        top_result = alias_results[0]
        metadata = top_result.get("metadata") or {}
        occupation_code = metadata.get("onet_soc_code")
        occupation_title = metadata.get("occupation_title")
        if isinstance(occupation_code, str):
            occupation_code = occupation_code.strip() or None
        else:
            occupation_code = None
        if isinstance(occupation_title, str):
            occupation_title = occupation_title.strip() or None
        else:
            occupation_title = None

        return alias_results, occupation_code, occupation_title

    def _normal_occupation_evidence(
        self,
        query: str,
        occupation_code: str,
        k: int,
    ) -> list[dict[str, Any]]:
        """Retrieve normal section/full evidence for a known occupation code."""
        return self._safe_retrieve(
            lambda: self.retriever.retrieve_for_occupation(
                query=query,
                onet_soc_code=occupation_code,
                k=k,
            )
        )

    def _safe_retrieve(
        self,
        retrieve_func: Any,
    ) -> list[dict[str, Any]]:
        """Run a retrieval call and treat failures as empty results."""
        try:
            results = retrieve_func()
        except Exception:
            return []
        return results or []

    def _analysis_query(
        self,
        query_analysis: dict[str, Any],
        rewrite_result: dict[str, Any],
    ) -> str:
        """Return the best single query string for targeted supplemental calls."""
        queries = query_analysis.get("search_queries") or []
        for query in queries:
            if isinstance(query, str) and query.strip():
                return query.strip()
        return str(rewrite_result.get("original_question") or "").strip()

    def _merge_evidence_groups(
        self,
        evidence_groups: list[list[dict[str, Any]]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Merge evidence groups in priority order without duplicate documents."""
        merged: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()

        for group in evidence_groups:
            for result in group:
                evidence_item = self._clean_evidence_item(result)
                result_key = (
                    str(evidence_item.get("collection") or ""),
                    str(evidence_item.get("id") or ""),
                )
                if result_key in seen_keys:
                    continue
                seen_keys.add(result_key)
                merged.append(evidence_item)
                if len(merged) >= limit:
                    return merged

        return merged

    def _retrieve_evidence_from_rewrite(
        self,
        rewrite_result: dict[str, Any],
        k_per_query: int,
    ) -> list[dict[str, Any]]:
        """Run retrieval for the original question and rewritten queries."""
        original_question = str(rewrite_result.get("original_question") or "").strip()
        search_queries = rewrite_result.get("search_queries") or []
        queries = self._dedupe_queries([original_question, *search_queries])

        best_by_id: dict[str, dict[str, Any]] = {}

        for query in queries:
            try:
                results = self.retriever.retrieve_smart(query, k=k_per_query)
            except Exception:
                continue

            for result in results:
                evidence_item = self._clean_evidence_item(result)
                doc_id = evidence_item["id"]
                current_best = best_by_id.get(doc_id)
                if current_best is None or evidence_item["score"] > current_best["score"]:
                    best_by_id[doc_id] = evidence_item

        ranked = list(best_by_id.values())
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        response_format_json: bool = False,
    ) -> str:
        """Call the OpenAI chat completions API and return message content."""
        if self.client is None:
            raise RuntimeError(self.client_setup_warning or "OpenAI client is not available.")

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned an empty response.")
        return content.strip()

    def _fallback_rewrite(self, user_question: str) -> dict[str, Any]:
        """Build a safe rewrite result when LLM rewriting is unavailable."""
        return {
            "original_question": user_question,
            "search_queries": [user_question],
            "detected_intent": self._map_retriever_intent(detect_query_type(user_question)),
            "occupation_hint": None,
        }

    def _normalize_rewrite_result(
        self,
        parsed: dict[str, Any],
        user_question: str,
    ) -> dict[str, Any]:
        """Validate and clean the LLM rewrite JSON."""
        original_question = str(parsed.get("original_question") or user_question)
        raw_queries = parsed.get("search_queries") or [user_question]

        search_queries: list[str] = []
        if isinstance(raw_queries, list):
            for query in raw_queries:
                if isinstance(query, str) and query.strip():
                    search_queries.append(query.strip())

        search_queries = self._dedupe_queries(search_queries)[:4]
        if not search_queries:
            search_queries = [user_question]

        detected_intent = str(parsed.get("detected_intent") or "general").strip()
        allowed_intents = {
            "skills",
            "software",
            "day_in_life",
            "education",
            "career_match",
            "general",
        }
        if detected_intent not in allowed_intents:
            detected_intent = self._map_retriever_intent(detect_query_type(user_question))

        occupation_hint = parsed.get("occupation_hint")
        if isinstance(occupation_hint, str):
            occupation_hint = occupation_hint.strip() or None
        elif occupation_hint is not None:
            occupation_hint = None

        return {
            "original_question": original_question,
            "search_queries": search_queries,
            "detected_intent": detected_intent,
            "occupation_hint": occupation_hint,
        }

    def _clean_evidence_item(self, result: dict[str, Any]) -> dict[str, Any]:
        """Keep a stable evidence shape for answer generation and callers."""
        metadata = dict(result.get("metadata") or {})
        doc_id = str(result.get("id") or "")
        if not doc_id:
            doc_id = f"{result.get('collection', 'unknown')}:{hash(result.get('text', ''))}"

        return {
            "id": doc_id,
            "text": str(result.get("text") or ""),
            "score": float(result.get("score") or 0.0),
            "distance": float(result.get("distance") or 0.0),
            "collection": result.get("collection") or "unknown",
            "metadata": {
                **metadata,
                "occupation_title": metadata.get("occupation_title"),
                "onet_soc_code": metadata.get("onet_soc_code"),
                "section": metadata.get("section"),
                "doc_type": metadata.get("doc_type"),
            },
        }

    def _build_evidence_summary(
        self,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build compact source metadata for display or JSON output."""
        summaries: list[dict[str, Any]] = []

        for index, item in enumerate(evidence, 1):
            metadata = item.get("metadata") or {}
            doc_type = self._display_doc_type(item, metadata)
            summaries.append(
                {
                    "rank": index,
                    "occupation_title": metadata.get("occupation_title"),
                    "onet_soc_code": metadata.get("onet_soc_code"),
                    "section": metadata.get("section"),
                    "doc_type": doc_type,
                    "score": item.get("score"),
                    "distance": item.get("distance"),
                    "collection": item.get("collection"),
                    "id": item.get("id"),
                }
            )

        return summaries

    @staticmethod
    def _display_doc_type(
        item: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str | None:
        """Show supplemental doc_type only; old source_type values stay separate."""
        doc_type = metadata.get("doc_type")
        if item.get("collection") == "onet_supplemental":
            return doc_type
        if doc_type in {"onet_section", "onet_full_occupation"}:
            return None
        return doc_type

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        """Parse a JSON object, tolerating accidental Markdown code fences."""
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in response.")

        parsed = json.loads(content[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object.")
        return parsed

    @staticmethod
    def _dedupe_queries(queries: list[Any]) -> list[str]:
        """Deduplicate non-empty queries while preserving order."""
        seen: set[str] = set()
        deduped: list[str] = []

        for query in queries:
            if not isinstance(query, str):
                continue
            cleaned = query.strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                deduped.append(cleaned)

        return deduped

    @staticmethod
    def _map_retriever_intent(intent: str) -> str:
        """Map retriever-specific intents to generator rewrite intents."""
        if intent == "technology_career":
            return "career_match"
        if intent in {"tasks", "interests"}:
            return "general"
        allowed = {"skills", "software", "day_in_life", "education"}
        return intent if intent in allowed else "general"

    @staticmethod
    def _load_dotenv_if_available() -> None:
        """Load .env variables from the project root when python-dotenv is installed."""
        try:
            from dotenv import load_dotenv
        except ImportError:
            return

        load_dotenv(dotenv_path=ENV_PATH)

    @staticmethod
    def _validate_question(user_question: str) -> None:
        """Raise ValueError for empty user questions."""
        if not user_question or not user_question.strip():
            raise ValueError("Query cannot be empty.")

    @staticmethod
    def _validate_k(k: int) -> None:
        """Raise ValueError for invalid result counts."""
        if k <= 0:
            raise ValueError("k must be greater than 0.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the single-query CLI."""
    parser = argparse.ArgumentParser(
        description="Ask one career guidance question using O*NET RAG.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Career question. Quoting is recommended for multi-word questions.",
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Career question. Overrides the positional question if provided.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of retrieved evidence chunks to use.",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable LLM query rewriting before retrieval.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved source summaries after the answer.",
    )
    parser.add_argument(
        "--use-research",
        action="store_true",
        help="Force AI-impact research retrieval even if the detector does not trigger.",
    )
    parser.add_argument(
        "--research-top-k",
        type=int,
        default=DEFAULT_RESEARCH_TOP_K,
        help="Number of research claims to retrieve for AI-impact questions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result dictionary as JSON.",
    )
    return parser.parse_args()


def print_answer(result: dict[str, Any], show_sources: bool = False) -> None:
    """Print the default CLI answer format."""
    print("QUESTION:")
    print(result["question"])
    print("\nANSWER:")
    print(result["answer"])

    if show_sources:
        print("\n=== O*NET SOURCES ===")
        print("collection | occupation title | SOC code | section | score")
        for source in result.get("evidence_summary", []):
            print(
                f"{source.get('collection') or 'N/A'} | "
                f"{source.get('occupation_title') or 'N/A'} | "
                f"{source.get('onet_soc_code') or 'N/A'} | "
                f"{source.get('section') or 'N/A'} | "
                f"{float(source.get('score') or 0.0):.4f}"
            )

        ai_sources = result.get("ai_impact_summary") or []
        if ai_sources:
            print("\n=== AI IMPACT EVIDENCE ===")
            print(
                "collection | source | release/file | page | doc_type | "
                "occupation | SOC | task | impact_type | metric | value | unit | "
                "confidence | score"
            )
            for source in ai_sources:
                release_or_file = source.get("source_release") or source.get("source_file") or "N/A"
                metric_value = (
                    source.get("metric_value")
                    if source.get("metric_value") not in (None, "")
                    else "N/A"
                )
                print(
                    f"{source.get('collection') or 'N/A'} | "
                    f"{source.get('source_name') or 'N/A'} | "
                    f"{release_or_file} | "
                    f"{source.get('page') or 'N/A'} | "
                    f"{source.get('doc_type') or 'N/A'} | "
                    f"{source.get('occupation_title') or 'N/A'} | "
                    f"{source.get('soc_code') or 'N/A'} | "
                    f"{source.get('task_text') or 'N/A'} | "
                    f"{source.get('impact_type') or 'N/A'} | "
                    f"{source.get('metric_name') or 'N/A'} | "
                    f"{metric_value} | "
                    f"{source.get('metric_unit') or 'N/A'} | "
                    f"{source.get('confidence') or 'N/A'} | "
                    f"{float(source.get('score') or 0.0):.4f}"
                )

        inference_sources = result.get("research_inference_summary") or []
        if inference_sources:
            print("\n=== RESEARCH INFERENCE ===")
            print("collection | source | page | allowed_usage | score")
            for source in inference_sources:
                print(
                    f"{source.get('collection') or 'N/A'} | "
                    f"{source.get('source_name') or 'N/A'} | "
                    f"{source.get('page') or 'N/A'} | "
                    f"{source.get('allowed_usage') or 'N/A'} | "
                    f"{float(source.get('score') or 0.0):.4f}"
                )

        research_sources = result.get("research_summary") or []
        if research_sources:
            print("\n=== LEGACY RESEARCH CLAIM SOURCES ===")
            print(
                "collection | claim_id | title | year | pages | "
                "impact_type_clean | impact_direction_clean | ai_relevance | "
                "generator_use_scope | score"
            )
            for source in research_sources:
                print(
                    f"{source.get('collection') or 'N/A'} | "
                    f"{source.get('claim_id') or 'N/A'} | "
                    f"{source.get('title') or 'N/A'} | "
                    f"{source.get('year') or 'N/A'} | "
                    f"{source.get('pages') or 'N/A'} | "
                    f"{source.get('impact_type_clean') or 'N/A'} | "
                    f"{source.get('impact_direction_clean') or 'N/A'} | "
                    f"{source.get('ai_relevance') or 'N/A'} | "
                    f"{source.get('generator_use_scope') or 'N/A'} | "
                    f"{float(source.get('score') or 0.0):.4f}"
                )


def _get_query_from_args(args: argparse.Namespace) -> str:
    """Read the query from --query, positional args, or interactive input."""
    if args.query is not None:
        query = args.query.strip()
    elif args.question:
        query = " ".join(args.question).strip()
    else:
        query = input("Enter your career question: ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")
    return query


def _is_quota_error(exc: Exception) -> bool:
    """Return True when an OpenAI exception looks like quota or billing trouble."""
    text = str(exc).lower()
    quota_markers = (
        "insufficient_quota",
        "insufficient quota",
        "quota",
        "billing",
        "usage limit",
        "usage limits",
    )
    return any(marker in text for marker in quota_markers)


def main() -> int:
    """Run the single-query command-line interface."""
    args = parse_args()

    try:
        query = _get_query_from_args(args)
        generator = CareerRAGGenerator(use_query_rewriting=not args.no_rewrite)
        result = generator.generate_answer(
            query,
            k=args.k,
            use_research=args.use_research,
            research_top_k=args.research_top_k,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if MISSING_API_KEY_MESSAGE in message:
            print(MISSING_API_KEY_MESSAGE, file=sys.stderr)
        else:
            print(message, file=sys.stderr)
        return 1
    except Exception as exc:
        if _is_quota_error(exc):
            print(QUOTA_MESSAGE, file=sys.stderr)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_answer(result, show_sources=args.show_sources)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
