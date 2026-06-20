#!/usr/bin/env python3
"""Generator layer and simple CLI for the O*NET career guidance RAG system.

Flow:
user question -> optional query rewrite -> retrieve O*NET evidence -> generate answer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
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


@dataclass
class EvidenceChunk:
    """Normalized evidence metadata used by AI-impact synthesis and source display."""

    collection: str
    source_type: str
    source_name: str
    title: str = ""
    authors: str = ""
    year: str = ""
    url: str = ""
    page: str = ""
    section: str = ""
    occupation_title: str = ""
    soc_code: str = ""
    task_text: str = ""
    score: float = 0.0
    doc_id: str = ""
    text: str = ""
    role: str = ""
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskAIExposure:
    """Task-level AI exposure interpretation separated by evidence role."""

    task_text: str
    direct_anthropic_evidence: str
    inferred_applicability: str
    confidence: str
    main_reason: str
    source_refs: list[str] = field(default_factory=list)


@dataclass
class AIImpactEvidencePack:
    """All evidence used for one AI-impact answer."""

    occupation_context: dict[str, Any]
    search_queries: list[str]
    chunks: list[EvidenceChunk]
    task_exposures: list[TaskAIExposure]
    ai_impact_evidence: list[dict[str, Any]] = field(default_factory=list)
    research_inference: list[dict[str, Any]] = field(default_factory=list)
    research_claims: list[dict[str, Any]] = field(default_factory=list)


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

    def format_ai_evidence_pack_context(self, pack: AIImpactEvidencePack) -> str:
        """Format the unified AI-impact evidence pack for model prompts."""
        context_parts: list[str] = []
        occupation_title = self._one_line(
            pack.occupation_context.get("occupation_title")
        ) or "N/A"
        soc_code = self._one_line(pack.occupation_context.get("soc_code")) or "N/A"
        context_parts.append(
            "\n".join(
                [
                    "[AI Evidence Pack]",
                    f"Occupation: {occupation_title}",
                    f"SOC: {soc_code}",
                    f"Search queries: {' | '.join(pack.search_queries[:8])}",
                ]
            )
        )

        if pack.task_exposures:
            rows = [
                "Task | Direct Anthropic evidence | Inferred applicability | Confidence | Main reason",
                "--- | --- | --- | --- | ---",
            ]
            for exposure in pack.task_exposures:
                rows.append(
                    " | ".join(
                        [
                            self._one_line(exposure.task_text),
                            self._one_line(exposure.direct_anthropic_evidence),
                            self._one_line(exposure.inferred_applicability),
                            self._one_line(exposure.confidence),
                            self._one_line(exposure.main_reason),
                        ]
                    )
                )
            context_parts.append("[Task AI Exposure Table Input]\n" + "\n".join(rows))

        for index, chunk in enumerate(pack.chunks, 1):
            text = self._one_line(chunk.text)
            if chunk.source_type == "research_inference":
                text = self._redact_numbers_for_inference(text)
            context_parts.append(
                "\n".join(
                    [
                        f"[Evidence Chunk E{index}]",
                        f"Role: {chunk.role or 'supporting_evidence'}",
                        f"Collection: {chunk.collection or 'N/A'}",
                        f"Source type: {chunk.source_type or 'N/A'}",
                        f"Source: {chunk.source_name or chunk.source_id or 'N/A'}",
                        f"Title: {chunk.title or 'N/A'}",
                        f"Page: {chunk.page or 'N/A'}",
                        f"Section: {chunk.section or 'N/A'}",
                        f"Occupation: {chunk.occupation_title or 'N/A'}",
                        f"SOC: {chunk.soc_code or 'N/A'}",
                        f"Task: {chunk.task_text or 'N/A'}",
                        f"Score: {chunk.score:.4f}",
                        "Numeric-use rule: research_inference chunks are qualitative only."
                        if chunk.source_type == "research_inference"
                        else "Numeric-use rule: use numbers only when present in structured metadata.",
                        f"Text: {text}",
                    ]
                )
            )

        return "\n\n".join(context_parts) if context_parts else "No AI-impact evidence pack was found."

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
        fallback_task_texts: list[str] = []
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
                section = self._one_line(metadata.get("section") or metadata.get("doc_type")).lower()
                collection = self._one_line(item.get("collection")).lower()
                extracted_tasks = self._extract_task_lines(item.get("text"))
                if "task" in section and "occupation title" not in section:
                    task_texts.extend(extracted_tasks)
                elif collection == "onet_full_occupations":
                    fallback_task_texts.extend(extracted_tasks)

        occupation_title = first_title or (max(title_counts, key=title_counts.get) if title_counts else None)
        soc_code = first_soc or (max(soc_counts, key=soc_counts.get) if soc_counts else None)
        return {
            "occupation_title": occupation_title,
            "soc_code": soc_code,
            "task_texts": self._dedupe_queries(task_texts or fallback_task_texts)[:12],
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

    def retrieve_ai_impact_evidence_pack(
        self,
        user_question: str,
        onet_evidence: list[dict[str, Any]],
        top_k: int = DEFAULT_RESEARCH_TOP_K,
    ) -> AIImpactEvidencePack:
        """Retrieve, normalize, and deduplicate all AI-impact evidence roles."""
        occupation_context = self._infer_occupation_context(onet_evidence)
        queries = self._build_ai_impact_search_queries(
            user_question=user_question,
            onet_evidence=onet_evidence,
            occupation_context=occupation_context,
        )

        ai_rows: list[dict[str, Any]] = []
        inference_rows: list[dict[str, Any]] = []
        research_claims: list[dict[str, Any]] = []
        ai_query_k = max(4, min(top_k, 8))
        inference_query_k = max(2, min(DEFAULT_RESEARCH_INFERENCE_TOP_K, 4))
        research_query_k = max(3, min(top_k, 6))

        for query in queries:
            try:
                ai_rows.extend(
                    retrieve_ai_impact(
                        query,
                        soc_code=occupation_context.get("soc_code"),
                        occupation_title=occupation_context.get("occupation_title"),
                        task_texts=occupation_context.get("task_texts") or [],
                        top_k=ai_query_k,
                    )
                )
            except Exception:
                pass

            try:
                inference_rows.extend(
                    retrieve_research_inference(
                        query,
                        top_k=inference_query_k,
                    )
                )
            except Exception:
                pass

            try:
                research_claims.extend(
                    retrieve_research_claims(
                        query,
                        top_k=research_query_k,
                    )
                )
            except Exception:
                pass

        ai_rows = self._dedupe_ai_result_rows(ai_rows, limit=max(top_k * 3, 18))
        inference_rows = self._dedupe_ai_result_rows(
            inference_rows,
            limit=max(DEFAULT_RESEARCH_INFERENCE_TOP_K * 3, 8),
        )
        research_claims = self._dedupe_research_claim_rows(
            research_claims,
            limit=max(top_k * 2, 12),
        )

        chunks: list[EvidenceChunk] = []
        chunks.extend(
            self._chunk_from_onet_evidence(item)
            for item in onet_evidence[: max(top_k, 8)]
        )
        chunks.extend(self._chunk_from_ai_result(item) for item in ai_rows)
        chunks.extend(
            self._chunk_from_research_inference(item) for item in inference_rows
        )
        chunks.extend(self._chunk_from_research_claim(item) for item in research_claims)
        chunks = self._dedupe_evidence_chunks(chunks, max_total=40)

        task_exposures = self._build_task_ai_exposures(
            occupation_context=occupation_context,
            ai_rows=ai_rows,
            chunks=chunks,
        )

        return AIImpactEvidencePack(
            occupation_context=occupation_context,
            search_queries=queries,
            chunks=chunks,
            task_exposures=task_exposures,
            ai_impact_evidence=ai_rows,
            research_inference=inference_rows,
            research_claims=research_claims,
        )

    def _build_ai_impact_search_queries(
        self,
        user_question: str,
        onet_evidence: list[dict[str, Any]],
        occupation_context: dict[str, Any],
    ) -> list[str]:
        """Create occupation-, task-, and AI-concept-aware retrieval queries."""
        occupation_title = self._one_line(occupation_context.get("occupation_title"))
        soc_code = self._one_line(occupation_context.get("soc_code"))
        tasks = occupation_context.get("task_texts") or []
        section_terms: list[str] = []
        related_titles: list[str] = []

        for item in onet_evidence[:10]:
            metadata = item.get("metadata") or {}
            section = self._one_line(metadata.get("section") or metadata.get("doc_type"))
            if section:
                section_terms.append(section)
            for key in (
                "related_occupation_title",
                "related_title",
                "alternate_title",
                "occupation_title",
            ):
                value = self._one_line(metadata.get(key))
                if value:
                    related_titles.append(value)

        base_title = occupation_title or user_question
        queries = [
            user_question,
            self._build_expanded_research_query(user_question, onet_evidence),
            f"AI impact on {base_title}",
            f"generative AI automation augmentation exposure for {base_title}",
            f"AI task transformation skills and work activities for {base_title}",
            f"labor market AI exposure {base_title} {soc_code}".strip(),
        ]
        if related_titles:
            queries.append(
                "AI impact on related occupation titles: "
                + "; ".join(self._dedupe_queries(related_titles)[:5])
            )
        if section_terms:
            queries.append(
                f"{base_title} AI exposure using O*NET sections "
                + "; ".join(self._dedupe_queries(section_terms)[:6])
            )
        for task in tasks[:6]:
            queries.append(f"{base_title} AI exposure for task: {task}")

        return self._dedupe_queries(queries)[:14]

    def _dedupe_ai_result_rows(
        self,
        rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Deduplicate Chroma-style rows by document and normalized claim fields."""
        best_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in rows:
            metadata = row.get("metadata") or {}
            key = (
                self._normal_text_key(row.get("collection")),
                self._normal_text_key(row.get("id") or row.get("doc_id")),
            )
            if not key[1]:
                key = (
                    self._normal_text_key(metadata.get("source_id")),
                    self._normal_text_key(metadata.get("source_file") or metadata.get("source_release")),
                    self._normal_text_key(metadata.get("soc_code")),
                    self._normal_text_key(metadata.get("task_text")),
                    self._normal_text_key(metadata.get("impact_type")),
                    self._normal_text_key(metadata.get("metric_name")),
                    self._normal_text_key(metadata.get("metric_value")),
                )
            current = best_by_key.get(key)
            if current is None or self._row_score(row) > self._row_score(current):
                best_by_key[key] = row

        ranked = list(best_by_key.values())
        ranked.sort(key=self._row_score, reverse=True)
        return ranked[:limit]

    def _dedupe_research_claim_rows(
        self,
        rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Deduplicate research claims by claim id and source/page/text."""
        best_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in rows:
            key = (
                self._normal_text_key(row.get("collection") or RESEARCH_COLLECTION_NAME),
                self._normal_text_key(row.get("claim_id") or row.get("id")),
            )
            if not key[1]:
                key = (
                    self._normal_text_key(row.get("source_id") or row.get("title")),
                    self._normal_text_key(row.get("page_start")),
                    self._normal_text_key(row.get("claim_text"))[:120],
                )
            current = best_by_key.get(key)
            if current is None or self._row_score(row) > self._row_score(current):
                best_by_key[key] = row

        ranked = list(best_by_key.values())
        ranked.sort(key=self._row_score, reverse=True)
        return ranked[:limit]

    def _chunk_from_onet_evidence(self, item: dict[str, Any]) -> EvidenceChunk:
        """Normalize an O*NET result into the unified source shape."""
        metadata = item.get("metadata") or {}
        occupation_title = self._one_line(metadata.get("occupation_title"))
        return EvidenceChunk(
            collection=self._one_line(item.get("collection")) or "onet",
            source_type="onet",
            source_name="O*NET",
            title="O*NET Online",
            url="https://www.onetonline.org/",
            section=self._one_line(metadata.get("section") or metadata.get("doc_type")),
            occupation_title=occupation_title,
            soc_code=self._one_line(metadata.get("onet_soc_code") or metadata.get("soc_code")),
            task_text="",
            score=float(item.get("score") or 0.0),
            doc_id=self._one_line(item.get("id")),
            text=self._one_line(item.get("text")),
            role="occupation_grounding",
            source_id="onet",
            metadata=dict(metadata),
        )

    def _chunk_from_ai_result(self, item: dict[str, Any]) -> EvidenceChunk:
        """Normalize a structured AI-impact result into the unified source shape."""
        metadata = item.get("metadata") or {}
        source_id = self._one_line(metadata.get("source_id"))
        doc_type = self._one_line(metadata.get("doc_type"))
        source_type = self._source_type_for_ai_metadata(metadata)
        role = "structured_ai_metric"
        if source_id == "anthropic_economic_index" and metadata.get("task_text"):
            role = "direct_anthropic_task_evidence"
        elif source_id == "nber_w31222":
            role = "nber_structured_evidence"
        elif doc_type in {"ai_methodology", "ai_core_supplemental_exposure"}:
            role = "methodology"

        metric_name = self._one_line(metadata.get("metric_name"))
        metric_value = self._one_line(metadata.get("metric_value"))
        metric_text = ""
        if metric_name or metric_value:
            metric_text = f" Metric: {metric_name} = {metric_value} {self._metric_unit_for_display(metadata)}."

        return EvidenceChunk(
            collection=self._one_line(item.get("collection")) or AI_IMPACT_COLLECTION_NAME,
            source_type=source_type,
            source_name=self._one_line(metadata.get("source_name") or source_id),
            title=self._one_line(metadata.get("source_file") or metadata.get("source_release") or metadata.get("source_name")),
            url=self._one_line(metadata.get("source_url") or metadata.get("url") or metadata.get("final_url")),
            page=self._one_line(metadata.get("source_page") or metadata.get("page")),
            section=doc_type,
            occupation_title=self._one_line(metadata.get("occupation_title")),
            soc_code=self._one_line(metadata.get("soc_code")),
            task_text=self._one_line(metadata.get("task_text")),
            score=float(item.get("reranked_score") or item.get("score") or 0.0),
            doc_id=self._one_line(item.get("doc_id") or item.get("id")),
            text=(self._one_line(item.get("text")) + metric_text).strip(),
            role=role,
            source_id=source_id,
            metadata=dict(metadata),
        )

    def _chunk_from_research_inference(self, item: dict[str, Any]) -> EvidenceChunk:
        """Normalize a qualitative research inference chunk."""
        metadata = item.get("metadata") or {}
        return EvidenceChunk(
            collection=self._one_line(item.get("collection")) or RESEARCH_INFERENCE_COLLECTION_NAME,
            source_type="research_inference",
            source_name=self._one_line(metadata.get("source_name") or metadata.get("source_id")),
            title=self._one_line(metadata.get("source_name") or metadata.get("source_file")),
            url=self._one_line(metadata.get("source_url") or metadata.get("url") or metadata.get("final_url")),
            page=self._one_line(metadata.get("source_page") or metadata.get("page")),
            section=self._one_line(metadata.get("allowed_usage") or metadata.get("doc_type")),
            occupation_title=self._one_line(metadata.get("occupation_title")),
            soc_code=self._one_line(metadata.get("soc_code")),
            task_text=self._one_line(metadata.get("task_text")),
            score=float(item.get("score") or 0.0),
            doc_id=self._one_line(item.get("doc_id") or item.get("id")),
            text=self._one_line(item.get("text")),
            role="qualitative_methodology_or_caveat",
            source_id=self._one_line(metadata.get("source_id")),
            metadata=dict(metadata),
        )

    def _chunk_from_research_claim(self, claim: dict[str, Any]) -> EvidenceChunk:
        """Normalize an uploaded research-claim result."""
        text_parts = [
            self._one_line(claim.get("claim_text")),
            self._one_line(claim.get("evidence_quote")),
        ]
        return EvidenceChunk(
            collection=self._one_line(claim.get("collection")) or RESEARCH_COLLECTION_NAME,
            source_type="uploaded_research_claim",
            source_name=self._one_line(claim.get("title") or claim.get("source_id")),
            title=self._one_line(claim.get("title")),
            authors=self._clean_authors_for_citation(claim.get("authors")),
            year=self._one_line(claim.get("year")),
            url=self._one_line(claim.get("final_url") or claim.get("url")),
            page=self._page_range(claim),
            section=self._one_line(claim.get("impact_type_clean") or claim.get("generator_use_scope")),
            occupation_title=self._one_line(claim.get("affected_entity_text")),
            soc_code="",
            task_text="",
            score=float(claim.get("score") or 0.0),
            doc_id=self._one_line(claim.get("claim_id") or claim.get("id")),
            text=". ".join(part for part in text_parts if part),
            role="uploaded_research_claim",
            source_id=self._one_line(claim.get("source_id")),
            metadata=dict(claim),
        )

    def _dedupe_evidence_chunks(
        self,
        chunks: list[EvidenceChunk],
        max_total: int,
    ) -> list[EvidenceChunk]:
        """Deduplicate normalized chunks and keep a source-diverse set."""
        best_by_key: dict[tuple[str, ...], EvidenceChunk] = {}
        for chunk in chunks:
            text_key = self._normal_text_key(chunk.text)
            text_hash = hashlib.sha1(text_key[:500].encode("utf-8")).hexdigest()[:12]
            key = (
                self._normal_text_key(chunk.collection),
                self._normal_text_key(chunk.doc_id),
            )
            if not key[1]:
                key = (
                    self._normal_text_key(chunk.source_id or chunk.source_name),
                    self._normal_text_key(chunk.title),
                    self._normal_text_key(chunk.page),
                    self._normal_text_key(chunk.section),
                    self._normal_text_key(chunk.occupation_title),
                    self._normal_text_key(chunk.task_text),
                    text_hash,
                )
            current = best_by_key.get(key)
            if current is None or chunk.score > current.score:
                best_by_key[key] = chunk

        role_priority = {
            "occupation_grounding": 5,
            "direct_anthropic_task_evidence": 4,
            "structured_ai_metric": 3,
            "uploaded_research_claim": 2,
            "qualitative_methodology_or_caveat": 1,
        }
        ranked = list(best_by_key.values())
        ranked.sort(
            key=lambda chunk: (
                role_priority.get(chunk.role, 0),
                chunk.score,
            ),
            reverse=True,
        )

        selected: list[EvidenceChunk] = []
        source_counts: dict[str, int] = {}
        for chunk in ranked:
            source_key = self._normal_text_key(
                chunk.source_id or chunk.source_name or chunk.collection
            )
            max_per_source = 8 if chunk.source_type in {"onet", "structured_ai_dataset"} else 5
            if source_counts.get(source_key, 0) >= max_per_source:
                continue
            selected.append(chunk)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            if len(selected) >= max_total:
                break
        return selected

    def _build_task_ai_exposures(
        self,
        occupation_context: dict[str, Any],
        ai_rows: list[dict[str, Any]],
        chunks: list[EvidenceChunk],
    ) -> list[TaskAIExposure]:
        """Build conservative task-level AI exposure interpretations."""
        tasks = list(occupation_context.get("task_texts") or [])
        if not tasks:
            for row in ai_rows:
                task = self._one_line((row.get("metadata") or {}).get("task_text"))
                if task:
                    tasks.append(task)
        tasks = self._dedupe_queries(tasks)[:7]

        exposures: list[TaskAIExposure] = []
        for task in tasks:
            direct_rows = self._matching_anthropic_task_rows(
                task=task,
                ai_rows=ai_rows,
                occupation_context=occupation_context,
            )
            direct_text = self._direct_anthropic_evidence_label(direct_rows)
            applicability, confidence, reason = self._infer_task_ai_applicability(
                task=task,
                direct_rows=direct_rows,
                chunks=chunks,
            )
            exposures.append(
                TaskAIExposure(
                    task_text=task,
                    direct_anthropic_evidence=direct_text,
                    inferred_applicability=applicability,
                    confidence=confidence,
                    main_reason=reason,
                    source_refs=self._supporting_source_refs(task, chunks),
                )
            )

        return exposures

    def _matching_anthropic_task_rows(
        self,
        task: str,
        ai_rows: list[dict[str, Any]],
        occupation_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Find Anthropic rows that map to the exact or near O*NET task."""
        expected_soc = self._normal_text_key(occupation_context.get("soc_code"))
        expected_title = self._normal_text_key(occupation_context.get("occupation_title"))
        matches: list[dict[str, Any]] = []
        for row in ai_rows:
            metadata = row.get("metadata") or {}
            if self._one_line(metadata.get("source_id")) != "anthropic_economic_index":
                continue
            row_task = self._one_line(metadata.get("task_text"))
            if not row_task:
                continue
            actual_soc = self._normal_text_key(metadata.get("soc_code"))
            actual_title = self._normal_text_key(metadata.get("occupation_title"))
            soc_or_title_matches = (
                bool(expected_soc and actual_soc == expected_soc)
                or bool(expected_title and actual_title == expected_title)
                or not (expected_soc or expected_title)
            )
            if not soc_or_title_matches:
                continue
            similarity = self._text_similarity(task, row_task)
            if similarity >= 0.58:
                row["_task_similarity"] = similarity
                matches.append(row)
        matches.sort(
            key=lambda row: (
                float(row.get("_task_similarity") or 0.0),
                self._row_score(row),
            ),
            reverse=True,
        )
        return matches[:4]

    def _direct_anthropic_evidence_label(
        self,
        direct_rows: list[dict[str, Any]],
    ) -> str:
        """Describe direct Anthropic evidence without treating zeros as no impact."""
        positive_rows = [
            row for row in direct_rows if self._metric_value_float(row.get("metadata") or {}) > 0
        ]
        if not positive_rows:
            return "No direct observed/mapped Anthropic evidence for this exact O*NET task."

        row = positive_rows[0]
        metadata = row.get("metadata") or {}
        metric_name = self._one_line(metadata.get("metric_name")) or "metric"
        metric_value = self._one_line(metadata.get("metric_value"))
        metric_unit = self._metric_unit_for_display(metadata)
        impact_type = self._one_line(metadata.get("impact_type"))
        source_scope = self._one_line(metadata.get("doc_type")) or "retrieved row"
        return (
            f"Direct Anthropic {impact_type or 'AI'} evidence: "
            f"{metric_name} = {metric_value} {metric_unit} ({source_scope})."
        ).strip()

    def _infer_task_ai_applicability(
        self,
        task: str,
        direct_rows: list[dict[str, Any]],
        chunks: list[EvidenceChunk],
    ) -> tuple[str, str, str]:
        """Infer semantic AI applicability from task text and qualitative evidence."""
        task_lower = task.lower()
        information_terms = {
            "analyze",
            "analysis",
            "assess",
            "calculate",
            "compare",
            "data",
            "database",
            "design",
            "document",
            "draft",
            "estimate",
            "evaluate",
            "forecast",
            "mathematical",
            "model",
            "monitor",
            "numerical",
            "plan",
            "problem",
            "prepare",
            "program",
            "report",
            "research",
            "review",
            "software",
            "statistical",
            "study",
            "summarize",
            "technique",
            "theories",
            "write",
        }
        compliance_terms = {
            "compliance",
            "permit",
            "policy",
            "regulation",
            "regulatory",
            "risk",
            "standards",
        }
        physical_terms = {
            "backfill",
            "clean",
            "compact",
            "construction",
            "equipment",
            "hazard",
            "inspect",
            "inspection",
            "install",
            "maintain",
            "operate",
            "picks",
            "repair",
            "shovels",
            "sample",
            "site",
            "supervise",
            "test",
            "traffic",
            "trenches",
        }
        score = 0
        score += sum(1 for term in information_terms if term in task_lower)
        score += sum(1 for term in compliance_terms if term in task_lower)
        score -= min(sum(1 for term in physical_terms if term in task_lower), 2)
        if any(self._metric_value_float(row.get("metadata") or {}) > 0 for row in direct_rows):
            score += 3
        if self._has_qualitative_research_support(task, chunks):
            score += 1

        if score >= 5:
            applicability = "High"
        elif score >= 3:
            applicability = "Medium"
        elif score >= 1:
            applicability = "Low-Medium"
        else:
            applicability = "Low"

        if any(self._metric_value_float(row.get("metadata") or {}) > 0 for row in direct_rows):
            confidence = "Direct Anthropic signal plus task inference"
        else:
            confidence = "Inferred from task language and qualitative research context"

        return applicability, confidence, self._task_ai_reason(task_lower, applicability)

    def _task_ai_reason(self, task_lower: str, applicability: str) -> str:
        """Return a concise, non-numeric reason for the task exposure table."""
        if any(term in task_lower for term in ("monitor", "inspect", "inspection", "field research", "field inspection", "site", "sample", "test", "equipment", "traffic", "ditch", "trench", "concrete", "pipes", "shovel", "rake", "machinery")):
            return "AI may support planning, monitoring data, and documentation, but site conditions, safety, and hands-on execution remain human-led."
        if any(term in task_lower for term in ("data", "analysis", "model", "statistical", "forecast", "calculate", "mathematical", "computational", "numerical", "theories", "techniques")):
            return "AI can assist with analysis, modeling, pattern finding, and drafting, but domain validation remains human-led."
        if any(term in task_lower for term in ("report", "document", "draft", "write", "prepare", "review", "summarize")):
            return "AI can assist with document review, summarization, and first drafts, while humans remain accountable for final judgment."
        if any(term in task_lower for term in ("compliance", "permit", "regulation", "policy", "standards", "risk")):
            return "AI can help compare rules, flag risks, and draft compliance materials, but legal and professional accountability stays human."
        if applicability in {"High", "Medium"}:
            return "The task has repeatable information-processing components that current AI systems can assist with."
        return "The task appears more context-specific or hands-on, so AI support is more likely to be indirect."

    def _has_qualitative_research_support(
        self,
        task: str,
        chunks: list[EvidenceChunk],
    ) -> bool:
        """Return True when qualitative research chunks overlap with the task."""
        task_key = self._normal_text_key(task)
        if not task_key:
            return False
        task_tokens = set(task_key.split())
        for chunk in chunks:
            if chunk.source_type not in {"uploaded_research_claim", "research_inference"}:
                continue
            chunk_tokens = set(self._normal_text_key(chunk.text).split())
            if len(task_tokens & chunk_tokens) >= 2:
                return True
        return False

    def _supporting_source_refs(
        self,
        task: str,
        chunks: list[EvidenceChunk],
    ) -> list[str]:
        """Pick compact source references for a task exposure row."""
        refs: list[str] = []
        for chunk in chunks:
            if chunk.source_id == "nber_w31222":
                continue
            similarity = self._text_similarity(task, chunk.task_text or chunk.text)
            if chunk.source_type == "onet" or similarity >= 0.12:
                label = chunk.source_name or chunk.title or chunk.collection
                if label and label not in refs:
                    refs.append(label)
            if len(refs) >= 3:
                break
        return refs

    def _build_structured_ai_impact_answer(
        self,
        user_question: str,
        onet_evidence: list[dict[str, Any]],
        pack: AIImpactEvidencePack,
    ) -> str:
        """Build a grounded AI-impact answer with enforceable cautious wording."""
        del user_question
        occupation_context = pack.occupation_context
        occupation_title = self._one_line(occupation_context.get("occupation_title")) or "this occupation"
        soc_code = self._one_line(occupation_context.get("soc_code"))
        description = self._first_occupation_description(onet_evidence)
        tasks = list(occupation_context.get("task_texts") or [])

        parts = [
            "Short Career Explanation",
            description
            or f"{occupation_title} work on the tasks and responsibilities described in the retrieved O*NET evidence.",
        ]
        if soc_code:
            parts[-1] = f"{parts[-1]} O*NET SOC: {soc_code}."

        parts.extend(["", "Main Tasks"])
        if tasks:
            parts.extend(f"- {task}" for task in tasks[:6])
        else:
            parts.append("- I found O*NET occupation evidence, but not a clean task list in the retrieved chunks.")

        parts.extend(
            [
                "",
                "AI Exposure By Task",
                "| Task | Direct Anthropic Evidence | Inferred AI Applicability | Confidence | Main Reason |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        if pack.task_exposures:
            for exposure in pack.task_exposures:
                parts.append(
                    "| "
                    + " | ".join(
                        [
                            self._markdown_cell(exposure.task_text),
                            self._markdown_cell(exposure.direct_anthropic_evidence),
                            self._markdown_cell(exposure.inferred_applicability),
                            self._markdown_cell(exposure.confidence),
                            self._markdown_cell(exposure.main_reason),
                        ]
                    )
                    + " |"
                )
        else:
            parts.append(
                "| Occupation-level work | No direct observed/mapped Anthropic evidence for this exact O*NET task. | Low-Medium | Inferred from available O*NET and research evidence | Retrieved task evidence was not specific enough for a task-level table. |"
            )

        parts.extend(
            [
                "",
                "AI Impact Synthesis",
                self._ai_impact_synthesis(pack),
                "",
                "Where AI May Help Most",
            ]
        )
        parts.extend(self._where_ai_may_help_lines(pack))
        parts.extend(["", "Where Human Expertise Remains Important"])
        parts.extend(self._where_human_expertise_lines(pack))
        parts.extend(["", "Sources Used"])
        parts.extend(self._format_answer_sources(pack, onet_evidence))
        return "\n".join(parts)

    def _ai_impact_synthesis(self, pack: AIImpactEvidencePack) -> str:
        """Summarize the evidence pack without overclaiming."""
        positive_direct = sum(
            1
            for exposure in pack.task_exposures
            if exposure.direct_anthropic_evidence.startswith("Direct Anthropic")
        )
        no_direct = sum(
            1
            for exposure in pack.task_exposures
            if exposure.direct_anthropic_evidence.startswith("No direct")
        )
        pieces = [
            "The safest reading is task transformation and augmentation, not a claim that the occupation is disappearing.",
            "The Anthropic column reports only direct observed or mapped evidence from the indexed dataset.",
        ]
        if positive_direct:
            pieces.append(
                f"{positive_direct} retrieved task row(s) had a positive direct Anthropic signal."
            )
        if no_direct:
            pieces.append(
                "'No direct observed/mapped Anthropic evidence' means the retrieved Anthropic data did not map an observed signal to that exact O*NET task; it does not mean zero AI impact."
            )
        if any(chunk.source_type in {"uploaded_research_claim", "research_inference"} for chunk in pack.chunks):
            pieces.append(
                "Uploaded research and research-inference chunks are used qualitatively for scope, caveats, and likely task mechanisms, not for new numeric AI-impact claims."
            )
        return " ".join(pieces)

    def _where_ai_may_help_lines(self, pack: AIImpactEvidencePack) -> list[str]:
        """Return task-focused lines for the highest-applicability rows."""
        prioritized = [
            exposure
            for exposure in pack.task_exposures
            if exposure.inferred_applicability in {"High", "Medium"}
        ]
        if not prioritized:
            prioritized = pack.task_exposures[:3]
        lines: list[str] = []
        for exposure in prioritized[:4]:
            lines.append(
                f"- {self._truncate(exposure.task_text, 110)}: {exposure.main_reason}"
            )
        return lines or ["- The retrieved task evidence did not identify a strong task-specific AI assistance area."]

    def _where_human_expertise_lines(self, pack: AIImpactEvidencePack) -> list[str]:
        """Return conservative lines for human-led parts of the work."""
        lower_priority = [
            exposure
            for exposure in pack.task_exposures
            if exposure.inferred_applicability in {"Low", "Low-Medium"}
        ]
        lines: list[str] = []
        for exposure in lower_priority[:3]:
            lines.append(
                f"- {self._truncate(exposure.task_text, 110)}: human oversight matters because the task is context-specific, accountable, or hands-on."
            )
        if not lines:
            lines.append(
                "- Final decisions, professional accountability, stakeholder communication, and safety-sensitive judgments remain human-led."
            )
        return lines

    def _format_answer_sources(
        self,
        pack: AIImpactEvidencePack,
        onet_evidence: list[dict[str, Any]],
    ) -> list[str]:
        """Format only sources actually used by the deterministic AI answer."""
        del onet_evidence
        selected: list[EvidenceChunk] = []
        seen: set[str] = set()
        source_lines: list[str] = []
        source_plan = [
            ("structured_ai_dataset", 2),
            ("uploaded_research_claim", 2),
            ("research_inference", 1),
        ]

        if any(chunk.source_type == "onet" for chunk in pack.chunks):
            source_lines.append("- O*NET Online. https://www.onetonline.org/")

        def add_chunk(chunk: EvidenceChunk) -> None:
            if chunk.source_id == "nber_w31222":
                return
            if chunk.source_type == "onet":
                return
            key = self._source_citation_key(chunk)
            if not key or key in seen:
                return
            seen.add(key)
            selected.append(chunk)

        for source_type, limit in source_plan:
            added = 0
            for chunk in pack.chunks:
                if chunk.source_type != source_type:
                    continue
                before = len(selected)
                add_chunk(chunk)
                if len(selected) > before:
                    added += 1
                if added >= limit:
                    break

        for chunk in pack.chunks:
            if len(selected) >= 5:
                break
            add_chunk(chunk)

        for chunk in selected[:5]:
            citation = self._format_source_citation(chunk)
            if chunk.source_type == "research_inference":
                citation = f"{citation} [qualitative only]"
            source_lines.append(f"- {citation}")

        return source_lines or ["- O*NET Online. https://www.onetonline.org/"]

    def _format_source_citation(self, chunk: EvidenceChunk) -> str:
        """Format one source with paper-style citation metadata when available."""
        if chunk.source_type == "onet":
            return "O*NET Online. https://www.onetonline.org/"

        if chunk.source_type == "structured_ai_dataset":
            parts = [chunk.source_name or "Structured AI dataset"]
            source_title = self._display_source_title(chunk.title)
            if source_title and source_title != parts[0]:
                parts.append(source_title)
            if chunk.url:
                parts.append(chunk.url)
            return self._join_citation_parts(parts)

        title = chunk.title or chunk.source_name or chunk.source_id or chunk.collection
        parts = [title]
        if chunk.authors:
            parts.append(chunk.authors)
        if chunk.year:
            parts.append(chunk.year)
        page = self._page_for_citation(chunk.page)
        if page:
            parts.append(page)
        if chunk.url:
            parts.append(chunk.url)
        elif chunk.source_type == "research_inference" and chunk.metadata.get("source_url"):
            parts.append(self._one_line(chunk.metadata.get("source_url")))
        return self._join_citation_parts(parts)

    def _source_citation_key(self, chunk: EvidenceChunk) -> str:
        """Deduplicate sources by citation identity rather than retrieved chunk id."""
        if chunk.source_type in {
            "uploaded_research_claim",
            "research_inference",
            "structured_research_claim",
            "research_methodology",
        }:
            return self._normal_text_key(
                f"{chunk.source_type} {chunk.title or chunk.source_name} "
                f"{chunk.authors} {chunk.year} {chunk.url or chunk.source_id}"
            )
        return self._normal_text_key(
            f"{chunk.source_type} {chunk.source_id} {chunk.source_name} "
            f"{chunk.title} {chunk.section}"
        )

    @staticmethod
    def _join_citation_parts(parts: list[str]) -> str:
        """Join citation fragments with periods and preserve URL readability."""
        cleaned = [CareerRAGGenerator._one_line(part).rstrip(".") for part in parts if CareerRAGGenerator._one_line(part)]
        if not cleaned:
            return "Unknown source."
        citation = cleaned[0]
        for part in cleaned[1:]:
            separator = " " if citation.endswith(("?", "!")) else ". "
            citation = f"{citation}{separator}{part}"
        if cleaned[-1].startswith(("http://", "https://")) or citation.endswith((".", "?", "!")):
            return citation
        return f"{citation}."

    @staticmethod
    def _clean_authors_for_citation(value: Any) -> str:
        """Keep author metadata citation-like and drop subtitles/abstract snippets."""
        text = CareerRAGGenerator._one_line(value)
        if not text:
            return ""
        segments = [segment.strip() for segment in text.split(";") if segment.strip()]
        if len(segments) > 1:
            first_lower = segments[0].lower()
            if first_lower.startswith(("evidence from", "a review of", "working paper")):
                segments = segments[1:]
            else:
                segments = [segments[0]]
        cleaned = "; ".join(segments)
        noisy = re.search(
            r"\b(this paper|abstract|study|report|evaluates claims|implications)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if noisy and segments:
            cleaned = segments[0]
        if len(cleaned.split()) > 18:
            cleaned = ""
        return cleaned

    @staticmethod
    def _page_for_citation(page: str) -> str:
        """Format page metadata for citation text."""
        cleaned = CareerRAGGenerator._one_line(page)
        if not cleaned or cleaned == "N/A":
            return ""
        if "-" in cleaned:
            return f"pp. {cleaned}"
        return f"p. {cleaned}"

    @staticmethod
    def _display_source_title(title: str) -> str:
        """Prefer readable source names over long local file paths."""
        cleaned = CareerRAGGenerator._one_line(title)
        if not cleaned:
            return ""
        if "\\" in cleaned or "/" in cleaned:
            return Path(cleaned.replace("\\", "/")).name
        return cleaned

    def _build_evidence_pack_summary(
        self,
        chunks: list[EvidenceChunk],
    ) -> list[dict[str, Any]]:
        """Build the unified source table used by --show-sources."""
        summaries: list[dict[str, Any]] = []
        for chunk in chunks:
            summaries.append(
                {
                    "collection": chunk.collection,
                    "source_type": chunk.source_type,
                    "source_name": chunk.source_name,
                    "title": chunk.title,
                    "authors": chunk.authors,
                    "year": chunk.year,
                    "url": chunk.url,
                    "citation": self._format_source_citation(chunk),
                    "page": chunk.page,
                    "section": chunk.section,
                    "occupation": chunk.occupation_title,
                    "task": chunk.task_text,
                    "score": chunk.score,
                    "doc_id": chunk.doc_id,
                    "role": chunk.role,
                }
            )
        return summaries

    @staticmethod
    def _source_type_for_ai_metadata(metadata: dict[str, Any]) -> str:
        """Classify structured AI-impact metadata into source table source_type."""
        source_id = CareerRAGGenerator._one_line(metadata.get("source_id"))
        doc_type = CareerRAGGenerator._one_line(metadata.get("doc_type"))
        if source_id == "anthropic_economic_index":
            return "structured_ai_dataset"
        if source_id == "nber_w31222":
            return "structured_research_claim"
        if doc_type in {"ai_methodology", "ai_core_supplemental_exposure"}:
            return "research_methodology"
        return "structured_ai_evidence"

    @staticmethod
    def _row_score(row: dict[str, Any]) -> float:
        """Return the best available score for a retrieved row."""
        return float(row.get("reranked_score") or row.get("score") or 0.0)

    @staticmethod
    def _metric_value_float(metadata: dict[str, Any]) -> float:
        """Parse a metric value, returning 0.0 when absent or unparsable."""
        value = CareerRAGGenerator._one_line(metadata.get("metric_value"))
        if not value:
            return 0.0
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return 0.0

    @staticmethod
    def _normal_text_key(value: Any) -> str:
        """Normalize text for lightweight dedupe and matching."""
        text = CareerRAGGenerator._one_line(value).lower()
        return re.sub(r"[^a-z0-9]+", " ", text).strip()

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        """Compute simple token-overlap similarity."""
        left_tokens = set(CareerRAGGenerator._normal_text_key(left).split())
        right_tokens = set(CareerRAGGenerator._normal_text_key(right).split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)

    @staticmethod
    def _markdown_cell(value: Any) -> str:
        """Escape table cell delimiters and keep cells compact."""
        text = CareerRAGGenerator._one_line(value)
        return text.replace("|", "\\|")

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
                    "3. AI exposure by task as a Markdown table\n"
                    "4. AI impact synthesis\n"
                    "5. Where AI may help most\n"
                    "6. Where human expertise remains important\n"
                    "7. Sources used\n\n"
                    "Rules:\n"
                    "- Use O*NET evidence for what the occupation does and its tasks.\n"
                    "- Use structured AI impact evidence for AI statistics.\n"
                    "- Prefer Anthropic task-level rows for automation, augmentation, task penetration, observed usage, and job exposure.\n"
                    "- Use NBER W31222 for exposure methodology, direct/indirect exposure, and core/supplemental distinctions.\n"
                    "- Research inference chunks are only for caveats or methodology, not numeric statistics.\n"
                    "- Do not fabricate values. Every numeric AI-impact value must appear in the structured AI evidence block.\n"
                    "- Do not convert share/score metrics into percentages unless metric_unit is percent.\n"
                    "- A metric value of 0.0 share means the retrieved indexed dataset did not observe/map a direct signal for that exact task; do not call it no AI impact, no exposure, or no automation unless impact_type is explicitly no_exposure.\n"
                    "- In the task table, write 'No direct observed/mapped Anthropic evidence for this exact O*NET task' instead of showing raw 0.0 share rows.\n"
                    "- Do not say exposure means replacement, disappearance, layoffs, or safety from AI.\n"
                    "- Discuss task change and skill adaptation only. Do not forecast demand, hiring, employment, openings, growth, or decline unless retrieved evidence explicitly says so.\n"
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
            parts.extend(
                [
                    "",
                    "AI exposure by task",
                    "| Task | Direct Anthropic Evidence | Inferred AI Applicability | Confidence | Main Reason |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            metric_lines = self._fallback_ai_metric_lines(ai_impact_evidence)
            if metric_lines:
                parts.extend(metric_lines[:5])
            else:
                parts.append(
                    "| Occupation-level work | No direct observed/mapped Anthropic evidence for this exact O*NET task. | Low-Medium | Inferred from available evidence | I did not find a direct indexed Anthropic task match. |"
                )

            parts.extend(
                [
                    "",
                    "AI impact synthesis",
                    "The retrieved evidence should be read as exposure or observed usage, not as proof that the occupation will be replaced. Coding, modeling, drafting, and repeatable information-processing tasks are usually the parts to inspect first; judgment-heavy communication and accountable decisions need more caution.",
                    "",
                    "Where AI may help most",
                    "- Repeatable information processing, drafting, review, and analysis tasks are the most plausible assistance areas when supported by retrieved evidence.",
                    "",
                    "Where human expertise remains important",
                    "- Final decisions, professional accountability, stakeholder communication, and safety-sensitive judgments remain human-led.",
                    "",
                    "Sources used",
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
                if self._metric_caution(metadata).startswith("0.0 share"):
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                self._markdown_cell(task),
                                "No direct observed/mapped Anthropic evidence for this exact O*NET task.",
                                "Low-Medium",
                                "Inferred from available evidence",
                                "The zero-valued retrieved metric is not interpreted as zero AI impact.",
                            ]
                        )
                        + " |"
                    )
                else:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                self._markdown_cell(task),
                                self._markdown_cell(
                                    f"{source} reports {impact_type} metric {metric_name} = {metric_value} {metric_unit}."
                                ),
                                "Medium",
                                "Direct metric plus inference",
                                "Use the metric only within its source scope.",
                            ]
                        )
                        + " |"
                    )
            elif source:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            self._markdown_cell(task),
                            self._markdown_cell(
                                f"{source} provides {impact_type or 'methodology'} evidence, but no specific numeric value is available in this row."
                            ),
                            "Low-Medium",
                            "Inferred from available evidence",
                            "No direct numeric task metric was available.",
                        ]
                    )
                    + " |"
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
                "no direct observed/mapped Anthropic evidence for this exact O*NET task",
            ),
            (
                r"\bno observed AI penetration\b",
                "no direct observed/mapped Anthropic evidence for this exact O*NET task",
            ),
            (
                r"\bno significant AI penetration\b",
                "no direct observed/mapped Anthropic evidence in the retrieved task row",
            ),
            (
                r"\bno current AI exposure\b",
                "no direct observed/mapped Anthropic evidence for this exact O*NET task",
            ),
            (
                r"\bnot currently exposed to AI\b",
                "not directly observed or mapped in the retrieved Anthropic task evidence",
            ),
            (
                r"\b0\.0 share in the indexed measure\b",
                "no direct observed/mapped Anthropic evidence for this exact O*NET task",
            ),
            (
                r"\b0\.0 share in the indexed exposure measure\b",
                "no direct observed/mapped Anthropic evidence for this exact O*NET task",
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
            "aiimpactsynthesis",
            "futureoutlook",
            "sources",
            "sourcesused",
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
        ai_evidence_pack: AIImpactEvidencePack | None = None

        if ai_impact_needed:
            ai_evidence_pack = self.retrieve_ai_impact_evidence_pack(
                user_question=user_question,
                onet_evidence=evidence,
                top_k=research_top_k,
            )
            ai_impact_evidence = ai_evidence_pack.ai_impact_evidence
            research_inference = ai_evidence_pack.research_inference
            research_claims = ai_evidence_pack.research_claims
            expanded_research_query = " | ".join(ai_evidence_pack.search_queries)
            ai_impact_context = self.format_ai_evidence_pack_context(ai_evidence_pack)
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

        if ai_impact_needed and ai_evidence_pack is not None:
            answer = self._build_structured_ai_impact_answer(
                user_question=user_question,
                onet_evidence=evidence,
                pack=ai_evidence_pack,
            )
        else:
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
            "ai_evidence_pack_summary": self._build_evidence_pack_summary(
                ai_evidence_pack.chunks if ai_evidence_pack is not None else []
            ),
            "ai_task_exposures": [
                {
                    "task_text": exposure.task_text,
                    "direct_anthropic_evidence": exposure.direct_anthropic_evidence,
                    "inferred_applicability": exposure.inferred_applicability,
                    "confidence": exposure.confidence,
                    "main_reason": exposure.main_reason,
                    "source_refs": exposure.source_refs,
                }
                for exposure in (ai_evidence_pack.task_exposures if ai_evidence_pack is not None else [])
            ],
            "ai_evidence_pack_queries": (
                ai_evidence_pack.search_queries if ai_evidence_pack is not None else []
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
        if query_analysis.get("ai_impact_query"):
            grounded_evidence = self._retrieve_ai_occupation_grounding(
                query_analysis=query_analysis,
                rewrite_result=rewrite_result,
                k_per_query=k_per_query,
            )
            if grounded_evidence:
                return grounded_evidence

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

    def _retrieve_ai_occupation_grounding(
        self,
        query_analysis: dict[str, Any],
        rewrite_result: dict[str, Any],
        k_per_query: int,
    ) -> list[dict[str, Any]]:
        """Resolve a named occupation for AI-impact questions before O*NET retrieval."""
        query = self._analysis_query(query_analysis, rewrite_result)
        alias_results, occupation_code, occupation_title = self._resolve_occupation(query)
        if not occupation_code:
            return []

        top_alias_score = float(alias_results[0].get("score") or 0.0) if alias_results else 0.0
        if top_alias_score < 0.56:
            return []

        title_or_code = occupation_title or occupation_code
        evidence_groups = [
            alias_results[:1],
            self._normal_occupation_evidence(
                query=f"{title_or_code} tasks main duties work activities",
                occupation_code=occupation_code,
                k=max(k_per_query, 8),
            ),
            self._normal_occupation_evidence(
                query=f"{title_or_code} occupation overview skills knowledge abilities",
                occupation_code=occupation_code,
                k=min(max(k_per_query, 5), 8),
            ),
        ]
        return self._merge_evidence_groups(evidence_groups, k_per_query)

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
        unified_sources = result.get("ai_evidence_pack_summary") or []
        if unified_sources:
            print("\n=== RETRIEVED SOURCES ===")
            print(
                "collection | source_type | source_name | title | authors | year | "
                "page | url | citation | section | occupation | task | score | doc_id"
            )
            for source in unified_sources:
                task = CareerRAGGenerator._truncate(
                    CareerRAGGenerator._one_line(source.get("task")),
                    120,
                )
                title = CareerRAGGenerator._truncate(
                    CareerRAGGenerator._one_line(source.get("title")),
                    90,
                )
                citation = CareerRAGGenerator._truncate(
                    CareerRAGGenerator._one_line(source.get("citation")),
                    160,
                )
                print(
                    f"{source.get('collection') or 'N/A'} | "
                    f"{source.get('source_type') or 'N/A'} | "
                    f"{source.get('source_name') or 'N/A'} | "
                    f"{title or 'N/A'} | "
                    f"{source.get('authors') or 'N/A'} | "
                    f"{source.get('year') or 'N/A'} | "
                    f"{source.get('page') or 'N/A'} | "
                    f"{source.get('url') or 'N/A'} | "
                    f"{citation or 'N/A'} | "
                    f"{source.get('section') or 'N/A'} | "
                    f"{source.get('occupation') or 'N/A'} | "
                    f"{task or 'N/A'} | "
                    f"{float(source.get('score') or 0.0):.4f} | "
                    f"{source.get('doc_id') or 'N/A'}"
                )
            return

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
