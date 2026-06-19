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
    from career_rag.research_retriever import (
        DEFAULT_COLLECTION as RESEARCH_COLLECTION_NAME,
        format_pages as format_research_pages,
        format_research_source,
        retrieve_research_claims,
    )
except ImportError:  # Allows: py career_rag/generator.py
    from retriever import OnetRetriever, detect_query_type  # type: ignore
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
- Contains AI/labor-market impact claims from papers and reports.
- Use it for AI impact, automation exposure, augmentation, job loss, job creation, skill change, employment effects, wage effects, productivity effects, worker vulnerability, and uncertainty.

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
2. If research evidence is broad, task-level, industry-level, or general labor-market evidence rather than occupation-specific, say so clearly.
3. Do not treat automation exposure as guaranteed job loss.
4. Distinguish automation exposure, augmentation potential, task transformation, job loss, job creation, skill change, and worker vulnerability.
5. Use O*NET evidence for what the occupation involves.
6. Use research evidence for how AI may affect it.
7. If evidence is mixed or uncertain, say so.
8. Cite research sources using title, year, and page metadata.

Useful wording for AI-impact answers:
- The research evidence suggests exposure rather than guaranteed replacement.
- This claim is task-level rather than occupation-specific, so it should be interpreted as an indirect signal.

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
    "future of work",
    "job impact",
    "job loss",
    "job creation",
    "skill change",
    "augmentation",
    "displacement",
    "exposure",
    "replace",
    "replacement",
    "at risk",
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
        if not api_key:
            raise RuntimeError(MISSING_API_KEY_MESSAGE)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI Python package is not installed. Install it with "
                "`pip install openai` before using CareerRAGGenerator."
            ) from exc

        self.client = OpenAI(api_key=api_key)
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
        if not self.use_query_rewriting:
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
        research_needed = bool(use_research or query_analysis.get("ai_impact_query"))
        query_analysis["use_research"] = research_needed
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
        expanded_research_query = ""

        if research_needed:
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

        answer = self._chat_completion(messages=messages, temperature=0.2)

        return {
            "question": user_question,
            "answer": answer,
            "rewritten_queries": rewrite_result.get("search_queries", []),
            "query_analysis": query_analysis,
            "evidence": evidence,
            "evidence_summary": self._build_evidence_summary(evidence),
            "research_claims": research_claims,
            "research_summary": self._build_research_summary(research_claims),
            "used_research": research_needed,
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
        print("\nRETRIEVED SOURCES:")
        for source in result.get("evidence_summary", []):
            print(
                f"{source['rank']}. "
                f"{source.get('collection') or 'N/A'} | "
                f"{source.get('doc_type') or 'N/A'} | "
                f"{source.get('occupation_title') or 'N/A'} | "
                f"{source.get('section') or 'N/A'} | "
                f"{float(source.get('score') or 0.0):.4f} | "
                f"{source.get('id') or 'N/A'}"
            )

        research_sources = result.get("research_summary") or []
        if research_sources:
            print("\n=== RESEARCH SOURCES ===")
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
        query = input("Enter your career question:\n").strip()

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
