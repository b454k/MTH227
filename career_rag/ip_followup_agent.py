"""Open-ended follow-up filtering for the local O*NET Interest Profiler."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from career_rag.config import ENV_PATH
from career_rag.interest_profiler_local import (
    RIASEC_BY_CODE,
    RIASEC_CODES,
    RIASEC_INTERESTS,
    canonical_interest,
    make_holland_code,
)


MAX_FOLLOWUP_QUESTIONS = 10
CAREER_FILTERING_STAGE = "career_filtering"

CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}

CAREER_FILTERING_QUESTIONS = [
    {
        "id": "work_setting",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Work Context - Physical Setting",
        "source": "O*NET Work Context (onetonline.org/find/descriptor/result/4.C.2)",
        "question": (
            "Describe where you'd most want to spend your working hours. "
            "Think about the physical space, the atmosphere, indoors or outdoors - "
            "paint the picture of your ideal work environment."
        ),
        "llm_instruction": (
            "Infer whether the user prefers outdoor/field settings, physical workspaces, "
            "office/desk environments, or remote/mobile work. Map to: Realistic "
            "(outdoor, physical), Conventional/Investigative (office), "
            "Artistic/Enterprising (flexible, varied)."
        ),
    },
    {
        "id": "work_with",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Work Activities - People / Data / Things / Ideas",
        "source": "Holland (1997); O*NET Work Activities",
        "question": (
            "When you imagine yourself fully absorbed in work you love, what are you "
            "actually doing? Not the job title - what's happening in the room, what "
            "are you interacting with?"
        ),
        "llm_instruction": (
            "Identify primary work object: people (Social/Enterprising), "
            "data/information (Investigative/Conventional), physical things "
            "(Realistic), ideas/creative output (Artistic/Investigative). Weight the "
            "dominant category."
        ),
    },
    {
        "id": "independence",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Work Values - Independence",
        "source": "O*NET Work Importance Locator; CareerOneStop Work Values Matcher",
        "question": (
            "Tell me about a time you felt most free and effective at work or school. "
            "What made it work - were you on your own, collaborating, following a "
            "structure, or making it up as you went?"
        ),
        "llm_instruction": (
            "Assess autonomy preference: high independence "
            "(Artistic/Investigative/Enterprising), structured with some discretion "
            "(Social/Realistic), clear direction and process "
            "(Conventional/Realistic). Look for language around freedom, control, "
            "rules, flexibility."
        ),
    },
    {
        "id": "team_dynamic",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Work Context - Interpersonal Relationships",
        "source": "O*NET Work Context descriptor 4.C.1",
        "question": (
            "How do you feel about working with other people day to day? Do you find "
            "it energizing or draining? What's the ideal amount and type of human "
            "interaction in your work?"
        ),
        "llm_instruction": (
            "Determine interpersonal preference: solo/minimal contact "
            "(Investigative/Realistic/Artistic), small close team "
            "(Investigative/Artistic/Realistic), large group coordination "
            "(Social/Enterprising/Conventional), client-facing variety "
            "(Social/Enterprising). Note emotional tone - energized vs drained "
            "signals Social vs Investigative strongly."
        ),
    },
    {
        "id": "impact_type",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Work Values - Achievement / Altruism / Recognition",
        "source": "O*NET Work Importance Locator",
        "question": (
            "At the end of a really good workday, what would make you feel like it "
            "actually mattered? What kind of difference do you want your work to make?"
        ),
        "llm_instruction": (
            "Identify impact orientation: helping individuals directly (Social), "
            "advancing knowledge or solving problems (Investigative/Realistic), "
            "building or scaling something (Enterprising/Conventional), creating "
            "lasting work (Artistic/Investigative). This strongly differentiates "
            "Social from Enterprising even with similar surface answers."
        ),
    },
    {
        "id": "structure_preference",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Work Values - Working Conditions / Independence",
        "source": "O*NET Work Values; Holland (1997) consistency principle",
        "question": (
            "How do you feel when your day has no clear agenda or plan? And on the "
            "flip side, how do you feel when every hour is scheduled for you?"
        ),
        "llm_instruction": (
            "Assess structure tolerance using both ends of the spectrum. Comfortable "
            "with both = Enterprising/Social. Prefers no agenda = "
            "Artistic/Investigative. Prefers full schedule = Conventional/Realistic. "
            "Distress at ambiguity signals Conventional strongly."
        ),
    },
    {
        "id": "skill_anchor",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Skills - Core competency self-assessment",
        "source": "Schein (1978) Career Anchors; O*NET Skills taxonomy",
        "question": (
            "What's something you've done - at school, a job, a project, anywhere - "
            "where someone said 'you're really good at this' and you actually "
            "believed them? What were you doing?"
        ),
        "llm_instruction": (
            "Extract natural competency signal from concrete example. "
            "Technical/mechanical skill = Realistic/Conventional. Analysis, research, "
            "debugging = Investigative. Communication, empathy, coaching = Social. "
            "Persuasion, leadership, selling = Enterprising. Creative production, "
            "design, expression = Artistic. Concrete examples are more reliable than "
            "self-labels - weight the activity, not the adjective."
        ),
    },
    {
        "id": "recognition",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Work Values - Recognition / Achievement",
        "source": "O*NET Work Importance Locator; CareerOneStop",
        "question": (
            "What kind of acknowledgment or result makes your work feel worthwhile? "
            "Is it a visible outcome, being seen as an expert, someone's gratitude, "
            "or something else entirely?"
        ),
        "llm_instruction": (
            "Identify recognition driver: tangible visible result "
            "(Realistic/Enterprising), expert status/respect "
            "(Investigative/Conventional), personal gratitude from those helped "
            "(Social), creative work appreciated by others (Artistic). "
            "Differentiates within tied codes especially I vs S and E vs A."
        ),
    },
    {
        "id": "future_concern",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Work Values - Security / Achievement",
        "source": "Jeon et al. (2025) Letters from Future Self - concern elicitation",
        "question": (
            "What worries you most when you think about your career future? Not the "
            "practical stuff like salary - what deeper fear comes up when you imagine "
            "making the wrong choice?"
        ),
        "llm_instruction": (
            "Map concern to suppressed work value: fear of instability = Security "
            "dominant (Conventional/Realistic). Fear of meaninglessness = "
            "Achievement/Altruism (Investigative/Social). Fear of losing creative "
            "identity = Artistic strongly. Fear of missing impact = "
            "Social/Enterprising. Concerns often reveal values more clearly than "
            "stated preferences."
        ),
    },
    {
        "id": "dream_scenario",
        "stage": CAREER_FILTERING_STAGE,
        "onet_dimension": "Holistic career vision - narrative elicitation",
        "source": "Jeon et al. (2025); Renji et al. (2025) cognitive profiling",
        "question": (
            "Five years from now, someone who knows you well runs into you and you "
            "tell them what you've been up to. What do you want to be telling them?"
        ),
        "llm_instruction": (
            "This is the highest-signal question. Extract: setting, activity, who "
            "they're with, what they built or did. Built/fixed something real = "
            "Realistic/Conventional. Became expert in a field = Investigative. "
            "Running own thing or leading team = Enterprising. Created work others "
            "connect with = Artistic. Helped people through a hard thing = Social. "
            "Cross-reference with all prior answers for consistency. If this answer "
            "contradicts RIASEC code, flag it as worth discussing."
        ),
    },
]

ANSWER_ANALYSIS_SYSTEM_PROMPT = """You are analyzing a career exploration answer.

Question asked: {question}
User's answer: {answer}

Your task:
{llm_instruction}

Output only valid JSON with:
- dominant_categories: list of 1-3 RIASEC letters this answer supports
- confidence: high / medium / low
- reasoning: one sentence explaining why
- flag: any contradiction with prior answers worth noting, or empty string
"""


class OpenAIChatProvider:
    """Tiny OpenAI chat client wrapper used only for follow-up refinement."""

    def __init__(self, model: str | None = None) -> None:
        self._load_dotenv_if_available()
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.client = None
        self.setup_error = ""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.setup_error = "OPENAI_API_KEY is not available."
            return
        try:
            from openai import OpenAI
        except ImportError:
            self.setup_error = "The OpenAI Python package is not installed."
            return
        self.client = OpenAI(api_key=api_key)

    @property
    def available(self) -> bool:
        """Return True when the provider can make LLM calls."""
        return self.client is not None

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
    ) -> str:
        """Call the OpenAI chat completions API."""
        if self.client is None:
            raise RuntimeError(self.setup_error or "OpenAI client is not available.")

        kwargs: dict[str, Any] = {
            "model": self.model,
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

    @staticmethod
    def _load_dotenv_if_available() -> None:
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        load_dotenv(dotenv_path=ENV_PATH)


def build_system_prompt(
    profile_result: dict[str, Any],
    questions_asked: list[dict[str, str]] | None = None,
) -> str:
    """Build a compact context prompt for compatibility with older callers."""
    return "\n".join(
        [
            "You are analyzing open-ended career filtering answers.",
            f"Initial Holland code: {profile_result.get('initial_holland_code')}",
            f"Initial top interests: {profile_result.get('initial_top_interests')}",
            f"Current Job Zone: {profile_result.get('current_job_zone')}",
            f"Future Job Zone: {profile_result.get('future_job_zone')}",
            f"Previous answers: {json.dumps(questions_asked or [], ensure_ascii=True)}",
        ]
    )


def build_followup_question_plan(
    profile_result: dict[str, Any],
    max_questions: int = MAX_FOLLOWUP_QUESTIONS,
) -> list[dict[str, str]]:
    """Return the fixed open-ended career filtering question sequence."""
    del profile_result
    max_questions = min(max(1, int(max_questions)), MAX_FOLLOWUP_QUESTIONS)
    return [dict(question) for question in CAREER_FILTERING_QUESTIONS[:max_questions]]


def get_next_followup_question(
    question_plan: list[dict[str, str]],
    questions_asked: list[dict[str, str]],
) -> dict[str, str] | None:
    """Return the next pending follow-up question."""
    if len(questions_asked) >= len(question_plan):
        return None
    return question_plan[len(questions_asked)]


def record_followup_answer(
    question_plan: list[dict[str, str]],
    questions_asked: list[dict[str, str]],
    answer: str,
    max_questions: int = MAX_FOLLOWUP_QUESTIONS,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Append an answer without inserting any extra follow-up question."""
    del max_questions
    current = get_next_followup_question(question_plan, questions_asked)
    if current is None:
        return question_plan, questions_asked

    updated_answers = [
        *questions_asked,
        {
            "id": current.get("id", ""),
            "stage": current.get("stage", CAREER_FILTERING_STAGE),
            "onet_dimension": current.get("onet_dimension", ""),
            "source": current.get("source", ""),
            "question": current["question"],
            "llm_instruction": current.get("llm_instruction", ""),
            "answer": str(answer).strip(),
        },
    ]
    return list(question_plan)[:MAX_FOLLOWUP_QUESTIONS], updated_answers


def complete_followup_refinement(
    profile_result: dict[str, Any],
    questions_asked: list[dict[str, str]],
    provider: OpenAIChatProvider | None = None,
) -> dict[str, Any]:
    """Analyze open-ended answers and return the saved refinement object."""
    if not questions_asked:
        return {
            "method": "initial_result_no_followup",
            "questions_asked": [],
            "question_results": [],
            "filtering_result": aggregate_filtering_results(
                _initial_holland_letters(profile_result),
                [],
            ),
            "final_refinement": build_template_refinement(profile_result, []),
            "json_valid": True,
        }

    provider = provider or OpenAIChatProvider()
    question_results: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in questions_asked:
        if provider.available:
            try:
                question_results.append(_llm_answer_analysis(item, provider))
                continue
            except Exception as exc:
                errors.append(f"{item.get('id') or item.get('question')}: {exc}")
        question_results.append(_template_answer_analysis(item))

    filtering_result = aggregate_filtering_results(
        _initial_holland_letters(profile_result),
        question_results,
    )
    final_refinement = build_final_refinement_from_filtering(
        profile_result=profile_result,
        questions_asked=questions_asked,
        question_results=question_results,
        filtering_result=filtering_result,
    )
    result = {
        "method": "llm_per_question" if provider.available and not errors else "template_fallback",
        "questions_asked": questions_asked,
        "question_results": question_results,
        "filtering_result": filtering_result,
        "final_refinement": final_refinement,
        "json_valid": True,
    }
    if errors:
        result["analysis_errors"] = errors
    return result


def aggregate_filtering_results(
    holland_code: list[str] | tuple[str, ...] | str,
    question_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-question RIASEC evidence into a refined code."""
    original_letters = _normalize_holland_letters(holland_code)
    votes = {cat: 0.0 for cat in "RIASEC"}

    for result in question_results:
        confidence = str(result.get("confidence") or "low").strip().lower()
        weight = CONFIDENCE_WEIGHT.get(confidence, 0.3)
        for category in _dominant_letters(result.get("dominant_categories")):
            votes[category] += weight

    filtered = {cat: votes[cat] for cat in original_letters}
    ranked = sorted(
        filtered.items(),
        key=lambda item: (-item[1], original_letters.index(item[0])),
    )
    refined_letters = [cat for cat, _ in ranked] or original_letters
    return {
        "original_code": "".join(original_letters),
        "refined_primary": refined_letters[0] if refined_letters else "",
        "refined_code": "".join(refined_letters),
        "vote_detail": {key: round(value, 4) for key, value in votes.items()},
        "filtered_vote_detail": {key: round(value, 4) for key, value in filtered.items()},
    }


def build_template_refinement(
    profile_result: dict[str, Any],
    questions_asked: list[dict[str, str]],
) -> dict[str, Any]:
    """Build a deterministic refinement when no LLM answer analyses exist."""
    question_results = [_template_answer_analysis(item) for item in questions_asked]
    filtering_result = aggregate_filtering_results(
        _initial_holland_letters(profile_result),
        question_results,
    )
    return build_final_refinement_from_filtering(
        profile_result=profile_result,
        questions_asked=questions_asked,
        question_results=question_results,
        filtering_result=filtering_result,
    )


def build_final_refinement_from_filtering(
    profile_result: dict[str, Any],
    questions_asked: list[dict[str, str]],
    question_results: list[dict[str, Any]],
    filtering_result: dict[str, Any],
) -> dict[str, Any]:
    """Convert aggregate filtering output into the app's final refinement schema."""
    refined_letters = list(str(filtering_result.get("refined_code") or ""))
    refined_top = [
        RIASEC_BY_CODE[letter]
        for letter in refined_letters
        if letter in RIASEC_BY_CODE
    ]
    if not refined_top:
        refined_top = [canonical_interest(item) for item in profile_result["initial_top_interests"]]

    original_top = [canonical_interest(item) for item in profile_result["initial_top_interests"]]
    key_preferences = _key_sub_preferences(questions_asked, question_results)
    future_summary = _future_vision_summary(questions_asked)
    concerns = _concerns_noted(questions_asked)

    return {
        "tie_resolution": {
            "was_needed": bool((profile_result.get("score_ambiguity") or {}).get("has_ambiguity")),
            "ambiguous_categories": [
                canonical_interest(item)
                for item in (profile_result.get("score_ambiguity") or {}).get("ambiguous_categories", [])
            ],
            "resolved_order": refined_top,
            "reasoning_summary": (
                "Open-ended career filtering questions re-ordered the original Holland code "
                f"from {', '.join(original_top)} to {', '.join(refined_top)} using weighted RIASEC votes."
            ),
        },
        "refined_top_interests": refined_top,
        "refined_holland_code": make_holland_code(refined_top),
        "key_sub_preferences": key_preferences,
        "future_vision_summary": future_summary,
        "concerns_noted": concerns,
        "career_matching_guidance": {
            "prioritize_interests": refined_top,
            "job_zone_to_use": int(profile_result["future_job_zone"]),
            "notes": (
                "Use the original O*NET Interest Profiler matches as the candidate pool, "
                "then rank them by the open-ended filtering answers and O*NET occupation signals."
            ),
        },
        "ready_for_job_matching": True,
        "filtering_result": filtering_result,
        "answer_analysis_summary": question_results,
    }


def validate_final_refinement_json(value: Any) -> dict[str, Any]:
    """Validate an existing final refinement object for compatibility."""
    if not isinstance(value, dict):
        raise ValueError("Final refinement must be a JSON object.")
    refined_top = [
        canonical_interest(item)
        for item in _ensure_list(value.get("refined_top_interests"))
    ]
    if not refined_top:
        raise ValueError("refined_top_interests must contain at least one interest.")
    refined_top = refined_top[:3]
    guidance = value.get("career_matching_guidance") or {}
    try:
        job_zone = int(guidance.get("job_zone_to_use") or 0)
    except (TypeError, ValueError):
        job_zone = 0
    if job_zone not in {1, 2, 3, 4, 5}:
        raise ValueError("career_matching_guidance.job_zone_to_use must be 1-5.")
    return {
        **value,
        "refined_top_interests": refined_top,
        "refined_holland_code": make_holland_code(refined_top),
        "career_matching_guidance": {
            **guidance,
            "prioritize_interests": [
                canonical_interest(item)
                for item in _ensure_list(guidance.get("prioritize_interests") or refined_top)
            ][:3],
            "job_zone_to_use": job_zone,
        },
        "ready_for_job_matching": bool(value.get("ready_for_job_matching", True)),
    }


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating accidental Markdown code fences."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response.")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _llm_answer_analysis(
    item: dict[str, Any],
    provider: OpenAIChatProvider,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": ANSWER_ANALYSIS_SYSTEM_PROMPT.format(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                llm_instruction=item.get("llm_instruction", ""),
            ),
        },
        {
            "role": "user",
            "content": "Analyze this answer and return only the requested JSON.",
        },
    ]
    raw = provider.complete(messages, temperature=0.0, response_format_json=True)
    return _validate_answer_analysis(parse_json_object(raw), item)


def _validate_answer_analysis(value: Any, item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Answer analysis must be a JSON object.")
    dominant = _dominant_letters(value.get("dominant_categories"))[:3]
    if not dominant:
        dominant = _template_answer_analysis(item)["dominant_categories"]
    confidence = str(value.get("confidence") or "low").strip().lower()
    if confidence not in CONFIDENCE_WEIGHT:
        confidence = "low"
    return {
        "question_id": item.get("id", ""),
        "dominant_categories": dominant,
        "confidence": confidence,
        "reasoning": _clean_text(value.get("reasoning")),
        "flag": _clean_text(value.get("flag")),
    }


def _template_answer_analysis(item: dict[str, Any]) -> dict[str, Any]:
    answer = _normalize(item.get("answer"))
    question_id = str(item.get("id") or "")
    scores = {letter: 0 for letter in "RIASEC"}

    keyword_scores = {
        "R": [
            "outdoor",
            "outside",
            "field",
            "physical",
            "hands",
            "tools",
            "machines",
            "equipment",
            "build",
            "fix",
            "repair",
            "technical",
        ],
        "I": [
            "research",
            "analyze",
            "analysis",
            "data",
            "information",
            "science",
            "problem",
            "debug",
            "expert",
            "knowledge",
            "understand",
        ],
        "A": [
            "creative",
            "design",
            "art",
            "write",
            "writing",
            "music",
            "visual",
            "flexible",
            "original",
            "create",
        ],
        "S": [
            "people",
            "help",
            "support",
            "teach",
            "coach",
            "empathy",
            "gratitude",
            "community",
            "client",
            "human",
        ],
        "E": [
            "lead",
            "leader",
            "sell",
            "persuade",
            "business",
            "startup",
            "run",
            "scale",
            "manage",
            "visible",
        ],
        "C": [
            "structure",
            "schedule",
            "organized",
            "process",
            "rules",
            "office",
            "stable",
            "security",
            "records",
            "clear",
        ],
    }
    for letter, keywords in keyword_scores.items():
        scores[letter] += sum(1 for keyword in keywords if keyword in answer)

    fallback_by_question = {
        "work_setting": ["R", "I", "C"],
        "work_with": ["S", "I", "R"],
        "independence": ["A", "I", "E"],
        "team_dynamic": ["S", "E", "I"],
        "impact_type": ["S", "I", "E"],
        "structure_preference": ["C", "R", "I"],
        "skill_anchor": ["I", "S", "R"],
        "recognition": ["I", "S", "E"],
        "future_concern": ["C", "S", "I"],
        "dream_scenario": ["I", "E", "S"],
    }
    ranked = sorted(scores.items(), key=lambda item_score: (-item_score[1], item_score[0]))
    dominant = [letter for letter, score in ranked if score > 0][:3]
    if not dominant:
        dominant = fallback_by_question.get(question_id, ["I"])[:1]
    confidence = "high" if ranked and ranked[0][1] >= 3 else "medium" if ranked and ranked[0][1] >= 1 else "low"
    return {
        "question_id": question_id,
        "dominant_categories": dominant,
        "confidence": confidence,
        "reasoning": (
            "Template analysis matched answer keywords to the supplied O*NET/Holland filtering instruction."
        ),
        "flag": "",
    }


def _key_sub_preferences(
    questions_asked: list[dict[str, str]],
    question_results: list[dict[str, Any]],
) -> list[str]:
    by_id = {result.get("question_id"): result for result in question_results}
    preferences = []
    for item in questions_asked:
        answer = _clean_text(item.get("answer"))
        if not answer:
            continue
        result = by_id.get(item.get("id")) or {}
        cats = "".join(result.get("dominant_categories") or [])
        preferences.append(f"{item.get('id')}: {cats} - {_shorten(answer, 120)}")
    return preferences[:10]


def _future_vision_summary(questions_asked: list[dict[str, str]]) -> str:
    dream = _answer_for_id(questions_asked, "dream_scenario")
    impact = _answer_for_id(questions_asked, "impact_type")
    if dream and impact:
        return f"Dream scenario: {dream} Desired impact: {impact}"
    if dream:
        return f"Dream scenario: {dream}"
    if impact:
        return f"Desired impact: {impact}"
    return "Open-ended filtering answers were recorded."


def _concerns_noted(questions_asked: list[dict[str, str]]) -> list[str]:
    concern = _answer_for_id(questions_asked, "future_concern")
    return [concern] if concern else []


def _answer_for_id(questions_asked: list[dict[str, str]], question_id: str) -> str:
    for item in questions_asked:
        if item.get("id") == question_id:
            return _clean_text(item.get("answer"))
    return ""


def _initial_holland_letters(profile_result: dict[str, Any]) -> list[str]:
    code = (
        profile_result.get("initial_holland_code")
        or profile_result.get("initial_code")
        or profile_result.get("holland_code")
        or ""
    )
    letters = _normalize_holland_letters(str(code))
    if letters:
        return letters
    return [
        RIASEC_CODES[canonical_interest(interest)]
        for interest in profile_result.get("initial_top_interests", [])[:3]
    ]


def _normalize_holland_letters(value: list[str] | tuple[str, ...] | str) -> list[str]:
    raw = list(value) if not isinstance(value, str) else list(value)
    letters = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        if len(text) == 1 and text.upper() in RIASEC_BY_CODE:
            letter = text.upper()
        else:
            letter = RIASEC_CODES[canonical_interest(text)]
        if letter not in letters:
            letters.append(letter)
    return letters[:3]


def _dominant_letters(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    letters = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        try:
            letter = text.upper() if len(text) == 1 else RIASEC_CODES[canonical_interest(text)]
        except ValueError:
            continue
        if letter in RIASEC_BY_CODE and letter not in letters:
            letters.append(letter)
    return letters


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _shorten(value: Any, limit: int) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
