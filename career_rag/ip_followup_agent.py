"""Follow-up conversation helpers for the local O*NET Interest Profiler."""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

from career_rag.config import ENV_PATH
from career_rag.interest_profiler_local import (
    RIASEC_CODES,
    RIASEC_INTERESTS,
    canonical_interest,
    make_holland_code,
)


MAX_FOLLOWUP_QUESTIONS = 10
TIE_BREAKING_STAGE = "tie_breaking"
DYNAMIC_DEEPENING_STAGE = "dynamic_deepening"
FUTURE_VISION_STAGE = "future_vision"

SYSTEM_PROMPT_TEMPLATE = """You are a career exploration assistant. Your role is to help users discover their career interests through natural conversation.

CONTEXT:
The user has already completed the O*NET Interest Profiler Short Form locally.
You have:

* their raw RIASEC scores
* their initial top 3 interests
* their current Job Zone
* their future Job Zone
* any ambiguous or tied categories

Your job is to:

1. Resolve tied or unclear categories with follow-up questions.
2. Deepen the profile with dynamic follow-up questions.
3. Explore future vision.
4. Produce a structured final refinement JSON.

GENERAL RULES:

* Ask one question at a time.
* Use a warm, non-judgmental tone.
* Never ask "what career do you want?"
* Never ask the user to self-categorize into RIASEC.
* Use activity-based questions.
* If the user gives a very short answer, ask one gentle clarification.
* Maximum total follow-up questions across all stages: 10.
* If the user seems disengaged, shorten the remaining questions.
* Do not overwrite the raw O*NET scores.
* Your output should refine interpretation, not change the official score.

STAGE 1: TIE-BREAKING

Run this only if:

* two or more top RIASEC categories are exactly tied, or
* two or more top RIASEC categories are within 1 point.

Rules:

* Ask one question at a time.
* Use activity-based questions.
* Do not ask "Do you prefer Realistic or Investigative?"
* Infer the answer from the user's response.
* After each answer, update the internal estimate of which category is stronger.
* Stop when the tie is resolved or after maximum 3 tie-breaking questions.

Tie-breaking question bank:

R vs I:
"When something breaks at home, what is your first instinct - try to fix it hands-on, or research and understand what is wrong first?"

R vs A:
"If you had a free weekend project, would building something physical or creating something artistic feel more natural to you?"

R vs S:
"Would you rather spend a day working with tools, machines, or equipment, or spend it helping and talking with people?"

R vs E:
"Do you usually prefer doing the practical work yourself, or organizing and directing others to get it done?"

R vs C:
"Would you rather physically build or repair something, or follow a precise technical process step by step?"

I vs A:
"When you get curious about something, do you usually research and analyze it deeply, or imagine and create something inspired by it?"

I vs S:
"Do you find it more satisfying to solve a complex problem, or help someone work through a difficult situation?"

I vs E:
"Would you rather analyze data and evidence, or persuade people and lead a project?"

I vs C:
"Do you prefer open-ended research where the answer is uncertain, or structured tasks with a clear correct process?"

A vs S:
"If you were volunteering, would you rather run a creative workshop, or provide one-on-one support to someone?"

A vs E:
"Do you get more energy from expressing your own creative ideas, or from pitching and selling ideas to others?"

A vs C:
"Would you rather design something original with freedom, or organize and manage detailed records and processes?"

S vs E:
"Do you prefer supporting people personally, or leading a team toward a shared goal?"

S vs C:
"Would you rather spend your day counseling and supporting people, or maintaining organized systems and records?"

E vs C:
"Do you prefer making decisions and taking risks to grow something, or maintaining accuracy and order in established systems?"

STAGE 2: DYNAMIC DEEPENING

Purpose:
After tie-breaking, ask 3-4 follow-up questions to understand the user's sub-preferences within their top RIASEC categories.

Research inspiration:
Renji et al. (2025), "Steve: LLM Powered ChatBot for Career Progression"
https://arxiv.org/abs/2504.03789

Use this as design inspiration for structured AI-led career interviews, not as a claim that these exact questions are validated.

Rules:

* Ask one question at a time.
* Ask 3-4 questions total in this stage.
* Generate questions dynamically based on the user's top 3 RIASEC categories and previous answers.
* Never repeat something already answered.
* Each question should reveal what type of career path within the top categories might fit.
* Keep answers short and conversational.

Question templates by category:

If Social is high:

1. "In a group setting, do you prefer teaching and guiding others, or listening and supporting them through problems?"
2. "Do you find more meaning in working with children, adults going through difficulties, or communities at a larger scale?"

If Investigative is high:

1. "When you research something, are you more drawn to understanding how things work scientifically, or solving real-world practical problems?"
2. "Do you prefer working with data and numbers, living organisms, technology systems, or abstract theories?"

If Artistic is high:

1. "Is your creative side more visual, written, musical/performance-based, or design-oriented?"
2. "Do you prefer creating alone, or collaborating with others on a creative project?"

If Realistic is high:

1. "Do you prefer working with technology and machines, animals and nature, or construction and physical structures?"
2. "Do you like working outdoors, in a workshop, or with technical equipment in a controlled setting?"

If Enterprising is high:

1. "Are you more drawn to starting your own thing, leading teams inside an organization, or persuading and selling to clients?"
2. "Do you prefer fast-paced business environments or strategic long-term planning?"

If Conventional is high:

1. "Do you prefer working with financial data, administrative systems, legal/regulatory details, or technology infrastructure?"
2. "Do you like working independently on structured tasks, or maintaining systems that other people rely on?"

STAGE 3: FUTURE VISION

Purpose:
Ask future-oriented questions to understand aspirations, fears, and hidden preferences.

Research inspiration:
Jeon et al. (2025), "Letters from Future Self: Augmenting the Letter-Exchange Exercise with LLM-based Agents to Enhance Young Adults' Career Exploration"
https://arxiv.org/abs/2502.18881

Use this as design inspiration for future-self reflection and career exploration. Do not claim that these exact questions are psychometrically validated.

Ask exactly these 3 questions, one at a time, in this order:

Q1:
"Imagine you wake up five years from now on a perfect workday. Where are you, what are you doing, and who are you working with?"

Q2:
"What worries you most about choosing a career path right now?"

Q3:
"Is there something you have always thought 'I'd love to do that, but it's probably not realistic' - what is it?"

STAGE 4: FINAL REFINEMENT JSON

After all stages are complete, produce structured JSON internally.

Required JSON:

{
"tie_resolution": {
"was_needed": true,
"ambiguous_categories": ["Investigative", "Conventional"],
"resolved_order": ["Investigative", "Conventional"],
"reasoning_summary": "Short explanation based on user answers."
},
"refined_top_interests": ["Investigative", "Conventional", "Social"],
"refined_holland_code": "ICS",
"key_sub_preferences": [
"Prefers data and real-world problem solving",
"Likes structured tasks but not repetitive clerical work",
"Wants some human impact in career choice"
],
"future_vision_summary": "2-3 sentence summary.",
"concerns_noted": [
"Concern about job stability",
"Concern about choosing too narrow a path"
],
"career_matching_guidance": {
"prioritize_interests": ["Investigative", "Conventional", "Social"],
"job_zone_to_use": 5,
"notes": "Use future job zone for aspirational recommendations, current job zone for immediate options."
},
"ready_for_job_matching": true
}

Important:
The LLM should return valid JSON for the final refinement.
Validate the JSON before saving.
If JSON parsing fails, ask the LLM once to repair the JSON.
If it still fails, save the raw response and mark json_valid=false.

CURRENT USER CONTEXT:
{context_block}
"""

TIE_BREAKING_QUESTIONS = {
    ("Realistic", "Investigative"): "When something breaks at home, what is your first instinct - try to fix it hands-on, or research and understand what is wrong first?",
    ("Realistic", "Artistic"): "If you had a free weekend project, would building something physical or creating something artistic feel more natural to you?",
    ("Realistic", "Social"): "Would you rather spend a day working with tools, machines, or equipment, or spend it helping and talking with people?",
    ("Realistic", "Enterprising"): "Do you usually prefer doing the practical work yourself, or organizing and directing others to get it done?",
    ("Realistic", "Conventional"): "Would you rather physically build or repair something, or follow a precise technical process step by step?",
    ("Investigative", "Artistic"): "When you get curious about something, do you usually research and analyze it deeply, or imagine and create something inspired by it?",
    ("Investigative", "Social"): "Do you find it more satisfying to solve a complex problem, or help someone work through a difficult situation?",
    ("Investigative", "Enterprising"): "Would you rather analyze data and evidence, or persuade people and lead a project?",
    ("Investigative", "Conventional"): "Do you prefer open-ended research where the answer is uncertain, or structured tasks with a clear correct process?",
    ("Artistic", "Social"): "If you were volunteering, would you rather run a creative workshop, or provide one-on-one support to someone?",
    ("Artistic", "Enterprising"): "Do you get more energy from expressing your own creative ideas, or from pitching and selling ideas to others?",
    ("Artistic", "Conventional"): "Would you rather design something original with freedom, or organize and manage detailed records and processes?",
    ("Social", "Enterprising"): "Do you prefer supporting people personally, or leading a team toward a shared goal?",
    ("Social", "Conventional"): "Would you rather spend your day counseling and supporting people, or maintaining organized systems and records?",
    ("Enterprising", "Conventional"): "Do you prefer making decisions and taking risks to grow something, or maintaining accuracy and order in established systems?",
}

DYNAMIC_QUESTION_TEMPLATES = {
    "Social": [
        "In a group setting, do you prefer teaching and guiding others, or listening and supporting them through problems?",
        "Do you find more meaning in working with children, adults going through difficulties, or communities at a larger scale?",
    ],
    "Investigative": [
        "When you research something, are you more drawn to understanding how things work scientifically, or solving real-world practical problems?",
        "Do you prefer working with data and numbers, living organisms, technology systems, or abstract theories?",
    ],
    "Artistic": [
        "Is your creative side more visual, written, musical/performance-based, or design-oriented?",
        "Do you prefer creating alone, or collaborating with others on a creative project?",
    ],
    "Realistic": [
        "Do you prefer working with technology and machines, animals and nature, or construction and physical structures?",
        "Do you like working outdoors, in a workshop, or with technical equipment in a controlled setting?",
    ],
    "Enterprising": [
        "Are you more drawn to starting your own thing, leading teams inside an organization, or persuading and selling to clients?",
        "Do you prefer fast-paced business environments or strategic long-term planning?",
    ],
    "Conventional": [
        "Do you prefer working with financial data, administrative systems, legal/regulatory details, or technology infrastructure?",
        "Do you like working independently on structured tasks, or maintaining systems that other people rely on?",
    ],
}

FUTURE_VISION_QUESTIONS = [
    "Imagine you wake up five years from now on a perfect workday. Where are you, what are you doing, and who are you working with?",
    "What worries you most about choosing a career path right now?",
    "Is there something you have always thought 'I'd love to do that, but it's probably not realistic' - what is it?",
]


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
    """Build the follow-up system prompt with this user's profiler context."""
    return SYSTEM_PROMPT_TEMPLATE.replace(
        "{context_block}",
        _profile_context_block(profile_result, questions_asked or []),
    )


def build_followup_question_plan(
    profile_result: dict[str, Any],
    max_questions: int = MAX_FOLLOWUP_QUESTIONS,
) -> list[dict[str, str]]:
    """Create a deterministic one-question-at-a-time follow-up plan."""
    max_questions = min(max(1, int(max_questions)), MAX_FOLLOWUP_QUESTIONS)
    plan: list[dict[str, str]] = []

    ambiguous = _ordered_ambiguous_categories(profile_result)
    if len(ambiguous) >= 2:
        for first, second in combinations(ambiguous, 2):
            question = _tie_breaking_question(first, second)
            if question:
                plan.append(
                    {
                        "stage": TIE_BREAKING_STAGE,
                        "question": question,
                    }
                )
            if len([item for item in plan if item["stage"] == TIE_BREAKING_STAGE]) >= 3:
                break

    remaining_after_future = max_questions - len(plan) - len(FUTURE_VISION_QUESTIONS)
    dynamic_count = min(4, max(0, remaining_after_future))
    plan.extend(_dynamic_deepening_questions(profile_result, dynamic_count))

    for question in FUTURE_VISION_QUESTIONS:
        if len(plan) >= max_questions:
            break
        plan.append({"stage": FUTURE_VISION_STAGE, "question": question})

    return plan[:max_questions]


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
    """Append an answer and insert one clarification for very short responses."""
    current = get_next_followup_question(question_plan, questions_asked)
    if current is None:
        return question_plan, questions_asked

    cleaned_answer = str(answer).strip()
    updated_answers = [
        *questions_asked,
        {
            "stage": current["stage"],
            "question": current["question"],
            "answer": cleaned_answer,
        },
    ]
    updated_plan = list(question_plan)

    if (
        _answer_is_very_short(cleaned_answer)
        and current.get("is_clarification") != "true"
        and len(updated_plan) < max_questions
        and not _stage_has_clarification(current["stage"], updated_plan, updated_answers)
    ):
        updated_plan.insert(
            len(updated_answers),
            {
                "stage": current["stage"],
                "question": "Could you say a little more about what makes that feel appealing or unappealing?",
                "is_clarification": "true",
            },
        )

    return updated_plan[:max_questions], updated_answers


def complete_followup_refinement(
    profile_result: dict[str, Any],
    questions_asked: list[dict[str, str]],
    provider: OpenAIChatProvider | None = None,
) -> dict[str, Any]:
    """Return the saved follow-up refinement object."""
    if not questions_asked:
        return {
            "method": "initial_result_no_followup",
            "questions_asked": [],
            "final_refinement": build_template_refinement(profile_result, []),
            "json_valid": True,
        }

    provider = provider or OpenAIChatProvider()
    if provider.available:
        final_refinement, json_valid, raw_response, error = _llm_final_refinement(
            profile_result,
            questions_asked,
            provider,
        )
        refinement = {
            "method": "llm",
            "questions_asked": questions_asked,
            "final_refinement": final_refinement,
            "json_valid": json_valid,
        }
        if raw_response and not json_valid:
            refinement["raw_response"] = raw_response
        if error:
            refinement["error"] = error
        return refinement

    return {
        "method": "template_fallback",
        "questions_asked": questions_asked,
        "final_refinement": build_template_refinement(profile_result, questions_asked),
        "json_valid": True,
    }


def build_template_refinement(
    profile_result: dict[str, Any],
    questions_asked: list[dict[str, str]],
) -> dict[str, Any]:
    """Build a conservative no-LLM refinement without reinterpreting raw scores."""
    initial_top = [canonical_interest(item) for item in profile_result["initial_top_interests"]]
    ambiguity = profile_result.get("score_ambiguity") or {}
    ambiguous = [canonical_interest(item) for item in ambiguity.get("ambiguous_categories", [])]
    tie_was_needed = bool(ambiguity.get("has_ambiguity"))
    resolved_order = [interest for interest in initial_top if interest in ambiguous] or ambiguous
    future_answers = [item for item in questions_asked if item["stage"] == FUTURE_VISION_STAGE]
    deepening_answers = [item for item in questions_asked if item["stage"] == DYNAMIC_DEEPENING_STAGE]

    key_sub_preferences = [
        _answer_note(item["answer"])
        for item in deepening_answers
        if item.get("answer")
    ][:5]

    concerns_noted = []
    if len(future_answers) >= 2 and future_answers[1].get("answer"):
        concerns_noted.append(_answer_note(future_answers[1]["answer"]))

    return {
        "tie_resolution": {
            "was_needed": tie_was_needed,
            "ambiguous_categories": ambiguous,
            "resolved_order": resolved_order,
            "reasoning_summary": (
                "Template fallback recorded tie-breaking answers and kept the initial deterministic order."
                if tie_was_needed
                else "No tie-breaking was needed from the raw O*NET scores."
            ),
        },
        "refined_top_interests": initial_top,
        "refined_holland_code": make_holland_code(initial_top),
        "key_sub_preferences": key_sub_preferences,
        "future_vision_summary": _future_vision_summary(future_answers),
        "concerns_noted": concerns_noted,
        "career_matching_guidance": {
            "prioritize_interests": initial_top,
            "job_zone_to_use": int(profile_result["future_job_zone"]),
            "notes": "Use future job zone for aspirational recommendations, current job zone for immediate options.",
        },
        "ready_for_job_matching": True,
    }


def validate_final_refinement_json(value: Any) -> dict[str, Any]:
    """Validate the LLM final refinement JSON before saving it."""
    if not isinstance(value, dict):
        raise ValueError("Final refinement must be a JSON object.")

    required_keys = {
        "tie_resolution",
        "refined_top_interests",
        "refined_holland_code",
        "key_sub_preferences",
        "future_vision_summary",
        "concerns_noted",
        "career_matching_guidance",
        "ready_for_job_matching",
    }
    missing = required_keys - set(value)
    if missing:
        raise ValueError(f"Final refinement missing keys: {sorted(missing)}.")

    tie_resolution = value["tie_resolution"]
    if not isinstance(tie_resolution, dict):
        raise ValueError("tie_resolution must be an object.")
    tie_resolution = {
        "was_needed": bool(tie_resolution.get("was_needed")),
        "ambiguous_categories": [
            canonical_interest(item)
            for item in _ensure_list(tie_resolution.get("ambiguous_categories"))
        ],
        "resolved_order": [
            canonical_interest(item)
            for item in _ensure_list(tie_resolution.get("resolved_order"))
        ],
        "reasoning_summary": str(tie_resolution.get("reasoning_summary") or "").strip(),
    }

    refined_top = [canonical_interest(item) for item in _ensure_list(value["refined_top_interests"])]
    if not refined_top:
        raise ValueError("refined_top_interests must contain at least one interest.")
    refined_top = refined_top[:3]
    holland_code = str(value.get("refined_holland_code") or make_holland_code(refined_top)).strip().upper()
    expected_code = make_holland_code(refined_top)
    if holland_code != expected_code:
        holland_code = expected_code

    guidance = value["career_matching_guidance"]
    if not isinstance(guidance, dict):
        raise ValueError("career_matching_guidance must be an object.")
    job_zone = int(guidance.get("job_zone_to_use") or 0)
    if job_zone not in {1, 2, 3, 4, 5}:
        raise ValueError("career_matching_guidance.job_zone_to_use must be 1-5.")

    return {
        "tie_resolution": tie_resolution,
        "refined_top_interests": refined_top,
        "refined_holland_code": holland_code,
        "key_sub_preferences": [str(item).strip() for item in _ensure_list(value["key_sub_preferences"]) if str(item).strip()],
        "future_vision_summary": str(value["future_vision_summary"]).strip(),
        "concerns_noted": [str(item).strip() for item in _ensure_list(value["concerns_noted"]) if str(item).strip()],
        "career_matching_guidance": {
            "prioritize_interests": [
                canonical_interest(item)
                for item in _ensure_list(guidance.get("prioritize_interests"))
            ][:3],
            "job_zone_to_use": job_zone,
            "notes": str(guidance.get("notes") or "").strip(),
        },
        "ready_for_job_matching": bool(value["ready_for_job_matching"]),
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


def _llm_final_refinement(
    profile_result: dict[str, Any],
    questions_asked: list[dict[str, str]],
    provider: OpenAIChatProvider,
) -> tuple[dict[str, Any], bool, str, str]:
    transcript = json.dumps(questions_asked, indent=2, ensure_ascii=True)
    messages = [
        {"role": "system", "content": build_system_prompt(profile_result, questions_asked)},
        {
            "role": "user",
            "content": (
                "The follow-up conversation is complete. Return only valid JSON matching "
                "the required final refinement schema. Do not include Markdown.\n\n"
                f"Follow-up transcript:\n{transcript}"
            ),
        },
    ]

    raw_response = ""
    try:
        raw_response = provider.complete(messages, temperature=0.2, response_format_json=True)
        parsed = parse_json_object(raw_response)
        return validate_final_refinement_json(parsed), True, raw_response, ""
    except Exception as first_error:
        repair_messages = [
            {
                "role": "system",
                "content": "Repair the user's invalid JSON. Return only one valid JSON object.",
            },
            {
                "role": "user",
                "content": (
                    f"Error: {first_error}\n\n"
                    f"Invalid response:\n{raw_response}\n\n"
                    "Return valid JSON matching the Interest Profiler final refinement schema."
                ),
            },
        ]
        try:
            repaired = provider.complete(repair_messages, temperature=0.0, response_format_json=True)
            parsed = parse_json_object(repaired)
            return validate_final_refinement_json(parsed), True, repaired, ""
        except Exception as second_error:
            fallback = build_template_refinement(profile_result, questions_asked)
            return fallback, False, raw_response, str(second_error)


def _profile_context_block(
    profile_result: dict[str, Any],
    questions_asked: list[dict[str, str]],
) -> str:
    raw_scores = profile_result.get("raw_riasec_scores") or {}
    ambiguity = profile_result.get("score_ambiguity") or {}
    return "\n".join(
        [
            f"Raw RIASEC scores: {json.dumps(raw_scores, ensure_ascii=True)}",
            f"Initial top 3 interests: {profile_result.get('initial_top_interests')}",
            f"Initial Holland code: {profile_result.get('initial_holland_code')}",
            f"Current Job Zone: {profile_result.get('current_job_zone')}",
            f"Future Job Zone: {profile_result.get('future_job_zone')}",
            f"Ambiguous categories: {ambiguity.get('ambiguous_categories', [])}",
            f"Previous answers: {json.dumps(questions_asked, ensure_ascii=True)}",
        ]
    )


def _ordered_ambiguous_categories(profile_result: dict[str, Any]) -> list[str]:
    ambiguity = profile_result.get("score_ambiguity") or {}
    if not ambiguity.get("has_ambiguity"):
        return []
    ambiguous = [canonical_interest(item) for item in ambiguity.get("ambiguous_categories", [])]
    top_order = [canonical_interest(item) for item in profile_result.get("initial_top_interests", [])]
    ordered = [interest for interest in top_order if interest in ambiguous]
    ordered.extend([interest for interest in RIASEC_INTERESTS if interest in ambiguous and interest not in ordered])
    return ordered


def _tie_breaking_question(first: str, second: str) -> str | None:
    first = canonical_interest(first)
    second = canonical_interest(second)
    ordered = tuple(sorted((first, second), key=RIASEC_INTERESTS.index))
    return TIE_BREAKING_QUESTIONS.get(ordered)


def _dynamic_deepening_questions(
    profile_result: dict[str, Any],
    count: int,
) -> list[dict[str, str]]:
    if count <= 0:
        return []
    top_interests = [canonical_interest(item) for item in profile_result.get("initial_top_interests", [])]
    questions: list[dict[str, str]] = []
    template_index = 0
    while len(questions) < count and template_index < 2:
        for interest in top_interests:
            templates = DYNAMIC_QUESTION_TEMPLATES.get(interest, [])
            if template_index < len(templates):
                questions.append(
                    {
                        "stage": DYNAMIC_DEEPENING_STAGE,
                        "question": templates[template_index],
                    }
                )
                if len(questions) >= count:
                    break
        template_index += 1
    return questions


def _answer_is_very_short(answer: str) -> bool:
    words = [word for word in answer.strip().split() if word]
    return len(words) <= 2 or len(answer.strip()) < 12


def _stage_has_clarification(
    stage: str,
    question_plan: list[dict[str, str]],
    questions_asked: list[dict[str, str]],
) -> bool:
    planned = any(
        item.get("stage") == stage and item.get("is_clarification") == "true"
        for item in question_plan
    )
    asked = any(
        item.get("stage") == stage and item.get("is_clarification") == "true"
        for item in questions_asked
    )
    return planned or asked


def _answer_note(answer: str, max_chars: int = 140) -> str:
    cleaned = " ".join(str(answer).split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return f"User mentioned: {cleaned}"


def _future_vision_summary(future_answers: list[dict[str, str]]) -> str:
    if not future_answers:
        return "No future-vision answers were recorded."
    answers = [item.get("answer", "").strip() for item in future_answers]
    parts: list[str] = []
    if len(answers) >= 1 and answers[0]:
        parts.append(f"Five-year workday image: {answers[0]}")
    if len(answers) >= 2 and answers[1]:
        parts.append(f"Career concern: {answers[1]}")
    if len(answers) >= 3 and answers[2]:
        parts.append(f"Stretch interest: {answers[2]}")
    return " ".join(parts) or "Future-vision answers were recorded but left mostly blank."


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
