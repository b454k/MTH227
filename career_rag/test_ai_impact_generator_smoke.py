#!/usr/bin/env python3
"""Smoke tests for grounded AI-impact answer generation."""

from __future__ import annotations

import unittest

from career_rag.generator import CareerRAGGenerator


class AIImpactGeneratorSmokeTest(unittest.TestCase):
    """Exercise representative AI-impact generation paths end-to-end."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = CareerRAGGenerator(use_query_rewriting=False)

    def _run_case(self, query: str, expected_soc: str) -> dict:
        result = self.generator.generate_answer(query, k=8, research_top_k=8)
        answer = result["answer"]
        self.assertTrue(result["used_structured_ai_impact"])
        self.assertIn(expected_soc, answer)
        self.assertIn("AI Exposure By Task", answer)
        self.assertIn("| Task | Direct Anthropic Evidence |", answer)
        self.assertIn("No direct observed/mapped Anthropic evidence", answer)
        self.assertNotIn("0.0 share", answer)
        self.assertNotIn("no current AI penetration", answer.lower())
        self.assertTrue(result["ai_evidence_pack_summary"])
        for source in result["ai_evidence_pack_summary"]:
            for key in (
                "collection",
                "source_type",
                "source_name",
                "title",
                "page",
                "section",
                "occupation",
                "task",
                "score",
                "doc_id",
            ):
                self.assertIn(key, source)
        return result

    def test_environmental_engineers(self) -> None:
        self._run_case("ai impact on environmental engineer", "17-2081.00")

    def test_civil_engineers(self) -> None:
        self._run_case("ai impact on civil engineer", "17-2051.00")

    def test_mathematicians(self) -> None:
        self._run_case("ai impact on mathematician", "15-2021.00")

    def test_data_scientists(self) -> None:
        self._run_case("ai impact on data scientist", "15-2051.00")

    def test_construction_laborers(self) -> None:
        self._run_case("ai impact on construction laborer", "47-2061.00")


if __name__ == "__main__":
    unittest.main()
