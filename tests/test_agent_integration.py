from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


class AgentIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "A",
                "title": "Blue Cotton Running Shoe",
                "categories": ["Clothing", "Shoes", "Running"],
                "features": ["lightweight sole"],
                "details": {"Department": "Women"},
                "store": "Example",
                "description": ["comfortable running shoe"],
                "average_rating": 4.5,
                "rating_number": 100,
            },
            {
                "parent_asin": "B",
                "title": "Red Leather Winter Boot",
                "categories": ["Clothing", "Shoes", "Boots"],
                "features": ["insulated"],
                "details": {"Department": "Women"},
                "store": "Example",
                "description": ["warm winter boot"],
                "average_rating": 4.4,
                "rating_number": 80,
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(catalog_path)
        self.profile = {
            "purchase_frequency": "frequent",
            "average_prior_rating": 4.0,
            "rating_style": "critical",
            "preference_tags": ["comfort"],
            "summary": "Prefers comfort.",
        }

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_multi_turn_response_preserves_preferences(self) -> None:
        self.agent.reset("buying", self.profile)
        first = self.agent.respond(
            "buying",
            "I'm looking for Shoes Running. A key requirement is: cotton.",
            1,
            2,
        )
        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(first["recommendations"][0]["parent_asin"], "A")

        second = self.agent.respond(
            "buying",
            "For that, what matters is: lightweight sole.",
            2,
            2,
        )
        state = self.agent.memory.get("buying")
        self.assertEqual(
            [constraint.text for constraint in state.active_constraints],
            ["cotton", "lightweight sole"],
        )
        self.assertEqual(second["recommendations"][0]["parent_asin"], "A")
        self.assertEqual(len(second["recommendations"]), 2)

    def test_question_pool_is_wider_than_contract_payload(self) -> None:
        products = [
            {
                "parent_asin": f"P{index:03d}",
                "title": f"Blue Item {index}",
                "categories": ["Clothing", "Shoes"],
                "features": [f"feature {index}"],
                "details": {"Department": "Women"},
                "store": "Example",
                "description": ["catalog item"],
                "average_rating": 4.0,
                "rating_number": 200 - index,
            }
            for index in range(120)
        ]
        catalog_path = Path(self.directory.name) / "wide-catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        agent = Agent(catalog_path)
        captured: list[list] = []
        original = agent.questions.decide

        def capture(state, turn, candidates=()):
            captured.append(list(candidates))
            return original(state, turn, candidates)

        agent.questions.decide = capture  # type: ignore[method-assign]
        agent.reset("wide", self.profile)
        response = agent.respond(
            "wide",
            "I'm looking for Shoes. A key requirement is: blue.",
            1,
            10,
        )
        self.assertEqual(len(response["recommendations"]), 10)
        self.assertEqual(len(captured[0]), 100)
        pool_ids = [item.parent_asin for item in captured[0]]
        contract_ids = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(contract_ids, pool_ids[:10])

    def test_browsing_boundary_and_override_flows(self) -> None:
        self.agent.reset("browsing", self.profile)
        browsing = self.agent.respond(
            "browsing",
            "I'm looking for Shoes Running, but I'm still exploring.",
            1,
            2,
        )
        self.assertEqual(browsing["ask_attribute"], "other")

        self.agent.respond(
            "browsing",
            "I don't have a preference for other; please use your judgment.",
            2,
            2,
        )
        self.assertIn("other", self.agent.memory.get("browsing").no_preference)

        self.agent.reset("override", self.profile)
        self.agent.respond(
            "override",
            "I'm looking for Shoes. red leather winter boot",
            1,
            2,
        )
        final = self.agent.respond(
            "override",
            "Actually, ignore my earlier preference. What I need is: cotton.",
            10,
            2,
        )
        state = self.agent.memory.get("override")
        self.assertEqual([item.text for item in state.active_constraints], ["cotton"])
        self.assertIsNone(final["ask_attribute"])

    def test_respond_requires_reset(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond("missing", "hello", 1, 2)


if __name__ == "__main__":
    unittest.main()
