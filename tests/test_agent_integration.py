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

        third = self.agent.respond(
            "buying",
            "For that, what matters is: running.",
            3,
            2,
        )
        self.assertEqual(len(third["recommendations"]), 2)

    def test_browsing_boundary_and_override_flows(self) -> None:
        self.agent.reset("browsing", self.profile)
        browsing = self.agent.respond(
            "browsing",
            "I'm looking for Shoes Running, but I'm still exploring.",
            1,
            2,
        )
        self.assertEqual(browsing["ask_attribute"], "material")

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

    def test_respond_slices_recommendations_to_top_k(self) -> None:
        catalog_path = Path(self.directory.name) / "pool_catalog.jsonl"
        products = [
            {
                "parent_asin": f"P{index}",
                "title": f"Cotton Dress {index}",
                "categories": ["Clothing", "Dresses"],
                "features": ["cotton fabric"],
                "details": {"Department": "Women"},
                "store": "Example",
                "description": ["everyday dress"],
                "average_rating": 4.0,
                "rating_number": 10 + index,
            }
            for index in range(12)
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        agent = Agent(catalog_path)
        agent.reset("pool", self.profile)
        top_k = 5
        self.assertGreater(agent.questions.candidate_pool, top_k)
        agent.respond(
            "pool",
            "I'm looking for Women Dresses. A key requirement is: cotton.",
            1,
            top_k,
        )
        response = agent.respond(
            "pool",
            "For that, what matters is: lightweight.",
            2,
            top_k,
        )
        self.assertEqual(len(response["recommendations"]), top_k)

        agent.reset("one", self.profile)
        one = agent.respond(
            "one",
            "I'm looking for Women Dresses. A key requirement is: cotton.",
            1,
            top_k,
        )
        self.assertEqual(len(one["recommendations"]), 1)

    def test_thin_evidence_returns_a_short_list(self) -> None:
        self.agent.reset("browse", self.profile)
        browsing = self.agent.respond(
            "browse",
            "I'm looking for Shoes Running, but I'm still exploring.",
            1,
            10,
        )
        self.assertEqual(len(browsing["recommendations"]), 1)

    def test_late_turns_return_the_full_list(self) -> None:
        self.agent.reset("late", self.profile)
        response = self.agent.respond(
            "late",
            "I'm looking for Shoes Running, but I'm still exploring.",
            9,
            2,
        )
        self.assertEqual(len(response["recommendations"]), 2)


if __name__ == "__main__":
    unittest.main()
