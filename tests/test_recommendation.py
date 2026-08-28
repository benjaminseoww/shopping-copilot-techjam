from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.models import Constraint, SessionState, UserProfile
from starter.recommendation import RecommendationEngine


class RecommendationEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "SHOE",
                "title": "Blue Cotton Running Shoe",
                "categories": ["Clothing", "Shoes", "Running"],
                "features": ["lightweight sole"],
                "details": {"Department": "Women"},
                "store": "Example",
                "description": ["comfortable running shoe"],
                "average_rating": 4.2,
                "rating_number": 20,
            },
            {
                "parent_asin": "HAT",
                "title": "Red Wool Winter Hat",
                "categories": ["Clothing", "Accessories", "Hats"],
                "features": ["warm knit"],
                "details": {"Department": "Women"},
                "store": "Example",
                "description": ["winter headwear"],
                "average_rating": 4.8,
                "rating_number": 100,
            },
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.engine = RecommendationEngine(self.catalog_path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_uses_accumulated_active_constraints(self) -> None:
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shoes Running",
            active_constraints=[
                Constraint("blue", "color", 1, "initial"),
                Constraint("lightweight sole", "feature", 2, "clarification"),
            ],
        )
        results = self.engine.recommend(state, top_k=2)
        self.assertEqual(results[0].parent_asin, "SHOE")
        self.assertEqual(len({item.parent_asin for item in results}), 2)

    def test_fallback_is_non_empty_and_catalog_valid(self) -> None:
        state = SessionState("session-1", UserProfile())
        results = self.engine.recommend(state, top_k=2)
        self.assertEqual([item.parent_asin for item in results], ["HAT", "SHOE"])
        self.assertTrue(
            all(item.parent_asin in self.engine.catalog_ids for item in results)
        )


if __name__ == "__main__":
    unittest.main()
