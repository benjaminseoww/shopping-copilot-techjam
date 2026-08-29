from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.models import Constraint, SessionState, UserProfile
from starter.recommendation import RecommendationEngine


def _product(
    parent_asin: str,
    title: str,
    categories: list[str],
    features: list[str] | None = None,
    details: dict | None = None,
    store: str = "Example",
    description: list[str] | None = None,
    average_rating: float = 4.0,
    rating_number: float = 10.0,
    price: float | None = None,
) -> dict:
    product = {
        "parent_asin": parent_asin,
        "title": title,
        "categories": categories,
        "features": features or [],
        "details": details or {"Department": "Women"},
        "store": store,
        "description": description or [],
        "average_rating": average_rating,
        "rating_number": rating_number,
    }
    if price is not None:
        product["price"] = price
    return product


class RecommendationEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _engine(self, products: list[dict]) -> RecommendationEngine:
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        return RecommendationEngine(self.catalog_path)

    def test_uses_accumulated_active_constraints(self) -> None:
        engine = self._engine(
            [
                _product(
                    "SHOE",
                    "Blue Cotton Running Shoe",
                    ["Clothing", "Shoes", "Running"],
                    features=["lightweight sole"],
                    description=["comfortable running shoe"],
                    average_rating=4.2,
                    rating_number=20,
                ),
                _product(
                    "HAT",
                    "Red Wool Winter Hat",
                    ["Clothing", "Accessories", "Hats"],
                    features=["warm knit"],
                    description=["winter headwear"],
                    average_rating=4.8,
                    rating_number=100,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shoes Running",
            active_constraints=[
                Constraint("blue", "color", 1, "initial"),
                Constraint("lightweight sole", "feature", 2, "clarification"),
            ],
        )
        results = engine.recommend(state, top_k=2).for_contract(2)
        self.assertEqual(results[0].parent_asin, "SHOE")
        self.assertEqual(len({item.parent_asin for item in results}), 2)

    def test_fallback_is_non_empty_and_catalog_valid(self) -> None:
        engine = self._engine(
            [
                _product(
                    "SHOE",
                    "Blue Cotton Running Shoe",
                    ["Clothing", "Shoes", "Running"],
                    features=["lightweight sole"],
                    average_rating=4.2,
                    rating_number=20,
                ),
                _product(
                    "HAT",
                    "Red Wool Winter Hat",
                    ["Clothing", "Accessories", "Hats"],
                    features=["warm knit"],
                    average_rating=4.8,
                    rating_number=100,
                ),
            ]
        )
        state = SessionState("session-1", UserProfile())
        results = engine.recommend(state, top_k=2).for_contract(2)
        self.assertEqual([item.parent_asin for item in results], ["HAT", "SHOE"])
        self.assertTrue(
            all(item.parent_asin in engine.catalog_ids for item in results)
        )

    def test_joint_coverage_outranks_single_attribute_match(self) -> None:
        engine = self._engine(
            [
                _product(
                    "PARTIAL",
                    "Blue Blue Blue Jacket",
                    ["Clothing", "Jackets"],
                    features=["everyday wear"],
                    rating_number=900,
                ),
                _product(
                    "FULL",
                    "Running Shoe",
                    ["Clothing", "Shoes", "Running"],
                    features=["blue lightweight sole"],
                    rating_number=5,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shoes Running",
            active_constraints=[
                Constraint("blue", "color", 1, "initial"),
                Constraint("lightweight sole", "feature", 2, "clarification"),
            ],
        )
        results = engine.recommend(state, top_k=2).for_contract(2)
        self.assertEqual(results[0].parent_asin, "FULL")

    def test_phrase_match_outranks_scattered_tokens(self) -> None:
        engine = self._engine(
            [
                _product(
                    "TOKENS",
                    "Waterproof coat",
                    ["Clothing", "Coats"],
                    features=["breathable mesh lining"],
                    description=["membrane accents"],
                    rating_number=400,
                ),
                _product(
                    "PHRASE",
                    "Rain Shell",
                    ["Clothing", "Coats"],
                    features=["waterproof breathable membrane"],
                    rating_number=8,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Coats",
            active_constraints=[
                Constraint("waterproof breathable membrane", "feature", 1, "initial"),
            ],
        )
        results = engine.recommend(state, top_k=2).for_contract(2)
        self.assertEqual(results[0].parent_asin, "PHRASE")

    def test_overfetch_reranks_target_that_is_outside_raw_top_ten(self) -> None:
        products = [
            _product(
                f"POP{index:02d}",
                "Blue Lightweight Running Shoes",
                ["Shoes", "Running"],
                features=["rubber sole support"],
                rating_number=800 - index,
            )
            for index in range(30)
        ]
        products.append(
            _product(
                "TARGET",
                "Blue Trail Footwear",
                ["Shoes", "Running"],
                features=["lightweight sole for racing"],
                rating_number=1,
            )
        )
        engine = self._engine(products)
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shoes Running",
            active_constraints=[
                Constraint("blue", "color", 1, "initial"),
                Constraint("lightweight sole", "feature", 2, "clarification"),
            ],
        )
        raw_top_ten = engine._search(
            "Shoes Running blue lightweight sole",
            10,
        )
        self.assertNotIn("TARGET", raw_top_ten)
        ranked = engine.recommend(state, top_k=10)
        self.assertEqual(ranked.for_contract(10)[0].parent_asin, "TARGET")

    def test_superseded_constraints_are_not_scored(self) -> None:
        engine = self._engine(
            [
                _product(
                    "RED",
                    "Red Leather Boot",
                    ["Shoes", "Boots"],
                    features=["insulated"],
                    rating_number=50,
                ),
                _product(
                    "BLUE",
                    "Blue Cotton Shoe",
                    ["Shoes", "Running"],
                    features=["lightweight"],
                    rating_number=5,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shoes",
            active_constraints=[Constraint("blue", "color", 3, "override")],
            superseded_constraints=[Constraint("red leather", "feature", 1, "initial")],
        )
        results = engine.recommend(state, top_k=2).for_contract(2)
        self.assertEqual(results[0].parent_asin, "BLUE")

    def test_overfetch_keeps_more_than_contract_prefix(self) -> None:
        products = [
            _product(
                f"P{index:03d}",
                f"Item {index}",
                ["Clothing"],
                features=[f"feature {index}"],
                rating_number=200 - index,
            )
            for index in range(120)
        ]
        engine = self._engine(products)
        ranked = engine.recommend(SessionState("session-1", UserProfile()), top_k=10)
        contract = ranked.for_contract(10)
        self.assertGreaterEqual(len(ranked.items), 100)
        self.assertEqual(len(contract), 10)
        self.assertEqual(
            [item.parent_asin for item in contract],
            [item.parent_asin for item in ranked.items[:10]],
        )
        self.assertEqual(len({item.parent_asin for item in ranked.items}), len(ranked.items))

    def test_new_attribute_changes_rank_without_previous_top_ten(self) -> None:
        engine = self._engine(
            [
                _product(
                    "FIRST",
                    "Blue Shirt",
                    ["Clothing", "Shirts"],
                    features=["casual cotton"],
                    rating_number=80,
                ),
                _product(
                    "SECOND",
                    "Blue Shirt",
                    ["Clothing", "Shirts"],
                    features=["nylon shell"],
                    rating_number=70,
                ),
            ]
        )
        before = SessionState(
            "session-1",
            UserProfile(),
            category="Shirts",
            active_constraints=[Constraint("blue", "color", 1, "initial")],
        )
        after = SessionState(
            "session-1",
            UserProfile(),
            category="Shirts",
            active_constraints=[
                Constraint("blue", "color", 1, "initial"),
                Constraint("nylon shell", "feature", 2, "clarification"),
            ],
        )
        self.assertEqual(
            engine.recommend(before, top_k=2).for_contract(2)[0].parent_asin,
            "FIRST",
        )
        self.assertEqual(
            engine.recommend(after, top_k=2).for_contract(2)[0].parent_asin,
            "SECOND",
        )

    def test_store_match_ranks_brand_constraint(self) -> None:
        engine = self._engine(
            [
                _product(
                    "OTHER",
                    "Running Shoe",
                    ["Shoes", "Running"],
                    store="Acme",
                    rating_number=400,
                ),
                _product(
                    "NIKE",
                    "Running Shoe",
                    ["Shoes", "Running"],
                    store="Nike Official",
                    rating_number=5,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shoes Running",
            active_constraints=[Constraint("Nike", "brand", 1, "clarification")],
        )
        results = engine.recommend(state, top_k=2).for_contract(2)
        self.assertEqual(results[0].parent_asin, "NIKE")

    def test_missing_price_is_not_hard_filtered(self) -> None:
        engine = self._engine(
            [
                _product(
                    "PRICED",
                    "Cotton Shirt",
                    ["Clothing", "Shirts"],
                    features=["cotton"],
                    price=18.0,
                    rating_number=50,
                ),
                _product(
                    "MISSING",
                    "Cotton Shirt",
                    ["Clothing", "Shirts"],
                    features=["cotton"],
                    rating_number=40,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shirts",
            active_constraints=[
                Constraint("cotton", "material", 1, "initial"),
                Constraint("budget around $20", "budget", 2, "clarification"),
            ],
        )
        ids = [item.parent_asin for item in engine.recommend(state, top_k=2).for_contract(2)]
        self.assertIn("MISSING", ids)
        self.assertEqual(len(ids), 2)

    def test_profile_tags_do_not_exclude_the_target(self) -> None:
        engine = self._engine(
            [
                _product(
                    "TARGET",
                    "Blue Shirt",
                    ["Clothing", "Shirts"],
                    features=["cotton"],
                    rating_number=5,
                ),
                _product(
                    "COMFY",
                    "Comfort Tee",
                    ["Clothing", "Shirts"],
                    features=["comfort fit"],
                    rating_number=90,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(preference_tags=("comfort",)),
            category="Shirts",
            active_constraints=[Constraint("blue", "color", 1, "initial")],
        )
        results = engine.recommend(state, top_k=2).for_contract(2)
        self.assertEqual(results[0].parent_asin, "TARGET")
        self.assertEqual({item.parent_asin for item in results}, {"TARGET", "COMFY"})


if __name__ == "__main__":
    unittest.main()
