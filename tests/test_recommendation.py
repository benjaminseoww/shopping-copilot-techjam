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
        return RecommendationEngine(self.catalog_path, embedder=None)

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
        results = engine.recommend(state, 2)
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
        results = engine.recommend(state, 2)
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
        results = engine.recommend(state, 2)
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
        results = engine.recommend(state, 2)
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
        pool = engine.recommend(state, 10)
        self.assertGreater(len(pool), 10)
        self.assertEqual(pool[0].parent_asin, "TARGET")

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
        results = engine.recommend(state, 2)
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
        pool = engine.recommend(SessionState("session-1", UserProfile()), 10)
        prefix = pool[:10]
        self.assertGreaterEqual(len(pool), 100)
        self.assertEqual(len(pool), min(len(products), engine.RETRIEVE_K))
        self.assertEqual(len(prefix), 10)
        self.assertEqual(
            [item.parent_asin for item in prefix],
            [item.parent_asin for item in pool[:10]],
        )
        self.assertEqual(len({item.parent_asin for item in pool}), len(pool))

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
        self.assertEqual(engine.recommend(before, 2)[0].parent_asin, "FIRST")
        self.assertEqual(engine.recommend(after, 2)[0].parent_asin, "SECOND")

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
        results = engine.recommend(state, 2)
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
        ids = [item.parent_asin for item in engine.recommend(state, 2)[:2]]
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
        results = engine.recommend(state, 2)
        self.assertEqual(results[0].parent_asin, "TARGET")
        self.assertEqual({item.parent_asin for item in results}, {"TARGET", "COMFY"})

    def test_catalog_text_strips_root_fragments(self) -> None:
        engine = self._engine(
            [
                _product(
                    "DRESS",
                    "Blue Evening Dress",
                    ["Clothing, Shoes & Jewelry", "Women", "Dresses"],
                    features=["silk lining"],
                    details={"Department": "Women"},
                    store="Atelier",
                    description=["hidden description should not appear"],
                ),
                _product(
                    "ALT",
                    "Cotton Tee",
                    ["Clothing Shoes & Jewelry", "Men", "Shirts"],
                    features=["crew neck"],
                    store="Basics",
                ),
            ]
        )
        dress = engine.catalog_text("DRESS")
        self.assertIn("Blue Evening Dress", dress)
        self.assertIn("Women", dress)
        self.assertIn("Dresses", dress)
        self.assertIn("silk lining", dress)
        self.assertIn("Atelier", dress)
        self.assertNotIn("Clothing, Shoes & Jewelry", dress)
        self.assertNotIn("hidden description", dress)
        alt = engine.catalog_text("ALT")
        self.assertIn("Cotton Tee", alt)
        self.assertNotIn("Clothing Shoes & Jewelry", alt)
        self.assertEqual(engine.catalog_text("missing"), "")

    def test_missing_minilm_keeps_lexical_ranking(self) -> None:
        engine = self._engine(
            [
                _product(
                    "SHOE",
                    "Blue Cotton Running Shoe",
                    ["Clothing", "Shoes", "Running"],
                    features=["lightweight sole"],
                ),
                _product(
                    "HAT",
                    "Red Wool Winter Hat",
                    ["Clothing", "Accessories", "Hats"],
                    features=["warm knit"],
                ),
            ]
        )
        self.assertIsNone(engine._embedder)
        self.assertIsNone(engine._embed_matrix)
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shoes Running",
            active_constraints=[Constraint("lightweight sole", "feature", 1, "initial")],
        )
        self.assertEqual(engine.recommend(state, 2)[0].parent_asin, "SHOE")

    def test_embedding_rerank_promotes_paraphrase_match(self) -> None:
        self.catalog_path.write_text(
            "".join(
                json.dumps(product) + "\n"
                for product in [
                    _product(
                        "JACKET",
                        "Water resistant hiking jacket",
                        ["Outdoor", "Jackets"],
                        features=["water resistant shell for hiking"],
                        rating_number=400,
                    ),
                    _product(
                        "BOOTS",
                        "Waterproof trail footwear",
                        ["Outdoor", "Boots"],
                        features=["sealed hiking boots"],
                        rating_number=5,
                    ),
                ]
            ),
            encoding="utf-8",
        )
        lexical = RecommendationEngine(self.catalog_path, embedder=None)
        hybrid = RecommendationEngine(self.catalog_path, embedder=_FakeEmbedder())
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Outdoor",
            active_constraints=[
                Constraint("water resistant hiking boots", "feature", 1, "initial"),
            ],
        )
        self.assertEqual(lexical.recommend(state, 2)[0].parent_asin, "JACKET")
        self.assertEqual(hybrid.recommend(state, 2)[0].parent_asin, "BOOTS")

    def test_labeled_color_constraint_ranks_the_color_not_the_word_color(self) -> None:
        engine = self._engine(
            [
                _product(
                    "LABEL",
                    "Color size chart",
                    ["Clothing", "Accessories"],
                    features=["color guide for matching"],
                    rating_number=900,
                ),
                _product(
                    "BLUE",
                    "Everyday Shirt",
                    ["Clothing", "Shirts"],
                    features=["blue cotton jersey"],
                    rating_number=5,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shirts",
            active_constraints=[Constraint("color: blue", "color", 1, "clarification")],
        )
        self.assertEqual(engine.recommend(state, 2)[0].parent_asin, "BLUE")

    def test_material_mismatch_is_outranked_by_requested_material(self) -> None:
        engine = self._engine(
            [
                _product(
                    "POLY",
                    "Everyday Shirt",
                    ["Clothing", "Shirts"],
                    features=["polyester shell"],
                    rating_number=900,
                ),
                _product(
                    "COTTON",
                    "Everyday Shirt",
                    ["Clothing", "Shirts"],
                    features=["cotton jersey"],
                    rating_number=5,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shirts",
            active_constraints=[Constraint("cotton", "material", 1, "initial")],
        )
        self.assertEqual(engine.recommend(state, 2)[0].parent_asin, "COTTON")

    def test_compatible_superseded_feature_still_helps_rank(self) -> None:
        engine = self._engine(
            [
                _product(
                    "OTHER",
                    "Leather loafer",
                    ["Shoes", "Loafers"],
                    features=["smooth finish"],
                    rating_number=400,
                ),
                _product(
                    "TARGET",
                    "Leather boot",
                    ["Shoes", "Boots"],
                    features=["waterproof breathable membrane"],
                    rating_number=5,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shoes",
            active_constraints=[Constraint("leather", "material", 3, "override")],
            superseded_constraints=[
                Constraint("waterproof breathable membrane", "feature", 1, "initial")
            ],
        )
        self.assertEqual(engine.recommend(state, 2)[0].parent_asin, "TARGET")

    def test_leaf_category_outranks_same_material_in_another_shelf(self) -> None:
        engine = self._engine(
            [
                _product(
                    "BELT",
                    "Leather belt",
                    ["Clothing", "Accessories", "Belts"],
                    features=["full grain leather"],
                    rating_number=400,
                ),
                _product(
                    "WALLET",
                    "Slim leather wallet",
                    ["Clothing", "Wallets"],
                    features=["leather interior"],
                    rating_number=5,
                ),
            ]
        )
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Card Cases Money Organizers Wallets",
            active_constraints=[Constraint("leather", "material", 1, "initial")],
        )
        self.assertEqual(engine.recommend(state, 2)[0].parent_asin, "WALLET")

    def test_prf_expands_rare_terms_shared_by_top_hits(self) -> None:
        products = [
            _product(
                f"COMMON{index}",
                "Cotton Shirt",
                ["Clothing", "Shirts"],
                features=["cotton jersey"],
            )
            for index in range(77)
        ]
        products.extend(
            _product(
                f"RARE{index}",
                "Cotton Shirt rarefiber",
                ["Clothing", "Shirts"],
                features=["cotton rarefiber"],
            )
            for index in range(3)
        )
        engine = self._engine(products)
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Shirts",
            active_constraints=[Constraint("cotton", "material", 1, "initial")],
        )
        expansion = engine._prf_terms(state, ["RARE0", "RARE1", "RARE2"])
        self.assertIn("rarefiber", expansion)
        self.assertNotIn("cotton", expansion)
        rare = engine._products["RARE0"]
        common = engine._products["COMMON0"]
        self.assertGreater(engine._prf_bonus(expansion, rare), engine._prf_bonus(expansion, common))


class _FakeEmbedder:
    """Tiny stand-in that clusters boot/footwear separately from jackets."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vector = [0.0, 0.0, 0.0]
            if any(token in lowered for token in ("boot", "footwear", "shoe")):
                vector[0] = 1.0
            if "jacket" in lowered:
                vector[1] = 1.0
            if any(token in lowered for token in ("water", "waterproof", "resistant")):
                vector[2] = 0.3
            vectors.append(vector)
        return vectors


if __name__ == "__main__":
    unittest.main()
