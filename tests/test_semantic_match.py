from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.intent import IntentUnderstander
from starter.models import Constraint, ScoredProduct, SessionState, UserProfile
from starter.questions import QuestionsEngine, extract_values
from starter.recommendation import RecommendationEngine
from starter.semantic_match import SemanticMatcher


class _HintEmbedder:
    """Axis embedder: similar words share an axis, evaluator lists are not required."""

    AXES = (
        ("color", ("color", "hue", "shade", "navy", "burgundy", "maroon", "ivory", "beige")),
        ("material", ("material", "fabric", "leather", "suede", "canvas", "textile")),
        ("size", ("size", "width", "dimensions")),
        ("style", ("style", "fit", "cut", "sleeve", "neckline")),
        ("use_case", ("hiking", "running", "winter", "occasion", "activity", "season")),
        ("budget", ("price", "budget", "costs")),
        ("brand", ("brand", "store")),
        ("feature", ("feature", "function", "detail", "capability")),
    )

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vector = [
                float(sum(1 for hint in hints if hint in lowered))
                for _label, hints in self.AXES
            ]
            if not any(vector):
                vector[-1] = 0.1
            vectors.append(vector)
        return vectors


def _product(parent_asin: str, title: str, rating_number: float) -> dict:
    return {
        "parent_asin": parent_asin,
        "title": title,
        "categories": ["Clothing", "Women", "Dresses"],
        "features": [],
        "details": {"Department": "Women"},
        "store": "Example",
        "description": [],
        "average_rating": 4.0,
        "rating_number": rating_number,
    }


class SemanticMatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = SemanticMatcher(_HintEmbedder())

    def test_labels_values_the_gazetteer_never_listed(self) -> None:
        self.assertEqual(self.matcher.classify("navy"), "color")
        self.assertEqual(self.matcher.classify("suede"), "material")
        self.assertIsNone(self.matcher.classify("required feature"))

    def test_title_span_is_taken_from_the_product_text(self) -> None:
        self.assertEqual(self.matcher.best_span("suede wrap dress", "material"), "suede")
        self.assertIsNone(self.matcher.best_span("suede wrap dress", "color"))

    def test_disabled_matcher_does_nothing(self) -> None:
        matcher = SemanticMatcher(None)
        self.assertFalse(matcher.available())
        self.assertIsNone(matcher.classify("navy"))
        self.assertIsNone(matcher.best_span("suede dress", "material"))


class SemanticIntentTest(unittest.TestCase):
    def test_unknown_values_are_labeled_without_the_evaluator_list(self) -> None:
        lexical = IntentUnderstander()
        semantic = IntentUnderstander(semantic=SemanticMatcher(_HintEmbedder()))
        self.assertEqual(lexical.classify_constraint("navy"), "feature")
        self.assertEqual(semantic.classify_constraint("navy"), "color")
        self.assertEqual(semantic.classify_constraint("suede"), "material")
        self.assertEqual(semantic.classify_constraint("cotton"), "material")
        self.assertEqual(semantic.classify_constraint("red leather"), "material")


class SemanticQuestionsTest(unittest.TestCase):
    def test_suede_canvas_titles_can_split_material(self) -> None:
        snippets = {f"P{index}": text for index, text in enumerate(
            ["suede dress"] * 20 + ["canvas dress"] * 20
        )}
        pile = [ScoredProduct(parent_asin=parent_asin) for parent_asin in snippets]
        state = SessionState("session-1", UserProfile(), category="Women Dresses")
        closed = QuestionsEngine()
        semantic = QuestionsEngine(semantic=SemanticMatcher(_HintEmbedder()))
        self.assertEqual(
            closed.decide(state, 1, pile, snippets.get).ask_attribute,
            "other",
        )
        self.assertEqual(
            semantic.decide(state, 1, pile, snippets.get).ask_attribute,
            "material",
        )

    def test_extract_values_keeps_gazetteer_hits(self) -> None:
        extracted = extract_values(
            "soft cotton dress in blue",
            SemanticMatcher(_HintEmbedder()),
        )
        self.assertEqual(extracted["material"], "cotton")
        self.assertEqual(extracted["color"], "blue")


class SemanticRankingTest(unittest.TestCase):
    def test_synonym_title_outranks_unrelated_color(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        catalog_path = Path(directory.name) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(
                json.dumps(product) + "\n"
                for product in [
                    _product("GREEN", "Fern sheath", 900),
                    _product("MAROON", "Maroon evening gown", 5),
                ]
            ),
            encoding="utf-8",
        )
        engine = RecommendationEngine(catalog_path, embedder=_HintEmbedder())
        engine.EMBED_WEIGHT = 0.0
        state = SessionState(
            "session-1",
            UserProfile(),
            category="Women Dresses",
            active_constraints=[Constraint("burgundy", "color", 1, "clarification")],
        )
        self.assertEqual(engine.recommend(state, 2)[0].parent_asin, "MAROON")


if __name__ == "__main__":
    unittest.main()
