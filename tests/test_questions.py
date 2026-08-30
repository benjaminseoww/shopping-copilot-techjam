from __future__ import annotations

import unittest
from collections.abc import Callable, Sequence

from starter.models import Constraint, ScoredProduct, SessionState, UserProfile
from starter.questions import QuestionsEngine, extract_values, family_from_category


def _pile(texts: Sequence[str]) -> tuple[list[ScoredProduct], Callable[[str], str]]:
    snippets = {f"P{index}": text for index, text in enumerate(texts)}
    products = [ScoredProduct(parent_asin=parent_asin) for parent_asin in snippets]
    return products, snippets.get


class QuestionsEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = QuestionsEngine()
        self.state = SessionState("session-1", UserProfile())

    def test_asks_other_on_turns_one_through_nine(self) -> None:
        for turn in range(1, 10):
            with self.subTest(turn=turn):
                decision = self.engine.decide(self.state, turn)
                self.assertEqual(decision.ask_attribute, "other")
                self.assertTrue(decision.message)

    def test_does_not_ask_on_turn_ten(self) -> None:
        decision = self.engine.decide(self.state, 10)
        self.assertIsNone(decision.ask_attribute)
        self.assertEqual(
            decision.message,
            "Here are the closest matches based on your preferences.",
        )

    def test_empty_pile_asks_other(self) -> None:
        decision = self.engine.decide(self.state, 1, [])
        self.assertEqual(decision.ask_attribute, "other")
        self.assertEqual(
            decision.message,
            "What other requirement or priority matters most to you?",
        )

    def test_clothing_split_materials_asks_material(self) -> None:
        self.state.category = "Women Dresses"
        pile, catalog_text = _pile(["cotton dress"] * 20 + ["polyester dress"] * 20)
        decision = self.engine.decide(self.state, 1, pile, catalog_text)
        self.assertEqual(decision.ask_attribute, "material")
        self.assertEqual(decision.message, "Do you have a material preference?")

    def test_constant_material_asks_other(self) -> None:
        self.state.category = "Women Dresses"
        pile, catalog_text = _pile(["cotton dress"] * 40)
        decision = self.engine.decide(self.state, 1, pile, catalog_text)
        self.assertEqual(decision.ask_attribute, "other")

    def test_jewelry_does_not_ask_material(self) -> None:
        self.state.category = "Jewelry Necklaces"
        pile, catalog_text = _pile(["gold necklace"] * 20 + ["silver necklace"] * 20)
        decision = self.engine.decide(self.state, 1, pile, catalog_text)
        self.assertNotEqual(decision.ask_attribute, "material")
        self.assertEqual(decision.ask_attribute, "other")

    def test_jewelry_split_colors_asks_color(self) -> None:
        self.state.category = "Jewelry Necklaces"
        pile, catalog_text = _pile(["black necklace"] * 20 + ["white necklace"] * 20)
        decision = self.engine.decide(self.state, 1, pile, catalog_text)
        self.assertEqual(decision.ask_attribute, "color")
        self.assertEqual(decision.message, "Do you have a color preference?")

    def test_known_constraint_asks_open_followup_not_another_typed_field(self) -> None:
        self.state.category = "Women Dresses"
        self.state.active_constraints.append(Constraint("cotton", "material", 1, "initial"))
        pile, catalog_text = _pile(["blue cotton dress"] * 20 + ["red cotton dress"] * 20)
        decision = self.engine.decide(self.state, 2, pile, catalog_text)
        self.assertEqual(decision.ask_attribute, "other")
        self.assertEqual(
            decision.message,
            "What other requirement or priority matters most to you?",
        )

    def test_no_preference_switches_to_open_followup(self) -> None:
        self.state.category = "Women Dresses"
        self.state.no_preference.add("material")
        pile, catalog_text = _pile(["blue cotton dress"] * 20 + ["red polyester dress"] * 20)
        decision = self.engine.decide(self.state, 2, pile, catalog_text)
        self.assertEqual(decision.ask_attribute, "other")

    def test_answered_no_preference_exhausted_material_not_reasked(self) -> None:
        self.state.category = "Women Dresses"
        pile, catalog_text = _pile(["cotton dress"] * 20 + ["polyester dress"] * 20)

        answered = SessionState("answered", UserProfile(), category="Women Dresses")
        answered.active_constraints.append(Constraint("cotton", "material", 1, "initial"))
        self.assertNotEqual(
            self.engine.decide(answered, 2, pile, catalog_text).ask_attribute,
            "material",
        )

        declined = SessionState("declined", UserProfile(), category="Women Dresses")
        declined.no_preference.add("material")
        self.assertNotEqual(
            self.engine.decide(declined, 2, pile, catalog_text).ask_attribute,
            "material",
        )

        exhausted = SessionState("exhausted", UserProfile(), category="Women Dresses")
        exhausted.exhausted_attributes.add("material")
        self.assertNotEqual(
            self.engine.decide(exhausted, 2, pile, catalog_text).ask_attribute,
            "material",
        )

    def test_never_asks_use_case_brand_budget_category(self) -> None:
        cases = [
            SessionState("empty", UserProfile()),
            SessionState("dresses", UserProfile(), category="Women Dresses"),
            SessionState("jewelry", UserProfile(), category="Jewelry Necklaces"),
            SessionState("shoes", UserProfile(), category="Shoes Slippers"),
        ]
        piles = [
            _pile([]),
            _pile(["cotton dress"] * 20 + ["polyester dress"] * 20),
            _pile(["black necklace"] * 20 + ["white necklace"] * 20),
            _pile(["leather shoe"] * 20 + ["nylon shoe"] * 20),
        ]
        forbidden = {"use_case", "brand", "budget", "category"}
        for state, (pile, catalog_text) in zip(cases, piles):
            decision = self.engine.decide(state, 1, pile, catalog_text)
            self.assertNotIn(decision.ask_attribute, forbidden)

    def test_family_from_category(self) -> None:
        self.assertEqual(family_from_category("Jewelry Necklaces"), "jewelry")
        self.assertEqual(family_from_category("Women Dresses"), "clothing")
        self.assertEqual(family_from_category("Shoes Slippers"), "shoes")
        self.assertEqual(family_from_category("Watches Wrist Watches"), "watches")
        self.assertEqual(family_from_category("Handbags & Wallets Totes"), "bags")
        shoe_pile, catalog_text = _pile(["running shoe sneaker", "walking shoe boot"])
        self.assertEqual(
            family_from_category("Athletic Walking", shoe_pile, catalog_text),
            "shoes",
        )

    def test_extract_values(self) -> None:
        extracted = extract_values("soft cotton dress in blue")
        self.assertEqual(extracted["material"], "cotton")
        self.assertEqual(extracted["color"], "blue")
        alloy = extract_values("alloy necklace")
        self.assertIsNone(alloy["material"])


if __name__ == "__main__":
    unittest.main()
