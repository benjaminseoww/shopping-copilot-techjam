from __future__ import annotations

import unittest

from starter.models import Constraint, ScoredProduct, SessionState, UserProfile
from starter.questions import QuestionsEngine, extract_values, family_from_category


def _products(texts: list[str]) -> list[ScoredProduct]:
    return [
        ScoredProduct(parent_asin=f"P{index}", text=text)
        for index, text in enumerate(texts)
    ]


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
        pile = _products(["cotton dress"] * 20 + ["polyester dress"] * 20)
        decision = self.engine.decide(self.state, 1, pile)
        self.assertEqual(decision.ask_attribute, "material")
        self.assertEqual(decision.message, "Do you have a material preference?")

    def test_constant_material_asks_other(self) -> None:
        self.state.category = "Women Dresses"
        pile = _products(["cotton dress"] * 40)
        decision = self.engine.decide(self.state, 1, pile)
        self.assertEqual(decision.ask_attribute, "other")

    def test_jewelry_does_not_ask_material(self) -> None:
        self.state.category = "Jewelry Necklaces"
        pile = _products(["gold necklace"] * 20 + ["silver necklace"] * 20)
        decision = self.engine.decide(self.state, 1, pile)
        self.assertNotEqual(decision.ask_attribute, "material")
        self.assertEqual(decision.ask_attribute, "other")

    def test_jewelry_split_colors_asks_color(self) -> None:
        self.state.category = "Jewelry Necklaces"
        pile = _products(["black necklace"] * 20 + ["white necklace"] * 20)
        decision = self.engine.decide(self.state, 1, pile)
        self.assertEqual(decision.ask_attribute, "color")
        self.assertEqual(decision.message, "Do you have a color preference?")

    def test_answered_no_preference_exhausted_material_not_reasked(self) -> None:
        self.state.category = "Women Dresses"
        pile = _products(["cotton dress"] * 20 + ["polyester dress"] * 20)

        answered = SessionState("answered", UserProfile(), category="Women Dresses")
        answered.active_constraints.append(Constraint("cotton", "material", 1, "initial"))
        self.assertNotEqual(
            self.engine.decide(answered, 2, pile).ask_attribute,
            "material",
        )

        declined = SessionState("declined", UserProfile(), category="Women Dresses")
        declined.no_preference.add("material")
        self.assertNotEqual(
            self.engine.decide(declined, 2, pile).ask_attribute,
            "material",
        )

        exhausted = SessionState("exhausted", UserProfile(), category="Women Dresses")
        exhausted.exhausted_attributes.add("material")
        self.assertNotEqual(
            self.engine.decide(exhausted, 2, pile).ask_attribute,
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
            [],
            _products(["cotton dress"] * 20 + ["polyester dress"] * 20),
            _products(["black necklace"] * 20 + ["white necklace"] * 20),
            _products(["leather shoe"] * 20 + ["nylon shoe"] * 20),
        ]
        forbidden = {"use_case", "brand", "budget", "category"}
        for state, pile in zip(cases, piles):
            decision = self.engine.decide(state, 1, pile)
            self.assertNotIn(decision.ask_attribute, forbidden)

    def test_family_from_category(self) -> None:
        self.assertEqual(family_from_category("Jewelry Necklaces"), "jewelry")
        self.assertEqual(family_from_category("Women Dresses"), "clothing")
        self.assertEqual(family_from_category("Shoes Slippers"), "shoes")
        self.assertEqual(family_from_category("Watches Wrist Watches"), "watches")
        self.assertEqual(family_from_category("Handbags & Wallets Totes"), "bags")
        shoe_pile = _products(["running shoe sneaker", "walking shoe boot"])
        self.assertEqual(family_from_category("Athletic Walking", shoe_pile), "shoes")

    def test_extract_values(self) -> None:
        extracted = extract_values("soft cotton dress in blue")
        self.assertEqual(extracted["material"], "cotton")
        self.assertEqual(extracted["color"], "blue")
        alloy = extract_values("alloy necklace")
        self.assertIsNone(alloy["material"])


if __name__ == "__main__":
    unittest.main()
