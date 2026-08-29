from __future__ import annotations

import unittest

from starter.intent import IntentUnderstander


class IntentUnderstanderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = IntentUnderstander()

    def test_parses_buying_and_browsing_messages(self) -> None:
        buying = self.intent.parse(
            "I'm looking for Shoes Running. A key requirement is: cotton.",
            1,
        )
        self.assertEqual(buying.interaction_kind, "buying")
        self.assertEqual(buying.category, "Shoes Running")
        self.assertEqual(buying.constraints[0].text, "cotton")
        self.assertEqual(buying.constraints[0].attribute, "material")

        browsing = self.intent.parse(
            "I'm looking for Jewelry Necklaces, but I'm still exploring.",
            1,
        )
        self.assertEqual(browsing.interaction_kind, "browsing")
        self.assertEqual(browsing.category, "Jewelry Necklaces")
        self.assertEqual(browsing.constraints, [])

    def test_parses_clarification_and_negative_replies(self) -> None:
        clarification = self.intent.parse(
            "For that, what matters is: lightweight; color: blue.",
            2,
            "other",
        )
        self.assertEqual(
            [constraint.text for constraint in clarification.constraints],
            ["lightweight", "color: blue"],
        )

        no_preference = self.intent.parse(
            "I don't have a preference for material; please use your judgment.",
            2,
            "material",
        )
        self.assertEqual(no_preference.no_preference, {"material"})
        self.assertEqual(no_preference.constraints, [])

        exhausted = self.intent.parse(
            "I don't have an additional preference for color.",
            3,
            "color",
        )
        self.assertEqual(exhausted.exhausted, {"color"})

    def test_override_requests_replacement(self) -> None:
        update = self.intent.parse(
            "Actually, ignore my earlier preference. What I need is: leather.",
            3,
            "other",
        )
        self.assertTrue(update.supersede_preferences)
        self.assertEqual(update.interaction_kind, "override")
        self.assertEqual(update.constraints[0].text, "leather")

    def test_unknown_message_uses_conservative_fallback(self) -> None:
        update = self.intent.parse("Could it be a waterproof hiking jacket?", 4)
        self.assertEqual(update.parser, "fallback")
        self.assertEqual(update.constraints[0].attribute, "use_case")
        self.assertIn("waterproof", update.constraints[0].text)


if __name__ == "__main__":
    unittest.main()
