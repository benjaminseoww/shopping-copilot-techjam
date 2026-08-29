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

    def test_paraphrase_handles_buying_browsing_and_override(self) -> None:
        buying = self.intent.parse(
            "Need running shoes — cotton is required.",
            1,
        )
        self.assertEqual(buying.interaction_kind, "buying")
        self.assertEqual(buying.category, "running shoes")
        self.assertEqual(buying.constraints[0].text, "cotton")
        self.assertEqual(buying.parser, "phrase")

        browsing = self.intent.parse("Just exploring necklaces for now.", 1)
        self.assertEqual(browsing.interaction_kind, "browsing")
        self.assertEqual(browsing.category, "necklaces")
        self.assertEqual(browsing.constraints, [])

        override = self.intent.parse(
            "Forget the earlier preference; make it leather.",
            3,
        )
        self.assertEqual(override.interaction_kind, "override")
        self.assertTrue(override.supersede_preferences)
        self.assertEqual(override.constraints[0].text, "leather")

    def test_paraphrase_uses_last_ask_and_extract_clarifications(self) -> None:
        no_preference = self.intent.parse(
            "It doesn't matter, you pick.",
            2,
            "material",
        )
        self.assertEqual(no_preference.interaction_kind, "no_preference")
        self.assertEqual(no_preference.no_preference, {"material"})
        self.assertEqual(no_preference.constraints, [])

        exhausted = self.intent.parse("Nothing more on color.", 3, "other")
        self.assertEqual(exhausted.interaction_kind, "exhausted")
        self.assertEqual(exhausted.exhausted, {"color"})

        clarification = self.intent.parse(
            "The important part is lightweight, and also color: blue.",
            2,
        )
        self.assertEqual(
            [constraint.text for constraint in clarification.constraints],
            ["lightweight", "color: blue"],
        )

    def test_mixed_forget_and_no_preference_is_override(self) -> None:
        update = self.intent.parse(
            "Forget the earlier preference, it doesn't matter, make it leather.",
            3,
        )
        self.assertEqual(update.interaction_kind, "override")
        self.assertTrue(update.supersede_preferences)
        self.assertEqual(update.constraints[0].text, "leather")
        self.assertEqual(update.no_preference, set())

    def test_negation_is_not_added_as_positive_search_evidence(self) -> None:
        update = self.intent.parse(
            "I'm looking for boots. I don't want leather.",
            2,
        )
        self.assertEqual(update.category, "boots")
        self.assertEqual(update.constraints, [])
        self.assertEqual(update.fallback_terms, [])
        self.assertFalse(update.supersede_preferences)

        vague_actual = self.intent.parse(
            "Actually this could be a waterproof jacket.",
            2,
        )
        self.assertFalse(vague_actual.supersede_preferences)

    def test_vague_need_or_want_is_not_buying(self) -> None:
        keep_looking = self.intent.parse("I want to keep looking", 2)
        self.assertEqual(keep_looking.interaction_kind, "noop")
        self.assertEqual(keep_looking.constraints, [])
        self.assertFalse(keep_looking.supersede_preferences)

        think = self.intent.parse("I need to think", 2)
        self.assertEqual(think.interaction_kind, "noop")
        self.assertEqual(think.constraints, [])

    def test_answer_ask_binds_short_replies_to_last_ask(self) -> None:
        leather = self.intent.parse("leather", 2, "material")
        self.assertEqual(leather.interaction_kind, "answer_ask")
        self.assertEqual(leather.constraints[0].text, "leather")
        self.assertEqual(leather.constraints[0].attribute, "material")
        self.assertFalse(leather.supersede_preferences)

        prefer = self.intent.parse("I prefer cotton", 2, "material")
        self.assertEqual(prefer.interaction_kind, "answer_ask")
        self.assertEqual(prefer.constraints[0].text, "cotton")
        self.assertEqual(prefer.constraints[0].attribute, "material")

        navy = self.intent.parse("navy", 3, "color")
        self.assertEqual(navy.interaction_kind, "answer_ask")
        self.assertEqual(navy.constraints[0].text, "navy")
        self.assertEqual(navy.constraints[0].attribute, "color")

        mismatched = self.intent.parse("blue", 3, "material")
        self.assertEqual(mismatched.constraints[0].attribute, "color")

    def test_answer_ask_does_not_steal_questions_or_thinking(self) -> None:
        question = self.intent.parse(
            "Could it be a waterproof hiking jacket?",
            4,
            "material",
        )
        self.assertEqual(question.parser, "fallback")
        self.assertNotEqual(question.interaction_kind, "answer_ask")

        think = self.intent.parse("I need to think", 4, "material")
        self.assertEqual(think.interaction_kind, "noop")
        self.assertEqual(think.constraints, [])

    def test_keyword_gazetteers_match_whole_words(self) -> None:
        self.assertEqual(self.intent.classify_constraint("red leather"), "material")
        self.assertEqual(self.intent.classify_constraint("required feature"), "feature")
        self.assertEqual(self.intent.classify_constraint("color: navy"), "color")

    def test_unknown_message_uses_conservative_fallback(self) -> None:
        update = self.intent.parse("Could it be a waterproof hiking jacket?", 4)
        self.assertEqual(update.parser, "fallback")
        self.assertEqual(update.constraints[0].attribute, "use_case")
        self.assertIn("waterproof", update.constraints[0].text)


if __name__ == "__main__":
    unittest.main()
