from __future__ import annotations

import unittest

from starter.memory import MemoryStore
from starter.models import Constraint, IntentUpdate


class MemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = MemoryStore()
        self.profile = {
            "purchase_frequency": "frequent",
            "average_prior_rating": 4.2,
            "rating_style": "critical",
            "preference_tags": ["comfort"],
            "summary": "Prefers comfortable products.",
        }

    def test_profile_is_separate_from_accumulated_intent(self) -> None:
        state = self.memory.reset("session-1", self.profile)
        self.assertEqual(state.profile.preference_tags, ("comfort",))
        self.assertEqual(state.active_constraints, [])

        update = IntentUpdate(
            category="Shoes",
            constraints=[Constraint("leather", "material", 1, "initial")],
        )
        state = self.memory.apply("session-1", update, "message", 1)
        self.assertEqual(state.category, "Shoes")
        self.assertEqual(state.active_constraints[0].text, "leather")
        self.assertEqual(state.profile.preference_tags, ("comfort",))

    def test_duplicate_updates_are_idempotent(self) -> None:
        self.memory.reset("session-1", self.profile)
        update = IntentUpdate(
            constraints=[Constraint("Blue", "color", 1, "clarification")]
        )
        self.memory.apply("session-1", update, "first", 1)
        state = self.memory.apply("session-1", update, "second", 2)
        self.assertEqual(len(state.active_constraints), 1)

    def test_override_moves_active_constraints_out_of_retrieval_state(self) -> None:
        self.memory.reset("session-1", self.profile)
        self.memory.apply(
            "session-1",
            IntentUpdate(
                category="Shoes",
                constraints=[Constraint("red", "color", 1, "initial")],
            ),
            "initial",
            1,
        )
        state = self.memory.apply(
            "session-1",
            IntentUpdate(
                constraints=[Constraint("blue", "color", 3, "override")],
                supersede_preferences=True,
            ),
            "override",
            3,
        )
        self.assertEqual([item.text for item in state.active_constraints], ["blue"])
        self.assertEqual([item.text for item in state.superseded_constraints], ["red"])
        self.assertEqual(state.category, "Shoes")

    def test_shown_ids_accumulate_and_clear_on_override(self) -> None:
        self.memory.reset("session-1", self.profile)
        self.memory.record_agent_action("session-1", "material", ["A"], 1)
        self.memory.record_agent_action("session-1", "color", ["B"], 2)
        state = self.memory.get("session-1")
        self.assertEqual(state.previous_recommendations, ["A", "B"])

        self.memory.apply(
            "session-1",
            IntentUpdate(
                constraints=[Constraint("blue", "color", 3, "override")],
                supersede_preferences=True,
            ),
            "override",
            3,
        )
        self.assertEqual(self.memory.get("session-1").previous_recommendations, [])

    def test_shown_ids_keep_a_full_ten_turn_window(self) -> None:
        self.memory.reset("session-1", self.profile)
        for turn in range(1, 12):
            self.memory.record_agent_action(
                "session-1",
                None,
                [f"P{turn}-{index}" for index in range(10)],
                turn,
            )
        shown = self.memory.get("session-1").previous_recommendations
        self.assertEqual(len(shown), 100)
        self.assertNotIn("P1-0", shown)
        self.assertIn("P2-0", shown)
        self.assertIn("P11-9", shown)

    def test_reset_prevents_cross_session_leakage(self) -> None:
        self.memory.reset("session-1", self.profile)
        self.memory.apply(
            "session-1",
            IntentUpdate(constraints=[Constraint("wool", "material", 1, "initial")]),
            "initial",
            1,
        )
        state = self.memory.reset("session-2", self.profile)
        self.assertEqual(state.active_constraints, [])
        with self.assertRaises(RuntimeError):
            self.memory.get("session-1")


if __name__ == "__main__":
    unittest.main()
